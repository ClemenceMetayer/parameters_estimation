""" Parallelized CMAES + soft constraints on the loss for the parameters estimation """

import pickle as pkl
import warnings

import cma
import numpy as np
from cma.optimization_tools import EvalParallel2

from functions import from_log_010_to_ab, wrapper_fit, fitness_sw, clock_model_sw
from utils import preprocessing_rna_seq_data, get_parameter_bounds


def main():
   
    # Importation of the data
    filename = "data/Jeu_8/data/data_dict_concentration_cc.dat"

    with open(filename, "rb") as f:
        rna_seq_data = pkl.load(f)

    rna_seq_matrix = preprocessing_rna_seq_data(rna_seq_data)

    time_rna = rna_seq_data["CTs"]
    time = np.concatenate((time_rna, time_rna, time_rna))  # 3 rep

    y0 = np.loadtxt("data/y0.txt") # Initial values of the state variables
    vc, vn = 0.8, 0.2
    # each species concentration is multiplied by the proportion of
    # total volume of the compartment it belongs to
    # used only for constraints after.
    w = np.zeros(y0.shape[0])
    w[0:4] = vn
    w[4:9] = 1
    w[9:] = vc
    w[-1] = 1

    rna_seq_max = np.nanmax(rna_seq_matrix, axis=0)

    tspan = np.linspace(0, 600, 6001)
    t_interp_rna_seq = np.array((tspan[-1] - 60 + time) * 10, dtype=np.int32)

    params_min, params_max = get_parameter_bounds(62)
    params = np.random.uniform(params_min, params_max)
    
    lp = len(params)
    lb, ub = np.zeros(lp), 10 * np.ones(lp)

    # see here for a description of these options
    # https://github.om/CMA-ES/pycma/blob/master/cma_signals.in
    options = {  #'tolfun': 1e-6,
        #'tolfunhist': 1e-,
        "tolx": 1e-20,
        "CMA_elitist": True,  # keep only the best 50% for next gen
        "bounds": [lb, ub],
        "popsize": 30,  # popsize evals of fitness per generation
        "tolflatfitness": 1000,
    }

    scaler = from_log_010_to_ab
    sigma = 2
    likelihood = 1e10
    
    with EvalParallel2(wrapper_fit) as eval_all:
        for _ in range(10):
            start_scale = (
                10
                * (np.log10(params) - np.log10(params_min))
                / (np.log10(params_max) - np.log10(params_min))
            )
            options["fixed_variables"] = {
                59: start_scale[59]
            }  # won't be touched anyway, just fix it to some value in (0, 10)
            es = cma.CMAEvolutionStrategy(start_scale, sigma, options)
            while not es.stop():
                # get parameter set to evaluate fitness
                X = es.ask()
                # evaluate and update cov matrix
                es.tell(
                    X,
                    eval_all(
                        X,
                        args=(
                            params_max,
                            params_min,
                            scaler,
                            fitness_sw,
                            clock_model_sw,
                            rna_seq_matrix,
                            rna_seq_max,
                            w,
                            t_interp_rna_seq,
                            tspan,
                            y0,
                        ),
                    ),
                )
                es.logger.add()
                es.disp()
            print(es.result[-1])  # stopping criterion
            if es.result[1] < likelihood:
                likelihood = es.result[1]
                params = es.result[0]
                print(f"found new parameter set with likelihood {likelihood}")
                # the cmaes optimal parameter set needs to be
                # scaled back to the original parameter domain.
                params = scaler(params, params_max, params_min)
                # foldbmal not estimated
                params[59] = 12
                np.savetxt(
                    f"example_new _cmaes_result_{likelihood}.txt",
                    np.array([params, params_min, params_max]),
                )
            else:
                params = np.random.uniform(params_min, params_max)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
