""" Parallelized CMAES + soft constraints on the loss for the parameters estimation """

import pickle as pkl
import subprocess
import warnings

import cma
import numpy as np
import pandas as pd
from cma.optimization_tools import EvalParallel2
from numba.typed import List
from numba.types import Array, float32

from functions import clock_model, fitness, from_log_010_to_ab, wrapper_fit
from utils import (bounds_reach, get_parameter_bounds,
                   get_parameter_bounds_percent, get_random_parameters,
                   new_preprocessing_rna_seq_data,
                   next_preprocessing_prot_data)

# LIST OF THE PARAMETERS TO RE-ESTIMATE #######################################
resdir = ""
list_params = (
    ["dx{}".format(i) for i in [1, 2, 5, 6]]
    + ["dy{}".format(i) for i in range(1, 7)]
    + ["dz{}".format(i) for i in [1, 2, 5, 6, 7, 8]]
    + ["kfz9", "kdz9", "kfz4", "kdz4"]
    + ["V{}max".format(i) for i in [1, 2, 5]]
    + [
        "kt1",
        "ki1",
        "kt2",
        "ki2",
        "ki21",
        "kt3",
        "ki3",
        "kt4",
        "ki4",
        "kt5",
        "ki5",
        "kt6",
        "ki6",
    ]
    + ["kp{}".format(i) for i in range(1, 7)]
    + ["kiz{}".format(i) for i in [4, 6, 7, 9]]
    + ["fold_per", "fold_cry", "fold_rev", "fold_ror", "fold_clock"]
    + ["hill_inhi", "hill_acti", "hill_inhi_cry", "hill_acti_cry", "hill_inhi_cry_rev"]
    + ["V3max", "V4max", "V6max", "fold_bmal", "kex1", "kex2"]
)

params_to_reestimate = pd.read_csv("data/params_indirect_5_params.csv")
params_to_reestimate = [name[0] for name in params_to_reestimate.values]

###############################################################################


def main():
    # IMPORTATION OF THE RNA-SEQ DATA #########################################
    filename = "data/Jeu_9/data/data_dict_concentration_cc.dat"
    with open(filename, "rb") as f:
        rna_seq_data = pkl.load(f)

    # Data formating
    rna_seq_list, rna_seq_cts = new_preprocessing_rna_seq_data(
        rna_seq_data, factor=10**9
    )
    rna_seq_list = [np.array(sum(r, [])) for r in rna_seq_list]

    # arr = Array(float32, 1, 'C')
    # typed_rna = List(arr)
    # [typed_rna.append(x) for x in rna_seq_list]
    # rna_seq_list = typed_rna

    rna_seq_max = [max(r) for r in rna_seq_list]

    ###########################################################################

    # IMPORTATION OF THE PROTEIN DATA #########################################
    filename = "data/dict_comparison"
    with open(filename, "rb") as f:
        prot_data = pkl.load(f)

    # Data formating and time points initialisation
    prot_data = next_preprocessing_prot_data(prot_data)
    prot_list = [d[0] for d in prot_data]
    prot_cts = [d[1] for d in prot_data]
    prot_max = np.array([d[2] for d in prot_data])

    ###########################################################################

    # INITIALISATION OF THE VARIABLES NEEDED FOR CMA-ES #######################

    # Initial values of the state variables
    y0 = np.loadtxt("data/y0.txt")

    # Volume of the cytoplasm and the nucleus
    vc, vn = 0.72, 0.28

    # Weights vector initialisation
    w = np.zeros(y0.shape[0])
    w[0:4] = vn
    w[4:9] = 1
    w[9:] = vc
    w[-1] = 1

    # Time points vector initialisation
    tspan = np.linspace(0, 600, 6001)

    t_interp_rna_seq = [
        np.array([(tspan[-1] - 60 + ct) * 10 for ct in CT], dtype=np.int32)
        for CT in rna_seq_cts
    ]
    t_interp_prot = [
        np.array([(tspan[-1] - 60 + ct) * 10 for ct in CT], dtype=np.int32)
        for CT in prot_cts
    ]

    # Initial values of the model parameters
    params_julien = np.loadtxt('data/error2.6612_sw480_pcrmicro.txt')[0]
    # params_min, params_max = get_parameter_bounds_percent(params_julien,5)
    params_min, params_max = get_parameter_bounds(len(params_julien), "sw")

    # Logarithmic interval
    lp = len(params_julien)
    lb, ub = np.zeros(lp), 10 * np.ones(lp)

    # Options of the CMA-ES algoritghm
    # see here for a description of these options
    # https://github.om/CMA-ES/pycma/blob/master/cma_signals.in
    options = {
        "tolx": 1e-20,
        "CMA_elitist": True,
        "bounds": [lb, ub],
        "popsize": 64,
        "tolflatfitness": 1000,
        "tolfun": 1e-5,
        "tolfunhist": 1e-5,
        "minstd": 1e-10,
    }

    # Initialisation of the scaler function
    scaler = from_log_010_to_ab

    # Standard deviation of the distribution and initiamisation of the likelihood
    sigma = 2
    likelihood = 1e10

    ###########################################################################

    # INITIALISATION OF THE FIXED PARAMETERS ##################################
    # fixed_params = [list_params.index(name) for name in list_params if name not in params_to_reestimate]
    fixed_params = []

    ###########################################################################

    # THE CMA-ES ALGORITHM ####################################################
    # params = get_random_parameters(fixed_params, params_julien, params_min, params_max)
    params = params_julien
    with EvalParallel2(wrapper_fit) as eval_all:
        for _ in range(10):
            # Random parameters values in the interval
            # params = get_random_parameters(fixed_params, params_julien, params_min, params_max)
            # params = params_julien
            # Scaling
            start_scale = (
                10
                * (np.log10(params) - np.log10(params_min))
                / (np.log10(params_max) - np.log10(params_min))
            )

            # Fixed parameters values
            options["fixed_variables"] = {i: start_scale[i] for i in fixed_params}

            # CMAE-ES algorithm
            es = cma.CMAEvolutionStrategy(start_scale, sigma, options)
            while not es.stop():
                X = es.ask()
                es.tell(
                    X,
                    eval_all(
                        X,
                        args=(
                            params_max,
                            params_min,
                            scaler,
                            fitness,
                            clock_model,
                            rna_seq_list,
                            rna_seq_max,
                            prot_list,
                            prot_max,
                            w,
                            t_interp_rna_seq,
                            t_interp_prot,
                            tspan,
                            y0,
                        ),
                    ),
                )
                es.logger.add()
                es.disp()

            # Test of a better likelihood and parameters set
            if es.result[1] < likelihood:
                likelihood = es.result[1]
                params = es.result[0]
                print(f"found new parameter set with likelihood {likelihood}")
                params = scaler(params, params_max, params_min)

                # Test of the reached boundaries
                bound_reach = bounds_reach(params, params_min, params_max)
                if len(bound_reach) == 0:
                    print("Bounds OK")
                else:
                    print("Bounds reached :" + str(bound_reach))

                # Save the parameters set
                np.savetxt(
                    f"results/example_new_cmaes_result_{likelihood}.txt",
                    np.array([params, params_min, params_max]),
                )
                subprocess.call(
                    f"mv outcmaes/xrecentbest.dat outcmaes/params_dist_{likelihood}.dat",
                    shell=True,
                )
            else:
                params = np.random.uniform(params_min, params_max)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
