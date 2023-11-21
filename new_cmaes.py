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
                   new_preprocessing_rna_seq_data, preprocessing_prot_data)

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
    (
        jeu2_matrix,
        jeu2_timepoints,
        jeu3_matrix,
        jeu3_timepoints,
        jeu4_matrix,
        jeu4_timepoints,
        jeu5_matrix,
        jeu5_timepoints,
        jeu7_matrix,
        jeu7_timepoints,
        jeu8_matrix,
        jeu8_timepoints,
        jeu10_matrix,
        jeu10_timepoints,
    ) = preprocessing_prot_data(prot_data)

    jeu2_matrix /= 10**9
    jeu3_matrix /= 10**9
    jeu4_matrix /= 10**9
    jeu5_matrix /= 10**9
    jeu7_matrix /= 10**9
    jeu8_matrix /= 10**9
    jeu10_matrix /= 10**9
    # Missing data handling

    # Jeu 2
    jeu2_max = np.nanmax(jeu2_matrix, axis=0)

    # Jeu 3
    jeu3_max = np.nanmax(jeu3_matrix, axis=0)

    # Jeu 4
    jeu4_max = np.nanmax(jeu4_matrix, axis=0)

    # Jeu 5
    jeu5_max = np.nanmax(jeu5_matrix, axis=0)
    nanmean = np.nanmean(jeu5_matrix, axis=0)
    inds = np.where(np.isnan(jeu5_matrix))
    jeu5_matrix[inds] = np.take(nanmean, inds[1])

    # Jeu 7
    nan_columns_to_delete = [3, 5, 6, 11]
    jeu7_matrix = np.delete(jeu7_matrix, nan_columns_to_delete, axis=1)
    jeu7_max = np.nanmax(jeu7_matrix, axis=0)
    nanmean = np.nanmean(jeu7_matrix, axis=0)
    inds = np.where(np.isnan(jeu7_matrix))
    jeu7_matrix[inds] = np.take(nanmean, inds[1])

    # Jeu 8
    jeu8_max = np.nanmax(jeu8_matrix, axis=0)

    # Jeu 10
    jeu10_max = np.nanmax(jeu10_matrix, axis=0)

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
    tspan = np.linspace(0, 1200, 12001)

    t_interp_rna_seq = [
        np.array([(tspan[-1] - 60 + ct) * 10 for ct in CT], dtype=np.int32)
        for CT in rna_seq_cts
    ]
    t_interp_jeu2 = np.array((tspan[-1] - 60 + jeu2_timepoints) * 10, dtype=np.int32)
    t_interp_jeu3 = np.array((tspan[-1] - 60 + jeu3_timepoints) * 10, dtype=np.int32)
    t_interp_jeu4 = np.array((tspan[-1] - 60 + jeu4_timepoints) * 10, dtype=np.int32)
    t_interp_jeu5 = np.array((tspan[-1] - 60 + jeu5_timepoints) * 10, dtype=np.int32)
    t_interp_jeu7 = np.array((tspan[-1] - 60 + jeu7_timepoints) * 10, dtype=np.int32)
    t_interp_jeu8 = np.array((tspan[-1] - 60 + jeu8_timepoints) * 10, dtype=np.int32)
    t_interp_jeu10 = np.array((tspan[-1] - 60 + jeu10_timepoints) * 10, dtype=np.int32)

    # Initial values of the model parameters
    # params_julien = np.loadtxt('data/error2.6612_sw480_pcrmicro.txt')[0]
    params_julien = np.loadtxt(
        "results/example_new_cmaes_result_2.032966400368863.txt"
    )[0]
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
        "tolfun": 1e-7,
        "tolfunhist": 1e-7,
        "minstd": 1e-12,
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
                            jeu2_matrix,
                            jeu2_max,
                            jeu3_matrix,
                            jeu3_max,
                            jeu4_matrix,
                            jeu4_max,
                            jeu5_matrix,
                            jeu5_max,
                            jeu7_matrix,
                            jeu7_max,
                            jeu8_matrix,
                            jeu8_max,
                            jeu10_matrix,
                            jeu10_max,
                            w,
                            t_interp_rna_seq,
                            t_interp_jeu2,
                            t_interp_jeu3,
                            t_interp_jeu4,
                            t_interp_jeu5,
                            t_interp_jeu7,
                            t_interp_jeu8,
                            t_interp_jeu10,
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
