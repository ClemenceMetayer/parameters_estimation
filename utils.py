""" Various useful functions """

import numpy as np
import pandas as pd

def preprocessing_rna_seq_data(rna_seq_data):

    """
    From the RNA-Seq data dictionary, get the data and the structure
    that will be given to cma-es.
    
    
    """

    names = ['ARNTL','CLOCK', 'CRY', 'REV-ERB', 'PER', 'ROR']
    nbio_rep, n_timepoints = 3, 6
    
    rna_seq_matrix = np.zeros([len(names), n_timepoints, nbio_rep])
    
    for i, n in enumerate(names):
        
        if n == "ARNTL" or n == "CLOCK" :
            rna_seq_matrix[i, :, :] = [list(l) for l in rna_seq_data["ctrl"][n]["pts"]]
            
        if n == "CRY" : 
            rna_seq_matrix[i, :, :] = [list(np.nansum([l1, l2], axis=0)) for l1, l2 in zip(rna_seq_data["ctrl"]["CRY1"]["pts"], rna_seq_data["ctrl"]["CRY2"]["pts"])]
            
        if n == "REV-ERB" : 
            rna_seq_matrix[i, :, :] = [list(np.nansum([l1, l2], axis=0)) for l1, l2 in zip(rna_seq_data["ctrl"]["NR1D1"]["pts"], rna_seq_data["ctrl"]["NR1D2"]["pts"])]

        if n == "PER" : 
            rna_seq_matrix[i, :, :] = [list(np.nansum([l1, l2, l3], axis=0)) for l1, l2, l3 in zip(rna_seq_data["ctrl"]["PER1"]["pts"], rna_seq_data["ctrl"]["PER2"]["pts"], rna_seq_data["ctrl"]["PER3"]["pts"])]
    
        if n == "ROR" : 
            rna_seq_matrix[i, :, :] = [list(np.nansum([l1, l2], axis=0)) for l1, l2 in zip(rna_seq_data["ctrl"]["RORA"]["pts"], rna_seq_data["ctrl"]["RORB"]["pts"])]
            
    col_arntl = np.concatenate((rna_seq_matrix[0][:,0], rna_seq_matrix[0][:,1], rna_seq_matrix[0][:,2]), axis=0)
    col_clock = np.concatenate((rna_seq_matrix[1][:,0], rna_seq_matrix[1][:,1], rna_seq_matrix[1][:,2]), axis=0)
    col_cry = np.concatenate((rna_seq_matrix[2][:,0], rna_seq_matrix[2][:,1], rna_seq_matrix[2][:,2]), axis=0)
    col_rev = np.concatenate((rna_seq_matrix[3][:,0], rna_seq_matrix[3][:,1], rna_seq_matrix[3][:,2]), axis=0)
    col_per = np.concatenate((rna_seq_matrix[4][:,0], rna_seq_matrix[4][:,1], rna_seq_matrix[4][:,2]), axis=0)
    col_ror = np.concatenate((rna_seq_matrix[5][:,0], rna_seq_matrix[5][:,1], rna_seq_matrix[5][:,2]), axis=0)
            
    rna_seq_matrix = pd.DataFrame({'col1': col_arntl,
                                   'col2': col_clock,
                                   'col3': col_cry,
                                   'col4': col_rev,
                                   'col5': col_per,
                                   'col6': col_ror})

    return np.array(rna_seq_matrix)


def get_parameter_bounds(n_params):

    """Parameter lower and upper bounds"""

    params_min, params_max = np.zeros(n_params), np.zeros(n_params)
    for i in range(n_params):
        # degradations
        if i < 16 or i in [17, 19]:
            params_min[i] = 1e-2
            params_max[i] = 3
        # complexations as 10**9 * decomplexation
        if i in [16, 18]:
            params_min[i] = 1e5
            params_max[i] = 1e9
        # transcription rates are bounded
        # with the crude approx x_mean = vmax/deg
        if i in [20, 21, 22]:
            params_min[i] = 1e-13
            params_max[i] = 5e-7
        # regulations ratios. No hint here really.
        # typically these parameters change all the time.
        if i > 22 and i < 36:
            params_min[i] = 1e-13
            params_max[i] = 1e-5
        # production rates scale with x_prot_val/x_gene_val so around 1e3
        if i > 35 and i < 42:
            params_min[i] = 1
            params_max[i] = 1e5
        # transport rates
        if i in [42, 43, 44, 45, 56, 57]:
            params_min[i] = 1e-3
            params_max[i] = 1
        # fold activation ratios
        if ((i > 45) and (i < 51)):
            params_min[i] = 1
            params_max[i] = 200
        # hill coefficients
        if i > 50 and i < 56:
            params_min[i] = 1
            params_max[i] = 8

    params_min[[56, 57, 58]] = 1e-13
    params_max[[56, 57, 58]] = 5e-7
    params_min[[60, 61]] = 1e-3
    params_max[[60, 61]] = 1
    params_min[59] = 11.9
    params_max[59] = 12.1
    return params_min, params_max

