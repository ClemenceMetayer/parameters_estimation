""" Various useful functions """

import numpy as np
import pandas as pd

def preprocessing_rna_seq_data(rna_seq_data):

    """
    From the RNA-Seq data dictionary, get the data and the structure
    that will be given to cma-es.
    
    rna_seq_data : python dictionary that contains the RNA-seq data
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


def preprocessing_prot_data(prot_data):

    """
    From the Protein data dictionary, get the data and the structure
    that will be given to cma-es.
    
    prot_data : python dictionary that contains the Protein data
    """
    
    # JEU 2 ###################################################################
    names_prot = ["BMAL1", "CRY"]
    len_jeu2_timepoints = len(prot_data["BMAL1"]["Jeu_2"]["cts"])
    jeu2_timepoints = prot_data["BMAL1"]["Jeu_2"]["cts"]
    jeu2_matrix = np.zeros([len_jeu2_timepoints, len(names_prot)])
    for idx_data, name_prot in enumerate(names_prot) : 
       jeu2_matrix[:,idx_data] = prot_data[name_prot]["Jeu_2"]["pts"]
    
    # JEU 3 ###################################################################
    names_prot = ["CRY", "REV-ERB"]
    len_jeu3_timepoints = len(prot_data["CRY"]["Jeu_3"]["cts"])
    jeu3_timepoints = prot_data["CRY"]["Jeu_3"]["cts"]
    jeu3_matrix = np.zeros([len_jeu3_timepoints, len(names_prot)])
    for idx_data, name_prot in enumerate(names_prot) : 
       jeu3_matrix[:,idx_data] = prot_data[name_prot]["Jeu_3"]["pts"]
    
    # JEU 4 ###################################################################
    names_prot = ["BMAL1_N", "CRY", "CRY_N", "CRY_C", "PER_N"]
    len_jeu4_timepoints = len(prot_data["BMAL1_N"]["Jeu_4"]["cts"])
    jeu4_timepoints = prot_data["BMAL1_N"]["Jeu_4"]["cts"]
    jeu4_matrix = np.zeros([len_jeu4_timepoints, len(names_prot)])
    for idx_data, name_prot in enumerate(names_prot) : 
       jeu4_matrix[:,idx_data] = prot_data[name_prot]["Jeu_4"]["pts"]
    
    # JEU 5 ###################################################################
    names_prot = ["CRY", "CRY_N", "CRY_C", "PER"]
    len_jeu5_timepoints = len(prot_data["CRY"]["Jeu_5"]["cts"])
    jeu5_timepoints = prot_data["CRY"]["Jeu_5"]["cts"]
    jeu5_matrix = np.zeros([len_jeu5_timepoints, len(names_prot)])
    for idx_data, name_prot in enumerate(names_prot) : 
       jeu5_matrix[:,idx_data] = prot_data[name_prot]["Jeu_5"]["pts"]
    
    # JEU 7 ###################################################################
    names_prot = ["PER", "REV-ERB", "CLOCK", "ROR"]
    len_jeu7_timepoints = len(prot_data["PER"]["Jeu_7_rep1"]["cts"])
    jeu7_timepoints = prot_data["PER"]["Jeu_7_rep1"]["cts"]
    nb_rep = 4
    jeu7_matrix = np.zeros([len_jeu7_timepoints, len(names_prot)*nb_rep])
    idx_data = 0
    for name_prot in names_prot : 
       jeu7_matrix[:,idx_data] = prot_data[name_prot]["Jeu_7_rep1"]["pts"]
       idx_data +=1
       jeu7_matrix[:,idx_data] = prot_data[name_prot]["Jeu_7_rep2"]["pts"]
       idx_data +=1
       jeu7_matrix[:,idx_data] = prot_data[name_prot]["Jeu_7_rep3"]["pts"]
       idx_data +=1
       jeu7_matrix[:,idx_data] = prot_data[name_prot]["Jeu_7_rep4"]["pts"]
       idx_data +=1
    
    # JEU 8 ###################################################################
    names_prot = ["BMAL1", "CRY"]
    len_jeu8_timepoints = len(prot_data["BMAL1"]["Jeu_8"]["cts"])
    jeu8_timepoints = prot_data["BMAL1"]["Jeu_8"]["cts"]
    jeu8_matrix = np.zeros([len_jeu8_timepoints, len(names_prot)])
    for idx_data, name_prot in enumerate(names_prot) : 
       jeu8_matrix[:,idx_data] = prot_data[name_prot]["Jeu_8"]["pts"]

    # JEU 10 ###################################################################
    names_prot = ["BMAL1", "CRY", "REV-ERB"]
    len_jeu10_timepoints = len(prot_data["BMAL1"]["Jeu_10"]["cts"])
    jeu10_timepoints = prot_data["BMAL1"]["Jeu_10"]["cts"]
    jeu10_matrix = np.zeros([len_jeu10_timepoints, len(names_prot)])
    for idx_data, name_prot in enumerate(names_prot) : 
       jeu10_matrix[:,idx_data] = prot_data[name_prot]["Jeu_10"]["pts"]
    
    return jeu2_matrix, jeu2_timepoints, jeu3_matrix, jeu3_timepoints, jeu4_matrix, jeu4_timepoints, jeu5_matrix, jeu5_timepoints, jeu7_matrix, jeu7_timepoints, jeu8_matrix, jeu8_timepoints, jeu10_matrix, jeu10_timepoints



def get_parameter_bounds_percent(params, percent):

    """
    Determine the parameter lower and upper bounds based on the SW480 circadian
    clock model
    
    params : list of the circadian clock model parameters 
    percent : percentage used to calculate the bounds
    """

    n_params = len(params)
    params_min, params_max = np.zeros(n_params), np.zeros(n_params)
    for i in range(n_params):  
        params_min[i] = params[i] - abs(params[i])*0.99
        params_max[i] = params[i] + abs(params[i])*percent
            
    return params_min, params_max


def get_parameter_bounds(n_params, model):

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
    # coordinates are diff for
    # kex1, kex2 & V3max, V4max, V6max depending on model
    if model == 'liver':
        params_min[[56, 57]] = 1e-3
        params_max[[56, 57]] = 1
    elif model == 'sw':
        params_min[[56, 57, 58]] = 1e-13
        params_max[[56, 57, 58]] = 5e-7
        params_min[[60, 61]] = 1e-3
        params_max[[60, 61]] = 1
    params_min[59] = 11.9
    params_max[59] = 12.1
    return params_min, params_max




def get_random_parameters(fixed_params, params, params_min, params_max):

    """
    Get random parameters set between the interval and keep the original 
    values when the parameter is fixed
    
    fixed_params : list of the fixed parameters names
    params : list of the original circadian clock model parameters
    params_min : list of the minimum boundaries of the parameters
    params_max : list of the maximum boundaries of the parameters
    """

    params_rand = [np.random.uniform(params_min[i], params_max[i]) if i not in fixed_params else params[i] for i in range(len(params))]
        
    return params_rand


def bounds_reach(params, params_min, params_max):
    
    """
    Verify if the parameters estimation reaches the boundaries of the interval
    
    params : list of the parameters obtained with CMA-ES
    params_min : list of the minimum boundaries of the parameters
    params_max : list of the maximum boundaries of the parameters
    """
    
    lp = len(params)
    ind_reached = []
    
    for i in range(lp):
        if params[i] == params_min[i] or params[i] == params_max[i] :
            ind_reached.append(i)
        
    return ind_reached
    
    
def new_preprocessing_rna_seq_data(rna_seq_data):

    paralogs = ["ARNTL", "CLOCK", "CRY1", "CRY2", "NR1D1", "NR1D2", "PER1", "PER2", "PER3", "RORA", "RORB"]
    names = ["BMAL1", "CLOCK", "CRY", "REV-ERB", "PER", "ROR"]
    idx = [[0], [1], [2, 3], [4, 5], [6, 7, 8], [9, 10]]
    GENES, CTs = [], []

    for i in range(len(names)):
        concat = np.array([rna_seq_data["ctrl"][paralogs[j]]["pts"] for j in idx[i]])
        nansum = np.nansum(concat, axis=0) # multiply this by 10**3 for pmol/L visualization
        gene, ct = [], []
        print(nansum.shape)
        for l in range(nansum.shape[0]):
            gene.append([])
            for k in range(nansum.shape[1]):
                # if we have a zero in the nansum: either it's a true zero, or it's a sum of nans.
                if not nansum[l, k]:
                    if not np.isnan(concat[:, l, k]).all():
                        gene[l].append(nansum[l, k])
                        ct.append(rna_seq_data["CTs"][l])
                else:
                    gene[l].append(nansum[l, k])
                    ct.append(rna_seq_data["CTs"][l])
        GENES.append(gene)
        CTs.append(ct)
    return GENES, CTs