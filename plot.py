""" Plot of the results for the parameters set """

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from functions import clock_model
from utils_plot import get_mean_std_hbec

# IMPORTATION ##################################################################################

params = np.loadtxt('results/example_new _cmaes_result_14.867930843313475.txt')[0]
tspan = np.linspace(0, 600, 6001)
y0 = np.loadtxt('data/y0.txt')


with open('data/Jeu_9/data/data_dict_concentration.dat', 'rb') as fun:
    data_hbec_rna = pkl.load(fun)
    
with open("data/dict_comparison", "rb") as f:
    data_hbec_prot = pkl.load(f)
      
    

# SOLUTION OF THE ODE ##########################################################################
Y = odeint(clock_model, y0, tspan, args=(params,), rtol=10**(-12), atol=10**(-12)) 
Y_plot = Y[-601:]*10**12


# PLOT RNA-SEQUENCING ##########################################################################

names_genes = ['PER', 'CRY', 'REV-ERB', 'ROR', 'BMAL1', 'CLOCK']
gene_coords = [4, 5, 6, 7, 8, 17] 
t_plot = np.linspace(0, 60, 601)



count, count2 = 0, 0
for i, j in enumerate(gene_coords):
    fig = plt.figure(dpi=300)
    mean_vector, std_vector = get_mean_std_hbec(names_genes[i], data_hbec_rna)
    plt.errorbar(data_hbec_rna["CTs"], mean_vector, std_vector, fmt='o', linewidth=1, capsize=3, mfc= "#283F53", mec = "#283F53", ecolor ="#283F53")
    plt.plot(t_plot, Y_plot[:, j], color='purple', label='sw480 clock model', linewidth=2.5)
    plt.title(names_genes[i], fontsize=20)
    plt.ylabel('Gene Expression (pmol/L)', fontsize=18)
    plt.xlabel('CT (h)', fontsize=18)
    plt.savefig('results/simu_RNA_Seq_'+str(names_genes[i])+'.png', dpi=300)
    plt.show()


# PLOT PROTEINS #############################################################################

Y_plot = Y[-601:]*10**9

# Weights vector initialisation 
vc, vn = 0.72, 0.28
w = np.zeros(y0.shape[0])
w[0:4] = vn
w[4:9] = 1
w[9:] = vc
w[-1] = 1

data_Y_plot = {}
# BMAL1 = CLOCK/BMAL_N + BMAL_C + CLOCK/BMAL_C
data_Y_plot["BMAL1"] = w[0]*Y_plot[:,0] + w[15]*Y_plot[:,15] + w[16]*Y_plot[:,16]

# CRY = PER/CRY_N + CRY_C + PER/CRY_C
data_Y_plot["CRY"] = w[1]*Y_plot[:,1] + w[9]*Y_plot[:,9] + w[11]*Y_plot[:,11] 

# REV-ERB = REV-ERB_N + REV-ERB_C
data_Y_plot["REV-ERB"] = w[2]*Y_plot[:,2] + w[13]*Y_plot[:,13] 

# BMAL1_ N = CLOCK/BMAL_N
data_Y_plot["BMAL1_N"] = Y_plot[:,0]

# CRY_N = PER/CRY_N
data_Y_plot["CRY_N"] = Y_plot[:,1]

# CRY_C = CRY_C + PER/CRY_C
data_Y_plot["CRY_C"] =  Y_plot[:,9] + Y_plot[:,11] 

# PER_N = PER/CRY_N
data_Y_plot["PER_N"] = Y_plot[:,1]

# PER =  PER/CRY_N + PER_C + PER/CRY_C
data_Y_plot["PER"] = w[1]*Y_plot[:,1] + w[10]*Y_plot[:,10] + w[11]*Y_plot[:,11] 

# CLOCK = CLOCK/BMAL_N + CLOCK_C + CLOCK/BMAL_C
data_Y_plot["CLOCK"] = w[0]*Y_plot[:,0] + w[12]*Y_plot[:,12] + w[16]*Y_plot[:,16]

# ROR = ROR_N + ROR_C 
data_Y_plot["ROR"] = w[3]*Y_plot[:,3] + w[14]*Y_plot[:,14]


t_plot = np.linspace(0, 60, 601)



for name_prot in data_hbec_prot.keys() :
    cts_lengths = [len(jeu_data['cts']) if isinstance(jeu_data, dict) and 'cts' in jeu_data else 0 for jeu_data in data_hbec_prot[name_prot].values()]
    longest_cts_index = np.argmax(cts_lengths)
    longest_cts_list = data_hbec_prot[name_prot][list(data_hbec_prot[name_prot].keys())[longest_cts_index]]['cts']

    len_cts_list = len(longest_cts_list)
    
    plt.figure(dpi=300)
    plt.clf()
    plt.errorbar(longest_cts_list,  data_hbec_prot[name_prot]["mean"], yerr= data_hbec_prot[name_prot]["std"], linestyle='None', marker='o', color ="#283F53", capsize=3, capthick=1)
    plt.plot(t_plot, data_Y_plot[name_prot], color='purple', label='sw480 clock model', linewidth=2.5)
    plt.xlabel('Time (Circadian times)')
    plt.ylabel('Protein concentration (nmol/L)')
    plt.title(str(name_prot))
    plt.savefig('results/simu_prot_'+str(name_prot)+'.png', dpi = 300)
    plt.show()



