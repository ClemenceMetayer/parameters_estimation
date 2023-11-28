""" Various useful functions for the CMA-ES algorithm run and the cost calculation """

import numpy as np
from numba import jit
from scipy.integrate import odeint

from utils_numba import np_max, np_min


@jit(nopython=True)
def clock_model(y, t, p):
    """
    Definition of the ODE dynamical system of the circadian clock

    Variables:
    y = [x1, x2, x5, x6,
         y1, y2, y3, y4, y5,
         z1, z2, z4, z5, z6, z7, z8, z9, y6]

    Species:
    y = [CLOCK/BMAL_N, PER/CRY_N, REV-ERB_N, ROR_N,
         Per, Cry, Rev-Erb, Ror, Bmal1,
         CRY_C, PER_C, PER/CRY_C, CLOCK_C, REV-ERB_C, ROR_C,
         BMAL_C, CLOCK/BMAL_C, Clock]

    y : list of the state variables values
    t : time vector
    p : parameters vector

    """

    vol = 0.72 / 0.28
    # vol = .8 / .2

    dx1_dt = p[45] * y[16] - (p[0] + p[60]) * y[0]
    dx2_dt = p[42] * y[11] - (p[1] + p[61]) * y[1]
    dx5_dt = p[43] * y[13] - p[2] * y[2]
    dx6_dt = p[44] * y[14] - p[3] * y[3]

    dy1_dt = (
        p[20]
        / (
            1
            + (y[1] / p[24]) ** p[51] * (y[0] / p[23]) ** p[52]
            + (y[0] / p[23]) ** p[52]
        )
        * (1 + p[46] * (y[0] / p[23]) ** p[52])
        - p[4] * y[4]
    )
    dy2_dt = (
        p[21]
        / (
            1
            + (y[1] / p[26]) ** p[53] * (y[0] / p[25]) ** p[54]
            + (y[0] / p[25]) ** p[54]
        )
        * (1 + p[47] * (y[0] / p[25]) ** p[54])
        * (1 / (1 + (y[2] / p[27]) ** p[55]))
        - p[5] * y[5]
    )
    dy3_dt = (
        p[56]
        / (
            1
            + (y[1] / p[29]) ** p[51] * (y[0] / p[28]) ** p[52]
            + (y[0] / p[28]) ** p[52]
        )
        * (1 + p[48] * (y[0] / p[28]) ** p[52])
        - p[6] * y[6]
    )
    dy4_dt = (
        p[57]
        / (
            1
            + (y[1] / p[31]) ** p[51] * (y[0] / p[30]) ** p[52]
            + (y[0] / p[30]) ** p[52]
        )
        * (1 + p[49] * (y[0] / p[30]) ** p[52])
        - p[7] * y[7]
    )
    dy5_dt = (
        p[22]
        / (1 + (y[2] / p[33]) ** p[51] + (y[3] / p[32]) ** p[52])
        * (1 + 12 * (y[3] / p[32]) ** p[52])
        - p[8] * y[8]
    )
    dy6_dt = (
        p[58]
        / (1 + (y[2] / p[35]) ** p[51] + (y[3] / p[34]) ** p[52])
        * (1 + p[50] * (y[3] / p[34]) ** p[52])
        - p[9] * y[17]
    )

    dz1_dt = p[37] * y[5] + p[19] * y[11] - (p[18] * y[10] + p[10]) * y[9]
    dz2_dt = p[36] * y[4] + p[19] * y[11] - (p[18] * y[9] + p[11]) * y[10]
    dz4_dt = p[18] * y[9] * y[10] + vol * p[61] * y[1] - (p[42] * vol + p[19]) * y[11]
    dz5_dt = p[41] * y[17] + p[17] * y[16] - (p[16] * y[15] + p[12]) * y[12]
    dz6_dt = p[38] * y[6] - (p[43] * vol + p[13]) * y[13]
    dz7_dt = p[39] * y[7] - (p[44] * vol + p[14]) * y[14]
    dz8_dt = p[40] * y[8] + p[17] * y[16] - (p[16] * y[12] + p[15]) * y[15]
    dz9_dt = p[16] * y[15] * y[12] + vol * p[60] * y[0] - (vol * p[45] + p[17]) * y[16]

    dXdt = [
        dx1_dt,
        dx2_dt,
        dx5_dt,
        dx6_dt,
        dy1_dt,
        dy2_dt,
        dy3_dt,
        dy4_dt,
        dy5_dt,
        dz1_dt,
        dz2_dt,
        dz4_dt,
        dz5_dt,
        dz6_dt,
        dz7_dt,
        dz8_dt,
        dz9_dt,
        dy6_dt,
    ]

    return dXdt


@jit(nopython=True)
def compute_loss(
    Y,
    arn_seq,
    arn_seq_max,
    prot,
    prot_max,
    w,
    t_interp_rna,
    t_interp_prot,
):
    """
    Compute the loss function using a sum of squared errors and a maximum
    weighting.

    Y : list of the solution of the ODE system values
    arn_seq : numpy array pf the RNA-Sequencing data
    arn_seq_max : list of the maximum values for each gene
    t_interp_rna : time points vector

    RNA-Seq : [BMAL1, CLOCK, CRY, REV-ERB, PER, ROR]
    Jeu 2 : [BMAL1, CRY]
    Jeu 3 : [CRY, REV-ERB]
    Jeu 4 : [BMAL1_N, CRY, CRY_N, CRY_C, PER_N]
    Jeu 5 : [CRY, CRY_N, CRY_C, PER]
    Jeu 7 : [PER, REV-ERB, CLOCK, ROR]
    Jeu 8 : [BMAL1, CRY]
    Jeu 10 : [BMAL1, CRY, REV-ERB]

    """
    # RNA - SEQUENCING ########################################################
    ls_rna = 0
    idx = [8, 17, 5, 6, 4, 7]
    # BMAL CLOCK CRY REV PER ROR
    for i in range(len(arn_seq)):
        ls_rna += np.sum(
            ((arn_seq[i] - Y[:, idx[i]][t_interp_rna[i]]) / arn_seq_max[i]) ** 2
        ) / len(t_interp_rna[i])

    # INITIALISATION OF THE DATA ##############################################
    # BMAL1 = CLOCK/BMAL_N + BMAL_C + CLOCK/BMAL_C
    bmal1_data = w[0] * Y[:, 0] + w[15] * Y[:, 15] + w[16] * Y[:, 16]

    # CRY = PER/CRY_N + CRY_C + PER/CRY_C
    cry_data = w[1] * Y[:, 1] + w[9] * Y[:, 9] + w[11] * Y[:, 11]

    # REV-ERB = REV-ERB_N + REV-ERB_C
    reverb_data = w[2] * Y[:, 2] + w[13] * Y[:, 13]

    # BMAL1_ N = CLOCK/BMAL_N
    bmal1n_data = Y[:, 0]

    # CRY_N = PER/CRY_N
    cryn_data = Y[:, 1]

    # CRY_C = CRY_C + PER/CRY_C
    cryc_data = Y[:, 9] + Y[:, 11]

    # PER_N = PER/CRY_N
    pern_data = Y[:, 1]

    # PER =  PER/CRY_N + PER_C + PER/CRY_C
    per_data = w[1] * Y[:, 1] + w[10] * Y[:, 10] + w[11] * Y[:, 11]

    # CLOCK = CLOCK/BMAL_N + CLOCK_C + CLOCK/BMAL_C
    clock_data = w[0] * Y[:, 0] + w[12] * Y[:, 12] + w[16] * Y[:, 16]

    # ROR = ROR_N + ROR_C
    ror_data = w[3] * Y[:, 3] + w[14] * Y[:, 14]

    # order BMAL1, BMAL1_N, CRY, CRY_N, CRY_C, REV-ERB, CLOCK, PER, PER_N, ROR]
    variables = [
        bmal1_data,
        bmal1n_data,
        cry_data,
        cryn_data,
        cryc_data,
        reverb_data,
        clock_data,
        per_data,
        pern_data,
        ror_data,
    ]
    ls_prot = 0
    for i in range(len(variables)):
        ls_prot += np.sum(
            ((prot[i] - variables[i][t_interp_prot[i]]) / prot_max[i]) ** 2
        ) / len(t_interp_prot[i])

    return ls_rna + ls_prot


@jit(nopython=True)
def constraints_and_loss(
    Y,
    arn_seq,
    arn_seq_max,
    prot,
    prot_max,
    w,
    t_interp_rna,
    t_interp_prot,
    t,
):
    """
    Check for several constraints and compute loss.

    Y : list of the solution of the ODE system values
    arn_seq : numpy array pf the RNA-Sequencing data
    arn_seq_max : list of the maximum values for each gene
    t_interp_rna : time points vector

    """

    ma = np_max(Y[-601:], 0)
    if ma.max() == 0:
        return np.random.normal(1e10, 1e8)

    loss = compute_loss(
        Y, arn_seq, arn_seq_max, prot, prot_max, w, t_interp_rna, t_interp_prot
    )
    # return loss
    # # do not care about constraint until the loss is low enough.
    # if loss < 1e4:
    #     # cb_max = CLOCK/BMAL_C + CLOCK/BMAL_N
    #     cb_max = w[0] * ma[0] + w[-2] * ma[-2]
    #     # clock_tot = CLOCK/BMAL_C + CLOCK/BMAL_N + CLOCK_C
    #     clock_tot = w[-6] * ma[-6] + cb_max
    #     #  per_tot = PER/CRY_C + PER_C
    #     per_tot = ma[11] + ma[10]

    #     mi = np_min(Y[-601:], 0)
    #     amp = (ma - mi) / ma
    #     a = np.array([0.15 * clock_tot, cb_max, 0.5 * per_tot, 1e-14, ma.max()])
    #     b = np.array([cb_max, 0.85 * clock_tot, ma[11], mi.min(), 1e-5])

    if loss < 1e4:
        mi = np_min(Y[-601:], 0)
        amp = (ma - mi) / ma
        a = np.array([1e-14, ma.max()])
        b = np.array([mi.min(), 1e-5])
        # period should be between 14 and 45 (large)
        # release the period and amplitude check once fit is good enough
        a = np.append(a, 5e-2)
        b = np.append(b, amp.min())
        p = np.zeros(Y.shape[1])
        for j in range(Y.shape[1]):
            p[j] = indiv_period_comp(Y[-601:, j], t[-601:])
        a = np.append(a, 14 * np.ones(Y.shape[1]))
        a = np.append(a, p)
        b = np.append(b, p)
        b = np.append(b, 45 * np.ones(Y.shape[1]))
        c = new_transform_constraint_penalty(a, b)

        full_loss = loss + c
    else:
        full_loss = loss

    return full_loss


def fitness(
    params,
    f,
    arn_seq,
    arn_seq_max,
    prot,
    prot_max,
    w,
    t_interp_rna,
    t_interp_prot,
    t,
    y0,
):
    """
    The fitness function called by CMA-ES at each feval.

    params : vector of parameters
    f : function that describe the ODEs to solve
    arn_seq : numpy array pf the RNA-Sequencing data
    arn_seq_max : list of the maximum values for each gene
    t_interp_rna : time points vector
    t : time points vector
    y0 : initial values of the sate variables

    """

    Y = odeint(f, y0, t, args=(params,), rtol=1e-12, atol=1e-12)
    if check_nanneg(Y):
        return 1e100

    loss = constraints_and_loss(
        Y,
        arn_seq,
        arn_seq_max,
        prot,
        prot_max,
        w,
        t_interp_rna,
        t_interp_prot,
        t,
    )

    return loss


def wrapper_fit(params, params_max, params_min, scaler, fitness, *args):
    """
    Wrapper for fitness function and parameter scaling.

    params : vector of parameters
    params_max : list of maximum boundaries for each parameter
    params_min : list of minimum boundaries for each parameter
    scaler : function for scaling
    fitness : fitnes function

    """

    params = scaler(params, params_max, params_min)
    return fitness(params, *args)


@jit(nopython=True)
def indiv_period_comp(Y, t, thresh=1.5):
    a, b = np.empty(0), np.empty(0)
    for i in range(1, len(Y) - 1):
        # just looking for a peak
        if (Y[i - 1] < Y[i]) and (Y[i] > Y[i + 1]):
            a = np.append(a, t[i])
            b = np.append(b, Y[i])
    pers = np.diff(a)
    mean = 0 if len(pers) == 0 else np.sum(pers) / len(pers)
    if b.size != 0:
        # this part checks whether the peaks are similar from one period to another
        # problematic if you have peaks      at 24h and peaks at 12 though
        mb = np.min(b)
        if mb > 0:
            if np.max(b) / mb < thresh:
                per = mean
            else:
                per = 0
        else:
            per = 0
    else:
        per = 0
    return per


@jit(nopython=True)
def from_log_010_to_ab(x, xmax, xmin):
    """
    Log space scaling.
    See http://cma.gforge.inria.fr/cmaes_sourcecode_page.html for a explanation.
    """

    return xmin * (xmax / xmin) ** (x / 10)


@jit(nopython=True)
def check_nanneg(Y):
    """
    Check whether or not a solution Y contains negative values.

    Y : solution of the ODE system

    """

    for yrow in Y:
        for yval in yrow:
            if yval != yval or yval < 0:
                return 1


@jit(nopython=True)
def new_transform_constraint_penalty(a, b):
    """Create a penalty term from a list of constraints.
    syntax is such that if a > b then penalize
    """
    return (np.maximum(a - b, 0) ** 2).sum()
