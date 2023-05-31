import numpy as np

from numba import jit
from scipy.integrate import odeint

from utils_numba import np_max, np_min


@jit(nopython=True)
def clock_model_sw(y, t, p):

    """
    Variables:
    y = [x1, x2, x5, x6,
         y1, y2, y3, y4, y5,
         z1, z2, z4, z5, z6, z7, z8, z9, y6]

    Species:
    y = [CLOCK/BMAL_N, PER/CRY_N, REV-ERB_N, ROR_N,
         Per, Cry, Rev-Erb, Ror, Bmal1,
         CRY_C, PER_C, PER/CRY_C, CLOCK_C, REV-ERB_C, ROR_C,
         BMAL_C, CLOCK/BMAL_C, Clock]

    Parameter list description in supp material.
    Parameter order can be found in the example notebook.
    """

    # swcells are colon cells:
    vol = .8 / .2

    dx1_dt = p[45] * y[16] - (p[0] + p[60]) * y[0]
    dx2_dt = p[42] * y[11] - (p[1] + p[61]) * y[1]
    dx5_dt = p[43] * y[13] - p[2] * y[2]
    dx6_dt = p[44] * y[14] - p[3] * y[3]

    dy1_dt = p[20] / (1 + (y[1] / p[24])**p[51] * (y[0] / p[23])**p[52] + (y[0] / p[23])**p[52]) * (1 + p[46] * (y[0] / p[23])**p[52]) - p[4] * y[4]
    dy2_dt = p[21] / (1 + (y[1] / p[26])**p[53] * (y[0] / p[25])**p[54] + (y[0] / p[25])**p[54]) * (1 + p[47] * (y[0] / p[25])**p[54]) * (1 / (1 + (y[2] / p[27])**p[55])) - p[5] * y[5]
    dy3_dt = p[56] / (1 + (y[1] / p[29])**p[51] * (y[0] / p[28])**p[52] + (y[0] / p[28])**p[52]) * (1 + p[48] * (y[0] / p[28])**p[52]) - p[6] * y[6]
    dy4_dt = p[57] / (1 + (y[1] / p[31])**p[51] * (y[0] / p[30])**p[52] + (y[0] / p[30])**p[52]) * (1 + p[49] * (y[0] / p[30])**p[52]) - p[7] * y[7]
    dy5_dt = p[22] / (1 + (y[2] / p[33])**p[51] + (y[3] / p[32])**p[52]) * (1 + 12 * (y[3] / p[32])**p[52]) - p[8] * y[8]
    dy6_dt = p[58] / (1 + (y[2] / p[35])**p[51] + (y[3] / p[34])**p[52]) * (1 + p[50] * (y[3] / p[34])**p[52]) - p[9] * y[17]

    dz1_dt = p[37] * y[5] + p[19] * y[11] - (p[18] * y[10] + p[10]) * y[9]
    dz2_dt = p[36] * y[4] + p[19] * y[11] - (p[18] * y[9] + p[11]) * y[10]
    dz4_dt = p[18] * y[9] * y[10] + vol * p[61] * y[1] - (p[42] * vol + p[19]) * y[11]
    dz5_dt = p[41] * y[17] + p[17] * y[16] - (p[16] * y[15] + p[12]) * y[12]
    dz6_dt = p[38] * y[6] - (p[43] * vol + p[13]) * y[13]
    dz7_dt = p[39] * y[7] - (p[44] * vol + p[14]) * y[14]
    dz8_dt = p[40] * y[8] + p[17] * y[16] - (p[16] * y[12] + p[15]) * y[15]
    dz9_dt = p[16] * y[15] * y[12] + vol * p[60] * y[0] - (vol * p[45] + p[17]) * y[16]
    dXdt = [dx1_dt, dx2_dt, dx5_dt, dx6_dt,
            dy1_dt, dy2_dt, dy3_dt, dy4_dt, dy5_dt,
            dz1_dt, dz2_dt, dz4_dt, dz5_dt, dz6_dt, dz7_dt,
            dz8_dt, dz9_dt, dy6_dt]
    return dXdt



@jit(nopython=True)
def compute_loss(Y, arn_seq, arn_seq_max, t_interp_rna):

    """
    Loss function computation. sum of squared errors. the errors for gene j
    are divided by gene_j maximum value across time
    so that each gene weights the same however highly expressed they are.
    """
    # loss divided by number of points: equal importance pcr or micro
    
    ls = np.sum(((arn_seq - Y[:, np.array([8, 17, 5, 6, 4, 7])][t_interp_rna]) / arn_seq_max)**2) / len(t_interp_rna)
    return ls


@jit(nopython=True)
def constraints_and_loss(Y, arn_seq, arn_seq_max,
                         w, t_interp_rna, t):

    """
    Check for several constraints and compute loss. Constraints derived
    from the literature are described in the supp material. The others ensure
    the model behaves nicely. These are:
    - Periodicity 18<P<30
    - Relative amplitudes ((x_max - x_min) / x_max) > 0.05
    - Species concentration values not unrealistic (for unconstrained species).
    """
    
    ma = np_max(Y[-601:], 0)
    if ma.max() == 0:
        return np.random.normal(1e10, 1e8)

    loss = compute_loss(Y, arn_seq, arn_seq_max, t_interp_rna)
    # do not care about constraint until the loss is low enough.
    if loss < 1e4:
        # cb_max = CLOCK/BMAL_C + CLOCK/BMAL_N
        cb_max = w[0] * ma[0] + w[-2] * ma[-2]
        # clock_tot = CLOCK/BMAL_C + CLOCK/BMAL_N + CLOCK_C
        clock_tot = w[-6] * ma[-6] + cb_max
        #  per_tot = PER/CRY_C + PER_C
        per_tot = ma[11] + ma[10]

        mi = np_min(Y[-601:], 0)
        amp = (ma - mi) / ma
        p = np.zeros(Y.shape[1])
        for j in range(Y.shape[1]):
            p[j] = indiv_period_comp(Y[-601:, j], t[-601:])
        a = np.array([.15*clock_tot, cb_max, 0.5 * per_tot, 5e-2, 1e-14, ma.max()])
        b = np.array([cb_max, .85 * clock_tot, ma[11], amp.min(), mi.min(), 1e-5])
        #a = np.array([5e-2, 1e-14, ma.max()])
        # period should be between 14 and 35 (large)
        a = np.append(a, 14 * np.ones(Y.shape[1]))
        a = np.append(a, p)
        #b = np.array([amp.min(), mi.min(), 1e-5])
        b = np.append(b, p)
        b = np.append(b, 35 * np.ones(Y.shape[1]))
        c = new_transform_constraint_penalty(a, b)

        full_loss = loss + c
    else:
        full_loss = loss
    return full_loss


def fitness_sw(params, f, arn_seq, arn_seq_max, w, t_interp_rna, t, y0):

    """The fitness function called by CMA-ES at each feval."""

    Y = odeint(f, y0, t, args=(params,), rtol=1e-12, atol=1e-12)
    if check_nanneg(Y):
        return 1e100

    loss = constraints_and_loss(Y, arn_seq, arn_seq_max,
                                w, t_interp_rna, t)
    return loss


def wrapper_fit(params, params_max, params_min, scaler, fitness, *args):

    """
    Wrapper for fitness function + parameter scaling.
    This is done so that CMA-ES draws points to evaluate the fitness
    function with a normal distribution whose variances along each axis
    are somehow equivalent. This is not the case when you have 2 parameters
    that can have several orders of magnitude of differences in their values.
    """

    params = scaler(params, params_max, params_min)
    return fitness(params, *args)


@jit(nopython=True)
def from_log_010_to_ab(x, xmax, xmin):

    """
    Log space scaling.
    See http://cma.gforge.inria.fr/cmaes_sourcecode_page.html for a explanation.
    """

    return xmin * (xmax / xmin) ** (x / 10)


@jit(nopython=True)
def indiv_period_comp(Y, t, thresh=1.5):

    m = np.empty(0)
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
        # problematic if you have peaks at 24h and peaks at 12 though
        mb = np.min(b)
        if mb > 0:
            if np.max(b) / mb < thresh:
                per = mean
            else:
                per = 0
        else:
            per = 0
    else:
        per =0
    return per


@jit(nopython=True)
def check_nanneg(Y):

    """Check whether or not a solution Y contains negative or nan values."""

    for yrow in Y:
        for yval in yrow:
            if yval != yval or yval < 0 :
                return 1


@jit(nopython=True)
def new_transform_constraint_penalty(a, b):

    """Create a penalty term from a list of constraints.
    syntax is such that if a > b then penalize
    """
    
    return (np.maximum(a-b, 0)**2).sum()
