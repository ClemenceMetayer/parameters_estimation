"""Utils file for numba-related functions only."""

import numpy as np

from numba import jit


def make_2D_array(lis):

    """
    Function to get 2D array from a list of lists.
    Prevents from using reflected lists.
    """

    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = np.max(lengths)
    arr = np.zeros((n, max_len), dtype=np.int32)
    for i in range(n):
        arr[i, :lengths[i]] = lis[i]
    return arr, lengths

# see here for more details
# https://github.com/numba/numba/issues/1269#issuecomment-472574352


@jit(nopython=True)
def np_max(array, axis):
    return np_apply_along_axis(np.max, axis, array)


@jit(nopython=True)
def np_min(array, axis):
    return np_apply_along_axis(np.min, axis, array)


@jit(nopython=True)
def np_sum(array, axis):
    return np_apply_along_axis(np.sum, axis, array)


@jit(nopython=True)
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)


@jit(nopython=True)
def np_apply_along_axis(func1d, axis, arr):
    # assert arr.ndim == 2
    # assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result
