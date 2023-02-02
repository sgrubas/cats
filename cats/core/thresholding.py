import numpy as np
import numba as nb
from .utils import ReshapeInputArray

################### THRESHOLDING ###################


@nb.njit(["b1[:, :](f8[:, :], f8[:, :], i8[:, :])",
          "b1[:, :](f4[:, :], f8[:, :], i8[:, :])"], parallel=True)
def _Thresholding(PSD, Eta, frames):
    B = np.empty_like(PSD, dtype=np.bool_)
    M = len(frames)
    N = len(PSD)
    for j in nb.prange(M):
        j1, j2 = frames[j]
        for i in range(N):
            B[i, j1 : j2] = PSD[i, j1 : j2] > Eta[i, j]
    return B


@ReshapeInputArray(dim=2, num=2, methodfunc=False)
def Thresholding(PSD, Eta, frames):
    return _Thresholding(PSD, Eta, frames)