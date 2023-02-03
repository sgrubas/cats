import numpy as np
import numba as nb
from .utils import ReshapeInputArray
from .projection import _giveIntervals


@nb.njit(["f8[:, :](f8[:, :], f8[:, :], b1[:, :], i8[:, :])",
          "f4[:, :](f4[:, :], f8[:, :], b1[:, :], i8[:, :])"], parallel=True)
def _WienerNaive(PSD, Sgm, Mask, frames):
    A = np.zeros_like(PSD)
    n = len(frames)
    M, N = PSD.shape

    for i in nb.prange(M):
        for j in range(n):
            j1, j2 = frames[j]
            sgm_j = Sgm[i, j]
            intervals = _giveIntervals(Mask[i, j1: j2 + 1])
            for j11, j22 in intervals:
                psd_ij = PSD[i, j1 + j11: j1 + j22 + 1]
                A[i, j1 + j11: j1 + j22 + 1] = psd_ij ** 2 / (psd_ij ** 2 + (j22 - j11) * sgm_j ** 2)
    return A


@ReshapeInputArray(dim=2, num=3, methodfunc=False)
def WienerNaive(PSD, Sgm, Mask, frames):
    return _WienerNaive(PSD, Sgm, Mask, frames)