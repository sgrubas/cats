"""
    Functions for trimming spectrograms (in Time-Frequency domain).
    Main functions:
        ...

"""

import numpy as np
import numba as nb
from .utils import ReshapeArraysDecorator

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
            B[i, j1 : j2 + 1] = PSD[i, j1 : j2 + 1] > Eta[i, j]
    return B


@ReshapeArraysDecorator(dim=2, input_num=2, methodfunc=False, output_num=1, first_shape=True)
def Thresholding(PSD, Eta, /, frames):
    return _Thresholding(PSD, Eta, frames)


@nb.njit(["b1[:, :](f8[:, :], f8[:, :], i8[:, :], f8)",
          "b1[:, :](f4[:, :], f8[:, :], i8[:, :], f8)"], parallel=True)
def _ThresholdingBySNR(PSD, Sgm, frames, minSNR):
    B = np.empty_like(PSD, dtype=np.bool_)
    M = len(frames)
    N = len(PSD)
    for j in nb.prange(M):
        j1, j2 = frames[j]
        for i in nb.prange(N):
            snr = PSD[i, j1: j2 + 1] / Sgm[i, j]
            B[i, j1: j2 + 1] = snr > minSNR
    return B


@ReshapeArraysDecorator(dim=2, input_num=2, methodfunc=False, output_num=1, first_shape=True)
def ThresholdingBySNR(PSD, Sgm, /, frames, minSNR):
    return _ThresholdingBySNR(PSD, Sgm, frames, minSNR)


@nb.njit(["f8[:, :](f8[:, :], f8[:, :], f8[:, :], i8[:, :])",
          "f8[:, :](f4[:, :], f8[:, :], f8[:, :], i8[:, :])"], parallel=True)
def _ThresholdingSNR(PSD, Sgm, Eta, frames):
    SNR = np.empty(PSD.shape)
    M = len(frames)
    N = len(PSD)
    for j in nb.prange(M):
        j1, j2 = frames[j]
        for i in nb.prange(N):
            psd = PSD[i, j1: j2 + 1]
            SNR[i, j1: j2 + 1] = (psd > Eta[i, j]) * psd / Sgm[i, j]
    return SNR


@ReshapeArraysDecorator(dim=2, input_num=3, methodfunc=False, output_num=1, first_shape=True)
def ThresholdingSNR(PSD, Sgm, Eta, /, frames):
    return _ThresholdingSNR(PSD, Sgm, Eta, frames)