"""
    Metrics for evaluating detection quality with respect to true answers.
    Main metrics:
        Binary crossentropy : classical crossentropy for binary classification
        Error On Intervals : flexible evaluation of detected intervals, accounts for False Negatives, False Positives,
                             True positives, True Negatives (confusion matrix)
        Confusion matrix : computes confusion matrix

"""

import numpy as np
import numba as nb
from cats.core.utils import ReshapeArraysDecorator


_EPS = 1e-9


@nb.njit("f8[:](i8[:], i8[:])")
def _BinaryCrossentropy(y_true, y_pred):
    bce = y_true * np.log(y_pred + _EPS)
    bce += (1 - y_true) * np.log(1 - y_pred + _EPS)
    return -bce


@nb.njit("f8[:, :](i8[:, :], i8[:, :])", parallel=True)
def _BinaryCrossentropyN(y_true, y_pred):
    M = y_true.shape[0]
    bce = np.empty(y_true.shape)
    for i in nb.prange(M):
        bce[i] = _BinaryCrossentropy(y_true[i], y_pred[i])
    return bce


@ReshapeArraysDecorator(dim=2, input_num=2)
def BinaryCrossentropy(y_true, y_pred, /):
    return _BinaryCrossentropyN(y_true.astype(np.int64), y_pred.astype(np.int64))


""" Classification labels """
_FN = -1  # False Negative  (missed event)
_TP = 1   # True Positive   (detected event)
_FP = 2   # False Positive  (false alarm)
_TN = 0   # True Negative   (detected noise)


@nb.njit("UniTuple(i8, 3)(i8[:], f8)")
def _intervalFalseNeg(classes, fn_max):
    n = len(classes)
    fn_count = np.count_nonzero(classes == _FN)
    if fn_count > fn_max * n:
        label = _FN
        fn, tp = 1, 0
    else:
        label = _TP
        fn, tp = 0, 1
    return label, fn, tp


@nb.njit("UniTuple(i8, 3)(i8[:], i8, i8)")
def _intervalFalsePos(classes, fp_max, overlap):
    fp_count = np.count_nonzero(classes == _FP)
    if (fp_count > fp_max) or (overlap == 0):
        label, fp, tn = _FP, max(fp_count // fp_max, 1), 0
    else:
        label, fp, tn = _TN, 0, (overlap == 2) * 1
    return label, fp, tn


@nb.njit("UniTuple(i8[:], 2)(b1[:], b1[:], f8, i8)")
def _evaluateDetection(y_true, y_pred, fn_max, fp_max):
    c = np.hstack((y_pred * 2 - y_true, np.array([0])))
    N = len(c)
    R = np.array([0, 0, 0, 0])  # placeholder for result with [FN, TP, FP, TN]
    classified = np.full(N - 1, 0)
    i1 = i2 = 0
    ci_ = c[0]
    group = [_FN, _TP]
    for i in range(1, N):
        ci = c[i]
        if ((ci in group) and (ci_ in group)) or (ci == ci_):
            i2 = i
        else:
            if ci_ in group:
                label, fn, tp = _intervalFalseNeg(c[i1: i2 + 1], fn_max)
                R[0] += fn
                R[1] += tp
            elif ci_ == _FP:
                label, fp, tn = _intervalFalsePos(c[i1: i2 + 1], fp_max, (c[i1 - 1] == _TP) + (ci == _TP))
                R[2] += fp
                R[3] += tn
            else:
                label = _TN
                R[3] += 1
            classified[i1: i2 + 1] = label
            i1 = i2
            ci_ = ci
    return classified, R


@nb.njit("UniTuple(i8[:, :], 2)(b1[:, :], b1[:, :], f8, i8)")
def _evaluateDetectionN(y_true, y_pred, fn_max, fp_max):
    M, N = y_true.shape
    R = np.empty((M, 4), dtype=np.int64)
    labels = np.empty((M, N), dtype=np.int64)
    for i in nb.prange(M):
        labels_i, R_i = _evaluateDetection(y_true[i], y_pred[i], fn_max, fp_max)
        labels[i] = labels_i
        R[i] = R_i
    return labels, R


@ReshapeArraysDecorator(dim=2, input_num=2, methodfunc=False, output_num=2, first_shape=True)
def EvaluateDetection(y_true, y_pred, /, fn_max, fp_max):
    labels, R = _evaluateDetectionN(y_true, y_pred, fn_max, fp_max)
    return labels, R