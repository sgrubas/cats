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
from cats.core.association import MatchSequences
from .linalg import find_word_starting_with
import re
from functools import partial


_EPS = 1e-9
num_pattern = r"-?\d+\.?\d*"


@ReshapeArraysDecorator(dim=2, input_num=2, output_num=0)
def binary_crossentropy(y_true, y_pred, /, eps=1e-32):
    bce = y_true * np.log(y_pred + eps) + (1.0 - y_true) * np.log(1.0 - y_pred + eps)
    bce = -bce.mean(axis=-1)
    return bce


@nb.njit("f8(i8[:], i8[:], f8)")
def _f_beta_raw(y_true, y_pred, beta):
    true_ids = (y_true == 1)
    det = y_pred[true_ids]
    TP = det.sum()
    FN = len(det) - TP
    FP = y_pred[~true_ids].sum()

    Precision = TP / (TP + FP + _EPS)
    Recall = TP / (TP + FN + _EPS)

    f = (1 + beta**2) * Precision * Recall
    denom = (Precision * beta**2 + Recall) + _EPS

    return f / denom


@ReshapeArraysDecorator(dim=2, input_num=2)
def f_beta_raw(y_true, y_pred, beta, /):
    return _f_beta_raw(y_true.astype(np.int64), y_pred.astype(np.int64), beta)


def f_beta(recall, precision, beta):
    f = (1 + beta**2) * precision * recall
    denom = (precision * beta**2 + recall) + _EPS
    return f / denom


def f_beta_on_picks(picks_true, picks_pred, max_time_dist=2.5, beta=0.5):
    Recall, Precision = recall_precision_picks(picks_true=picks_true, picks_pred=picks_pred,
                                               max_time_dist=max_time_dist)
    return f_beta(Recall, Precision, beta)


def true_false_missed_picks(picks_true, picks_pred, max_time_dist=2.5):
    if picks_pred.dtype.name == 'object':
        shape = picks_pred.shape
    else:
        shape = picks_pred.shape[:-1]
    TP, FP, FN = 0, 0, 0
    for ind in np.ndindex(*shape):
        onsets = np.array(picks_pred[ind], ndmin=1)
        ref_onsets = np.array(picks_true[ind], ndmin=1)
        matched = MatchSequences(ref_onsets, onsets, max_dist=max_time_dist, verbose=False)
        if len(matched.shape) == 1:
            matched = matched.reshape(2, 1)
        detection_status = np.isnan(matched.T)
        tp = (~detection_status).prod(axis=-1).sum()
        fp, fn = detection_status.sum(axis=0)
        TP += tp
        FP += fp
        FN += fn

    return TP, FP, FN


@nb.njit("UniTuple(i8, 3)(f8[:, :], f8[:])",
         parallel=True, cache=True)
def true_false_missed_intervals(pred_intervals, true_picks):
    """ The metric is NOT for tuning, it focuses on recall only,
        because the best result can be easily achieved by a single huge interval
    """
    interval_events = np.zeros(len(pred_intervals), dtype=np.int64)

    for pi in true_picks:
        for j, intv in enumerate(pred_intervals):
            inside = (intv[0] <= pi) & (pi <= intv[1])
            if inside:
                interval_events[j] += 1
                break  # only the first interval covering the event

    TP = np.sum(interval_events)  # true detections
    FP = np.sum(interval_events == 0)  # false detections
    FN = len(true_picks) - TP  # missed events

    return TP, FP, FN


def recall_precision_picks(picks_true, picks_pred, max_time_dist=2.5):
    TP, FP, FN = true_false_missed_picks(picks_true, picks_pred, max_time_dist)
    Precision = TP / (TP + FP + _EPS)
    Recall = TP / (TP + FN + _EPS)
    return Recall, Precision


def binary_metric_func(metric_name):
    picks = find_word_starting_with(metric_name, "pick", case_insensitive=True)
    f_score = find_word_starting_with(metric_name, "f", case_insensitive=True)
    crossentropy = bool(find_word_starting_with(metric_name, "crossentr", case_insensitive=True))
    crossentropy = crossentropy or bool(find_word_starting_with(metric_name, "entropy", case_insensitive=True))
    Re_Pr = find_word_starting_with(metric_name, "recall", case_insensitive=True)
    Re_Pr = Re_Pr or find_word_starting_with(metric_name, "precision", case_insensitive=True)

    if f_score:  # if F score is needed
        beta = float(re.findall(num_pattern, f_score[0])[0])
        if picks:  # if only picks are provided
            picks_proximity_sec = float(re.findall(num_pattern, picks[0])[0])
            err_func = partial(f_beta_on_picks, max_time_dist=picks_proximity_sec, beta=beta)
        else:  # If binary sequence is provided (True - event, False - noise)
            err_func = partial(f_beta_raw, beta=beta)
    elif Re_Pr:  # if Recall & Presision are needed
        if picks:  # if only picks are provided
            picks_proximity_sec = float(re.findall(num_pattern, picks[0])[0])
            err_func = partial(recall_precision_picks, max_time_dist=picks_proximity_sec)
        else:  # If binary sequence is provided (True - event, False - noise)
            raise NotImplementedError("This metric hasn't been implemented yet")
    elif crossentropy:  # if crossentropy is needed, works if binary sequence is provided (True - event, False - noise)
        err_func = binary_crossentropy
    else:
        raise ValueError(f"Unknown metric given {metric_name}, only 'F', 'crossentropy', or 'picks F'")

    return err_func


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
