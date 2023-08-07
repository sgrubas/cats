"""
    Functions for projection detection from Time-Frequency onto Time.
"""

import numpy as np
import numba as nb
from .utils import ReshapeArraysDecorator


############################################################################
#####################  TIME PROJECTION AND FIILTERING  #####################
############################################################################


@nb.njit("i8[:, :](b1[:])", cache=True)
def _giveIntervals(detection):
    """
        Extracts time intervals of `True` values from boolean array.

        Arguments:
            detection : boolean np.ndarray (N,) : boolean array, assuming `True` is earthquake, `False` is noise

        Returns:
            intervals : np.ndarray (nt, 2) : time intervals, where `nt` is number of found intervals,
                                             e.g. if `nt = 2` then [[0, 1], [5, 8]]
    """
    N = len(detection)
    intervals = np.empty((N // 2 + 1, 2), dtype=np.int64)
    i1 = i2 = -1.0
    j = 0
    for i in range(N):
        d_i = detection[i]
        if d_i and (i < N - 1):
            if i1 < 0:
                i1 = i
            else:
                i2 = i
        elif d_i and (i == N - 1) and (i1 >= 0):
            intervals[j, 0] = i1
            intervals[j, 1] = i
            i1 = i2 = -1.0
            j += 1
        elif (not d_i) and (i1 >= 0):
            intervals[j, 0] = i1
            intervals[j, 1] = i1 if i2 < 0 else i2
            i1 = i2 = -1.0
            j += 1
    return intervals[:j]


def GiveIntervals(detection, /):
    intervals = np.empty(detection.shape[:-1], dtype=object)
    for i, _ in np.ndenumerate(intervals):
        intervals[i] = _giveIntervals(detection[i])
    return intervals


@nb.njit("b1[:, :](b1[:, :], i8, i8)", cache=True)
def _removeGaps_OLD(detection, min_separation, min_width):
    M, Nt = detection.shape
    filtered = np.full_like(detection, False)
    for i in nb.prange(M):
        intervals = _giveIntervals(detection[i])
        n = len(intervals)
        if (min_separation > 0) and (n > 0):
            buffer = [(intervals[0, 0], intervals[0, 1])]
            for j in range(1, n):
                intv_new = intervals[j]
                intv_old = buffer[-1]
                if ((intv_new[0] - intv_old[1] - 1) <= min_separation) and \
                   (((intv_new[1] - intv_new[0] + 1) < min_width) or
                   ((intv_old[1] - intv_old[0] + 1) < min_width)):

                    buffer[-1] = (intv_old[0], intv_new[1])
                else:
                    buffer.append((intv_new[0], intv_new[1]))
            buffer = np.array(buffer) if len(buffer) > 0 else np.zeros((0, 2), dtype=np.int64)
        else:
            buffer = intervals
        for j1, j2 in buffer:
            filtered[i, j1: j2 + 1] = True
    return filtered


@nb.njit(["f8[:, :](f8[:, :], f8, f8)", "f4[:, :](f4[:, :], f8, f8)",
          "i8[:, :](i8[:, :], i8, i8)", "i4[:, :](i4[:, :], i8, i8)"],
         cache=True, parallel=True)
def filter_intervals(intervals, min_separation, min_duration):
    if len(intervals > 1):
        # subsequent start - preceding end
        combine_inds = (intervals[1:, 0] - intervals[:-1, 1]) >= min_separation
        combine_inds = np.concatenate((np.array([True]),
                                       combine_inds,
                                       np.array([True])))  # include first start and last end

        combined_intervals = np.stack(
            (intervals[:, 0][combine_inds[:-1]],  # starts with separation `> min` from preceding
             intervals[:, 1][combine_inds[1:]]  # ends with separation `> min` from subsequent
             ), axis=-1)
    else:
        combined_intervals = intervals

    if len(combined_intervals) > 0:
        duration_inds = (combined_intervals[:, 1] - combined_intervals[:,
                                                    0]) >= min_duration - 1  # `-` is for single point
        return combined_intervals[duration_inds]
    else:
        return combined_intervals


def FilterIntervals(intervals, min_separation, min_duration):
    for ind, intv in np.ndenumerate(intervals):
        intervals[ind] = filter_intervals(intv, min_separation, min_duration)
    return intervals


@nb.njit(["(i8[:, :], b1[:])",
          "(i4[:, :], b1[:])"], cache=True, parallel=True)
def project_intervals(intervals, output_array):
    for i in nb.prange(len(intervals)):
        t1, t2 = intervals[i]
        output_array[t1: t2 + 1] = True


def FilterDetection(detection, min_separation, min_duration):
    filtered_intervals = np.empty(detection.shape[:-1], dtype=object)
    filtered_detection = np.full_like(detection, False, dtype=bool)
    for ind, _ in np.ndenumerate(filtered_intervals):
        interval_ind = _giveIntervals(detection[ind])
        filtered_intervals[ind] = filter_intervals(interval_ind, min_separation, min_duration)
        project_intervals(filtered_intervals[ind], filtered_detection[ind])

    return filtered_detection, filtered_intervals


@nb.njit(["b1[:](i8[:])", "b1[:](i4[:])", "b1[:](u2[:])", "b1[:](u4[:])"])
def _fill_gaps(labeled_sequence):
    binary_sequence = np.full_like(labeled_sequence, False, dtype=np.bool_)
    K = labeled_sequence.max()
    for k in range(1, K + 1):
        ids = np.argwhere(labeled_sequence == k)
        if len(ids) > 0:
            i1, i2 = np.min(ids), np.max(ids)
            binary_sequence[i1: i2 + 1] = True
    return binary_sequence


@nb.njit(["b1[:, :](i8[:, :])", "b1[:, :](i4[:, :])",
          "b1[:, :](u2[:, :])", "b1[:, :](u4[:, :])"], parallel=True)
def _fill_gaps_nd(labeled_sequences):
    M, N = labeled_sequences.shape
    binary_sequences = np.empty((M, N), dtype=np.bool_)
    for i in nb.prange(M):
        binary_sequences[i] = _fill_gaps(labeled_sequences[i])
    return binary_sequences


@ReshapeArraysDecorator(dim=2, input_num=1, output_num=1)
def FillGaps(labeled_sequences):
    return _fill_gaps_nd(labeled_sequences)
