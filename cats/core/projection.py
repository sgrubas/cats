"""
    Functions for projection detection from Time-Frequency onto Time.
    Main functions:
        GiveIntervals : extracts intervals of True from binary classification
    ...

"""

import numpy as np
import numba as nb
from .utils import ReshapeArraysDecorator


############################################################################
#####################  TIME PROJECTION AND FIILTERING  #####################
############################################################################


@nb.njit("i8[:, :](b1[:])")
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


@nb.njit("b1[:, :](b1[:, :], i8)")
def _removeGaps(detection, max_gap):
    M, Nt = detection.shape
    filtered = np.full_like(detection, False)
    for i in nb.prange(M):
        intervals = _giveIntervals(detection[i])
        n = len(intervals)
        if (max_gap > 0) and (n > 0):
            buffer = [(intervals[0, 0], intervals[0, 1])]
            for j in range(1, n):
                intv_new = intervals[j]
                intv_old = buffer[-1]
                if (intv_new[0] - intv_old[1]) > max_gap:
                    buffer.append((intv_new[0], intv_new[1]))
                else:
                    buffer[-1] = (intv_old[0], intv_new[1])
            buffer = np.array(buffer) if len(buffer) > 0 else np.zeros((0, 2), dtype=np.int64)
        else:
            buffer = intervals
        for j1, j2 in buffer:
            filtered[i, j1 : j2 + 1] = True
    return filtered

@ReshapeArraysDecorator(dim=2)
def RemoveGaps(detection, /, max_gap):
    return _removeGaps(detection, max_gap)


@nb.njit("f8[:, :](b1[:], f8[:])")
def _giveTimeIntervals(detection, time):
    """
        Extracts time intervals of `True` values from boolean array.

        Arguments:
            detection : boolean np.ndarray (N,) : boolean array, assuming `True` is earthquake, `False` is noise
            time : float np.ndarray (N,) : time axis to assign time values to the extracted intervals

        Returns:
            intervals : np.ndarray (nt, 2) : time intervals, where `nt` is number of found intervals,
                                             e.g. if `nt = 2` then [[0.0, 2.0], [5.0, 5.3]]
    """
    N = len(detection)
    intervals = np.empty((N // 2 + 1, 2))
    t1 = t2 = -1.0
    j = 0
    for i in range(N):
        d_i = detection[i]
        t_i = time[i]
        if d_i and (i < N - 1):
            if t1 < 0:
                t1 = t_i
            else:
                t2 = t_i
        elif d_i and (i == N - 1) and (t1 >= 0):
            intervals[j, 0] = t1
            intervals[j, 1] = t_i
            t1 = t2 = -1.0
            j += 1
        elif (not d_i) and (t1 >= 0):
            intervals[j, 0] = t1
            intervals[j, 1] = t1 if t2 < 0 else t2
            t1 = t2 = -1.0
            j += 1
    return intervals[:j]


@nb.njit("f8[:, :](f8[:, :], f8, f8)")
def _filterIntervals(intervals, max_gap, min_width):
    """
        Filters intervals by combining close intervals with gap <= `max_gap` and removing short intervals < `min_width`.

        Arguments:
            intervals : np.ndarray (nt, 2) : time intervals, where `nt` is number of found intervals,
            max_gap : float : maximum gap between two close intervals to be considered as one big interval
            min_width : float : minimum width of intervals to be kept

        Returns:
            filtered_intervals : np.ndarray (ntf, 2) : `ntf` filtered time intervals
    """
    n = len(intervals)
    if (max_gap > 0) and (n > 0):
        filtered = [(intervals[0, 0], intervals[0, 1])]
        for i in range(1, n):
            intv_new = intervals[i]
            intv_old = filtered[-1]
            if (intv_new[0] - intv_old[1]) > max_gap:
                if (intv_old[1] - intv_old[0]) < min_width:
                    filtered.pop(-1)
                filtered.append((intv_new[0], intv_new[1]))
            else:
                filtered[-1] = (intv_old[0], intv_new[1])
        if (len(filtered) > 0) and (filtered[-1][1] - filtered[-1][0]) < min_width:
            filtered.pop(-1)
        filtered = np.array(filtered) if len(filtered) > 0 else np.zeros((0, 2))
    else:
        filtered = intervals
    
    return filtered


@nb.njit("List(f8[:, :])(b1[:, :], f8[:], f8, f8)")
def _filterIntervalsN(detection, time, max_gap, min_width):
    """
        Extracts and filters time intervals of `True` values from boolean array.
        Filters by combining close intervals with gap <= `max_gap` and removing short intervals < `min_width`.

        Arguments:
            detection : boolean np.ndarray (M, N,) : `M` boolean arrays, assuming `True` is earthquake, `False` is noise
            time : float np.ndarray (N,) : time axis to assign time values to the extracted intervals
            max_gap : float : maximum gap between two close intervals to be considered as one big interval
            min_width : float : minimum width of intervals to be kept

        Returns:
            filtered_intervals : List(np.ndarray (ntf, 2)) : `ntf` filtered time intervals
    """
    intervals = []
    for di in detection:
        intervals.append(_filterIntervals(_giveTimeIntervals(di, time), max_gap, min_width))
    return intervals


def FilterIntervals(detection, time, max_gap, min_width):
    """
        Extracts and filters time intervals of `True` values from boolean array.
        Filters by combining close intervals with gap <= `max_gap` and removing short intervals < `min_width`.

        Arguments:
            detection : boolean np.ndarray (M, N,) : `M` boolean arrays, assuming `True` is earthquake, `False` is noise
            time : float np.ndarray (N,) : time axis to assign time values to the extracted intervals
            max_gap : float : maximum gap between two close intervals to be considered as one big interval
            min_width : float : minimum width of intervals to be kept

        Returns:
            filtered_intervals : List(np.ndarray (ntf, 2)) : `ntf` filtered time intervals
    """
    return _filterIntervalsN(detection, time, max_gap, min_width)


@nb.njit("b1[:](f8[:, :], i8, f8)")
def _projectIntervals(intervals, N, dt):
    """
        Projects `True` values onto time axis with sampling `dt` and length `N`.

        Arguments:
            intervals : np.ndarray (nt, 2) : `nt` time intervals
            N : int : number of elements in time axis
            dt : int : sampling of time axis

        Returns:
            detection : boolean np.ndarray (N,) : boolean array, `True` is earthquake, `False` is noise
    """
    b = np.full(N, False)
    for (t1, t2) in intervals:
        i1, i2 = int(t1 / dt), int(np.ceil(t2 / dt)) + 1
        b[i1 : i2] = True
    return b


@nb.njit("b1[:, :](b1[:, :], f8[:], f8, f8, f8[:])", parallel=True)
def _projectFilterIntervalsN(detection, time, max_gap, min_width, new_time):
    """
        Filters boolean array `detection` by combining close `True` intervals with gap <= `max_gap`
        and removing short intervals < `min_width`. Then, projects onto `new_time` axis.

        Arguments:
            detection : boolean np.ndarray (M, Nt) : `M` boolean arrays, `True` is earthquake, `False` is noise
            time : float np.ndarray (Nt,) : time axis for extraction of intervals
            max_gap : float : maximum gap between two close intervals to be considered as one big interval
            min_width : float : minimum width of intervals to be kept
            time : float np.ndarray (N,) : new time axis for projection of filtered intervals

        Returns:
            filtered_detection : boolean np.ndarray (M, N) : filtered and re-projected boolean array
    """
    dt = new_time[1] - new_time[0]
    n = len(detection)
    N = len(new_time)
    b = np.empty((n, N), dtype=np.bool_)
    for i in nb.prange(n):
        intervals = _giveTimeIntervals(detection[i], time)
        filtered = _filterIntervals(intervals, max_gap, min_width)
        b[i] = _projectIntervals(filtered, N, dt)
    return b


@ReshapeArraysDecorator(dim=2, input_num=1, methodfunc=False, output_num=1, first_shape=True)
def ProjectFilterIntervals(detection, /, time, max_gap, min_width, new_time):
    """
        Filters boolean array `detection` by combining close `True` intervals with gap <= `max_gap`
        and removing short intervals < `min_width`. Then, projects onto `new_time` axis.

        Arguments:
            detection : boolean np.ndarray (..., Nt) : boolean arrays, `True` is earthquake, `False` is noise
            time : float np.ndarray (Nt,) : time axis for extraction of intervals
            max_gap : float : maximum gap between two close intervals to be considered as one big interval
            min_width : float : minimum width of intervals to be kept
            time : float np.ndarray (N,) : new time axis for projection of filtered intervals

        Returns:
            filtered_detection : boolean np.ndarray (..., N) : filtered and re-projected boolean array
    """
    return _projectFilterIntervalsN(detection, time, max_gap, min_width, new_time)
