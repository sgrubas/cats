import numpy as np
import numba as nb
from numba.types import b1, f8, i8, string, List, UniTuple
from .utils import ReshapeInputArray


############################################################################
#####################  TIME PROJECTION AND FIILTERING  #####################
############################################################################

@nb.njit("f8[:, :](b1[:], f8[:])")
def _giveIntervals(detection, time):
    N = len(detection)
    intervals = np.empty((N // 2 + 1, 2))
    t1 = t2 = -1.0; j = 0
    for i in range(N):
        di = detection[i]
        ti = time[i]
        if di and (i < N - 1):
            if t1 < 0: t1 = ti
            else: t2 = ti
        elif di and (i == N - 1) and (t1 >= 0):
            intervals[j] = np.array([t1, ti])
            t1 = t2 = -1.0; j += 1
        elif (not di) and (t1 >= 0):
            if (t2 < 0): t2 = t1
            intervals[j] = np.array([t1, t2])
            t1 = t2 = -1.0; j += 1
    return intervals[:j]

@nb.njit("f8[:, :](f8[:, :], f8, f8)")
def _filterIntervals(intervals, max_gap, min_width):
    n = len(intervals)
    if (max_gap > 0) and (len(intervals) > 0):
        filtered = [(intervals[0, 0], intervals[0, 1])]
        for i in range(1, len(intervals)):
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
    intervals = []
    for di in detection:
        intervals.append(_filterIntervals(_giveIntervals(di, time), max_gap, min_width))
    return intervals

def FilterIntervals(detection, time, max_gap, min_width):
    return _filterIntervalsN(detection, time, max_gap, min_width)

@nb.njit("b1[:](f8[:, :], f8[:], f8)")
def _projectIntervals(intervals, time, dt):
    b = np.full(len(time), False)
    for (i1, i2) in intervals:
        b[int(i1 / dt) : int(np.ceil(i2 / dt)) + 1] = True
    return b

@nb.njit("b1[:, :](b1[:, :], f8[:], f8, f8, f8[:])", parallel=True)
def _projectFilterIntervalsN(detection, time, max_gap, min_width, new_time):
    dt = new_time[1] - new_time[0]
    n = len(detection)
    b = np.full((n, len(new_time)), False)
    for i in nb.prange(n):
        filtered = _filterIntervals(_giveIntervals(detection[i], time), max_gap, min_width)
        b[i] = _projectIntervals(filtered, new_time, dt)
    return b

@ReshapeInputArray(dim=2)
def ProjectFilterIntervals(detection, time, max_gap, min_width, new_time):
    return _projectFilterIntervalsN(detection, time, max_gap, min_width, new_time)


############################### SAME FUNCTION BUT RIGHT AWAY WITH CLUSTERING RESULT

# @nb.njit("b1[:, :](b1[:, :], f8[:], f8, f8, f8[:])", parallel=True)
# def _projectFilterIntervalsN(detection, time, max_gap, min_width, new_time):
#     dt = new_time[1] - new_time[0]
#     n = len(detection)
#     b = np.full((n, len(new_time)), False)
#     for i in nb.prange(n):
#         filtered = _filterIntervals(_giveIntervals(detection[i], time), max_gap, min_width)
#         b[i] = _projectIntervals(filtered, new_time, dt)
#     return b