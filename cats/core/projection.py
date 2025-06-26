"""
    Functions for projecting detection from Time-Frequency onto Time.
"""

import numpy as np
import numba as nb


# ------------------  TIME PROJECTION AND FILTERING  ------------------ #


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
        if d_i and (i < N - 1):  # True and not last
            if i1 < 0:
                i1 = i
            else:
                i2 = i
        elif d_i and (i == N - 1) and (i1 >= 0):  # True and last and ...
            intervals[j, 0] = i1
            intervals[j, 1] = i
            i1 = i2 = -1.0
            j += 1
        elif (not d_i) and (i1 >= 0):  # False and ...
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


@nb.njit(["f8[:, :](f8[:, :], f8, f8)",
          "f4[:, :](f4[:, :], f8, f8)",
          "i8[:, :](i8[:, :], i8, i8)",
          "i4[:, :](i4[:, :], i8, i8)"],
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

    if (min_duration > 0) and (len(combined_intervals) > 0):
        # Below, `-1` is to include a single point
        duration_inds = (combined_intervals[:, 1] - combined_intervals[:, 0]) >= min_duration
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
    for ind in np.ndindex(filtered_intervals.shape):
        interval_ind = _giveIntervals(detection[ind])  # get intervals first
        filtered_intervals[ind] = filter_intervals(interval_ind, min_separation, min_duration)  # filter intervals
        project_intervals(filtered_intervals[ind], filtered_detection[ind])  # project filtered intervals

    return filtered_detection, filtered_intervals


# def ProjectCatalogs(trace_shape, cluster_catalogs, dt_sec, min_separation_sec, min_duration_sec):
#     shape = trace_shape
#
#     detected_intervals = np.empty(shape, dtype=object)
#     picked_features = np.empty(shape, dtype=object)
#
#     interval_cols = ["Time_start_sec", "Time_end_sec"]
#     detection_cols = ["Interval_ID", "Interval_start_sec", "Interval_end_sec"]
#
#     # cols for 'picked_features'
#     features_cols = ["Time_peak_sec", "Energy_peak", "Frequency_peak_Hz"]  # STRICTLY THESE TWO FIRST
#
#     for ind in np.ndindex(shape):
#         cat = index_cluster_catalog(cluster_catalogs, ind)
#
#         if len(cat) > 0:  # skip if empty
#             intervals_i = cat[interval_cols].values  # these will be merged
#
#             projected_intervals, merge_inds = MergeIntervals(intervals_i, dt_sec, min_separation_sec,
#                                                              min_duration_sec)
#             values = np.concatenate((merge_inds[:, None], projected_intervals[merge_inds]), axis=1)
#             assign_by_index_cluster_catalog(cluster_catalogs, ind, detection_cols, values)
#
#             detected_intervals[ind] = projected_intervals
#
#             # ------- going to be deprecated --------
#             features = cat[features_cols].values  # 2D array
#             max_inds = maximum_position(features[:, 1], merge_inds + 1,
#                                         np.arange(merge_inds.max() + 1) + 1)  # peak energy criterion
#             features = features[np.array(max_inds).squeeze()]
#             picked_features[ind] = features if features.ndim > 1 else features.reshape(-1, len(features_cols))
#
#         else:  # if empty, then it is `pyobject` array, and fails numba JIT below
#             detected_intervals[ind] = np.zeros((0, 2), dtype=float)
#             picked_features[ind] = np.zeros((0, len(features_cols)), dtype=float)
#
#     # cluster_catalogs = cluster_catalogs.astype({detection_cols[0]: int})
#     return detected_intervals, picked_features


@nb.njit(["Tuple((f8[:, :], i8[:]))(f8[:, :], f8, f8, f8)",
          "Tuple((f8[:, :], i8[:]))(f4[:, :], f8, f8, f8)"], cache=True)
def MergeIntervals(intervals, dt_sec, min_separation_sec, min_duration_sec):
    """
        Merges intervals based on the given minimum separation and duration.
    """
    if len(intervals) > 1:  # no point to merge less than 2 intervals
        # centers = np.mean(intervals, axis=1)
        centers = (intervals[:, 0] + intervals[:, 1]) * 0.5  # mean of start and end

        # 1. Project intervals onto a discrete sequence
        # (easier to implement, because it does not depend on sorting)
        N = round(intervals.max() / dt_sec) + 2  # don't need to know full length, max is enough
        intervals_inds = (intervals / dt_sec).astype(np.int64)

        binary_projection = np.full(N, False, dtype=np.bool_)
        project_intervals(intervals_inds, binary_projection)

        # 2. Extract new projected intervals, and merge those within min_separation_sec
        merged_intervals = _giveIntervals(binary_projection) * dt_sec
        merged_intervals[:, 1] += dt_sec  # append to allow single pixel intervals
        merged_intervals = filter_intervals(merged_intervals, min_separation_sec, min_duration_sec)

        # 3. Aggregate indexes falling in one interval, by feature_max
        merge_inds = np.arange(len(intervals))
        for i in range(len(merged_intervals)):
            d1, d2 = merged_intervals[i]
            curr_inds = (d1 <= centers) & (centers <= d2)  # which ones are in the new merged interval?
            merge_inds[curr_inds] = i  # index allows assigning interval ID to each feature row
    else:
        merged_intervals = intervals.astype(np.float64)
        merge_inds = np.arange(len(intervals))

    return merged_intervals, merge_inds


@nb.njit(["f8[:, :](f8[:, :], i8[:], i8, i8)"], cache=True)
def GetMaxRowsByLabels(array, labels, col_ind, col_num):
    """
        Get rows from `array` with maximum values in `col_ind` for each label in `labels`.

        Arguments:
            array : np.ndarray (N, M) : 2D array, where N is number of rows, M is number of columns
            labels : np.ndarray (N,) : 1D array with labels for each row
            col_ind : int : index of column to find maximum in
            col_num : int : number of columns in output array

        Returns:
            new_array : np.ndarray (L, M) : 2D array with L rows, where L is number of unique labels,
                                            and M is number of columns
    """
    # index = np.arange(labels.max()) + 1  # labels start from 1, not 0
    index = np.unique(labels)

    new_array = np.zeros((len(index), col_num))

    array_col = array[:, col_ind]
    original_inds = np.arange(len(array_col))
    for i in range(len(index)):
        j = index[i]
        cond_j = labels == j
        max_j = np.argmax(array_col[cond_j])  # if empty then Error
        orig_max_j = original_inds[cond_j][max_j]
        new_array[i] = array[orig_max_j]

    return new_array


def ProjectCatalogs(trace_shape, cluster_catalogs, dt_sec, min_separation_sec, min_duration_sec):
    """
        Projects intervals in cluster catalogs onto time domain, merging the overlapping intervals.

        Arguments:
            trace_shape : tuple : shape of the traces, e.g. (N_channels, N_traces)
            cluster_catalogs : pd.DataFrame : DataFrame with cluster catalog data
            dt_sec : float : time step in seconds
            min_separation_sec : float : minimum separation between intervals in seconds
            min_duration_sec : float : minimum duration of intervals in seconds

        Returns:
            detected_intervals : np.ndarray : array of detected intervals for each trace
            picked_features : np.ndarray : array of picked features for each trace
    """
    cluster_catalogs.sort_index(inplace=True)  # for faster and easier indexing over trace indexes
    unique_inds, iter_inds = np.unique(cluster_catalogs.index, return_index=True)
    N_all = cluster_catalogs.shape[0]
    N_inds = len(iter_inds)

    interval_cols = ["Time_start_sec", "Time_end_sec"]
    detection_cols = ["Interval_ID", "Interval_start_sec", "Interval_end_sec"]
    features_cols = ["Time_peak_sec", "Energy_peak", "Frequency_peak_Hz"]  # STRICTLY THESE TWO FIRST

    shape = trace_shape
    detections = np.zeros((N_all, len(detection_cols)), dtype=float)
    detected_intervals = np.empty(shape, dtype=object)
    picked_features = np.empty(shape, dtype=object)

    if N_all > 0:
        intervals = cluster_catalogs[interval_cols].values  # numpy arrays
        features = cluster_catalogs[features_cols].values  # numpy arrays

        for j, i1 in enumerate(iter_inds):
            ind = unique_inds[j]

            i2 = iter_inds[j + 1] if (j + 1 < N_inds) else N_all
            arr_slice = slice(i1, i2)

            intervals_j = intervals[arr_slice]
            projected_intervals, merge_inds = MergeIntervals(intervals_j, dt_sec, min_separation_sec, min_duration_sec)
            values = np.concatenate((merge_inds[:, None], projected_intervals[merge_inds]), axis=1)
            detections[arr_slice] = values

            detected_intervals[ind] = projected_intervals

            # ------- going to be deprecated --------
            picked_features[ind] = GetMaxRowsByLabels(features[arr_slice], merge_inds + 1, 1, len(features_cols))

        cluster_catalogs[detection_cols] = detections

    # populate traces with no detected intervals with zero-length arrays
    for ind, arr in np.ndenumerate(detected_intervals):
        if arr is None:
            detected_intervals[ind] = np.zeros((0, 2), dtype=float)
            picked_features[ind] = np.zeros((0, len(features_cols)), dtype=float)

    return detected_intervals, picked_features


def FilterIntervalsFromClusterLabels(detected_cluster_labels):
    filtered_intervals = np.empty(detected_cluster_labels.shape[:-1], dtype=object)
    filtered_detection = np.full_like(detected_cluster_labels, False, dtype=bool)
    for ind, _ in np.ndenumerate(filtered_intervals):
        filtered_detection[ind] = _projectLabeledSequence(detected_cluster_labels[ind])
        filtered_intervals[ind] = _giveIntervals(filtered_detection[ind])

    return filtered_detection, filtered_intervals


@nb.njit(["b1[:](i8[:])", "b1[:](i4[:])",
          "b1[:](u2[:])", "b1[:](u4[:])"])
def _projectLabeledSequence(labeled_sequence):
    binary_sequence = np.full_like(labeled_sequence, False, dtype=np.bool_)
    cluster_IDs = np.unique(labeled_sequence)
    cluster_IDs = cluster_IDs[cluster_IDs != 0]

    for j, k in enumerate(cluster_IDs):
        ids = np.argwhere(labeled_sequence == k)
        i1, i2 = np.min(ids), np.max(ids)
        binary_sequence[i1: i2 + 1] = True
    return binary_sequence


def generate_time_segments(intervals, time_len, dt_sec, segment_separation_sec, segment_extend_time_sec=None):
    # 1. Projecting onto time (binary sequence)
    binary_projection = np.full(time_len + 2, False, dtype=np.bool_)
    for intv in intervals:
        if intv.dtype.name == 'object':
            for ind in np.ndindex(intv.shape):
                intv_i = np.rint(intv[ind] / dt_sec).astype(int)
                project_intervals(intv_i, binary_projection)
        else:
            intv_i = np.rint(intv / dt_sec).astype(int)
            project_intervals(intv_i, binary_projection)

    # 2. Extracting and merging intervals
    if (segment_extend_time_sec is not None) and \
            (segment_separation_sec < segment_extend_time_sec):
        segment_separation_sec = segment_extend_time_sec  # to avoid plotting same events many times
    time_segments = _giveIntervals(binary_projection)
    if len(time_segments) > 0:
        time_segments = time_segments * dt_sec
        time_segments[:, 1] += dt_sec  # append to allow single pixel intervals
        time_segments = filter_intervals(time_segments, segment_separation_sec, 0.0)
        if segment_extend_time_sec is None:
            segment_extend_time_sec = np.diff(time_segments, axis=-1).squeeze() * 0.5
        # print(segment_extend_time_sec.shape)

        time_segments[:, 0] -= segment_extend_time_sec
        time_segments[:, 1] += segment_extend_time_sec

    return time_segments
