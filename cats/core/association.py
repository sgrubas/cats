"""
    Functions for association detected events from different stations / channels.
"""

from typing import Union
from pydantic import BaseModel
from functools import partial
from tqdm.notebook import tqdm

import numpy as np
import numba as nb
from scipy.signal import find_peaks
from scipy.spatial import KDTree
import networkx as nx
from .utils import ReshapeArraysDecorator


@nb.njit(["f8(f8[:], f8[:], f8[:], i8)",
          "f8(f4[:], f4[:], f8[:], i8)",
          "f8(f8[:], f4[:], f8[:], i8)",
          "f8(f4[:], f8[:], f8[:], i8)"
          ], cache=True)
def thresholded_distance(v1, v2, threshold, order):
    diff = np.abs(v1 - v2)
    if len(threshold) > 1:
        close = np.all(diff <= threshold)
    else:
        close = False

    diff = (diff ** order).sum() ** (1 / order)

    if len(threshold) == 1:
        close = (diff <= threshold[0])

    diff = diff if close else np.inf
    return diff


@nb.njit(["f8[:](f8[:], f8[:, :], f8[:], i8)",
          "f8[:](f4[:], f4[:, :], f8[:], i8)",
          "f8[:](f8[:], f4[:, :], f8[:], i8)",
          "f8[:](f4[:], f8[:, :], f8[:], i8)"], cache=True)
def distance_on_vector(val, vector, threshold, order):
    n = len(vector)
    dists = np.empty(n)
    for i in nb.prange(n):
        dists[i] = thresholded_distance(val, vector[i], threshold, order)
    return dists


@nb.njit(["f8[:, :](f8[:, :], f8[:, :], f8[:], i8)",
          "f8[:, :](f4[:, :], f4[:, :], f8[:], i8)",
          "f8[:, :](f8[:, :], f4[:, :], f8[:], i8)",
          "f8[:, :](f4[:, :], f8[:, :], f8[:], i8)"], parallel=True, cache=True)
def distance_matrix(vec1, vec2, threshold, order):
    n1, n2 = len(vec1), len(vec2)
    D = np.empty((n1, n2))
    for i in nb.prange(n1):
        D[i] = distance_on_vector(vec1[i], vec2, threshold, order)
    return D


@nb.njit("i8[:](f8[:])", cache=True)
def sorted_indices_1D_vector(vector):
    finite_inds = np.isfinite(vector)
    inds = np.ravel(np.argwhere(finite_inds))
    argsort = np.argsort(vector[inds])
    return inds[argsort]


@nb.njit("UniTuple(i8[:], 2)(f8[:, :])", cache=True)
def search_pairs(D):
    n1, n2 = D.shape
    paired_1, paired_2 = [], []
    for i in range(n1):
        di = D[i]
        j_inds = sorted_indices_1D_vector(di)
        for j_by_i in j_inds:
            if j_by_i not in paired_2:
                dj = D[:, j_by_i]
                i_by_j = sorted_indices_1D_vector(dj)[0]
                if i_by_j == i:
                    paired_1.append(i_by_j)
                    paired_2.append(j_by_i)
                    break
    paired_1 = np.array(paired_1, dtype=np.int64)
    paired_2 = np.array(paired_2, dtype=np.int64)
    return paired_1, paired_2


def NonlinearPairAssignment(vec1, vec2, threshold, order):
    D = distance_matrix(vec1, vec2, threshold, order)
    paired_1_by_1, paired_2_by_1 = search_pairs(D)
    paired_2_by_2, paired_1_by_2 = search_pairs(D.T)
    del D
    pairs_1 = set((i, j) for i, j in zip(paired_1_by_1, paired_2_by_1))
    pairs_2 = set((i, j) for i, j in zip(paired_1_by_2, paired_2_by_2))
    pairs = list(pairs_1 | pairs_2)
    return pairs


def MinWeightPairs(sequence1, sequence2, max_dist, p):
    kd_tree1 = KDTree(sequence1)
    kd_tree2 = KDTree(sequence2)
    n1 = len(sequence1)

    D = kd_tree1.sparse_distance_matrix(kd_tree2, max_dist, p=p, output_type='coo_matrix')
    G = nx.bipartite.from_biadjacency_matrix(D)
    min_edges = nx.min_weight_matching(G)
    del D, G
    pairs = []
    for p in min_edges:
        i, j = sorted(p)
        pairs.append((i, j - n1))
    return pairs


def MinWeightBipartiteMatching(sequence1, sequence2, max_dist, order, method='manual'):
    max_dist = max_dist if (max_dist is not None) else np.inf
    if method == 'kdtree':
        pairs = MinWeightPairs(sequence1, sequence2, max_dist=max_dist, p=order)
    else:
        dim = sequence1.shape[-1]
        if isinstance(max_dist, (tuple, list, np.ndarray)):
            max_dist = max_dist + type(max_dist)([np.inf] * (dim - len(max_dist)))
        elif isinstance(max_dist, (int, float)):
            max_dist = [max_dist]
        else:
            ValueError(f"{max_dist}")
        max_dist = np.array(max_dist)
        pairs = NonlinearPairAssignment(sequence1, sequence2, threshold=max_dist, order=order)

    paired_1 = np.empty(len(pairs), dtype=np.int64)
    paired_2 = np.empty(len(pairs), dtype=np.int64)
    for k, (i, j) in enumerate(pairs):
        paired_1[k], paired_2[k] = i, j

    unpaired_1 = list(set(range(len(sequence1))) - set(paired_1))
    unpaired_2 = list(set(range(len(sequence2))) - set(paired_2))

    matched = (np.concatenate((paired_1, unpaired_1, np.full(len(unpaired_2), -1))),
               np.concatenate((paired_2, np.full(len(unpaired_1), -1), unpaired_2)))
    return matched


# TODO:
#  ? aggregation with weighted average closer to last trace


def _last_not_nan(sequences):
    reverted_seqs = sequences[::-1]
    aggregated = reverted_seqs[0]
    for seqi in reverted_seqs[1:]:
        aggregated = np.where(np.isnan(aggregated), seqi, aggregated)
    return aggregated


AGGREGATORS = {"min": partial(np.nanmin, axis=0),
               "mean": partial(np.nanmean, axis=0),
               "max": partial(np.nanmax, axis=0),
               "median": partial(np.nanmedian, axis=0),
               "last": lambda x: x[-1],
               "last_not_nan": _last_not_nan}


class Aggregator(BaseModel):
    method: Union[int, str]

    def __call__(self, sequences):
        stacked = np.stack(sequences, axis=0)
        if isinstance(self.method, str):
            return AGGREGATORS[self.method](stacked)
        else:
            return stacked[self.method]


def index_naned(naned_seq, ref_seq):
    sort_inds = np.argsort(naned_seq).argsort()
    ref_inds = np.concatenate((np.argsort(ref_seq), np.full(len(naned_seq) - len(ref_seq), -1)))
    return ref_inds[sort_inds]


def fill_with_nan(reference, inds):
    ndims = reference.ndim
    dim = reference.shape[-1] if ndims > 1 else 1
    filling = np.array([np.nan] * dim) if ndims > 1 else np.nan
    shape = (-1, dim)[:ndims]

    def func(i):
        return reference[i] if i >= 0 else filling

    return np.array([func(i) for i in inds.astype(int)]).reshape(*shape)


def dim_expand_2d(arr):
    return np.expand_dims(arr, -1) if arr.ndim == 1 else arr


def atleast_2d(*sequences):
    arrays = map(np.array, sequences)
    return tuple(map(dim_expand_2d, arrays))


def sort_sequences(*sequences, aggregator=np.nanmin, sort_key=0):
    stacked = np.stack(sequences, axis=0)
    sort_inds = np.argsort(aggregator(stacked[..., sort_key], axis=0))
    return stacked[:, sort_inds]


def MatchSequences(*sequences, max_dist, aggregate='mean', metric_order=1, method='manual', verbose=True):
    sequences = atleast_2d(*sequences)
    match_func = partial(MinWeightBipartiteMatching, max_dist=max_dist, order=metric_order, method=method)

    with tqdm(desc="Association", total=len(sequences) - 1, disable=not verbose) as pbar:
        seq_12 = sequences[:2]
        inds_12 = match_func(*seq_12)
        matched = list(map(fill_with_nan, seq_12, inds_12))
        pbar.update()
        for seq_i in sequences[2:]:
            seq_prev = Aggregator(method=aggregate)(matched)
            inds_prev, inds_next = match_func(seq_prev, seq_i)
            matched = list(map(partial(fill_with_nan, inds=inds_prev), matched))
            matched.append(fill_with_nan(seq_i, inds_next))
            pbar.update()

    return sort_sequences(*matched).squeeze()


@ReshapeArraysDecorator(dim=2, input_num=-1, output_num=1)
def PickFeatures(likelihood, /, *features, time, min_likelihood, min_width_sec, num_features, **kwargs):
    num_features = num_features or 2 + len(features)
    dt = time[1] - time[0]
    min_width = int(min_width_sec / dt)

    feature_sequences = []
    for i in range(len(likelihood)):
        peak_inds, props = find_peaks(likelihood[i], height=min_likelihood, width=min_width, **kwargs)
        onset, heights = time[peak_inds], props['peak_heights']
        feats = tuple(feat_j[i][peak_inds] for feat_j in features)
        feats = np.stack([onset, heights, *feats], axis=-1)
        feats = feats.reshape(-1, num_features)
        feature_sequences.append(feats)
    feature_sequences = np.array(feature_sequences, dtype=object)
    return feature_sequences


@nb.njit(["f8[:, :](f8[:], i8[:, :], f8, f8)", "f4[:, :](f4[:], i8[:, :], f8, f8)",
          "f8[:, :](f8[:], i4[:, :], f8, f8)", "f4[:, :](f4[:], i4[:, :], f8, f8)"],
         parallel=True, cache=True)
def pick_detected_peaks(likelihood, intervals, dt, t0):
    features = np.zeros(intervals.shape, dtype=likelihood.dtype)
    for i in nb.prange(len(intervals)):
        i1, i2 = intervals[i]
        l12 = likelihood[i1: i2 + 1]
        i_max = np.argmax(l12)
        features[i, 0] = (i_max + i1) * dt + t0
        features[i, 1] = l12[i_max]

    return features


def PickDetectedPeaks(likelihood, intervals, dt, t0):
    features = np.empty(likelihood.shape[:-1], dtype=object)
    for ind, _ in np.ndenumerate(features):
        features[ind] = pick_detected_peaks(likelihood[ind], intervals[ind], dt, t0)
    return features


@ReshapeArraysDecorator(dim=0, input_num=2, output_num=1)
def Associate(sequences, location_order, /, vote_rate,
              max_dist_assignment, assignment_aggregate='mean',
              metric_order=1, method='manual', verbose=True):

    # Ordering traces
    location = location_order.squeeze()
    edge_loc = location[location.argmin()]
    R = abs(location - edge_loc)
    sort_inds = R.argsort()
    unsort_inds = sort_inds.argsort()
    sorted_sequences = tuple(sequences[i] for i in sort_inds)

    # Association
    associated_sequences = MatchSequences(*sorted_sequences, max_dist=max_dist_assignment,
                                          aggregate=assignment_aggregate, metric_order=metric_order,
                                          method=method, verbose=verbose)
    del sorted_sequences

    associated_sequences = associated_sequences[unsort_inds]
    dims = (..., 0) if associated_sequences.ndim == 3 else (...,)

    # deleting events which are detected not enough times
    counts = np.isfinite(associated_sequences[dims]).sum(axis=0)
    voting = counts >= (vote_rate * len(location))
    associated_sequences = associated_sequences[:, voting]

    return associated_sequences


# def get_association(cats_result, location_order, /, vote_rate,
#                     max_dist_assignment, assignment_aggregate='mean',
#                     metric_order=1, method='manual', verbose=True):
#     picks = to2d_features(cats_result.picked_features)
#     associated = Associate(np.array(picks, dtype=object), location_order,
#                            vote_rate=vote_rate, max_dist_assignment=max_dist_assignment,
#                            assignment_aggregate=assignment_aggregate, metric_order=metric_order,
#                            method=method, verbose=verbose)
#     return associated
