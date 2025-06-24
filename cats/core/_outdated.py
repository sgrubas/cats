# import numpy as np
# from scipy import optimize
# from scipy.spatial.distance import cdist
#
#
# _metrics = {'cityblock': 1, 'euclidean': 2,
#             1: 'cityblock', 2: 'euclidean'}
#
#
# def CheckDistance(vec1, vec2, max_dist, metric):
#     assert vec1.shape[-1] == vec2.shape[-1]
#     dim = vec1.shape[-1]
#     max_dist = max_dist if (max_dist is not None) else np.inf
#     metric = metric if (metric is not None) else 1
#
#     if isinstance(max_dist, (tuple, list, np.ndarray)):
#         max_dist = max_dist + type(max_dist)([np.inf] * (dim - len(max_dist)))
#
#         R = np.abs(np.expand_dims(vec1, 1) - np.expand_dims(vec2, 0))
#         bool_inds = np.prod(R <= np.expand_dims(max_dist, (0, 1)), axis=-1, dtype=bool)
#         norm = _metrics.get(metric, 1) if isinstance(metric, str) else metric
#         R = np.linalg.norm(R, ord=norm, axis=-1)
#
#     elif isinstance(max_dist, (int, float)):
#         norm = _metrics.get(metric, 'cityblock') if isinstance(metric, int) else metric
#         R = cdist(vec1, vec2, metric=norm)
#         bool_inds = (R <= max_dist)
#     else:
#         raise ValueError(f"`max_dist` must be `int`/`float` or `array[`int`/`float`]`, given {type(max_dist)}")
#     R[~bool_inds] = float('inf')
#     return R, bool_inds
#
#
# def SplitByDistance(vec1, vec2, max_dist, metric):
#     R, bool_inds = CheckDistance(vec1, vec2, max_dist, metric)
#
#     close_sums = tuple(bool_inds.sum(axis=dim) for dim in (1, 0))
#     distant_inds = tuple(cs == 0 for cs in close_sums)
#
#     paired_inds = tuple(cs == 1 for cs in close_sums)
#     pairs = np.argwhere(bool_inds * paired_inds[0][:, None] * paired_inds[1][None, :])
#     pairs = tuple(pairs.T)
#
#     close_sums[0][pairs[0]] = -1
#     close_sums[1][pairs[1]] = -1
#     close_inds = tuple(cs > 0 for cs in close_sums)
#
#     R = R[close_inds[0]][:, close_inds[1]]
#
#     return R, distant_inds, close_inds, pairs
#
#
# def LSAMatchPair(sequence1, sequence2, max_dist=None, metric=1):
#     seq1, seq2 = sequence1, sequence2
#
#     R, *inds = SplitByDistance(seq1, seq2, max_dist, metric)
#     matched_1, matched_2 = optimize.linear_sum_assignment(R)
#     del R  # to clean memory
#
#     distant_inds_1, distant_inds_2 = inds[0]
#     close_inds_1, close_inds_2 = inds[1]
#     pairs_1, pairs_2 = inds[2]
#
#     inds1, inds2 = np.arange(len(seq1)), np.arange(len(seq2))
#     reindexed_1, reindexed_2 = inds1[close_inds_1], inds2[close_inds_2]
#
#     n_distant_1 = np.count_nonzero(distant_inds_1)
#     n_distant_2 = np.count_nonzero(distant_inds_2)
#
#     # Elements of sequences may not always have a pair
#     remained_1 = list(set(range(len(reindexed_1))) - set(matched_1))
#     remained_2 = list(set(range(len(reindexed_2))) - set(matched_2))
#
#     matched_seq1_inds = (pairs_1,
#                          reindexed_1[matched_1],
#                          reindexed_1[remained_1],
#                          np.full(len(remained_2), -1),
#                          inds1[distant_inds_1],
#                          np.full(n_distant_2, -1))
#     matched_seq2_inds = (pairs_2,
#                          reindexed_2[matched_2],
#                          np.full(len(remained_1), -1),
#                          reindexed_2[remained_2],
#                          np.full(n_distant_1, -1),
#                          inds2[distant_inds_2])
#     matched = (np.concatenate(matched_seq1_inds),
#                np.concatenate(matched_seq2_inds))
#     return matched
#
#
# def give_unpaired(n1, n2, pairs):
#     i1, i2 = tuple(set(pi) for pi in np.array(pairs).reshape(-1, 2).T)
#     ref1, ref2 = set(range(n1)), set(range(n2))
#     ind1, ind2 = list(ref1 - i1), list(ref2 - i2)
#     return ind1, ind2
#
#
# def sparsify_path(path, vec1, vec2, dist):
#     N = len(path)
#     sparse, buffer = [], []
#     i0, j0 = -1, -1
#     for cnt, (i, j) in enumerate(path):
#         dist_ij = dist(vec1[i], vec2[j])
#         last = (cnt == N - 1)
#         if ((i != i0) and (j != j0) and len(buffer) > 0) or last:
#             if last:
#                 buffer.append(((i, j), dist_ij))
#             min_ind, min_dist = min(buffer, key=lambda x: x[-1])
#             if np.isfinite(min_dist):
#                 sparse.append(min_ind)
#             buffer.clear()
#         buffer.append(((i, j), dist_ij))
#         i0, j0 = i, j
#
#     ind1, ind2 = give_unpaired(len(vec1), len(vec2), sparse)
#     return sparse, (ind1, ind2)
#
#
# @nb.njit("b1[:, :](b1[:, :], i8, i8)", cache=True)
# def _removeGaps_OLD(detection, min_separation, min_width):
#     M, Nt = detection.shape
#     filtered = np.full_like(detection, False)
#     for i in nb.prange(M):
#         intervals = _giveIntervals(detection[i])
#         n = len(intervals)
#         if (min_separation > 0) and (n > 0):
#             buffer = [(intervals[0, 0], intervals[0, 1])]
#             for j in range(1, n):
#                 intv_new = intervals[j]
#                 intv_old = buffer[-1]
#                 if ((intv_new[0] - intv_old[1] - 1) <= min_separation) and \
#                    (((intv_new[1] - intv_new[0] + 1) < min_width) or
#                    ((intv_old[1] - intv_old[0] + 1) < min_width)):
#
#                     buffer[-1] = (intv_old[0], intv_new[1])
#                 else:
#                     buffer.append((intv_new[0], intv_new[1]))
#             buffer = np.array(buffer) if len(buffer) > 0 else np.zeros((0, 2), dtype=np.int64)
#         else:
#             buffer = intervals
#         for j1, j2 in buffer:
#             filtered[i, j1: j2 + 1] = True
#     return filtered


# @nb.njit(["UniTuple(f8[:, :], 2)(f8[:, :], f8[:, :], f8)"])
# def FilterIntervalsFeatures(detected_intervals, picked_features, dt_sec):
#     """
#         `detected_intervals` are assumed to be properly projected from cluster catalogs
#     """
#     if len(detected_intervals) > 1:  # no point to merge less than 2 intervals
#         N = round(detected_intervals.max() / dt_sec) + 2  # don't need to know full length
#         bool_detection = np.full(N, False, dtype=np.bool_)
#         for intv in detected_intervals:  # projecting first, easier
#             i1, i2 = intv / dt_sec
#             i1, i2 = round(i1 - dt_sec/2), round(i2 + dt_sec/2)  # extending, to avoid argmax of empty slices below
#             bool_detection[i1: i2 + 1] = True
#
#         merged_intervals = _giveIntervals(bool_detection) * dt_sec  # new, merged intervals from projection
#         onsets = picked_features[:, 0]
#         likelihood = picked_features[:, 1]
#         merged_features = np.full((len(merged_intervals),
#                                    picked_features.shape[1]), -1.0)  # default `-1` will pop up if empty slice below
#         for i, intv in enumerate(merged_intervals):
#             curr_inds = (intv[0] <= onsets) & (onsets <= intv[1])  # which ones are in the new merged interval
#             curr_feats = picked_features[curr_inds]
#             likel_i = likelihood[curr_inds]
#             if len(likel_i) > 0:
#                 ind_max = np.argmax(likel_i)  # choose only ones with max energy
#                 merged_features[i] = curr_feats[ind_max]
#     else:
#         merged_intervals, merged_features = detected_intervals, picked_features
#
#     return merged_intervals, merged_features
#
#
# def get_associated_events_from_mCATS_catalog(full_shape, cluster_catalogs, dt_sec):
#
#     shape = full_shape[:-2]  # reduce dim in multitrace
#
#     detected_intervals = np.empty(shape, dtype=object)
#     picked_features = np.empty(shape, dtype=object)
#
#     interval_cols = ["Time_start_sec", "Time_end_sec"]
#     feature_cols = ["Time_peak_sec", "Energy_peak"]  # STRICTLY THESE TWO FIRST, important for filtering
#     feature_cols = [col for col in cluster_catalogs.columns
#                     if col not in (interval_cols + feature_cols)]
#
#     for ind in np.ndindex(shape):
#         cat = index_cluster_catalog(cluster_catalogs, ind)
#
#         intervals, features = [], []
#         for i, df_grp in cat.groupby('Cluster_ID'):
#             intervals.append([df_grp.Time_start_sec.min(), df_grp.Time_end_sec.max()])
#             peak_id = df_grp.Energy_peak.values.argmax()
#             features.append(df_grp[feature_cols].values[peak_id])
#
#         intervals_i = np.array(intervals)
#         features_i = np.array(features)
#
#         detected_intervals[ind], picked_features[ind] = FilterIntervalsFeatures(detected_intervals=intervals_i,
#                                                                                 picked_features=features_i,
#                                                                                 dt_sec=dt_sec)
#
#     return detected_intervals, picked_features
#
#
# def IntervalsFeaturesFromCatalogs(full_shape, cluster_catalogs, dt_sec):
#
#     shape = full_shape[:-1]
#
#     detected_intervals = np.empty(shape, dtype=object)
#     picked_features = np.empty(shape, dtype=object)
#
#     interval_cols = ["Time_start_sec", "Time_end_sec"]
#     features_cols = ["Time_peak_sec", "Energy_peak"]  # STRICTLY THESE TWO FIRST, important for filtering
#     features_cols += [col for col in cluster_catalogs.columns
#                       if col not in (interval_cols + features_cols)]
#
#     for ind in np.ndindex(shape):
#         if len(cluster_catalogs) > 0:  # empty if empty
#             cat = index_cluster_catalog(cluster_catalogs, ind)
#             intervals_i = cat[interval_cols].values
#             features_i = cat[features_cols].values
#
#             if len(intervals_i) == 0:  # if shape = (0, ...) then it is `pyobject` array, and fails numba JIT below
#                 intervals_i = intervals_i.astype(np.float64)
#                 features_i = features_i.astype(np.float64)
#         else:
#             intervals_i = np.zeros((0, 2), dtype=float)
#             features_i = np.zeros((0, len(features_cols)), dtype=float)
#
#         detected_intervals[ind], picked_features[ind] = FilterIntervalsFeatures(detected_intervals=intervals_i,
#                                                                                 picked_features=features_i,
#                                                                                 dt_sec=dt_sec)
#
#     return detected_intervals, picked_features


# ------------------ Experimental ------------------ #


# def _optimalNeighborhoodDistance(p, pmin, qmax, maxN):
#     qi = qmax  # in case the cycle is empty
#     for qi in range(1, qmax + 1):
#         Q = (qi * 2 + 1)**2  # clustering kernel is square
#         cdf = stats.binom.cdf(maxN, Q, p)  # probability that maximum `maxN` elements in kernel `Q` are nonzero
#         if cdf < pmin:  # choose `qi` which provides probability of noisy nonzero elements not to be present in `Q`
#             break
#     return max(1, qi - 1)
#
#
# def OptimalNeighborhoodDistance(minSNR, d=2, pmin=0.95, qmax=10, maxN=1):
#
#     # thresholding function value from DATE, defined for `noise variance = 1`
#     xi = _xi(d=d, rho=minSNR)
#     # percentage of chi-distributed noise elements to be > `xi` (`d` degrees of freedom)
#     p = special.gammaincc(d / 2, xi**2 / 2)
#     q_opt = _optimalNeighborhoodDistance(p, pmin=pmin, qmax=qmax, maxN=maxN)
#     return q_opt


# @nb.njit(["f8[:, :](f8[:, :], UniTuple(i8, 2), b1)",
#           "f4[:, :](f4[:, :], UniTuple(i8, 2), b1)",
#           "b1[:, :](b1[:, :], UniTuple(i8, 2), b1)"],
#          cache=True, parallel=True)
# def _topo_density_estimation(Z, q, smoothing):
#     """ Basically, it is equivalent to 2D convolution. But since I can ignore 'zero' pixels,
#         it can speed up it a little bit, as well as I use 'parallel' loop.
#     """
#     Nf, Nt = Z.shape
#     q_f, q_t = q
#     k_weight = 1.0 / ((2 * q_f + 1) * (2 * q_t + 1))
#     D = np.zeros_like(Z)
#
#     for flat_index in nb.prange(0, Z.size):  # parallel
#         if not Z.flat[flat_index] > 0.0:
#             continue
#
#         i, j = ind = divmod(flat_index, Nt)
#
#         i1, i2 = max(i - q_f, 0), min(i + q_f + 1, Nf)
#         j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)
#
#         for ii in range(i1, i2):
#             for jj in range(j1, j2):
#                 if Z[ii, jj] > 0.0:
#                     if smoothing:
#                         D[ind] += k_weight * Z[ii, jj]  # smoother
#                     else:
#                         D[ind] += k_weight  # w/o smoothing
#         if not smoothing:
#             D[ind] *= Z[ind]  # w/o smoothing
#
#     return D
#
#
# @nb.njit([f"i4[:, :](f4[:, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2), f8, i8, b1)",
#           f"i4[:, :](f8[:, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2), f8, i8, b1)",
#           f"i4[:, :](b1[:, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2), f8, i8, b1)"],
#          cache=True)
# def _TopoClustering2D(Z, q, s, freq_octaves, prominence_thr, merge_rule, smoothing):
#     """
#     This is an adaptation of ToMATo algorithm [1].
#
#     Notes:
#         1. This implementation adds additional criterion on size of clusters.
#         2. Small clusters get merged into neighbouring clusters, otherwise removed, see 'merge_rule' below.
#
#     Arguments:
#         Z : np.ndarray (Nf, Nt) : 2D array of density values.
#         q : tuple[int, int] : neighbourhood distance for clustering `(q_f, q_t)`.
#         s : tuple[int, int] : minimum cluster sizes `(s_f, s_t)`.
#         freq_octaves : tuple[float, float] : equivalent to `s_f` and `q_f` (see above), but in log2 scale (octaves),
#                        for frequency only, in the order: (cluster_size_f_octaves, cluster_distance_f_octaves).
#                        By default, zeros and not used, but if non-zero, they replace `s_f` and `q_f` respectively.
#         prominence_thr : float : threshold for cluster density prominence.
#         merge_rule : int : Merging rule for small clusters:
#                            0 - Kept as is, no rule applied, as long as they are prominent enough
#                            1 - Delete small clusters,
#                            2 - Merge to the closest prominent neighbouring cluster.
#
#     Reference: [1] Chazal, F., Guibas, L. J., Oudot, S. Y., & Skraba, P. (2013).
#                Persistence-based clustering in Riemannian manifolds. Journal of the ACM (JACM), 60(6), 1-38.
#     """
#     Nf, Nt = shape = Z.shape
#     freq_width_octaves, freq_distance_octaves = freq_octaves
#
#     C = np.zeros(shape, dtype=np.int64)  # cluster IDs
#
#     D = _topo_density_estimation(Z, q, smoothing)  # smoothed density map, takes into account connectivity
#     q_f, q_t = q
#     sort_inds = np.argsort(-D.ravel())  # decreasing order sorting, 1D flat index
#
#     clusters = {}  # to store cluster members
#     prominences = {}  # to store cluster prominences
#     neighbour_clusters = {-1: -1}  # to keep the largest neighbouring cluster to a cluster
#     cluster_id = 1
#     for flat_index in sort_inds:
#         D_ij = D.flat[flat_index]
#         if not D_ij > 0.0:
#             continue  # skip null pixels
#
#         # 1. ---------- Find neighbours ----------
#         ind = divmod(flat_index, Nt)
#         i, j = ind
#
#         # define lookout window to search for neighbors
#         if freq_distance_octaves > 0.0:  # log2 step for frequency
#             i1 = min(i - 1, round(i * 2 ** (-freq_distance_octaves)))  # octave is power of 2
#             i2 = max(i + 1, round(i * 2 ** freq_distance_octaves)) + 1  # octave is power of 2
#         else:  # normal linear step
#             i1, i2 = i - q_f, i + q_f + 1
#
#         i1, i2 = max(i1, 0), min(i2, Nf)
#         j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)
#
#         neighbours = []
#         for ii in range(i1, i2):
#             for jj in range(j1, j2):
#                 ind_kk = (ii, jj)
#                 D_kk = D[ind_kk]
#                 if (ind_kk != ind) and (D_kk > D_ij):
#                     neighbours.append((D_kk, ind_kk))
#
#         # 2. ---------- Assign a cluster ----------
#         # 2.1. ---------- Declare a new cluster (peak) ----------
#         if len(neighbours) == 0:
#             C[ind] = cluster_id  # assign cluster id, `+1` to avoid `0` id
#             clusters[cluster_id] = {
#                 ind: D_ij}  # add to the cluster list, NUMBA does not support List and Set as values (0.60.0)
#             prominences[cluster_id] = (ind, D_ij, -np.inf)  # ID, birth, death
#             cluster_id += 1
#
#         # 2.2. ---------- Assign to existing cluster (peak) ----------
#         else:
#             max_i = np.argmax(np.array([xi[0] for xi in neighbours]))
#             D_j, ind_j = neighbours[max_i]  # max among the neighbours
#
#             C[ind] = cid_j = C[ind_j]  # assign ID of the highest cluster in the neighbourhood
#             clusters[cid_j][ind] = D_ij  # add current point to the cluster 'j'
#             D_rj = prominences[cid_j][1]  # root vertex height of local max gradient
#
#             # 2.3 ---------- Merge neighbouring clusters ----------
#             for Z_k, ind_k in neighbours:
#                 cid_k = C[ind_k]
#                 D_rk = prominences[cid_k][1]  # current root vertex height
#                 low_prominence = (min(D_rk, D_rj) - D_ij < prominence_thr)
#
#                 # This 'if' branch is for merging not prominent clusters
#                 if ((cid_k != cid_j) and  # to avoid repeated merging 'k -> j'
#                    low_prominence and  # prominence threshold
#                    (clusters.get(cid_j, None) is not None)):  # to avoid repeated merging 'j -> k'
#
#                     cids = np.array([cid_j, cid_k])  # cluster IDs
#                     roots = np.array([D_rj, D_rk])  # their root vertices, to compare
#                     cids_sort = np.argsort(roots)  # sort, to find the most prominent cluster
#
#                     cid_d, cid_l = cids[cids_sort]  # to die, to live
#
#                     prominences[cid_d] = (prominences[cid_d][0],  # index
#                                           prominences[cid_d][1],  # birth level
#                                           D_ij)  # kill and update death
#
#                     clusters_d = clusters.pop(cid_d)  # clear dead cluster
#                     clusters[cid_l].update(clusters_d)  # merge dead cluster with live cluster
#
#                     # update cluster mask after merging
#                     for ind_d in clusters_d.keys():
#                         C[ind_d] = cid_l  # update cluster IDs
#
#                 # This 'elif' branch is for remembering bigger neighbours of clusters
#                 elif ((cid_k != cid_j) and  # if different cluster than 'j'
#                       (not low_prominence) and  # if prominent
#                       (merge_rule == 2)):  # if merging is needed
#                     cids = np.array([cid_j, cid_k])  # cluster IDs
#                     roots = np.array([D_rj, D_rk])  # their root vertices, to compare
#                     cids_sort = np.argsort(roots)  # sort, to find the most prominent cluster
#
#                     cid_d, cid_l = cids[cids_sort]  # to die, to live (probably)
#                     D_rd, D_rl = roots[cids_sort]  # root vertex of the dead cluster
#
#                     ind_d = prominences[cid_d][0]
#                     ind_l = prominences[cid_l][0]
#                     dist_new = np.sqrt((D_rd - D_rl) ** 2 +
#                                        (ind_d[0] - ind_l[0]) ** 2 +
#                                        (ind_d[1] - ind_l[1]) ** 2)
#
#                     neighb_cid = neighbour_clusters.get(cid_d, cid_d)
#                     if cid_d == neighb_cid:
#                         dist_old = np.inf
#                     else:
#                         dist_old = np.sqrt((D_rd - prominences[neighb_cid][1]) ** 2 +
#                                            (ind_d[0] - prominences[neighb_cid][0][0]) ** 2 +
#                                            (ind_d[1] - prominences[neighb_cid][0][1]) ** 2)
#
#                     if dist_new < dist_old:
#                         neighbour_clusters[cid_d] = cid_l
#
#                     # there are two ways to merge
#                     # 1. To a bigger neighbour
#                     # if D_rj > neighbour_stats[1]:  # if neighbour is more prominent
#                     #     neighbour_clusters[cid_k] = cid_j
#
#                     # 2. To a closest neighbour (by peak-to-peak distance)
#
#     # 3. ---------- Filter clusters by size and merge if needed ----------
#     if merge_rule > 0:
#         s_f, s_t = s
#         proms, cluster_ids = [], []
#         for cid in clusters.keys():
#             birth, death = prominences[cid][1:]
#             if death < 0:  # i.e. -np.inf, means that this cluster is alive by prominence
#                 proms.append(birth)  # its root height
#                 cluster_ids.append(cid)
#         sort_ind = np.argsort(np.array(proms))
#
#         for ind in sort_ind:
#             cid = cluster_ids[ind]
#             cluster = clusters.get(cid, None)
#             if cluster is None:
#                 continue  # skip clusters already removed (should not happen)
#
#             # check size
#             cluster_inds = list(cluster.keys())
#             cl_ids = np.array(cluster_inds)
#             f_max, f_min = cl_ids[:, 0].max(), cl_ids[:, 0].min()
#             t_max, t_min = cl_ids[:, 1].max(), cl_ids[:, 1].min()
#
#             if freq_width_octaves > 0.0:
#                 min_frequency_width = np.log2((f_max + 1) / (f_min + 1e-8)) >= freq_width_octaves  # log2 for octaves
#             else:  # linear width
#                 min_frequency_width = (f_max - f_min + 1 >= s_f)
#
#             # remove, merge, or keep
#             if (not min_frequency_width) or (t_max - t_min + 1 < s_t):
#                 cluster_cid = clusters.pop(cid)  # remove
#
#                 # Merging step (iteratively searches for the first alive closest neighbour via their links)
#                 if merge_rule == 2:  # if merging is allowed
#                     neighbour = neighbour_clusters.get(cid, 0)
#                     neighbour_dead = clusters.get(neighbour, None) is None  # if neighbour still exists
#                     neigh_cntr = 1  # counter to avoid infinite loops
#                     while neighbour_dead and (neighbour > 0) and neigh_cntr < 50:
#                         neighbour = neighbour_clusters.get(neighbour, 0)
#                         neighbour_dead = clusters.get(neighbour, None) is None
#                         neigh_cntr += 1
#
#                     if (not neighbour_dead) and (neighbour > 0):
#                         clusters[neighbour].update(cluster_cid)  # merge
#
#     # 4. ---------- Assign new labels 1...K ----------
#     C = np.zeros(shape, dtype=np.int32)
#     for k, cluster in enumerate(clusters.values()):
#         for ind in cluster.keys():
#             C[ind] = k + 1
#
#     return C
