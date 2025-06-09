"""
    Function for phase separation based on topological clustering and spline interpolation.
    It works like "feature_func" in `cats.clustering.ClusterCatalogs`,
    but instead provides 'Cluster_ID'-like map, called 'Phase_ID'.

    1. Topological separation based on ToMATo algorithm (see docstring _TopoClustering2D and _TopoClustering1D)
    2. Spline separation based on smooth cubic spline interpolation and peaks detection.

    Note: Splines (2) are tens of times faster than Topological (1)
"""

from typing import List, Dict, Callable, Union, Tuple
import numpy as np
import numba as nb
from scipy import interpolate
from .clustering import aggregate_feature_values
from .utils import replace_int_by_slice


def NewMapFromClusters(values_dict: Dict[str, np.ndarray],
                       CID: np.ndarray,
                       freq: np.ndarray,
                       time: np.ndarray,
                       map_funcs: List[Callable],
                       out_dtypes: List[str] = None,
                       out_is_aggr: List[bool] = None,
                       aggr_clustering_axis: Union[int, Tuple[int], None] = None):
    """
        Acts very much like `cats.clustering.ClusterCatalogs`, but instead 'pd.DataFrame' returns 'np.ndarray'
        which the same shape as 'Cluster_ID' values or 'values_dict' arrays. It is useful if a new distribution
        is needed to be calculated based on the distribution in the clusters.

        Arguments:
            values_dict : Dict[str, np.ndarray] : dictionary of feature distributions.
            CID : np.ndarray : Cluster_ID array.
            freq : np.ndarray : frequency values.
            time : np.ndarray : time values.
            map_funcs : List[Callable] : list of mapping functions to apply on feature distributions.
            out_dtypes : List[str] : list of output dtypes for each mapping function.
            out_is_aggr : List[bool] : list of flags to indicate if the output is aggregated or not.
            aggr_clustering_axis : Union[int, Tuple[int], None] : axis to aggregate the clustering.
    """

    assert isinstance(map_funcs, (list, tuple)), "`map_funcs` must be list or tuple"
    assert all([callable(mf) for mf in map_funcs]), "`map_funcs` must be callable"

    assert all([val.shape[-2:] == CID.shape[-2:] for val in values_dict.values()]), \
        "Key time-freq dimensions (last two) must coincide with CID array for all 'values_dict'"

    shape_aggr = CID.shape
    shape_raw = list(values_dict.values())[0].shape
    output_maps = [np.zeros(shape_aggr if aggr else shape_raw, dtype=dtype)
                   for aggr, dtype in zip(out_is_aggr, out_dtypes)]

    trace_shape = CID.shape[:-2]
    ndim = len(trace_shape)

    # iterate over clusters
    for ind in np.ndindex(*trace_shape):  # iter over traces
        val_ind = replace_int_by_slice(ind, aggr_clustering_axis)  # for slicing values_dict (needed to handle aggr.)

        values_dict_ind = {name: val[val_ind] for name, val in values_dict.items()}
        labels = CID[ind]
        index = range(1, labels.max() + 1)

        # reset mapping funcs
        for mf in map_funcs:
            if hasattr(mf, 'reset'):
                mf.reset()

        for cid in index:
            # Find elements for cluster index == 'cid'
            # Get time and freq indexes
            freq_inds, time_inds = sep_inds = np.nonzero(labels == cid)  # always 2D !
            nonzero_num = len(freq_inds)

            if nonzero_num > 0:  # skip non-existing cluster IDs
                f = freq[freq_inds]  # exact time values
                t = time[time_inds]  # exact freq values

                # Slice feature distributions
                vals_slices_dict = {}
                for name, val in values_dict_ind.items():
                    val_shape = val.shape[:-2]  # the last 2 dims are (freq, time)

                    if len(val_shape) > 0:  # handles arrays in case of aggregated clustering
                        val_new = np.zeros(val_shape + (len(freq_inds),), dtype=val.dtype)
                        for val_ind_i in np.ndindex(*val_shape):
                            val_new[val_ind_i] = val[val_ind_i][
                                sep_inds]  # first indexer - traces, second - time x freq
                    else:
                        val_new = val[sep_inds]

                    vals_slices_dict[name] = val_new

                # Iterate over mapping funcs on feature distributions
                results = [func(f, t, vals_slices_dict, sep_inds) for func in map_funcs]
                for out, res in zip(output_maps, results):
                    if res.ndim > 1:  # not aggregated output array
                        for ind_i in np.ndindex(*res.shape[:-1]):
                            out[val_ind][ind_i][sep_inds] = res[ind_i]
                    else:  # aggregated
                        out[ind][sep_inds] = res

    return output_maps


class PhaseSeparator:
    dt_sec: float
    on_distribution_name: str
    aggr_func: Callable
    counter: int

    def reset(self):
        self.counter = 1

    def set_dt_sec(self, dt_sec):
        self.dt_sec = dt_sec


@nb.njit([f"Tuple((DictType(i8, Array(i8, 1, 'C')), DictType(i8, f8)))(f4[:], i8, f8)",
          f"Tuple((DictType(i8, Array(i8, 1, 'C')), DictType(i8, f8)))(f8[:], i8, f8)"],
         cache=True)
def _TopoClustering1D(Z, t_dist, prominence_thr):
    shape = Z.shape
    Nt = shape[0]
    C = np.zeros(shape, dtype=np.int64)  # cluster IDs, for easy access

    sort_inds = np.argsort(-Z)  # decreasing order sorting, 1D flat index

    clusters = {}  # to store cluster members
    prominences = {}  # to store cluster prominences

    cluster_id = 1
    for j in sort_inds:
        Z_j = Z[j]

        if not Z_j > 0.0:  # strictly non-negative as prominences base level is '0'
            continue  # skip null pixels

        # 1. ---------- Find neighbours ----------
        j1, j2 = max(j - t_dist, 0), min(j + t_dist + 1, Nt)

        neighbours = []
        for jj in range(j1, j2):
            Z_jj = Z[jj]
            if (jj != j) and (Z_jj > Z_j):
                neighbours.append((Z_jj, jj))

        # 2. ---------- Assign a cluster ----------
        # 2.1. ---------- Declare a new cluster (peak) ----------
        if len(neighbours) == 0:
            C[j] = cluster_id  # assign cluster id, `+1` to avoid `0` id
            clusters[cluster_id] = {
                j: Z_j}  # add to the cluster list, NUMBA (v0.60.0) does not support List and Set as values
            prominences[cluster_id] = (j, Z_j, 0)  # ID, birth, death (non-negative values only)
            cluster_id += 1

        # 2.2. ---------- Assign to existing cluster (peak) ----------
        else:
            max_i = np.argmax(np.array([xi[0] for xi in neighbours]))
            Z_jm, jm = neighbours[max_i]  # max among the neighbours

            C[j] = cid_jm = C[jm]  # assign ID of the highest cluster in the neighbourhood
            clusters[cid_jm][j] = Z_j  # add to the cluster 'j'
            Z_rj = prominences[cid_jm][1]  # root vertex height of local max gradient

            # 2.3 ---------- Merge neighbouring clusters ----------
            for Z_k, k in neighbours:
                cid_k = C[k]
                Z_rk = prominences[cid_k][1]  # current root vertex height
                low_prominence = (min(Z_rk, Z_rj) - Z_j < prominence_thr)

                if ((cid_k != cid_jm) and  # to avoid repeated merging 'k -> j'
                        low_prominence and  # prominence threshold
                        (clusters.get(cid_jm, None) is not None)):  # to avoid repeated merging 'j -> k'

                    cids = np.array([cid_jm, cid_k])  # cluster IDs
                    roots = np.array([Z_rj, Z_rk])  # their root vertices, to compare
                    cids_sort = np.argsort(roots)  # sort, to find the most prominent cluster

                    cid_d, cid_l = cids[cids_sort]  # to die, to live

                    prominences[cid_d] = (prominences[cid_d][0],  # index
                                          prominences[cid_d][1],  # birth level
                                          Z_j)  # kill and update death

                    clusters_d = clusters.pop(cid_d)  # remove dead cluster
                    clusters[cid_l].update(clusters_d)  # merge dead cluster with live cluster

                    # update cluster mask after merging
                    for ind_d in clusters_d.keys():
                        C[ind_d] = cid_l  # update cluster IDs

    # Re-structure output
    clusters_list = {}
    prominences_list = {}
    for cid, prom in prominences.items():
        peak_id = prom[0]
        prominences_list[peak_id] = prom[1] - prom[2]
        cluster = clusters.get(cid, None)
        if cluster is not None:
            clusters_list[peak_id] = np.array(list(cluster.keys()))

    return clusters_list, prominences_list


@nb.njit(["f8[:, :](f8[:, :], UniTuple(i8, 2))",
          "f4[:, :](f4[:, :], UniTuple(i8, 2))",
          "b1[:, :](b1[:, :], UniTuple(i8, 2))"],
         cache=True, parallel=True)
def _topo_density_estimation(Z, q):
    """ Basically, it is equivalent to 2D convolution. But since I can ignore 'zero' pixels,
        it can speed up it a little bit, as well as I use 'parallel' loop.
    """
    Nf, Nt = Z.shape
    q_f, q_t = q
    K = (2 * q_f + 1) * (2 * q_t + 1)
    k_weight = 1.0 / K
    D = np.zeros_like(Z)

    for flat_index in nb.prange(0, Z.size):  # parallel
        if not Z.flat[flat_index] > 0.0:
            continue

        i, j = ind = divmod(flat_index, Nt)

        i1, i2 = max(i - q_f, 0), min(i + q_f + 1, Nf)
        j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

        D[ind] = Z[i1: i2, j1: j2].mean()  # normal smoothing
        # nzeros = K - np.count_nonzero(Z[i1: i2, j1: j2])
        # D[ind] = Z[ind] / nzeros

    return D


IND_ND = lambda n: f'UniTuple(i8, {n})'


@nb.njit([f"Tuple((i4[:, :], DictType(UniTuple(i8, 2), f8)))(f4[:, :], {IND_ND(2)}, f8, f8, b1)",
          f"Tuple((i4[:, :], DictType(UniTuple(i8, 2), f8)))(f8[:, :], {IND_ND(2)}, f8, f8, b1)",
          f"Tuple((i4[:, :], DictType(UniTuple(i8, 2), f8)))(b1[:, :], {IND_ND(2)}, f8, f8, b1)"],
         cache=True)
def _TopoClustering2D(Z, q, q_f_oct, prominence_thr, smoothing):
    """
    This is an adaptation of ToMATo algorithm [1].

    Notes:
        1. This implementation adds additional criterion on size of clusters.
        2. Small clusters get merged into neighbouring clusters, otherwise removed, see 'merge_rule' below.

    Arguments:
        Z : np.ndarray (Nf, Nt) : 2D array of density values.
        q : tuple[int, int] : neighbourhood distance for clustering `(q_f, q_t)`.
        q_f_oct : float : equivalent to `q_f` (see above), but in log2 scale (octaves), for frequency only.
                       By default, zero and not used, but if non-zero, replaces `q_f`.
        prominence_thr : float : threshold for cluster density prominence.

    Reference: [1] Chazal, F., Guibas, L. J., Oudot, S. Y., & Skraba, P. (2013).
               Persistence-based clustering in Riemannian manifolds. Journal of the ACM (JACM), 60(6), 1-38.
    """
    Nf, Nt = shape = Z.shape
    freq_distance_octaves = q_f_oct

    C = np.zeros(shape, dtype=np.int64)  # cluster IDs

    if smoothing:
        D = _topo_density_estimation(Z, q)  # smoothed density map, takes into account connectivity
    else:
        D = Z

    q_f, q_t = q
    sort_inds = np.argsort(-D.ravel())  # decreasing order sorting, 1D flat index

    clusters = {}  # to store cluster members
    prominences = {}  # to store cluster prominences
    neighbour_clusters = {-1: -1}  # to keep the largest neighbouring cluster to a cluster
    cluster_id = 1
    for flat_index in sort_inds:
        D_ij = D.flat[flat_index]
        if not D_ij > 0.0:
            continue  # skip null pixels

        # 1. ---------- Find neighbours ----------
        ind = divmod(flat_index, Nt)
        i, j = ind

        # define lookout window to search for neighbors
        if freq_distance_octaves > 0.0:  # log2 step for frequency
            i1 = min(i - 1, round(i * 2 ** (-freq_distance_octaves)))  # octave is power of 2
            i2 = max(i + 1, round(i * 2 ** freq_distance_octaves)) + 1  # octave is power of 2
        else:  # normal linear step
            i1, i2 = i - q_f, i + q_f + 1

        i1, i2 = max(i1, 0), min(i2, Nf)
        j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

        neighbours = []
        for ii in range(i1, i2):
            for jj in range(j1, j2):
                ind_kk = (ii, jj)
                D_kk = D[ind_kk]
                if (ind_kk != ind) and (D_kk > D_ij):
                    neighbours.append((D_kk, ind_kk))

        # 2. ---------- Assign a cluster ----------
        # 2.1. ---------- Declare a new cluster (peak) ----------
        if len(neighbours) == 0:
            C[ind] = cluster_id  # assign cluster id, `+1` to avoid `0` id
            clusters[cluster_id] = {
                ind: D_ij}  # add to the cluster list, NUMBA does not support List and Set as values (0.60.0)
            prominences[cluster_id] = (ind, D_ij, -np.inf)  # ID, birth, death
            cluster_id += 1

        # 2.2. ---------- Assign to existing cluster (peak) ----------
        else:
            max_i = np.argmax(np.array([xi[0] for xi in neighbours]))
            D_j, ind_j = neighbours[max_i]  # max among the neighbours

            C[ind] = cid_j = C[ind_j]  # assign ID of the highest cluster in the neighbourhood
            clusters[cid_j][ind] = D_ij  # add current point to the cluster 'j'
            D_rj = prominences[cid_j][1]  # root vertex height of local max gradient

            # 2.3 ---------- Merge neighbouring clusters ----------
            for Z_k, ind_k in neighbours:
                cid_k = C[ind_k]
                D_rk = prominences[cid_k][1]  # current root vertex height
                low_prominence = (min(D_rk, D_rj) - D_ij < prominence_thr)

                # This 'if' branch is for merging not prominent clusters
                if ((cid_k != cid_j) and  # to avoid repeated merging 'k -> j'
                        low_prominence and  # prominence threshold
                        (clusters.get(cid_j, None) is not None)):  # to avoid repeated merging 'j -> k'

                    cids = np.array([cid_j, cid_k])  # cluster IDs
                    roots = np.array([D_rj, D_rk])  # their root vertices, to compare
                    cids_sort = np.argsort(roots)  # sort, to find the most prominent cluster

                    cid_d, cid_l = cids[cids_sort]  # to die, to live

                    prominences[cid_d] = (prominences[cid_d][0],  # index
                                          prominences[cid_d][1],  # birth level
                                          D_ij)  # kill and update death

                    clusters_d = clusters.pop(cid_d)  # clear dead cluster
                    clusters[cid_l].update(clusters_d)  # merge dead cluster with live cluster

                    # update cluster mask after merging
                    for ind_d in clusters_d.keys():
                        C[ind_d] = cid_l  # update cluster IDs

    # 4. Re-structure prominence output
    prominences_list = {}
    for cid, prom in prominences.items():
        peak_id = prom[0]
        prominences_list[peak_id] = prom[1] - prom[2]

    # 5. ---------- Assign new labels 1...K ----------
    C = np.zeros(shape, dtype=np.int32)
    for k, cluster in enumerate(clusters.values()):
        for ind in cluster.keys():
            C[ind] = k + 1

    return C, prominences_list


class TopoPhaseSeparator(PhaseSeparator):
    def __init__(self,
                 dt_sec=None,
                 is_1D_separation=True,
                 time_dist_len=None,
                 min_prominence=None,
                 num_periods_smooth=0.25,
                 max_num_phases=None,
                 min_num_phases=1,
                 period_estimator='freq',
                 norm_func=np.max,
                 on_distribution_name='spectrogram_SNR',
                 density_estimate_1D='rel_mean',
                 density_estimate_2D=True,
                 freq_dist_len=None,
                 freq_dist_octaves=None,
                 aggr_func=np.max):

        self.dt_sec = dt_sec

        # Topological clustering general params (1D & 2D)
        self.time_dist_len = time_dist_len
        self.freq_dist_len = freq_dist_len or 1
        self.freq_dist_octaves = freq_dist_octaves or 0
        self.min_prominence = min_prominence

        self.max_num_phases = max_num_phases
        self.min_num_phases = min_num_phases

        self.is_1D_separation = is_1D_separation

        self.norm_func = norm_func
        self.num_periods_smooth = num_periods_smooth
        self.period_estimator = period_estimator

        self.density_estimate_1D = density_estimate_1D
        self.density_estimate_2D = density_estimate_2D

        # Auxiliary params
        self.on_distribution_name = on_distribution_name
        self.aggr_func = aggr_func
        self.reset()

    def time_projector_1D(self, time_inds, freq_inds, distribution):

        t_min_ind, t_max_ind = np.min(time_inds), np.max(time_inds)
        if self.density_estimate_1D == 'max':
            # 1) saves the scale/magnitude and does not scatter energy unlike sum or mean,
            # 2) gives the brightest spots from t-f domain,
            max_energy = np.zeros(t_max_ind + 1)
            np.maximum.at(max_energy, time_inds, distribution)  # the fastest and most concise way so far
            proj_curve = max_energy[t_min_ind:]

        elif self.density_estimate_1D == 'rel_mean':
            # Relative mean - normalized by the cluster bandwidth (max bandwidth)
            # 1) saves the scale/magnitude and does not scatter energy unlike naive 'sum' or 'mean',
            # 2) unlike 'max', it reduces the influence of narrow-band high-energy parts
            counts = np.bincount(time_inds)
            max_count = np.max(counts)
            proj_curve = np.bincount(time_inds, weights=distribution)
            proj_curve = proj_curve[t_min_ind:] / max_count

        elif callable(self.density_estimate_1D):
            proj_curve = self.density_estimate_1D(time_inds, freq_inds, distribution)
        else:
            raise NotImplementedError(f"Mode {self.density_estimate_1D} is not implemented")

        # Normalizing by local Max value
        if callable(self.norm_func):
            proj_curve /= self.norm_func(proj_curve)

        return proj_curve

    @staticmethod
    def estimate_period(period_estimator, freq, time, distribution_aggr):

        d_sum = distribution_aggr.sum()
        if 'freq' in period_estimator.casefold():
            f_peak = np.sum(freq * distribution_aggr) / d_sum  # centroid frequency
            time_window_sec = (1 / f_peak)
        else:
            t_peak = np.sum(time * distribution_aggr) / d_sum  # centroid time
            t_var = np.sum((time - t_peak) ** 2 * distribution_aggr) / d_sum
            time_window_sec = np.sqrt(t_var)  # std around centroid time

        return time_window_sec

    def __call__(self, freq, time, values_dict, inds):

        distribution_aggr = aggregate_feature_values(values_dict[self.on_distribution_name],
                                                     self.aggr_func)

        if self.dt_sec is None:
            freq_inds, time_inds = inds
            t_min_ind, t_max_ind = np.min(time_inds), np.max(time_inds)
            time_index = np.arange(0, t_max_ind - t_min_ind + 1)
            self.dt_sec = (time.max() - time.min()) / (len(time_index) - 1)

        if self.time_dist_len is None:
            period = self.estimate_period(self.period_estimator, freq, time, distribution_aggr)
            time_dist = round(period / 2 * self.num_periods_smooth / self.dt_sec)
        else:
            time_dist = self.time_dist_len

        time_dist = max(1, time_dist)  # at least 1

        if self.is_1D_separation:
            return self.phase_separation_1D(freq, time, distribution_aggr, inds, time_dist)
        else:
            assert (self.freq_dist_len is not None) or (self.freq_dist_octaves is not None)
            return self.phase_separation_2D(freq, time, distribution_aggr, inds, time_dist)

    def phase_separation_1D(self, freq, time, distribution_aggr, inds, time_dist):
        freq_inds, time_inds = inds

        t_min_ind = np.min(time_inds)
        time_inds_local = time_inds - t_min_ind

        # 1. Characteristic curve / projection onto time

        projected_curve = self.time_projector_1D(time_inds, freq_inds, distribution_aggr)

        # 2. Find peaks - separate into separate time segments of prominent peaks (phases)

        if self.min_prominence:
            phases_id, _ = _TopoClustering1D(projected_curve, time_dist,
                                             self.min_prominence)  # prominent phases (clusters)

        else:  # if 'min_prominence' not given, then approximate by using big GAP in Persistence diagram
            _, prominences = _TopoClustering1D(projected_curve, time_dist, np.inf)  # to get prominences for all peaks
            min_prominence = self.estimate_min_prominence(prominences.values(),
                                                          max_k=self.max_num_phases,
                                                          min_k=self.min_num_phases)
            phases_id, _ = _TopoClustering1D(projected_curve, time_dist, min_prominence)  # prominent phases (clusters)

        # 3. Write Phase_ID

        phases_id = dict(phases_id)
        peak_ids = list(phases_id.keys())
        intervals = [phases_id[pid] for pid in peak_ids]
        intervals = [(np.min(intv), np.max(intv)) for intv in intervals if len(intv) > 0]

        Phase_ID = np.zeros_like(time_inds)
        for t1, t2 in intervals:
            pi_inds = (t1 <= time_inds_local) & (time_inds_local <= t2)
            Phase_ID[pi_inds] = self.counter
            self.counter += 1

        return Phase_ID

    def phase_separation_2D(self, freq, time, distribution_aggr, inds, time_dist):
        freq_inds, time_inds = inds

        t_min_ind, t_max_ind = np.min(time_inds), np.max(time_inds)
        f_min_ind, f_max_ind = np.min(freq_inds), np.max(freq_inds)
        time_inds_local = time_inds - t_min_ind
        freq_inds_local = freq_inds - f_min_ind

        # 1. Distribution on grid (matrix)
        flen = f_max_ind - f_min_ind + 1
        tlen = t_max_ind - t_min_ind + 1

        Z = np.zeros((flen, tlen), dtype=distribution_aggr.dtype)
        Z[freq_inds_local, time_inds_local] = distribution_aggr

        # 2. Find peaks - separate into separate time segments of prominent peaks (phases)
        q = (self.freq_dist_len, time_dist)

        if self.min_prominence:
            min_prominence = self.min_prominence

        else:  # if 'min_prominence' not given, approximate by big GAP in Persistence diagram
            _, prominences = _TopoClustering2D(Z, q, self.freq_dist_octaves,
                                               np.inf, self.density_estimate_2D)  # to get prominences for all peaks
            min_prominence = self.estimate_min_prominence(prominences.values())

        PID, _ = _TopoClustering2D(Z, q, self.freq_dist_octaves,
                                   min_prominence,
                                   self.density_estimate_2D)  # prominent phases (clusters)

        Phase_ID = PID[freq_inds_local, time_inds_local] + self.counter - 1
        self.counter += 1

        return Phase_ID

    @staticmethod
    def estimate_min_prominence(prominences, max_k=None, min_k=1, ndiff=1):
        prominences = sorted(prominences)  # sort in ascending order

        if max_k is not None:  # min prominence by max number of phases
            assert (not max_k < 0)
            k_ind = len(prominences) - min(len(prominences), max_k) - 1  # `-1` to index form
            min_prominence = prominences[0] / 2 if k_ind < 0 else np.mean(prominences[k_ind: k_ind + 2])
        else:  # min prominence by biggest gap in persistence diagram
            inds = slice(0, len(prominences) - min_k)  # min number of peaks to be detected
            prominences_diff = np.diff(prominences[inds], n=ndiff)  # take diff
            if len(prominences_diff) > 2:
                max_diff_id = prominences_diff.argmax()  # find big GAP, except for the last INF value
                min_prominence = np.mean(prominences[max_diff_id + ndiff - 1: max_diff_id + 1 + ndiff])
                # mean of two with the biggest difference
            else:
                min_prominence = prominences[0] / 2  # when only 2 peaks in total

        return min_prominence


class SplinePhaseSeparator(PhaseSeparator):
    def __init__(self,
                 dt_sec=None,
                 min_prominence=None,
                 num_periods_smooth=None,
                 min_phase_periods=0.0,
                 max_num_phases=None,
                 min_num_phases=1,
                 norm_func=np.max,
                 on_distribution_name='spectrogram_SNR',
                 period_estimator='frequency',
                 density_estimator='rel_mean',
                 aggr_func=np.max):

        self.dt_sec = dt_sec

        self.min_prominence = min_prominence
        self.num_periods_smooth = num_periods_smooth or 0
        self.min_phase_periods = min_phase_periods
        self.max_num_phases = max_num_phases
        self.min_num_phases = min_num_phases
        self.density_estimator = density_estimator
        self.norm_func = norm_func
        self.period_estimator = period_estimator  # "time_std"

        # Auxiliary params
        self.on_distribution_name = on_distribution_name
        self.aggr_func = aggr_func
        self.reset()

    def time_projector(self, time_inds, freq_inds, distribution):

        t_min_ind, t_max_ind = np.min(time_inds), np.max(time_inds)
        if self.density_estimator == 'max':
            # 1) saves the scale/magnitude and does not scatter energy unlike sum or mean,
            # 2) gives the brightest spots from t-f domain,
            max_energy = np.zeros(t_max_ind + 1)
            np.maximum.at(max_energy, time_inds, distribution)  # the fastest and most concise way so far
            proj_curve = max_energy[t_min_ind:]

        elif self.density_estimator == 'rel_mean':
            # Relative mean - normalized by the cluster bandwidth (max bandwidth)
            # 1) saves the scale/magnitude and does not scatter energy unlike naive 'sum' or 'mean',
            # 2) unlike 'max', it reduces the influence of narrow-band high-energy parts
            counts = np.bincount(time_inds)
            max_count = np.max(counts)
            proj_curve = np.bincount(time_inds, weights=distribution)
            proj_curve = proj_curve[t_min_ind:] / max_count

        elif callable(self.density_estimator):
            proj_curve = self.density_estimator(time_inds, freq_inds, distribution)
        else:
            raise NotImplementedError(f"Mode {self.density_estimator} is not implemented")

        # Normalizing by local Max value
        if callable(self.norm_func):
            proj_curve /= self.norm_func(proj_curve)

        return proj_curve

    @staticmethod
    def estimate_period(period_estimator, freq, time, distribution_aggr):

        if 'freq' in period_estimator.casefold():
            # f_peak = np.sum(freq * distribution_aggr) / d_sum  # centroid frequency
            f_peak = freq[distribution_aggr.argmax()]  # peak frequency, does not underestimate due to low-freq coda
            time_window_len = (1 / f_peak)
        else:
            # t_peak = np.sum(time * distribution_aggr) / d_sum  # centroid time
            t_peak = time[distribution_aggr.argmax()]  # peak time, does not overestimate due to long coda
            d_sum = distribution_aggr.sum()
            t_std = np.sum((time - t_peak) ** 2 * distribution_aggr) / d_sum
            time_window_len = np.sqrt(t_std)

        return time_window_len

    @staticmethod
    def filter_phases(bounds, prominence, min_width, min_phases, max_phases, min_prominence):
        if len(prominence) == 0:
            return np.array([[0, np.inf]])
        else:
            if min_prominence is None:
                min_prominence = TopoPhaseSeparator.estimate_min_prominence(prominence,
                                                                            max_k=max_phases,
                                                                            min_k=min_phases,
                                                                            ndiff=1)

            # 1. Filter by time width and prominence
            # Merging rule: Short and not prominent phases are merged to:
            # Right: if it is stronger than the right one
            # Left: otherwise

            intervals = np.stack([bounds[:-1], bounds[1:]], axis=-1)
            widths = np.diff(intervals, axis=-1)
            mask = np.full(len(prominence), True, dtype=bool)
            if len(widths) > 1:
                for i, (wi, pi) in enumerate(zip(widths, prominence)):
                    if (wi < min_width) or (pi < min_prominence):
                        within_len = (i < len(mask) - 1)
                        if (i == 0) or (within_len and (pi > prominence[i + 1])):
                            mask[i + 1] = False
                        else:
                            mask[i] = False

            bounds = intervals[mask, 0]
            prominence = prominence[mask]

            # 2. Get 'N' most prominent
            # Merging rule: Weak are merged to the left ones as "energy decreases with time"
            # Assumption: a weaker part cannot arrive before the stronger part, only after.
            # Therefore, the weak small phases are assigned to the preceding stronger phase.
            if max_phases is not None:
                prominent_phases = np.argsort(prominence)[::-1][:max_phases]  # get 'N' most prominent
                bounds = np.sort(bounds[prominent_phases])

            # 3. Add 'start' and 'end' bounds
            bounds = np.r_[0, bounds[1:], np.inf]

            return np.stack([bounds[:-1], bounds[1:]], axis=-1)

    def __call__(self, freq, time, values_dict, inds):

        freq_inds, time_inds = inds
        distribution_aggr = aggregate_feature_values(values_dict[self.on_distribution_name],
                                                     self.aggr_func)

        t_min_ind, t_max_ind = np.min(time_inds), np.max(time_inds)
        time_index = np.arange(0, t_max_ind - t_min_ind + 1)

        if self.dt_sec is None:
            self.dt_sec = (time.max() - time.min()) / (len(time_index) - 1)

        k = 3  # cubic spline (we use 3rd derivatives, k_min = 3)

        if len(time_index) < k + 2:  # 5 points min for spline
            Phase_ID = np.full_like(time_inds, self.counter)
            self.counter += 1
            return Phase_ID

        else:
            # 1. Characteristic curve / projection onto time
            projected_curve = self.time_projector(time_inds, freq_inds, distribution_aggr)

            # 2. Smoothed BSpline

            # 2.1 Smmothing window
            # Smoothing window is proportional to centroid cluster frequency (quasi-monochromatic event)
            # Smoothing is to suppress noise oscillations shorter than the event period
            # Phases inside one cluster are assumed to have similar frequency

            period_sec = self.estimate_period(self.period_estimator, freq, time, distribution_aggr)
            time_window_len = period_sec * self.num_periods_smooth / self.dt_sec

            if time_window_len > time_index[-1] / 2:
                time_window_len = time_index[-1] / 2

            # 2.2 Spline knots
            # Position of Spline knots is proportional to centroid frequency (main period)
            # it will smooth out high-freq noise
            # Note: min 3 points for the cluster (min, mid, max)
            if time_window_len > 0:
                knots = np.r_[[time_index[0]] * (k + 1),
                              np.arange(time_window_len, time_index[-1], time_window_len),
                              [time_index[-1]] * (
                                          k + 1)]  # composing all knots, contain at least 3 points in the cluster
                knots = knots.astype(float)
                # 'knots' have at least (min, mid, max)

                # 2.3 Spline fit
                try:
                    spline = interpolate.make_lsq_spline(time_index, projected_curve, t=knots, k=k)  # LSQ spline
                except:
                    spline = interpolate.make_interp_spline(time_index, projected_curve, k=k)  # Normal spline
            else:
                spline = interpolate.make_interp_spline(time_index, projected_curve, k=k)  # Normal spline

            # 3. Getting Phases intervals
            interp = interpolate.PPoly.from_spline(spline)  # Make piecewise-polynomial to have `roots` methods

            # 3.1 Derivatives roots
            der1_interp = interp.derivative(1)
            der2_interp = interp.derivative(2)

            zeros1 = der1_interp.roots()

            # 3.2 Filter roots for increasing parts only
            sign12 = der2_interp(zeros1)
            maxs = zeros1[(sign12 < 0)]  # maximums (peaks)
            mins = zeros1[(sign12 > 0)]  # minimums (valleys)

            peaks = np.sort(maxs)  # reference points
            bounds = []

            # 3.3 Compose list of phases bounds
            for i, p in enumerate(peaks):
                zp = mins[mins <= p]  # potential borders
                zp = zp.max() if len(zp) else None

                if (zp is not None) and (zp not in bounds):
                    zb = zp
                else:  # average with previous if no mininum to the left
                    pn = peaks[i - 1] if i > 0 else p
                    zb = (p + pn) / 2

                bounds.append(zb)
            bounds = np.array(bounds)

            # 3.4 Filter bounds (peaks) by prominence and merge to the biggest neighbour
            prominence = spline(peaks) - spline(bounds)  # prominence of peaks
            bounds = np.r_[0, bounds[1:], time_index[-1]]
            min_width_len = period_sec * self.min_phase_periods / self.dt_sec

            intervals = self.filter_phases(bounds, prominence, min_width_len,
                                           self.min_num_phases, self.max_num_phases,
                                           self.min_prominence)

            intervals += t_min_ind

            # 4. Labeling intervals
            Phase_ID = np.zeros_like(time_inds)

            for p1, p2 in intervals:
                pi_inds = (p1 <= time_inds) & (time_inds < p2)
                Phase_ID[pi_inds] = self.counter
                self.counter += 1

            return Phase_ID


def event_id_recorder(freq, time, values_dict, inds):
    return {'Event_ID': values_dict['spectrogram_event_ID'].flat[0]}

