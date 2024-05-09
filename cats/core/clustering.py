"""
    Functions for clustering trimmed spectrograms (in Time-Frequency or Space-Time-Frequency domain).
"""

import numpy as np
import numba as nb
from .utils import ReshapeArraysDecorator
from .date import _xi
from scipy import special, stats, ndimage
import pandas as pd
from functools import partial


# ------------------- CLUSTERING ------------------- #


def Clustering(SNR, /, q, s, log_freq_cluster):
    """
        Performs neighbor-based clustering/labelling of trimmed SNR spectrograms (parallelized).
        It is based on Depth-First Search (DFS) algorithm. Works analogous to `scipy.ndimage.label`,
        but this supports `q` > 1 (see below for `q`).

        If `len(q) = 2` then 2-dimensional clustering in "Frequency x Time" is used, `SNR.shape=(M, Nf, Nt)`
        If `len(q) = 3` then 3-dimensional clustering in "Trace x Frequency x Time" is used, `SNR.shape=(M, Nc, Nf, Nt)`

        Arguments:
            SNR : np.ndarray (M, Nf, Nt) or (M, Nc, Nf, Nt) :
            Trimmed SNR spectrogram where nonzero elements (SNR > 0) will be labelled.
                                    `M` is number of spectrograms,
                                    `Nc` is number of traces,
                                    `Nf` is frequency axis,
                                    `Nt` is time axis.
            q : tuple(int, 2) / tuple(int, 3) : neighborhood distance for clustering `(q_f, q_t)` or `(q_c, q_f, q_t)`.
                                                `q_c` for traces, `q_f` for frequency, `q_t` for time.
            s : tuple(int, 2) / tuple(int, 3) : minimum cluster sizes `(s_f, s_t)` or `(s_c, s_f, s_t)`.
                                                `s_c` for traces, `s_f` for frequency, `s_t` for time.
            log_freq_cluster : tuple(float, float) : equivalent to `s_f` and `q_f` (see above), but in
                                logarithmic scale (log10), for frequency axis only, in the following order:
                                (cluster_size_f_logHz, cluster_distance_f_logHz).
                                By default, they are zeros and not used, but if given '> 0',
                                 then they replace `s_f` and `q_f` respectively.
    """
    func = {2: _ClusteringN2D, 3: _ClusteringN3D}
    return func[len(q)](SNR, q, s, log_freq_cluster)


IND_ND = lambda n: f'UniTuple(i8, {n})'
CLUSTER_STAT_SIGNATURE_ND = lambda n: f"Tuple(({IND_ND(n)}, {IND_ND(n)}))"


@nb.njit([f"{CLUSTER_STAT_SIGNATURE_ND(2)}({IND_ND(2)}, {IND_ND(2)}, f8[:, :], i4[:, :], i4, f8)",
          f"{CLUSTER_STAT_SIGNATURE_ND(2)}({IND_ND(2)}, {IND_ND(2)}, f4[:, :], i4[:, :], i4, f8)"],
         cache=True)
def depth_first_search_2D(ind, q, SNR, C, cid, log_freq_distance):
    q_f, q_t = q
    Nf, Nt = SNR.shape

    mins = ind
    maxs = ind

    stack = [ind]

    while stack:
        # Depth-First Search (DFS, First-In / Last-Out --> stack.pop(-1))
        # DFS scales better with `q` than Breadth-First (BFS, First-In / First-Out --> stack.pop(0))
        curr = stack.pop(-1)  # take element to label and scan neighbors

        if C[curr] == cid:  # check if it was labelled before
            continue

        C[curr] = cid  # label element

        mins = (min(mins[0], curr[0]), min(mins[1], curr[1]))  # update min indexes
        maxs = (max(maxs[0], curr[0]), max(maxs[1], curr[1]))  # update max indexes

        # define lookout window to search for neighbors
        i, j = curr
        if log_freq_distance > 0.0:
            i1 = min(i - 1, round(i * 10 ** (-log_freq_distance)))
            i2 = max(i + 1, round(i * 10 ** log_freq_distance)) + 1
        else:
            i1, i2 = i - q_f, i + q_f + 1

        i1, i2 = max(i1, 0), min(i2, Nf)
        j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

        # search for neighbors to scan next
        for li in range(i1, i2):
            for lj in range(j1, j2):
                lind = (li, lj)
                nonzero = (SNR[lind] > 0.0)  # SNR must be > 0
                nonvisited = (C[lind] == 0)  # only non-visited elements are added to stack
                if nonzero and nonvisited:
                    stack.append(lind)  # put to the stack
                    C[lind] = -cid  # label as added to the stack

    return mins, maxs


@nb.njit([f"Tuple((f8[:, :], i4[:, :]))(f8[:, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))",
          f"Tuple((f4[:, :], i4[:, :]))(f4[:, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))"],
         cache=True)
def _Clustering2D(SNR, q, s, log_freq_cluster):
    shape = SNR.shape
    log_freq_width, log_freq_distance = log_freq_cluster

    # Finding and labelling clusters
    C = np.zeros(shape, dtype=np.int32)
    cluster_size = {}  # remembers cluster sizes
    cid = 1
    for i, j in np.argwhere(SNR):
        ind = (i, j)
        if C[ind] < 1:
            cluster_size[cid] = depth_first_search_2D(ind, q, SNR, C, cid, log_freq_distance)
            cid += 1

    # Filtering clusters by sizes
    s_f, s_t = s
    k = 1
    cluster_newid = {}  # remembers old cluster IDs to relabel them
    for cid, (mins, maxs) in cluster_size.items():
        if log_freq_width > 0.0:
            min_frequency_width = np.log10((maxs[0] + 1) / (mins[0] + 1e-8)) >= log_freq_width
        else:
            min_frequency_width = (maxs[0] - mins[0] + 1 >= s_f)
        cluster_counted = (min_frequency_width and  # min frequency width
                           (maxs[1] - mins[1] + 1 >= s_t))  # min time duration
        if cluster_counted:
            cluster_newid[cid] = k  # old ID ---> new ID
            k += 1

    # Relabelling clusters
    SNRK = np.zeros_like(SNR)
    for i, j in np.argwhere(C):
        ind = (i, j)
        C[ind] = new_id = cluster_newid.get(C[ind], 0)
        if new_id > 0:
            SNRK[ind] = SNR[ind]

    return SNRK, C


@ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=False, output_num=2, first_shape=True)
@nb.njit([f"Tuple((f8[:, :, :], i4[:, :, :]))(f8[:, :, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))",
          f"Tuple((f4[:, :, :], i4[:, :, :]))(f4[:, :, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))"],
         parallel=True, cache=True)
def _ClusteringN2D(SNR, q, s, log_freq_cluster):
    SNRK = np.empty_like(SNR)
    C = np.empty(SNR.shape, dtype=np.int32)
    for i in nb.prange(SNR.shape[0]):
        SNRK[i], C[i] = _Clustering2D(SNR[i], q, s, log_freq_cluster)
    return SNRK, C


@nb.njit([f"{CLUSTER_STAT_SIGNATURE_ND(3)}({IND_ND(3)}, {IND_ND(3)}, f8[:, :, :], i4[:, :, :], i4, f8)",
          f"{CLUSTER_STAT_SIGNATURE_ND(3)}({IND_ND(3)}, {IND_ND(3)}, f4[:, :, :], i4[:, :, :], i4, f8)"],
         cache=True)
def depth_first_search_3D(ind, q, SNR, C, cid, log_freq_distance):
    """ See detailed comments in `depth_first_search_2D` """

    q_c, q_f, q_t = q
    Nc, Nf, Nt = SNR.shape

    mins = ind
    maxs = ind

    stack = [ind]

    while stack:
        curr = stack.pop(-1)

        if C[curr] == cid:
            continue

        C[curr] = cid

        mins = (min(mins[0], curr[0]), min(mins[1], curr[1]), min(mins[2], curr[2]))
        maxs = (max(maxs[0], curr[0]), max(maxs[1], curr[1]), max(maxs[2], curr[2]))

        k, i, j = curr
        if log_freq_distance > 0.0:
            i1 = min(i - 1, round(i * 10**(-log_freq_distance)))
            i2 = max(i + 1, round(i * 10**log_freq_distance)) + 1
        else:
            i1, i2 = i - q_f, i + q_f + 1

        k1, k2 = max(k - q_c, 0), min(k + q_c + 1, Nc)
        i1, i2 = max(i1, 0), min(i2, Nf)
        j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

        for lk in range(k1, k2):
            for li in range(i1, i2):
                for lj in range(j1, j2):
                    lind = (lk, li, lj)
                    nonzero = (SNR[lind] > 0.0)
                    nonvisited = (C[lind] == 0)
                    if nonzero and nonvisited:
                        stack.append(lind)
                        C[lind] = -cid

    return mins, maxs


@nb.njit([f"Tuple((f8[:, :, :], i4[:, :, :]))(f8[:, :, :], {IND_ND(3)}, {IND_ND(3)}, UniTuple(f8, 2))",
          f"Tuple((f4[:, :, :], i4[:, :, :]))(f4[:, :, :], {IND_ND(3)}, {IND_ND(3)}, UniTuple(f8, 2))"],
         cache=True)
def _Clustering3D(SNR, q, s, log_freq_cluster):
    shape = SNR.shape
    log_freq_width, log_freq_distance = log_freq_cluster

    # finding clusters
    C = np.zeros(shape, dtype=np.int32)
    cluster_size = {}
    cid = 1
    for k, i, j in np.argwhere(SNR):
        ind = (k, i, j)
        if C[ind] < 1:
            cluster_size[cid] = depth_first_search_3D(ind, q, SNR, C, cid, log_freq_distance)
            cid += 1

    # filtering clusters
    s_c, s_f, s_t = s
    k = 1
    cluster_newid = {}
    for cid, (mins, maxs) in cluster_size.items():
        c1, f1, t1 = mins
        c2, f2, t2 = maxs

        if log_freq_width > 0.0:
            min_frequency_width = np.log10((f2 + 1) / (f1 + 1e-8)) >= log_freq_width
        else:
            min_frequency_width = (f2 - f1 + 1 >= s_f)

        cluster_counted = ((c2 - c1 + 1 >= s_c) and  # min number of traces
                           min_frequency_width and  # min frequency width
                           (t2 - t1 + 1 >= s_t))  # min time duration
        if cluster_counted:
            cluster_newid[cid] = k
            k += 1

    # clusters = {new_cid: clusters[old_cid] for old_cid, new_cid in passed_clusters_cid.items()}

    # assigning allowed clusters
    SNRK = np.zeros_like(SNR)
    for k, i, j in np.argwhere(C):
        ind = (k, i, j)
        C[ind] = newid = cluster_newid.get(C[ind], 0)
        if newid > 0:
            SNRK[ind] = SNR[ind]

    return SNRK, C


@ReshapeArraysDecorator(dim=4, input_num=1, methodfunc=False, output_num=2, first_shape=True)
@nb.njit([f"Tuple((f8[:, :, :, :], i4[:, :, :, :]))(f8[:, :, :, :], {IND_ND(3)}, {IND_ND(3)}, UniTuple(f8, 2))",
          f"Tuple((f4[:, :, :, :], i4[:, :, :, :]))(f4[:, :, :, :], {IND_ND(3)}, {IND_ND(3)}, UniTuple(f8, 2))"],
         parallel=True, cache=True)
def _ClusteringN3D(SNR, q, s, log_freq_cluster):
    SNRK = np.empty_like(SNR)
    C = np.empty(SNR.shape, dtype=np.int32)
    for i in nb.prange(SNR.shape[0]):
        SNRK[i], C[i] = _Clustering3D(SNR[i], q, s, log_freq_cluster)
    return SNRK, C


# ---------------- UTILS ---------------- #

@nb.njit(["f8[:](UniTuple(i8, 2), f8, f8, f8[:], i4[:])",
          "f8[:](UniTuple(i8, 2), f8, f8, f4[:], i4[:])",
          "f8[:](UniTuple(i8, 2), f8, f8, f8[:], i8[:])",
          "f8[:](UniTuple(i8, 2), f8, f8, f4[:], i8[:])"], cache=True)
def _cluster_stats(shape, df, dt, arr, inds):
    nf, nt = shape
    freq_inds, time_inds = np.divmod(inds, nt)

    t1, t2 = np.min(time_inds) - 0.5, np.max(time_inds) + 0.5
    f1, f2 = np.min(freq_inds) - 0.5, np.max(freq_inds) + 0.5
    f1, f2 = max(0.0, f1), min(nf, f2)

    snr_sum = np.sum(arr)
    N = len(arr)

    t_center = np.sum(arr * time_inds) / snr_sum * dt
    f_center = np.sum(arr * freq_inds) / snr_sum * df

    peak_snr = np.max(arr)

    return np.array([t1 * dt,
                     t2 * dt,
                     t_center,
                     f1 * df,
                     f2 * df,
                     f_center,
                     snr_sum / N,
                     peak_snr,
                     N])


def clusters_stats_2D(SNR, CID, df, dt):
    cids = np.arange(1, CID.max() + 1)
    if len(cids) > 0:
        cluster_stats = ndimage.labeled_comprehension(input=SNR,
                                                      labels=CID,
                                                      index=cids,
                                                      func=partial(_cluster_stats, SNR.shape, df, dt),
                                                      out_dtype=np.ndarray,
                                                      default=np.nan,
                                                      pass_positions=True)
    else:
        cluster_stats = []
    names = ['Time_start_sec',
             'Time_end_sec',
             'Time_center_of_mass_sec',
             'Frequency_start_Hz',
             'Frequency_end_Hz',
             'Frequency_center_of_mass_Hz',
             'Average_SNR',
             'Peak_SNR',
             'Area']
    df_stats = pd.DataFrame(columns=pd.Index(names, name='Statistics'),
                            index=pd.Index(cids, name='Cluster_ID'),
                            dtype=float)
    for i, stat in zip(cids, cluster_stats):
        df_stats.loc[i] = stat
    return df_stats


def ClusterCatalogs(SNR, CID, df, dt):
    shape = SNR.shape[:-2]
    clusters_stats = np.empty(shape, dtype=pd.DataFrame)
    for ind in np.ndindex(*shape):
        clusters_stats[ind] = clusters_stats_2D(SNR[ind], CID[ind], df, dt)
    return clusters_stats


def get_clusters_catalogs(cats_result):
    SNR, CID = cats_result.spectrogram_SNR_clustered, cats_result.spectrogram_cluster_ID
    assert (SNR is not None) and (CID is not None), "CATS result must contain `spectrogram_SNR_clustered` and " \
                                                    "`spectrogram_cluster_ID` (use `full_info`)"
    return ClusterCatalogs(SNR, CID, cats_result.stft_frequency, cats_result.stft_dt_sec)


def concatenate_cluster_catalogs(catalog1, catalog2, t0):
    time_shifting = ["Time_start_sec", "Time_end_sec", "Time_center_of_mass_sec"]
    if len(catalog2) > 0:
        catalog2[time_shifting] += t0
        catalog2.index += max(catalog1.index, default=0)
    return pd.concat([catalog1, catalog2])


def concatenate_arrays_of_cluster_catalogs(catalog1, catalog2, t0):
    if (catalog1 is not None) and (catalog2 is not None):
        assert catalog1.shape == catalog2.shape
        for ind in np.ndindex(catalog1.shape):
            catalog1[ind] = concatenate_cluster_catalogs(catalog1[ind], catalog2[ind], t0)
        del catalog2
    return catalog1


# ------------------ Experimental ------------------ #


def _optimalNeighborhoodDistance(p, pmin, qmax, maxN):
    qi = qmax  # in case the cycle is empty
    for qi in range(1, qmax + 1):
        Q = (qi * 2 + 1)**2  # clustering kernel is square
        cdf = stats.binom.cdf(maxN, Q, p)  # probability that maximum `maxN` elements in kernel `Q` are nonzero
        if cdf < pmin:  # choose `qi` which provides probability of noisy nonzero elements not to be present in `Q`
            break
    return max(1, qi - 1)


def OptimalNeighborhoodDistance(minSNR, d=2, pmin=0.95, qmax=10, maxN=1):

    # thresholding function value from DATE, defined for `noise variance = 1`
    xi = _xi(d=d, rho=minSNR)
    # percentage of chi-distributed noise elements to be > `xi` (`d` degrees of freedom)
    p = special.gammaincc(d / 2, xi**2 / 2)
    q_opt = _optimalNeighborhoodDistance(p, pmin=pmin, qmax=qmax, maxN=maxN)
    return q_opt


# -------------------- OLD -------------------- #


@nb.njit(["Tuple((f8[:, :], u4[:, :]))(f8[:, :], UniTuple(i8, 2), UniTuple(i8, 2), f8, f8)",
          "Tuple((f4[:, :], u4[:, :]))(f4[:, :], UniTuple(i8, 2), UniTuple(i8, 2), f8, f8)"], cache=True)
def _Clustering2D_old(SNR, q, s, minSNR, alpha):
    shape = SNR.shape
    Nf, Nt = shape
    C = np.full(shape, -1)
    q_f, q_t = q
    s_f, s_t = s
    clusters = []

    for (i, j), SNRij in np.ndenumerate(SNR):
        if SNRij:
            # selecting area of interest
            i1, i2 = max(i - q_f, 0), min(i + q_f + 1, Nf)
            j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

            snr = SNR[i1: i2, j1: j2]
            if np.count_nonzero(snr) < 2:  # isolated single points are deleted right away
                continue

            c = C[i1: i2, j1: j2]

            cluster_k = []
            neighbor_clusters = []

            # checking existing clusters and remembering not assigned
            for m, cl in np.ndenumerate(c):
                if snr[m]:
                    if cl >= 0:
                        if cl not in neighbor_clusters:
                            neighbor_clusters.append(cl)
                    else:
                        cluster_k.append((i1 + m[0],
                                          j1 + m[1],
                                          snr[m]))

            k = len(neighbor_clusters)
            if k == 0:  # updating collection of clusters
                cluster_assigned = len(clusters)
                clusters.append(cluster_k)
            elif (k == 1) and (len(cluster_k) == 0):  # nothing added
                continue
            else:  # combining different clusters into one
                cluster_assigned = min(neighbor_clusters)
                neighbor_clusters.remove(cluster_assigned)
                for cli in neighbor_clusters:
                    cluster_k += clusters[cli]
                    clusters[cli] = [(Nf, Nt, SNR[-1, -1])]  # meaningless point needed for compilation of list types
                clusters[cluster_assigned] += cluster_k  # add new points

            # assigning clusters
            for (l1, l2, snr) in cluster_k:
                C[l1, l2] = cluster_assigned

    # filtering clusters and projection
    K = np.full(shape, 0, dtype=np.uint32)
    SNRK = np.zeros_like(SNR)
    k = 1
    for cl in clusters:
        if len(cl) > 1:  # to skip those meaningless points which we put for compilation
            cl_arr = np.array(cl)
            cl_f_min, cl_f_max = cl_arr[:, 0].min(), cl_arr[:, 0].max() + 1
            cl_t_min, cl_t_max = cl_arr[:, 1].min(), cl_arr[:, 1].max() + 1
            cl_snr = cl_arr[:, 2].mean()

            cluster_pass = ((cl_f_max - cl_f_min >= s_f) and  # min frequency width
                            (cl_t_max - cl_t_min >= s_t) and  # min time duration
                            (cl_snr >= minSNR) and            # min average energy
                            (len(cl) >= alpha * s_f * s_t))   # min cluster fullness

            if cluster_pass:
                for (l1, l2, snr) in cl:
                    K[l1, l2] = k
                    SNRK[l1, l2] = snr
                k += 1

    return SNRK, K
