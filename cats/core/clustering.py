"""
    Functions for clustering trimmed spectrograms (in Time-Frequency or Space-Time-Frequency domain).
"""

import numpy as np
import numba as nb
from .utils import ReshapeArraysDecorator
from .date import _xi
from scipy import special, stats

###################  CLUSTERING  ###################


@nb.njit("Tuple((i8[:, :], i8[:]))(f8[:, :], UniTuple(i8, 2), UniTuple(i8, 2), f8)")
def _Clustering2D(SNR, q, s, minSNR):
    B = SNR > 0
    shape = B.shape
    Nf, Nt = shape
    C = np.full(shape, -1)
    q_f, q_t = q
    s_f, s_t = s
    clusters = []

    for (i, j), bij in np.ndenumerate(B):
        if bij:
            # selecting area of interest
            i1, i2 = max(i - q_f, 0), min(i + q_f + 1, Nf)
            j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

            b = B[i1: i2, j1: j2]
            if np.sum(b) < 2:  # isolated single points are deleted right away
                continue

            c = C[i1: i2, j1: j2]
            snr = SNR[i1: i2, j1: j2]
            cluster_k = []
            neighbor_clusters = []

            # checking existing clusters and remembering not assigned
            for l, cl in np.ndenumerate(c):
                if b[l]:
                    if (cl >= 0):
                        if (cl not in neighbor_clusters):
                            neighbor_clusters.append(cl)
                    else:
                        cluster_k.append((i1 + l[0],
                                          j1 + l[1],
                                          snr[l]))

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
                    clusters[cli] = [(Nf, Nt, 0.0)]  # meaningless isolated point, will not be used
                clusters[cluster_assigned] += cluster_k  # add new points

            # assigning clusters
            for (l1, l2, snr) in cluster_k:
                C[l1, l2] = cluster_assigned

    # filtering clusters and projection
    K = np.full(shape, 0)
    k = 1
    projection = np.zeros(Nt, dtype=np.int64)
    for cl in clusters:
        if len(cl) > 1:  # all isolated points are skipped
            cl_arr = np.array(cl)
            cl_f_min, cl_f_max = cl_arr[:, 0].min(), cl_arr[:, 0].max() + 1
            cl_t_min, cl_t_max = cl_arr[:, 1].min(), cl_arr[:, 1].max() + 1
            cl_snr = cl_arr[:, 2].mean()
            if (cl_f_max - cl_f_min >= s_f) and (cl_t_max - cl_t_min >= s_t) and (cl_snr >= minSNR):
                projection[cl_t_min: cl_t_max] += k
                for (l1, l2, snr) in cl:
                    K[l1, l2] = k
                k += 1

    return K, projection


@nb.njit("Tuple((i8[:, :, :], i8[:, :]))(f8[:, :, :], UniTuple(i8, 2), UniTuple(i8, 2), f8)", parallel=True)
def _ClusteringN2D(SNR, q, s, minSNR):

    K = np.empty(SNR.shape, dtype=np.int64)
    P = np.empty((SNR.shape[0], SNR.shape[-1]), dtype=np.int64)
    for i in nb.prange(SNR.shape[0]):
        K[i], P[i] = _Clustering2D(SNR[i], q, s, minSNR)
    return K, P


@ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=False, output_num=2, first_shape=True)
def _ClusteringN2D_API(SNR, /, q, s, minSNR):
    return _ClusteringN2D(SNR, q, s, minSNR)


@nb.njit("Tuple((i8[:, :, :], i8[:, :]))(f8[:, :, :], UniTuple(i8, 3), UniTuple(i8, 3), f8)")
def _Clustering3D(SNR, q, s, minSNR):

    B = SNR > 0
    shape = B.shape
    Nc, Nf, Nt = shape
    C = np.full(shape, -1)
    q_c, q_f, q_t = q
    s_c, s_f, s_t = s
    clusters = []

    for (i, j, k), bijk in np.ndenumerate(B):
        if bijk:
            # selecting area of interest
            i1, i2 = max(i - q_c, 0), min(i + q_c + 1, Nc)
            j1, j2 = max(j - q_f, 0), min(j + q_f + 1, Nf)
            k1, k2 = max(k - q_t, 0), min(k + q_t + 1, Nt)

            b = B[i1: i2, j1: j2, k1: k2]
            if np.sum(b) < 2:  # isolated single points are deleted right away
                continue

            c = C[i1: i2, j1: j2, k1: k2]
            snr = SNR[i1: i2, j1: j2, k1: k2]
            cluster_k = []
            neighbor_clusters = []

            # checking existing clusters and remembering not assigned
            for l, cl in np.ndenumerate(c):
                if b[l]:
                    if (cl >= 0):
                        if (cl not in neighbor_clusters):
                            neighbor_clusters.append(cl)
                    else:
                        cluster_k.append((i1 + l[0],
                                          j1 + l[1],
                                          k1 + l[2],
                                          snr[l]))

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
                    clusters[cli] = [(Nc, Nf, Nt, 0.0)]  # meaningless isolated point for compiler, will not be used
                clusters[cluster_assigned] += cluster_k  # add new points

            # assigning clusters
            for (l1, l2, l3, snr) in cluster_k:
                C[l1, l2, l3] = cluster_assigned

    # filtering clusters and projection
    K = np.full(shape, 0)
    projection = np.zeros((Nc, Nt), dtype=np.int64)
    k = 1
    for cl in clusters:
        if len(cl) > 1:
            cl_arr = np.array(cl)
            cl_c_min, cl_c_max = cl_arr[:, 0].min(), cl_arr[:, 0].max() + 1
            cl_f_min, cl_f_max = cl_arr[:, 1].min(), cl_arr[:, 1].max() + 1
            cl_t_min, cl_t_max = cl_arr[:, 2].min(), cl_arr[:, 2].max() + 1
            cl_snr = cl_arr[:, 3].mean()
            if (cl_c_max - cl_c_min >= s_c) and (cl_f_max - cl_f_min >= s_f) and \
               (cl_t_max - cl_t_min >= s_t) and (cl_snr >= minSNR):
                projection[cl_c_min: cl_c_max, cl_t_min: cl_t_max] += k
                for (l1, l2, l3, snr) in cl:
                    K[l1, l2, l3] = k
                k += 1

    return K, projection


@nb.njit("Tuple((i8[:, :, :, :], i8[:, :, :]))(f8[:, :, :, :], UniTuple(i8, 3), UniTuple(i8, 3), f8)", parallel=True)
def _ClusteringN3D(SNR, q, s, minSNR):
    N, Nc, Nf, Nt = SNR.shape
    K = np.empty(SNR.shape, dtype=np.int64)
    P = np.empty((N, Nc, Nt), dtype=np.int64)
    for i in nb.prange(SNR.shape[0]):
        K[i], P[i] = _Clustering3D(SNR[i], q, s, minSNR)
    return K, P


@ReshapeArraysDecorator(dim=4, input_num=1, methodfunc=False, output_num=2, first_shape=True)
def _ClusteringN3D_API(SNR, /, q, s, minSNR):
    return _ClusteringN3D(SNR, q, s, minSNR)


def Clustering(SNR, /, q, s, minSNR):
    """
        Performs clustering (density-based / neighbor-based) of many trimmed spectrograms in parallel.
        If `len(q) = 2` then 2-dimensional clustering in "Frequency x Time" is used, `SNR.shape=(M, Nf, Nt)`
        If `len(q) = 3` then 3-dimensional clustering in "Trace x Frequency x Time" is used, `SNR.shape=(M, Nc, Nf, Nt)`

        Arguments:
            SNR : np.ndarray (M, Nf, Nt) or (M, Nc, Nf, Nt) : Trimmed spectrogram wherein nonzero elements represent
                                    SNR values. `M' is number of spectrograms, `Nc` number of traces,
                                    `Nf` is frequency axis, `Nt` is time axis.
            q : tuple(int, 2) / tuple(int, 3) : neighborhood distance for clustering `(q_f, q_t)` or `(q_c, q_f, q_t)`.
                                                `q_c` for traces, `q_f` for frequency, `q_t` for time.
            s : tuple(int, 2) / tuple(int, 3) : minimum cluster sizes `(s_f, s_t)` or `(s_c, s_f, s_t)`.
                                                `s_c` for traces, `s_f` for frequency, `s_t` for time.
    """
    func = {2 : _ClusteringN2D_API, 3 : _ClusteringN3D_API}
    dim = len(q)
    K, P = func[dim](SNR, q, s, minSNR)
    return K, P


## Experimental ##


@nb.njit("b1[:, :](b1[:, :], UniTuple(i8, 2), i8)")
def _ClusterFilling2D(B, q, min_neighbors):
    Nf, Nt = B.shape
    q_f, q_t = q
    F = np.empty((Nf - q_f * 2,
                  Nt - q_t * 2), dtype=np.bool_)
    for (i, j), fij in np.ndenumerate(F):
        if B[i + q_f, j + q_t]:
            F[i, j] = True
        else:
            F[i, j] = (B[i : i + q_f * 2 + 1,
                         j : j + q_t * 2 + 1].sum() >= min_neighbors)
    return F


@nb.njit("b1[:, :, :](b1[:, :, :], UniTuple(i8, 2), i8)")
def _ClusterFillingN2D(B, q, min_neighbors):
    M, Nf, Nt = B.shape
    q_f, q_t = q
    F = np.empty((M,
                  Nf - q_f * 2,
                  Nt - q_t * 2), dtype=np.bool_)
    for i in nb.prange(M):
        F[i] = _ClusterFilling2D(B[i], q, min_neighbors)
    return F


@ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=False, output_num=1, first_shape=True)
def _ClusterFilling2D_API(B, /, q, min_neighbors):
    q_f, q_t = q
    B_pad = np.pad(B, [(0, 0), (q_f, q_f), (q_t, q_t)], mode='constant', constant_values=0)
    return _ClusterFillingN2D(B_pad.astype(bool), q, min_neighbors)


@nb.njit("b1[:, :, :](b1[:, :, :], UniTuple(i8, 3), i8)")
def _ClusterFilling3D(B, q, min_neighbors):
    Nc, Nf, Nt = B.shape
    q_c, q_f, q_t = q
    F = np.empty((Nc - q_c * 2,
                  Nf - q_f * 2,
                  Nt - q_t * 2), dtype=np.bool_)
    for (i, j, k), fijk in np.ndenumerate(F):
        if B[i + q_c, j + q_f, k + q_t]:
            F[i, j, k] = True
        else:
            F[i, j, k] = (B[i : i + q_c * 2 + 1,
                            j : j + q_f * 2 + 1,
                            k : k + q_t * 2 + 1].sum() >= min_neighbors)
    return F


@nb.njit("b1[:, :, :, :](b1[:, :, :, :], UniTuple(i8, 3), i8)")
def _ClusterFillingN3D(B, q, min_neighbors):
    M, Nc, Nf, Nt = B.shape
    q_c, q_f, q_t = q
    F = np.empty((M,
                  Nc - q_c * 2,
                  Nf - q_f * 2,
                  Nt - q_t * 2), dtype=np.bool_)
    for i in nb.prange(M):
        F[i] = _ClusterFilling3D(B[i], q, min_neighbors)
    return F


@ReshapeArraysDecorator(dim=4, input_num=1, methodfunc=False, output_num=1, first_shape=True)
def _ClusterFilling3D_API(B, /, q, min_neighbors):
    q_c, q_f, q_t = q
    B_pad = np.pad(B, [(0, 0), (q_c, q_c), (q_f, q_f), (q_t, q_t)], mode='constant', constant_values=0)
    return _ClusterFillingN3D(B_pad.astype(bool), q, min_neighbors)


def ClusterFilling(B, /, q, min_neighbors):
    """
        Fills the zero elements if they have at least `min_neighbors` nonzero neighbors within distance `d`

        Arguments:
            B : boolean np.ndarray (..., Nf, Nt) : binary spectrograms
            q : int : neighborhood distance
            min_neighbors : int : minimum number of neighbors for zero element to be re-assigned to nonzero

        Returns:
            F : boolean np.ndarray (..., Nf, Nt) : filled binary spectrograms
    """
    func = {2: _ClusterFilling2D_API, 3: _ClusterFilling3D_API}
    dim = len(q)
    return func[dim](B, q, min_neighbors)


@nb.njit("b1[:](f8[:, :], UniTuple(i8, 2), UniTuple(i8, 2), f8)")
def _ClusteringToProjection2D(SNR, q, s, minSNR):
    B = SNR > 0
    shape = B.shape
    Nf, Nt = shape
    C = np.full(shape, -1)
    q_f, q_t = q
    s_f, s_t = s
    clusters = []
    move_clusters = []  # for combining the connecting clusters

    for (i, j), bij in np.ndenumerate(B):
        if bij:
            # selecting area of interest
            i1, i2 = max(i - q_f, 0), min(i + q_f + 1, Nf)
            j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

            b = B[i1: i2, j1: j2]
            if np.sum(b) < 2:  # isolated single points are deleted right away
                continue

            c = C[i1: i2, j1: j2]
            snr = SNR[i1: i2, j1: j2]
            cluster_k = []
            neighbor_clusters = []

            # checking existing clusters and remembering not assigned
            for l, cl in np.ndenumerate(c):
                if b[l]:
                    if (cl >= 0):
                        if (cl not in neighbor_clusters):
                            neighbor_clusters.append(cl)
                    else:
                        l1, l2 = l[0] + i1, l[1] + j1
                        cluster_k.append((l1, l2, snr[l]))

            # combining different clusters into one
            # and updating collection of clusters
            k = len(neighbor_clusters)
            if k == 0:
                cluster_assigned = len(clusters)
                clusters.append(cluster_k)
            elif (k == 1) and (len(cluster_k) == 0):
                continue
            else:
                cluster_assigned = neighbor_clusters[0]
                clusters[cluster_assigned] += cluster_k
                for cli in neighbor_clusters[1:]:
                    move_clusters.append((cluster_assigned, cli))

            # assigning clusters
            for (l1, l2, snr) in cluster_k:
                C[l1, l2] = cluster_assigned

    counted = []
    for (di, dj) in move_clusters:
        if (dj not in counted) and (di not in counted):
            counted.append(dj)
            clusters[di] += clusters[dj]
            clusters[dj] = [(Nf, Nt, 0.0)]  # meaningless isolated point, will not be used

    intervals = []
    for cl in clusters:
        if len(cl) > 1:
            cl_arr = np.array(cl)
            cl_df = cl_arr[:, 0].max() - cl_arr[:, 0].min() + 1
            t_min, t_max = cl_arr[:, 1].min(), cl_arr[:, 1].max() + 1
            cl_snr = cl_arr[:, 2].mean()
            cl_dt = t_max - t_min
            if (cl_dt >= s_t) and (cl_df >= s_f) and (cl_snr >= minSNR):
                intervals.append((t_min, t_max))

    projection = np.full(Nt, False)
    for i1, i2 in intervals:
        projection[i1 : i2 + 1] = True
    return projection


@nb.njit("b1[:, :](f8[:, :, :], UniTuple(i8, 2), UniTuple(i8, 2), f8)", parallel=True)
def _ClusteringToProjectionN2D(SNR, q, s, minSNR):
    M, Nf, Nt = SNR.shape
    projection = np.empty((M, Nt), dtype=np.bool_)
    for i in nb.prange(M):
        projection[i] = _ClusteringToProjection2D(SNR[i], q, s, minSNR)
    return projection


@ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=False, output_num=1, first_shape=True)
def ClusteringToProjection(SNR, /, q, s, minSNR):
    P = _ClusteringToProjectionN2D(SNR, q, s, minSNR)
    return P


def _optimalNeighborhoodDistance(p, pmin, qmax, maxN):
    qi = qmax  # in case the cycle is empty
    for qi in range(1, qmax + 1):
        Q = (qi * 2 + 1)**2  # clustering kernel is square
        cdf = stats.binom.cdf(maxN, Q, p)  # probability that maximum `maxN` elements in kernel `Q` are nonzero
        if cdf < pmin:  # choose `qi` which provides probability of noisy nonzero elements not to be present in `Q`
            break
    return max(1, qi - 1)


def OptimalNeighborhoodDistance(minSNR, d=2, pmin=0.95, qmax=10, maxN=1):
    """
        Estimates optimal value for neighborhood distance for the clustering.
        Estimation is based on probability `pmin` that in a given neighborhood distance
        there are no more than `maxN` noise elements. The main idea behind this estimation is that
        the sparsity of the trimmed spectrogram is directly dependent on the `minSNR` used for the trimming from DATE.
        This estimation is to reduce the number of manually adjusted non-intuitive parameters by using `minSNR`.

        Arguments:
            minSNR : float : minimum SNR used in DATE algorithm
            d : int : dimension of random numbers. Default `d=2`, i.e. spectrograms almost all the frequency bins
                      are complex numbers (2-dim real numbers)
            pmin : float : minimum confidence level that within neighborhood distance,
                           there are no more than `maxN` noise elements.
                           Default `0.95`, empirically estimated as the optimal level of confidence for various `minSNR`
            qmax : int : maximal size of the neighborhood distance.
                         Default `10`, but should not be bigger than minimum cluster size
            maxN : int : maximal number of allowed noise elements to be present within the neighborhood distance.
                         Default `1`, based on that isolated single points are automatically considered as noise
    """
    # thresholding function value from DATE, defined for `noise variance = 1`
    xi = _xi(d=d, rho=minSNR)
    # percentage of chi-distributed noise elements to be > `xi` (`d` degrees of freedom)
    p = special.gammaincc(d / 2, xi**2 / 2)
    q_opt = _optimalNeighborhoodDistance(p, pmin=pmin, qmax=qmax, maxN=maxN)
    return q_opt


