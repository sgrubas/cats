"""
    Functions for clustering trimmed spectrograms (in Time-Frequency or Space-Time-Frequency domain).
"""

import numpy as np
import numba as nb
from .utils import ReshapeArraysDecorator
from .date import _xi
from scipy import special, stats

###################  CLUSTERING  ###################

# TODO:
#  - make sure that uint16 for cluster indexes is enough


@nb.njit(["Tuple((f8[:, :], u2[:, :]))(f8[:, :], UniTuple(i8, 2), UniTuple(i8, 2), f8)",
          "Tuple((f4[:, :], u2[:, :]))(f4[:, :], UniTuple(i8, 2), UniTuple(i8, 2), f8)"], cache=True)
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
                    clusters[cli] = [(Nf, Nt, SNR[-1, -1])]  # meaningless isolated point, will not be used
                clusters[cluster_assigned] += cluster_k  # add new points

            # assigning clusters
            for (l1, l2, snr) in cluster_k:
                C[l1, l2] = cluster_assigned

    # filtering clusters and projection
    K = np.full(shape, 0, dtype=np.uint16)
    SNRK = np.zeros_like(SNR)
    k = 1
    for cl in clusters:
        if len(cl) > 1:  # all isolated points are skipped
            cl_arr = np.array(cl)
            cl_f_min, cl_f_max = cl_arr[:, 0].min(), cl_arr[:, 0].max() + 1
            cl_t_min, cl_t_max = cl_arr[:, 1].min(), cl_arr[:, 1].max() + 1
            cl_snr = cl_arr[:, 2].mean()
            if (cl_f_max - cl_f_min >= s_f) and (cl_t_max - cl_t_min >= s_t) and (cl_snr >= minSNR):
                for (l1, l2, snr) in cl:
                    K[l1, l2] = k
                    SNRK[l1, l2] = snr
                k += 1

    return SNRK, K


@nb.njit(["Tuple((f8[:, :, :], u2[:, :, :]))(f8[:, :, :], UniTuple(i8, 2), UniTuple(i8, 2), f8)",
          "Tuple((f4[:, :, :], u2[:, :, :]))(f4[:, :, :], UniTuple(i8, 2), UniTuple(i8, 2), f8)"],
         parallel=True, cache=True)
def _ClusteringN2D(SNR, q, s, minSNR):
    SNRK = np.empty_like(SNR)
    K = np.empty(SNR.shape, dtype=np.uint16)
    for i in nb.prange(SNR.shape[0]):
        SNRK[i], K[i] = _Clustering2D(SNR[i], q, s, minSNR)
    return SNRK, K


@ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=False, output_num=2, first_shape=True)
def _ClusteringN2D_API(SNR, /, q, s, minSNR):
    return _ClusteringN2D(SNR, q, s, minSNR)


@nb.njit(["Tuple((f8[:, :, :], u2[:, :, :]))(f8[:, :, :], UniTuple(i8, 3), UniTuple(i8, 3), f8)",
          "Tuple((f4[:, :, :], u2[:, :, :]))(f4[:, :, :], UniTuple(i8, 3), UniTuple(i8, 3), f8)"], cache=True)
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
                    clusters[cli] = [(Nc, Nf, Nt, SNR[-1, -1, -1])]  # meaningless isolated point for compiler, will not be used
                clusters[cluster_assigned] += cluster_k  # add new points

            # assigning clusters
            for (l1, l2, l3, snr) in cluster_k:
                C[l1, l2, l3] = cluster_assigned

    # filtering clusters and projection
    K = np.full(shape, 0, dtype=np.uint16)
    SNRK = np.zeros_like(SNR)
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
                for (l1, l2, l3, snr) in cl:
                    K[l1, l2, l3] = k
                    SNRK[l1, l2, l3] = snr
                k += 1

    return SNRK, K


@nb.njit(["Tuple((f8[:, :, :, :], u2[:, :, :, :]))(f8[:, :, :, :], UniTuple(i8, 3), UniTuple(i8, 3), f8)",
          "Tuple((f4[:, :, :, :], u2[:, :, :, :]))(f4[:, :, :, :], UniTuple(i8, 3), UniTuple(i8, 3), f8)"],
         parallel=True, cache=True)
def _ClusteringN3D(SNR, q, s, minSNR):
    SNRK = np.empty_like(SNR)
    K = np.empty(SNR.shape, dtype=np.uint16)
    for i in nb.prange(SNR.shape[0]):
        SNRK[i], K[i] = _Clustering3D(SNR[i], q, s, minSNR)
    return SNRK, K


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
    func = {2: _ClusteringN2D_API, 3: _ClusteringN3D_API}
    return func[len(q)](SNR, q, s, minSNR)


## Experimental ##


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


