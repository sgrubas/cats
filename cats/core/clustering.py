import numpy as np
import numba as nb
from .utils import ReshapeInputArray, _scalarORarray_to_tuple
from .date import _xi
from scipy import special, stats

###################  CLUSTERING  ###################

@nb.njit("b1[:, :](b1[:, :], UniTuple(i8, 2), UniTuple(i8, 2))")
def _Clustering2D(B, q, s):
    """
        Performs clustering (density-based / neighbor-based) of binary spectrogram.

        Arguments:
            B : np.ndarray (Nf, Nt) : where `Nf` is frequency axis, `Nt` is time axis
            q : tuple(int, 2) : neighborhood distance for clustering `(q_f, q_t)` for time and frequency respectively.
            s : tuple(int, 2) : minimum cluster size (width & height) `(s_f, s_t)` for time and frequency respectively.
    """
    shape = B.shape
    Nf, Nt = shape
    C = np.full(shape, -1)
    q_f, q_t = q
    s_f, s_t = s
    clusters = []
    move_clusters = [] # for combining the connecting clusters
    
    for (i, j), bij in np.ndenumerate(B):
        if bij:
            # selecting area of interest
            i1, i2 = max(i - q_f, 0), min(i + q_f + 1, Nf)
            j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

            b = B[i1 : i2, j1 : j2]
            if np.sum(b) < 2:  # isolated single points are deleted right away
                continue
            
            c = C[i1 : i2, j1 : j2]
            cluster_k = []
            neighbor_clusters = []
                
            # checking existing clusters and remembering not assigned
            for l, cl in np.ndenumerate(c):
                if b[l]:
                    if (cl >= 0):
                        if (cl not in neighbor_clusters):
                            neighbor_clusters.append(cl)
                    else:
                        l1 = i - q_f * (i > 0) + l[0]
                        l2 = j - q_t * (j > 0) + l[1]
                        cluster_k.append((l1, l2))

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
            for ind in cluster_k:
                C[ind] = cluster_assigned
            
    F = np.full(shape, False)

    counted = []
    for (di, dj) in move_clusters:
        if (dj not in counted) and (di not in counted):
            counted.append(dj)
            clusters[di] += clusters[dj]
            clusters[dj] = [(Nf, Nt)]  # meaningless isolated point, will not be used

    F = np.full(shape, False)
    for cl in clusters:
        if len(cl) > 1:
            cl_arr = np.array(cl)
            cl_df = cl_arr[:, 0].max() - cl_arr[:, 0].min()
            cl_dt = cl_arr[:, 1].max() - cl_arr[:, 1].min()
            if (cl_dt >= s_t) and (cl_df >= s_f):
                for ind in cl:
                    F[ind] = True

    return F


@nb.njit("b1[:, :, :](b1[:, :, :], UniTuple(i8, 2), UniTuple(i8, 2))", parallel=True)
def _ClusteringN2D(B, q, s):
    """
        Performs clustering (density-based / neighbor-based) of many binary spectrograms in parallel.

        Arguments:
            B : np.ndarray (M, Nf, Nt) : `M' is number of spectrograms, `Nf` is frequency axis, `Nt` is time axis
            q : tuple(int, 2) : neighborhood distance for clustering `(q_f, q_t)` for time and frequency respectively.
            s : tuple(int, 2) : minimum cluster size (width & height) `(s_f, s_t)` for time and frequency respectively.
    """
    C = np.empty(B.shape, dtype=np.bool_)
    for i in nb.prange(B.shape[0]):
        C[i] = _Clustering2D(B[i], q, s)
    return C


@ReshapeInputArray(dim=3, num=1, methodfunc=False)
def Clustering(B, q=1, s=(5, 5)):
    """
        Performs clustering (density-based / neighbor-based) of many binary spectrograms in parallel.

        Arguments:
            B : np.ndarray (M, Nf, Nt) : `M' is number of spectrograms, `Nf` is frequency axis, `Nt` is time axis
            q : tuple(int, 2) : neighborhood distance for clustering `(q_f, q_t)` for time and frequency respectively.
            s : tuple(int, 2) : minimum cluster size (width & height) `(s_f, s_t)` for time and frequency respectively.
    """
    q = _scalarORarray_to_tuple(q, minsize=2)
    s = _scalarORarray_to_tuple(s, minsize=2)
    C = _ClusteringN2D(B.astype(bool), q, s)
    return C


@nb.njit("b1[:](b1[:, :], UniTuple(i8, 2), UniTuple(i8, 2), i8)")
def _ClusteringToProjection2D(B, q, s, max_gap):
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
            cluster_k = []
            neighbor_clusters = []

            # checking existing clusters and remembering not assigned
            for l, cl in np.ndenumerate(c):
                if b[l]:
                    if cl >= 0:
                        if cl not in neighbor_clusters:
                            neighbor_clusters.append(cl)
                    else:
                        l1 = i - q_f * (i > 0) + l[0]
                        l2 = j - q_t * (j > 0) + l[1]
                        cluster_k.append((l1, l2))

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
            for ind in cluster_k:
                C[ind] = cluster_assigned

    counted = []
    for (di, dj) in move_clusters:
        if (dj not in counted) and (di not in counted):
            counted.append(dj)
            clusters[di] += clusters[dj]
            clusters[dj] = [(Nf, Nt)]  # meaningless isolated point, will not be used

    intervals = []
    for cl in clusters:
        if len(cl) > 1:
            cl_arr = np.array(cl)
            cl_df = cl_arr[:, 0].max() - cl_arr[:, 0].min()
            t_min, t_max = cl_arr[:, 1].min(), cl_arr[:, 1].max()
            cl_dt = t_max - t_min
            if (cl_dt >= s_t) and (cl_df >= s_f):
                intervals.append((t_min, t_max))
                # if len(intervals) == 0:
                #     intervals.append((t_min, t_max))
                # else:
                #     prev_interval = intervals[-1]
                #     if (t_min - prev_interval[1]) > max_gap:
                #         intervals.append((t_min, t_max))
                #     else:
                #         intervals[-1] = (prev_interval[0], t_max)

    projection = np.full(Nt, False)
    for i1, i2 in intervals:
        projection[i1 : i2 + 1] = True
    return projection


@nb.njit("b1[:, :](b1[:, :, :], UniTuple(i8, 2), UniTuple(i8, 2), i8)", parallel=True)
def _ClusteringToProjectionN2D(B, q, s, max_gap):
    M, Nf, Nt = B.shape
    projection = np.empty((M, Nt), dtype=np.bool_)
    for i in nb.prange(M):
        projection[i] = _ClusteringToProjection2D(B[i], q, s, max_gap)
    return projection


@ReshapeInputArray(dim=3, num=1, methodfunc=False)
def ClusteringToProjection(B, q, s, max_gap):
    q = _scalarORarray_to_tuple(q, minsize=2)
    s = _scalarORarray_to_tuple(s, minsize=2)
    P = _ClusteringToProjectionN2D(B.astype(bool), q, s, max_gap)
    return P


@nb.njit("b1[:, :](b1[:, :], i8, i8)")
def _ClusterFiling2D(B, d, t):
    N, M = B.shape
    u = 2 * d
    F = np.empty((N - u, M - u), dtype=np.bool_)
    for (i, j), fij in np.ndenumerate(F):
        if B[i + d, j + d]:
            F[i, j] = True
        else:
            F[i, j] = (B[i : i + u + 1, j : j + u + 1].sum() >= t)
    return F


def _optimalNeighborhoodDistance(p, pmin, qmax, maxN):
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
