import numpy as np
import numba as nb
from numba.types import b1, f8, i8, UniTuple

from .utils import ReshapeInputArray, _scalarORarray_to_tuple

################### THRESHOLDING ###################

@nb.njit(["b1[:, :](f8[:, :], f8[:, :], i8[:, :])", 
          "b1[:, :](f4[:, :], f8[:, :], i8[:, :])"], parallel=True)
def _Thresholding(PSD, Eta, frames):
    B = np.empty_like(PSD, dtype=np.bool_)
    M = len(frames); N = len(PSD)
    for j in nb.prange(M):
        j1, j2 = frames[j]
        for i in nb.prange(N):
            B[i, j1 : j2] = PSD[i, j1 : j2] > Eta[i, j]
    return B

@ReshapeInputArray(dim=2, num=2, methodfunc=False)
def Thresholding(PSD, Eta, frames):
    return _Thresholding(PSD, Eta, frames)

###################  CLUSTERING  ###################

@nb.njit("b1[:, :](b1[:, :], UniTuple(i8, 2), i8, i8)")
def _Clustering2D(B, q, dt_min, df_min):
    """
        B : np.ndarray (Nf, Nt) : where `Nf` is frequency axis, `Nt` is time axis
    """
    shape = B.shape
    Nf, Nt = shape
    C = np.full(shape, -1)
    q_f, q_t = q
    clusters = []
    moved_clusters = [] # for combining the connecting clusters
    
    for (i, j), bij in np.ndenumerate(B):
        if bij:
            # selecting area of interest
            i1, i2 = max(i - q_f, 0), min(i + q_f + 1, Nf)
            j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

            b = B[i1 : i2, j1 : j2]
            if np.sum(b) < 2: # isolated single points are deleted right away
                continue
            
            c = C[i1 : i2, j1 : j2]
            clusts = []
            clnums = []
                
            # checking existing clusters and remembering not assigned
            for l, cl in np.ndenumerate(c):
                if b[l]:
                    if (cl >= 0):
                        if (cl not in clnums):
                            clnums.append(cl)
                    else:
                        l1 = i - q_f * (i > 0) + l[0]
                        l2 = j - q_t * (j > 0) + l[1]
                        clusts.append((l1, l2))

            # combining different clusters into one
            # and udpdating collection of clusters
            k = len(clnums)
            if k == 0:
                clnum = len(clusters)
                clusters.append(clusts)
            elif (k == 1) and (len(clusts) == 0):
                continue
            else:
                clnum = clnums[0]
                clusters[clnum] += clusts
                for cli in clnums[1:]:
                    moved_clusters.append((clnum, cli))
            
            # assigning clusters    
            for l in clusts:
                C[l] = clnum
            
    F = np.full(shape, False)
    
    counted = []
    for (di, dj) in moved_clusters:
        if (dj not in counted) and (di not in counted):
            counted.append(dj)
            clusters[di] += clusters[dj]
            clusters[dj] = [(Nf, Nt)] # meaningless point, will not be used

    for cl in clusters:
        if len(cl) > 1:
            cl_arr = np.array(cl)
            cl_df = cl_arr[:, 0].max() - cl_arr[:, 0].min() 
            cl_dt = cl_arr[:, 1].max() - cl_arr[:, 1].min() 
            if (cl_dt >= dt_min) and (cl_df >= df_min):
                for ind in cl:
                    F[ind] = True

    return F

@nb.njit("b1[:, :, :](b1[:, :, :], UniTuple(i8, 2), i8, i8)", parallel=True)
def _ClusteringN2D(B, q, dt_min, df_min):
    C = np.empty(B.shape, dtype=np.bool_)
    for i in nb.prange(B.shape[0]):
        C[i] = _Clustering2D(B[i], q, dt_min, df_min)
    return C

@ReshapeInputArray(dim=3, num=1, methodfunc=False)
def Clustering(B, q=1, dt_min=10, df_min=5):
    q = _scalarORarray_to_tuple(q, minsize=2)
    C = _ClusteringN2D(B.astype(bool), q, dt_min, df_min)
    return C


# @nb.njit("f8[:, :](b1[:, :], f8[:], f8, f8)")
# def _ClusteringIntervals2D(B, time, dt_gap, dt_min):
#     """
#         B : np.ndarray (Nf, Nt) : where `Nf` is frequency axis, `Nt` is time axis
#     """
#     dt        =     time[1] - time[0]
#     dt_min    =     max(dt_min - dt_min % dt, dt)
#     q         =     max(dt_gap // dt, 1)
#     shape     =     B.shape
#     Nf, Nt    =     shape
#     C         =     np.full(shape, -1)
#     clusters        = []
#     moved_clusters  = [] # for combining the connecting clusters
    
#     for (i, j), aij in np.ndenumerate(B):
#         if aij:
#             # selecting area of interest
#             i1, i2 = max(i - q, 0), min(i + q + 1, Nf)
#             j1, j2 = max(j - q, 0), min(j + q + 1, Nt)

#             b = B[i1 : i2, j1 : j2]

#             if np.sum(b) < 2: # isolated single points are deleted right away
#                 continue
            
#             c = C[i1 : i2, j1 : j2]
#             clusts = []
#             clnums = []
                
#             # checking existing clusters and remembering not assigned
#             for l, cl in np.ndenumerate(c):
#                 if b[l]:
#                     if (cl >= 0):
#                         if (cl not in clnums):
#                             clnums.append(cl)
#                     else:
#                         l1 = i - q * (i > 0) + l[0]
#                         l2 = j - q * (j > 0) + l[1]
#                         clusts.append((l1, l2))

#             # combining different clusters into one
#             # and udpdating collection of clusters
#             if len(clnums) == 0:
#                 clnum = len(clusters)
#                 clusters.append(clusts)
#             elif (len(clnums) == 1) and (not clusts):
#                 continue
#             else:
#                 clnum = clnums[0]
#                 for cli in clnums[1:]:
#                     moved_clusters.append((clnum, cli))
#                 clusters[clnum] += clusts
            
#             # assigning clusters    
#             for l in clusts:
#                 C[l] = clnum
    
#     counted = []
#     for (di, dj) in moved_clusters:
#         if (dj not in counted) and (di not in counted):
#             counted.append(dj)
#             clusters[di] += clusters[dj]
#             clusters[dj] = [(Nf, Nt)] # meaningless point, will not be used

#     intervals = np.empty((len(clusters), 2), dtype=np.int64)
#     k = 0
#     for cl in clusters:
#         cl_t_arr = np.array([inds[1] for inds in cl])
#         t1, t2 = time[cl_t_arr.max()], time[cl_t_arr.min()]
#         cl_dt = t1 - t2
#         if cl_dt >= dt_min:
#             intervals[k, 0] = t1
#             intervals[k, 1] = t2
#             k += 1
#     return intervals

# @nb.njit("List(i8[:, :])(b1[:, :, :], f8, f8)", parallel=True)
# def _ClusteringIntervalsN2D(B, dt_gap, dt_min):
#     n = B.shape[0]
#     intervals = []
#     for i in nb.prange(n):
#         C[i] = _ClusteringIntervals2D(B[i], dt_gap, dt_min)
#     return C

# @ReshapeInputArray(dim=3, num=1, methodfunc=False)
# def Clustering(B, q=1, dt_min=25):
#     q = _scalarORarray_to_tuple(q, minsize=2)
#     C = _ClusteringN2D(B.astype(bool), q, dt_min)
#     return C