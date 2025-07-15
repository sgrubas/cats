"""
    Functions for clustering trimmed spectrograms (in Time-Frequency or Space-Time-Frequency domain).
"""

import numpy as np
import numba as nb
from .utils import ReshapeArraysDecorator, replace_int_by_list, replace_int_by_slice, give_trace_dim_names
from scipy import stats
import pandas as pd
from collections import ChainMap, defaultdict
from typing import List, Dict, Callable, Union, Tuple

# ------------------- CLUSTERING ------------------- #


def Clustering(T_mask, /, q, s, freq_octaves):
    """
        Performs clustering [Note 1] of trimmed spectrograms via Connected-Component labeling [Ch 7.7 P 152, 1; 2].
        Algorithm based on method "one component at a time" via graph traversal using Depth-First Search [3, 4, 5][Note 2].
        Works analogous to `scipy.ndimage.label`, but also:
            1. Supports arbitrary neighbourhood pixel distances `q` > 1 (see below for `q`).
            2. Supports log2 neighbourhood pixel distances for frequency (see below for `freq_octaves`).
            3. Filters out labeled clusters smaller than `s` (see below for `s`).
        This function is a wrapper for `_Clustering2D` and `_Clustering3D` functions.

        Add-on: TopoClustering [Note 3] is also available, which is an adaptation of ToMATo algorithm [7].
                Used if arguments `prominence_thr` is not None (`merge_small` = {True, False}, False is default)

        Arguments:
            T_mask : np.ndarray (M, Nf, Nt) or (M, Nc, Nf, Nt) :
                  Trimmed binary mask where nonzero elements will be labelled.
                  `M` is number of spectrograms, `Nc` is number of traces, `Nf` is frequency axis, `Nt` is time axis.

            q : tuple[int, int] / tuple[int, int, int] : neighborhood distance for clustering
                `(q_f, q_t)` or `(q_c, q_f, q_t)`, where q_c` for traces, `q_f` for frequency, `q_t` for time.
                If `len(q) = 2` then 2D clustering in "Frequency x Time" is used, `SNR.shape = (M, Nf, Nt)`
                If `len(q) = 3` then 3D clustering in "Trace x Frequency x Time" is used, `SNR.shape = (M, Nc, Nf, Nt)`

            s : tuple[int, int] / tuple[int, int, int] : minimum cluster sizes `(s_f, s_t)` or `(s_c, s_f, s_t)`.
                `s_c` for traces, `s_f` for frequency, `s_t` for time.

            freq_octaves : tuple[float, float] : equivalent to `s_f` and `q_f` (see above), but in log2 scale (octaves),
                           for frequency only, in the order: (cluster_size_f_octaves, cluster_distance_f_octaves).
                           By default, zeros and not used, but if non-zero, they replace `s_f` and `q_f` respectively.

        Returns:
            C : np.ndarray (M, Nf, Nt) or (M, Nc, Nf, Nt) : Array of clustered/labeled pixels.

        Notes:
            1. We call this "clustering" because:
                1.1. It labels clusters of energy outliers in time-frequency domain. Energy criterion was used on the
                     previous stage where spectrogram has been trimmed and classified into "outliers" and "noise".
                1.2. This clustering/labeling algorithm is very similar to DBSCAN [6] but works on image pixels
                     like Region Growing [Ch 7.5 P 148, 1] or Connected-Component labeling [Ch 7.7 P 152, 1; 2],
                     and in time-frequency domain.
                1.3. Filters out clusters that small in size in time-frequency domain.
            2. I tried to implement graph traversal using "Depth-First Search" (DFS) algorithm [3, 4], however,
               my naive implementation may actually be "Stack-based graph traversal" [3, 5],
               which is commonly confused with proper DFS.
            3. TopoClustering is an adaptation of ToMATo algorithm [1] with additional criterion on size of clusters.

        References:
            1. Acharya, T., & Ray, A. K. (2005). Image processing: principles and applications. John Wiley & Sons.
            2. https://en.wikipedia.org/wiki/Connected-component_labeling
            3. Kleinberg, J., & Tardos, E. (2006). Algorithm design. Pearson Education India.
            4. https://en.wikipedia.org/wiki/Depth-first_search
            5. https://11011110.github.io/blog/2013/12/17/stack-based-graph-traversal.html
            6. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996, August). A density-based algorithm for discovering
               clusters in large spatial databases with noise. In kdd (Vol. 96, No. 34, pp. 226-231).
            7. Chazal, F., Guibas, L. J., Oudot, S. Y., & Skraba, P. (2013).
               Persistence-based clustering in Riemannian manifolds. Journal of the ACM (JACM), 60(6), 1-38.
    """

    dim = len(q)
    assert dim == len(s)

    if dim == 2:  # 2D clustering (single-trace)
        res = _ClusteringN2D(T_mask, q, s, freq_octaves)
    else:  # 3D clustering (multi-trace)
        res = _ClusteringN3D(T_mask, q, s, freq_octaves)

    return res


IND_ND = lambda n: f'UniTuple(i8, {n})'
CLUSTER_OUT_SIGNATURE_ND = lambda n: f"Tuple(({IND_ND(n)}, {IND_ND(n)}))"


@nb.njit([f"{CLUSTER_OUT_SIGNATURE_ND(2)}({IND_ND(2)}, {IND_ND(2)}, b1[:, :], i4[:, :], i4, f8)",
          f"{CLUSTER_OUT_SIGNATURE_ND(2)}({IND_ND(2)}, {IND_ND(2)}, f4[:, :], i4[:, :], i4, f8)",
          f"{CLUSTER_OUT_SIGNATURE_ND(2)}({IND_ND(2)}, {IND_ND(2)}, f8[:, :], i4[:, :], i4, f8)"],
         cache=True)
def depth_first_search_2D(ind, q, T_mask, C, cid, freq_distance_octaves):
    q_f, q_t = q
    Nf, Nt = T_mask.shape

    mins = ind
    maxs = ind

    stack = [ind]

    while stack:
        # Depth-First Search (DFS, First-In / Last-Out --> stack.pop(-1))
        # DFS seems to scale better with `q` than Breadth-First (BFS, First-In / First-Out --> stack.pop(0))
        curr = stack.pop(-1)  # take element to label and scan neighbors

        if C[curr] == cid:  # check if it was labelled before
            continue

        C[curr] = cid  # label element

        mins = (min(mins[0], curr[0]), min(mins[1], curr[1]))  # update min indexes
        maxs = (max(maxs[0], curr[0]), max(maxs[1], curr[1]))  # update max indexes

        # define lookout window to search for neighbors
        i, j = curr
        if freq_distance_octaves > 0.0:  # log2 step for frequency
            i1 = min(i - 1, round(i * 2 ** (-freq_distance_octaves)))  # octave is power of 2
            i2 = max(i + 1, round(i * 2 ** freq_distance_octaves)) + 1  # octave is power of 2
        else:  # normal linear step
            i1, i2 = i - q_f, i + q_f + 1

        i1, i2 = max(i1, 0), min(i2, Nf)
        j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

        # search for neighbors to scan next
        for li in range(i1, i2):
            for lj in range(j1, j2):
                lind = (li, lj)
                nonzero = T_mask[lind]  # mask value must be True
                non_visited = (C[lind] == 0)  # only non-visited elements are added to stack
                if nonzero and non_visited:
                    stack.append(lind)  # add to stack
                    C[lind] = -cid  # negative label means it's been visited

    return mins, maxs


@nb.njit([f"i4[:, :](b1[:, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))",
          f"i4[:, :](f4[:, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))",
          f"i4[:, :](f8[:, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))",], cache=True)
def _Clustering2D(T_mask, q, s, freq_octaves):
    shape = T_mask.shape
    freq_width_octaves, freq_distance_octaves = freq_octaves

    # Finding and labelling clusters
    C = np.zeros(shape, dtype=np.int32)
    cluster_size = {}  # remembers cluster sizes
    cid = 1
    for i, j in np.argwhere(T_mask):  # iteratively scan each pixel
        ind = (i, j)
        if C[ind] < 1:  # if pixel does not have label, do DFS
            cluster_size[cid] = depth_first_search_2D(ind, q, T_mask, C, cid, freq_distance_octaves)
            cid += 1

    # Filtering clusters by sizes
    s_f, s_t = s
    k = 1
    cluster_newid = {}  # memorizes old cluster IDs to relabel them
    for cid, (mins, maxs) in cluster_size.items():
        if freq_width_octaves > 0.0:
            min_frequency_width = np.log2((maxs[0] + 1) / (mins[0] + 1e-8)) >= freq_width_octaves  # log2 for octaves
        else:
            min_frequency_width = (maxs[0] - mins[0] + 1 >= s_f)
        cluster_counted = (min_frequency_width and  # min frequency width
                           (maxs[1] - mins[1] + 1 >= s_t))  # min time duration
        if cluster_counted:
            cluster_newid[cid] = k  # old ID ---> new ID
            k += 1

    # Re-labeling clusters: IMPORTANT as ClusterCatalogs rely on the order (1, ..., N)
    for i, j in np.argwhere(C):
        ind = (i, j)
        C[ind] = cluster_newid.get(C[ind], np.int32(0))

    return C


@ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=False, output_num=2, first_shape=True)
@nb.njit([f"i4[:, :, :](b1[:, :, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))",
          f"i4[:, :, :](f4[:, :, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))",
          f"i4[:, :, :](f8[:, :, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))"],
         parallel=True, cache=True)
def _ClusteringN2D(T_mask, q, s, freq_octaves):
    C = np.empty(T_mask.shape, dtype=np.int32)

    for i in nb.prange(T_mask.shape[0]):
        C[i] = _Clustering2D(T_mask[i], q, s, freq_octaves)
    return C


@nb.njit(f"{CLUSTER_OUT_SIGNATURE_ND(3)}({IND_ND(3)}, {IND_ND(3)}, b1[:, :, :], i4[:, :, :], i4, f8)", cache=True)
def depth_first_search_3D(ind, q, T_mask, C, cid, freq_distance_octaves):
    """ See detailed comments in `depth_first_search_2D` """

    q_c, q_f, q_t = q
    Nc, Nf, Nt = T_mask.shape

    mins = ind
    maxs = ind
    unique_c = []  # to count num of unique traces

    stack = [ind]

    while stack:
        curr = stack.pop(-1)

        if C[curr] == cid:
            continue

        C[curr] = cid

        if curr[0] not in unique_c:
            unique_c.append(curr[0])
        mins = (0, min(mins[1], curr[1]), min(mins[2], curr[2]))  # (min number of traces always 0!, ...)
        maxs = (len(unique_c), max(maxs[1], curr[1]), max(maxs[2], curr[2]))  # (max number of unique traces!, ...)

        k, i, j = curr
        if freq_distance_octaves > 0.0:
            i1 = min(i - 1, round(i * 2 ** (-freq_distance_octaves)))  # octaves is power of 2
            i2 = max(i + 1, round(i * 2 ** freq_distance_octaves)) + 1  # octaves is power of 2
        else:
            i1, i2 = i - q_f, i + q_f + 1

        k1, k2 = max(k - q_c, 0), min(k + q_c + 1, Nc)
        i1, i2 = max(i1, 0), min(i2, Nf)
        j1, j2 = max(j - q_t, 0), min(j + q_t + 1, Nt)

        for lk in range(k1, k2):
            for li in range(i1, i2):
                for lj in range(j1, j2):
                    lind = (lk, li, lj)
                    nonzero = T_mask[lind]  # mask must be True
                    non_visited = (C[lind] == 0)
                    if nonzero and non_visited:
                        stack.append(lind)
                        C[lind] = -cid

    return mins, maxs


@nb.njit(f"i4[:, :, :](b1[:, :, :], {IND_ND(3)}, {IND_ND(3)}, UniTuple(f8, 2))", cache=True)
def _Clustering3D(T_mask, q, s, freq_octaves):
    shape = T_mask.shape
    freq_width_octaves, freq_distance_octaves = freq_octaves

    # finding clusters
    C = np.zeros(shape, dtype=np.int32)
    cluster_size = {}
    cid = 1
    for k, i, j in np.argwhere(T_mask):
        ind = (k, i, j)
        if C[ind] < 1:
            cluster_size[cid] = depth_first_search_3D(ind, q, T_mask, C, cid, freq_distance_octaves)
            cid += 1

    # filtering clusters
    s_c, s_f, s_t = s
    k = 1
    cluster_newid = {}
    for cid, (mins, maxs) in cluster_size.items():
        c1, f1, t1 = mins
        c2, f2, t2 = maxs

        if freq_width_octaves > 0.0:
            min_frequency_width = np.log2((f2 + 1) / (f1 + 1e-8)) >= freq_width_octaves  # log2 for octaves
        else:
            min_frequency_width = (f2 - f1 + 1 >= s_f)

        cluster_counted = ((c2 - c1 >= s_c) and  # min number of traces (no `+ 1` as it is number of unique traces)
                           min_frequency_width and  # min frequency width
                           (t2 - t1 + 1 >= s_t))  # min time duration
        if cluster_counted:
            cluster_newid[cid] = k
            k += 1

    # clusters = {new_cid: clusters[old_cid] for old_cid, new_cid in passed_clusters_cid.items()}

    # Re-labeling clusters
    for k, i, j in np.argwhere(C):
        ind = (k, i, j)
        C[ind] = cluster_newid.get(C[ind], 0)

    # SNRK = np.zeros_like(SNR)
    # for k, i, j in np.argwhere(C):
    #     ind = (k, i, j)
    #     C[ind] = newid = cluster_newid.get(C[ind], 0)
    #     if newid > 0:
    #         SNRK[ind] = SNR[ind]

    return C


@ReshapeArraysDecorator(dim=4, input_num=1, methodfunc=False, output_num=1, first_shape=True)
@nb.njit(f"i4[:, :, :, :](b1[:, :, :, :], {IND_ND(3)}, {IND_ND(3)}, UniTuple(f8, 2))", parallel=True, cache=True)
def _ClusteringN3D(T_mask, q, s, freq_octaves):
    C = np.empty(T_mask.shape, dtype=np.int32)
    for i in nb.prange(T_mask.shape[0]):
        C[i] = _Clustering3D(T_mask[i], q, s, freq_octaves)
    return C


# ---------------- Cluster catalogs ---------------- #

# TODO:
#  - parallelize 'calculate_cluster_features' with 'numba'? (seems not feasible due to 'funcs' and 'dicts' types)
#  - implement ClusterCatalogs for 3D, when `multitrace=True`? (is it useful?)

def calculate_cluster_features(values_dict: Dict[str, np.ndarray],
                               labels: np.ndarray,
                               funcs: List[Callable],
                               freq: np.ndarray,
                               time: np.ndarray,
                               id_name: str) -> Dict[str, List[float]]:

    index = range(1, labels.max() + 1)
    catalog = defaultdict(list)

    for cid in index:
        # Find elements for cluster index == 'cid'

        # Get time and freq indexes
        freq_inds, time_inds = sep_inds = np.nonzero(labels == cid)  # always 2D !
        nonzero_num = len(freq_inds)

        if nonzero_num > 0:  # skip non-existing indexes
            f = freq[freq_inds]  # exact time values
            t = time[time_inds]  # exact freq values

            # Slice feature distributions
            vals_slices_dict = {}
            for name, val in values_dict.items():
                val_shape = val.shape[:-2]  # the last 2 dims are (freq, time)

                if len(val_shape) > 0:  # handles arrays in case of aggregated clustering
                    val_new = np.zeros(val_shape + (len(freq_inds),), dtype=val.dtype)
                    for val_ind in np.ndindex(*val_shape):
                        val_new[val_ind] = val[val_ind][sep_inds]
                else:
                    val_new = val[sep_inds]

                vals_slices_dict[name] = val_new

            # Iterate over feature funcs on feature distributions
            res_dicts = [func(f, t, vals_slices_dict, sep_inds) for func in funcs]

            # Compose single long dict (must be unique keys / feature names, else it picks the first occured)
            func_res_dict = ChainMap(*res_dicts)

            # Make 'dict of lists of vals' from 'dict of vals'
            catalog[id_name].append(cid)
            for name, vals in func_res_dict.items():
                catalog[name].append(vals)

    return catalog


def ClusterCatalogs(values_dict: Dict[str, np.ndarray],
                    CID: np.ndarray,
                    freq: np.ndarray,
                    time: np.ndarray,
                    feature_funcs: List[Callable] = None,
                    aggr_clustering_axis: Union[int, Tuple[int], None] = None,
                    frequency_unit: str = 'Hz',
                    time_unit: str = 'sec',
                    trace_dim_names: List[str] = None,
                    id_name: str = 'Cluster_ID') -> pd.DataFrame:
    """
        Calculates statistics/features of cluster in each trace

        Arguments:
             values_dict : dict[str, np.ndarray (..., Nf, Nt)] : array of values (PSD, SNR, ...)
             CID : np.ndarray (..., Nf, Nt) : array cluster ID values
             freq : np.ndarray (Nf,) : array of frequencies
             time : np.ndarray (Nt,) : array of time samples
             feature_funcs : list[callable] : Funcs of signature
                         `func(freq, time, values_dict, inds) -> dict[str, float]`,
                         where `freq`, `time`, `values` are 1D arrays, `inds = [freq_inds, time_inds]`.
                         Must return dict with the calculated features.
             aggr_clustering_axis : int | tuple[int] : axis for aggregated clustering
             frequency_unit : str : unit name for adding suffix to column names in Catalog DataFrame
             time_unit : str : unit name for adding suffix to column names in Catalog DataFrame
             trace_dim_names : list[str] : names of trace dimensions for DataFrame MultiIndex
             id_name : str : name of the column with cluster IDs (default: 'Cluster_ID')

        Returns:
            Catalogs: pd.DataFrame : dataframe with calculated features for each trace and cluster
    """

    if feature_funcs is None:
        feature_funcs = [bbox_peaks]
    assert isinstance(feature_funcs, (list, tuple)), "`feature_funcs` must be list or tuple"

    assert all([val.shape[-2:] == CID.shape[-2:] for val in values_dict.values()]), \
        "Key time-freq dimensions (last two) must coincide with CID array for all 'values_dict'"

    trace_shape = CID.shape[:-2]
    ndim = len(trace_shape)

    trace_dim_names = give_trace_dim_names(ndim, aggr_clustering_axis, trace_dim_names)

    # 1. get features data in dicts; 2. then DataFrame. This order is much more efficient (~2-3x)
    catalogs = defaultdict(list)  # list type is important for appending/extending lists
    for ind in np.ndindex(*trace_shape):  # iter over traces
        val_ind = replace_int_by_slice(ind, aggr_clustering_axis)  # for slicing values_dict (needed to handle aggr.)

        # calculate features
        values_dict_ind = {name: val[val_ind] for name, val in values_dict.items()}
        catalog = calculate_cluster_features(values_dict=values_dict_ind,
                                             labels=CID[ind],
                                             funcs=feature_funcs,
                                             freq=freq,
                                             time=time,
                                             id_name=id_name)
        for name, data in catalog.items():
            catalogs[name].extend(data)

        if (cids := catalog.get(id_name, None)) is not None:  # if non-empty, at least one cluster
            for name, tr_i in zip(trace_dim_names, ind):  # iter over trace dims
                catalogs[name].extend([tr_i] * len(cids))

    # create DataFrame
    catalogs = pd.DataFrame(catalogs)  # Unified DataFrame for all traces

    if len(catalogs) > 0:  # otherwise throws errors
        catalogs.set_index(trace_dim_names, inplace=True)  # Create (Multi)Index for proper indexing over traces

        # append unit suffixes to appropriate feature names
        apply_units(catalogs, frequency_unit=frequency_unit, time_unit=time_unit)

        # sort columns order
        sorted_cols = sort_feature_names(catalogs.columns)
        catalogs = catalogs[sorted_cols]

    return catalogs


def apply_units(catalog, frequency_unit='Hz', time_unit='sec'):
    cols = list(catalog.columns)
    unit_names = {'frequency': frequency_unit, 'time': time_unit}
    combine = '_'.join
    renames = []
    for name, unit in unit_names.items():
        only_one_name = lambda col: sum(nm in col.casefold() for nm in unit_names.keys()) < 2
        condition_func = lambda col: (name in col.casefold()) and only_one_name(col) and not (unit in col.casefold())
        if name:
            renames.append({ci: combine([ci, unit]) for ci in cols if condition_func(ci)})
    renames = ChainMap(*renames)

    catalog.rename(columns=renames, inplace=True)


def sort_feature_names(cols):
    sorted_cols = ["Event_ID", "Cluster_ID"]
    sorted_cols += [ci for ci in cols if "time" in ci.casefold() and ci not in sorted_cols]
    sorted_cols += [ci for ci in cols if "frequency" in ci.casefold() and ci not in sorted_cols]
    # sorted_cols += [ci for ci in ["First_arrival", "Strong_arrival"] if ci in cols]
    # sorted_cols += [ci for ci in cols if "energy" in ci.casefold() and ci not in sorted_cols]
    # sorted_cols += [ci for ci in cols if "ellipse" in ci.casefold() and ci not in sorted_cols]
    sorted_cols += [ci for ci in cols if ci not in sorted_cols]
    return sorted_cols


def index_cluster_catalog(catalog, ind):
    if isinstance(catalog.index, pd.MultiIndex):
        catalog_ind = replace_int_by_list(ind)  # to ensure that slicing always returns DataFrame (not Series)
    else:
        catalog_ind = ind if isinstance(ind, int) else ind[0]
        catalog_ind = replace_int_by_list(catalog_ind)

    try:
        catalog_slice = catalog.loc[catalog_ind, :]  # last slice for columns
    except KeyError:  # if such trace index is absent --> empty data frame
        catalog_slice = pd.DataFrame(columns=catalog.columns,
                                     index=pd.MultiIndex.from_tuples([],
                                                                     names=catalog.index.names))
    return catalog_slice


def assign_by_index_cluster_catalog(catalog, ind, columns, values):
    if isinstance(catalog.index, pd.MultiIndex):
        catalog_ind = replace_int_by_list(ind)  # to ensure that slicing always returns DataFrame (not Series)
    else:
        catalog_ind = ind if isinstance(ind, int) else ind[0]
        catalog_ind = replace_int_by_list(catalog_ind)

    try:
        catalog.loc[catalog_ind, columns] = values
    except KeyError:  # if such trace index is absent --> empty data frame
        pass


def concatenate_cluster_catalogs(catalog1: pd.DataFrame,
                                 catalog2: pd.DataFrame,
                                 id_cols: List[str],
                                 t0: float):
    time_shift_cols = ["Time_start_sec", "Time_end_sec", "Time_peak_sec", "Time_centroid_sec"]
    id_cols = id_cols or ["Cluster_ID"]

    if catalog2 is not None:
        time_shift_cols = [col_i for col_i in time_shift_cols if col_i in catalog2.columns]
        if len(catalog2) > 0:
            # time shift
            catalog2[time_shift_cols] += t0

            if catalog1 is not None:
                # cluster ID shift
                for id_col in id_cols:
                    id_col2 = f"{id_col}_2"
                    # assign new col to handle empty traces with no clusters (NaNs)
                    catalog2[id_col2] = catalog1[id_col].groupby(catalog1.index.names).max()
                    # empty traces with NaN are replaced with 0, and added to original Cluster_IDs
                    catalog2[id_col] += catalog2.pop(id_col2).fillna(0).astype(int)

        catalog_new = pd.concat([catalog1, catalog2], copy=False)
        # catalog_new.sort_values(by=catalog_new.index.names, inplace=True)
        catalog_new.sort_index(inplace=True)
        return catalog_new

    elif (catalog1 is not None) and (catalog2 is None):
        return catalog1

    else:
        return None


def update_cluster_ID_by_catalog(cluster_ID, catalog, keep_catalog_rows, id_name=None):
    id_name = id_name or "Cluster_ID"
    # the fastest so far
    remove_inds = ~keep_catalog_rows
    bad_catalog = catalog[remove_inds]
    for row in bad_catalog.itertuples():  # over traces with bad clusters
        ind, cid = row.Index, getattr(row, id_name)
        CID = cluster_ID[ind]
        CID[CID == cid] = 0  # remove bad cluster ID


# ---------------- Feature/Attribute functions ---------------- #

def aggregate_feature_values(values, aggr_func):
    """ Convenience function """
    aggr_axes = tuple(range(values.ndim - 1))
    return aggr_func(values, axis=aggr_axes) if values.ndim > 1 else values


def bbox_peaks(freq, time, values_dict, inds, aggr_func=np.linalg.norm,
               distribution='spectrogram_SNR'):
    energy = aggregate_feature_values(values_dict[distribution], aggr_func)

    f_inds, t_inds = inds

    # Time interval
    t_min_id, t_max_id = np.argmin(time), np.argmax(time)
    t_min, t_max = time[t_min_id], time[t_max_id]
    t_min -= t_min / (t_inds[t_min_id] + 1) * 0.5  # half-pixel extension (left)
    t_max += t_max / (t_inds[t_max_id] + 1) * 0.5  # half-pixel extension (right)

    # Frequency bandwidth
    f_min_id, f_max_id = np.argmin(freq), np.argmax(freq)
    f_min, f_max = freq[f_min_id], freq[f_max_id]
    # f_min -= f_min / (f_inds[f_min_id] + 1) * 0.5  # half-pixel extension (left), work BAD for CWT logscale
    # f_max += f_max / (f_inds[f_max_id] + 1) * 0.5  # half-pixel extension (right), work BAD for CWT logscale

    # Peak location
    peak_id = np.argmax(energy)
    t_peak, f_peak = time[peak_id], freq[peak_id]
    v_peak = energy[peak_id]
    v_sum = energy.sum()
    v_mean = v_sum / len(energy)

    output = {"Time_start": t_min,
              "Time_end": t_max,
              "Frequency_start": f_min,
              "Frequency_end": f_max,
              "Time_peak": t_peak,
              "Frequency_peak": f_peak,
              "Energy_peak": v_peak,
              "Energy_mean": v_mean,
              "Energy_sum": v_sum,
              }

    snr = values_dict.get('spectrogram_SNR', None)
    if snr is not None:
        snr = aggregate_feature_values(snr, aggr_func)
        output["SNR_sum"] = np.sum(snr)
        output["SNR_mean"] = output["SNR_sum"] / len(snr)
        output["SNR_peak"] = np.max(snr)
    return output


def peak_freq_arrival(freq, time, values_dict, inds, aggr_func=np.max, mode='first_valley'):

    assert mode in ['first_valley', 'cluster_start'], "Mode must be 'first_valley' or 'cluster_start'"

    energy = aggregate_feature_values(values_dict['spectrogram'], aggr_func)

    freq_inds, time_inds = inds

    # Mode / Peak to identify the strongest event phase
    peak_id = np.argmax(energy)
    f_peak_id = freq_inds[peak_id]
    f_peak_inds = (freq_inds == f_peak_id)

    if mode == 'first_valley':
        t_peak_id = time_inds[peak_id]
        tf_peak_inds = f_peak_inds & (time_inds <= t_peak_id)  # only the left part (before peak)
        t_freq_slice = time[tf_peak_inds]
        e_freq_slice = energy[tf_peak_inds]
        sorting = np.argsort(-t_freq_slice)  # decreasing order (from right to left)
        ind_1 = 0  # default in case only 1 pixel in `t_freq_slice`
        for i in range(1, len(sorting)):
            ind_1 = sorting[i - 1]
            ind_2 = sorting[i]
            if e_freq_slice[ind_2] >= e_freq_slice[ind_1]:  # if energy started to increase
                break
            else:
                ind_1 = ind_2  # in case it reached the end
        arrival = t_freq_slice[ind_1]

    elif mode == 'cluster_start':
        arrival = time[f_peak_inds].min()
    else:
        raise ValueError("Mode must be 'first_valley' or 'cluster_start'")

    return {"Peak_freq_arrival": arrival}


def energy_statistics(freq, time, values_dict, inds, aggr_func=np.max):
    """
        References: (Section 6) http://dx.doi.org/10.1016/j.dsp.2017.07.015
    """
    energy = values_dict['spectrogram']
    aggr_axes = tuple(range(energy.ndim - 1))
    energy = aggr_func(energy, axis=aggr_axes) if energy.ndim > 1 else energy
    N = energy.size
    N = N if N > 1 else 2

    # Energy stats
    vmin = energy.min()
    vmax = energy.max()
    amean = energy.mean()
    std = np.sqrt(np.sum((energy - amean) ** 2) / (N - 1))
    skew = stats.skew(energy, bias=False)
    kurtosis = stats.kurtosis(energy, fisher=True, bias=False)
    gmean = stats.gmean(energy)
    flatness = gmean / amean

    entropy = stats.entropy(energy)  # Shannon entropy

    output = {"Energy_min": vmin,
              "Energy_max": vmax,
              "Energy_mean": amean,
              "Energy_std": std,
              "Energy_skew": skew,
              "Energy_kurtosis": kurtosis,
              "Energy_flatness": flatness,
              "Energy_entropy": entropy}

    return output


def freq_of_time_polynomial(freq, time, values_dict, inds, order, aggr_func=np.max,
                            freq_octaves=False, f_octaves_zero=1e-3):
    energy = values_dict['spectrogram']
    aggr_axes = tuple(range(energy.ndim - 1))
    energy = aggr_func(energy, axis=aggr_axes) if energy.ndim > 1 else energy

    freq = np.log2(np.where(freq == 0.0, f_octaves_zero, freq)) if freq_octaves else freq

    try:
        if len(energy) <= order:
            raise ValueError
        poly = np.polynomial.Polynomial.fit(time - time.min(), freq, deg=order, rcond=np.nan, w=energy)
        coefs = poly.coef
    except:
        coefs = [np.nan] * (order + 1)
    naming = lambda x: f"f(t^{x})"
    coefs = {naming(i): ci for i, ci in enumerate(coefs)}
    return coefs


@nb.njit(["f8[:, :](f8[:], f8[:], f8[:], UniTuple(i4[:], 2), i8, boolean)",
          "f8[:, :](f8[:], f8[:], f4[:], UniTuple(i4[:], 2), i8, boolean)",
          "f8[:, :](f8[:], f8[:], f8[:], UniTuple(i8[:], 2), i8, boolean)",
          "f8[:, :](f8[:], f8[:], f4[:], UniTuple(i8[:], 2), i8, boolean)"],
         cache=True, parallel=True)
def central_moments(freq, time, values, inds, order, interpretable_only):
    """
       Pseudo-normalized pseudo-central geometrical moments up to order=`order`.
       Pseudo-normalized means that all moments are normalized by m00 (zeroth-order moment) except m00 (m00 != 1).
       Pseudo-central means that all moments are centered by m01 and m10 except themselves (m01 != m10 != 0).
       These Pseudo invariants carry energy and centroid information which is useful for our application.
    """
    N = len(values)
    order += 1  # to include zeroth order
    M = np.zeros((order, order))

    # Zeroth-order moment
    M[0, 0] = np.sum(values)
    values /= M[0, 0]  # normalize
    # First-order moments
    M[1, 0] = np.sum(values * freq)  # centroid f
    M[0, 1] = np.sum(values * time)  # centroid t
    f_c = (freq - M[1, 0])  # centered f
    t_c = (time - M[0, 1])  # centered t
    M[1, 1] = np.sum(values * f_c * t_c)  # covariance

    # compute all powers in advance for efficiency
    f_c_powers = np.ones((order, N), dtype=f_c.dtype)
    t_c_powers = np.ones((order, N), dtype=t_c.dtype)
    for power in range(1, order):
        f_c_powers[power] = f_c_powers[power - 1] * f_c
        t_c_powers[power] = t_c_powers[power - 1] * t_c

    interpretable_moments = [(0, 0), (1, 1),
                             (1, 0), (0, 1),
                             (2, 0), (0, 2),
                             (3, 0), (0, 3),
                             (4, 0), (0, 4)]

    # main loop for moments
    for i_f in nb.prange(order):
        for j_t in nb.prange(order):
            if (i_f > 1) or (j_t > 1):  # skip up to first-order mixed moments as they've been calculated
                allow_calc = ((i_f, j_t) in interpretable_moments) if interpretable_only else True

                if allow_calc:
                    M[i_f, j_t] = np.sum(values * f_c_powers[i_f] * t_c_powers[j_t])

    return M


def interpret_central_moments(catalog, interpretable_only):

    interpretable = ['m00', 'm11', 'm01', 'm10', 'm02', 'm20', 'm03', 'm30', 'm04', 'm40']
    if interpretable_only:
        for kw in list(catalog.keys()):
            if kw not in interpretable:
                catalog.pop(kw)

    if (m00 := catalog.get('m00', None)) is not None:
        catalog['Energy_sum'] = m00
    if (m10 := catalog.get('m10', None)) is not None:
        catalog['Frequency_centroid'] = m10
    if (m01 := catalog.get('m01', None)) is not None:
        catalog['Time_centroid'] = m01
    if (m11 := catalog.get('m11', None)) is not None:
        catalog['Time_Frequency_covar'] = m11
    if (m20 := catalog.get('m20', None)) is not None:
        catalog['Frequency_std'] = np.sqrt(m20)
    if (m02 := catalog.get('m02', None)) is not None:
        catalog['Time_std'] = np.sqrt(m02)
    if (((m30 := catalog.get('m30', None)) is not None) and
       ((f_std := catalog.get('Frequency_std', None)) is not None)):
        catalog['Frequency_skew'] = m30 / f_std ** 3
    if (((m03 := catalog.get('m03', None)) is not None) and
       ((t_std := catalog.get('Time_std', None)) is not None)):
        catalog['Time_skew'] = m03 / t_std ** 3
    if (((m40 := catalog.get('m40', None)) is not None) and
       ((m20 := catalog.get('m20', None)) is not None)):
        catalog['Frequency_kurtosis'] = m40 / m20 ** 2
    if (((m04 := catalog.get('m04', None)) is not None) and
       ((m02 := catalog.get('m02', None)) is not None)):
        catalog['Time_kurtosis'] = m04 / m02 ** 2

    if (((m11 := catalog.get('m11', None)) is not None) and
       ((m20 := catalog.get('m20', None)) is not None) and
       ((m02 := catalog.get('m02', None)) is not None)):
        angle = 0.5 * np.arctan2(2 * m11, m20 - m02)
        catalog['Ellipse_angle'] = angle

        root = np.sqrt((catalog['m20'] - catalog['m02']) ** 2 + 4 * catalog['m11'] ** 2)
        eigvals = [0.5 * (catalog['m20'] + catalog['m02'] + root),
                   0.5 * (catalog['m20'] + catalog['m02'] - root)]
        ecc = np.sqrt(1 - eigvals[1] ** 2 / eigvals[0] ** 2)  # eigvals[0] > eigvals[1]
        catalog['Ellipse_eccentricity'] = ecc


def calculate_moments(freq, time, values_dict, inds, order, interpretable_only,
                      freq_octaves=False, f_octaves_zero=1e-3, aggr_func=np.max):
    energy = values_dict['spectrogram']
    aggr_axes = tuple(range(energy.ndim - 1))
    energy = aggr_func(energy, axis=aggr_axes) if energy.ndim > 1 else energy

    freq = np.log2(np.where(freq == 0.0, f_octaves_zero, freq)) if freq_octaves else freq

    moments = central_moments(freq, time, energy, inds, order=order, interpretable_only=interpretable_only)
    naming = "m{0}{1}".format
    moment_dicts = {naming(*ind): mij for ind, mij in np.ndenumerate(moments)}
    del moments
    interpret_central_moments(moment_dicts, interpretable_only=interpretable_only)
    return moment_dicts

