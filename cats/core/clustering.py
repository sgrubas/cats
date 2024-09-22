"""
    Functions for clustering trimmed spectrograms (in Time-Frequency or Space-Time-Frequency domain).
"""

import numpy as np
import numba as nb
from .utils import ReshapeArraysDecorator, replace_int_by_list
from scipy import ndimage, stats, interpolate
import pandas as pd
from collections import ChainMap, defaultdict

# ------------------- CLUSTERING ------------------- #


def Clustering(SNR, /, q, s, freq_octaves):
    """
        Performs clustering [Note 1] of trimmed spectrograms via Connected-Component labeling [Ch 7.7 P 152, 1; 2].
        Algorithm based on method "one component at a time" via graph traversal using Depth-First Search [3, 4, 5][Note 2].
        Works analogous to `scipy.ndimage.label`, but also:
            1. Supports arbitrary neighbourhood pixel distances `q` > 1 (see below for `q`).
            2. Supports log2 neighbourhood pixel distances for frequency (see below for `freq_octaves`).
            3. Filters out labeled clusters smaller than `s` (see below for `s`).

        Arguments:
            SNR : np.ndarray (M, Nf, Nt) or (M, Nc, Nf, Nt) :
                  Trimmed SNR spectrogram where nonzero elements (SNR > 0) will be labelled.
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

        References:
            1. Acharya, T., & Ray, A. K. (2005). Image processing: principles and applications. John Wiley & Sons.
            2. https://en.wikipedia.org/wiki/Connected-component_labeling
            3. Kleinberg, J., & Tardos, E. (2006). Algorithm design. Pearson Education India.
            4. https://en.wikipedia.org/wiki/Depth-first_search
            5. https://11011110.github.io/blog/2013/12/17/stack-based-graph-traversal.html
            6. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996, August). A density-based algorithm for discovering
               clusters in large spatial databases with noise. In kdd (Vol. 96, No. 34, pp. 226-231).
    """

    func = {2: _ClusteringN2D, 3: _ClusteringN3D}
    assert len(q) == len(s)
    return func[len(q)](SNR, q, s, freq_octaves)


IND_ND = lambda n: f'UniTuple(i8, {n})'
CLUSTER_STAT_SIGNATURE_ND = lambda n: f"Tuple(({IND_ND(n)}, {IND_ND(n)}))"


@nb.njit([f"{CLUSTER_STAT_SIGNATURE_ND(2)}({IND_ND(2)}, {IND_ND(2)}, f8[:, :], i4[:, :], i4, f8)",
          f"{CLUSTER_STAT_SIGNATURE_ND(2)}({IND_ND(2)}, {IND_ND(2)}, f4[:, :], i4[:, :], i4, f8)"],
         cache=True)
def depth_first_search_2D(ind, q, SNR, C, cid, freq_distance_octaves):
    q_f, q_t = q
    Nf, Nt = SNR.shape

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
                nonzero = (SNR[lind] > 0.0)  # SNR must be > 0
                non_visited = (C[lind] == 0)  # only non-visited elements are added to stack
                if nonzero and non_visited:
                    stack.append(lind)  # add to stack
                    C[lind] = -cid  # negative label means it's been visited

    return mins, maxs


@nb.njit([f"i4[:, :](f8[:, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))",
          f"i4[:, :](f4[:, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))"],
         cache=True)
def _Clustering2D(SNR, q, s, freq_octaves):
    shape = SNR.shape
    freq_width_octaves, freq_distance_octaves = freq_octaves

    # Finding and labelling clusters
    C = np.zeros(shape, dtype=np.int32)
    cluster_size = {}  # remembers cluster sizes
    cid = 1
    for i, j in np.argwhere(SNR):  # iteratively scan each pixel
        ind = (i, j)
        if C[ind] < 1:  # if pixel does not have label, do DFS
            cluster_size[cid] = depth_first_search_2D(ind, q, SNR, C, cid, freq_distance_octaves)
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
        C[ind] = cluster_newid.get(C[ind], 0)

    return C


@ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=False, output_num=2, first_shape=True)
@nb.njit([f"i4[:, :, :](f8[:, :, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))",
          f"i4[:, :, :](f4[:, :, :], {IND_ND(2)}, {IND_ND(2)}, UniTuple(f8, 2))"],
         parallel=True, cache=True)
def _ClusteringN2D(SNR, q, s, freq_octaves):
    C = np.empty(SNR.shape, dtype=np.int32)
    for i in nb.prange(SNR.shape[0]):
        C[i] = _Clustering2D(SNR[i], q, s, freq_octaves)
    return C


@nb.njit([f"{CLUSTER_STAT_SIGNATURE_ND(3)}({IND_ND(3)}, {IND_ND(3)}, f8[:, :, :], i4[:, :, :], i4, f8)",
          f"{CLUSTER_STAT_SIGNATURE_ND(3)}({IND_ND(3)}, {IND_ND(3)}, f4[:, :, :], i4[:, :, :], i4, f8)"],
         cache=True)
def depth_first_search_3D(ind, q, SNR, C, cid, freq_distance_octaves):
    """ See detailed comments in `depth_first_search_2D` """

    q_c, q_f, q_t = q
    Nc, Nf, Nt = SNR.shape

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
                    nonzero = (SNR[lind] > 0.0)
                    non_visited = (C[lind] == 0)
                    if nonzero and non_visited:
                        stack.append(lind)
                        C[lind] = -cid

    return mins, maxs


@nb.njit([f"i4[:, :, :](f8[:, :, :], {IND_ND(3)}, {IND_ND(3)}, UniTuple(f8, 2))",
          f"i4[:, :, :](f4[:, :, :], {IND_ND(3)}, {IND_ND(3)}, UniTuple(f8, 2))"],
         cache=True)
def _Clustering3D(SNR, q, s, freq_octaves):
    shape = SNR.shape
    freq_width_octaves, freq_distance_octaves = freq_octaves

    # finding clusters
    C = np.zeros(shape, dtype=np.int32)
    cluster_size = {}
    cid = 1
    for k, i, j in np.argwhere(SNR):
        ind = (k, i, j)
        if C[ind] < 1:
            cluster_size[cid] = depth_first_search_3D(ind, q, SNR, C, cid, freq_distance_octaves)
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
@nb.njit([f"i4[:, :, :, :](f8[:, :, :, :], {IND_ND(3)}, {IND_ND(3)}, UniTuple(f8, 2))",
          f"i4[:, :, :, :](f4[:, :, :, :], {IND_ND(3)}, {IND_ND(3)}, UniTuple(f8, 2))"],
         parallel=True, cache=True)
def _ClusteringN3D(SNR, q, s, freq_octaves):
    C = np.empty(SNR.shape, dtype=np.int32)
    for i in nb.prange(SNR.shape[0]):
        C[i] = _Clustering3D(SNR[i], q, s, freq_octaves)
    return C


# ---------------- Cluster catalogs ---------------- #


def bbox_peaks(freq, time, values, inds):

    peak_id = np.argmax(values)

    output = {"Time_start": np.min(time),
              "Time_end": np.max(time),
              "Frequency_start": np.min(freq),
              "Frequency_end": np.max(freq),
              "Time_peak": time[peak_id],
              "Frequency_peak": freq[peak_id],
              "Energy_peak": values[peak_id],
              }
    return output


def get_last_rising_point_spline(x, y, default_value, spline_lam=None):
    last_rising_point = default_value
    argsort = x.argsort()
    x = x[argsort]
    y = y[argsort]

    if len(x) < 5:  # min number of points for splines
        return default_value

    # Create smooth cubic spline
    if spline_lam is not None:
        spline = interpolate.make_smoothing_spline(x, y, lam=spline_lam)
        interp = interpolate.PPoly.from_spline(spline)  # Make piecewise-polynomial to have `roots` methods
    else:
        interp = interpolate.CubicSpline(x, y, bc_type='clamped')

    der1_interp = interp.derivative(1)
    der2_interp = interp.derivative(2)
    der3_interp = interp.derivative(3)

    zeros = der2_interp.roots()  # defines potential position phase arrivals (2nd derivative is zero)

    der1 = der1_interp(zeros)
    # der2 = der3_interp(zeros)
    der3 = der3_interp(zeros)
    picks = zeros[(der1 > 0) & (der3 < 0)]  # keep only those where energy is rising only

    if len(picks):
        # last_rising_point = picks[-1]  # not the last but second??
        last_rising_point = picks[interp(picks).argmax()]  # with max energy

    return last_rising_point


def first_and_strong_arrivals(freq, time, values, inds, spline_lam=0.0):
    freq_inds, time_inds = inds

    # Mode / Peak to identify the strongest event phase
    peak_id = np.argmax(values)
    f_peak_id = freq_inds[peak_id]
    t_peak_id = time_inds[peak_id]
    f_peak_inds = (freq_inds == f_peak_id)

    # 1. P-arrival estimate: when peak frequency first appears (may differ from `Time_start`)
    # Assumptions:
    #   1.1. P is the first phase
    #   1.2. P-phase has similar peak frequency as the strongest peak
    first_arrival = time[f_peak_inds].min()

    # 2. S-arrival estimate: zero of 2nd derivative at the strongest peak
    # Assumptions:
    #   2.1. S is the strongest phase
    #   2.2. Characteristic curve is energy slice at peak freq
    #   2.3. Polynomial approximation
    slice_ids = (time_inds <= t_peak_id) & f_peak_inds  # only preceding points at peak frequency
    # slice_ids = f_peak_inds  # only preceding points at peak frequency
    strong_arrival = get_last_rising_point_spline(time[slice_ids], values[slice_ids], time[peak_id], spline_lam)

    # strong_arrival = get_last_rising_point_poly(t[slice_ids], values[slice_ids], poly_deg, first_arrival)

    output = {"Time_first_arrival": first_arrival,
              "Time_strong_arrival": strong_arrival}

    return output


def phase_picks_from_spline_roots(freq, time, values, inds, spline_lam):
    # TODO:
    #   - check `freq` & `time` validity
    #   - check sorting of ch_curve
    raise NotImplementedError("Not fully implemented feature")

    # NOTE: A way to smooth characteristic curve - binning and averaging
    # sum_vals, bin_edges, binnumber = scipy.stats.binned_statistic(time_index[nonzero_sum], sum_vals,
    #                                                               statistic='mean', bins=bin_num)

    freq_inds, time_inds = inds

    # Estimate of all phase arrivals in a cluster via smooth spline interpolation
    # It can detect arrivals of multiple events and other phases (P/S/Surface/...) by energy change

    # Characteristic function
    t0_ind = time_inds.min()
    time_inds_shift = time_inds - t0_ind
    ch_curve = np.bincount(time_inds_shift, weights=values)  # Sum over frequencies
    ch_curve = ch_curve * np.bincount(time_inds_shift)  # Normalize by number of frequency bins

    # Remove zero-len time indices (side effect of `np.bincount`)
    nonzero_ids = (ch_curve > 0)
    ch_curve = ch_curve[nonzero_ids]
    # N = len(ch_curve)

    # Time axis
    time_index = np.arange(time_inds_shift.max() + 1) + t0_ind
    time_index = time_index[nonzero_ids]
    time_vals = time[time_index]

    try:
        argsort = time_vals.argsort()
        x = time_vals[argsort]
        y = ch_curve[argsort]

        # Create smooth cubic spline
        spline = interpolate.make_smoothing_spline(x, y, lam=spline_lam)
        interp = interpolate.PPoly.from_spline(spline)  # Make piecewise-polynomial to have `roots` methods

        der1_interp = interp.derivative(1)
        der2_interp = interp.derivative(2)
        der3_interp = interp.derivative(3)

        zeros = der2_interp.roots()  # defines potential position phase arrivals (2nd derivative is zero)

        der1 = der1_interp(zeros)
        der3 = der3_interp(zeros)
        picks = zeros[(der1 > 0) & (der3 < 0)]  # keep only those where energy is rising only

        # ---- Adjustment of picks by associated 3rd derivative zeros ---- seems unwarranted, better change `lam`
        # zero_der3 = der3_interp.roots()
        # pre_picks = picks.copy()
        # for i, pi in enumerate(pre_picks):
        #     zj = zero_der3[zero_der3 < pi]  # check missing der3-zeros neighbours to der2-zeros
        #     if len(zj) > 0:
        #         pre_picks[i] = zj[-1]
        # picks = (picks + pre_picks) / 2  # maybe non-uniform weighting?

    except ValueError:
        picks = []

    output = {"Phases": picks}

    return output


def energy_statistics(freq, time, values, inds):
    """
        References: (Section 6) http://dx.doi.org/10.1016/j.dsp.2017.07.015
    """
    N = values.size
    N = N if N > 1 else 2

    # Energy stats
    vmin = values.min()
    vmax = values.max()
    amean = values.mean()
    std = np.sqrt(np.sum((values - amean) ** 2) / (N - 1))
    skew = stats.skew(values, bias=False)
    kurtosis = stats.kurtosis(values, fisher=True, bias=False)
    gmean = stats.gmean(values)
    flatness = gmean / amean

    entropy = stats.entropy(values)  # Shannon entropy

    output = {"Energy_min": vmin,
              "Energy_max": vmax,
              "Energy_mean": amean,
              "Energy_std": std,
              "Energy_skew": skew,
              "Energy_kurtosis": kurtosis,
              "Energy_flatness": flatness,
              "Energy_entropy": entropy}

    return output


def freq_of_time_polynomial(freq, time, values, inds, order):
    try:
        if len(values) <= order:
            raise ValueError
        poly = np.polynomial.Polynomial.fit(time - time.min(), freq, deg=order, rcond=np.nan, w=values)
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


def calculate_moments(freq, time, values, inds, order, interpretable_only):
    moments = central_moments(freq, time, values, inds, order=order, interpretable_only=interpretable_only)
    naming = "m{0}{1}".format
    moment_dicts = {naming(*ind): mij for ind, mij in np.ndenumerate(moments)}
    del moments
    interpret_central_moments(moment_dicts, interpretable_only=interpretable_only)
    return moment_dicts


def default_features(freq, time, values, inds, **params):
    basic = params.get('bbox_peaks', True)
    polynomial = params.get('polynomial', False)
    energy = params.get('energy', False)
    moments = params.get('moments', False)
    arrivals = params.get('arrivals', False)
    spline_phases = params.get('spline_phases', False)

    calculated_features = []
    if basic:  # 1. bounding box
        bbox_arrivals = bbox_peaks(freq, time, values, inds)
        calculated_features.append(bbox_arrivals)

    if polynomial or isinstance(polynomial, dict):  # 2. cubic polynomial `freq(t)`
        order = 3
        if isinstance(polynomial, dict):
            order = polynomial.get('order', order)
        coefs = freq_of_time_polynomial(freq, time, values, inds, order=order)
        calculated_features.append(coefs)

    if energy:  # 3. energy statistics
        energy_stats = energy_statistics(freq, time, values, inds)
        calculated_features.append(energy_stats)

    if moments or isinstance(moments, dict):  # 4. moments up to 4th order
        order = 4
        interpretable_only = True
        if isinstance(moments, dict):
            order = moments.get('order', order)
            interpretable_only = moments.get('interpretable_only', order)

        moment_dicts = calculate_moments(freq, time, values, inds, order, interpretable_only)
        calculated_features.append(moment_dicts)

    if arrivals or isinstance(arrivals, dict):  # 5. P/S arrivals
        spline_lam = 0.0  # spline to get S arrival
        if isinstance(arrivals, dict):
            spline_lam = arrivals.get('spline_lam', spline_lam)
        arrivals = first_and_strong_arrivals(freq, time, values, inds, spline_lam=spline_lam)
        calculated_features.append(arrivals)

    if spline_phases or isinstance(spline_phases, dict):  # 6. All pickable phases arrivals
        spline_lam = 0.0
        if isinstance(spline_phases, dict):
            spline_lam = spline_phases.get('spline_lam', spline_lam)
        phases = phase_picks_from_spline_roots(freq, time, values, inds, spline_lam=spline_lam)
        calculated_features.append(phases)

    features = ChainMap(*calculated_features)

    return features


def feature_func_wrapper(funcs, freq, time):
    def wrapper(values, inds):
        sep_inds = np.divmod(inds, len(time))
        freq_inds, time_inds = sep_inds
        t = time[time_inds]  # time axis of `values`
        f = freq[freq_inds]  # freq axis of `values`

        output_dicts = [func(f, t, values, sep_inds) for func in funcs]
        return ChainMap(*output_dicts)

    return wrapper


def calculate_cluster_features(Val, CID, freq, time, feature_funcs):
    cids = np.arange(1, CID.max() + 1).tolist()  # list is important for unpacking further in ClusterCatalogs
    func = feature_func_wrapper(feature_funcs, freq, time)
    if len(cids) > 0:
        cluster_features = ndimage.labeled_comprehension(input=Val, labels=CID,
                                                         index=cids, func=func,
                                                         out_dtype=dict,
                                                         default={},  # empty dict by default!
                                                         pass_positions=True)
        # `if cldict` to skip non-existing cluster IDs (happens if multitrace)
        features_dict = {'Cluster_ID': [cid for cid, cldict in zip(cids, cluster_features) if cldict]}
        # extracts all unique keys
        ref_keys = ChainMap(*cluster_features).keys()
        for name in ref_keys:  # iter over features
            features_dict[name] = [cldict.get(name, np.nan)
                                   for cldict in cluster_features if cldict]  # list is important, see above
    else:
        features_dict = {}

    return features_dict


# TODO:
#   - implement cluster catalogs for 3D, when `multitrace=True`


def ClusterCatalogs(Val, CID, freq, time, feature_funcs=None, frequency_unit='Hz', time_unit='sec', energy_unit='',
                    trace_dim_names=None):
    """
        Calculates statistics/features of cluster in each trace

        Arguments:
             Val : np.ndarray (..., Nf, Nt) : array of values (PSD, SNR, ...)
             CID : np.ndarray (..., Nf, Nt) : array cluster ID values
             freq : np.ndarray (Nf,) : array of frequencies
             time : np.ndarray (Nt,) : array of time samples
             feature_funcs : list[callable] : Funcs of signature `func(freq, time, values, inds) -> dict[str, float]`,
                         where `freq`, `time`, `values` are 1D arrays, `inds = [freq_inds, time_inds]`.
                         Must return dict with calculated features.
             frequency_unit : str : unit name for adding suffix to column names in Catalog DataFrame
             time_unit : str : unit name for adding suffix to column names in Catalog DataFrame
             energy_unit : str : unit name for adding suffix to column names in Catalog DataFrame
             trace_dim_names : list[str] : names of trace dimensions for DataFrame MultiIndex

        Returns:
            Catalogs: pd.DataFrame : dataframe with calculated features for each trace and cluster
    """
    if feature_funcs is None:
        feature_funcs = [default_features]
    assert isinstance(feature_funcs, (list, tuple)), "`feature_funcs` must be list or tuple"

    shape = Val.shape[:-2]
    ndim = len(shape)

    if trace_dim_names is None:
        trace_dim_names = [f"Trace_dim_{i}" for i in range(ndim)]

    # 1. get features data in dicts; 2. then DataFrame. This order is much more efficient (~2-3x)
    catalogs = defaultdict(list)  # list type is important for appending/extending lists
    for ind in np.ndindex(*shape):
        # calculate features
        catalog = calculate_cluster_features(Val[ind], CID[ind], freq, time, feature_funcs)
        for name, data in catalog.items():
            catalogs[name].extend(data)

        if (cids := catalog.get("Cluster_ID", None)) is not None:  # if non-empty, at least one cluster
            for name, tr_i in zip(trace_dim_names, ind):  # iter over trace dims
                catalogs[name].extend([tr_i] * len(cids))

    # create DataFrame
    catalogs = pd.DataFrame(catalogs)  # Unified DataFrame for all traces

    if len(catalogs) > 0:  # otherwise throws errors
        catalogs.set_index(trace_dim_names, inplace=True)  # Create (Multi)Index for proper indexing over traces

        # append unit suffixes to appropriate feature names
        apply_units(catalogs, frequency_unit=frequency_unit, time_unit=time_unit, energy_unit=energy_unit)

        # sort columns order
        sorted_cols = sort_feature_names(catalogs.columns)
        catalogs = catalogs[sorted_cols]

    return catalogs


def apply_units(catalog, frequency_unit='Hz', time_unit='sec', energy_unit=''):
    cols = list(catalog.columns)
    unit_names = {'frequency': frequency_unit, 'time': time_unit, 'energy': energy_unit}
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
    sorted_cols = ["Cluster_ID"]
    sorted_cols += [ci for ci in cols if "time" in ci.casefold() and ci not in sorted_cols]
    sorted_cols += [ci for ci in cols if "frequency" in ci.casefold() and ci not in sorted_cols]
    sorted_cols += [ci for ci in ["First_arrival", "Strong_arrival"] if ci in cols]
    sorted_cols += [ci for ci in cols if "energy" in ci.casefold() and ci not in sorted_cols]
    sorted_cols += [ci for ci in cols if "ellipse" in ci.casefold() and ci not in sorted_cols]
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
                                 t0: float):
    time_shift_cols = ["Time_start_sec", "Time_end_sec",
                       "Time_peak_sec", "Time_centroid_sec",
                       "Time_first_arrival_sec", "Time_strong_arrival_sec"]

    if catalog2 is not None:
        time_shift_cols = [col_i for col_i in time_shift_cols if col_i in catalog2.columns]
        if len(catalog2) > 0:
            # time shift
            catalog2[time_shift_cols] += t0

            if catalog1 is not None:
                # cluster ID shift
                catalog2["Cluster_ID_2"] = catalog1.Cluster_ID.groupby(
                    catalog1.index.names).max()  # assign new col to handle empty traces with no clusters (NaNs)
                catalog2.Cluster_ID += catalog2.pop("Cluster_ID_2").fillna(0).astype(
                    int)  # empty traces with NaN are replaced with 0, and added to original Cluster_IDs

        catalog_new = pd.concat([catalog1, catalog2], copy=False)
        catalog_new.sort_values(by=catalog_new.index.names, inplace=True)
        return catalog_new

    elif (catalog1 is not None) and (catalog2 is None):
        return catalog1

    else:
        return None


def update_cluster_ID_by_catalog(cluster_ID, catalog, keep_catalog_rows):
    # the fastest so far
    remove_inds = ~keep_catalog_rows
    bad_catalog = catalog[remove_inds]
    for row in bad_catalog.itertuples():  # over traces with bad clusters
        ind, cid = row.Index, row.Cluster_ID
        CID = cluster_ID[ind]
        CID[CID == cid] = 0


def get_last_rising_point_poly(x, y, poly_deg, default_value):
    last_rising_point = default_value

    try:
        polynomial = np.polynomial.Polynomial.fit(x, y, deg=poly_deg, rcond=np.nan)
    except:
        return last_rising_point

    bound = polynomial.domain[-1]

    der1_poly = polynomial.deriv(1)
    der2_poly = polynomial.deriv(2)

    zeros1 = der1_poly.roots()
    # print(f"{zeros1 = }")
    der2_1 = der2_poly(zeros1)  # sign of 2nd deriv defines max or min
    minimums = zeros1[der2_1.real > 0].real
    if len(minimums):
        last_min = minimums[-1]
        last_rising_point = last_min
        after_min = (x >= last_min)

        try:
            polynomial = np.polynomial.Polynomial.fit(x[after_min], y[after_min], deg=3, rcond=np.nan)
        except:
            return last_rising_point

        der2_poly = polynomial.deriv(2)
        der3_poly = polynomial.deriv(3)
        zeros2 = der2_poly.roots()  # potential peak arrival (2nd derivative is zero)
        # print(f"{zeros2 = }")
        der3_2 = der3_poly(zeros2)
        last_zeros = zeros2[(zeros2.real >= last_min) &  # after min
                            (zeros2.real <= bound) &  # before peak
                            (der3_2 < 0)]  # when from min to max

        if len(last_zeros):
            last_rising_point = last_zeros[-1]
        else:  # if 2nd deriv is not zero after last min, try 3rd deriv approx
            zeros3 = der3_poly.roots()  # another potential peak arrival (3rd derivative is zero)
            # print(f"{zeros3 = }")
            der2_3 = der2_poly(zeros3)
            last_zeros = zeros3[(zeros3.real >= last_min) &  # after min
                                (der2_3.real > 0) &  # from min to rising
                                (zeros3.real <= bound)]  # before peak
            if len(last_zeros):
                last_rising_point = last_zeros[-1]

    return np.real(last_rising_point)
