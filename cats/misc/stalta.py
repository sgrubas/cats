"""
    Implements similar detector API for STA/LTA
"""

from pydantic import BaseModel, Extra
from typing import Callable, Union, Tuple, List

import numpy as np
import holoviews as hv
import numba as nb
from tqdm.notebook import tqdm

from cats.core.utils import ReshapeArraysDecorator, give_rectangles, update_object_params
from cats.core.utils import format_index_by_dims, format_interval_by_limits, give_index_slice_by_limits
from cats.core.projection import RemoveGaps, GiveIntervals
from cats.core.association import PickDetectedPeaks
from cats.core.utils import aggregate_array_by_axis_and_func, cast_to_bool_dict, del_vals_by_keys, StatusKeeper
from cats.baseclass import CATSResult, CATSBase
from cats.detection import CATSDetector


###### BACKENDS ######

@nb.njit(["f8[:](f8[:], i8, i8, i8, i8, i8)",
          "f8[:](f4[:], i8, i8, i8, i8, i8)"], cache=True)
def cpu_STA_LTA_backend_vector(x, left, lleft, right, rright, step):
    n = len(x)
    M = int((n - rright) / step) + 1

    y = np.zeros(M)

    lta = np.sum(x[: left])
    sta = np.sum(x[lleft: rright])
    y[0] = sta / (lta + 1e-16) * left / right
    for i in range(1, M):
        k0, k1 = step * (i - 1), step * i
        if step > 1:
            lta = lta - np.sum(x[k0: k1]) + np.sum(x[left + k0: left + k1])
            sta = sta - np.sum(x[lleft + k0: lleft + k1]) + np.sum(x[rright + k0: rright + k1])
        else:
            lta = lta - x[k0] + x[left + k0]
            sta = sta - x[lleft + k0] + x[rright + k0]
        y[i] = sta / (lta + 1e-16) * left / right

    return y


@nb.njit(["f8[:, :](f8[:, :], i8, i8, i8, i8)",
          "f8[:, :](f4[:, :], i8, i8, i8, i8)"], parallel=True, cache=True)
def cpu_STA_LTA_backend(X, left, right, step, overlap):
    lleft = left - overlap
    rright = lleft + right
    cpu = nb.config.NUMBA_NUM_THREADS
    N, n = X.shape

    M = int((n - rright) / step) + 1
    Mp = int(M / cpu)

    Y = np.zeros((N, M))
    for i in nb.prange(N):
        xi = X[i]
        for j in nb.prange(cpu):
            j0, j1 = j * Mp, (j + 1) * Mp
            k0, k1 = j0 * step, (j1 - 1) * step + rright
            if j != cpu - 1:
                j1, k1 = M, n
            Y[i, j0: j1] = cpu_STA_LTA_backend_vector(xi[k0: k1], left, lleft,
                                                      right, rright, step)

    return Y


@ReshapeArraysDecorator(dim=2, input_num=1, methodfunc=False, output_num=1, first_shape=True)
def STA_LTA(x, long, short, step, overlap, padmode='reflect'):
    X = np.pad(x, [(0, 0), (long - overlap, short - 1)], mode=padmode)
    return cpu_STA_LTA_backend(X, long, short, step, overlap)


##### API #####


class STALTADetector(BaseModel, extra=Extra.allow):
    dt_sec: float
    long_window_sec: float
    short_window_sec: float
    min_duration_sec: float
    min_separation_sec: float
    threshold: float
    windows_overlap_sec: float = None
    step_sec: float = None
    padmode: str = 'reflect'
    characteristic: str = 'square'

    aggregate_axis_for_likelihood: Union[int, Tuple[int]] = None
    aggregate_func_for_likelihood: Callable[[np.ndarray], np.ndarray] = np.max
    pick_features: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_params()

    def _set_params(self):
        self.windows_overlap_sec = max(self.windows_overlap_sec or 0, 0)
        self.step_sec = max(self.step_sec or self.dt_sec, self.dt_sec)

        self.long_window_len = round(self.long_window_sec / self.dt_sec)
        self.short_window_len = round(self.short_window_sec / self.dt_sec)
        self.min_duration_len = round(self.min_duration_sec / self.dt_sec)
        self.min_separation_len = round(self.min_separation_sec / self.dt_sec)
        self.windows_overlap_len = round(self.windows_overlap_sec / self.dt_sec)
        self.step_len = round(self.step_sec / self.dt_sec)

        self.ch_functions = {'abs': np.abs, 'square': np.square}
        self.ch_func = self.ch_functions[self.characteristic]

    def reset_params(self, **params):
        """
            Updates the instance with changed parameters.
        """
        update_object_params(self, **params)

    def _detect(self, x, verbose=True, full_info=False):
        full_info = self._parse_info_dict(full_info)

        result = dict.fromkeys(full_info.keys())
        result['signal'] = x if full_info['signal'] else None

        history = StatusKeeper(verbose=verbose)

        with history(current_process='STA/LTA'):
            result['likelihood'] = STA_LTA(self.ch_func(x), self.long_window_len, self.short_window_len,
                                           self.step_len, self.windows_overlap_len, self.padmode)

        with history(current_process='Thresholding'):
            result['likelihood'] = aggregate_array_by_axis_and_func(result['likelihood'],
                                                                    self.aggregate_axis_for_likelihood,
                                                                    self.aggregate_func_for_likelihood,
                                                                    min_last_dims=1)

            result['detection'] = RemoveGaps(result['likelihood'] > self.threshold,
                                             self.min_separation_len,
                                             self.min_duration_len)

        del_vals_by_keys(result, full_info, ['detection'])

        stalta_time = np.arange(result['likelihood'].shape[-1]) * self.step_sec

        # Picking
        if full_info['picked_features']:
            with history(current_process='Picking'):
                result['picked_features'] = PickDetectedPeaks(result['likelihood'], result['detection'],
                                                              dt=self.step_sec)
        del_vals_by_keys(result, full_info, ['likelihood'])

        history.print_total_time()

        from_full_info = {kw: result.get(kw, None) for kw in full_info}
        kwargs = {**from_full_info,
                  "dt_sec": self.dt_sec,
                  "stalta_dt_sec": self.step_sec,
                  "npts": x.shape[-1],
                  "stalta_npts": len(stalta_time),
                  "threshold": self.threshold,
                  "history": history}

        return STALTADetectionResult(**kwargs)

    def detect(self, x: np.ndarray,
               /,
               verbose: bool = False,
               full_info: Union[bool, str, List[str]] = False):
        """
            Performs the detection on the given dataset. If the data processing does not fit the available memory,
            the data are split into chunks.

            Arguments:
                x : np.ndarray (..., N) : input data with any number of dimensions, but the last axis `N` must be Time.
                verbose : bool : whether to print status and timing
                full_info : bool / str / List[str] : whether to save workflow stages for further quality control.
                        If `True`, then all stages are saved, if `False` then only the `detection` is saved.
                        If string "plot" or "plt" is given, then only stages needed for plotting are saved.
                        Available workflow stages, if any is listed then saved to result:
                            - "signal" - input signal
                            - "likelihood" - calculated STA/LTA
                            - "detection" - binary classification [noise / signal], always returned.
        """

        n_chunks = self._split_data_by_memory(x, full_info=full_info, to_file=False)
        single_chunk = n_chunks > 1
        data_chunks = np.array_split(x, n_chunks, axis=-1)
        results = []
        for dc in tqdm(data_chunks, desc='Data chunks', display=single_chunk):
            results.append(self._detect(dc, verbose=verbose, full_info=full_info))
        return STALTADetectionResult.concatenate(*results)

    def __mul__(self, x):
        return self.detect(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.detect(x, verbose=True, full_info=True)

    def _parse_info_dict(self, full_info):
        info_keys = ["signal",
                     "likelihood",
                     "detection",
                     "picked_features"]

        if isinstance(full_info, str) and \
           (full_info in ['plot', 'plotting', 'plt', 'qc', 'main']):  # only those needed for plotting step-by-step
            full_info = {'signal':     True,
                         'likelihood': True}

        full_info = cast_to_bool_dict(full_info, info_keys)

        full_info["detection"] = True  # default saved result
        full_info["picked_features"] = self.pick_features  # non-default result

        return full_info

    def memory_usage_estimate(self, x, /, full_info=False):
        stalta_time_len = int(x.shape[-1] / self.step_len)
        stalta_shape = x.shape[:-1] + (stalta_time_len,)
        stalta_size = np.prod(stalta_shape)

        x_bytes = x.nbytes
        precision_order = x_bytes / x.size

        memory_usage_bytes = {
            "signal": 1. * x_bytes,  # float / int
            "likelihood": 1. * precision_order * stalta_size,  # float
            "detection": 1. * stalta_size,  # bool always
        }

        used_together = [('likelihood', 'detection')]
        base_info = ["signal"]
        full_info = self._parse_info_dict(full_info)

        return CATSBase.memory_info(memory_usage_bytes, used_together, base_info, full_info)

    def _split_data_by_memory(self, x, /, full_info, to_file=False):
        memory_info = self.memory_usage_estimate(x, full_info=full_info)
        return CATSBase.memory_chunks(memory_info, to_file)

    def detect_to_file(self, x, /, path_destination, verbose=False, full_info=False, compress=False):
        CATSDetector._detect_to_file(self, x, path_destination, verbose, full_info, compress)

    def detect_on_files(self, data_folder, data_format, result_folder=None,
                        verbose=False, full_info=False, compress=False):
        CATSDetector._detect_on_files(self, data_folder, data_format, result_folder, verbose, full_info, compress)


class STALTADetectionResult(CATSResult):
    def __init__(self,
                 signal=None,
                 likelihood=None,
                 detection=None,
                 picked_features=None,
                 threshold=None,
                 dt_sec=None,
                 stalta_dt_sec=None,
                 npts=None,
                 stalta_npts=None,
                 history=None,
                 **kwargs):

        # Numpy arrays
        self.signal = signal
        self.likelihood = likelihood
        self.detection = detection
        self.picked_features = picked_features

        # Scalars
        self.dt_sec = dt_sec
        self.stalta_dt_sec = stalta_dt_sec
        self.npts = npts
        self.stalta_npts = stalta_npts
        self.threshold = threshold
        self.history = history

    def plot(self, ind=None, time_interval_sec=(None, None)):
        if ind is None:
            ind = (0,) * (self.signal.ndim - 1)
        t_dim = hv.Dimension('Time', unit='s')
        L_dim = hv.Dimension('Likelihood')

        ind = format_index_by_dims(ind, self.signal.shape)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.npts - 1) * self.dt_sec))
        t1, t2 = time_interval_sec

        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        i_stalta = give_index_slice_by_limits(time_interval_sec, self.stalta_dt_sec)
        inds_time = ind + (i_time,)
        inds_stalta = ind + (i_stalta,)

        time = self.time(time_interval_sec)
        stalta_time = self.stalta_time(time_interval_sec)

        fig0 = hv.Curve((time, self.signal[inds_time]), kdims=[t_dim], vdims='Amplitude',
                        label='0. Input data: $x(t)$').opts(xlabel='', linewidth=0.2)

        likelihood = np.nan_to_num(self.likelihood[inds_stalta],
                                   posinf=10 * self.threshold,
                                   neginf=-10 * self.threshold)  # POSSIBLE `NAN` AND `INF` VALUES!
        likelihood_fig = hv.Curve((stalta_time, likelihood), kdims=[t_dim], vdims=L_dim)

        # Peaks
        if getattr(self, 'picked_features', None) is not None:
            P = self.picked_features[ind]
            P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
            peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
            peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim).opts(marker='D', color='r')
        else:
            peaks_fig = None

        # Intervals
        intervals = GiveIntervals(self.detection[inds_stalta])
        interv_height = (np.max(likelihood) / 2) * 1.1
        rectangles = give_rectangles(intervals, stalta_time, [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles,
                                      kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                             linewidth=0,
                                                                             alpha=0.2)
        snr_level_fig = hv.HLine(self.threshold, kdims=[t_dim, L_dim]).opts(color='k', linestyle='--', alpha=0.7)

        last_figs = [intervals_fig, snr_level_fig, likelihood_fig]
        if peaks_fig is not None:
            last_figs.append(peaks_fig)

        fig1 = hv.Overlay(last_figs, label='1. Likelihood and Detection:'
                                           ' $\mathcal{L}(t)$ and $\mathcal{D}(t)$')

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250

        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=time_interval_sec)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        fig = (fig0 + fig1).cols(1).opts(layout_opts, curve_opts)
        return fig

    def stalta_time(self, time_interval_sec=None):
        if time_interval_sec is None:
            return np.arange(self.stalta_npts) * self.stalta_dt_sec
        else:
            return np.arange(time_interval_sec[0], time_interval_sec[1] + self.stalta_dt_sec, self.stalta_dt_sec)

    def append(self, other):

        concat_attrs = ["signal",
                        "likelihood",
                        "detection"]

        for name in concat_attrs:
            self._concat(other, name, -1)

        stalta_t0 = self.stalta_dt_sec * self.stalta_npts

        self._concat(other, "picked_features", -2, stalta_t0, (..., 0))

        self.npts += other.npts
        self.stalta_npts += other.stalta_npts

        self.history.merge(other.history)

        assert self.threshold == other.threshold
        assert self.dt_sec == other.dt_sec
        assert self.stalta_dt_sec == other.stalta_dt_sec
