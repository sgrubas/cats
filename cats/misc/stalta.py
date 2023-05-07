"""
    Implements similar detector API for STA/LTA
"""

from pydantic import BaseModel, Extra
from typing import Callable, Union, Tuple

import numpy as np
import holoviews as hv
import numba as nb
from cats.core.utils import ReshapeArraysDecorator, give_rectangles
from cats.core.utils import format_index_by_dims, format_interval_by_limits, give_index_slice_by_limits
from cats.core.projection import RemoveGaps, GiveIntervals
from cats.core.association import PickFeatures
from cats.core.utils import aggregate_array_by_axis_and_func, cast_to_bool_dict, del_vals_by_keys, StatusKeeper


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

    def detect(self, x, verbose=True, full_info=False):
        result = {"likelihood": None, "detection": None, "picked_features": None}

        full_info = cast_to_bool_dict(full_info, list(result.keys()))
        full_info['detection'] = True

        History = StatusKeeper(verbose=verbose)

        time = np.arange(x.shape[-1]) * self.dt_sec

        with History(current_process='STA/LTA'):
            result['likelihood'] = STA_LTA(self.ch_func(x), self.long_window_len, self.short_window_len,
                                           self.step_len, self.windows_overlap_len, self.padmode)

        with History(current_process='Thresholding'):
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
            with History(current_process='Picking'):
                result['picked_features'] = PickFeatures(result['likelihood'],
                                                         time=stalta_time,
                                                         min_likelihood=self.threshold,
                                                         min_width_sec=self.min_duration_sec,
                                                         num_features=None)
        del_vals_by_keys(result, full_info, ['likelihood'])

        return STALTAResult(signal=x,
                            stalta=result.get('likelihood', None),
                            detection=result.get('detection', None),
                            picked_features=result.get('picked_features', None),
                            threshold=self.threshold,
                            time=time,
                            stalta_time=stalta_time,
                            history=History)

    def __mul__(self, x):
        return self.detect(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.detect(x, verbose=True, full_info=True)


class STALTAResult:
    def __init__(self, signal, stalta, detection, picked_features, threshold, time, stalta_time, history):
        self.signal = signal
        self.stalta = stalta
        self.detection = detection
        self.picked_features = picked_features
        self.threshold = threshold
        self.time = time
        self.stalta_time = stalta_time
        self.history = history

    def plot(self, ind, time_interval_sec=(None, None)):
        t_dim = hv.Dimension('Time', unit='s')
        L_dim = hv.Dimension('Likelihood')

        ind = format_index_by_dims(ind, self.signal.shape)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (self.time[0], self.time[-1]))
        t1, t2 = time_interval_sec
        i_t = give_index_slice_by_limits(time_interval_sec, self.time[1] - self.time[0])
        i_st = give_index_slice_by_limits(time_interval_sec, self.stalta_time[1] - self.stalta_time[0])
        inds_t = ind + (i_t,)
        inds_st = ind + (i_st,)

        fig0 = hv.Curve((self.time[i_t], self.signal[inds_t]), kdims=[t_dim], vdims='Amplitude',
                        label='0. Input data: $x_n$').opts(xlabel='', linewidth=0.2)

        likelihood = self.stalta[inds_st]
        likelihood_fig = hv.Curve((self.stalta_time[i_st], likelihood),
                                  kdims=[t_dim], vdims=L_dim)

        # Peaks
        P = self.picked_features[ind]
        P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
        peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
        peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim).opts(marker='D', color='r')

        # Intervals
        intervals = GiveIntervals(self.detection[inds_st])
        interv_height = (likelihood.max() / 2) * 1.1
        rectangles = give_rectangles(intervals, self.stalta_time[i_st], [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles, kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                                         linewidth=0, alpha=0.1)
        snr_level_fig = hv.HLine(self.threshold, kdims=[t_dim, L_dim]).opts(color='k', linestyle='--', alpha=0.7)

        fig1 = hv.Overlay([intervals_fig, snr_level_fig, likelihood_fig, peaks_fig],
                          label='1. STA/LTA: $r_n$')

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250

        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=time_interval_sec)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        fig = (fig0 + fig1).cols(1).opts(layout_opts, curve_opts)
        return fig
