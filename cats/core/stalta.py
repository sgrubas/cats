"""
    Implements similar detector API for STA/LTA
"""

import numpy as np
import holoviews as hv
import numba as nb
from .utils import ReshapeArraysDecorator
from .projection import ProjectFilterIntervals


###### BACKENDS ######


@nb.njit(["f8[:](f8[:], i8, i8, i8, i8, i8)",
          "f8[:](f4[:], i8, i8, i8, i8, i8)"])
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
          "f8[:, :](f4[:, :], i8, i8, i8, i8)"], parallel=True)
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


class DetectorSTALTA:
    def __init__(self, dt_sec, long_window_sec, short_window_sec,
                 min_duration_sec, separation_sec, threshold,
                 windows_overlap_sec=None, step_sec=None, padmode='reflect',
                 charateristic='square'):
        self.dt_sec = dt_sec
        self.long_window_sec = long_window_sec
        self.short_window_sec = short_window_sec
        self.min_duration_sec = min_duration_sec
        self.separation_sec = separation_sec
        self.windows_overlap_sec = max(windows_overlap_sec or 0, 0)
        self.step_sec = max(step_sec or self.dt_sec, self.dt_sec)

        self.long_window_len = int(long_window_sec / self.dt_sec)
        self.short_window_len = int(short_window_sec / self.dt_sec)
        self.min_duration_len = int(min_duration_sec / self.dt_sec)
        self.separation_len = int(separation_sec / self.dt_sec)
        self.windows_overlap_len = int(self.windows_overlap_sec / self.dt_sec)
        self.step_len = int(self.step_sec / self.dt_sec)

        self.threshold = threshold
        self.padmode = padmode
        self.charateristic = charateristic
        ch_funcs = {'abs': np.abs, 'square': np.square}
        self.ch_func = ch_funcs[charateristic]

    def __call__(self, x):
        time = np.arange(x.shape[-1]) * self.dt_sec
        R = STA_LTA(self.ch_func(x), self.long_window_len, self.short_window_len,
                    self.step_len, self.windows_overlap_len, self.padmode)
        D = ProjectFilterIntervals(R > self.threshold, time, self.separation_sec, self.min_duration_sec, time)
        return ResultSTALTA(x, R, D, self.threshold, time)


class ResultSTALTA:
    def __init__(self, signal, stalta, detection, threshold, time):
        self.signal = signal
        self.stalta = stalta
        self.detection = detection
        self.threshold = threshold
        self.time = time

    def plot(self, ind, time_interval_sec=(None, None)):
        t_dim = hv.Dimension('Time', unit='s')

        if isinstance(time_interval_sec, tuple):
            t1 = 0 if (y := time_interval_sec[0]) is None else y
            t2 = self.time[-1] if (y := time_interval_sec[1]) is None else y
        elif time_interval_sec is None:
            t1, t2 = 0, 0
        else:
            t1, t2 = time_interval_sec

        dt = self.time[1] - self.time[0]
        ti1, ti2 = int(t1 / dt), int(t2 / dt)
        time = self.time[ti1: ti2]
        fig0 = hv.Curve((time, self.signal[ind][ti1: ti2]), kdims=[t_dim], vdims='Amplitude',
                        label='0. Input data: $x_n$').opts(xlabel='', linewidth=0.2)
        fig1 = hv.Curve((time, self.stalta[ind][ti1: ti2]), kdims=[t_dim], vdims='Ratio',
                        label='1. STA/LTA: $r_n$').opts(xlabel='')
        fig1 = fig1 * hv.HLine(self.threshold).opts(color='k', alpha=0.6)
        fig2 = hv.Curve((time, self.detection[ind][ti1: ti2].astype(float)), kdims=[t_dim], vdims='Classification',
                        label='2. Detection: $o_n$')

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250
        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=(t1, t2))
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        fig = (fig0 + fig1 + fig2).cols(1).opts(layout_opts, curve_opts)
        return fig
