import numpy as np
import holoviews as hv
from .utils import format_interval_by_limits, give_index_slice_by_limits, give_rectangles, intervals_intersection
hv.extension('matplotlib')


def plot_traces(data, time, intervals=None, picks=None, associated_picks=None, trace_loc=None,
                time_interval_sec=None, gain=1, amplitude_scale=None, clip=False, each_trace=1, **kwargs):

    trace_loc = trace_loc if (trace_loc is not None) else np.arange(data.shape[0]).astype(float)

    t1, t2 = time_interval_sec = format_interval_by_limits(time_interval_sec, (time.min(), time.max()))
    i_t = give_index_slice_by_limits(time_interval_sec, dt=time[1] - time[0], t0=time[0])
    i_trace = slice(None, None, each_trace)
    ind_slice = (i_trace, i_t)

    data_slice = data[ind_slice]
    loc_slice = trace_loc[ind_slice[0]]

    kwargs.setdefault('color', 'black')
    kwargs.setdefault('lw', 0.5)
    kwargs.setdefault('fig_size', 400)

    assert (data_slice.ndim == 2)

    trace_dim = hv.Dimension("Trace")
    t_dim = hv.Dimension("Time", unit='s')

    dloc = np.nanmin(np.diff(loc_slice)) if len(loc_slice) > 1 else 1
    scale = gain * dloc
    amax = amplitude_scale or np.nanmedian(np.nanmax(abs(data_slice), axis=-1))
    amax = amax or 1.0
    dc = (data_slice / amax * scale)
    if clip:
        level = dloc / 2.1
        dc[-level > dc] = -level
        dc[level < dc] = level
    dc = dc + np.expand_dims(loc_slice, -1)

    #  Data
    traces = hv.Overlay([hv.Curve((time[i_t], xi), kdims=[t_dim], vdims=[trace_dim]) for xi in dc])
    traces = traces.opts(hv.opts.Curve(color=kwargs['color'], linewidth=kwargs['lw']))

    #  Events intervals
    rects = []
    if intervals is not None:
        sliced_intervals = np.empty(len(intervals), dtype=object)
        for ind, intv in enumerate(intervals):
            sliced_intervals[ind] = intervals_intersection(intv, (t1, t2))

        rectangles = give_rectangles(sliced_intervals, loc_slice, scale / gain / 2.2)
        rects.append(hv.Rectangles(rectangles).opts(facecolor='blue', color='blue',
                                                    alpha=kwargs.get('alpha', 0.3)))
    rects = hv.Overlay(rects)

    #  Not associated picks
    onsets = []
    if picks is not None:
        onset_picks = []
        for pi, locy in zip(picks, trace_loc):
            p_i = pi[(t1 <= pi) & (pi <= t2)]
            onset_picks.append(np.stack([p_i, np.full_like(p_i, locy)], axis=-1))
        onset_picks = np.concatenate(onset_picks, axis=0)

        onsets = [hv.Points(onset_picks, kdims=[t_dim, trace_dim]).opts(marker='|', facecolors='red',
                                                                        edgecolors=None, s=600)]

    onsets = hv.Overlay(onsets)

    #  Associated Picks
    curves = []
    if associated_picks is not None:
        min_times = np.nanmin(associated_picks, axis=0)
        slice_ons = (t1 <= min_times) & (min_times <= t2)
        ons_inds = (i_trace, slice_ons)

        curves = [hv.Curve((pi, loc_slice), label=f"Event {i}",
                           kdims=[t_dim], vdims=[trace_dim]).opts(marker='|', ms=25, linewidth=2)
                  for i, pi in enumerate(associated_picks[ons_inds].T)]
    curves = hv.Overlay(curves)

    #  Summary of all
    ylims = (np.nanmin(trace_loc) - dloc, np.nanmax(trace_loc) + dloc)
    figure = hv.Overlay((traces, rects, onsets, curves))
    figure = figure.opts(ylim=ylims, xlim=time_interval_sec, fig_size=kwargs['fig_size'])
    return figure
