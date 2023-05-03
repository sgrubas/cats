import numpy as np
import holoviews as hv
hv.extension('matplotlib')

from .projection import GiveIntervals
from .utils import format_interval_by_limits, give_index_slice_by_limits, give_rectangles


def plot_traces(data, time, detection=None, detection_time=None, picks=None, trace_loc=None,
                tlims=None, gain=1, clip=True, each_trace=1, **kwargs):

    trace_loc = trace_loc if (trace_loc is not None) else np.arange(data.shape[0]).astype(float)

    tlims = format_interval_by_limits(tlims, (time.min(), time.max()))
    i_t = give_index_slice_by_limits(tlims, time[1] - time[0])
    i_trace = slice(None, None, each_trace)
    ind_slice = (i_trace, i_t)

    data_slice = data[ind_slice]
    loc_slice = trace_loc[ind_slice[0]]

    kwargs.setdefault('color', 'black')
    kwargs.setdefault('lw', 0.5)
    kwargs.setdefault('figsize', 400)

    assert (data_slice.ndim == 2)

    trace_dim = hv.Dimension("Trace")
    t_dim = hv.Dimension("Time", unit='s')

    dloc = min(np.diff(loc_slice))
    scale = gain * dloc
    amax = abs(data_slice).max(axis=-1).mean()
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
    if detection is not None:
        tlims_det = format_interval_by_limits(tlims, (detection_time[0], detection_time[-1]))
        i_det = give_index_slice_by_limits(tlims_det, detection_time[1] - detection_time[0])
        ind_slice_det = (i_trace, i_det)
        det_slice = detection[ind_slice_det]
        det_time_slice = detection_time[i_det]

        intervals = GiveIntervals(det_slice)
        rectangles = give_rectangles(intervals, det_time_slice, loc_slice, scale / gain / 2.2)
        rects.append(hv.Rectangles(rectangles).opts(facecolor='blue', color='blue',
                                                    alpha=kwargs.get('alpha', 0.3)))
    rects = hv.Overlay(rects)

    #  Picks
    curves = []
    if picks is not None:
        min_times = np.nanmin(picks, axis=0)
        slice_ons = (tlims[0] <= min_times) & (min_times <= tlims[1])
        ons_inds = (i_trace, slice_ons)

        curves = [hv.Curve((pi, loc_slice), label=f"Event {i}",
                           kdims=[t_dim], vdims=[trace_dim]).opts(marker='|', ms=25, linewidth=2)
                  for i, pi in enumerate(picks[ons_inds].T)]
    curves = hv.Overlay(curves)

    #  Summary of all
    ylims = (trace_loc.min() - 2 * dloc, trace_loc.max() + 2 * dloc)
    figure = hv.Overlay((traces, rects, curves))
    figure = figure.opts(ylim=ylims, fig_size=kwargs['figsize'])
    return figure
