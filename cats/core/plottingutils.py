import numpy as np
import numba as nb
import holoviews as hv

from .projection import GiveIntervals


@nb.njit("List(UniTuple(f8, 4))(i8[:, :], f8[:], f8, f8)")
def _give_rectangles(events, time, yloc, dy):
    rectangles = []
    for e1, e2 in events:
        rectangles.append((time[e1], yloc - dy, time[e2], yloc + dy))
    return rectangles


def give_rectangles(events, time, yloc, dy):
    rectangles = []
    for i, (trace, yi) in enumerate(zip(events, yloc)):
        if len(trace) > 0:
            rectangles += _give_rectangles(trace, time, yi, dy)
    return rectangles


def plot_traces(traces, detection, fname, comp, gain=1, rsp=1, **kwargs):
    kwargs.setdefault('color', 'black')
    kwargs.setdefault('lw', 0.5)
    kwargs.setdefault('figsize', 400)

    t_dim = hv.Dimension('Time', unit='s')
    dco = traces[fname]
    dco = dco[dco.Component == comp].squeeze()

    locy = dco.Location.values
    dy = min(np.diff(locy))
    scale = ((rsp - 1) + 1) * gain * dy
    dc = (dco / abs(dco).max(axis=-1).mean() * scale) + dco.Location

    ### Data
    traces = hv.Overlay([hv.Curve(xi, kdims=[t_dim]) for xi in dc])
    traces = traces.opts(hv.opts.Curve(color=kwargs['color'],
                                       linewidth=kwargs['lw']))
    trace_vis = [(i % rsp == 0) for i in range(len(dc))]
    traces = traces.opts(hv.opts.Curve(visible=hv.Cycle(trace_vis)))

    ### Events intervals
    vis = trace_vis
    colors = ['blue', 'green', 'red']
    names = ['Detected', 'False Alarm', 'Missed Event']
    rects = []
    if detection is not None:
        time = detection[fname].coords['Time'].values
        det = detection[fname]
        det = det[det.Component == comp].squeeze().values
        if det.ndim == 2:
            det = det[..., None]

        for i in range(det.shape[-1]):
            intervals = GiveIntervals(det[..., i])
            intervals = [intvs for vi, intvs in zip(vis, intervals) if vi]
            rectangles = give_rectangles(intervals, time, dc.Location[vis].values,
                                         scale / gain / 2.2)
            rects.append(hv.Rectangles(rectangles, label=names[i]).opts(facecolor=colors[i],
                                                                        color=colors[i],
                                                                        alpha=kwargs.get('alpha', 0.3)))
    rects = hv.Overlay(rects)

    ### Summary of all
    ylims = (locy.min() - 2 * dy, locy.max() + 2 * dy)
    figure = (traces * rects)
    figure = figure.opts(ylim=ylims, fig_size=kwargs['figsize'])
    return figure