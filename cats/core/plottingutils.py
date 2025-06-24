import numpy as np
import holoviews as hv
from holoviews.plotting.links import DataLink
from typing import Union
from .utils import format_interval_by_limits, give_index_slice_by_limits, give_rectangles, intervals_intersection
from scipy.signal import detrend as detrend_func
hv.extension('matplotlib', 'bokeh')


def scale_funcs(x, spec, per_station_scale=False):
    funcs = {'median': np.nanmedian,
             'mean': np.nanmean,
             'max': np.nanmax,
             'min': np.nanmin}

    func_names = spec.split('_')

    if len(func_names) == 1:
        func1 = funcs[func_names[0]]
        axes = (0, 2) if x.ndim == 3 else (1,)
        axes = axes if per_station_scale else None
        return func1(x, axis=axes, keepdims=True)

    elif len(func_names) == 2:
        func1 = funcs[func_names[0]]
        func2 = funcs[func_names[1]]
        x2 = func2(x, axis=-1, keepdims=True)
        if x.ndim == 3:
            x1 = func1(x2, axis=0 if per_station_scale else None, keepdims=True)
        else:
            x1 = x2 if per_station_scale else func1(x2, axis=None, keepdims=True)
        return x1

    else:
        raise ValueError(f"Invalid scale function: {spec}, max 2 is allowed")


def plot_traces(data: np.ndarray,
                time: Union[float, np.ndarray] = None,
                intervals: np.ndarray = None,
                picks: np.ndarray = None,
                station_loc: Union[float, np.ndarray] = None,
                time_interval_sec: tuple[float, float] = None,
                gain: float = 1,
                detrend_type: str = 'constant',
                amplitude_scale: Union[float, np.ndarray] = None,
                per_station_scale: bool = False,
                scale_func: str = 'mean_max',
                clip: bool = False,
                clip_height: float = 0.95,
                each_station: int = 1,
                interactive: bool = False,
                allow_picking: bool = False,
                component_labels: list[str] = None,
                station_labels: list[str] = None,
                **kwargs):

    # ---- Argument parsing ---- #
    if data.ndim == 1:
        data = data.reshape(1, -1)

    assert (data.ndim == 2) or (data.ndim == 3)

    if not isinstance(time, np.ndarray):
        dt = 1 if time is None else time
        time = np.arange(data.shape[-1]) * dt

    if not isinstance(station_loc, np.ndarray):
        dx = 1 if station_loc is None else station_loc
        station_loc = np.arange(data.shape[-2], dtype=float) * dx

    # ---- Getting data interval ---- #
    t1, t2 = time_interval_sec = format_interval_by_limits(time_interval_sec, (time.min(), time.max()))
    i_t = give_index_slice_by_limits(time_interval_sec, dt=time[1] - time[0], t0=time[0])
    i_trace = slice(None, None, each_station)

    ind_slice = (..., i_trace, i_t)
    data_slice = data[ind_slice]
    loc_slice = station_loc[i_trace]
    time_slice = time[i_t]

    if detrend_type:
        detrend_type = 'linear' if detrend_type == 'linear' else 'constant'
        data_slice = detrend_func(data_slice, axis=-1, type=detrend_type)

    num_stations = data_slice.shape[-2]
    multi_comp = data_slice.ndim == 3
    num_comp = data_slice.shape[0] if multi_comp else 1

    # ---- Scaling data ---- #
    dloc = np.nanmin(np.diff(loc_slice)) if len(loc_slice) > 1 else 1
    dloc = dloc if dloc > 0 else 1
    scale = gain * dloc

    if not per_station_scale:
        # amax = amplitude_scale or np.nanmedian(np.nanmax(abs(data_slice), axis=-1))
        amax = amplitude_scale or scale_funcs(abs(data_slice), scale_func, per_station_scale=per_station_scale)
        amax = np.where(amax != 0, amax, 1.0)  # remove zero divisor
    else:
        if amplitude_scale is not None:
            amax = np.expand_dims(amplitude_scale, axis=-1)
            amax = amax if not multi_comp else np.expand_dims(amax, axis=0)
        else:
            # amax = np.nanmax(abs(data_slice), axis=-1, keepdims=True)  # per station
            # amax = amax if not multi_comp else np.nanmedian(amax, axis=0, keepdims=True)  # per component
            amax = scale_funcs(abs(data_slice), scale_func, per_station_scale=per_station_scale)
            amax = np.where(amax != 0, amax, 1.0)  # remove zero divisor

    data_slice = (data_slice / amax * scale)
    clip_level = dloc / 2 * clip_height  # clip level, divided by 2 (1 for UP and 1 for DOWN)
    if clip:
        data_slice[-clip_level > data_slice] = -clip_level
        data_slice[clip_level < data_slice] = clip_level

    # ---- Labels ---- #

    if component_labels is None:
        component_labels = ["E", "N", "Z"] if num_comp <= 3 else list(range(num_comp))

    if station_labels is None:
        station_labels = list(range(1, data.shape[-2] + 1, each_station))
    station_labels = [(i, st) for i, st in zip(loc_slice, station_labels)]

    trace_dim = hv.Dimension("Station")
    t_dim = hv.Dimension("Time", unit='s')

    # ---- Creating plotting objects ---- #
    #  Data traces
    traces = []
    for i in range(num_stations):  # iter over traces
        for cj in range(num_comp):  # iter over components
            j_ind = cj if multi_comp else Ellipsis  # ellipsis is for absent component axis
            trace_curve = data_slice[j_ind, i, :] + loc_slice[i]
            traces.append(hv.Curve((time_slice, trace_curve),
                                   kdims=[t_dim], vdims=[trace_dim],
                                   label=component_labels[cj] if num_comp > 1 else ''))
    traces = hv.Overlay(traces)

    #  Events intervals
    rects = []
    if intervals is not None:
        object_dtype = (intervals.dtype == object)
        indexer = (..., i_trace) + (slice(None),) * (not object_dtype)
        shape_slice = slice(None, None if object_dtype else -1)
        intervals_slice = intervals[indexer]
        shape = intervals_slice.shape[shape_slice]

        sliced_intervals = []
        intv_locy = []
        for ind in np.ndindex(shape):
            intv_i = intervals_slice[ind]
            if intv_i is not None:
                sliced_intervals.append(intervals_intersection(intv_i, reference_interval=(t1, t2)))
                intv_locy.append(loc_slice[ind[-1]])

        rectangles = give_rectangles(sliced_intervals, intv_locy, clip_level)
        rects.append(hv.Rectangles(rectangles, kdims=[t_dim, trace_dim, 'x2', 'y2'],
                                   label="Intervals" * interactive))
    rects = hv.Overlay(rects)

    #  Not associated picks
    onset_picks = np.zeros((0, 2))
    if picks is not None:
        object_dtype = (picks.dtype == object)
        indexer = (..., i_trace) + (slice(None),) * (not object_dtype)
        shape_slice = slice(None, None if object_dtype else -1)
        picks_slice = picks[indexer]
        shape = picks_slice.shape[shape_slice]

        onset_picks = []
        for ind in np.ndindex(shape):
            pi = picks_slice[ind]
            if pi is not None:
                locy = loc_slice[ind[-1]]
                p_i = pi[(t1 <= pi) & (pi <= t2)]
                onset_picks.append(np.stack([p_i, np.full_like(p_i, locy)], axis=-1))
        onset_picks = np.concatenate(onset_picks, axis=0) if onset_picks else np.empty((0, 2))
    original_onsets = hv.Points(onset_picks, kdims=[t_dim, trace_dim], label='Picks')
    onsets = hv.Overlay([original_onsets])

    #  Associated Picks
    # curves = []
    # if associated_picks is not None:
    #     min_times = np.nanmin(associated_picks, axis=0)
    #     slice_ons = (t1 <= min_times) & (min_times <= t2)
    #     ons_inds = (i_trace, slice_ons)
    #
    #     curves = [hv.Curve((pi, loc_slice), label=f"Event {i}", kdims=[t_dim], vdims=[trace_dim])
    #               for i, pi in enumerate(associated_picks[ons_inds].T)]
    # curves = hv.Overlay(curves)

    # ---- Parameters and options ---- #
    ylims = (np.nanmin(station_loc) - dloc, np.nanmax(station_loc) + dloc)
    aspect = 2
    trace_color = kwargs.get('color', 'black') if num_comp == 1 else hv.Cycle(['red', 'blue', 'green'][:num_comp])
    linewidth = kwargs.get('linewidth', 1)
    figsize = kwargs.get('fig_size', 400)
    alpha = kwargs.get('alpha', 0.15)
    fontsize = dict(labels=15, ticks=14, title=18, legend=12)
    # bokeh params
    width = int(figsize * aspect * 1.1)
    height = int(figsize * 1.1)
    hover_tooltips = ["$label", "@Time", "@Trace"]
    intv_tooltips = ["$label", ("Start", "@Time"), ("End", "@x2")]
    tools = ['hover']

    # Matplotlib opts
    backend = 'matplotlib'
    mpl_traces_opts = hv.opts.Curve(color=trace_color, show_legend=True, linewidth=linewidth, backend=backend)
    mpl_rect_opts = hv.opts.Rectangles(facecolor='blue', color='blue', show_legend=True, alpha=alpha, backend=backend)
    mpl_points_opts = hv.opts.Points(marker='|', edgecolors=None, linewidth=4, show_legend=True, s=1250,
                                     backend=backend)
    # mpl_picks_opts = hv.opts.Curve(marker='|', ms=25, linewidth=2, backend=backend)
    mpl_overlay_opts = hv.opts.Overlay(ylim=ylims, xlim=time_interval_sec, fig_size=figsize, aspect=aspect,
                                       fontsize=fontsize, legend_position='top_right', yticks=station_labels,
                                       backend=backend)

    # Bokeh opts (duplicate of Matplotlib but with Bokeh syntax)
    backend = 'bokeh'
    bkh_traces_opts = hv.opts.Curve(line_color=trace_color, line_width=linewidth, tools=tools,
                                    show_legend=True, hover_tooltips=hover_tooltips, backend=backend)
    bkh_rect_opts = hv.opts.Rectangles(fill_color='blue', line_color='blue', alpha=alpha, muted_alpha=0.025,
                                       show_legend=True, tools=tools, hover_tooltips=intv_tooltips, backend=backend)
    bkh_points_opts = hv.opts.Points(marker='dash', angle=90, size=40, line_width=4, legend_position='top_right',
                                     show_legend=True, tools=tools, hover_tooltips=hover_tooltips, backend=backend)
    # bkh_picks_opts = hv.opts.Curve(line_width=2, tools=tools, hover_tooltips=hover_tooltips, backend=backend)
    bkh_overlay_opts = hv.opts.Overlay(ylim=ylims, xlim=time_interval_sec, fontsize=fontsize, width=width,
                                       legend_position='top_right',
                                       height=height, legend_opts={"click_policy": "hide"}, yticks=station_labels,
                                       backend=backend)

    # ---- Finalizing figure object ---- #
    figure = hv.Overlay((traces.opts(mpl_traces_opts, bkh_traces_opts),
                         rects.opts(mpl_rect_opts, bkh_rect_opts),
                         onsets.opts(mpl_points_opts, bkh_points_opts)))
    figure = figure.opts(mpl_overlay_opts, bkh_overlay_opts)

    switch_plotting_backend(interactive, fig_mpl='png', fig_bokeh='auto')  # switch MPL or BKH

    # ---- Enabling interactive picking tool ---- #
    if allow_picking:
        assert interactive, "Picking is possible only when interactive plot is used"

        point_stream = hv.streams.PointDraw(data=original_onsets.columns(), num_objects=None, source=original_onsets)
        table = hv.Table(original_onsets, [t_dim, trace_dim])
        DataLink(original_onsets, table)

        bkh_table_opts = hv.opts.Table(title='', width=170, editable=True, backend=backend)
        bkh_overlay_opts = hv.opts.Overlay(width=width, height=height, backend=backend)
        bkh_layout_opts = hv.opts.Layout(merge_tools=True)

        figure = (figure + table).opts(bkh_table_opts,
                                       bkh_overlay_opts,
                                       bkh_layout_opts)
        figure = (figure, point_stream)  # point stream stores created points/picks

    return figure


def switch_plotting_backend(interactive: bool, fig_mpl='png', fig_bokeh='auto'):
    """
    Switches between `matplotlib` (`interactive=False`) and `bokeh` (`interactive=True`)
    """
    backend = 'bokeh' if interactive else 'matplotlib'
    static_format = fig_bokeh if interactive else fig_mpl
    hv.output(backend=backend, fig=static_format)
