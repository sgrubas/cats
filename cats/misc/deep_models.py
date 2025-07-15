from typing import Any, Callable, Union
from pydantic import BaseModel
import numpy as np

from cats.io import convert_dict_to_stream, convert_stream_to_dict, save_pickle, load_pickle
from cats.core.projection import FilterDetection
from cats.core.association import PickDetectedPeaks
from cats.core.utils import give_rectangles, intervals_intersection
from cats.core.utils import format_index_by_dimensions, format_interval_by_limits, give_index_slice_by_limits
from cats.core.utils import StatusKeeper
from cats.core.utils import make_default_index_if_outrange

from scipy.signal import decimate
import seisbench.models as sbm
from cats.core.plottingutils import detrend_func
from typing import Tuple
import holoviews as hv


class DeepModel(BaseModel, extra='allow'):
    dt_sec: float
    scale: float = 1.0
    name: str = "EQTransformer"
    pretrained: str = "original"
    seisbench_models_cache: dict = None
    comp_axis: int = 0
    comp_labels: list = ['E', 'N', 'Z']
    annotate_kwargs: dict = {}
    max_ndims: int = 3  # max number of dimensions for input data
    reshape_rule: Union[Callable, None] = None  # function to reshape output data

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.get_model_from_cache()
        self.scale = self.scale if self.scale > 0 else 1 / self.model.sampling_rate / self.dt_sec  # scale freqs

    def apply(self, x, verbose=True, **kwargs):
        assert x.ndim <= self.max_ndims, f"Number of data dimensions {x.ndim} exceeded max {self.max_ndims}!"

    def detect(self, x, verbose=False, **kwargs):
        return self.apply(x, verbose=verbose)

    def __mul__(self, x):
        return self.apply(x, verbose=False)

    def __pow__(self, x):
        return self.apply(x, verbose=True)

    def __matmul__(self, x):
        return self.apply(x, verbose=True)

    def get_stats(self, shape):
        """
            Creates channel names to preserve the order
        """
        stats = np.empty(shape, dtype=dict)
        assert shape[0] == 3  # first is always component!
        comp_names = self.comp_labels  # strictly, this order
        # 'ZNE' order for PhaseNet?
        model_dt = self.dt_sec * self.scale

        for ind in np.ndindex(shape):
            stats[ind] = dict(delta=model_dt,
                              station="_".join(map(str, ind[1:])),
                              channel=comp_names[ind[0]])
        return stats

    def get_model_from_cache(self):
        if self.seisbench_models_cache is None:
            fs = 1 / self.dt_sec
            model = getattr(sbm, self.name)(sampling_rate=fs).from_pretrained(self.pretrained)
        else:
            model = self.seisbench_models_cache.get((self.name, self.pretrained), None)
        return model

    def padding(self, x):
        # to avoid cases when the number of samples is lower than a minimum for 'picker'

        # 'ratio' - from 'scale sampling rate' to 'model sampling rate'
        ratio = int(1 / (self.dt_sec * self.scale * self.model.sampling_rate))
        len_x = x.shape[-1] // ratio
        min_n = self.model.in_samples
        pad_n = max(0, (min_n - len_x) * ratio + 1)

        pads = [(0, 0)] * (x.ndim - 1)  # no padding for "not time" axes
        pads.append((0, pad_n))  # zero-pad from right only

        x_pad = np.pad(x, pad_width=pads, mode='constant', constant_values=0.0)
        return x_pad

    def decimate(self, x, original_dt_sec, **kwargs):
        dec_c = int(self.dt_sec / original_dt_sec)
        if dec_c > 1:
            return decimate(x, q=dec_c, **kwargs)
        else:
            return x

    def reshape_data(self, x):
        """
            Reshapes data according to the `reshape_rule`.
            If `reshape_rule` is None, returns the input data.
        """
        if self.reshape_rule is None:
            return x
        elif callable(self.reshape_rule):
            return self.reshape_rule(x)
        else:
            raise ValueError(f"Unknown reshape rule: {self.reshape_rule}")

    @property
    def main_params(self):
        params = {kw: val for kw in type(self).model_fields.keys()
                  if ((val := getattr(self, kw, None)) is not None) and ('cache' not in kw)}
        return params

    def reset_params(self, **params):
        """
            Updates the instance with changed parameters.
        """
        kwargs = self.main_params
        kwargs.update(params)
        self.__init__(**kwargs)

    def save(self, filename):
        save_pickle(self.main_params, filename)

    @classmethod
    def load(cls, filename):
        return cls(**load_pickle(filename))


class DeepModelResult(BaseModel):
    dt_sec: float
    signal: Any = None
    likelihood: Any = None
    t0_offset: float = 0.0
    history: Any = None
    time_npts: int = None
    new_dt_sec: float = None
    new_npts: int = None
    threshold: float = None

    @staticmethod
    def base_time_func(npts, dt_sec, t0, time_interval_sec):
        if time_interval_sec is None:
            return t0 + np.arange(npts) * dt_sec
        else:
            time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (npts - 1) * dt_sec))
            time_slice = give_index_slice_by_limits(time_interval_sec, dt_sec)
            return t0 + np.arange(time_slice.start, time_slice.stop) * dt_sec

    def time(self, time_interval_sec=None):
        return DeepModelResult.base_time_func(self.time_npts, self.dt_sec, 0, time_interval_sec)

    def new_time(self, time_interval_sec=None):
        return DeepModelResult.base_time_func(self.new_npts, self.new_dt_sec, self.t0_offset, time_interval_sec)

    def plot(self, *args, **kwargs):
        return None


class DeepDetector(DeepModel):
    threshold: float = 0.1
    min_separation_sec: float = 0.5
    min_duration_sec: float = 1.0

    def apply(self, x, verbose=True, **kwargs):
        super().apply(x, verbose=verbose, **kwargs)

        history = StatusKeeper(verbose=verbose)
        shape = x.shape[:-1]

        with history(f"Applying {self.name}"):
            x_stream = convert_dict_to_stream({'data': self.padding(x),
                                               'stats': self.get_stats(shape)})
            annotated = self.model.annotate(x_stream, **self.annotate_kwargs)

        with history("Aggregating"):
            likelihood = self.aggregate_annotations(annotated, verbose)
            if likelihood.ndim < x.ndim:
                likelihood = np.expand_dims(likelihood, self.comp_axis)

        new_dt_sec = annotated[0].stats.delta
        t0_offset = (annotated[0].stats.starttime - x_stream[0].stats.starttime)  # time offset

        with history("Detecting intervals"):
            make_int = lambda n: round(n / new_dt_sec)
            detection, detected_intervals = FilterDetection(likelihood > self.threshold,
                                                            min_separation=make_int(self.min_separation_sec),
                                                            min_duration=make_int(self.min_duration_sec))

            picked_features = PickDetectedPeaks(likelihood, detected_intervals, dt=new_dt_sec, t0=t0_offset)

            detected_intervals = (detected_intervals * new_dt_sec + t0_offset) / self.scale

        history.print_total_time()

        return DeepDetectorResult(signal=x,
                                  likelihood=likelihood,
                                  threshold=self.threshold,
                                  dt_sec=self.dt_sec,
                                  time_npts=x.shape[-1],
                                  new_dt_sec=new_dt_sec,
                                  new_npts=likelihood.shape[-1],
                                  t0_offset=t0_offset,
                                  detected_intervals=detected_intervals,
                                  picked_features=picked_features,
                                  history=history)

    def aggregate_annotations(self, annotated, verbose=False):
        available = [tr.stats.channel for tr in annotated]
        detection_trace = f"{self.name}_Detection"
        noise_trace = f"{self.name}_N"
        P_trace = f"{self.name}_P"
        S_trace = f"{self.name}_S"

        if detection_trace in available:
            phase = detection_trace
            aggregated = annotated.select(channel=detection_trace)
            likelihood = convert_stream_to_dict(aggregated)['data']
        elif noise_trace in available:
            phase = noise_trace
            aggregated = annotated.select(channel=noise_trace)
            likelihood_N = convert_stream_to_dict(aggregated)['data']
            likelihood = 1 - likelihood_N  # presence of signal
        else:
            phase = P_trace + " and " + S_trace
            aggregated_P = annotated.select(channel=P_trace)
            aggregated_S = annotated.select(channel=S_trace)

            likelihood_P = convert_stream_to_dict(aggregated_P)['data']
            likelihood_S = convert_stream_to_dict(aggregated_S)['data']

            likelihood = np.maximum(likelihood_P, likelihood_S)  # max of P and S likelihoods

        if verbose:
            print(f"{phase}", end='\t')
        return likelihood


class DeepDetectorResult(DeepModelResult):
    detected_intervals: Any
    picked_features: Any

    def plot(self,
             ind: Tuple[int] = None,
             time_interval_sec: Tuple[float] = None,
             detrend_type='constant',
             **kwargs):

        if ind is None:
            ind = (0,) * (self.signal.ndim - 1)
        t_dim = hv.Dimension('Time', unit='s')
        L_dim = hv.Dimension('Likelihood')

        ind = format_index_by_dimensions(ind=ind, shape=self.signal.shape[:-1], slice_dims=0, default_ind=0)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.time_npts - 1) * self.dt_sec))
        t1, t2 = time_interval_sec

        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        i_model = give_index_slice_by_limits(time_interval_sec, self.new_dt_sec)
        inds_time = ind + (i_time,)

        time = self.time(time_interval_sec)
        model_time = self.new_time(time_interval_sec)

        signal = self.signal[inds_time]
        if detrend_type:
            detrend_type = 'linear' if detrend_type == 'linear' else 'constant'
            signal = detrend_func(signal, axis=-1, type=detrend_type)

        fig0 = hv.Curve((time, signal), kdims=[t_dim], vdims='Amplitude',
                        label='0. Input data: $x(t)$').opts(xlabel='')

        ind = make_default_index_if_outrange(ind, self.likelihood.shape[:-1], default_ind_value=0)
        inds_model = ind + (i_model,)

        likelihood = np.nan_to_num(self.likelihood[inds_model],
                                   posinf=1e8, neginf=-1e8)  # POSSIBLE `NAN` AND `INF` VALUES!
        likelihood_fig = hv.Curve((model_time, likelihood), kdims=[t_dim], vdims=L_dim)

        # Peaks
        P = self.picked_features[ind]
        P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
        peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
        peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim).opts(marker='D', color='r')

        # Intervals
        intervals = intervals_intersection(self.detected_intervals[ind], (t1, t2))

        interv_height = (np.max(likelihood) / 2) * 1.1
        rectangles = give_rectangles([intervals], [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles,
                                      kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                             linewidth=0,
                                                                             alpha=0.2)
        snr_level_fig = hv.HLine(self.threshold, kdims=[t_dim, L_dim]).opts(color='k', linestyle='--', alpha=0.7)

        last_figs = [intervals_fig, snr_level_fig, likelihood_fig, peaks_fig]
        fig1 = hv.Overlay(last_figs, label=r'1. Likelihood and Detection: $\mathcal{L}(t)$ and $\tilde{\alpha}(t)$')

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250

        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, show_frame=True,
                                   xlim=time_interval_sec)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        fig = (fig0 + fig1).cols(1).opts(layout_opts, curve_opts)
        return fig

    def plot_multi(self,
                   inds,
                   share_time_labels=True,
                   share_vertical_axes=True,
                   hspace=None,
                   vspace=None,
                   **kwargs):

        figs = [self.plot(ind, **kwargs) for ind in inds]
        nrows = len(figs[0])
        ncols = len(figs)

        if share_time_labels:
            for i, figs_i in enumerate(zip(*figs)):
                if i < nrows - 1:
                    for fi in figs_i:
                        fi = fi.opts(xaxis='bare')

        if share_vertical_axes:
            for i, fig_i in enumerate(figs):
                if i > 0:
                    fig_i = fig_i.opts(hv.opts.Curve(yaxis='bare'))

        fig = hv.Layout(figs).cols(nrows).opts(**figs[0].opts.get().kwargs)

        vspace = vspace or 0.050 + 0.02 * (not share_time_labels)
        hspace = hspace or 0.025 + 0.10 * sum([not share_vertical_axes])

        fig = fig.opts(shared_axes=share_vertical_axes, vspace=vspace, hspace=hspace, transpose=True)

        return fig


class DeepPicker(DeepModel):
    """
        Can be applied only on a single event (takes MAX likelihood of picks)
    """
    threshold_P: float = 0.0
    threshold_S: float = 0.0

    def apply(self, x, verbose=True, **kwargs):
        super().apply(x, verbose=verbose, **kwargs)

        history = StatusKeeper(verbose=verbose)
        shape = x.shape[:-1]
        with history(f"Applying {self.name}"):
            x_stream = convert_dict_to_stream({'data': self.padding(x),
                                               'stats': self.get_stats(shape)})
            annotated = self.model.annotate(x_stream, **self.annotate_kwargs)

        new_dt_sec = annotated[0].stats.delta
        t0_offset = (annotated[0].stats.starttime - x_stream[0].stats.starttime)  # time offset

        with history("Getting picks"):
            Lp, Ls, P_picks, S_picks = self.get_picks(annotated, verbose)

            picks = np.stack([P_picks, S_picks], axis=-1)
            picks = (picks * new_dt_sec + t0_offset) / self.scale
            P_picks = picks[..., 0]
            S_picks = picks[..., 1]

        history.print_total_time()

        return DeepPickerResult(dt_sec=new_dt_sec,
                                threshold_P=self.threshold_P,
                                threshold_S=self.threshold_S,
                                likelihood_P=Lp,
                                likelihood_S=Ls,
                                P_picks=np.expand_dims(P_picks, self.comp_axis),
                                S_picks=np.expand_dims(S_picks, self.comp_axis),
                                t0_offset=t0_offset,
                                picks=np.expand_dims(picks, self.comp_axis),
                                history=history)

    def get_picks(self, annotated, verbose=False):

        aggregated_P = annotated.select(channel=f"{self.name}_P")
        aggregated_S = annotated.select(channel=f"{self.name}_S")

        likelihood_P = convert_stream_to_dict(aggregated_P)['data']
        likelihood_P[likelihood_P < self.threshold_P] = 0.0
        likelihood_S = convert_stream_to_dict(aggregated_S)['data']
        likelihood_S[likelihood_S < self.threshold_S] = 0.0

        P_picks = np.argmax(likelihood_P, axis=-1) if np.count_nonzero(likelihood_P) > 0 else np.nan
        S_picks = np.argmax(likelihood_S, axis=-1) if np.count_nonzero(likelihood_S) > 0 else np.nan

        return likelihood_P, likelihood_S, P_picks, S_picks


class DeepPickerResult(DeepModelResult):
    t0_offset: float
    threshold_P: float
    threshold_S: float
    likelihood_P: Any
    likelihood_S: Any
    P_picks: Any
    S_picks: Any
    picks: Any


class DeepDenoiser(DeepModel):
    name: str = "DeepDenoiser"

    def apply(self, x, verbose=True, **kwargs):
        super().apply(x, verbose=verbose, **kwargs)

        history = StatusKeeper(verbose=verbose)
        shape = x.shape[:-1]
        with history(f"Applying {self.name}"):
            x_stream = convert_dict_to_stream({'data': self.padding(x),
                                               'stats': self.get_stats(shape)})
            annotated = self.model.annotate(x_stream, **self.annotate_kwargs)

        signal_denoised = convert_stream_to_dict(annotated)['data']
        signal_denoised = self.reshape_data(signal_denoised)

        new_dt_sec = annotated[0].stats.delta
        t0_offset = (annotated[0].stats.starttime - x_stream[0].stats.starttime)  # time offset

        return DeepDenoiserResult(dt_sec=self.dt_sec,
                                  new_dt_sec=new_dt_sec,
                                  signal=x,
                                  signal_denoised=signal_denoised,
                                  t0_offset=t0_offset,
                                  time_npts=x.shape[-1],
                                  new_npts=signal_denoised.shape[-1],
                                  history=history,)

    def denoise(self, x, verbose=True):
        return self.apply(x, verbose=verbose)


class DeepDenoiserResult(DeepModelResult):
    signal_denoised: Any
