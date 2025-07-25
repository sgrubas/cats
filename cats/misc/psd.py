"""
    Implements detector API for Power-Spectral-Density detector (Yoones & van der Baan, GJI, 2015)
"""

from pydantic import BaseModel, Field, Extra
from typing import Callable, Union, Tuple, List, Any
from pathlib import Path
from tqdm.notebook import tqdm

import numpy as np
import holoviews as hv
from cats.core.projection import FilterDetection
from cats.core.timefrequency import STFTOperator
from cats.core.association import PickDetectedPeaks
from cats.core.utils import format_index_by_dimensions, format_interval_by_limits, give_index_slice_by_limits
from cats.core.utils import aggregate_array_by_axis_and_func, cast_to_bool_dict, del_vals_by_keys, give_rectangles
from cats.core.utils import give_nonzero_limits, complex_abs_square, intervals_intersection
from cats.core.utils import StatusKeeper, make_default_index_if_outrange
from cats.baseclass import CATSBase
from cats.detection import CATSDetector, CATSDetectionResult
from cats.denoising import CATSDenoiser, CATSDenoisingResult
from cats.io.utils import read_data, save_pickle, load_pickle
from cats.core.plottingutils import detrend_func


# --------------- PSD DETECTOR API --------------- #

class PSDDetector(BaseModel, extra="allow"):
    # STFT params
    dt_sec: float
    min_duration_sec: float
    min_separation_sec: float
    threshold: float  # threshold ~ 0.065

    stft_window_type: str = 'hann'
    stft_window_sec: float = 0.5
    stft_overlap: float = Field(0.75, ge=0.0, lt=1.0)
    stft_backend: str = 'ssqueezepy'
    stft_kwargs: dict = {}
    stft_nfft: int = -1
    freq_bandpass_Hz: Union[Tuple[float, float], Any, None] = None
    characteristic: str = 'abs'

    aggregate_axis_for_likelihood: Union[int, Tuple[int], None] = None
    aggregate_func_for_likelihood: Union[str, Callable, None] = None

    psd_noise_mean: Any = None
    psd_noise_std: Any = None

    name: str = "PSD"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_params()

    def _set_params(self):
        self.STFT = STFTOperator(window_specs=(self.stft_window_type, self.stft_window_sec), overlap=self.stft_overlap,
                                 dt_sec=self.dt_sec, nfft=self.stft_nfft, backend=self.stft_backend, **self.stft_kwargs)
        self.stft_overlap_len = self.STFT.noverlap
        self.stft_overlap_sec = self.stft_overlap_len * self.dt_sec
        self.stft_window = self.STFT.window
        self.stft_window_len = len(self.stft_window)
        self.stft_window_sec = (self.stft_window_len - 1) * self.dt_sec
        self.stft_frequency = self.STFT.f
        self.stft_df = self.STFT.df
        self.stft_nfft = self.STFT.nfft
        self.stft_hop_len = self.STFT.hop
        self.stft_hop_sec = self.dt_sec * self.stft_hop_len

        self.freq_bandpass_Hz = format_interval_by_limits(self.freq_bandpass_Hz,
                                                          (self.stft_frequency[0], self.stft_frequency[-1]))
        self.freq_bandpass_Hz = (min(self.freq_bandpass_Hz), max(self.freq_bandpass_Hz))
        self.freq_bandpass_slice = give_index_slice_by_limits(self.freq_bandpass_Hz, self.stft_df)
        self.freq_bandpass_len = (self.freq_bandpass_slice.start, self.freq_bandpass_slice.stop)

        self.min_duration_len = max(round(self.min_duration_sec / self.stft_hop_sec), 1)
        self.min_separation_len = max(round(self.min_separation_sec / self.stft_hop_sec), 1)

        self.ch_functions = {'abs': np.abs, 'square': np.square}
        self.ch_func = self.ch_functions[self.characteristic]

        self.time_edge = int(self.stft_window_len // 2 / self.stft_hop_len)

    @property
    def main_params(self):
        return {kw: val for kw in self.model_fields.keys()
                if (val := getattr(self, kw, None)) is not None}

    @classmethod
    def from_result(cls, PSDResult):
        return cls(**PSDResult.main_params)

    def reset_params(self, **params):
        """
            Updates the instance with changed parameters.
        """
        kwargs = self.main_params
        kwargs.update(params)
        self.__init__(**kwargs)

    def set_noise_model(self, noisy_pieces):
        psd_noise_models = np.concatenate([complex_abs_square(self.STFT * xi) for xi in noisy_pieces], axis=-1)
        self.psd_noise_mean = psd_noise_models.mean(axis=-1, keepdims=True)
        self.psd_noise_std = psd_noise_models.std(axis=-1, keepdims=True)
        del psd_noise_models

    def split_data_by_intervals(self, data, intervals_sec):
        intervals_slices = tuple(give_index_slice_by_limits(intv, self.dt_sec) for intv in intervals_sec)
        split_data = tuple(data[..., inds] for inds in intervals_slices)
        return split_data

    def set_noise_model_by_intervals(self, data, intervals_sec):
        self.set_noise_model(self.split_data_by_intervals(data, intervals_sec))

    def _detect(self, x, verbose=True, full_info=False):
        assert self.psd_noise_mean is not None, f"Noise model must be set priorly, use `.set_noise_model(...)`"

        full_info = self.parse_info_dict(full_info)

        result = dict.fromkeys(full_info.keys())
        result['signal'] = x if full_info['signal'] else None

        history = StatusKeeper(verbose=verbose)

        time = np.arange(x.shape[-1]) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(len(time))

        with history(current_process='STFT'):
            result['coefficients'] = self.STFT * x
            result['spectrogram'] = np.abs(result['coefficients'])**2

        del_vals_by_keys(result, full_info, ['coefficients'])

        # bandpass + time edges of STFT result
        bandpass_slice = (..., self.freq_bandpass_slice, slice(None))

        with history(current_process='Trimming'):
            result['spectrogram_SNR_trimmed'] = np.zeros_like(result['spectrogram'])
            result['spectrogram_SNR_trimmed'][bandpass_slice] = \
                (result['spectrogram'][bandpass_slice] - self.psd_noise_mean[bandpass_slice])
            result['spectrogram_SNR_trimmed'][bandpass_slice] /= self.psd_noise_std[bandpass_slice] + 1e-16
            del_vals_by_keys(result, full_info, ['spectrogram'])
            result['spectrogram_SNR_trimmed'] = \
                np.where(result['spectrogram_SNR_trimmed'] >= 1.0,
                         self.ch_func(result['spectrogram_SNR_trimmed']), 0.0)

            # Removing spiky high-energy edges
            result['spectrogram_SNR_trimmed'][..., :self.time_edge] = 0.0
            result['spectrogram_SNR_trimmed'][..., -self.time_edge - 1:] = 0.0

        with history(current_process='Likelihood'):
            # Aggregation
            result['likelihood'] = result['spectrogram_SNR_trimmed'].mean(axis=-2)
            result['likelihood'] = aggregate_array_by_axis_and_func(result['likelihood'],
                                                                    self.aggregate_axis_for_likelihood,
                                                                    self.aggregate_func_for_likelihood,
                                                                    min_last_dims=1)

        del_vals_by_keys(result, full_info, ['spectrogram_SNR_trimmed'])

        with history(current_process='Detecting intervals'):
            result['detection'], result['detected_intervals'] = \
                FilterDetection(result['likelihood'] > self.threshold,
                                min_separation=self.min_separation_len, min_duration=self.min_duration_len)

            result['picked_features'] = PickDetectedPeaks(result['likelihood'], result['detected_intervals'],
                                                          dt=self.stft_hop_sec, t0=stft_time[0])

            result['detected_intervals'] = result['detected_intervals'] * self.stft_hop_sec + stft_time[0]

        del_vals_by_keys(result, full_info, ['likelihood', 'detection', 'detected_intervals', 'picked_features'])

        history.print_total_time()

        from_full_info = {kw: result.get(kw, None) for kw in full_info}
        kwargs = {**from_full_info,
                  "noise_mean": self.psd_noise_mean,
                  "noise_std": self.psd_noise_std,
                  "dt_sec": self.dt_sec,
                  "tf_dt_sec": self.stft_hop_sec,
                  "tf_t0_sec": stft_time[0],
                  "time_npts": x.shape[-1],
                  "tf_time_npts": len(stft_time),
                  "frequencies": self.stft_frequency,
                  "threshold": self.threshold,
                  "history": history,
                  "aggregate_axis_for_likelihood": self.aggregate_axis_for_likelihood}

        return PSDDetectionResult(**kwargs)

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
                            - "coefficients" - STFT coefficients
                            - "spectrogram" - absolute value of `coefficients`
                            - "noise_mean" - average of noise model
                            - "noise_std" - average std of noise level
                            - "spectrogram_SNR_trimmed" - `spectrogram / noise_std` trimmed by `noise_threshold`
                            - "likelihood" - projected `spectrogram_SNR_clustered`
                            - "detection" - binary classification [noise / signal], always returned.
                            - "detected_intervals" - detected intervals (start, end) in seconds (always returned)
                            - "picked_features" - features in intervals [onset, peak likelihood] (always returned)
        """
        n_chunks = self.split_data_by_memory(x, full_info=full_info, to_file=False)
        single_chunk = n_chunks > 1
        data_chunks = np.array_split(x, n_chunks, axis=-1)
        results = []
        for dc in tqdm(data_chunks, desc='Data chunks', display=single_chunk):
            results.append(self._detect(dc, verbose=verbose, full_info=full_info))
        return PSDDetectionResult.concatenate(*results)

    def apply(self, x, verbose, full_info):
        return self.detect(x, verbose=verbose, full_info=full_info)

    def __mul__(self, x):
        return self.apply(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.apply(x, verbose=True, full_info='qc')

    def __matmul__(self, x):
        return self.apply(x, verbose=True, full_info=True)

    @staticmethod
    def get_qc_keys():
        return ["signal", "spectrogram", "spectrogram_SNR_trimmed",
                "likelihood", "detected_intervals", "picked_features"]

    @staticmethod
    def parse_info_dict(full_info):
        info_keys = ["signal",
                     "coefficients",
                     "spectrogram",
                     "spectrogram_SNR_trimmed",
                     "likelihood",
                     "detection",
                     "detected_intervals",
                     "picked_features"]

        if isinstance(full_info, str) and \
           (full_info in ['plot', 'plotting', 'plt', 'qc', 'main']):  # only those needed for plotting step-by-step
            full_info = PSDDetector.get_qc_keys()
        full_info = cast_to_bool_dict(full_info, info_keys)

        full_info["detected_intervals"] = True  # default saved result
        full_info["picked_features"] = True     # default saved result

        return full_info

    def memory_usage_estimate(self, x, /, full_info=False):
        stft_time_len = len(self.STFT.forward_time_axis(x.shape[-1]))
        stft_shape = x.shape[:-1] + (len(self.stft_frequency), stft_time_len)
        stft_size = np.prod(stft_shape)

        noise_shape = x.shape[:-1] + (len(self.stft_frequency),)
        noise_size = np.prod(noise_shape)

        likelihood_shape = x.shape[:-1] + (stft_time_len,)
        likelihood_size = np.prod(likelihood_shape)

        x_bytes = x.nbytes
        precision_order = x_bytes / x.size

        aggregated_axis_len = x.shape[ax] if (ax := self.aggregate_axis_for_likelihood) is not None else 1
        n, mod = divmod(stft_time_len, self.min_duration_len + self.min_separation_len)
        num_intervals = n + mod // self.min_duration_len  # upper bound estimate
        intervals_size = 2 * np.prod(x.shape[:-1]) / aggregated_axis_len * num_intervals

        memory_usage_bytes = {
            "frequencies":           8. * stft_shape[-2],                     # float64 always
            "signal":                   1. * x_bytes,                            # float / int
            "coefficients":             2. * precision_order * stft_size,        # complex
            "spectrogram":              1. * precision_order * stft_size,        # float
            "spectrogram_SNR_trimmed":  1. * precision_order * stft_size,        # float
            "noise_mean":               1. * precision_order * noise_size,       # float
            "noise_std":                1. * precision_order * noise_size,       # float
            "likelihood":               1. * precision_order * likelihood_size,  # float
            "detection":                1. * likelihood_size,                    # bool always
            "detected_intervals":       8. * intervals_size,                     # ~ float64, upper bound
            "picked_features":          8. * intervals_size,                     # ~ float64, upper bound
        }

        used_together = [('coefficients', 'spectrogram'),
                         ('spectrogram', 'spectrogram_SNR_trimmed'),
                         ('spectrogram_SNR_trimmed', 'likelihood'),
                         ('likelihood', 'detection', 'detected_intervals', 'picked_features')]
        base_info = ["signal", "frequencies", "noise_mean", "noise_std"]
        full_info = self.parse_info_dict(full_info)

        return CATSBase.memory_info(memory_usage_bytes, used_together, base_info, full_info)

    def split_data_by_memory(self, x, /, full_info, to_file):
        memory_info = self.memory_usage_estimate(x, full_info=full_info)
        return CATSBase.memory_chunks(memory_info, to_file)

    def detect_to_file(self, x, /, path_destination, verbose=False, full_info=False, compress=False):
        CATSDetector.basefunc_detect_to_file(self, x, path_destination, verbose, full_info, compress)

    def detect_on_files(self, data_folder, data_format, noise_model_intervals_sec=None, result_folder=None,
                        verbose=False, full_info=False, compress=False):
        folder = Path(data_folder)
        data_format = "." * ("." not in data_format) + data_format
        result_folder = Path(result_folder) if (result_folder is not None) else folder
        files = [fi for fi in folder.rglob("*" + data_format)]

        for fi in (pbar := tqdm(files, desc="Files")):
            pbar.set_postfix({"File": fi})
            rel_folder = fi.relative_to(folder).parent
            save_path = result_folder / rel_folder / Path(str(fi.name).replace(data_format, "_detection"))
            x = read_data(fi)['data']
            if noise_model_intervals_sec is not None:
                self.set_noise_model_by_intervals(x, noise_model_intervals_sec)
            self.detect_to_file(x, save_path, verbose=verbose, full_info=full_info, compress=compress)
            del x

    def save(self, filename):
        save_pickle(self.main_params, filename)

    @classmethod
    def load(cls, filename):
        loaded = load_pickle(filename)
        if isinstance(loaded, cls):
            loaded = loaded.main_params
        return cls(**loaded)


class PSDDetectionResult(CATSDetectionResult):
    noise_mean: Any = None
    threshold: float = None
    spectrogram_SNR_trimmed: Any = None

    def plot(self,
             ind=None,
             time_interval_sec=None,
             SNR_spectrograms: bool = True,
             detrend_type: str = 'constant',
             **kwargs):

        if ind is None:
            ind = (0,) * (self.signal.ndim - 1)
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        A_dim = hv.Dimension('Amplitude')
        L_dim = hv.Dimension('Likelihood')

        ind = format_index_by_dimensions(ind=ind, shape=self.signal.shape[:-1], slice_dims=0, default_ind=0)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.time_npts - 1) * self.dt_sec))
        t1, t2 = time_interval_sec

        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        i_stft = give_index_slice_by_limits(time_interval_sec, self.tf_dt_sec)

        inds_time = ind + (i_time,)
        inds_stft = ind + (i_stft,)
        inds_Stft = ind + (slice(None), i_stft)

        PSD = self.spectrogram[inds_Stft]
        SNR = self.spectrogram_SNR_trimmed[inds_Stft]

        PSD_clims = give_nonzero_limits(PSD, initials=(1e-1, 1e1))
        SNR_clims = give_nonzero_limits(SNR, initials=(1e-1, 1e1))

        time = self.time(time_interval_sec)
        stft_time = self.tf_time(time_interval_sec)

        signal = self.signal[inds_time]
        if detrend_type:
            detrend_type = 'linear' if detrend_type == 'linear' else 'constant'
            signal = detrend_func(signal, axis=-1, type=detrend_type)

        fig0 = hv.Curve((time, signal), kdims=[t_dim], vdims=A_dim,
                        label='0. Input data: $x(t)$').opts(xlabel='', linewidth=0.5)
        fig1 = hv.Image((stft_time, self.frequencies, PSD), kdims=[t_dim, f_dim],
                        label='1. Power spectrogram: $|X(t,f)|^2$').opts(clim=PSD_clims, clabel='Power')
        fig2 = hv.Image((stft_time, self.frequencies, SNR), kdims=[t_dim, f_dim],
                        label='2. Trimmed SNR spectrogram: $T(t,f)$').opts(clim=SNR_clims)

        ind = make_default_index_if_outrange(ind, self.likelihood.shape[:-1], default_ind_value=0)
        inds_stft = ind + (i_stft,)

        likelihood = np.nan_to_num(self.likelihood[inds_stft],
                                   posinf=1e8, neginf=-1e8)  # POSSIBLE `NAN` AND `INF` VALUES!
        likelihood_fig = hv.Curve((stft_time, likelihood),
                                  kdims=[t_dim], vdims=L_dim)

        P = self.picked_features[ind]
        P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
        peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
        peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim).opts(marker='D', color='r')

        intervals = intervals_intersection(self.detected_intervals[ind], (t1, t2))

        interv_height = (np.max(likelihood) / 2) * 1.1
        rectangles = give_rectangles([intervals], [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles,
                                      kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                             linewidth=0,
                                                                             alpha=0.2)
        snr_level_fig = hv.HLine(self.threshold, kdims=[t_dim, L_dim]).opts(color='k', linestyle='--', alpha=0.7)

        last_figs = [intervals_fig, snr_level_fig, likelihood_fig, peaks_fig]

        fig3 = hv.Overlay(last_figs, label=r'3. Likelihood and Detection: $\mathcal{L}(t)$ and $\tilde{\alpha}(t)$')

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250
        cmap = 'viridis'
        xlim = time_interval_sec
        ylim = (max(1e-1, self.frequencies[1]), None)
        spectr_opts = hv.opts.Image(cmap=cmap, colorbar=True,  logy=True, logz=True, xlim=xlim, ylim=ylim,
                                    cbar_width=0.02, xlabel='', clabel='', aspect=2,
                                    fig_size=figsize, fontsize=fontsize)
        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=xlim,
                                   show_frame=True, show_legend=False)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        figs = [fig0, fig1, fig2, fig3]
        fig = hv.Layout(figs).cols(1).opts(layout_opts, curve_opts, spectr_opts)
        return fig

    def plot_multi(self,
                   inds,
                   share_time_labels=True,
                   share_vertical_axes=True,
                   share_spectrogram_cmaps=True,
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
                    fig_i = fig_i.opts(hv.opts.Image(yaxis='bare'),
                                       hv.opts.Curve(yaxis='bare'))
        if share_spectrogram_cmaps:
            for i, figs_i in enumerate(zip(*figs)):
                if (0 < i < nrows - 1):
                    clims = [fi.opts.get().kwargs['clim'] for fi in figs_i]
                    if all([(cl is not None) for i, cl in np.ndenumerate(clims)]):
                        clim = (np.min(clims), np.max(clims))
                        for fi in figs_i:
                            fi = fi.opts(clim=clim)

            for i, fig_i in enumerate(figs):
                if i < ncols - 1:
                    fig_i = fig_i.opts(hv.opts.Image(colorbar=False))

        fig = hv.Layout(figs).cols(nrows).opts(**figs[0].opts.get().kwargs)

        vspace = vspace or 0.05 + 0.02 * (not share_time_labels)
        hspace = hspace or 0.025 + 0.1 * sum([not share_vertical_axes, not share_spectrogram_cmaps])

        fig = fig.opts(shared_axes=share_vertical_axes, vspace=vspace, hspace=hspace, transpose=True)

        return fig

    def append(self, other):

        concat_attrs = ["signal",
                        "coefficients",
                        "spectrogram",
                        "spectrogram_SNR_trimmed",
                        "likelihood",
                        "detection"]

        for name in concat_attrs:
            self._concat(other, name, -1)

        stft_t0 = self.tf_dt_sec * (self.tf_time_npts - 1)

        self._concat(other, "detected_intervals", -2, stft_t0)
        self._concat(other, "picked_features", -2, stft_t0, (..., 0))

        self.time_npts += other.time_npts
        self.tf_time_npts += other.tf_time_npts

        self.history.merge(other.history)

        assert self.threshold == other.threshold
        assert self.dt_sec == other.dt_sec
        assert self.tf_dt_sec == other.tf_dt_sec
        assert np.all(self.frequencies == other.frequencies)
        assert np.all(self.noise_mean == other.noise_mean)
        assert np.all(self.noise_std == other.noise_std)


# --------------- Denoising API --------------- #


class PSDDenoiser(PSDDetector):
    background_weight: float = 0.0

    def _denoise(self, x, verbose=True, full_info=False):
        assert self.psd_noise_mean is not None, f"Noise model must be set priorly, use `.set_noise_model(...)`"

        full_info = self.parse_info_dict(full_info)

        result = dict.fromkeys(full_info.keys())
        result['signal'] = x if full_info['signal'] else None

        history = StatusKeeper(verbose=verbose)

        time = np.arange(x.shape[-1]) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(len(time))

        with history(current_process='STFT'):
            result['coefficients'] = self.STFT * x
            result['spectrogram'] = np.abs(result['coefficients']) ** 2

        # bandpass + time edges of STFT result
        bandpass_slice = (..., self.freq_bandpass_slice, slice(None))

        with history(current_process='Trimming'):
            result['spectrogram_SNR_trimmed'] = np.zeros_like(result['spectrogram'])
            result['spectrogram_SNR_trimmed'][bandpass_slice] = \
                result['spectrogram'][bandpass_slice] - self.psd_noise_mean[bandpass_slice]
            result['spectrogram_SNR_trimmed'][bandpass_slice] /= self.psd_noise_std[bandpass_slice] + 1e-16
            del_vals_by_keys(result, full_info, ['spectrogram'])
            result['spectrogram_SNR_trimmed'] = \
                np.where(result['spectrogram_SNR_trimmed'] >= self.threshold,
                         self.ch_func(result['spectrogram_SNR_trimmed']), 0.0)

            # Removing spiky high-energy edges
            result['spectrogram_SNR_trimmed'][..., :self.time_edge] = 0.0
            result['spectrogram_SNR_trimmed'][..., -self.time_edge - 1:] = 0.0

        with history(current_process='Likelihood'):
            # Aggregation
            result['likelihood'] = result['spectrogram_SNR_trimmed'].mean(axis=-2)
            result['likelihood'] = aggregate_array_by_axis_and_func(result['likelihood'],
                                                                    self.aggregate_axis_for_likelihood,
                                                                    self.aggregate_func_for_likelihood,
                                                                    min_last_dims=1)

        with history(current_process='Detecting intervals'):
            result['detection'], result['detected_intervals'] = \
                FilterDetection(result['likelihood'] > self.threshold,
                                min_separation=self.min_separation_len, min_duration=self.min_duration_len)

            result['picked_features'] = PickDetectedPeaks(result['likelihood'], result['detected_intervals'],
                                                          dt=self.stft_hop_sec, t0=stft_time[0])

            result['detected_intervals'] = result['detected_intervals'] * self.stft_hop_sec + stft_time[0]

        del_vals_by_keys(result, full_info, ['likelihood', 'detection', 'detected_intervals', 'picked_features'])

        with history(current_process='Inverse STFT:'):
            weights = result['spectrogram_SNR_trimmed'] > 0
            if self.background_weight:
                weights = np.where(weights, 1.0, self.background_weight)

            weighted_coefficients = result['coefficients'] * weights

            result['signal_denoised'] = (self.STFT / weighted_coefficients)[..., :x.shape[-1]]
            del weights, weighted_coefficients

        del_vals_by_keys(result, full_info, ['coefficients', 'spectrogram_SNR_trimmed'])

        history.print_total_time()

        from_full_info = {kw: result.get(kw, None) for kw in full_info}

        kwargs = {**from_full_info,
                  "noise_mean": self.psd_noise_mean,
                  "noise_std": self.psd_noise_std,
                  "dt_sec": self.dt_sec,
                  "tf_dt_sec": self.stft_hop_sec,
                  "tf_t0_sec": stft_time[0],
                  "time_npts": x.shape[-1],
                  "tf_time_npts": len(stft_time),
                  "frequencies": self.stft_frequency,
                  "threshold": self.threshold,
                  "history": history,
                  "aggregate_axis_for_likelihood": self.aggregate_axis_for_likelihood}

        return PSDDenoisingResult(**kwargs)

    def denoise(self, x: np.ndarray,
                /,
                verbose: bool = False,
                full_info: Union[bool, str, List[str]] = False):

        n_chunks = self.split_data_by_memory(x, full_info=full_info, to_file=False)
        single_chunk = n_chunks > 1
        data_chunks = np.array_split(x, n_chunks, axis=-1)
        results = []
        for dc in tqdm(data_chunks, desc='Data chunks', display=single_chunk):
            results.append(self._denoise(dc, verbose=verbose, full_info=full_info))
        return PSDDenoisingResult.concatenate(*results)

    def __mul__(self, x):
        return self.denoise(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.denoise(x, verbose=True, full_info='qc')

    def __matmul__(self, x):
        return self.denoise(x, verbose=True, full_info=True)

    @staticmethod
    def get_qc_keys():
        return ["signal", "spectrogram", "spectrogram_SNR_trimmed", "likelihood", "signal_denoised"]

    @staticmethod
    def parse_info_dict(full_info):
        info_keys = ["signal",
                     "coefficients",
                     "spectrogram",
                     "spectrogram_SNR_trimmed",
                     "likelihood",
                     "detection",
                     "detected_intervals",
                     "picked_features",
                     "signal_denoised"]

        if isinstance(full_info, str) and \
                (full_info in ['plot', 'plotting', 'plt', 'qc', 'main']):  # only those needed for plotting step-by-step
            full_info = PSDDenoiser.get_qc_keys()
        full_info = cast_to_bool_dict(full_info, info_keys)

        full_info["signal_denoised"] = True  # default saved result

        return full_info

    def memory_usage_estimate(self, x, /, full_info=False):
        stft_time_len = len(self.STFT.forward_time_axis(x.shape[-1]))
        stft_shape = x.shape[:-1] + (len(self.stft_frequency), stft_time_len)
        stft_size = np.prod(stft_shape)

        noise_shape = x.shape[:-1] + (len(self.stft_frequency),)
        noise_size = np.prod(noise_shape)

        likelihood_shape = x.shape[:-1] + (stft_time_len,)
        likelihood_size = np.prod(likelihood_shape)

        x_bytes = x.nbytes
        precision_order = x_bytes / x.size

        aggregated_axis_len = x.shape[ax] if (ax := self.aggregate_axis_for_likelihood) is not None else 1
        n, mod = divmod(stft_time_len, self.min_duration_len + self.min_separation_len)
        num_intervals = n + mod // self.min_duration_len  # upper bound estimate
        intervals_size = 2 * np.prod(x.shape[:-1]) / aggregated_axis_len * num_intervals

        memory_usage_bytes = {
            "frequencies": 8. * stft_shape[-2],  # float64 always
            "signal": 1. * x_bytes,  # float / int
            "coefficients": 2. * precision_order * stft_size,  # complex
            "spectrogram": 1. * precision_order * stft_size,  # float
            "spectrogram_SNR_trimmed": 1. * precision_order * stft_size,  # float
            "noise_mean": 1. * precision_order * noise_size,  # float
            "noise_std": 1. * precision_order * noise_size,  # float
            "likelihood": 1. * precision_order * likelihood_size,  # float
            "detection": 1. * likelihood_size,  # bool always
            "detected_intervals": 8. * intervals_size,  # ~ float64, upper bound
            "picked_features": 8. * intervals_size,  # ~ float64, upper bound
            "signal_denoised": 1. * x_bytes,  # float / int
        }

        used_together = [('coefficients', 'spectrogram'),
                         ('coefficients', 'spectrogram', 'spectrogram_SNR_trimmed'),
                         ('coefficients', 'spectrogram_SNR_trimmed', 'likelihood'),
                         ('coefficients', 'spectrogram_SNR_trimmed', 'likelihood', 'detection',
                          'detected_intervals', 'picked_features'),
                         ('coefficients', 'spectrogram_SNR_trimmed', 'signal_denoised')
                         ]
        base_info = ["signal", "frequencies", "noise_mean", "noise_std"]
        full_info = self.parse_info_dict(full_info)

        return CATSBase.memory_info(memory_usage_bytes, used_together, base_info, full_info)

    def denoise_to_file(self, x, /, path_destination, verbose=False, full_info=False, compress=False):
        CATSDenoiser.basefunc_denoise_to_file(self, x, path_destination, verbose, full_info, compress)

    def denoise_on_files(self, data_folder, data_format, noise_model_intervals_sec=None, result_folder=None,
                         verbose=False, full_info=False, compress=False):
        folder = Path(data_folder)
        data_format = "." * ("." not in data_format) + data_format
        result_folder = Path(result_folder) if (result_folder is not None) else folder
        files = [fi for fi in folder.rglob("*" + data_format)]

        for fi in (pbar := tqdm(files, desc="Files")):
            pbar.set_postfix({"File": fi})
            rel_folder = fi.relative_to(folder).parent
            save_path = result_folder / rel_folder / Path(str(fi.name).replace(data_format, "_detection"))
            x = read_data(fi)['data']
            if noise_model_intervals_sec is not None:
                self.set_noise_model_by_intervals(x, noise_model_intervals_sec)
            self.denoise_to_file(x, save_path, verbose=verbose, full_info=full_info, compress=compress)
            del x


class PSDDenoisingResult(CATSDenoisingResult):
    noise_mean: Any = None
    signal_denoised: Any = None
    threshold: float = None

    def plot(self,
             ind=None,
             time_interval_sec=None,
             SNR_spectrograms: bool = False,
             weighted_coefficients: bool = False,
             intervals=False,
             picks=False,
             detrend_type='constant',
             interactive=False):

        if ind is None:
            ind = (0,) * (self.signal.ndim - 1)
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        A_dim = hv.Dimension('Amplitude')
        L_dim = hv.Dimension('Likelihood')

        ind = format_index_by_dimensions(ind=ind, shape=self.signal.shape[:-1], slice_dims=0, default_ind=0)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.time_npts - 1) * self.dt_sec))
        t1, t2 = time_interval_sec

        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        i_stft = give_index_slice_by_limits(time_interval_sec, self.tf_dt_sec)

        inds_time = ind + (i_time,)
        inds_Stft = ind + (slice(None), i_stft)

        PSD = self.spectrogram[inds_Stft]
        SNR = self.spectrogram_SNR_trimmed[inds_Stft]

        PSD_clims = give_nonzero_limits(PSD, initials=(1e-1, 1e1))

        if not SNR_spectrograms:
            SNR = PSD * (SNR > 0)
            label_trimmed = 'power spectrogram: $|X(t,f)|^2 \cdot (T(t,f) > 0)$'
            SNR_clims = PSD_clims
        else:
            label_trimmed = 'SNR spectrogram: $T(t,f)$'
            SNR_clims = give_nonzero_limits(SNR, initials=(1e-1, 1e1))

        time = self.time(time_interval_sec)
        stft_time = self.tf_time(time_interval_sec)

        signal = self.signal[inds_time]
        if detrend_type:
            detrend_type = 'linear' if detrend_type == 'linear' else 'constant'
            signal = detrend_func(signal, axis=-1, type=detrend_type)

        fig0 = hv.Curve((time, signal), kdims=[t_dim], vdims=A_dim,
                        label='0. Input data: $x(t)$').opts(xlabel='', linewidth=0.5)
        fig1 = hv.Image((stft_time, self.frequencies, PSD), kdims=[t_dim, f_dim],
                        label='1. Power spectrogram: $|X(t,f)|^2$').opts(clim=PSD_clims, clabel='Power')
        fig2 = hv.Image((stft_time, self.frequencies, SNR), kdims=[t_dim, f_dim],
                        label=f'2. Trimmed {label_trimmed}').opts(clim=SNR_clims)

        figs = [fig0, fig1, fig2]

        # Weighted coefficients
        if weighted_coefficients:
            psd_fig = fig1
            f_dim = psd_fig.dimensions()[1]

            C = fig2.data['z']
            weights = np.where(C > 0, 1.0, self.main_params['background_weight'])
            WPSD = fig1.data['z'] * weights
            psd_opts = psd_fig.opts.get().options
            fig21 = hv.Image((psd_fig.data[t_dim.name], psd_fig.data[f_dim.name], WPSD),
                             kdims=[t_dim, f_dim],
                             label='2.1. Weighted amplitude spectrogram: $|X(t,f) \cdot W(t,f)|$').opts(**psd_opts)
            figs.append(fig21)

        if (ax := self.aggregate_axis_for_likelihood) is not None:
            ind = make_default_index_if_outrange(ind, ax, 0)

        fig3 = hv.Curve((time, self.signal_denoised[inds_time]),
                        kdims=[t_dim], vdims=A_dim).opts(linewidth=0.5)
        last_figs = [fig3]

        if picks:
            P = self.picked_features[ind]
            P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
            peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
            peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim).opts(marker='D', color='r')
            last_figs.append(peaks_fig)

        if intervals:
            intervals = intervals_intersection(self.detected_intervals[ind], (t1, t2))

            interv_height = np.max(abs(self.signal_denoised[inds_time])) * 1.1
            rectangles = give_rectangles([intervals], [interv_height], interv_height)
            intervals_fig = hv.Rectangles(rectangles,
                                          kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                                 linewidth=0,
                                                                                 alpha=0.2)
            last_figs.append(intervals_fig)

        fig3 = hv.Overlay(last_figs, label='4. Denoised data: $\\tilde{s}(t)$')

        figs.append(fig3)

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250
        cmap = 'viridis'
        xlim = time_interval_sec
        ylim = (max(1e-1, self.frequencies[1]), None)
        spectr_opts = hv.opts.Image(cmap=cmap, colorbar=True,  logy=True, logz=True, xlim=xlim, ylim=ylim,
                                    xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize)
        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=xlim, show_legend=False)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')

        fig = hv.Layout(figs).cols(1).opts(layout_opts, curve_opts, spectr_opts)
        return fig

    def plot_multi(self,
                   inds,
                   share_time_labels=True,
                   share_vertical_axes=True,
                   share_spectrogram_cmaps=True,
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
                    fig_i = fig_i.opts(hv.opts.Image(yaxis='bare'),
                                       hv.opts.Curve(yaxis='bare'))
        if share_spectrogram_cmaps:
            for i, figs_i in enumerate(zip(*figs)):
                if (0 < i < nrows - 1):
                    clims = [fi.opts.get().kwargs['clim'] for fi in figs_i]
                    if all([(cl is not None) for i, cl in np.ndenumerate(clims)]):
                        clim = (np.min(clims), np.max(clims))
                        for fi in figs_i:
                            fi = fi.opts(clim=clim)

            for i, fig_i in enumerate(figs):
                if i < ncols - 1:
                    fig_i = fig_i.opts(hv.opts.Image(colorbar=False))

        fig = hv.Layout(figs).cols(nrows).opts(**figs[0].opts.get().kwargs)

        vspace = vspace or 0.05 + 0.02 * (not share_time_labels)
        hspace = hspace or 0.025 + 0.1 * sum([not share_vertical_axes, not share_spectrogram_cmaps])

        fig = fig.opts(shared_axes=share_vertical_axes, vspace=vspace, hspace=hspace, transpose=True)

        return fig

    def append(self, other):

        concat_attrs = ["signal",
                        "coefficients",
                        "spectrogram",
                        "spectrogram_SNR_trimmed",
                        "likelihood",
                        "detection",
                        "signal_denoised"]

        for name in concat_attrs:
            self._concat(other, name, -1)

        stft_t0 = self.tf_dt_sec * (self.tf_time_npts - 1)

        self._concat(other, "detected_intervals", -2, stft_t0)
        self._concat(other, "picked_features", -2, stft_t0, (..., 0))

        self.time_npts += other.time_npts
        self.tf_time_npts += other.tf_time_npts

        self.history.merge(other.history)

        assert self.threshold == other.threshold
        assert self.dt_sec == other.dt_sec
        assert self.tf_dt_sec == other.tf_dt_sec
        assert np.all(self.frequencies == other.frequencies)
        assert np.all(self.noise_mean == other.noise_mean)
        assert np.all(self.noise_std == other.noise_std)

