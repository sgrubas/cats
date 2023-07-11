"""
    Implements detector API for Power-Spectral-Density detector (Yoones & van der Baan, GJI, 2015)
"""

from pydantic import BaseModel, Field, Extra
from typing import Callable, Union, Tuple, List, Any
from pathlib import Path
from tqdm.notebook import tqdm

import numpy as np
import holoviews as hv
from cats.core.projection import RemoveGaps, GiveIntervals
from cats.core.timefrequency import STFTOperator
from cats.core.association import PickDetectedPeaks
from cats.core.utils import format_index_by_dims, format_interval_by_limits, give_index_slice_by_limits
from cats.core.utils import aggregate_array_by_axis_and_func, cast_to_bool_dict, del_vals_by_keys, StatusKeeper
from cats.core.utils import give_rectangles, update_object_params, give_nonzero_limits, complex_abs_square
from cats.baseclass import CATSResult, CATSBase
from cats.detection import CATSDetector
from cats.io import read_data


class PSDDetector(BaseModel, extra=Extra.allow):
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
    freq_bandpass_Hz: Tuple[float, float] = None
    characteristic: str = 'abs'

    aggregate_axis_for_likelihood: Union[int, Tuple[int]] = None
    aggregate_func_for_likelihood: Callable[[np.ndarray], np.ndarray] = np.max
    pick_features: bool = False

    psd_noise_models: Any = None
    psd_noise_mean: Any = None
    psd_noise_std: Any = None

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

        self.min_duration_len = int(self.min_duration_sec / self.stft_hop_sec)
        self.min_separation_len = int(self.min_separation_sec / self.stft_hop_sec)

        self.ch_functions = {'abs': lambda x: x, 'square': np.square}
        self.ch_func = self.ch_functions[self.characteristic]

    def reset_params(self, **params):
        """
            Updates the instance with changed parameters. Will delete the noise model
        """
        update_object_params(self, **params)

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

        full_info = self._parse_info_dict(full_info)

        result = dict.fromkeys(full_info.keys())
        result['signal'] = x if full_info['signal'] else None

        history = StatusKeeper(verbose=verbose)

        time = np.arange(x.shape[-1]) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(len(time))

        with history(current_process='STFT'):
            result['coefficients'] = self.STFT * x
            result['spectrogram'] = np.abs(result['coefficients'])**2

        del_vals_by_keys(result, full_info, ['coefficients'])
        bandpass_slice = (..., self.freq_bandpass_slice, slice(None))

        with history(current_process='Thresholding'):
            result['spectrogram_SNR_trimmed'] = np.zeros_like(result['spectrogram'])
            result['spectrogram_SNR_trimmed'][bandpass_slice] = \
                result['spectrogram'][bandpass_slice] - self.psd_noise_mean[bandpass_slice]
            result['spectrogram_SNR_trimmed'][bandpass_slice] /= self.psd_noise_std[bandpass_slice] + 1e-16
            del_vals_by_keys(result, full_info, ['spectrogram'])
            result['spectrogram_SNR_trimmed'] = \
                np.where(result['spectrogram_SNR_trimmed'] >= 1.0,
                         self.ch_func(result['spectrogram_SNR_trimmed']), 0.0)

        with history(current_process='Likelihood'):
            result['spectrogram_SNR_trimmed'] = \
                aggregate_array_by_axis_and_func(result['spectrogram_SNR_trimmed'],
                                                 self.aggregate_axis_for_likelihood,
                                                 self.aggregate_func_for_likelihood,
                                                 min_last_dims=2)

            result['likelihood'] = result['spectrogram_SNR_trimmed'].mean(axis=-2)
            result['detection'] = RemoveGaps(result['likelihood'] > self.threshold,
                                             self.min_separation_len,
                                             self.min_duration_len)

        del_vals_by_keys(result, full_info, ['spectrogram_SNR_trimmed'])

        # Picking
        if full_info['picked_features']:
            with history(current_process='Picking'):
                result['picked_features'] = PickDetectedPeaks(result['likelihood'], result['detection'],
                                                              dt=self.stft_hop_sec)
        del_vals_by_keys(result, full_info, ['likelihood'])

        history.print_total_time()

        from_full_info = {kw: result.get(kw, None) for kw in full_info}
        kwargs = {**from_full_info,
                  "noise_mean": self.psd_noise_mean,
                  "noise_std": self.psd_noise_std,
                  "dt_sec": self.dt_sec,
                  "stft_dt_sec": self.stft_hop_sec,
                  "stft_t0_sec": stft_time[0],
                  "npts": x.shape[-1],
                  "stft_npts": len(stft_time),
                  "stft_frequency": self.stft_frequency,
                  "threshold": self.threshold,
                  "history": history}

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
        """
        n_chunks = self._split_data_by_memory(x, full_info=full_info, to_file=False)
        single_chunk = n_chunks > 1
        data_chunks = np.array_split(x, n_chunks, axis=-1)
        results = []
        for dc in tqdm(data_chunks, desc='Data chunks', display=single_chunk):
            results.append(self._detect(dc, verbose=verbose, full_info=full_info))
        return PSDDetectionResult.concatenate(*results)

    def __mul__(self, x):
        return self.detect(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.detect(x, verbose=True, full_info=True)

    def _parse_info_dict(self, full_info):
        info_keys = ["signal",
                     "coefficients",
                     "spectrogram",
                     "spectrogram_SNR_trimmed",
                     "likelihood",
                     "detection",
                     "picked_features"]

        if isinstance(full_info, str) and \
           (full_info in ['plot', 'plotting', 'plt', 'qc', 'main']):  # only those needed for plotting step-by-step
            full_info = {'signal':                  True,
                         'spectrogram':             True,
                         'spectrogram_SNR_trimmed': True,
                         'likelihood':              True}

        full_info = cast_to_bool_dict(full_info, info_keys)

        full_info["detection"] = True  # default saved result
        full_info["picked_features"] = self.pick_features  # non-default result

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

        memory_usage_bytes = {
            "stft_frequency":           8. * stft_shape[-2],                     # float64 always
            "signal":                   1. * x_bytes,                            # float / int
            "coefficients":             2. * precision_order * stft_size,        # complex
            "spectrogram":              1. * precision_order * stft_size,        # float
            "spectrogram_SNR_trimmed":  1. * precision_order * stft_size,        # float
            "noise_mean":               1. * precision_order * noise_size,       # float
            "noise_std":                1. * precision_order * noise_size,       # float
            "likelihood":               1. * precision_order * likelihood_size,  # float
            "detection":                1. * likelihood_size,                    # bool always
        }

        used_together = [('coefficients', 'spectrogram'),
                         ('spectrogram', 'spectrogram_SNR_trimmed'),
                         ('spectrogram_SNR_trimmed', 'likelihood', 'detection'),
                         ('likelihood', 'detection')]
        base_info = ["signal", "stft_frequency", "noise_mean", "noise_std"]
        full_info = self._parse_info_dict(full_info)

        return CATSBase.memory_info(memory_usage_bytes, used_together, base_info, full_info)

    def _split_data_by_memory(self, x, /, full_info, to_file):
        memory_info = self.memory_usage_estimate(x, full_info=full_info)
        return CATSBase.memory_chunks(memory_info, to_file)

    def detect_to_file(self, x, /, path_destination, verbose=False, full_info=False, compress=False):
        CATSDetector._detect_to_file(self, x, path_destination, verbose, full_info, compress)

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


class PSDDetectionResult(CATSResult):
    def __init__(self,
                 signal=None,
                 coefficients=None,
                 spectrogram=None,
                 spectrogram_SNR_trimmed=None,
                 likelihood=None,
                 detection=None,
                 picked_features=None,
                 noise_mean=None,
                 noise_std=None,
                 dt_sec=None,
                 stft_dt_sec=None,
                 stft_t0_sec=None,
                 npts=None,
                 stft_npts=None,
                 stft_frequency=None,
                 threshold=None,
                 history=None
                 ):

        # Numpy arrays
        self.signal = signal
        self.coefficients = coefficients
        self.spectrogram = spectrogram
        self.spectrogram_SNR_trimmed = spectrogram_SNR_trimmed
        self.likelihood = likelihood
        self.detection = detection
        self.picked_features = picked_features
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.stft_frequency = stft_frequency

        # Scalars
        self.dt_sec = dt_sec
        self.stft_dt_sec = stft_dt_sec
        self.stft_t0_sec = stft_t0_sec
        self.npts = npts
        self.stft_npts = stft_npts
        self.threshold = threshold
        self.history = history

    def plot(self, ind=None, time_interval_sec=None):
        if ind is None:
            ind = (0,) * (self.signal.ndim - 1)
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        A_dim = hv.Dimension('Amplitude')
        L_dim = hv.Dimension('Likelihood')

        ind = format_index_by_dims(ind, self.signal.shape)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.npts - 1) * self.dt_sec))
        t1, t2 = time_interval_sec

        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        i_stft = give_index_slice_by_limits(time_interval_sec, self.stft_dt_sec)

        inds_time = ind + (i_time,)
        inds_stft = ind + (i_stft,)
        inds_Stft = ind + (slice(None), i_stft)

        PSD = self.spectrogram[inds_Stft]
        SNR = self.spectrogram_SNR_trimmed[inds_Stft]

        PSD_clims = give_nonzero_limits(PSD, initials=(1e-1, 1e1))
        SNR_clims = give_nonzero_limits(SNR, initials=(1e-1, 1e1))

        time = self.time(time_interval_sec)
        stft_time = self.stft_time(time_interval_sec)

        fig0 = hv.Curve((time, self.signal[inds_time]), kdims=[t_dim], vdims=A_dim,
                        label='0. Input data: $x(t)$').opts(xlabel='', linewidth=0.2)
        fig1 = hv.Image((stft_time, self.stft_frequency, PSD), kdims=[t_dim, f_dim],
                        label='1. Spectrogram: $|X(t,f)|^2$').opts(clim=PSD_clims)
        fig2 = hv.Image((stft_time, self.stft_frequency, SNR), kdims=[t_dim, f_dim],
                        label='2. Trimmed SNR spectrogram: $T(t,f)$').opts(clim=SNR_clims)

        likelihood = np.nan_to_num(self.likelihood[inds_stft],
                                   posinf=10 * self.threshold,
                                   neginf=-10 * self.threshold)  # POSSIBLE `NAN` AND `INF` VALUES!
        likelihood_fig = hv.Curve((stft_time, likelihood),
                                  kdims=[t_dim], vdims=L_dim)

        if getattr(self, 'picked_features', None) is not None:
            P = self.picked_features[ind]
            P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
            peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
            peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim).opts(marker='D', color='r')
        else:
            peaks_fig = None

        intervals = GiveIntervals(self.detection[inds_stft])
        interv_height = (np.max(likelihood) / 2) * 1.1
        rectangles = give_rectangles(intervals, stft_time, [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles,
                                      kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                             linewidth=0,
                                                                             alpha=0.2)
        snr_level_fig = hv.HLine(self.threshold, kdims=[t_dim, L_dim]).opts(color='k', linestyle='--', alpha=0.7)

        last_figs = [intervals_fig, snr_level_fig, likelihood_fig]
        if peaks_fig is not None:
            last_figs.append(peaks_fig)

        fig3 = hv.Overlay(last_figs, label='3. Likelihood and Detection:'
                                           ' $\mathcal{L}(t)$ and $\mathcal{D}(t)$')

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250; cmap = 'viridis'
        xlim = time_interval_sec
        ylim = (1e-1, None)
        spectr_opts = hv.opts.Image(cmap=cmap, colorbar=True,  logy=True, logz=True, xlim=xlim, ylim=ylim,
                                    xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize)
        curve_opts  = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=xlim, show_legend=False)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        figs = [fig0, fig1, fig2, fig3]
        fig = hv.Layout(figs).cols(1).opts(layout_opts, curve_opts, spectr_opts)
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

        stft_t0 = self.stft_dt_sec * self.stft_npts

        self._concat(other, "picked_features", -2, stft_t0, (..., 0))

        self.npts += other.npts
        self.stft_npts += other.stft_npts

        self.history.merge(other.history)

        assert self.threshold == other.threshold
        assert self.dt_sec == other.dt_sec
        assert self.stft_dt_sec == other.stft_dt_sec
        assert np.all(self.stft_frequency == other.stft_frequency)
        assert np.all(self.noise_mean == other.noise_mean)
        assert np.all(self.noise_std == other.noise_std)
