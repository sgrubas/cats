"""
    Implements detector API for Power-Spectral-Density detector (Yoones & van der Baan, GJI, 2015)
"""

from pydantic import BaseModel, Field, Extra
from typing import Callable, Union, Tuple

import numpy as np
import holoviews as hv
from cats.core.projection import RemoveGaps, GiveIntervals
from cats.core.timefrequency import STFTOperator
from cats.core.association import PickWithFeatures
from cats.core.utils import format_index_by_dims, format_interval_by_limits, give_index_slice_by_limits
from cats.core.utils import aggregate_array_by_axis_and_func, cast_to_bool_dict, del_vals_by_keys, StatusMessenger
from cats.core.utils import give_rectangles


# TODO: - logging msgs


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.STFT = STFTOperator(window=self.stft_window_sec, overlap=self.stft_overlap, dt=self.dt_sec,
                                 nfft=self.stft_nfft, backend=self.stft_backend, **self.stft_kwargs)
        self.stft_overlap_len = self.STFT.noverlap
        self.stft_overlap_sec = self.stft_overlap_len * self.dt_sec
        self.stft_window = self.STFT.window
        self.stft_window_len = len(self.stft_window)
        self.stft_window_sec = self.stft_window_len * self.dt_sec
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

        self.psd_noise_models = None
        self.psd_noise_mean = None
        self.psd_noise_std = None

    def set_noise_model(self, *noisy_pieces):
        psd_noise_models = np.concatenate([np.square(self.STFT * xi) for xi in noisy_pieces], axis=-1)
        self.psd_noise_mean = psd_noise_models.mean(axis=-1, keepdims=True)
        self.psd_noise_std = psd_noise_models.std(axis=-1, keepdims=True)
        del psd_noise_models

    def split_data_by_intervals(self, data, *intervals_sec):
        intervals_slices = tuple(give_index_slice_by_limits(intv, self.dt_sec) for intv in intervals_sec)
        split_data = tuple(data[..., inds] for inds in intervals_slices)
        return split_data

    def detect(self, x, verbose=True, full_info=False):
        assert self.psd_noise_mean is not None, f"Noise model must be set priorly, use `.set_noise_model(...)`"

        result = {"X": None, "PSD": None, "SNR": None,
                  "likelihood": None, "detection": None}

        full_info = cast_to_bool_dict(full_info, list(result.keys()))

        time = np.arange(x.shape[-1]) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(len(time))

        with StatusMessenger(verbose=verbose, operation='1. STFT'):
            result['X'] = self.STFT * x
            result['PSD'] = np.square(result['X'])

        del_vals_by_keys(result, full_info, ['X'])
        bandpass_slice = (..., self.freq_bandpass_slice, slice(None))

        with StatusMessenger(verbose=verbose, operation='2. Thresholding'):
            result['SNR'] = np.zeros_like(result['PSD'])
            result['SNR'][bandpass_slice] = result['PSD'][bandpass_slice] - self.psd_noise_mean[bandpass_slice]
            result['SNR'][bandpass_slice] /= self.psd_noise_std[bandpass_slice] + 1e-16
            del_vals_by_keys(result, full_info, ['PSD'])
            result['SNR'] = np.where(result['SNR'] >= 1.0, self.ch_func(result['SNR']), 0.0)

        with StatusMessenger(verbose=verbose, operation='3. Likelihood'):
            result['SNR'] = aggregate_array_by_axis_and_func(result['SNR'],
                                                             self.aggregate_axis_for_likelihood,
                                                             self.aggregate_func_for_likelihood,
                                                             min_last_dims=2)

            result['likelihood'] = result['SNR'].mean(axis=-2)
            result['detection'] = RemoveGaps(result['likelihood'] > self.threshold,
                                             self.min_separation_len,
                                             self.min_duration_len)

        del_vals_by_keys(result, full_info, ['SNR', 'detection'])

        # Picking
        with StatusMessenger(verbose=verbose, operation='4. Picking onsets'):
            result['picked_features'] = PickWithFeatures(result['likelihood'],
                                                         time=stft_time,
                                                         min_likelihood=self.threshold,
                                                         min_width_sec=self.min_duration_sec,
                                                         num_features=None)

        del_vals_by_keys(result, full_info, ['likelihood'])

        return PSDResult(signal=x,
                         coefficients=result.get('X', None),
                         spectrogram=result.get('PSD', None),
                         spectrogramSNR_trimmed=result.get('SNR', None),
                         likelihood=result.get('likelihood', None),
                         detection=result.get('detection', None),
                         picked_features=result.get('picked_features', None),
                         noise_mean=self.psd_noise_mean,
                         noise_std=self.psd_noise_std,
                         time=time,
                         stft_time=stft_time,
                         stft_frequency=self.stft_frequency,
                         threshold=self.threshold)

    def __mul__(self, x):
        return self.detect(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.detect(x, verbose=True, full_info=True)


class PSDResult:
    def __init__(self, signal, coefficients, spectrogram, spectrogramSNR_trimmed, likelihood, detection,
                 picked_features, noise_mean, noise_std, time, stft_time, stft_frequency, threshold):
        self.signal = signal
        self.coefficients = coefficients
        self.spectrogram = spectrogram
        self.spectrogramSNR_trimmed = spectrogramSNR_trimmed
        self.likelihood = likelihood
        self.detection = detection
        self.picked_features = picked_features
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.time = time
        self.stft_time = stft_time
        self.stft_frequency = stft_frequency
        self.threshold = threshold

    def plot(self, ind, time_interval_sec=None):
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        A_dim = hv.Dimension('Amplitude')
        L_dim = hv.Dimension('Likelihood')

        ind = format_index_by_dims(ind, self.signal.shape)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (self.time[0], self.time[-1]))
        t1, t2 = time_interval_sec
        i_t = give_index_slice_by_limits(time_interval_sec, self.time[1] - self.time[0])
        i_st = give_index_slice_by_limits(time_interval_sec, self.stft_time[1] - self.stft_time[0])
        inds_t = ind + (i_t,)
        inds_st = ind + (i_st,)
        inds_St = ind + (slice(None), i_st)

        PSD = self.spectrogram[inds_St]
        B = PSD * (self.spectrogramSNR_trimmed[inds_St] > 0)

        fig0 = hv.Curve((self.time[i_t], self.signal[inds_t]), kdims=[t_dim], vdims=A_dim,
                        label='0. Input data: $x_n$').opts(xlabel='', linewidth=0.2)
        fig1 = hv.Image((self.stft_time[i_st], self.stft_frequency, PSD), kdims=[t_dim, f_dim],
                        label='1. Spectrogram: $|X_{k,m}|$')
        fig2 = hv.Image((self.stft_time[i_st], self.stft_frequency, B), kdims=[t_dim, f_dim],
                        label='2. Trimming by noise level: $B_{k,m} \cdot |X_{k,m}|$')

        likelihood = self.likelihood[inds_st]
        likelihood_fig = hv.Curve((self.stft_time[i_st], likelihood),
                                  kdims=[t_dim], vdims=L_dim)

        P = self.picked_features[ind]
        P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
        peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
        peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim).opts(marker='D', color='r')

        intervals = GiveIntervals(self.detection[inds_st])
        interv_height = (likelihood.max() / 2) * 1.1
        rectangles = give_rectangles(intervals, self.stft_time[i_st], [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles, kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                                         linewidth=0, alpha=0.1)
        snr_level_fig = hv.HLine(self.threshold, kdims=[t_dim, L_dim]).opts(color='k', linestyle='--', alpha=0.7)

        fig3 = hv.Overlay([intervals_fig, snr_level_fig, likelihood_fig, peaks_fig],
                          label='3. Projection: $L_k$')

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250; cmap = 'viridis'
        PSD_pos = PSD[PSD > 0]
        cmin = 1e-1 if (PSD_pos.size == 0) else PSD_pos.min()
        cmax = 1e1 if (PSD_pos.size == 0) else PSD_pos.max()
        clim = (cmin, cmax)
        xlim = time_interval_sec
        ylim = (1e-1, None)
        spectr_opts = hv.opts.Image(cmap=cmap, colorbar=True,  logy=True, logz=True, xlim=xlim, ylim=ylim, clim=clim,
                                    xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize)
        curve_opts  = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=xlim, show_legend=False)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        figs = [fig0, fig1, fig2, fig3]
        fig = hv.Layout(figs).cols(1).opts(layout_opts, curve_opts, spectr_opts)
        return fig
