"""
    Implements baseclasses for Cluster Analysis of Trimmed Spectrograms (CATS)

        CATSBaseSTFT : baseclass to perform CATS
        CATSResult : baseclass for keeping the results of CATS
"""

import numpy as np
import holoviews as hv
from typing import Tuple
from pydantic import BaseModel, Field, Extra

from .core.timefrequency import STFTOperator
from .core.clustering import Clustering
from .core.date import BEDATE, EtaToSigma, MIN_DATE_BLOCK_SIZE
from .core.utils import get_interval_division, format_index_by_dims, cast_to_bool_dict, StatusKeeper
from .core.utils import format_interval_by_limits, give_index_slice_by_limits, del_vals_by_keys
from .core.thresholding import ThresholdingSNR


class CATSBaseSTFT(BaseModel, extra=Extra.allow):
    """
        Base class for CATS based on STFT. Implements unpacking of the main parameters.
        Implements 4 main steps:
            1) STFT
            2) Noise estimation via B-E-DATE
            3) Trimming spectrogram based minSNR
            4) Clustering trimmed spectrogram
        """
    # main STFT params
    dt_sec: float
    stft_window_type: str = 'hann'
    stft_window_sec: float = 0.5
    stft_overlap: float = Field(0.75, ge=0.0, lt=1.0)
    # main B-E-DATE params
    minSNR: float = 4.0
    stationary_frame_sec: float = None
    # main Clustering params
    cluster_size_t_sec: float = 0.2
    cluster_size_f_Hz: float = 20.0
    cluster_distance_t_sec: float = None
    cluster_distance_f_Hz: float = None

    # Minor clustering params
    freq_bandpass_Hz: Tuple[float, float] = None
    clustering_multitrace: bool = False
    cluster_size_trace: int = Field(1, ge=1)
    cluster_distance_trace: int = Field(1, ge=1)
    # Minor DATE params
    date_Q: float = 0.95
    date_detection_mode: bool = True
    # Minor STFT params
    stft_backend: str = 'ssqueezepy'
    stft_kwargs: dict = {}
    stft_nfft: int = -1

    def __init__(self, **kwargs):
        """
            Arguments:
                dt_sec : float : sampling time in seconds.

                stft_window_type: str : type of weighting window in STFT (e.g. `hann`, `hamming`, 'ones). Default 'hann'

                stft_window_sec : float : length of weighting window in seconds.

                stft_overlap : float [0, 1] : overlapping of STFT windows (e.g. `0.5` means 50% overlap).

                stft_nfft : int : zero-padding for each individual STFT window, recommended a power of 2 (e.g. 256).

                minSNR : float : minimum Signal-to-Noise Ratio (SNR) for B-E-DATE algorithm. Recommended `4.0`.

                stationary_frame_sec : float : length of time frame in seconds wherein noise is stationary. \
Length will be adjusted to have at least 256 elements in one frame. Default, 0.0 which leads to minimum possible

                cluster_size_t_sec : float : minimum cluster size in time, in seconds. \
Can be estimated as length of the strongest phases.

                cluster_size_f_Hz : float : minimum cluster size in frequency, in hertz, i.e. minimum frequency width.

                freq_bandpass_Hz : tuple/list[int/float] : bandpass frequency range, in hertz, i.e. everything out of the \
range is zero (e.g. (f_min, f_max)).

                cluster_distance_t_sec : float : neighborhood distance in time for clustering, in seconds. \
Minimum separation time between two different events.

                cluster_distance_f_Hz : float : neighborhood distance in frequency for clustering, in hertz. \
Minimum separation frequency width between two different events.

                clustering_multitrace : bool : whether to use multitrace clustering (Location x Frequency x Time). \
Increase accuracy for multiple arrays of receivers on regular grid. Performs cross-station association.

                cluster_size_trace : int : minimum cluster size for traces, minimum number of traces in one cluster.

                cluster_distance_trace : int : neighborhood distance across multiple traces for clustering.

                date_Q : float : probability that sorted elements after certain `Nmin` have amplitude higher \
than standard deviation. Used in Bienaymé–Chebyshev inequality to get `Nmin` to minimize cost of DATE. Default `0.95`

                date_detection_mode : bool : `True` means NOT to use original implementation of DATE algorithm. \
Original implementation assumes that if no outliers are found then standard deviation is estimated from `Nmin` \
to not overestimate the noise. `True` implies that noise can be overestimated. It is beneficial if no outliers \
are found, then no outliers will be present in the trimmed spectrogram.

                stft_backend : str : backend for STFT operator ['scipy', 'ssqueezepy', 'ssqueezepy_gpu']. \
The fastest CPU version is 'ssqueezepy', which is default.

                stft_kwargs : dict : additional keyword arguments for STFT operator (see `cats.STFTOperator`).
        """
        super().__init__(**kwargs)
        self._set_params()

    def _set_params(self):
        # Setting STFT
        self.STFT = STFTOperator(window=(self.stft_window_type, self.stft_window_sec), overlap=self.stft_overlap,
                                 dt=self.dt_sec, nfft=self.stft_nfft, backend=self.stft_backend, **self.stft_kwargs)
        self.stft_overlap_len = self.STFT.noverlap
        self.stft_overlap_sec = (self.stft_overlap_len - 1) * self.dt_sec
        self.stft_window = self.STFT.window
        self.stft_window_len = len(self.stft_window)
        self.stft_window_sec = (self.stft_window_len - 1) * self.dt_sec
        self.stft_frequency = self.STFT.f
        self.stft_df = self.STFT.df
        self.stft_nfft = self.STFT.nfft
        self.stft_hop_len = self.STFT.hop
        self.stft_hop_sec = (self.stft_hop_len - 1) * self.dt_sec

        # DATE params
        self.stationary_frame_sec = v if (v := self.stationary_frame_sec) is not None else 0.0
        self.stationary_frame_len = max(round(self.stationary_frame_sec / self.stft_hop_sec), MIN_DATE_BLOCK_SIZE)
        self.stationary_frame_sec = self.stationary_frame_len * self.stft_hop_sec

        # Clustering params
        self.freq_bandpass_Hz = format_interval_by_limits(self.freq_bandpass_Hz,
                                                          (self.stft_frequency[0], self.stft_frequency[-1]))
        self.freq_bandpass_Hz = (min(self.freq_bandpass_Hz), max(self.freq_bandpass_Hz))
        self.freq_bandwidth_Hz = self.freq_bandpass_Hz[1] - self.freq_bandpass_Hz[0]
        assert (fbw := self.freq_bandwidth_Hz) > (csf := self.cluster_size_f_Hz), \
                f"Frequency bandpass width `{fbw}` must be bigger than min frequency cluster size `{csf}`"
        self.freq_bandpass_slice = give_index_slice_by_limits(self.freq_bandpass_Hz, self.STFT.df)
        self.freq_bandpass_len = (self.freq_bandpass_slice.start, self.freq_bandpass_slice.stop)
        self.cluster_size_t_len = max(round(self.cluster_size_t_sec / self.stft_hop_sec), 1)
        self.cluster_size_f_len = max(round(self.cluster_size_f_Hz / self.STFT.df), 1)
        self.cluster_size_trace_len = self.cluster_size_trace

        self.cluster_distance_t_sec = v if (v := self.cluster_distance_t_sec) is not None else self.stft_hop_sec
        self.cluster_distance_f_Hz = v if (v := self.cluster_distance_f_Hz) is not None else self.stft_df
        self.cluster_distance_t_len = max(round(self.cluster_distance_t_sec / self.stft_hop_sec), 1)
        self.cluster_distance_f_len = max(round(self.cluster_distance_f_Hz / self.stft_df), 1)
        self.cluster_distance_trace_len = self.cluster_distance_trace

        self.edge_cut = int(self.stft_window_len // 2 / self.stft_hop_len)

    def reset_params(self, **params):
        """
            Updates the instance with changed parameters.
        """
        for attribute, value in params.items():
            if hasattr(self, attribute):
                setattr(self, attribute, value)
            else:
                raise AttributeError(f'{type(self)} has no attribute: {attribute}')
        self._set_params()

    def _apply(self, x, /, finish_on='clustering', verbose=True, full_info=True):

        result = {"coefficients": None, "spectrogram": None, "noise_threshold": None, "noise_std": None,
                  "spectrogram_SNR_trimmed": None, "spectrogram_SNR_clustered": None, "spectrogram_cluster_ID": None}

        full_info = cast_to_bool_dict(full_info, list(result.keys()))

        History = StatusKeeper(verbose=verbose)

        # STFT
        with History(current_process='STFT'):
            result['coefficients'] = self.STFT * x
            result['spectrogram'] = np.abs(result['coefficients'])

        del_vals_by_keys(result, full_info, ['coefficients'])

        if 'stft' in finish_on.casefold():
            return result, History

        # bandpass
        bandpass_slice = (..., self.freq_bandpass_slice, slice(None))

        # B-E-DATE
        with History(current_process='B-E-DATE'):
            frames = get_interval_division(N=result['spectrogram'].shape[-1], L=self.stationary_frame_len)
            zero_Nyq_freqs = (self.freq_bandpass_len[0] == 0,
                              len(self.stft_frequency) == self.freq_bandpass_len[1])
            result['noise_threshold'] = np.zeros(result['spectrogram'].shape[:-1] + (len(frames),))
            result['noise_threshold'][bandpass_slice] = BEDATE(result['spectrogram'][bandpass_slice],
                                                   frames=frames,
                                                   minSNR=self.minSNR,
                                                   Q=self.date_Q,
                                                   original_mode=not self.date_detection_mode,
                                                   zero_Nyq_freqs=zero_Nyq_freqs)
            result['noise_std'] = EtaToSigma(result['noise_threshold'], self.minSNR)

            # Thresholding
            result['spectrogram_SNR_trimmed'] = np.zeros_like(result['spectrogram'])
            result['spectrogram_SNR_trimmed'][bandpass_slice] = \
                ThresholdingSNR(result['spectrogram'][bandpass_slice],
                                result['noise_std'][bandpass_slice],
                                result['noise_threshold'][bandpass_slice],
                                frames)

            if self.edge_cut > 0:
                result['spectrogram_SNR_trimmed'][..., :self.edge_cut] = 0.0
                result['spectrogram_SNR_trimmed'][..., -self.edge_cut - 1:] = 0.0

        del_vals_by_keys(result, full_info, ['spectrogram', 'noise_threshold', 'noise_std'])

        if 'date' in finish_on.casefold():
            return result, History

        # Clustering
        with History(current_process='Clustering'):
            mc = self.clustering_multitrace
            q = (self.cluster_distance_trace_len,) * mc + (self.cluster_distance_f_len, self.cluster_distance_t_len)
            s = (self.cluster_size_trace_len,) * mc + (self.cluster_size_f_len, self.cluster_size_t_len)

            result['spectrogram_SNR_clustered'] = np.zeros_like(result['spectrogram_SNR_trimmed'])
            result['spectrogram_cluster_ID'] = np.zeros(result['spectrogram_SNR_trimmed'].shape, dtype=np.uint16)

            result['spectrogram_SNR_clustered'][bandpass_slice], result['spectrogram_cluster_ID'][bandpass_slice] = \
                Clustering(result['spectrogram_SNR_trimmed'][bandpass_slice], q=q, s=s, minSNR=self.minSNR)

        del_vals_by_keys(result, full_info, ['spectrogram_SNR_trimmed',
                                             'spectrogram_SNR_clustered',
                                             'spectrogram_cluster_ID'])

        return result, History


class CATSResult:
    def __init__(self, signal, coefficients, spectrogram, noise_threshold, noise_std, spectrogram_SNR_trimmed,
                 spectrogram_SNR_clustered, spectrogram_cluster_ID, likelihood, picked_features,
                 time, stft_time, stft_frequency, stationary_intervals, minSNR, history, **kwargs):
        self.signal = signal
        self.coefficients = coefficients
        self.spectrogram = spectrogram
        self.noise_threshold = noise_threshold
        self.noise_std = noise_std
        self.spectrogram_SNR_trimmed = spectrogram_SNR_trimmed
        self.spectrogram_SNR_clustered = spectrogram_SNR_clustered
        self.spectrogram_cluster_ID = spectrogram_cluster_ID
        self.likelihood = likelihood
        self.picked_features = picked_features
        self.time = time
        self.stft_time = stft_time
        self.stft_frequency = stft_frequency
        self.stationary_intervals = stationary_intervals
        self.minSNR = minSNR
        self.history = history

        for kw, v in kwargs.items():
            self.__setattr__(kw, v)

    def plot(self, ind, time_interval_sec=None):
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        A_dim = hv.Dimension('Amplitude')

        ind = format_index_by_dims(ind, self.signal.shape)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (self.time[0], self.time[-1]))
        i_time = give_index_slice_by_limits(time_interval_sec, self.time[1] - self.time[0])
        i_stft = give_index_slice_by_limits(time_interval_sec, self.stft_time[1] - self.stft_time[0])
        inds_time = ind + (i_time,)
        inds_stft = ind + (slice(None), i_stft)

        PSD = self.spectrogram[inds_stft]
        B = PSD * (self.spectrogram_SNR_trimmed[inds_stft] > 0)
        C = PSD * (self.spectrogram_SNR_clustered[inds_stft] > 0)

        fig0 = hv.Curve((self.time[i_time], self.signal[inds_time]), kdims=[t_dim], vdims=A_dim,
                        label='0. Input data: $x_n$').opts(xlabel='', linewidth=0.2)
        fig1 = hv.Image((self.stft_time[i_stft], self.stft_frequency, PSD), kdims=[t_dim, f_dim],
                        label='1. Spectrogram: $|X_{k,m}|$')
        fig2 = hv.Image((self.stft_time[i_stft], self.stft_frequency, B), kdims=[t_dim, f_dim],
                        label='2. Trimming by B-E-DATE: $B_{k,m} \cdot |X_{k,m}|$')
        fig3 = hv.Image((self.stft_time[i_stft], self.stft_frequency, C), kdims=[t_dim, f_dim],
                        label='3. Clustering: $C_{k,m} \cdot |X_{k,m}|$')

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250
        cmap = 'viridis'
        PSD_pos = PSD[PSD > 0]
        cmin = 1e-1 if (PSD_pos.size == 0) else PSD_pos.min()
        cmax = 1e1 if (PSD_pos.size == 0) else PSD_pos.max()
        clim = (cmin, cmax)
        xlim = time_interval_sec
        ylim = (1e-1, None)
        spectr_opts = hv.opts.Image(cmap=cmap, colorbar=True,  logy=True, logz=True, xlim=xlim, ylim=ylim, clim=clim,
                                    xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize)
        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=xlim)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        inds_slices = (ind, i_time, i_stft)
        figs = (fig0 + fig1 + fig2 + fig3)
        return figs, (layout_opts, spectr_opts, curve_opts), inds_slices
