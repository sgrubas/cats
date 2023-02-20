"""
    Implements baseclasses for Cluster Analysis of Trimmed Spectrograms (CATS)

        CATSBaseSTFT : baseclass to perform CATS
        CATSResult : baseclass for keeping the results of CATS
"""

import numpy as np
import holoviews as hv

from .core.timefrequency import STFTOperator
from .core.clustering import Clustering
from .core.date import BEDATE, EtaToSigma
from .core.utils import get_interval_division
from .core.thresholding import ThresholdingSNR


class CATSBaseSTFT:
    """
        Base class for CATS based on STFT. Implements unpacking of the main parameters.
        Implements 4 main steps:
            1) STFT
            2) Noise estimation via B-E-DATE
            3) Trimming spectrogram based minSNR
            4) Clustering trimmed spectrogram

    """
    def __init__(self, dt_sec, stft_window_sec, stft_overlap, stft_nfft,
                 minSNR, stationary_frame_sec, cluster_size_t_sec, cluster_size_f_Hz,
                 cluster_distance_t_sec=None, cluster_distance_f_Hz=None,
                 clustering_multitrace=False, cluster_size_trace=None, cluster_distance_trace=None,
                 date_Q=0.95, date_detection_mode=True, stft_backend='ssqueezepy', stft_kwargs=None):
        """
            Arguments:
                dt_sec : float : sampling time in seconds.
                stft_window_sec : float / tuple(str, float) : weighting window.
                                If `float`, defines length where window weights are ones [1, 1, ...].
                                If `tuple(str, float)`, `str` defines window type (e.g. `hann`, `hamming`),
                                and `float` is for length (e.g. ('hann', 0.5)).
                stft_overlap : float [0, 1] : overlapping of STFT windows (e.g. `0.5` means 50% overlap).
                stft_nfft : int : zero-padding for each individual STFT window, recommended a power of 2 (e.g. 256).
                minSNR : float : minimum Signal-to-Noise Ratio (SNR) for B-E-DATE algorithm. Recommended `4.0`.
                stationary_frame_sec : float : length of time frame in seconds wherein noise is stationary.
                                               Length will be adjusted to have at least 256 elements in one frame.
                cluster_size_t_sec : float : minimum cluster size in time, in seconds.
                                             Can be estimated as length of the strongest phases.
                cluster_size_f_Hz : float : minimum cluster size in frequency, in hertz, i.e. minimum frequency width.
                cluster_distance_t_sec : float : neighborhood distance in time for clustering, in seconds.
                                                 Minimum separation time between two different events.
                cluster_distance_f_Hz : float : neighborhood distance in frequency for clustering, in hertz.
                                                Minimum separation frequency width between two different events.
                clustering_multitrace : bool : whether to use multitrace clustering (Location x Frequency x Time).
                                               Increase accuracy for multiple arrays of receivers on regular grid.
                                               Performs cross-station association.
                cluster_size_trace : int : minimum cluster size for traces, minimum number of traces in one cluster.
                cluster_distance_trace : int : neighborhood distance across multiple traces for clustering.
                date_Q : float : probability that sorted elements after certain `Nmin` are have amplitude higher
                                 than standard deviation. Used in Bienaymé–Chebyshev inequality to estimate `Nmin`
                                 to reduce computational cost of DATE. Default `0.95`
                date_detection_mode : bool : `True` means NOT to use original implementation of DATE algorithm. Original
                                            implementation assumes that if no outliers are found then standard deviation
                                            is estimated from `Nmin` to not overestimate the noise. `True` implies that
                                            noise can be overestimated. It is beneficial if no outliers are found,
                                            then no outliers will be present in the trimmed spectrogram.
                stft_backend : str : backend for STFT operator ['scipy', 'ssqueezepy', 'ssqueezepy_gpu'].
                                     The fastest CPU version is 'ssqueezepy', which is default.
                stft_kwargs : dict : additional keyword arguments for STFT operator (see `cats.STFTOperator`).
        """

        # STFT params
        self.dt_sec = dt_sec
        self.stft_backend = stft_backend
        self.stft_kwargs = stft_kwargs or {}
        self.stft_overlap = stft_overlap
        self.stft_window_sec = stft_window_sec
        self.stft_nfft = stft_nfft
        # DATE params
        self.minSNR = minSNR
        self.date_Q = date_Q
        self.date_detection_mode = date_detection_mode
        self.stationary_frame_sec = stationary_frame_sec
        # Clustering params
        self.cluster_size_t_sec = cluster_size_t_sec
        self.cluster_size_f_Hz = cluster_size_f_Hz
        self.cluster_size_trace = cluster_size_trace

        self.cluster_distance_t_sec = cluster_distance_t_sec or self.cluster_size_t_sec / 2
        self.cluster_distance_f_Hz = cluster_distance_f_Hz or self.cluster_size_f_Hz / 2
        self.cluster_distance_trace = cluster_distance_trace or self.cluster_size_trace

        self.clustering_with_SNR = True
        self.clustering_multitrace = clustering_multitrace

        # Set other params and update correspondingly if needed
        self._set_params()

    def _set_params(self):
        # Setting STFT
        self.STFT = STFTOperator(window=self.stft_window_sec, overlap=self.stft_overlap, dt=self.dt_sec,
                                 nfft=self.stft_nfft, backend=self.stft_backend, **self.stft_kwargs)
        self.stft_overlap_len = self.STFT.noverlap
        self.stft_overlap_sec = self.stft_overlap_len * self.dt_sec
        self.stft_window = self.STFT.window
        self.stft_window_len = len(self.stft_window)
        self.stft_window_sec = self.stft_window_len * self.dt_sec
        self.stft_frequency = self.STFT.f
        self.stft_nfft = self.STFT.nfft
        self.stft_hop_len = self.STFT.hop
        self.stft_hop_sec = self.dt_sec * self.stft_hop_len

        # DATE params
        self.date_noise_mode = not self.date_detection_mode
        self.stationary_frame_len = max(int(self.stationary_frame_sec / self.stft_hop_sec), 256)
        self.stationary_frame_sec = self.stationary_frame_len * self.stft_hop_sec

        # Clustering params
        self.cluster_size_t_len = max(int(self.cluster_size_t_sec / self.stft_hop_sec), 1)
        self.cluster_size_f_len = max(int(self.cluster_size_f_Hz / self.STFT.df), 1)
        self.cluster_size_trace_len = max(self.cluster_size_trace or 1, 1)  # int(self.cluster_size_trace / self.dx)

        self.cluster_distance_t_len = max(int(self.cluster_distance_t_sec / self.stft_hop_sec), 1)
        self.cluster_distance_f_len = max(int(self.cluster_distance_f_Hz / self.STFT.df), 1)
        self.cluster_distance_trace_len = max(self.cluster_distance_trace or 1, 1)  # int(self.cluster_distance_trace / self.dx)

        self.edge_cut = int(self.stft_window_len // 2 / self.stft_hop_len)

        if self.clustering_multitrace:
            msg = "For clustering on multiple traces, `cluster_size_trace_len` and `cluster_distance_trace_len`"
            msg += f"must be `int`, given `{type(self.cluster_size_trace_len)}` and `{type(self.cluster_distance_trace_len)}`"
            assert isinstance(self.cluster_size_trace_len, int) and isinstance(self.cluster_distance_trace_len, int), msg

    def reset_params(self, **params):
        """
            Updates the instance with changed parameters.
        """
        for attribute, value in params.items():
            if hasattr(self, attribute):
                setattr(self, attribute, value)
            else:
                raise AttributeError(f'{self.__name__} has no attribute: {attribute}')
        self._set_params()

    def _apply(self, x, finish_on='clustering'):
        X = self.STFT * x
        PSD = np.abs(X)
        if 'stft' in finish_on.casefold():
            return X, PSD

        frames = get_interval_division(N=PSD.shape[-1], L=self.stationary_frame_len)
        Eta = BEDATE(PSD, frames=frames, minSNR=self.minSNR, Q=self.date_Q, original_mode=self.date_noise_mode)
        Sgm = EtaToSigma(Eta, self.minSNR)
        if 'date' in finish_on.casefold():
            return X, PSD, Eta, Sgm

        SNR = ThresholdingSNR(PSD, Sgm, Eta, frames)
        if self.edge_cut > 0:
            SNR[..., :self.edge_cut] = 0.0
            SNR[..., -self.edge_cut-1:] = 0.0
        if 'threshold' in finish_on.casefold():
            return X, PSD, Eta, Sgm, SNR

        mc = self.clustering_multitrace
        q = (self.cluster_distance_trace_len,) * mc + (self.cluster_distance_f_len, self.cluster_distance_t_len)
        s = (self.cluster_size_trace_len,) * mc + (self.cluster_size_f_len, self.cluster_size_t_len)
        minSNR = self.minSNR * self.clustering_with_SNR
        K, P = Clustering(SNR, q=q, s=s, minSNR=minSNR)

        return X, PSD, Eta, Sgm, SNR, K, P


class CATSResult:
    def __init__(self, signal, coefficients, spectrogram, noise_thresholding, noise_std, binary_spectrogram,
                 binary_spectrogram_clustered, spectrogram_clusters, time, stft_time, stft_frequency,
                 stationary_intervals, **kwargs):
        self.signal = signal
        self.coefficients = coefficients
        self.spectrogram = spectrogram
        self.noise_thresholding = noise_thresholding
        self.noise_std = noise_std
        self.binary_spectrogram = binary_spectrogram
        self.binary_spectrogram_clustered = binary_spectrogram_clustered
        self.spectrogram_clusters = spectrogram_clusters
        self.time = time
        self.stft_time = stft_time
        self.stft_frequency = stft_frequency
        self.stationary_intervals = stationary_intervals

        for kw, v in kwargs.items():
            self.__setattr__(kw, v)

    def plot(self, ind):
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        A_dim = hv.Dimension('Amplitude')

        PSD = self.spectrogram[ind]
        B = PSD * self.binary_spectrogram[ind].astype(float)
        C = PSD * self.binary_spectrogram_clustered[ind].astype(float)

        fig0 = hv.Curve((self.time, self.signal[ind]), kdims=[t_dim], vdims=A_dim,
                        label='0. Input data: $x_n$').opts(xlabel='', linewidth=0.2)
        fig1 = hv.Image((self.stft_time, self.stft_frequency, PSD), kdims=[t_dim, f_dim],
                        label='1. Spectrogram: $|X_{k,m}|$')
        fig2 = hv.Image((self.stft_time, self.stft_frequency, B), kdims=[t_dim, f_dim],
                        label='2. Trimming by B-E-DATE: $B_{k,m} \cdot |X_{k,m}|$')
        fig3 = hv.Image((self.stft_time, self.stft_frequency, C), kdims=[t_dim, f_dim],
                        label='3. Clustering: $C_{k,m} \cdot |X_{k,m}|$')

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250; cmap = 'viridis'; clim = (PSD.min(), PSD.max())
        spectr_opts = hv.opts.Image(cmap=cmap, colorbar=True,  logy=True, logz=True, ylim=(1e-1, None),
                                    clim=clim, xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize)
        curve_opts  = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                       aspect_weight=0, sublabel_format='')

        figs = (fig0 + fig1 + fig2 + fig3)
        return figs, (layout_opts, spectr_opts, curve_opts)
