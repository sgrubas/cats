import numpy as np
import holoviews as hv

from .core.timefrequency import STFTOperator
from .core.clustering import OptimalNeighborhoodDistance, Clustering
from .core.date import BEDATE
from .core.utils import get_interval_division
from .core.thresholding import Thresholding


class CATSBaseSTFT:
    """
        Base class for CATS based on STFT. Implements unpacking of base parameters.

    """
    def __init__(self, dt_sec, stft_window_sec, stft_overlap, stft_nfft,
                 minSNR, stationary_frame_sec, min_dt_width_sec, min_df_width_Hz,
                 neighbor_distance=0.95, date_Q=0.95, date_detection_mode=True,
                 stft_backend='ssqueezepy', stft_kwargs=None):
        """
            Arguments:
                dt_sec : float : sampling time in seconds
                stft_window_sec : float / tuple(str, float) : weighting window.
                                If `float`, `stft_window_sec` is length of window in seconds
                                and all window weights are ones.
                                If `tuple(str, float)`, `str` defines window type (e.g. `hann`, `hamming`),
                                and `float` is for length.
                stft_overlap : float [0, 1] : overlapping if STFT windows, e.g. `0.5` means 50% overlap
                stft_nfft : int : zero-padding for each individual STFT window,
                                  preferably should be a power of 2 (e.g. 256, 512).
                minSNR : float : minimum Signal-to-Noise Ratio (SNR) for DATE algorithm.
                                 Recommended value is `minSNR = 4.0`
                stationary_frame_sec : float : length of time frame in seconds wherein noise is assumed to be stationary
                                               One time frame must have at least 256 elements, which is used as minimum.
                min_dt_width_sec : float : minimum time duration of seismic event in seconds.
                                           It will be used in clustering
                min_df_width_Hz : float : minimum frequency band of seismic event in Hz. It will be used in clustering.
                neighbor_distance : int / float : Neighborhood distance for clustering.
                                   Value >= 1 means exactly distance that will be used.
                                   Value in [0, 1] is used as probability that at maximum 1 element will be noise
                                   within neighborhood distance. Provides an estimate of optimal distance takin into
                                   account influence of `minSNR` on the sparsity. Default `0.95`
                date_Q : float : probability that sorted elements after certain `Nmin` are have amplitude higher
                                 than standard deviation. Used in Bienaymé–Chebyshev inequality to estimate `Nmin`
                                 to reduce computational cost of DATE. Default `0.95`
                date_detection_mode : bool : `True` means not to use original implementation of DATE algorithm in case
                                             if no outliers are found and standard deviation is estimated from `Nmin`.
                                             Default `True`, i.e. if no outliers then noise estimated on everything.
                kwargs : dict : additional keyword arguments
        """

        # STFT params
        self.dt_sec = dt_sec
        self.stft_backend = stft_backend
        self.stft_kwargs = stft_kwargs or {}
        self.STFT = STFTOperator(window=stft_window_sec, overlap=stft_overlap, dt=dt_sec,
                                 nfft=stft_nfft, backend=self.stft_backend, **self.stft_kwargs)
        self.stft_overlap = stft_overlap
        self.stft_overlap_len = self.STFT.noverlap
        self.stft_overlap_sec = self.stft_overlap_len * dt_sec
        self.stft_window = self.STFT.window
        self.stft_window_len = len(self.stft_window)
        self.stft_window_sec = self.stft_window_len * dt_sec
        self.stft_frequency = self.STFT.f
        self.stft_nfft = self.STFT.nfft
        self.stft_hop_len = self.STFT.hop
        self.stft_hop_sec = dt_sec * self.stft_hop_len

        # DATE params
        self.minSNR = minSNR
        self.date_Q = date_Q
        self.date_detection_mode = date_detection_mode
        self.date_noise_mode = not date_detection_mode

        self.stationary_frame_len = max(int(stationary_frame_sec / self.stft_hop_sec), 256)
        self.stationary_frame_sec = self.stationary_frame_len * self.stft_hop_sec

        # Clustering params
        self.min_dt_width_sec = min_dt_width_sec
        self.min_dt_width_len = int(min_dt_width_sec / self.stft_hop_sec)
        self.min_df_width_Hz = min_df_width_Hz
        self.min_df_width_len = int(min_df_width_Hz / self.STFT.df)
        if 0.0 <= neighbor_distance < 1.0:
            self.neighbor_distance_len = OptimalNeighborhoodDistance(self.minSNR, d=2,
                                                                     pmin=neighbor_distance,
                                                                     qmax=min(self.min_df_width_len,
                                                                              self.min_dt_width_len),
                                                                     maxN=1)
        else:
            self.neighbor_distance_len = int(neighbor_distance)

    def _apply(self, x, finish_on='clustering'):
        X = self.STFT * x
        PSD = np.abs(X)
        if 'stft' in finish_on.casefold():
            return X, PSD

        frames = get_interval_division(N=PSD.shape[-1], L=self.stationary_frame_len)
        Eta = BEDATE(PSD, frames=frames, minSNR=self.minSNR, Q=self.date_Q, original_mode=self.date_noise_mode)
        if 'date' in finish_on.casefold():
            return X, PSD, Eta

        B = Thresholding(PSD, Eta, frames=frames)
        if 'threshold' in finish_on.casefold():
            return X, PSD, Eta, B

        K = Clustering(B, q=self.neighbor_distance_len, s=(self.min_df_width_len, self.min_dt_width_len))
        return X, PSD, Eta, B, K


class CATSResult:
    def __init__(self, signal, coefficients, spectrogram, noise_thresholding, noise_std, binary_spectrogram,
                 binary_spectrogram_clustered, spectrogram_clusters,
                 time, stft_time, stft_frequency, stationary_intervals, **kwargs):
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
        figsize = 250
        cmap = 'viridis'
        spectr_opts = hv.opts.Image(cmap=cmap, colorbar=True,  logy=True, logz=True, ylim=(1e-1, None),
                                      xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize)
        curve_opts  = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                       aspect_weight=0, sublabel_format='')

        figs = (fig0 + fig1 + fig2 + fig3)
        return figs, (layout_opts, spectr_opts, curve_opts)
