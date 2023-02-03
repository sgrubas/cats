import numpy as np
import holoviews as hv
hv.extension('matplotlib')

from .core.timefrequency import STFT_Operator
from .core.date import BEDATE, EtaToSigma
from .core.thresholding import Thresholding
from .core.clustering import Clustering, ClusterFilling, OptimalNeighborhoodDistance
from .core.utils import get_interval_division
from .core.wiener import WienerNaive

##################### Detector API #####################


class CATSDenoiser:
    def __init__(self, dt_sec, stft_window_sec, stft_overlap, stft_nfft,
                       minSNR, stationary_frame_sec, min_dt_width_sec, min_df_width_Hz,
                       neighbor_distance_len=None, min_neighbors=None,
                       date_Q=0.95, date_detection_mode=False, wiener=False, **stft_kwargs):
        self.dt_sec         =   dt_sec
        self.STFT           =   STFT_Operator(window=stft_window_sec, overlap=stft_overlap,
                                              dt=dt_sec, nfft=stft_nfft, **stft_kwargs)
        self.stft_overlap   =   stft_overlap
        self.stft_overlap_len   =   self.STFT.noverlap
        self.stft_overlap_sec   =   self.stft_overlap_len * dt_sec
        self.stft_window        =   self.STFT.window
        self.stft_window_len    =   len(self.stft_window)
        self.stft_window_sec    =   self.stft_window_len * dt_sec
        self.stft_frequency =   self.STFT.f
        self.stft_nfft      =   self.STFT.nfft
        self.stft_hop_len    =   self.STFT.hop
        self.stft_hop_sec    =   dt_sec * self.stft_hop_len
        self.minSNR   =   minSNR
        self.date_Q   =   date_Q
        self.date_detection_mode = date_detection_mode
        self.date_noise_mode     = not date_detection_mode

        self.stationary_frame_len   =   max(int(stationary_frame_sec / self.stft_hop_sec), 256)
        self.stationary_frame_sec   =   self.stationary_frame_len * self.stft_hop_sec

        # Clustering parameters
        self.min_dt_width_sec   =   min_dt_width_sec
        self.min_dt_width_len   =   int(min_dt_width_sec / self.stft_hop_sec)
        self.min_df_width_Hz    =   min_df_width_Hz
        self.min_df_width_len   =   int(min_df_width_Hz / self.STFT.df)
        if neighbor_distance_len is None:
            self.neighbor_distance_len = OptimalNeighborhoodDistance(self.minSNR, d=2, pmin=0.95,
                                                                     qmax=min(self.min_df_width_len,
                                                                              self.min_dt_width_len),
                                                                     maxN=1)
        else:
            self.neighbor_distance_len = neighbor_distance_len
        if min_neighbors is None:
            self.min_neighbors = int((self.neighbor_distance_len * 2 + 1)**2 / 2)
        else:
            self.min_neighbors = min_neighbors

        self.wiener = wiener

    def denoise_stepwise(self, x):
        N = x.shape[-1]
        time = np.arange(N) * self.dt_sec
        X = self.STFT * x
        stft_time = self.STFT.forward_time_axis(N)
        stationary_intervals = get_interval_division(len(stft_time), self.stationary_frame_len)
        PSD = abs(X)
        Eta = BEDATE(PSD, frames=stationary_intervals, minSNR=self.minSNR,
                     Q=self.date_Q, original_mode=self.date_noise_mode)
        Sgm = EtaToSigma(Eta, self.minSNR)
        B = Thresholding(PSD, Eta, stationary_intervals)
        C = Clustering(B, q=self.neighbor_distance_len,
                       s=(self.min_df_width_len, self.min_dt_width_len))
        F = ClusterFilling(C, self.neighbor_distance_len, self.min_neighbors)
        W = WienerNaive(PSD, Sgm, F, stationary_intervals) if self.wiener else F
        y = (self.STFT / (X * W))[..., :N]
        kwargs = {"signal" : x, "spectrogram" : X, "noise_thresholding" : Eta, "noise_std" : Sgm,
                  "binary_spectrogram" : B, "binary_spectrogram_clustered" : C,
                  "binary_spectrogram_filled" : F, "wiener_spectrogram" : W, "denoised_signal" : y,
                  "time" : time, "stft_time" : stft_time, "stft_frequency" : self.stft_frequency,
                  "stationary_intervals" : stationary_intervals, "wienered" : self.wiener}

        return CATSDenoisingResult(**kwargs)

    def denoise(self, x):
        N = x.shape[-1]
        X = self.STFT * x
        stft_time = self.STFT.forward_time_axis(N)
        stationary_intervals = get_interval_division(len(stft_time), self.stationary_frame_len)
        PSD = abs(X)
        Eta = BEDATE(PSD, frames=stationary_intervals, minSNR=self.minSNR,
                     Q=self.date_Q, original_mode=self.date_noise_mode)
        B = Thresholding(PSD, Eta, stationary_intervals)
        C = Clustering(B, q=self.neighbor_distance_len,
                       s=(self.min_df_width_len, self.min_dt_width_len))
        F = ClusterFilling(C, self.neighbor_distance_len, self.min_neighbors)
        y = (self.STFT / (X * F))[..., :N]
        return y


class CATSDenoisingResult:
    def __init__(self, **kwargs):
        for kw, v in kwargs.items():
            self.__setattr__(kw, v)

    def plot(self, ind, figsize=250, cmap='viridis'):
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        A_dim = hv.Dimension('Amplitude')

        PSD = np.abs(self.spectrogram[ind])
        B = PSD * self.binary_spectrogram[ind].astype(float)
        C = PSD * self.binary_spectrogram_clustered[ind].astype(float)
        F = PSD * self.binary_spectrogram_filled[ind].astype(float)

        fig0 = hv.Curve((self.time, self.signal[ind]), kdims=[t_dim], vdims=A_dim,
                        label='0. Input data: $x_n$').opts(xlabel='', linewidth=0.2)
        fig1 = hv.Image((self.stft_time, self.stft_frequency, PSD), kdims=[t_dim, f_dim],
                        label='1. Spectrogram: $|X_{k,m}|$')
        fig2 = hv.Image((self.stft_time, self.stft_frequency, B), kdims=[t_dim, f_dim],
                        label='2. Trimming by B-E-DATE: $B_{k,m} \cdot |X_{k,m}|$')
        fig3 = hv.Image((self.stft_time, self.stft_frequency, C), kdims=[t_dim, f_dim],
                        label='3. Clustering: $C_{k,m} \cdot |X_{k,m}|$')
        fig4 = hv.Image((self.stft_time, self.stft_frequency, F), kdims=[t_dim, f_dim],
                        label='4. Filling: $F_{k,m} \cdot |X_{k,m}|$')
        if self.wienered:
            W = self.wiener_spectrogram[ind] * PSD
            fig41 = hv.Image((self.stft_time, self.stft_frequency, W), kdims=[t_dim, f_dim],
                            label='4.1 Wiener: $W_{k,m} \cdot |X_{k,m}|$')
        fig5 = hv.Curve((self.time, self.denoised_signal[ind]),
                        kdims=[t_dim], vdims=A_dim,
                        label='5. Denoised signal: $y_n$').opts(xlabel='Time (s)')

        fontsize = dict(labels=15, title=16, ticks=14)
        spectr_kwargs = dict(cmap=cmap, colorbar=True,  logy=True, logz=True, ylim=(1e-1, None),
                             xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize)
        curve_kwargs  = dict(aspect=5, fig_size=figsize, fontsize=fontsize)


        if self.wienered:
            fig = fig0 + fig1 + fig2 + fig3 + fig4 + fig41 + fig5
        else:
            fig = fig0 + fig1 + fig2 + fig3 + fig4 + fig5

        fig = fig.opts(fig_size=figsize, shared_axes=True, vspace=0.4, aspect_weight=0,
                       sublabel_format='').opts(hv.opts.Image(**spectr_kwargs),
                                                hv.opts.Curve(**curve_kwargs))
        return fig.cols(1)
