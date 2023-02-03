import numpy as np
import holoviews as hv
hv.extension('matplotlib')

from .core.timefrequency import STFT_Operator
from .core.date import BEDATE, EtaToSigma
from .core.thresholding import Thresholding
from .core.clustering import Clustering, ClusteringToProjection, OptimalNeighborhoodDistance
from .core.projection import RemoveGaps
from .core.utils import get_interval_division

##################### Detector API #####################


class CATSDetector:
    def __init__(self, dt_sec, stft_window_sec, stft_overlap, stft_nfft,
                       minSNR, stationary_frame_sec, min_dt_width_sec, min_df_width_Hz,
                       max_dt_gap_sec, neighbor_distance_len=None,
                       date_Q=0.95, date_detection_mode=True, **stft_kwargs):
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

        # Filtering detected intervals parameters
        self.max_dt_gap_sec     =   max_dt_gap_sec
        self.max_dt_gap_len     =   int(max_dt_gap_sec / self.stft_hop_sec)

    def detect_stepwise(self, x):
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
        c = C.max(axis=-2)
        detection = RemoveGaps(c, self.max_dt_gap_len)

        kwargs = {"signal" : x, "spectrogram" : X, "noise_thresholding" : Eta, "noise_std" : Sgm,
                  "binary_spectrogram" : B, "binary_spectrogram_clustered" : C, 
                  "binary_projection" : c, "final_detection" : detection,
                  "time" : time, "stft_time" : stft_time, "stft_frequency" : self.stft_frequency,
                  "stationary_intervals" : stationary_intervals}

        return CATSDetectionResult(**kwargs)

    def detect(self, x):
        stft_time = self.STFT.forward_time_axis(x.shape[-1])
        PSD = np.abs(self.STFT * x)
        stationary_intervals = get_interval_division(N=PSD.shape[-1], L=self.stationary_frame_len)
        Eta = BEDATE(PSD, frames=stationary_intervals, minSNR=self.minSNR, Q=self.date_Q)
        B = Thresholding(PSD, Eta, frames=stationary_intervals)
        detection = ClusteringToProjection(B, q=self.neighbor_distance_len,
                                           s=(self.min_df_width_len, self.min_dt_width_len),
                                           max_gap=self.max_dt_gap_len)
        # intervals = _giveIntervals(detection, time)

        return stft_time, detection


class CATSDetectionResult:
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

        fig0 = hv.Curve((self.time, self.signal[ind]), kdims=[t_dim], vdims=A_dim, 
                        label='0. Input data: $x_n$').opts(xlabel='', linewidth=0.2)
        fig1 = hv.Image((self.stft_time, self.stft_frequency, PSD), kdims=[t_dim, f_dim], 
                        label='1. Spectrogram: $|X_{k,m}|$')
        fig2 = hv.Image((self.stft_time, self.stft_frequency, B), kdims=[t_dim, f_dim],
                        label='2. Trimming by B-E-DATE: $B_{k,m} \cdot |X_{k,m}|$')
        fig3 = hv.Image((self.stft_time, self.stft_frequency, C), kdims=[t_dim, f_dim],
                        label='3. Clustering: $C_{k,m} \cdot |X_{k,m}|$')
        fig4 = hv.Curve((self.stft_time, self.final_detection[ind].astype(float)),
                        kdims=[t_dim], vdims='Classification',
                        label='4. Projection: $c_k$').opts(xlabel='Time (s)')

        fontsize = dict(labels=15, title=16, ticks=14)
        spectr_kwargs = dict(cmap=cmap, colorbar=True,  logy=True, logz=True, ylim=(1e-1, None), 
                             xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize)
        curve_kwargs  = dict(aspect=5, fig_size=figsize, fontsize=fontsize)

        fig = (fig0 + fig1 + fig2 + 
               fig3 + fig4).opts(fig_size=figsize, shared_axes=True, vspace=0.4, aspect_weight=0, 
                                 sublabel_format='').opts(hv.opts.Image(**spectr_kwargs),
                                                          hv.opts.Curve(**curve_kwargs))
        return fig.cols(1)
