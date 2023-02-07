"""
    API for Detector based Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDetector : .................
        SNRDetector : experimental detector based on SNR values only (no CATS) .................
"""

import numpy as np
import holoviews as hv

from .core.date import EtaToSigma
from .core.clustering import ClusteringToProjection
from .core.projection import RemoveGaps, ProjectFilterIntervals
from .baseclass import CATSBaseSTFT, CATSResult
from .core.utils import get_interval_division
from .core.thresholding import ThresholdingSNR

class CATSDetector(CATSBaseSTFT):
    def __init__(self, dt_sec, stft_window_sec, stft_overlap, stft_nfft, minSNR, stationary_frame_sec,
                 min_dt_width_sec, min_df_width_Hz, max_dt_gap_sec, neighbor_distance=0.95, date_Q=0.95,
                 date_detection_mode=True, stft_backend='ssqueezepy', stft_kwargs=None):
        # Set basic parameter via baseclass
        super().__init__(dt_sec=dt_sec, stft_window_sec=stft_window_sec, stft_overlap=stft_overlap, stft_nfft=stft_nfft,
                         minSNR=minSNR, stationary_frame_sec=stationary_frame_sec, min_dt_width_sec=min_dt_width_sec,
                         min_df_width_Hz=min_df_width_Hz, neighbor_distance=neighbor_distance, date_Q=date_Q,
                         date_detection_mode=date_detection_mode, stft_backend=stft_backend, stft_kwargs=stft_kwargs)

        # Filtering detected intervals params
        self.max_dt_gap_sec = max_dt_gap_sec
        self.max_dt_gap_len = int(max_dt_gap_sec / self.stft_hop_sec)

    def detect_stepwise(self, x):
        X, Eta, B, C = super()._apply(x, finish_on='clustering')
        c = C.max(axis=-2)
        detection = RemoveGaps(c, self.max_dt_gap_len)

        Sgm = EtaToSigma(Eta, self.minSNR)
        time = np.arange(x.shape[-1]) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(x.shape[-1])
        frames = get_interval_division(N=X.shape[-1], L=self.stationary_frame_len)

        kwargs = {"signal" : x, "spectrogram" : X, "noise_thresholding" : Eta, "noise_std" : Sgm,
                  "binary_spectrogram" : B, "binary_spectrogram_clustered" : C,
                  "binary_projection" : c, "detection" : detection,
                  "time" : time, "stft_time" : stft_time, "stft_frequency" : self.stft_frequency,
                  "stationary_intervals" : frames}

        return CATSDetectionResult(**kwargs)

    def detect(self, x):
        X, Eta, B = super()._apply(x, finish_on='threshold')
        detection = ClusteringToProjection(B, q=self.neighbor_distance_len,
                                           s=(self.min_df_width_len, self.min_dt_width_len),
                                           max_gap=self.max_dt_gap_len)
        stft_time = self.STFT.forward_time_axis(x.shape[-1])
        return stft_time, detection


class CATSDetectionResult(CATSResult):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, ind):
        fig, opts = super().plot(ind)
        t_dim = hv.Dimension('Time', unit='s')
        fig4 = hv.Curve((self.stft_time, self.detection[ind].astype(float)),
                        kdims=[t_dim], vdims='Classification', label='4. Projection: $o_k$')
        return (fig + fig4).opts(*opts).cols(1)


class SNRDetector(CATSBaseSTFT):
    def __init__(self, dt_sec, stft_window_sec, stft_overlap, stft_nfft, minSNR, stationary_frame_sec,
                 min_dt_width_sec, max_dt_gap_sec, neighbor_distance=0.95, date_Q=0.95,
                 date_detection_mode=True, stft_backend='ssqueezepy', stft_kwargs=None):
        # Set basic parameter via baseclass
        super().__init__(dt_sec=dt_sec, stft_window_sec=stft_window_sec, stft_overlap=stft_overlap, stft_nfft=stft_nfft,
                         minSNR=minSNR, stationary_frame_sec=stationary_frame_sec, min_dt_width_sec=min_dt_width_sec,
                         min_df_width_Hz=0.0, neighbor_distance=neighbor_distance, date_Q=date_Q,
                         date_detection_mode=date_detection_mode, stft_backend=stft_backend, stft_kwargs=stft_kwargs)

        # Filtering detected intervals params
        self.max_dt_gap_sec = max_dt_gap_sec
        self.max_dt_gap_len = int(max_dt_gap_sec / self.stft_hop_sec)


    def detect(self, x):
        X, Eta = super()._apply(x, finish_on='date')
        Sgm = EtaToSigma(Eta, self.minSNR)
        frames = get_interval_division(N=X.shape[-1], L=self.stationary_frame_len)
        B = ThresholdingSNR(np.abs(X), Sgm, frames, self.minSNR)
        c = B.max(axis=-2)
        stft_time = self.STFT.forward_time_axis(x.shape[-1])
        detection = ProjectFilterIntervals(c, stft_time, self.max_dt_gap_sec, self.min_dt_width_sec, stft_time)

        time = np.arange(x.shape[-1]) * self.dt_sec
        kwargs = {"signal" : x, "spectrogram" : X, "noise_thresholding" : Eta, "noise_std" : Sgm,
                  "binary_spectrogram" : B, "binary_spectrogram_clustered" : B,
                  "binary_projection" : c, "detection" : detection,
                  "time" : time, "stft_time" : stft_time, "stft_frequency" : self.stft_frequency,
                  "stationary_intervals" : frames}

        return SNRDetectionResult(**kwargs)


class SNRDetectionResult(CATSResult):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, ind):
        fig, opts = super().plot(ind)
        t_dim = hv.Dimension('Time', unit='s')
        fig4 = hv.Curve((self.stft_time, self.detection[ind].astype(float)),
                        kdims=[t_dim], vdims='Classification', label='4. Projection: $o_k$')
        fig[-1] = fig4
        return fig.opts(*opts).cols(1)



