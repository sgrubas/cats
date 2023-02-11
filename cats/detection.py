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
from .core.thresholding import ThresholdingBySNR


class CATSDetector(CATSBaseSTFT):
    def __init__(self, dt_sec, stft_window_sec, stft_overlap, stft_nfft, minSNR, stationary_frame_sec,
                 min_dt_width_sec, min_df_width_Hz, max_dt_gap_sec, neighbor_distance=0.95, clustering_with_SNR=True,
                 clustering_multitrace=False, date_Q=0.95, date_detection_mode=True, stft_backend='ssqueezepy', stft_kwargs=None):
        # Filtering detected intervals params
        self.max_dt_gap_sec = max_dt_gap_sec
        # Set basic parameter via baseclass
        super().__init__(dt_sec=dt_sec, stft_window_sec=stft_window_sec, stft_overlap=stft_overlap, stft_nfft=stft_nfft,
                         minSNR=minSNR, stationary_frame_sec=stationary_frame_sec, min_dt_width_sec=min_dt_width_sec,
                         min_df_width_Hz=min_df_width_Hz, neighbor_distance=neighbor_distance,
                         clustering_with_SNR=clustering_with_SNR, clustering_multitrace=clustering_multitrace,
                         date_Q=date_Q, date_detection_mode=date_detection_mode, stft_backend=stft_backend,
                         stft_kwargs=stft_kwargs)

    def _set_params(self):
        super()._set_params()
        self.max_dt_gap_len = int(self.max_dt_gap_sec / self.stft_hop_sec)

    def detect(self, x):
        time = np.arange(x.shape[-1]) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(len(time))
        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        X, PSD, Eta, Sgm, SNR, K = super()._apply(x, finish_on='clustering')
        C = (K > 0)
        c = C.max(axis=-2)
        detection = RemoveGaps(c, self.max_dt_gap_len)

        kwargs = {"signal" : x, "coefficients" : X, "spectrogram" : PSD, "noise_thresholding" : Eta, "noise_std" : Sgm,
                  "binary_spectrogram" : SNR > 0, "binary_spectrogram_clustered" : C, "spectrogram_clusters" : K,
                  "binary_projection" : c, "detection" : detection, "SNR_spectrogram" : SNR,
                  "time" : time, "stft_time" : stft_time, "stft_frequency" : self.stft_frequency,
                  "stationary_intervals" : frames}

        return CATSDetectionResult(**kwargs)


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
        X, PSD, Eta = super()._apply(x, finish_on='date')
        Sgm = EtaToSigma(Eta, self.minSNR)
        frames = get_interval_division(N=X.shape[-1], L=self.stationary_frame_len)
        B = ThresholdingBySNR(PSD, Sgm, frames, self.minSNR)
        c = B.max(axis=-2)
        stft_time = self.STFT.forward_time_axis(x.shape[-1])
        detection = ProjectFilterIntervals(c, stft_time, self.max_dt_gap_sec, self.min_dt_width_sec, stft_time)

        time = np.arange(x.shape[-1]) * self.dt_sec
        kwargs = {"signal" : x, "coefficients" : X, "spectrogram" : PSD, "noise_thresholding" : Eta, "noise_std" : Sgm,
                  "binary_spectrogram" : B, "binary_spectrogram_clustered" : B, "spectrogram_clusters" : B,
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



