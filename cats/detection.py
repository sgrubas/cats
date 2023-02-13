"""
    API for Detector based Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDetector : .................
        SNRDetector : experimental detector based on SNR values only (no CATS) .................
"""

import numpy as np
import holoviews as hv

from .core.projection import RemoveGaps
from .baseclass import CATSBaseSTFT, CATSResult
from .core.utils import get_interval_division


class CATSDetector(CATSBaseSTFT):
    def __init__(self, dt_sec, stft_window_sec, stft_overlap, stft_nfft, minSNR, stationary_frame_sec,
                 min_t_duration_sec, min_t_separation_sec, cluster_size_t_sec, cluster_size_f_Hz,
                 cluster_distance_t_sec=None, cluster_distance_f_Hz=None, clustering_with_SNR=True,
                 clustering_multitrace=False, cluster_size_trace=None, cluster_distance_trace=None,
                 date_Q=0.95, date_detection_mode=True, stft_backend='ssqueezepy', stft_kwargs=None):
        # Filtering detected intervals params
        self.min_t_separation_sec = min_t_separation_sec
        self.min_t_duration_sec = min_t_duration_sec
        # Set basic parameter via baseclass
        super().__init__(dt_sec=dt_sec, stft_window_sec=stft_window_sec, stft_overlap=stft_overlap, stft_nfft=stft_nfft,
                         minSNR=minSNR, stationary_frame_sec=stationary_frame_sec,
                         cluster_size_t_sec=cluster_size_t_sec, cluster_size_f_Hz=cluster_size_f_Hz,
                         cluster_distance_t_sec=cluster_distance_t_sec, cluster_distance_f_Hz=cluster_distance_f_Hz,
                         clustering_with_SNR=clustering_with_SNR, clustering_multitrace=clustering_multitrace,
                         cluster_size_trace=cluster_size_trace, cluster_distance_trace=cluster_distance_trace,
                         date_Q=date_Q, date_detection_mode=date_detection_mode, stft_backend=stft_backend,
                         stft_kwargs=stft_kwargs)

    def _set_params(self):
        super()._set_params()
        self.min_t_separation_len = int(self.min_t_separation_sec / self.stft_hop_sec)
        self.min_t_duration_len = int(self.min_t_duration_sec / self.stft_hop_sec)

    def detect(self, x):
        time = np.arange(x.shape[-1]) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(len(time))
        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        X, PSD, Eta, Sgm, SNR, K = super()._apply(x, finish_on='clustering')
        C = (K > 0)
        c = C.max(axis=-2)
        detection = RemoveGaps(c, self.min_t_separation_len, self.min_t_duration_len)

        kwargs = {"signal" : x, "coefficients" : X, "spectrogram" : PSD, "noise_thresholding" : Eta, "noise_std" : Sgm,
                  "binary_spectrogram" : SNR > 0, "binary_spectrogram_clustered" : C, "spectrogram_clusters" : K,
                  "binary_projection" : c, "detection" : detection, "SNR_spectrogram" : SNR, "time" : time,
                  "stft_time" : stft_time, "stft_frequency" : self.stft_frequency, "stationary_intervals" : frames}

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




