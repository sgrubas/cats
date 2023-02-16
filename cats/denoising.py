"""
    API for Denoiser based on Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDenoiser : STFT based yet .................
"""

import numpy as np
import holoviews as hv

from .core.date import EtaToSigma
from .core.clustering import ClusterFilling
from .core.wiener import WienerNaive
from .baseclass import CATSBaseSTFT, CATSResult
from .core.utils import get_interval_division


class CATSDenoiser(CATSBaseSTFT):

    def __init__(self, dt_sec, stft_window_sec, stft_overlap, stft_nfft, minSNR, stationary_frame_sec,
                 cluster_size_t_sec, cluster_size_f_Hz, cluster_distance_t_sec=None, cluster_distance_f_Hz=None,
                 clustering_with_SNR=True, clustering_multitrace=False, cluster_size_trace=None,
                 cluster_distance_trace=None, min_neighbors=None,
                 date_Q=0.95, date_detection_mode=True, wiener=False, stft_backend='ssqueezepy', stft_kwargs=None):
        # Filling cluster params
        self.min_neighbors = min_neighbors
        # Wiener filtering
        self.wiener = wiener
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
        if self.min_neighbors is None:
            self.min_neighbors = ((self.cluster_distance_t_len * 2 + 1) *
                                  (self.cluster_distance_f_len * 2 + 1) *
                                  (self.cluster_distance_trace_len * self.clustering_multitrace * 2 + 1)) // 2
        else:
            self.min_neighbors = self.min_neighbors

    def denoise(self, x):
        N = x.shape[-1]
        time = np.arange(N) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(N)
        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        X, PSD, Eta, Sgm, SNR, K, P = super()._apply(x, finish_on='clustering')
        C = K > 0
        mc = self.clustering_multitrace
        q = (self.cluster_distance_trace_len,) * mc + (self.cluster_distance_f_len, self.cluster_distance_t_len)
        F = ClusterFilling(C, q, self.min_neighbors)
        W = WienerNaive(PSD, Sgm, F, frames) if self.wiener else F
        y = (self.STFT / (X * W))[..., :N]

        kwargs = {"signal" : x, "coefficients" : X, "spectrogram" : PSD, "noise_thresholding" : Eta, "noise_std" : Sgm,
                  "binary_spectrogram" : SNR > 0, "binary_spectrogram_clustered" : C, "spectrogram_clusters" : K,
                  "binary_spectrogram_filled" : F, "wiener_spectrogram" : W, "denoised_signal" : y,
                  "time" : time, "stft_time" : stft_time, "stft_frequency" : self.stft_frequency,
                  "stationary_intervals" : frames, "wienered" : self.wiener}

        return CATSDenoisingResult(**kwargs)


class CATSDenoisingResult(CATSResult):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, ind):
        fig, opts = super().plot(ind)

        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        A_dim = hv.Dimension('Amplitude')

        PSD = self.spectrogram[ind]
        F = PSD * self.binary_spectrogram_filled[ind].astype(float)

        fig4 = hv.Image((self.stft_time, self.stft_frequency, F), kdims=[t_dim, f_dim],
                                label='4. Filling: $F_{k,m} \cdot |X_{k,m}|$')
        fig = fig + fig4
        if self.wienered:
            W = self.wiener_spectrogram[ind] * PSD
            fig41 = hv.Image((self.stft_time, self.stft_frequency, W), kdims=[t_dim, f_dim],
                            label='4.1 Wiener: $W_{k,m} \cdot |X_{k,m}|$')
            fig = fig + fig41
        fig5 = hv.Curve((self.time, self.denoised_signal[ind]),
                        kdims=[t_dim], vdims=A_dim,
                        label='5. Denoised signal: $y_n$').opts(xlabel='Time (s)')
        fig = fig + fig5

        return fig.opts(*opts).cols(1)
