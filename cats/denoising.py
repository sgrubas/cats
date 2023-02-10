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
                 min_dt_width_sec, min_df_width_Hz, neighbor_distance=0.95, clusteringWithSNR=True, min_neighbors=None,
                 date_Q=0.95, date_detection_mode=True, wiener=False, stft_backend='ssqueezepy', stft_kwargs=None):
        # Filling cluster params
        self.min_neighbors = min_neighbors
        # Wiener filtering
        self.wiener = wiener
        # Set basic parameter via baseclass
        super().__init__(dt_sec=dt_sec, stft_window_sec=stft_window_sec, stft_overlap=stft_overlap, stft_nfft=stft_nfft,
                         minSNR=minSNR, stationary_frame_sec=stationary_frame_sec, min_dt_width_sec=min_dt_width_sec,
                         min_df_width_Hz=min_df_width_Hz, neighbor_distance=neighbor_distance,
                         clusteringWithSNR=clusteringWithSNR, date_Q=date_Q, date_detection_mode=date_detection_mode,
                         stft_backend=stft_backend, stft_kwargs=stft_kwargs)

    def _set_params(self):
        super()._set_params()
        self.min_neighbors = self.min_neighbors or (self.neighbor_distance_len * 2 + 1)**2 // 2

    def denoise_stepwise(self, x):
        N = x.shape[-1]
        time = np.arange(N) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(N)
        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        X, PSD, Eta, Sgm, SNR, K = super()._apply(x, finish_on='clustering')
        C = K > 0
        F = ClusterFilling(C, self.neighbor_distance_len, self.min_neighbors)
        W = WienerNaive(PSD, Sgm, F, frames) if self.wiener else F
        y = (self.STFT / (X * W))[..., :N]

        kwargs = {"signal" : x, "coefficients" : X, "spectrogram" : PSD, "noise_thresholding" : Eta, "noise_std" : Sgm,
                  "binary_spectrogram" : SNR > 0, "binary_spectrogram_clustered" : C, "spectrogram_clusters" : K,
                  "binary_spectrogram_filled" : F, "wiener_spectrogram" : W, "denoised_signal" : y,
                  "time" : time, "stft_time" : stft_time, "stft_frequency" : self.stft_frequency,
                  "stationary_intervals" : frames, "wienered" : self.wiener}

        return CATSDenoisingResult(**kwargs)

    def denoise(self, x):
        X, PSD, Eta, Sgm, SNR, K = super()._apply(x, finish_on='clustering')
        F = ClusterFilling(K > 0, self.neighbor_distance_len, self.min_neighbors)
        y = (self.STFT / (X * F))[..., :x.shape[-1]]
        return y


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
