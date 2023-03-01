"""
    API for Denoiser based on Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDenoiser : Denoiser of seismic events based on CATS
        CATSDenoisingResult : keeps all the results and can plot sample trace with step-by-step visualization
"""

import numpy as np
import holoviews as hv

from .baseclass import CATSBaseSTFT, CATSResult
from .core.utils import get_interval_division


class CATSDenoiser(CATSBaseSTFT):
    """
        Denoiser based on Cluster Analysis of Trimmed Spectrograms
    """
    def denoise(self, x):
        N = x.shape[-1]
        time = np.arange(N) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(N)
        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        X, PSD, Eta, Sgm, SNR, K, P = super()._apply(x, finish_on='clustering')
        C = K > 0
        y = (self.STFT / (X * C))[..., :N]

        kwargs = {"signal": x, "coefficients": X, "spectrogram": PSD, "noise_thresholding": Eta, "noise_std": Sgm,
                  "binary_spectrogram": SNR > 0, "binary_spectrogram_clustered": C, "spectrogram_clusters": K,
                  "denoised_signal": y, "detection": P > 0, "projected_clusters": P, "SNR_spectrogram": SNR,
                  "time": time, "stft_time": stft_time, "stft_frequency": self.stft_frequency,
                  "stationary_intervals": stft_time[frames]}

        return CATSDenoisingResult(**kwargs)


class CATSDenoisingResult(CATSResult):

    def plot(self, ind, time_interval_sec=(None, None)):
        fig, opts, tints = super().plot(ind, time_interval_sec)

        t_dim = hv.Dimension('Time', unit='s')
        A_dim = hv.Dimension('Amplitude')
        ti1, ti2 = tints[0]

        fig = fig + hv.Curve((self.time[ti1: ti2], self.denoised_signal[ind][ti1: ti2]),
                              kdims=[t_dim], vdims=A_dim,
                              label='5. Denoised signal: $y_n$').opts(xlabel='Time (s)')

        return fig.opts(*opts).cols(1)
