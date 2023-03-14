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
        out_slice = (..., slice(None, N))
        time = np.arange(N) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(N)
        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        X, PSD, Eta, Sgm, SNR, SNRK, K = super()._apply(x, finish_on='clustering')
        y = (self.STFT / (X * (SNRK > 0)))[out_slice]
        detection = K.max(axis=-2) > 0
        kwargs = {"signal": x, "coefficients": X, "spectrogram": PSD, "noise_thresholding": Eta, "noise_std": Sgm,
                  "spectrogramSNR_trimmed": SNR, "spectrogramSNR_clustered": SNRK, "spectrogramID_cluster": K,
                  "denoised_signal": y, "detection": detection, "time": time, "stft_time": stft_time,
                  "stft_frequency": self.stft_frequency, "stationary_intervals": stft_time[frames]}
        return CATSDenoisingResult(**kwargs)


class CATSDenoisingResult(CATSResult):

    def plot(self, ind, time_interval_sec=(None, None)):
        fig, opts, inds_slices = super().plot(ind, time_interval_sec)

        t_dim = hv.Dimension('Time', unit='s')
        A_dim = hv.Dimension('Amplitude')
        i_t = inds_slices[1]
        inds_t = inds_slices[0] + (i_t,)

        fig = fig + hv.Curve((self.time[i_t], self.denoised_signal[inds_t]), kdims=[t_dim], vdims=A_dim,
                              label='5. Denoised signal: $y_n$').opts(xlabel='Time (s)')

        return fig.opts(*opts).cols(1)
