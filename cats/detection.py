"""
    API for Detector based on Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDetector : Detector of seismic events based on CATS
        CATSDetectionResult : keeps all the results and can plot sample trace with step-by-step visualization
"""

import numpy as np
import holoviews as hv

from .baseclass import CATSBaseSTFT, CATSResult
from .core.utils import get_interval_division


class CATSDetector(CATSBaseSTFT):
    """
        Detector of events based on Cluster Analysis of Trimmed Spectrograms
    """
    def detect(self, x):
        """
            Performs the detection on the given dataset.

            Arguments:
                x : np.ndarray (..., N) : input data with any number of dimensions, but the last axis `N` must be Time.
        """
        time = np.arange(x.shape[-1]) * self.dt_sec
        stft_time = self.STFT.forward_time_axis(len(time))
        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        X, PSD, Eta, Sgm, SNR, K, P = super()._apply(x, finish_on='clustering')

        kwargs = {"signal": x, "coefficients": X, "spectrogram": PSD, "noise_thresholding": Eta, "noise_std": Sgm,
                  "binary_spectrogram": SNR > 0, "binary_spectrogram_clustered": K > 0, "spectrogram_clusters": K,
                  "detection": P > 0, "projected_clusters": P, "SNR_spectrogram": SNR, "time": time,
                  "stft_time": stft_time, "stft_frequency": self.stft_frequency,
                  "stationary_intervals": stft_time[frames]}

        return CATSDetectionResult(**kwargs)


class CATSDetectionResult(CATSResult):

    def plot(self, ind, time_interval_sec=(None, None)):
        fig, opts, tints = super().plot(ind, time_interval_sec)
        t_dim = hv.Dimension('Time', unit='s')
        sti1, sti2 = tints[1]
        fig4 = hv.Curve((self.stft_time[sti1: sti2], self.detection[ind][sti1: sti2].astype(float)),
                        kdims=[t_dim], vdims='Classification', label='4. Projection: $o_k$')
        return (fig + fig4).opts(*opts).cols(1)




