"""
-
"""

import numpy as np
import holoviews as hv
from typing import Callable, Union, Tuple
import psutil

from .baseclass import CATSBase, CATSResult
from .core.utils import get_interval_division
from .core.utils import cast_to_bool_dict, del_vals_by_keys, give_index_slice_by_limits

# TODO:
#  - make sure that this class is needed
#  - understand if clustering is needed for noise estimation


class CATSNoiseEstimator(CATSBase):
    """
        Noise estimator based on Cluster Analysis of Trimmed Spectrograms
    """

    def estimate(self, x, /, verbose=False, full_info=False):
        """
            Estimates noise spectrogram.

            Arguments:
                x : np.ndarray (..., N) : input data with any number of dimensions, but the last axis `N` must be Time.
                verbose : bool : whether to print status and timing
                full_info : bool / dict[str, bool] : whether to save workflow stages for further quality control.
                        If `True`, then all stages are saved, if `False` then only the `detection` is saved.
                        Available workflow stages in `full_info` which can be controlled by [True / False]:
                            - "coefficients" - STFT coefficients
                            - "spectrogram" - absolute value of `coefficients`
                            - "noise_threshold" - threshold defined by `noise_std`
                            - "noise_std" - noise level, standard deviation
                            - "spectrogram_SNR_trimmed" - `spectrogram / noise_std` trimmed by `noise_threshold`
                            - "spectrogram_SNR_clustered" - clustered `spectrogram_SNR_trimmed`
                            - "spectrogram_cluster_ID" - cluster indexes on `spectrogram_SNR_clustered`
        """

        full_info, pre_full_info = self._parse_info_dict(full_info)

        result, history = super()._apply(x, finish_on='clustering', verbose=verbose, full_info=pre_full_info)

        stft_time = self.STFT.forward_time_axis(x.shape[-1])

        history.print_total_time()

        time = np.arange(x.shape[-1]) * self.dt_sec
        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        from_full_info = {kw: result.get(kw, None) for kw in full_info}
        kwargs = {"signal": x,
                  **from_full_info,
                  "time": time,
                  "stft_time": stft_time,
                  "stft_frequency": self.stft_frequency,
                  "noise_stationary_intervals": stft_time[frames],
                  "minSNR": self.minSNR,
                  "history": history}

        return CATSNoiseEstimationResult(**kwargs)

    def __mul__(self, x):
        return self.estimate(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.estimate(x, verbose=True, full_info=True)

    def _parse_info_dict(self, full_info):
        info_keys = self._info_keys()
        pre_full_info = full_info.copy()
        pre_full_info['spectrogram_SNR_clustered'] = True  # needed for `CATSBase._apply`
        pre_full_info['spectrogram_cluster_ID'] = True     # needed for `CATSBase._apply`
        return full_info, pre_full_info

    def estimate_memory_usage(self, x, /, full_info=False):
        memory_usage_bytes, used_together = self._memory_usage(x)
        full_info, pre_full_info = self._parse_info_dict(full_info)

        stft_time_len = len(self.STFT.forward_time_axis(x.shape[-1]))

        likelihood_shape = x.shape[:-1] + (stft_time_len,)
        likelihood_size = np.prod(likelihood_shape)

        precision_order = memory_usage_bytes['signal'] / x.size

        memory_usage_bytes_new = {
            "likelihood":        precision_order * likelihood_size,     # float
            "detection":         likelihood_size,                       # bool always
        }
        memory_usage_bytes.update(memory_usage_bytes_new)

        used_together.append(('spectrogram_SNR_clustered', 'spectrogram_cluster_ID', 'likelihood', 'detection'))
        used_together_bytes = [0] * len(used_together)
        for i, ut in enumerate(used_together):
            used_together_bytes[i] += sum([memory_usage_bytes[kw] for kw in ut])

        base_info = ["time", "stft_time", "stft_frequency", "noise_stationary_intervals", "signal"]
        base_bytes = sum([memory_usage_bytes[kw] for kw in base_info])
        min_required = base_bytes + max(used_together_bytes)
        max_required = sum(memory_usage_bytes.values())
        info_required = base_bytes + sum([memory_usage_bytes.get(kw, 0) for kw, v in full_info.items() if v])

        memory_resources = psutil.virtual_memory()
        memory_info = {'total': memory_resources.total,
                       'available': memory_resources.available,
                       'min_required': min_required,
                       'max_required': max_required,
                       'required_by_info': info_required,
                       'safe_to_proceed_on_min': min_required < memory_resources.available,
                       'detailed_memory_usage': memory_usage_bytes}
        return memory_info


class CATSNoiseEstimationResult(CATSResult):

    def plot(self, ind, time_interval_sec=None):
        fig, opts, inds_slices = super().plot(ind, time_interval_sec)
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')

        dframe = self.noise_stationary_intervals.diff(axis=-1)[0]
        i_frame = give_index_slice_by_limits(time_interval_sec, dframe)

        inds_frames = ind + (slice(None), i_frame)

        frames = np.arange(dframe / 2, self.noise_stationary_intervals.shape[0] * dframe, dframe)

        fig4 = hv.Image((frames, self.stft_frequency,
                         self.noise_std[inds_frames]), kdims=[t_dim, f_dim],
                        label='4. Noise spectrogram')

        return (fig + fig4).opts(*opts).cols(1)




