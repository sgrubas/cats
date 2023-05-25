"""
    API for Detector based on Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDetector : Detector of seismic events based on CATS
        CATSDetectionResult : keeps all the results and can plot sample trace with step-by-step visualization
"""

import numpy as np
import holoviews as hv
from typing import Callable, Union, Tuple

from .baseclass import CATSBaseSTFT, CATSResult
from .core.utils import get_interval_division, aggregate_array_by_axis_and_func
from .core.utils import cast_to_bool_dict, del_vals_by_keys, give_rectangles
from .core.association import PickFeatures
from .core.projection import GiveIntervals, FillGaps


class CATSDetector(CATSBaseSTFT):
    """
        Detector of events based on Cluster Analysis of Trimmed Spectrograms
    """
    aggregate_axis_for_likelihood: Union[int, Tuple[int]] = None
    aggregate_func_for_likelihood: Callable[[np.ndarray], np.ndarray] = np.max
    pick_features: bool = False

    def detect(self, x, /, verbose=False, full_info=False):
        """
            Performs the detection on the given dataset.

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
                            - "likelihood" - projected `spectrogram_SNR_clustered`
                            - "picked_features - extract peak values and their onset times of `likelihood`
        """

        full_info = cast_to_bool_dict(full_info,
                                      ["coefficients", "spectrogram", "noise_threshold", "noise_std",
                                       "spectrogram_SNR_trimmed", "spectrogram_SNR_clustered",
                                       "spectrogram_cluster_ID", "likelihood"])
        full_info["picked_features"] = self.pick_features
        save_clustered = full_info['spectrogram_SNR_clustered']
        save_cluster_ID = full_info['spectrogram_cluster_ID']
        full_info['spectrogram_SNR_clustered'] = True
        full_info['spectrogram_cluster_ID'] = True

        result, history = super()._apply(x, finish_on='clustering', verbose=verbose, full_info=full_info)

        full_info['spectrogram_SNR_clustered'] = save_clustered
        full_info['spectrogram_cluster_ID'] = save_cluster_ID

        with history(current_process='Likelihood'):
            # Aggregation
            result['spectrogram_SNR_clustered'] = \
                aggregate_array_by_axis_and_func(result['spectrogram_SNR_clustered'],
                                                 self.aggregate_axis_for_likelihood,
                                                 self.aggregate_func_for_likelihood,
                                                 min_last_dims=2)

            counts = np.count_nonzero(result['spectrogram_SNR_clustered'], axis=-2)
            counts = np.where(counts == 0, 1, counts)

            result['likelihood'] = result['spectrogram_SNR_clustered'].sum(axis=-2) / counts
            result['detection'] = FillGaps(result['spectrogram_cluster_ID'].max(axis=-2))

        del counts
        del_vals_by_keys(result, full_info, ['spectrogram_SNR_clustered', 'spectrogram_cluster_ID'])

        stft_time = self.STFT.forward_time_axis(x.shape[-1])
        # Picking
        # peak_freqs = self.stft_frequency[np.argmax(SNRK_aggr, axis=-2)]
        if full_info['picked_features']:
            with history(current_process='Picking'):
                result['picked_features'] = PickFeatures(result['likelihood'],
                                                         time=stft_time,
                                                         min_likelihood=self.minSNR,
                                                         min_width_sec=self.cluster_size_t_sec,
                                                         num_features=None)

        del_vals_by_keys(result, full_info, ['likelihood'])

        history.print_total_time()

        time = np.arange(x.shape[-1]) * self.dt_sec
        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        kwargs = {"signal": x,
                  "coefficients": result.get('coefficients', None),
                  "spectrogram": result.get('spectrogram', None),
                  "noise_threshold": result.get('noise_threshold', None),
                  "noise_std": result.get('noise_std', None),
                  "spectrogram_SNR_trimmed": result.get('spectrogram_SNR_trimmed', None),
                  "spectrogram_SNR_clustered": result.get('spectrogram_SNR_clustered', None),
                  "spectrogram_cluster_ID": result.get('spectrogram_cluster_ID', None),
                  "detection": result.get('detection', None),
                  "likelihood": result.get('likelihood', None),
                  "picked_features": result.get('picked_features', None),
                  "time": time,
                  "stft_time": stft_time,
                  "stft_frequency": self.stft_frequency,
                  "stationary_intervals": stft_time[frames],
                  "minSNR": self.minSNR,
                  "history": history}

        return CATSDetectionResult(**kwargs)

    def __mul__(self, x):
        return self.detect(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.detect(x, verbose=True, full_info=True)


class CATSDetectionResult(CATSResult):

    def plot(self, ind, time_interval_sec=None):
        fig, opts, inds_slices = super().plot(ind, time_interval_sec)
        t_dim = hv.Dimension('Time', unit='s')
        L_dim = hv.Dimension('Likelihood')

        i_stft = inds_slices[2]
        inds_stft = inds_slices[0] + (i_stft,)

        ref_kw_opts = opts[-1].kwargs
        t1, t2 = ref_kw_opts['xlim']
        likelihood = np.nan_to_num(self.likelihood[inds_stft],
                                   posinf=10*self.minSNR,
                                   neginf=-10*self.minSNR)  # POSSIBLE `NAN` AND `INF` VALUES!
        likelihood_fig = hv.Curve((self.stft_time[i_stft], likelihood),
                                  kdims=[t_dim], vdims=L_dim)

        if self.picked_features is not None:
            P = self.picked_features[ind]
            P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
            peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
            peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim).opts(marker='D', color='r')
        else:
            peaks_fig = None

        intervals = GiveIntervals(self.detection[inds_stft])
        interv_height = (likelihood.max() / 2) * 1.1
        rectangles = give_rectangles(intervals, self.stft_time[i_stft], [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles,
                                      kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                             linewidth=0,
                                                                             alpha=0.2)
        snr_level_fig = hv.HLine(self.minSNR, kdims=[t_dim, L_dim]).opts(color='k', linestyle='--', alpha=0.7)

        last_figs = [intervals_fig, snr_level_fig, likelihood_fig]
        if peaks_fig is not None:
            last_figs.append(peaks_fig)

        fig4 = hv.Overlay(last_figs,
                          label='4. Projection: $L_k$').opts(**ref_kw_opts)

        return (fig + fig4).opts(*opts).cols(1)




