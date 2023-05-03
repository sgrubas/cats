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
from .core.utils import cast_to_bool_dict, del_vals_by_keys, StatusMessenger, give_rectangles
from .core.association import PickWithFeatures
from .core.projection import GiveIntervals, FillGaps


class CATSDetector(CATSBaseSTFT):
    """
        Detector of events based on Cluster Analysis of Trimmed Spectrograms
    """
    aggregate_axis_for_likelihood: Union[int, Tuple[int]] = None
    aggregate_func_for_likelihood: Callable[[np.ndarray], np.ndarray] = np.max

    def detect(self, x, /, verbose=False, full_info=False):
        """
            Performs the detection on the given dataset.

            Arguments:
                x : np.ndarray (..., N) : input data with any number of dimensions, but the last axis `N` must be Time.
        """
        full_info = cast_to_bool_dict(full_info, ["X", "PSD", "Eta", "Sgm", "SNR",
                                                  "SNRK", "K", 'detection', 'likelihood'])
        save_SNRK = full_info['SNRK']
        save_K = full_info['K']
        full_info['SNRK'] = True
        full_info['K'] = True

        result = super()._apply(x, finish_on='clustering', verbose=verbose, full_info=full_info)

        full_info['SNRK'] = save_SNRK
        full_info['K'] = save_K

        with StatusMessenger(verbose=verbose, operation='4. Likelihood'):
            # Aggregation
            result['SNRK'] = aggregate_array_by_axis_and_func(result['SNRK'],
                                                              self.aggregate_axis_for_likelihood,
                                                              self.aggregate_func_for_likelihood,
                                                              min_last_dims=2)
            counts = np.count_nonzero(result['SNRK'], axis=-2)
            counts = np.where(counts == 0, 1, counts)
            result['likelihood'] = result['SNRK'].sum(axis=-2) / counts

            # Detection projection
            # if full_info['detection']:

            result['detection'] = FillGaps(result['K'].max(axis=-2))

        del counts
        del_vals_by_keys(result, full_info, ['SNRK', 'K'])

        # Picking
        # peak_freqs = self.stft_frequency[np.argmax(SNRK_aggr, axis=-2)]
        with StatusMessenger(verbose=verbose, operation='5. Picking onsets'):
            stft_time = self.STFT.forward_time_axis(x.shape[-1])
            result['picked_features'] = PickWithFeatures(result['likelihood'],
                                                         time=stft_time,
                                                         min_likelihood=self.minSNR,
                                                         min_width_sec=self.cluster_size_t_sec,
                                                         num_features=None)

        del_vals_by_keys(result, full_info, ['likelihood'])

        time = np.arange(x.shape[-1]) * self.dt_sec
        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        kwargs = {"signal": x,
                  "coefficients": result.get('X', None),
                  "spectrogram": result.get('PSD', None),
                  "noise_thresholding": result.get('Eta', None),
                  "noise_std": result.get('Sgm', None),
                  "spectrogramSNR_trimmed": result.get('SNR', None),
                  "spectrogramSNR_clustered": result.get('SNRK', None),
                  "spectrogramID_cluster": result.get('K', None),
                  "detection": result.get('detection', None),
                  "likelihood": result.get('likelihood', None),
                  "picked_features": result.get('picked_features', None),
                  "time": time,
                  "stft_time": stft_time,
                  "stft_frequency": self.stft_frequency,
                  "stationary_intervals": stft_time[frames],
                  "minSNR": self.minSNR}

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
        i_st = inds_slices[2]
        inds_st = inds_slices[0] + (i_st,)
        ref_kw_opts = opts[-1].kwargs
        t1, t2 = ref_kw_opts['xlim']
        likelihood = self.likelihood[inds_st]
        likelihood_fig = hv.Curve((self.stft_time[i_st], likelihood),
                                  kdims=[t_dim], vdims=L_dim)

        P = self.picked_features[ind]
        P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
        peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
        peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim).opts(marker='D', color='r')

        intervals = GiveIntervals(self.detection[inds_st])
        interv_height = (likelihood.max() / 2) * 1.1
        rectangles = give_rectangles(intervals, self.stft_time[i_st], [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles, kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                                         linewidth=0, alpha=0.1)
        snr_level_fig = hv.HLine(self.minSNR, kdims=[t_dim, L_dim]).opts(color='k', linestyle='--', alpha=0.7)

        fig4 = hv.Overlay([intervals_fig, snr_level_fig, likelihood_fig, peaks_fig],
                          label='4. Projection: $L_k$').opts(**ref_kw_opts)

        return (fig + fig4).opts(*opts).cols(1)




