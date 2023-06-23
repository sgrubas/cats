"""
    API for Detector based on Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDetector : Detector of seismic events based on CATS
        CATSDetectionResult : keeps all the results and can plot sample trace with step-by-step visualization
"""

from typing import Callable, Union, Tuple, List

import holoviews as hv
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

from .baseclass import CATSBase, CATSResult
from .core.association import PickFeatures
from .core.projection import GiveIntervals, FillGaps
from .core.utils import cast_to_bool_dict, del_vals_by_keys, give_rectangles
from .core.utils import get_interval_division, aggregate_array_by_axis_and_func
from .io import read_data


# TODO:
#   - add StatusKeeper to auto data processing on files


class CATSDetector(CATSBase):
    """
        Detector of events based on Cluster Analysis of Trimmed Spectrograms
    """
    aggregate_axis_for_likelihood: Union[int, Tuple[int]] = None
    aggregate_func_for_likelihood: Callable[[np.ndarray], np.ndarray] = np.max
    pick_features: bool = False

    def _detect(self, x, /, verbose=False, full_info=False):

        full_info, pre_full_info = self._parse_info_dict(full_info)

        result, history = super()._apply(x, finish_on='clustering', verbose=verbose, full_info=pre_full_info)

        with history(current_process='Likelihood'):
            # Aggregation
            result['spectrogram_SNR_clustered'] = \
                aggregate_array_by_axis_and_func(result['spectrogram_SNR_clustered'],
                                                 self.aggregate_axis_for_likelihood,
                                                 self.aggregate_func_for_likelihood,
                                                 min_last_dims=2)

            counts = np.count_nonzero(result['spectrogram_SNR_clustered'], axis=-2)
            counts = np.where(counts == 0, 1, counts).astype(result['spectrogram_SNR_clustered'].dtype)

            result['likelihood'] = result['spectrogram_SNR_clustered'].sum(axis=-2) / counts
            result['detection'] = FillGaps(result['spectrogram_cluster_ID'].max(axis=-2))

        del counts
        del_vals_by_keys(result, full_info, ['spectrogram_SNR_clustered',
                                             'spectrogram_cluster_ID',
                                             'detection'])

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

        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        from_full_info = {kw: result.get(kw, None) for kw in full_info}
        kwargs = {**from_full_info,
                  "dt_sec": self.dt_sec,
                  "stft_dt_sec": self.stft_hop_sec,
                  "stft_t0_sec": stft_time[0],
                  "npts": x.shape[-1],
                  "stft_npts": len(stft_time),
                  "stft_frequency": self.stft_frequency,
                  "stationary_intervals": stft_time[frames],
                  "minSNR": self.minSNR,
                  "history": history}

        return CATSDetectionResult(**kwargs)

    def detect(self, x: np.ndarray,
               /,
               verbose: bool = False,
               full_info: Union[bool, str, List[str]] = False):
        """
            Performs the detection on the given dataset. If the data processing does not fit the available memory,
            the data are split into chunks.

            Arguments:
                x : np.ndarray (..., N) : input data with any number of dimensions, but the last axis `N` must be Time.
                verbose : bool : whether to print status and timing
                full_info : bool / str / List[str] : whether to save workflow stages for further quality control.
                        If `True`, then all stages are saved, if `False` then only the `detection` is saved.
                        If string "plot" or "plt" is given, then only stages needed for plotting are saved.
                        Available workflow stages, if any is listed then saved to result:
                            - "signal" - input signal
                            - "coefficients" - STFT coefficients
                            - "spectrogram" - absolute value of `coefficients`
                            - "noise_threshold" - threshold defined by `noise_std`
                            - "noise_std" - noise level, standard deviation
                            - "spectrogram_SNR_trimmed" - `spectrogram / noise_std` trimmed by `noise_threshold`
                            - "spectrogram_SNR_clustered" - clustered `spectrogram_SNR_trimmed`
                            - "spectrogram_cluster_ID" - cluster indexes on `spectrogram_SNR_clustered`
                            - "likelihood" - projected `spectrogram_SNR_clustered`
                            - "detection" - binary classification [noise / signal], always returned.
        """
        n_chunks = self._split_data_by_memory(x, full_info=full_info, to_file=False)
        single_chunk = n_chunks > 1
        data_chunks = np.array_split(x, n_chunks, axis=-1)
        results = []
        for dc in tqdm(data_chunks, desc='Data chunks', display=single_chunk):
            results.append(self._detect(dc, verbose=verbose, full_info=full_info))
        return CATSDetectionResult.concatenate(*results)

    def __mul__(self, x):
        return self.detect(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.detect(x, verbose=True, full_info=True)

    def _parse_info_dict(self, full_info):
        info_keys = self._info_keys()
        info_keys.extend(["likelihood", "detection", "picked_features"])
        if isinstance(full_info, str) and \
           (full_info in ['plot', 'plotting', 'plt', 'qc', 'main']):  # only those needed for plotting step-by-step
            full_info = {'signal':                    True,
                         'spectrogram':               True,
                         'spectrogram_SNR_trimmed':   True,
                         'spectrogram_SNR_clustered': True,
                         'likelihood':                True}
        full_info = cast_to_bool_dict(full_info, info_keys)

        full_info["detection"] = True  # default saved result
        full_info["picked_features"] = self.pick_features  # non-default result

        pre_full_info = full_info.copy()
        pre_full_info['spectrogram_SNR_clustered'] = True  # needed for `CATSBase._apply`
        pre_full_info['spectrogram_cluster_ID'] = True     # needed for `CATSBase._apply`
        return full_info, pre_full_info

    def memory_usage_estimate(self, x, /, full_info=False):
        memory_usage_bytes, used_together = self._memory_usage(x)
        full_info, pre_full_info = self._parse_info_dict(full_info)

        stft_time_len = len(self.STFT.forward_time_axis(x.shape[-1]))

        likelihood_shape = x.shape[:-1] + (stft_time_len,)
        likelihood_size = np.prod(likelihood_shape)

        precision_order = memory_usage_bytes['signal'] / x.size

        memory_usage_bytes_new = {
            "likelihood":        1. * precision_order * likelihood_size,   # float
            "detection":         1. * likelihood_size,                     # bool always
        }
        memory_usage_bytes.update(memory_usage_bytes_new)

        used_together.append(('spectrogram_SNR_clustered', 'spectrogram_cluster_ID', 'likelihood', 'detection'))
        base_info = ["signal", "stft_frequency", "stationary_intervals"]

        return self.memory_info(memory_usage_bytes, used_together, base_info, full_info)

    def _split_data_by_memory(self, x, /, full_info, to_file):
        memory_info = self.memory_usage_estimate(x, full_info=full_info)
        return self.memory_chunks(memory_info, to_file)

    @staticmethod
    def _detect_to_file(detector, x, /, path_destination, verbose=False, full_info=False, compress=False):

        path_destination = Path(path_destination)
        path_destination.parent.mkdir(parents=True, exist_ok=True)
        if path_destination.name[-4:] == '.mat':
            filepath = path_destination.name[:-4]
        else:
            filepath = path_destination.as_posix()

        n_chunks = detector._split_data_by_memory(x, full_info=full_info, to_file=True)
        multiple_chunks = n_chunks > 1
        file_chunks = np.array_split(x, n_chunks, axis=-1)
        for i, fc in enumerate(tqdm(file_chunks,
                                    desc='File chunks',
                                    display=multiple_chunks)):
            path_i = filepath + (f'_chunk_{i}' * multiple_chunks) + '.mat'
            detector.detect(fc, verbose=verbose, full_info=full_info).save(path_i, compress=compress)
            if verbose:
                print("Result" + f" chunk {i}" * multiple_chunks, f"has been saved to `{path_i}`",
                      sep=' ', end='\n\n')

    def detect_to_file(self, x, /, path_destination, verbose=False, full_info=False, compress=False):
        """
            Performs the detection the same as `detect` method, but also saves the results into files and
            more flexible with memory issues.

            Arguments:
                x : np.ndarray (..., N) : input data with any number of dimensions, but the last axis `N` must be Time.
                path_destination : str : path with filename to save in.
                verbose : bool : whether to print status and timing
                full_info : bool / str / dict[str, bool] : whether to save workflow stages for further quality control.
                        If `True`, then all stages are saved, if `False` then only the `detection` is saved.
                        If string "plot" or "plt" is given, then only stages needed for plotting are saved.
                        For available workflow stages in `full_info` see function `.detect()`.
                compress : bool : whether to compress the saved data
        """
        self._detect_to_file(self, x,
                             path_destination=path_destination,
                             verbose=verbose, full_info=full_info, compress=compress)

    @staticmethod
    def _detect_on_files(detector, data_folder, data_format, result_folder, verbose, full_info, compress):
        folder = Path(data_folder)
        data_format = "." * ("." not in data_format) + data_format
        result_folder = Path(result_folder) if (result_folder is not None) else folder
        files = [fi for fi in folder.rglob("*" + data_format)]

        for fi in (pbar := tqdm(files, desc="Files")):
            pbar.set_postfix({"File": fi})
            rel_folder = fi.relative_to(folder).parent
            save_path = result_folder / rel_folder / Path(str(fi.name).replace(data_format, "_detection"))
            x = read_data(fi)['data']
            detector.detect_to_file(x, save_path, verbose=verbose, full_info=full_info, compress=compress)
            del x

    def detect_on_files(self, data_folder, data_format, result_folder=None,
                        verbose=False, full_info=False, compress=False):
        """
            Performs the detection the same as `detect_to_file` method, but directly on folder with data.
        """
        self._detect_on_files(self,
                              data_folder=data_folder, data_format=data_format, result_folder=result_folder,
                              verbose=verbose, full_info=full_info, compress=compress)


class CATSDetectionResult(CATSResult):

    def plot(self, ind, time_interval_sec=None):
        fig, opts, inds_slices = super().plot(ind, time_interval_sec)
        t_dim = hv.Dimension('Time', unit='s')
        L_dim = hv.Dimension('Likelihood')

        i_stft = inds_slices[2]
        inds_stft = inds_slices[0] + (i_stft,)

        stft_time = self.stft_time(time_interval_sec)

        ref_kw_opts = opts[-1].kwargs
        t1, t2 = ref_kw_opts['xlim']
        likelihood = np.nan_to_num(self.likelihood[inds_stft],
                                   posinf=10 * self.minSNR,
                                   neginf=-10 * self.minSNR)  # POSSIBLE `NAN` AND `INF` VALUES!
        likelihood_fig = hv.Curve((stft_time, likelihood),
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
        rectangles = give_rectangles(intervals, stft_time, [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles,
                                      kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                             linewidth=0,
                                                                             alpha=0.2)
        snr_level_fig = hv.HLine(self.minSNR, kdims=[t_dim, L_dim]).opts(color='k', linestyle='--', alpha=0.7)

        last_figs = [intervals_fig, snr_level_fig, likelihood_fig]
        if peaks_fig is not None:
            last_figs.append(peaks_fig)

        fig4 = hv.Overlay(last_figs, label='4. Projection: $L_k$').opts(**ref_kw_opts)

        return (fig + fig4).opts(*opts).cols(1)

    def append(self, other):
        super().append(other)
        self._concat(other, "detection", -1)

