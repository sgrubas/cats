"""
    API for Detector based on Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDetector : Detector of seismic events based on CATS
        CATSDetectionResult : keeps all the results and can plot sample trace with step-by-step visualization
"""

from typing import Callable, Union, Tuple, List, Any

import holoviews as hv
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

from .baseclass import CATSBase, CATSResult
from .core.association import PickDetectedPeaks
from .core.projection import FilterIntervalsFromClusterLabels
from .core.clustering import concatenate_arrays_of_cluster_catalogs
from .core.utils import cast_to_bool_dict, del_vals_by_keys, give_rectangles, to2d_array_with_num_columns
from .core.utils import format_index_by_dimensions, give_index_slice_by_limits, intervals_intersection, StatusKeeper
from .core.utils import aggregate_array_by_axis_and_func, format_interval_by_limits, make_default_index_on_axis
from .core.plottingutils import plot_traces
from .io import read_data


# ------------------ CATS DETECTOR API ------------------ #

class CATSDetector(CATSBase):
    """
        Detector of events based on Cluster Analysis of Trimmed Spectrograms
    """
    aggregate_axis_for_likelihood: Union[int, Tuple[int]] = None
    aggregate_func_for_likelihood: Callable[[np.ndarray], np.ndarray] = np.max

    def apply_Likelihood(self, result_container, /, bandpass_slice, full_info):
        # Aggregation
        counts = np.count_nonzero(result_container['spectrogram_SNR_clustered'][bandpass_slice], axis=-2)
        counts = np.where(counts == 0, 1, counts).astype(result_container['spectrogram_SNR_clustered'].dtype)

        result_container['likelihood'] = \
            result_container['spectrogram_SNR_clustered'][bandpass_slice].sum(axis=-2) / counts

        del counts
        del_vals_by_keys(result_container, full_info, ['spectrogram_SNR_clustered'])

        result_container['likelihood'] = aggregate_array_by_axis_and_func(result_container['likelihood'],
                                                                          self.aggregate_axis_for_likelihood,
                                                                          self.aggregate_func_for_likelihood,
                                                                          min_last_dims=1)

    def apply_Intervals(self, result_container, /, full_info, t0):
        result_container['detection'], result_container['detected_intervals'] = \
            FilterIntervalsFromClusterLabels(result_container['spectrogram_cluster_ID'].max(axis=-2))

        del_vals_by_keys(result_container, full_info, ['spectrogram_cluster_ID'])

        result_container['picked_features'] = PickDetectedPeaks(result_container['likelihood'],
                                                                result_container['detected_intervals'],
                                                                dt=self.stft_hop_sec, t0=t0)

        result_container['detected_intervals'] = result_container['detected_intervals'] * self.stft_hop_sec + t0

    def _detect(self, x, /, verbose=False, full_info=False):
        full_info = self.parse_info_dict(full_info)

        result = dict.fromkeys(full_info.keys())
        result['signal'] = x
        history = StatusKeeper(verbose=verbose)

        # STFT
        self.apply_func(func_name='apply_STFT', result_container=result, status_keeper=history,
                        process_name='STFT')
        del_vals_by_keys(result, full_info, ['signal', 'coefficients'])

        stft_time = self.STFT.forward_time_axis(x.shape[-1])
        bandpass_slice = (..., self.freq_bandpass_slice, slice(None))

        # B-E-DATE
        self.apply_func(func_name='apply_BEDATE', result_container=result, status_keeper=history,
                        process_name='B-E-DATE trimming')
        del_vals_by_keys(result, full_info, ['spectrogram', 'noise_std', 'noise_threshold_conversion'])

        # Clustering
        self.apply_func(func_name='apply_Clustering', result_container=result, status_keeper=history,
                        process_name='Clustering')
        del_vals_by_keys(result, full_info, ['spectrogram_SNR_trimmed'])

        # Likelihood projection
        self.apply_func(func_name='apply_Likelihood', result_container=result, status_keeper=history,
                        process_name='Likelihood', bandpass_slice=bandpass_slice, full_info=full_info)

        # Detecting intervals
        self.apply_func(func_name='apply_Intervals', result_container=result, status_keeper=history,
                        process_name='Detecting intervals', full_info=full_info, t0=stft_time[0])

        del_vals_by_keys(result, full_info, ['likelihood', 'detection', 'detected_intervals', 'picked_features'])

        history.print_total_time()

        from_full_info = {kw: result.get(kw, None) for kw in full_info}

        return CATSDetectionResult(dt_sec=self.dt_sec,
                                   stft_dt_sec=self.stft_hop_sec,
                                   stft_t0_sec=stft_time[0],
                                   npts=x.shape[-1],
                                   stft_npts=len(stft_time),
                                   stft_frequency=self.stft_frequency,
                                   time_frames=result['time_frames'] * self.stft_hop_sec + stft_time[0],
                                   minSNR=self.minSNR,
                                   history=history,
                                   aggregate_axis_for_likelihood=self.aggregate_axis_for_likelihood,
                                   cluster_catalogs=result['cluster_catalogs'],
                                   frequency_groups=self.frequency_groups,
                                   **from_full_info)

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
                            - "noise_std" - noise level, standard deviation
                            - "noise_threshold_conversion" - conversion to threshold from `noise_std`
                            - "spectrogram_SNR_trimmed" - `spectrogram / noise_std` trimmed by `noise_threshold`
                            - "spectrogram_SNR_clustered" - clustered `spectrogram_SNR_trimmed`
                            - "spectrogram_cluster_ID" - cluster indexes on `spectrogram_SNR_clustered`
                            - "likelihood" - projected `spectrogram_SNR_clustered`
                            - "detection" - binary classification [noise / signal]
                            - "detected_intervals" - detected intervals (start, end) in seconds (always returned)
                            - "picked_features" - features in intervals [onset, peak likelihood] (always returned)
        """
        n_chunks = self.split_data_by_memory(x, full_info=full_info, to_file=False)
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

    @staticmethod
    def get_qc_keys():
        return ["signal", "spectrogram", "spectrogram_SNR_trimmed", "spectrogram_SNR_clustered",
                "likelihood", "detected_intervals", "picked_features"]

    def parse_info_dict(self, full_info):
        info_keys = self._info_keys()
        info_keys.extend(["likelihood", "detection", "detected_intervals", "picked_features"])
        if isinstance(full_info, str) and \
           (full_info in ['plot', 'plotting', 'plt', 'qc', 'main']):  # only those needed for plotting step-by-step
            full_info = self.get_qc_keys()

        full_info = cast_to_bool_dict(full_info, info_keys)

        full_info["detected_intervals"] = True  # default saved result
        full_info["picked_features"] = True     # default saved result

        return full_info

    def memory_usage_estimate(self, x, /, full_info=False):
        memory_usage_bytes, used_together = self._memory_usage(x)
        full_info = self.parse_info_dict(full_info)

        precision_order = memory_usage_bytes['signal'] / x.size
        stft_time_len = len(self.STFT.forward_time_axis(x.shape[-1]))

        likelihood_shape = x.shape[:-1] + (stft_time_len,)
        likelihood_size = np.prod(likelihood_shape)

        aggregated_axis_len = x.shape[ax] if (ax := self.aggregate_axis_for_likelihood) is not None else 1
        n, mod = divmod(stft_time_len, self.min_duration_len + self.min_separation_len)
        num_intervals = n + mod // self.min_duration_len  # upper bound estimate
        intervals_size = 2 * np.prod(x.shape[:-1]) / aggregated_axis_len * num_intervals

        memory_usage_bytes_new = {
            "likelihood":         1. * precision_order * likelihood_size,   # float
            "detection":          1. * likelihood_size,                     # bool always
            "detected_intervals": 8. * intervals_size,                      # ~ float64, upper bound
            "picked_features":    8. * intervals_size,                      # ~ float64, upper bound
        }
        memory_usage_bytes.update(memory_usage_bytes_new)

        used_together.append(('spectrogram_SNR_clustered', 'spectrogram_cluster_ID', 'likelihood'))
        used_together.append(("likelihood", "detection", "picked_features", "detected_intervals"))
        base_info = ["signal", "stft_frequency", "time_frames"]

        return self.memory_info(memory_usage_bytes, used_together, base_info, full_info)

    def split_data_by_memory(self, x, /, full_info, to_file):
        memory_info = self.memory_usage_estimate(x, full_info=full_info)
        return self.memory_chunks(memory_info, to_file)

    @staticmethod
    def basefunc_detect_to_file(detector, x, /, path_destination, verbose=False, full_info=False, compress=False):

        path_destination = Path(path_destination)
        path_destination.parent.mkdir(parents=True, exist_ok=True)
        if path_destination.name[-4:] == '.mat':
            filepath = path_destination.name[:-4]
        else:
            filepath = path_destination.as_posix()

        n_chunks = detector.split_data_by_memory(x, full_info=full_info, to_file=True)
        single_chunk = n_chunks <= 1
        file_chunks = np.array_split(x, n_chunks, axis=-1)
        for i, fc in enumerate(tqdm(file_chunks,
                                    desc='File chunks',
                                    disable=single_chunk)):
            chunk_suffix = (f'_chunk_{i}' * (not single_chunk))
            path_i = filepath + chunk_suffix + '.mat'
            detector.detect(fc, verbose=verbose, full_info=full_info).save(path_i, compress=compress)
            if verbose:
                print("Result" + f" chunk {i}" * (not single_chunk), f"has been saved to `{path_i}`",
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
        self.basefunc_detect_to_file(self, x, path_destination=path_destination, verbose=verbose,
                                     full_info=full_info, compress=compress)

    @staticmethod
    def basefunc_detect_on_files(detector, data_folder, data_format, result_folder, verbose, full_info, compress):
        folder = Path(data_folder)
        data_format = "." * ("." not in data_format) + data_format
        result_folder = Path(result_folder) if (result_folder is not None) else folder
        files = [fi for fi in folder.rglob("*" + data_format)]

        for fi in (pbar := tqdm(files, desc="Files")):
            pbar.set_postfix({"File": fi})
            rel_folder = fi.relative_to(folder).parent
            save_path = result_folder / rel_folder / Path(str(fi.name).replace(data_format, "_detection"))
            x = read_data(fi)['data']
            detector.denoise_to_file(x, save_path, verbose=verbose, full_info=full_info, compress=compress)
            del x

    def detect_on_files(self, data_folder, data_format, result_folder=None,
                        verbose=False, full_info=False, compress=False):
        """
            Performs the detection the same as `detect_to_file` method, but directly on folder with data.
        """
        self.basefunc_detect_on_files(self, data_folder=data_folder, data_format=data_format,
                                      result_folder=result_folder, verbose=verbose, full_info=full_info,
                                      compress=compress)


class CATSDetectionResult(CATSResult):
    likelihood: Any = None
    detection: Any = None
    detected_intervals: Any = None
    picked_features: Any = None
    aggregate_axis_for_likelihood: int = None

    def plot(self,
             ind: Tuple[int] = None,
             time_interval_sec: Tuple[float] = None,):

        fig, opts, inds_slices, time_interval_sec = super().plot(ind, time_interval_sec)
        t_dim = hv.Dimension('Time', unit='s')
        L_dim = hv.Dimension('Likelihood')

        ind = inds_slices[0]
        if (ax := self.aggregate_axis_for_likelihood) is not None:
            ind = make_default_index_on_axis(ind, ax, 0)
        inds_stft = ind + (inds_slices[2],)

        stft_time = self.stft_time(time_interval_sec)

        ref_kw_opts = opts[-1].kwargs
        t1, t2 = ref_kw_opts['xlim']
        likelihood = np.nan_to_num(self.likelihood[inds_stft],
                                   posinf=1e8, neginf=-1e8)  # POSSIBLE `NAN` AND `INF` VALUES!
        likelihood_fig = hv.Curve((stft_time, likelihood),
                                  kdims=[t_dim], vdims=L_dim)

        P = self.picked_features[ind]
        P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
        peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
        peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim).opts(marker='D', color='r')

        intervals = intervals_intersection(self.detected_intervals[ind], (t1, t2))

        interv_height = (np.max(likelihood) / 2) * 1.1
        rectangles = give_rectangles([intervals], [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles,
                                      kdims=[t_dim, L_dim, 't2', 'l2']).opts(color='blue',
                                                                             linewidth=0,
                                                                             alpha=0.2)
        snr_level_fig = hv.HLine(self.minSNR, kdims=[t_dim, L_dim]).opts(color='k', linestyle='--', alpha=0.7)

        last_figs = [intervals_fig, snr_level_fig, likelihood_fig, peaks_fig]

        fig4 = hv.Overlay(last_figs, label='4. Likelihood and Detection: '
                                           '$\mathcal{L}(t)$ and $\mathcal{D}(t)$').opts(**ref_kw_opts)

        return (fig + fig4).opts(*opts).cols(1)

    def plot_traces(self,
                    ind: Tuple[int] = None,
                    time_interval_sec: Tuple[float] = None,
                    intervals: bool = True,
                    picks: bool = True,
                    trace_loc: np.ndarray = None,
                    gain: int = 1,
                    clip: bool = False,
                    each_trace: int = 1,
                    **kwargs):

        ind = format_index_by_dimensions(ind=ind, shape=self.signal.shape[:-1], slice_dims=1, default_ind=0)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.npts - 1) * self.dt_sec))
        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        traces = self.signal[ind + (i_time,)]

        if (ax := self.aggregate_axis_for_likelihood) is not None:
            if ax < len(ind):
                ind = make_default_index_on_axis(ind, ax, 0)

        detected_intervals = getattr(self, 'detected_intervals', None) if intervals else None
        if detected_intervals is not None:
            detected_intervals = detected_intervals[ind]

        picked_features = getattr(self, 'picked_features', None) if picks else None
        if picked_features is not None:
            pf = picked_features[ind]
            picked_onsets = np.empty(pf.shape, dtype=object)
            for i, pi in np.ndenumerate(pf):
                picked_onsets[i] = pi[..., 0]
        else:
            picked_onsets = None

        fig = plot_traces(traces, self.time(time_interval_sec),
                          intervals=detected_intervals, picks=picked_onsets, associated_picks=None,
                          trace_loc=trace_loc, time_interval_sec=time_interval_sec,
                          gain=gain, clip=clip, each_trace=each_trace, **kwargs)
        return fig

    def append(self, other):
        concat_attrs = ["signal",
                        "coefficients",
                        "spectrogram",
                        "noise_std",
                        "spectrogram_SNR_trimmed",
                        "spectrogram_SNR_clustered",
                        "spectrogram_cluster_ID",
                        "likelihood",
                        "detection"]

        # update cluster id
        attr = "spectrogram_cluster_ID"
        if ((self_attr := getattr(self, attr, None)) is not None) and \
                ((other_attr := getattr(other, attr, None)) is not None):
            shape = other_attr.shape[:-2]
            for ind in np.ndindex(shape):
                other_attr[ind][other_attr[ind] > 0] += self_attr[ind].max()
            setattr(other, attr, other_attr)

        for name in concat_attrs:
            self._concat(other, name, -1)

        t0 = self.time_frames[-1, -1] + self.stft_dt_sec

        self._concat(other, "time_frames", 0, t0)
        self._concat(other, "detected_intervals", -2, t0)
        self._concat(other, "picked_features", -2, t0, (..., 0))

        self.cluster_catalogs = concatenate_arrays_of_cluster_catalogs(self.cluster_catalogs,
                                                                       other.cluster_catalogs, t0)

        self.npts += other.npts
        self.stft_npts += other.stft_npts

        self.history.merge(other.history)

        assert self.minSNR == other.minSNR
        assert self.dt_sec == other.dt_sec
        assert self.stft_dt_sec == other.stft_dt_sec
        assert np.all(self.stft_frequency == other.stft_frequency)
        assert np.all(self.noise_threshold_conversion == other.noise_threshold_conversion)

    @staticmethod
    def convert_dict_to_attributes(mdict):
        mdict = CATSResult.convert_dict_to_attributes(mdict)

        if (detected_intervals := mdict.get('detected_intervals', None)) is not None:
            for ind, intv in np.ndenumerate(detected_intervals):
                detected_intervals[ind] = to2d_array_with_num_columns(intv, num_columns=2)

        if (picked_features := mdict.get('picked_features', None)) is not None:
            for ind, feats in np.ndenumerate(picked_features):
                picked_features[ind] = to2d_array_with_num_columns(feats, num_columns=2)

        return mdict
