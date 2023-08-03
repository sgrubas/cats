"""
    API for Denoiser based on Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDenoiser : Denoiser of seismic events based on CATS
        CATSDenoisingResult : keeps all the results and can plot sample trace with step-by-step visualization
"""


from typing import Callable, Union, Tuple, List, Any

import holoviews as hv
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

from .baseclass import CATSBase, CATSResult
from .core.association import PickDetectedPeaks
from .core.projection import FilterDetection
from .core.utils import cast_to_bool_dict, del_vals_by_keys
from .core.utils import format_index_by_dimensions, give_index_slice_by_limits
from .core.utils import get_interval_division, aggregate_array_by_axis_and_func, format_interval_by_limits
from .core.plottingutils import plot_traces
from .io import read_data


class CATSDenoiser(CATSBase):
    """
        Denoiser prototype based on Cluster Analysis of Trimmed Spectrograms
    """
    aggregate_axis_for_likelihood: Union[int, Tuple[int]] = None
    aggregate_func_for_likelihood: Callable[[np.ndarray], np.ndarray] = np.max

    def _denoise(self, x, /, verbose=False, full_info=False):
        full_info, pre_full_info = self.parse_info_dict(full_info)

        result, history = super()._apply(x, finish_on='clustering', verbose=verbose, full_info=pre_full_info)

        stft_time = self.STFT.forward_time_axis(x.shape[-1])

        with history(current_process='Likelihood'):
            # Aggregation
            counts = np.count_nonzero(result['spectrogram_SNR_clustered'], axis=-2)
            counts = np.where(counts == 0, 1, counts).astype(result['spectrogram_SNR_clustered'].dtype)

            result['likelihood'] = result['spectrogram_SNR_clustered'].sum(axis=-2) / counts

            del counts
            del_vals_by_keys(result, full_info, ['spectrogram_SNR_clustered'])

            result['likelihood'] = aggregate_array_by_axis_and_func(result['likelihood'],
                                                                    self.aggregate_axis_for_likelihood,
                                                                    self.aggregate_func_for_likelihood,
                                                                    min_last_dims=1)

        with history(current_process='Detecting intervals'):
            result['detection'], result['detected_intervals'] = \
                FilterDetection(result['likelihood'] > 0.0,
                                min_separation=self.min_separation_len, min_duration=self.min_duration_len)

            result['picked_features'] = PickDetectedPeaks(result['likelihood'], result['detected_intervals'],
                                                          dt=self.stft_hop_sec, t0=stft_time[0])

            result['detected_intervals'] = result['detected_intervals'] * self.stft_hop_sec + stft_time[0]

        del_vals_by_keys(result, full_info, ['likelihood', 'detection', 'detected_intervals', 'picked_features'])

        with history(current_process='Inverse STFT'):
            result['denoised_signal'] = self.STFT / ((result['spectrogram_SNR_clustered'] > 0) * result['coefficients'])
            result['denoised_signal'] = result['denoised_signal'][..., :x.shape[-1]]

        del_vals_by_keys(result, full_info, ['coefficients', 'denoised_signal'])

        history.print_total_time()

        frames = get_interval_division(N=len(stft_time), L=self.stationary_frame_len)

        from_full_info = {kw: result.get(kw, None) for kw in full_info}

        return CATSDenoisingResult(dt_sec=self.dt_sec,
                                   stft_dt_sec=self.stft_hop_sec,
                                   stft_t0_sec=stft_time[0],
                                   npts=x.shape[-1],
                                   stft_npts=len(stft_time),
                                   stft_frequency=self.stft_frequency,
                                   noise_stationary_intervals=frames * self.stft_hop_sec + stft_time[0],
                                   minSNR=self.minSNR,
                                   history=history,
                                   aggregate_axis_for_likelihood=self.aggregate_axis_for_likelihood,
                                   **from_full_info)

    def denoise(self, x: np.ndarray,
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
                            - "detection" - binary classification [noise / signal]
                            - "detected_intervals" - detected intervals (start, end) in seconds
                            - "picked_features" - features in intervals [onset, peak likelihood]
                            - "denoised_signal" - denoised signal from sparse TF representation (always returned)
        """
        n_chunks = self.split_data_by_memory(x, full_info=full_info, to_file=False)
        single_chunk = n_chunks > 1
        data_chunks = np.array_split(x, n_chunks, axis=-1)
        results = []
        for dc in tqdm(data_chunks, desc='Data chunks', display=single_chunk):
            results.append(self._denoise(dc, verbose=verbose, full_info=full_info))
        return CATSDenoisingResult.concatenate(*results)

    def __mul__(self, x):
        return self.denoise(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.denoise(x, verbose=True, full_info=True)

    def parse_info_dict(self, full_info):
        info_keys = self._info_keys()
        info_keys.extend(["likelihood", "detection", "detected_intervals", "picked_features", "denoised_signal"])
        if isinstance(full_info, str) and \
           (full_info in ['plot', 'plotting', 'plt', 'qc', 'main']):  # only those needed for plotting step-by-step
            full_info = {'signal':                    True,
                         'spectrogram':               True,
                         'spectrogram_SNR_trimmed':   True,
                         'spectrogram_SNR_clustered': True,
                         'denoised_signal':           True}

        full_info = cast_to_bool_dict(full_info, info_keys)

        full_info["denoised_signal"] = True     # default saved result

        pre_full_info = full_info.copy()
        pre_full_info['coefficients'] = True  # needed for `CATSBase._apply`
        pre_full_info['spectrogram_SNR_clustered'] = True  # needed for `CATSBase._apply`
        return full_info, pre_full_info

    def memory_usage_estimate(self, x, /, full_info=False):
        memory_usage_bytes, used_together = self._memory_usage(x)
        full_info, pre_full_info = self.parse_info_dict(full_info)

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
            "denoised_signal":    memory_usage_bytes['signal'],             # float / int
        }
        memory_usage_bytes.update(memory_usage_bytes_new)

        # overriden since `coefficients` are needed till the end
        used_together = [('spectrogram', 'noise_threshold', 'noise_std', 'spectrogram_SNR_trimmed'),
                         ('spectrogram_SNR_trimmed', 'spectrogram_SNR_clustered', 'spectrogram_cluster_ID'),
                         ('spectrogram_SNR_clustered',  'likelihood'),
                         ("likelihood", "detection", "picked_features", "detected_intervals")]

        base_info = ["signal", 'coefficients', "stft_frequency", "noise_stationary_intervals"]

        return self.memory_info(memory_usage_bytes, used_together, base_info, full_info)

    def split_data_by_memory(self, x, /, full_info, to_file):
        memory_info = self.memory_usage_estimate(x, full_info=full_info)
        return self.memory_chunks(memory_info, to_file)

    @staticmethod
    def basefunc_denoise_to_file(detector, x, /, path_destination, verbose=False, full_info=False, compress=False):

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
            detector.denoise(fc, verbose=verbose, full_info=full_info).save(path_i, compress=compress)
            if verbose:
                print("Result" + f" chunk {i}" * (not single_chunk), f"has been saved to `{path_i}`",
                      sep=' ', end='\n\n')

    def denoise_to_file(self, x, /, path_destination, verbose=False, full_info=False, compress=False):
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
        self.basefunc_denoise_to_file(self, x, path_destination=path_destination, verbose=verbose, full_info=full_info,
                                      compress=compress)

    @staticmethod
    def basefunc_denoise_on_files(detector, data_folder, data_format, result_folder, verbose, full_info, compress):
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
        self.basefunc_denoise_on_files(self, data_folder=data_folder, data_format=data_format,
                                       result_folder=result_folder, verbose=verbose, full_info=full_info,
                                       compress=compress)


class CATSDenoisingResult(CATSResult):
    likelihood: Any = None
    detection: Any = None
    detected_intervals: Any = None
    picked_features: Any = None
    aggregate_axis_for_likelihood: int = None

    denoised_signal: Any = None

    def plot(self,
             ind: Tuple[int] = None,
             time_interval_sec: Tuple[float] = None,):

        fig, opts, inds_slices, time_interval_sec = super().plot(ind, time_interval_sec)
        t_dim = hv.Dimension('Time', unit='s')
        a_dim = hv.Dimension('Amplitude')

        fig4 = hv.Curve((self.time(time_interval_sec),
                         self.denoised_signal[inds_slices[:-1]]), kdims=[t_dim], vdims=a_dim,
                        label='0. Input data: $x(t)$').opts(linewidth=0.5)

        return (fig + fig4).opts(*opts).cols(1)

    def plot_traces(self,
                    ind: Tuple[int] = None,
                    time_interval_sec: Tuple[float] = None,
                    intervals: bool = True,
                    picks: bool = True,
                    trace_loc: np.ndarray = None,
                    gain: int = 1,
                    clip: bool = True,
                    each_trace: int = 1,
                    **kwargs):

        ind = format_index_by_dimensions(ind=ind, shape=self.signal.shape[:-1], slice_dims=1, default_ind=0)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.npts - 1) * self.dt_sec))
        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        traces = self.signal[ind + (i_time,)]

        if (ax := self.aggregate_axis_for_likelihood) is not None:
            if ax < len(ind): ind = list(ind); ind[ax] = 0; ind = tuple(ind)

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
                        "noise_threshold",
                        "noise_std",
                        "spectrogram_SNR_trimmed",
                        "spectrogram_SNR_clustered",
                        "spectrogram_cluster_ID",
                        "likelihood",
                        "detection",
                        "denoised_signal"]

        for name in concat_attrs:
            self._concat(other, name, -1)

        stft_t0 = self.stft_dt_sec * self.stft_npts  # must be calculated before `self.stft_npts += other.stft_npts`

        self._concat(other, "noise_stationary_intervals", 0, stft_t0)
        self._concat(other, "detected_intervals", -2, stft_t0)
        self._concat(other, "picked_features", -2, stft_t0, (..., 0))

        self.npts += other.npts
        self.stft_npts += other.stft_npts

        self.history.merge(other.history)

        assert self.minSNR == other.minSNR
        assert self.dt_sec == other.dt_sec
        assert self.stft_dt_sec == other.stft_dt_sec
        assert np.all(self.stft_frequency == other.stft_frequency)


