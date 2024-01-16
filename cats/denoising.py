"""
    - under development


    API for Denoiser based on Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDenoiser : Denoiser of seismic events based on CATS
        CATSDenoisingResult : keeps all the results and can plot sample trace with step-by-step visualization
"""


from typing import Union, Tuple, List, Any

import holoviews as hv
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

from .baseclass import CATSBase, CATSResult
from .core.utils import cast_to_bool_dict, del_vals_by_keys
from .core.utils import format_index_by_dimensions, give_index_slice_by_limits, give_rectangles
from .core.utils import format_interval_by_limits, StatusKeeper, intervals_intersection
from .core.clustering import concatenate_arrays_of_cluster_catalogs
from .core.plottingutils import plot_traces
from .io import read_data


# ------------------ CATS DENOISER API ------------------ #

class CATSDenoiser(CATSBase):
    """
        Denoiser of seismic events based on Cluster Analysis of Trimmed Spectrograms
    """

    def apply_ISTFT(self, result_container, /, N):
        sparse = result_container['coefficients'] * (result_container['spectrogram_SNR_clustered'] > 0)
        result_container['signal_denoised'] = (self.STFT / sparse)[..., :N]

    def _denoise(self, x, /, verbose=False, full_info=False):
        full_info = self.parse_info_dict(full_info)

        result = dict.fromkeys(full_info.keys())
        result['signal'] = x
        history = StatusKeeper(verbose=verbose)

        # STFT
        self.apply_func(func_name='apply_STFT', result_container=result, status_keeper=history,
                        process_name='STFT')
        del_vals_by_keys(result, full_info, ['signal'])
        stft_time = self.STFT.forward_time_axis(x.shape[-1])

        # B-E-DATE
        self.apply_func(func_name='apply_BEDATE', result_container=result, status_keeper=history,
                        process_name='B-E-DATE trimming')
        del_vals_by_keys(result, full_info, ['spectrogram', 'noise_std', 'noise_threshold_conversion'])

        # Clustering
        self.apply_func(func_name='apply_Clustering', result_container=result, status_keeper=history,
                        process_name='Clustering')
        del_vals_by_keys(result, full_info, ['spectrogram_SNR_trimmed', 'spectrogram_cluster_ID'])

        # Inverse STFT
        self.apply_func(func_name='apply_ISTFT', result_container=result, status_keeper=history,
                        process_name='Inverse STFT', N=x.shape[-1])

        del_vals_by_keys(result, full_info, ['coefficients', 'spectrogram_SNR_clustered'])

        history.print_total_time()

        from_full_info = {kw: result.get(kw, None) for kw in full_info}

        return CATSDenoisingResult(dt_sec=self.dt_sec,
                                   stft_dt_sec=self.stft_hop_sec,
                                   stft_t0_sec=stft_time[0],
                                   npts=x.shape[-1],
                                   stft_npts=len(stft_time),
                                   stft_frequency=self.stft_frequency,
                                   time_frames=result['time_frames'] * self.stft_hop_sec + stft_time[0],
                                   minSNR=self.minSNR,
                                   history=history,
                                   cluster_catalogs=result['cluster_catalogs'],
                                   frequency_groups=self.frequency_groups,
                                   main_params=self.export_main_params(),
                                   **from_full_info)

    def denoise(self, x: np.ndarray,
                /,
                verbose: bool = False,
                full_info: Union[bool, str, List[str]] = False):
        """
            Performs denoising of a given dataset. If the data processing does not fit the available memory,
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
                            - "signal_denoised" - inverse STFT of `coefficients` * (`spectrogram_SNR_clustered` > 0)
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

    @staticmethod
    def get_qc_keys():
        return ["signal", "spectrogram", "spectrogram_SNR_trimmed", "spectrogram_SNR_clustered", "signal_denoised"]

    def parse_info_dict(self, full_info):
        info_keys = self._info_keys()
        info_keys.extend(["signal_denoised"])
        if isinstance(full_info, str) and \
           (full_info in ['plot', 'plotting', 'plt', 'qc', 'main']):  # only those needed for plotting step-by-step
            full_info = self.get_qc_keys()

        full_info = cast_to_bool_dict(full_info, info_keys)

        full_info["signal_denoised"] = True     # default saved result

        return full_info

    def memory_usage_estimate(self, x, /, full_info=False):
        memory_usage_bytes, _ = self._memory_usage(x)
        full_info = self.parse_info_dict(full_info)

        memory_usage_bytes_new = {
                "signal_denoised":         memory_usage_bytes['signal'],        # float
        }
        memory_usage_bytes.update(memory_usage_bytes_new)

        used_together = [('coefficients', 'spectrogram'),
                         ('coefficients', 'spectrogram', 'noise_threshold_conversion',
                          'noise_std', 'spectrogram_SNR_trimmed'),
                         ('coefficients', 'spectrogram_SNR_trimmed', 'spectrogram_SNR_clustered',
                          'spectrogram_cluster_ID'),
                         ('coefficients', 'signal_denoised')]

        base_info = ["signal", "stft_frequency", "time_frames"]

        return self.memory_info(memory_usage_bytes, used_together, base_info, full_info)

    def split_data_by_memory(self, x, /, full_info, to_file):
        memory_info = self.memory_usage_estimate(x, full_info=full_info)
        return self.memory_chunks(memory_info, to_file)

    @staticmethod
    def basefunc_denoise_to_file(denoiser, x, /, path_destination, verbose=False, full_info=False, compress=False):

        path_destination = Path(path_destination)
        path_destination.parent.mkdir(parents=True, exist_ok=True)
        if path_destination.name[-4:] == '.mat':
            filepath = path_destination.name[:-4]
        else:
            filepath = path_destination.as_posix()

        n_chunks = denoiser.split_data_by_memory(x, full_info=full_info, to_file=True)
        single_chunk = n_chunks <= 1
        file_chunks = np.array_split(x, n_chunks, axis=-1)
        for i, fc in enumerate(tqdm(file_chunks,
                                    desc='File chunks',
                                    disable=single_chunk)):
            chunk_suffix = (f'_chunk_{i}' * (not single_chunk))
            path_i = filepath + chunk_suffix + '.mat'
            denoiser.denoise(fc, verbose=verbose, full_info=full_info).save(path_i, compress=compress)
            if verbose:
                print("Result" + f" chunk {i}" * (not single_chunk), f"has been saved to `{path_i}`",
                      sep=' ', end='\n\n')

    def denoise_to_file(self, x, /, path_destination, verbose=False, full_info=False, compress=False):
        """
            Performs the denoising the same as `denoise` method, but also saves the results into files and
            more flexible with memory issues.

            Arguments:
                x : np.ndarray (..., N) : input data with any number of dimensions, but the last axis `N` must be Time.
                path_destination : str : path with filename to save in.
                verbose : bool : whether to print status and timing
                full_info : bool / str / dict[str, bool] : whether to save workflow stages for further quality control.
                        If `True`, then all stages are saved, if `False` then only the `signal_denoised` is saved.
                        If string "plot" or "plt" is given, then only stages needed for plotting are saved.
                        For available workflow stages in `full_info` see function `.detect()`.
                compress : bool : whether to compress the saved data
        """
        self.basefunc_denoise_to_file(self, x, path_destination=path_destination, verbose=verbose, full_info=full_info,
                                      compress=compress)

    @staticmethod
    def basefunc_denoise_on_files(denoiser, data_folder, data_format, result_folder, verbose, full_info, compress):
        folder = Path(data_folder)
        data_format = "." * ("." not in data_format) + data_format
        result_folder = Path(result_folder) if (result_folder is not None) else folder
        files = [fi for fi in folder.rglob("*" + data_format)]

        for fi in (pbar := tqdm(files, desc="Files")):
            pbar.set_postfix({"File": fi})
            rel_folder = fi.relative_to(folder).parent
            save_path = result_folder / rel_folder / Path(str(fi.name).replace(data_format, "_detection"))
            x = read_data(fi)['data']
            denoiser.denoise_to_file(x, save_path, verbose=verbose, full_info=full_info, compress=compress)
            del x

    def detect_on_files(self, data_folder, data_format, result_folder=None,
                        verbose=False, full_info=False, compress=False):
        """
            Performs the denoising the same as `denoise_to_file` method, but directly on folder with data.
        """
        self.basefunc_denoise_on_files(self, data_folder=data_folder, data_format=data_format,
                                       result_folder=result_folder, verbose=verbose, full_info=full_info,
                                       compress=compress)


class CATSDenoisingResult(CATSResult):
    signal_denoised: Any = None

    def plot(self,
             ind: Tuple[int] = None,
             time_interval_sec: Tuple[float] = None,
             intervals: bool = False,
             picks: bool = False,
             ):

        fig, opts, inds_slices, time_interval_sec = super().plot(ind, time_interval_sec)
        ref_kw_opts = opts[-1].kwargs
        t1, t2 = ref_kw_opts['xlim']

        t_dim = hv.Dimension('Time', unit='s')
        a_dim = hv.Dimension('Amplitude')
        ind, i_time, i_stft = inds_slices
        trace_slice = ind + (i_time,)
        fig4 = hv.Curve((self.time(time_interval_sec), self.signal_denoised[trace_slice]),
                        kdims=[t_dim], vdims=a_dim).opts(linewidth=0.5)

        last_fig = [fig4]

        cluster_catalogs = getattr(self, 'cluster_catalogs', None) if (intervals or picks) else None
        catalog = cluster_catalogs[ind] if cluster_catalogs is not None else None

        if (catalog is not None) and intervals:
            detected_intervals = catalog[['Time_start_sec', 'Time_end_sec']].values
            intervals = intervals_intersection(detected_intervals, (t1, t2))
            interv_height = np.max(abs(self.signal_denoised[trace_slice])) * 1.1
            rectangles = give_rectangles([intervals], [0.0], interv_height)
            intervals_fig = hv.Rectangles(rectangles,
                                          kdims=[t_dim, a_dim, 't2', 'l2']).opts(color='blue',
                                                                                 linewidth=0,
                                                                                 alpha=0.2)
            last_fig.append(intervals_fig)

        if (catalog is not None) and picks:
            P = catalog['Time_center_of_mass_sec'].values
            P = P[(t1 <= P) & (P <= t2)]
            picks_fig = [hv.VLine(pi, kdims=[t_dim, a_dim]) for pi in P]
            picks_fig = hv.Overlay(picks_fig).opts(hv.opts.VLine(color='r'))
            last_fig.append(picks_fig)

        fig4 = hv.Overlay(last_fig, label='4. Denoised data: $\\tilde{s}(t)$').opts(**ref_kw_opts)

        return (fig + fig4).opts(*opts).cols(1)

    def plot_traces(self,
                    ind: Tuple[int] = None,
                    time_interval_sec: Tuple[float] = None,
                    intervals: bool = False,
                    picks: bool = False,
                    show_denoised: bool = True,
                    trace_loc: np.ndarray = None,
                    gain: int = 1,
                    clip: bool = False,
                    each_trace: int = 1,
                    amplitude_scale: float = None,
                    **kwargs):
        signal = self.signal_denoised if show_denoised else self.signal
        ind = format_index_by_dimensions(ind=ind, shape=signal.shape[:-1], slice_dims=1, default_ind=0)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.npts - 1) * self.dt_sec))
        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        traces = signal[ind + (i_time,)]

        cluster_catalogs = getattr(self, 'cluster_catalogs', None) if (intervals or picks) else None
        catalog = cluster_catalogs[ind] if cluster_catalogs is not None else None

        if (catalog is not None) and intervals:
            detected_intervals = np.empty(catalog.shape, dtype=object)
            for i, cati in np.ndenumerate(catalog):
                detected_intervals[i] = cati[['Time_start_sec', 'Time_end_sec']].values
        else:
            detected_intervals = None

        if (catalog is not None) and picks:
            picked_onsets = np.empty(catalog.shape, dtype=object)
            for i, cati in np.ndenumerate(catalog):
                picked_onsets[i] = cati['Time_center_of_mass_sec'].values
        else:
            picked_onsets = None

        fig = plot_traces(traces, self.time(time_interval_sec),
                          intervals=detected_intervals, picks=picked_onsets, associated_picks=None,
                          trace_loc=trace_loc, time_interval_sec=time_interval_sec, gain=gain, clip=clip,
                          each_trace=each_trace, amplitude_scale=amplitude_scale, **kwargs)
        return fig

    def append(self, other):
        concat_attrs = ["signal",
                        "coefficients",
                        "spectrogram",
                        "noise_std",
                        "spectrogram_SNR_trimmed",
                        "spectrogram_SNR_clustered",
                        "spectrogram_cluster_ID",
                        "signal_denoised"]

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
