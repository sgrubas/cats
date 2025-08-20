"""
    - under development


    API for Denoiser based on Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDenoiser : Denoiser of seismic events based on CATS
        CATSDenoisingResult : keeps all the results and can plot sample trace with step-by-step visualization
"""


from typing import Union, Tuple, List, Literal

import holoviews as hv
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
import obspy

from .baseclass import CATSBase, CATSResult
from .core.utils import make_default_index_if_outrange, del_vals_by_keys
from .core.utils import give_rectangles, StatusKeeper, intervals_intersection
from .core.clustering import index_cluster_catalog
from .io import read_data, convert_dict_to_stream

# TODO:
#   - denoise to/on files: support `obspy.Stream`

# ------------------ CATS DENOISER API ------------------ #


class CATSDenoiser(CATSBase):
    """
        Denoiser of seismic events based on Cluster Analysis of Trimmed Spectrograms

        background_weight : float : weight of the background noise in the inverse STFT. If None, then 0.0

    """
    inverse_transform: Literal[True] = True  # Inverse transform is always True for denoising

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
        del_vals_by_keys(result, full_info, ['spectrogram_trim_mask'] * self.full_aggregated_mask)

        # Clustering
        self.apply_func(func_name='apply_Clustering', result_container=result, status_keeper=history,
                        process_name='Clustering')

        # Phase separation
        if self.phase_separation is not None:
            self.apply_func(func_name='apply_PhaseSeparation', result_container=result, status_keeper=history,
                            process_name='Phase separation', tf_time=stft_time, frequencies=self.stft_frequency)

        # Cluster catalog
        self.cluster_catalogs_opts.setdefault('update_cluster_ID', True)  # update cluster_ID for inverse STFT
        self.apply_func(func_name='apply_ClusterCatalogs', result_container=result,
                        tf_time=stft_time, frequencies=self.stft_frequency, status_keeper=history,
                        process_name='Cluster catalog')

        del_vals_by_keys(result, full_info, ['spectrogram', 'spectrogram_trim_mask_aggr'])

        # Inverse STFT
        self.apply_func(func_name='apply_ISTFT', result_container=result, status_keeper=history,
                        process_name='Inverse STFT', N=x.shape[-1])

        del_vals_by_keys(result, full_info, ['spectrogram_trim_mask', 'coefficients', 'spectrogram_cluster_ID'])

        from_full_info = {kw: result.get(kw, None) for kw in full_info}

        result = CATSDenoisingResult(dt_sec=self.dt_sec,
                                     tf_dt_sec=self.stft_hop_sec,
                                     tf_t0_sec=stft_time[0],
                                     time_npts=x.shape[-1],
                                     tf_time_npts=len(stft_time),
                                     frequencies=self.stft_frequency,
                                     time_frames=result['time_frames'] * self.stft_hop_sec + stft_time[0],
                                     minSNR=self.minSNR,
                                     history=history,
                                     frequency_groups_indexes=self.frequency_groups_indexes,
                                     main_params=self.main_params,
                                     **from_full_info)

        history.print_total_time()

        return result

    def denoise(self, x: Union[np.ndarray, obspy.Stream],
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
                            - "spectrogram_trim_mask" - `spectrogram / noise_std` trimmed by `noise_threshold`
                            - "spectrogram_cluster_ID" - cluster indexes on `spectrogram_SNR_clustered`
                            - "signal_denoised" - inverse STFT of `coefficients` * (`spectrogram_SNR_clustered` > 0)
        """

        data, stats = self.convert_input_data(x)

        n_chunks = self.split_data_by_memory(data, full_info=full_info, to_file=False)
        single_chunk = n_chunks > 1
        data_chunks = np.array_split(data, n_chunks, axis=-1)
        results = []
        for dc in tqdm(data_chunks, desc='Data chunks', display=single_chunk):
            results.append(self._denoise(dc, verbose=verbose, full_info=full_info))

        result = CATSDenoisingResult.concatenate(*results)
        setattr(result, "stats", stats)
        return result

    def apply(self, x: Union[np.ndarray, obspy.Stream],
              /,
              verbose: bool = False,
              full_info: Union[bool, str, List[str]] = False):
        """ Alias of `.denoise`. For compatibility with CATSBase """
        return self.denoise(x, verbose=verbose, full_info=full_info)

    # @classmethod
    # def get_all_keys(cls):
    #     return super(cls, cls).get_all_keys() + ["signal_denoised"]

    @classmethod
    def get_qc_keys(cls):
        return super(cls, cls).get_qc_keys() + ["signal_denoised"]

    @classmethod
    def get_default_keys(cls):
        return super(cls, cls).get_default_keys() + ['signal_denoised']

    def memory_usage_estimate(self, x, /, full_info=False):
        memory_usage_bytes, _ = self._memory_usage(x)
        full_info = self.parse_info_dict(full_info)

        memory_usage_bytes_new = {
                "signal_denoised":         memory_usage_bytes['signal'],        # float
        }
        memory_usage_bytes.update(memory_usage_bytes_new)

        used_together = [('coefficients', 'spectrogram'),  # STFT step
                         ('coefficients', 'spectrogram', 'noise_threshold_conversion', 'noise_std',
                          'spectrogram_SNR', "spectrogram_trim_mask", "spectrogram_trim_mask_aggr"),  # BEDATE step
                         ('coefficients', 'spectrogram_trim_mask_aggr', 'spectrogram_cluster_ID'),  # Clustering
                         ('coefficients', 'signal_denoised')]  # Inverse STFT

        base_info = ["signal", "frequencies", "time_frames"]

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
            save_path = result_folder / rel_folder / Path(str(fi.name).replace(data_format, "_denoising"))
            x = read_data(fi)['data']
            denoiser.denoise_to_file(x, save_path, verbose=verbose, full_info=full_info, compress=compress)
            del x

    def denoise_on_files(self, data_folder, data_format, result_folder=None,
                         verbose=False, full_info=False, compress=False):
        """
            Performs the denoising the same as `denoise_to_file` method, but directly on folder with data.
        """
        self.basefunc_denoise_on_files(self, data_folder=data_folder, data_format=data_format,
                                       result_folder=result_folder, verbose=verbose, full_info=full_info,
                                       compress=compress)


class CATSDenoisingResult(CATSResult):

    @classmethod
    def get_qc_keys(cls):
        return CATSDenoiser.get_qc_keys()

    def _plot(self,
              weighted_coefficients,
              intervals,
              picks,
              fig,  # from CATSResult.plot()
              opts,  # from CATSResult.plot()
              inds_slices,  # from CATSResult.plot()
              time_interval_sec  # from CATSResult.plot()
              ):

        t1, t2 = time_interval_sec

        t_dim, a_dim = fig[0].dimensions()  # get dim params (no matter what fig index)
        f_dim = fig[1].dimensions()[1]  # get Freq dim from spectrogram fig

        ind, i_time, i_stft = inds_slices
        trace_slice = ind + (i_time,)

        # Weighted coefficients
        if weighted_coefficients:
            psd_fig = fig[1]  # original PSD
            psd_vdim = fig[1].dimensions()[2]

            sparse_spectr = fig[2]
            dim_key = sparse_spectr.dimensions()[2].name
            # take sparse spectrogram data values
            if isinstance(sparse_spectr, hv.QuadMesh):
                C = sparse_spectr.data[dim_key]
            else:  # hv.Overlay
                overlay_dim = list(sparse_spectr.keys())[0]
                C = sparse_spectr.data[overlay_dim].data[dim_key]

            weights = np.where(C > 0, 1.0, self.main_params['background_weight'])
            WPSD = psd_fig.data[psd_vdim.name] * weights  # multiply original PSD by weights
            clim = psd_fig.opts['clim']

            w_opts = (hv.opts.QuadMesh(clim=clim, backend='matplotlib'),
                      hv.opts.QuadMesh(clim=clim, backend='bokeh'))
            fig31 = hv.QuadMesh((psd_fig.data[t_dim.name], psd_fig.data[f_dim.name], WPSD),
                                kdims=[t_dim, f_dim], vdims=psd_vdim,
                                label='3.2. Weighted amplitude spectrogram: $|X(t,f) \cdot W(t,f)|$').opts(*w_opts)

            fig = fig + fig31  # Append the weighted spectrogram

        # Denoised signal
        fig4 = hv.Curve((fig[0].data[t_dim.name], self.signal_denoised[trace_slice]),
                        kdims=[t_dim], vdims=a_dim).opts(*opts[:2])  # apply opts for Curve

        last_fig = [fig4]

        aggr_ind = make_default_index_if_outrange(ind, self.spectrogram_cluster_ID.shape[:-2], default_ind_value=0)
        cluster_catalogs = getattr(self, 'cluster_catalogs', None)
        catalog = index_cluster_catalog(cluster_catalogs, aggr_ind).copy() if (cluster_catalogs is not None) else None

        if (catalog is not None) and intervals:
            detected_intervals = catalog[['Time_start_sec', 'Time_end_sec']].values
            intervals = intervals_intersection(detected_intervals, (t1, t2))
            interv_height = np.max(abs(self.signal_denoised[trace_slice])) * 1.1

            rectangles = give_rectangles([intervals], [0.0], interv_height)
            intervals_fig = hv.Rectangles(rectangles, kdims=[t_dim, a_dim, 't2', 'l2'])

            mpl_rect_opts = hv.opts.Rectangles(color='blue', linewidth=0, alpha=0.15, backend='matplotlib')
            bkh_rect_opts = hv.opts.Rectangles(fill_color='blue', line_width=0, alpha=0.15, backend='bokeh')
            intervals_fig = intervals_fig.opts(mpl_rect_opts, bkh_rect_opts)

            last_fig.append(intervals_fig)

        if (catalog is not None) and picks:
            arrival_cols = list(filter(lambda x: "arrival_sec" in x, self.cluster_catalogs.columns))
            vals = catalog.get(arrival_cols, None)
            vals = catalog.get('Time_peak_sec', None) if vals is None else vals

            mpl_vlines_opts = hv.opts.VLine(color='black', linewidth=2, backend='matplotlib')
            bkh_vlines_opts = hv.opts.VLine(color='black', line_width=2, backend='bokeh')
            vlines_opts = (mpl_vlines_opts, bkh_vlines_opts)
            if vals is not None:
                A = vals.values.flatten()
                A = A[(t1 <= A) & (A <= t2)]
                last_fig += [hv.VLine(ai, kdims=[t_dim, a_dim]).opts(*vlines_opts)
                             for ai in A]

        bkh_overlay_opts = hv.opts.Overlay(height=int(opts[1].kwargs['height'] * 1.15), backend='bokeh')
        fig4 = hv.Overlay(last_fig, label=r'4. Denoised data: $\tilde{s}(t)$').opts(bkh_overlay_opts)

        fig = (fig + fig4).opts(*opts).cols(1)

        return fig

    def plot(self,
             ind: Tuple[int] = None,
             time_interval_sec: Tuple[float] = None,
             SNR_spectrograms: bool = False,
             show_cluster_ID: bool = False,
             show_aggregated: bool = False,
             detrend_type: str = 'constant',
             weighted_coefficients: bool = False,
             intervals: bool = False,
             picks: bool = False,
             aggr_func: callable = np.max,
             interactive: bool = False
             ):

        fig, opts, inds_slices, time_interval_sec = super().plot(ind=ind,
                                                                 time_interval_sec=time_interval_sec,
                                                                 SNR_spectrograms=SNR_spectrograms,
                                                                 show_cluster_ID=show_cluster_ID,
                                                                 show_aggregated=show_aggregated,
                                                                 detrend_type=detrend_type,
                                                                 aggr_func=aggr_func,
                                                                 interactive=interactive,
                                                                 frequencies=self.frequencies)
        fig = self._plot(weighted_coefficients=weighted_coefficients,
                         intervals=intervals,
                         picks=picks,
                         fig=fig,
                         opts=opts,
                         inds_slices=inds_slices,
                         time_interval_sec=time_interval_sec)

        if show_cluster_ID:
            ind = -2 - weighted_coefficients
            fig31 = fig[ind].opts(hv.opts.QuadMesh(norm=None, backend='matplotlib'),
                                  hv.opts.QuadMesh(logz=False, backend='bokeh'))

        return fig

    def plot_traces(self,
                    show_denoised: bool = True,
                    ind: Tuple[int] = None,
                    time_interval_sec: Tuple[float] = None,
                    intervals: bool = False,
                    picks: bool = False,
                    station_loc: np.ndarray = None,
                    gain: int = 1,
                    detrend_type: str = 'constant',
                    clip: bool = False,
                    each_station: int = 1,
                    amplitude_scale: float = None,
                    per_station_scale: bool = False,
                    component_labels: list[str] = None,
                    station_labels: list[str] = None,
                    interactive: bool = False,
                    allow_picking: bool = False,
                    **kwargs):

        fig = super().plot_traces(signal=self.signal_denoised if show_denoised else self.signal,
                                  ind=ind,
                                  time_interval_sec=time_interval_sec,
                                  intervals=intervals,
                                  picks=picks,
                                  station_loc=station_loc,
                                  gain=gain,
                                  clip=clip,
                                  each_station=each_station,
                                  amplitude_scale=amplitude_scale,
                                  per_station_scale=per_station_scale,
                                  component_labels=component_labels,
                                  station_labels=station_labels,
                                  detrend_type=detrend_type,
                                  interactive=interactive,
                                  allow_picking=allow_picking,
                                  **kwargs)
        return fig

    def filter_and_update_result(self, cluster_catalogs_filter):
        super().filter_and_update_result(cluster_catalogs_filter)
        # TODO:
        #   - inverse transform

    @property
    def _concat_attr_list(self):
        return super()._concat_attr_list + ["signal_denoised"]

    def get_denoised_obspy_stream(self):
        return convert_dict_to_stream({"data": self.signal_denoised,
                                       "stats": self.stats})

    def save_denoised_obspy_stream(self, filename, format):
        """
            Saves via `obspy.Stream.write`
        """
        stream = self.get_denoised_obspy_stream()
        stream.write(filename=filename, format=format)
