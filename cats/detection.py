"""
    API for Detector based on Cluster Analysis of Trimmed Spectrograms (CATS)
    Main operators:
        CATSDetector : Detector of seismic events based on CATS
        CATSDetectionResult : keeps all the results and can plot sample trace with step-by-step visualization
"""

from typing import Union, Tuple, List, Any

import holoviews as hv
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm
import obspy
import datetime

from .baseclass import CATSBase, CATSResult
# from .core.association import PickDetectedPeaks
from .core.projection import ProjectCatalogs
from .core.utils import del_vals_by_keys, give_rectangles, to2d_array_with_num_columns, make_default_index_if_outrange
from .core.utils import give_index_slice_by_limits, intervals_intersection, StatusKeeper
from .core.clustering import index_cluster_catalog
from .io import read_data


# TODO:
#   - detect to/on files: support `obspy.Stream`

# ------------------ CATS DETECTOR API ------------------ #

class CATSDetector(CATSBase):
    """
        Detector of events based on Cluster Analysis of Trimmed Spectrograms
    """

    full_aggregated_mask: Union[bool, None] = True

    def apply_Projection(self, result_container, /, bandpass_slice, full_info):
        if full_info.get("likelihood", False):
            # Project SNR on time axis
            SNR, CID = (result_container['spectrogram_SNR'][bandpass_slice],
                        result_container['spectrogram_cluster_ID'][bandpass_slice])
            # counts = np.count_nonzero(clustered, axis=-2)
            # counts[counts == 0] = 1  # normalization by number of nonzero elements, not by all frequencies
            result_container['likelihood'] = (SNR * (CID > 0)).sum(axis=-2)  # removed the normalization, for simplicity

        # Extracts intervals of events
        trace_shape = result_container['spectrogram_cluster_ID'].shape[:-2]
        intervals, features = ProjectCatalogs(trace_shape,
                                              result_container['cluster_catalogs'],
                                              dt_sec=self.stft_hop_sec * 0.5,
                                              min_separation_sec=self.cluster_distance_t_sec,
                                              min_duration_sec=0.0)

        result_container['detected_intervals'] = intervals
        result_container['picked_features'] = features

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
        del_vals_by_keys(result, full_info, ['spectrogram_SNR', 'spectrogram_trim_mask',
                                             'noise_std', 'noise_threshold_conversion'])

        # Clustering
        self.apply_func(func_name='apply_Clustering', result_container=result, status_keeper=history,
                        process_name='Clustering')

        # Phase separation
        if self.phase_separation is not None:
            self.apply_func(func_name='apply_PhaseSeparation', result_container=result, status_keeper=history,
                            process_name='Phase separation', tf_time=stft_time, frequencies=self.stft_frequency)

        # Cluster catalog
        self.cluster_catalogs_opts.setdefault('update_cluster_ID', True)  # no need to update cluster_ID
        self.apply_func(func_name='apply_ClusterCatalogs', result_container=result,
                        tf_time=stft_time, frequencies=self.stft_frequency, status_keeper=history,
                        process_name='Cluster catalog')
        del_vals_by_keys(result, full_info, ['spectrogram'])

        # Projecting intervals
        self.apply_func(func_name='apply_Projection', result_container=result, status_keeper=history,
                        process_name='Projecting intervals', bandpass_slice=bandpass_slice, full_info=full_info)
        del_vals_by_keys(result, full_info, ['spectrogram_SNR', 'spectrogram_cluster_ID',
                                             'likelihood', 'detection', 'cluster_catalogs'])

        history.print_total_time()

        from_full_info = {kw: result.get(kw, None) for kw in full_info}

        result = CATSDetectionResult(dt_sec=self.dt_sec,
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

        return result

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
                        If `True`, then all stages are saved, if `False` then only the `detected_intervals' &
                        'picked_features` are saved.
                        If string "plot" or "plt" is given, then only stages needed for plotting are saved.
                        Available workflow stages, if any is listed then saved to result:
                            - "signal" - input signal
                            - "coefficients" - STFT coefficients
                            - "spectrogram" - absolute value of `coefficients`
                            - "noise_std" - noise level, standard deviation
                            - "noise_threshold_conversion" - conversion to threshold from `noise_std`
                            - "spectrogram_trim_mask" - `spectrogram / noise_std` trimmed by `noise_threshold`
                            - "spectrogram_cluster_ID" - cluster indexes on `spectrogram_SNR_clustered`
                            - "likelihood" - projected `spectrogram_SNR_clustered`
                            - "detected_intervals" - detected intervals (start, end) in seconds (always returned)
                            - "picked_features" - features in intervals [onset, peak likelihood] (always returned)
        """
        data, stats = self.convert_input_data(x)

        n_chunks = self.split_data_by_memory(data, full_info=full_info, to_file=False)
        single_chunk = n_chunks > 1
        data_chunks = np.array_split(data, n_chunks, axis=-1)
        results = []
        for dc in tqdm(data_chunks, desc='Data chunks', display=single_chunk):
            results.append(self._detect(dc, verbose=verbose, full_info=full_info))

        result = CATSDetectionResult.concatenate(*results)
        setattr(result, "stats", stats)
        return result

    def apply(self, x: Union[np.ndarray, obspy.Stream],
              /,
              verbose: bool = False,
              full_info: Union[bool, str, List[str]] = False):
        """ Alias of `.detect`. For compatibility with CATSBase """
        return self.detect(x, verbose=verbose, full_info=full_info)

    @classmethod
    def get_all_keys(cls):
        return super(cls, cls).get_all_keys() + ["likelihood", "detection", "detected_intervals", "picked_features"]

    @classmethod
    def get_qc_keys(cls):
        return super(cls, cls).get_qc_keys() + ["likelihood", "detected_intervals", "picked_features"]

    @classmethod
    def get_default_keys(cls):
        return super(cls, cls).get_default_keys() + ['cluster_catalogs', 'detected_intervals', 'picked_features']

    def memory_usage_estimate(self, x, /, full_info=False):
        # TODO: review changes related to cluster catalogs
        memory_usage_bytes, used_together = self._memory_usage(x)
        full_info = self.parse_info_dict(full_info)

        precision_order = memory_usage_bytes['signal'] / x.size
        stft_time_len = len(self.STFT.forward_time_axis(x.shape[-1]))

        likelihood_shape = x.shape[:-1] + (stft_time_len,)
        likelihood_size = np.prod(np.float64(likelihood_shape))

        n, mod = divmod(stft_time_len, self.min_duration_len + self.min_separation_len)
        num_intervals = n + mod // self.min_duration_len  # upper bound estimate
        intervals_size = 2 * np.prod(np.float64(x.shape[:-1])) * num_intervals

        memory_usage_bytes_new = {
            "likelihood":         1. * precision_order * likelihood_size,   # float
            "detection":          1. * likelihood_size,                     # bool always
            "detected_intervals": 8. * intervals_size,                      # ~ float64, upper bound
            "picked_features":    8. * intervals_size,                      # ~ float64, upper bound
        }
        memory_usage_bytes.update(memory_usage_bytes_new)

        used_together.append(('spectrogram_cluster_ID', 'likelihood'))  # Projection step
        used_together.append(("likelihood", "detection", "picked_features", "detected_intervals"))  # cluster catalog
        base_info = ["signal", "frequencies", "time_frames"]

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
            detector.detect_to_file(x, save_path, verbose=verbose, full_info=full_info, compress=compress)
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

    @classmethod
    def get_qc_keys(cls):
        return CATSDetector.get_qc_keys()

    def plot(self,
             ind: Tuple[int] = None,
             time_interval_sec: Tuple[float] = None,
             SNR_spectrograms: bool = False,
             show_cluster_ID: bool = False,
             show_aggregated: bool = False,
             detrend_type: str = 'constant',
             aggr_func: callable = np.max,
             interactive: bool = False,
             **kwargs):

        fig, opts, inds_slices, time_interval_sec = super().plot(ind=ind,
                                                                 time_interval_sec=time_interval_sec,
                                                                 SNR_spectrograms=SNR_spectrograms,
                                                                 show_cluster_ID=show_cluster_ID,
                                                                 show_aggregated=show_aggregated,
                                                                 detrend_type=detrend_type,
                                                                 aggr_func=aggr_func,
                                                                 interactive=interactive,
                                                                 frequencies=self.frequencies)

        t_dim, a_dim = fig[0].dimensions()  # get dim params (no matter what fig index)
        L_dim = hv.Dimension('Likelihood')
        t1, t2 = time_interval_sec

        ind, i_time, i_stft = inds_slices

        aggr_ind = make_default_index_if_outrange(ind, self.spectrogram_cluster_ID.shape[:-2], default_ind_value=0)
        inds_stft = aggr_ind + (i_stft,)

        stft_time = self.tf_time(time_interval_sec)

        likelihood = np.nan_to_num(self.likelihood[inds_stft], posinf=1e8, neginf=-1e8)  # POSSIBLE `NAN` AND `INF`

        # Likelihood curve
        likelihood_fig = hv.Curve((stft_time, likelihood), kdims=[t_dim], vdims=L_dim)

        # Arrivals
        cluster_catalogs = getattr(self, 'cluster_catalogs', None)
        catalog = index_cluster_catalog(cluster_catalogs, aggr_ind).copy() if (cluster_catalogs is not None) else None

        arrivals_figs = []
        if catalog is not None:
            vals = catalog.get(['Time_first_arrival_sec', 'Time_strong_arrival_sec'], None)
            vals = catalog.get('Time_peak_sec', None) if vals is None else vals

            mpl_vlines_opts = hv.opts.VLine(color='black', linewidth=2, backend='matplotlib')
            bkh_vlines_opts = hv.opts.VLine(color='black', line_width=2, backend='bokeh')
            vlines_opts = (mpl_vlines_opts, bkh_vlines_opts)
            if vals is not None:
                A = vals.values.flatten()
                A = A[(t1 <= A) & (A <= t2)]
                arrivals_figs += [hv.VLine(ai, kdims=[t_dim, L_dim]).opts(*vlines_opts)
                                  for ai in A]

        # Picks stems
        # P = self.picked_features[aggr_ind]
        # P = P[(t1 <= P[..., 0]) & (P[..., 0] <= t2)]
        # peaks_fig = hv.Spikes(P, kdims=t_dim, vdims=L_dim)
        # peaks_fig = peaks_fig * hv.Scatter(P, kdims=t_dim, vdims=L_dim)
        #
        # mpl_scatter_opts = hv.opts.Scatter(marker='D', color='red', s=20, backend='matplotlib')
        # bkh_scatter_opts = hv.opts.Scatter(marker='diamond', color='red', size=6, backend='bokeh')
        # peaks_fig = peaks_fig.opts(mpl_scatter_opts, bkh_scatter_opts)

        # Interval boxes
        intervals = intervals_intersection(self.detected_intervals[aggr_ind], (t1, t2))

        interv_height = (np.max(likelihood) / 2) * 1.1
        rectangles = give_rectangles([intervals], [interv_height], interv_height)
        intervals_fig = hv.Rectangles(rectangles, kdims=[t_dim, L_dim, 't2', 'l2'])

        intv_tooltips = [("Start", "@Time"), ("End", "@t2")]
        mpl_rects_opts = hv.opts.Rectangles(alpha=0.2, color='blue', linewidth=0, backend='matplotlib')
        bkh_rects_opts = hv.opts.Rectangles(alpha=0.2, color='blue', line_width=0, hover_tooltips=intv_tooltips,
                                            tools=['hover'], backend='bokeh')
        intervals_fig = intervals_fig.opts(mpl_rects_opts, bkh_rects_opts)

        # Final fig with likelihood, picks and boxes
        last_figs = [likelihood_fig, intervals_fig] + arrivals_figs
        fig4 = hv.Overlay(last_figs, label=r'4. Likelihood and Detection: $\mathcal{L}(t)$ and $\tilde{\alpha}(t)$')
        lmax = 1.1 * np.max(likelihood)
        cylim = (-0.02 * lmax, lmax)
        overlay_opts = (hv.opts.Overlay(ylim=cylim, show_frame=True, backend='matplotlib'),
                        hv.opts.Overlay(ylim=cylim, height=int(opts[1].kwargs['height'] * 1.15), backend='bokeh'))
        fig4 = fig4.opts(*opts[:2]).opts(*overlay_opts)

        # Output
        fig = (fig + fig4).opts(*opts).cols(1)

        if show_cluster_ID:
            fig31 = fig[-2].opts(hv.opts.QuadMesh(norm=None, backend='matplotlib'),
                                 hv.opts.QuadMesh(logz=False, backend='bokeh'))
        # fig = (fig + fig4).opts(*opts[-2:]).cols(1)  # apply layout opts
        return fig

    def plot_traces(self,
                    ind: Tuple[int] = None,
                    time_interval_sec: Tuple[float] = None,
                    intervals: bool = True,
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

        fig = super().plot_traces(signal=self.signal, ind=ind, time_interval_sec=time_interval_sec, intervals=intervals,
                                  picks=picks, station_loc=station_loc, gain=gain, clip=clip, each_station=each_station,
                                  detrend_type=detrend_type, amplitude_scale=amplitude_scale,
                                  per_station_scale=per_station_scale, component_labels=component_labels,
                                  station_labels=station_labels, interactive=interactive, allow_picking=allow_picking,
                                  **kwargs)
        return fig

    def filter_and_update_result(self, cluster_catalogs_filter):
        super().filter_and_update_result(cluster_catalogs_filter)
        # TODO:
        #   - update intervals

    @property
    def _concat_attr_list(self):
        return super()._concat_attr_list + ["likelihood", "detection", "detected_intervals", "picked_features"]

    @property
    def _update_funcs_before_concat(self):
        return super()._update_funcs_before_concat + [self._update_intervals_and_features]

    def _update_intervals_and_features(self, other):
        attrs = ["detected_intervals", "picked_features"]
        t0 = self.get_t0_for_append()

        for attr in attrs:
            self_attr = getattr(self, attr, None)
            other_attr = getattr(other, attr, None)
            if (self_attr is not None) and (other_attr is not None):
                if attr == 'detected_intervals':
                    other_attr += t0
                else:
                    shape = other_attr.shape
                    for ind in np.ndindex(shape):
                        other_attr[ind][..., 0] += t0  # onset time increment

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

    def extend_detected_intervals(self, pre_time_sec=None, post_time_sec=None):
        extended_intervals = np.empty_like(self.detected_intervals)
        auto_pre = (pre_time_sec is None)
        auto_post = (post_time_sec is None)
        auto_extension = auto_pre or auto_post
        for ind, intvs in np.ndenumerate(self.detected_intervals):
            ext_intv = intvs.copy()
            if auto_extension:
                d_intv = np.diff(ext_intv, axis=-1).squeeze()
            else:
                d_intv = None
            pre_time_sec = d_intv if auto_pre else pre_time_sec
            post_time_sec = d_intv if auto_post else post_time_sec
            ext_intv[:, 0] -= pre_time_sec
            ext_intv[:, 1] += post_time_sec
            extended_intervals[ind] = np.clip(ext_intv, 0.0, (self.time_npts - 1) * self.dt_sec)
        return extended_intervals

    def slice_input_traces(self, reference_traces=None, pre_time_sec=None, post_time_sec=None):
        signal = self.signal if reference_traces is None else reference_traces
        extended_intervals = self.extend_detected_intervals(pre_time_sec, post_time_sec)
        shape = signal.shape[:-1]
        start_time_sec = np.empty(shape, dtype=object)
        sliced_traces = np.empty(shape, dtype=object)
        convert_ind = lambda intv: give_index_slice_by_limits(intv, self.dt_sec)
        for ind in np.ndindex(shape):
            intv_ind = make_default_index_if_outrange(ind, extended_intervals.shape, default_ind_value=0)
            sliced_traces[ind] = np.array([signal[ind][convert_ind(intv)]
                                           for intv in extended_intervals[intv_ind]], dtype=object)
            start_time_sec[ind] = extended_intervals[intv_ind][:, 0]
        return sliced_traces, start_time_sec

    def write_sliced_traces(self, sliced_traces, start_time_sec, folder=None, prefix_name=None,
                            reference_header=None, format="SAC"):
        folder = folder or "SlicedTraces"
        prefix_name = prefix_name or "Event"
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        total = np.prod(sliced_traces.shape)
        for ind, sliced_trace in tqdm(np.ndenumerate(sliced_traces), total=total):
            for tr, t0 in zip(sliced_trace, start_time_sec[ind]):
                if len(tr) > 0:

                    header = obspy.core.trace.Stats({"delta": self.dt_sec})  # default header
                    if self.stats is not None:
                        header.update(self.stats[ind])  # update header with available obspy.Stats for trace[ind]
                    if reference_header is not None:
                        header.update(reference_header)  # update header with given reference header

                    header.update({"starttime": header.starttime + datetime.timedelta(seconds=t0)})  # time shift
                    trace = obspy.core.trace.Trace(tr, header=header)

                    t0_str = str(header.starttime).replace('-', '')
                    t0_str = t0_str.replace(':', '_').replace('.', '_')
                    # t0_str = f"{t0:.3f}"
                    filename = [prefix_name, header.network, header.station, header.channel, *map(str, ind), t0_str]
                    filename = '_'.join([fn for fn in filename if len(fn) > 0])

                    trace.write((folder / f"{filename}.{format.casefold()}").as_posix(), format)

    def save_detected_events(self, pre_time_sec=None, post_time_sec=None, folder='DetectedEvents',
                             prefix_name="", reference_header=None, format="SAC"):
        """
            Saves via obspy.Trace.write()
        """
        sliced_traces, start_time_sec = self.slice_input_traces(pre_time_sec=pre_time_sec, post_time_sec=post_time_sec)
        self.write_sliced_traces(sliced_traces, start_time_sec, folder=folder, prefix_name=prefix_name,
                                 reference_header=reference_header, format=format)
