import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import holoviews as hv
from typing import Any, Union, Literal, List, Tuple
from pydantic import BaseModel, Field, Extra
from tqdm.notebook import tqdm

from cats.core.plottingutils import plot_traces
from cats.core.timefrequency import CWTOperator
from cats.core.clustering import Clustering, ClusterCatalogs, concatenate_arrays_of_cluster_catalogs
from cats.core.date import BEDATE_trimming, group_frequency, bandpass_frequency_groups
from cats.core.env_variables import get_min_bedate_block_size, get_max_memory_available_for_cats
from cats.core.utils import get_interval_division, format_index_by_dimensions, cast_to_bool_dict, StatusKeeper
from cats.core.utils import format_interval_by_limits, give_index_slice_by_limits, save_pickle, load_pickle
from cats.core.utils import give_nonzero_limits, mat_structure_to_tight_dataframe_dict, del_vals_by_keys
from cats.core.utils import intervals_intersection, give_rectangles


class CATS_CWT(BaseModel, extra=Extra.allow):
    """
        CATS operator. Implements 4 main steps:
            1) TF transform
            2) Noise estimation via B-E-DATE
            3) Trimming spectrogram based minSNR
            4) Clustering trimmed spectrogram
            5) Inverse TF transform (if `denoise` mode is chosen)
        """
    # main TF params
    dt_sec: float
    wavelet_type: Union[str, tuple[str, dict]] = ('morlet', {'mu': 5})
    scales_type: Union[Literal['log', 'log-piecewise', 'linear', 'log:maximal'],
                       tuple[float],
                       list[float]] = 'log'
    nvoices: int = 16  # >= 16
    cwt_kwargs: dict = None

    # main B-E-DATE params
    minSNR: float = 5.5
    stationary_frame_sec: float = None
    # main Clustering params
    cluster_size_t_sec: float = 0.2
    cluster_size_scales: int = Field(5, ge=1)
    cluster_distance_t_sec: float = None
    cluster_distance_scales: int = Field(1, ge=1)

    # Extra CWT params
    bandpass_scales: Union[tuple[int, int], Any] = None

    # Extra B-E-DATE params
    bedate_scales_grouping: int = 1
    date_Q: float = 0.95
    date_detection_mode: bool = True

    # Extra clustering params
    background_weight: float = 0.0
    cluster_minSNR: float = 0.0
    cluster_fullness: float = Field(0, ge=0.0, le=1.0)
    cluster_catalogs: bool = True
    clustering_multitrace: bool = False
    cluster_size_trace: int = Field(1, ge=1)
    cluster_distance_trace: int = Field(1, ge=1)

    # Misc
    name: str = "CATS_CWT"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_params()

    def _set_params(self):
        self.cwt_kwargs = self.cwt_kwargs or {}

        # Setting CWT operator
        self.CWT = CWTOperator(dt_sec=self.dt_sec, wavelet=self.wavelet_type, scales=self.scales_type, nv=self.nvoices,
                               **self.cwt_kwargs)

        # DATE params
        self.stationary_frame_sec = v if (v := self.stationary_frame_sec) is not None else 0.0
        self.stationary_frame_len = max(round(self.stationary_frame_sec / self.dt_sec), get_min_bedate_block_size())
        self.stationary_frame_sec = self.stationary_frame_len * self.dt_sec

        # Clustering params
        self.cluster_size_t_len = max(round(self.cluster_size_t_sec / self.dt_sec), 1)
        self.cluster_size_trace_len = self.cluster_size_trace

        self.cluster_distance_t_sec = v if (v := self.cluster_distance_t_sec) is not None else self.dt_sec
        self.cluster_distance_t_len = max(round(self.cluster_distance_t_sec / self.dt_sec), 1)
        self.cluster_distance_trace_len = self.cluster_distance_trace

        self.time_edge = 2

        self.min_duration_len = max(self.cluster_size_t_len, 1)  # `-1` to include bounds (0, 1, 0)
        self.min_separation_len = max(self.cluster_distance_t_len + 1, 2)  # `+1` to exclude bounds (1, 0, 1)
        self.min_duration_sec = self.min_duration_len * self.dt_sec
        self.min_separation_sec = self.min_separation_len * self.dt_sec

        self.bandpass_scales = self.bandpass_scales or (None, None)
        self.bandpass_scales_slice = slice(*self.bandpass_scales)

    def export_main_params(self):
        return {kw: val for kw in type(self).__fields__.keys() if (val := getattr(self, kw, None)) is not None}

    @classmethod
    def from_result(cls, CATS_CWTResult):
        return cls(**CATS_CWTResult.main_params)

    def reset_params(self, **params):
        """
            Updates the instance with changed parameters.
        """
        kwargs = self.export_main_params()
        kwargs.update(params)
        self.__init__(**kwargs)

    def apply_CWT(self, result_container, /):
        result_container['coefficients'] = self.CWT * result_container['signal']
        result_container['spectrogram'] = np.abs(result_container['coefficients'])
        result_container['scales'] = self.CWT.get_scales(result_container['signal'].shape[-1])

    def apply_BEDATE(self, result_container, /):

        scales_idx = np.arange(len(result_container['scales']))
        bandpass_scales_slice = slice(self.bandpass_scales_slice.start or 0,
                                      self.bandpass_scales_slice.stop or (scales_idx[-1] + 1))
        scales_groups_index, scales_groups = group_frequency(scales_idx, self.bedate_scales_grouping, False)
        bandpassed_scales_groups_slice = bandpass_frequency_groups(scales_groups_index, bandpass_scales_slice)

        result_container["scales_groups"] = scales_groups
        result_container['time_frames'] = get_interval_division(N=result_container['spectrogram'].shape[-1],
                                                                L=self.stationary_frame_len)

        T, STD, THR = BEDATE_trimming(result_container['spectrogram'],
                                      scales_groups_index, bandpassed_scales_groups_slice,
                                      bandpass_scales_slice,
                                      result_container['time_frames'], self.minSNR, self.time_edge, self.date_Q,
                                      not self.date_detection_mode, fft_bounds=False)

        result_container['spectrogram_SNR_trimmed'] = T
        result_container['noise_std'] = STD
        result_container['noise_threshold_conversion'] = THR

    def apply_Clustering(self, result_container, /):
        mc = self.clustering_multitrace
        q = (self.cluster_distance_trace_len,) * mc + (self.cluster_distance_scales, self.cluster_distance_t_len)
        s = (self.cluster_size_trace_len,) * mc + (self.cluster_size_scales, self.cluster_size_t_len)
        alpha = self.cluster_fullness
        log_freq_cluster = (0.0, 0.0)

        result_container['spectrogram_SNR_clustered'] = np.zeros_like(result_container['spectrogram_SNR_trimmed'])
        result_container['spectrogram_cluster_ID'] = np.zeros(result_container['spectrogram_SNR_trimmed'].shape,
                                                              dtype=np.uint32)

        result_container['spectrogram_SNR_clustered'], result_container['spectrogram_cluster_ID'] = \
            Clustering(result_container['spectrogram_SNR_trimmed'], q=q, s=s,
                       minSNR=self.cluster_minSNR, alpha=alpha, log_freq_cluster=log_freq_cluster)

        if self.cluster_catalogs:
            result_container['cluster_catalogs'] = ClusterCatalogs(result_container['spectrogram_SNR_clustered'],
                                                                   result_container['spectrogram_cluster_ID'],
                                                                   result_container['scales'], self.dt_sec)
        else:
            result_container['cluster_catalogs'] = None

    def apply_ICWT(self, result_container, /):
        weights = result_container['spectrogram_SNR_clustered'] > 0
        if self.background_weight:
            weights = np.where(weights, 1.0, self.background_weight)

        weighted_coefficients = result_container['coefficients'] * weights

        result_container['signal_denoised'] = (self.CWT / weighted_coefficients)
        del weights, weighted_coefficients

    def apply_func(self, func_name, result_container, status_keeper, process_name=None, **kwargs):
        process_name = process_name or func_name
        with status_keeper(current_process=process_name):
            getattr(self, func_name)(result_container, **kwargs)

    def _denoise(self, x, /, verbose=False, full_info=False):
        full_info = self.parse_info_dict(full_info)

        result = dict.fromkeys(full_info.keys())
        result['signal'] = x
        history = StatusKeeper(verbose=verbose)

        # STFT
        self.apply_func(func_name='apply_CWT', result_container=result, status_keeper=history,
                        process_name='CWT')
        del_vals_by_keys(result, full_info, ['signal'])

        # B-E-DATE
        self.apply_func(func_name='apply_BEDATE', result_container=result, status_keeper=history,
                        process_name='B-E-DATE trimming')
        del_vals_by_keys(result, full_info, ['spectrogram', 'noise_std', 'noise_threshold_conversion'])

        # Clustering
        self.apply_func(func_name='apply_Clustering', result_container=result, status_keeper=history,
                        process_name='Clustering')
        del_vals_by_keys(result, full_info, ['spectrogram_SNR_trimmed', 'spectrogram_cluster_ID'])

        # Inverse STFT
        self.apply_func(func_name='apply_ICWT', result_container=result, status_keeper=history,
                        process_name='Inverse CWT')

        del_vals_by_keys(result, full_info, ['coefficients', 'spectrogram_SNR_clustered'])

        history.print_total_time()

        from_full_info = {kw: result.get(kw, None) for kw in full_info}

        return CATS_CWTResult(dt_sec=self.dt_sec,
                              npts=x.shape[-1],
                              time_frames=result['time_frames'] * self.dt_sec,
                              minSNR=self.minSNR,
                              history=history,
                              cluster_catalogs=result['cluster_catalogs'],
                              main_params=self.export_main_params(),
                              **from_full_info)

    def denoise(self, x: np.ndarray,
                /,
                verbose: bool = False,
                full_info: Union[bool, str, List[str]] = False):

        n_chunks = self.split_data_by_memory(x, full_info=full_info, to_file=False)
        single_chunk = n_chunks > 1
        data_chunks = np.array_split(x, n_chunks, axis=-1)
        results = []
        for dc in tqdm(data_chunks, desc='Data chunks', display=single_chunk):
            results.append(self._denoise(dc, verbose=verbose, full_info=full_info))
        return CATS_CWTResult.concatenate(*results)

    def __mul__(self, x):
        return self.denoise(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.denoise(x, verbose=True, full_info='qc')

    def __matmul__(self, x):
        return self.denoise(x, verbose=True, full_info=True)

    @staticmethod
    def _info_keys():
        return ["signal",
                "coefficients",
                "spectrogram",
                "noise_std",
                "noise_threshold_conversion",
                "spectrogram_SNR_trimmed",
                "spectrogram_SNR_clustered",
                "spectrogram_cluster_ID",
                "scales",
                "scales_groups",
                "signal_denoised"]

    @staticmethod
    def get_qc_keys():
        return ["signal", "spectrogram", "spectrogram_SNR_trimmed",
                "spectrogram_SNR_clustered", "signal_denoised", "scales"]

    def parse_info_dict(self, full_info):
        info_keys = self._info_keys()
        if isinstance(full_info, str) and \
           (full_info in ['plot', 'plotting', 'plt', 'qc', 'main']):  # only those needed for plotting step-by-step
            full_info = self.get_qc_keys()

        full_info = cast_to_bool_dict(full_info, info_keys)

        full_info["signal_denoised"] = True     # default saved result

        return full_info

    def _memory_usage(self, x):
        time_len = x.shape[-1]
        scale_len = len(self.CWT.get_scales(time_len))
        ft_shape = x.shape[:-1] + (scale_len, time_len)
        ft_size = np.prod(ft_shape)

        bedate_shape = x.shape[:-1] + (scale_len // self.bedate_scales_grouping,
                                       len(get_interval_division(N=time_len, L=self.stationary_frame_len)))
        bedate_size = np.prod(bedate_shape)

        x_bytes = x.nbytes
        precision_order = x_bytes / x.size

        memory_usage_bytes = {
            "stft_frequency": 8. * ft_shape[-2],                 # float64 always
            "time_frames":                 8. * bedate_shape[-1],               # float64 always
            "signal":                      1. * x_bytes,                        # float / int
            "coefficients": 2. * precision_order * ft_size,    # complex
            "spectrogram": 1. * precision_order * ft_size,    # float
            "noise_threshold_conversion":  8. * bedate_shape[-2],               # float
            "noise_std":                   1. * precision_order * bedate_size,  # float
            "spectrogram_SNR_trimmed": 1. * precision_order * ft_size,    # float
            "spectrogram_SNR_clustered": 1. * precision_order * ft_size,    # float
            "spectrogram_cluster_ID": 4. * ft_size,                      # uint32 always
        }
        used_together = [('coefficients', 'spectrogram'),
                         ('spectrogram', 'noise_threshold_conversion', 'noise_std', 'spectrogram_SNR_trimmed'),
                         ('spectrogram_SNR_trimmed', 'spectrogram_SNR_clustered', 'spectrogram_cluster_ID')]

        return memory_usage_bytes, used_together

    @staticmethod
    def memory_info(memory_usage_bytes, used_together, base_info, full_info):
        used_together_bytes = [0] * len(used_together)
        for i, ut in enumerate(used_together):
            used_together_bytes[i] += sum([memory_usage_bytes[kw] for kw in ut])

        base_bytes = sum([memory_usage_bytes[kw] for kw in base_info])

        min_required = base_bytes + max(used_together_bytes)  # minimum needed to process
        max_required = sum(memory_usage_bytes.values())  # maximum needed to store everything
        info_required = base_bytes + sum([memory_usage_bytes.get(kw, 0)
                                          for kw, v in full_info.items() if v])  # to store only the final result

        memory_info = {'available_for_cats': get_max_memory_available_for_cats(),
                       'min_required': min_required,
                       'max_required': max_required,
                       'required_by_info': info_required,
                       'detailed_memory_usage': memory_usage_bytes}
        return memory_info

    @staticmethod
    def memory_chunks(memory_info, to_file):
        large_result = memory_info["required_by_info"] > memory_info["available_for_cats"]
        if large_result and not to_file:
            raise MemoryError("The final result cannot fit the memory, "
                              f"min required {memory_info['required_by_info']} bytes, "
                              f"available memory {memory_info['available_for_cats']} bytes. "
                              "Consider function `detect_to_file` instead, or use less info in `full_info`.")
        elif to_file:
            n_chunks = int(np.ceil(memory_info["required_by_info"] / memory_info["available_for_cats"]))
        else:
            n_chunks = int(np.ceil(memory_info["min_required"] / memory_info["available_for_cats"]))

        return n_chunks

    def save(self, filename):
        save_pickle(self.export_main_params(), filename)

    @classmethod
    def load(cls, filename):
        loaded = load_pickle(filename)
        if isinstance(loaded, cls):
            loaded = loaded.export_main_params()
        return cls(**loaded)

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
        return self.memory_chunks(self.memory_usage_estimate(x, full_info=full_info), to_file)


class CATS_CWTResult(BaseModel):
    signal: Any = None
    coefficients: Any = None
    spectrogram: Any = None
    noise_threshold_conversion: Any = None
    noise_std: Any = None
    spectrogram_SNR_trimmed: Any = None
    spectrogram_SNR_clustered: Any = None
    spectrogram_cluster_ID: Any = None
    signal_denoised: Any = None
    scales: Any = None
    scales_groups: Any = None
    cluster_catalogs: Any = None
    dt_sec: float = None
    npts: int = None
    time_frames: Any = None
    minSNR: float = None
    history: Any = None
    main_params: dict = None
    header_info: dict = None

    @staticmethod
    def base_time_func(npts, dt_sec, t0, time_interval_sec):
        if time_interval_sec is None:
            return t0 + np.arange(npts) * dt_sec
        else:
            time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (npts - 1) * dt_sec))
            time_slice = give_index_slice_by_limits(time_interval_sec, dt_sec)
            return t0 + np.arange(time_slice.start, time_slice.stop) * dt_sec

    def time(self, time_interval_sec=None):
        return CATS_CWTResult.base_time_func(self.npts, self.dt_sec, 0, time_interval_sec)

    def _plot(self, ind=None, time_interval_sec=None, SNR_spectrograms=True):
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Scale number')
        a_dim = hv.Dimension('Amplitude')

        ind = format_index_by_dimensions(ind=ind, shape=self.signal.shape[:-1], slice_dims=0, default_ind=0)

        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.npts - 1) * self.dt_sec))

        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        inds_time = ind + (i_time,)
        inds_stft = ind + (slice(None), i_time)

        PSD = self.spectrogram[inds_stft]
        SNR = self.spectrogram_SNR_trimmed[inds_stft]
        C = self.spectrogram_SNR_clustered[inds_stft]

        PSD_clims = give_nonzero_limits(PSD, initials=(1e-1, 1e1))

        if not SNR_spectrograms:
            SNR = PSD * (SNR > 0)
            C = PSD * (C > 0)
            label_trimmed = 'spectrogram: $|X(t,f)| \cdot (T(t,f) > 0)$'
            label_clustered = 'spectrogram: $|X(t,f)| \cdot (\mathcal{L}(t,f) > 0)$'
            SNR_clims = PSD_clims
        else:
            label_trimmed = 'SNR spectrogram: $T(t,f)$'
            label_clustered = 'SNR spectrogram: $\mathcal{L}(t,f)$'
            SNR_clims = give_nonzero_limits(SNR, initials=(1e-1, 1e1))

        time = self.time(time_interval_sec)
        nscales = np.arange(len(self.scales))

        fig0 = hv.Curve((time, self.signal[inds_time]), kdims=[t_dim], vdims=a_dim,
                        label='0. Input data: $x(t)$').opts(xlabel='', linewidth=0.5)
        fig1 = hv.Image((time, nscales, PSD), kdims=[t_dim, f_dim],
                        label='1. Amplitude spectrogram: $|X(t,f)|$').opts(clim=PSD_clims, clabel='Amplitude')
        fig2 = hv.Image((time, nscales, SNR), kdims=[t_dim, f_dim],
                        label=f'2. Trimmed {label_trimmed}').opts(clim=SNR_clims)
        fig3 = hv.Image((time, nscales, C), kdims=[t_dim, f_dim],
                        label=f'3. Clustered {label_clustered}').opts(clim=SNR_clims)

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250
        cmap = 'viridis'
        xlim = time_interval_sec
        ylim = (None, None)
        spectr_opts = hv.opts.Image(cmap=cmap, colorbar=True, logy=False, invert_yaxis=True,
                                    logz=True, xlim=xlim, ylim=ylim,
                                    xlabel='', clabel='', aspect=5, fig_size=figsize, fontsize=fontsize)
        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=xlim)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        inds_slices = (ind, i_time, i_time)
        figs = (fig0 + fig1 + fig2 + fig3)
        return figs, (layout_opts, spectr_opts, curve_opts), inds_slices, time_interval_sec

    def plot(self,
             ind: Tuple[int] = None,
             time_interval_sec: Tuple[float] = None,
             SNR_spectrograms: bool = False,
             weighted_coefficients: bool = False,
             intervals: bool = False,
             picks: bool = False,
             ):

        fig, opts, inds_slices, time_interval_sec = self._plot(ind, time_interval_sec, SNR_spectrograms)
        ref_kw_opts = opts[-1].kwargs
        t1, t2 = ref_kw_opts['xlim']

        t_dim, a_dim = fig[0].dimensions()

        ind, i_time, i_stft = inds_slices
        trace_slice = ind + (i_time,)

        # Weighted coefficients
        if weighted_coefficients:
            psd_fig = fig[1]
            f_dim = psd_fig.dimensions()[1]

            C = fig[-1].data['z']
            weights = np.where(C > 0, 1.0, self.main_params['background_weight'])
            WPSD = fig[1].data['z'] * weights
            psd_opts = psd_fig.opts.get().options
            fig31 = hv.Image((psd_fig.data[t_dim.name], psd_fig.data[f_dim.name], WPSD),
                             kdims=[t_dim, f_dim],
                             label='3.1. Weighted amplitude spectrogram: $|X(t,f) \cdot W(t,f)|$').opts(**psd_opts)
            fig = fig + fig31

        # Denoised signal
        fig4 = hv.Curve((fig[0].data[t_dim.name], self.signal_denoised[trace_slice]),
                        kdims=[t_dim], vdims=a_dim).opts(linewidth=1)

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
                        "spectrogram_cluster_ID"]

        # update cluster id
        attr = "spectrogram_cluster_ID"
        if ((self_attr := getattr(self, attr, None)) is not None) and \
           ((other_attr := getattr(other, attr, None)) is not None):
            shape = other_attr.shape[:-2]
            for ind in np.ndindex(shape):
                other_attr[ind][other_attr[ind] > 0] += self_attr[ind].max()
            setattr(other, attr, other_attr)

        # concatenation
        for name in concat_attrs:
            self._concat(other, name, -1)

        t0 = self.time_frames[-1, -1] + self.dt_sec

        self._concat(other, "time_frames", 0, t0)
        self.cluster_catalogs = concatenate_arrays_of_cluster_catalogs(self.cluster_catalogs,
                                                                       other.cluster_catalogs, t0)

        self.npts += other.npts

        self.history.merge(other.history)

        assert self.minSNR == other.minSNR
        assert self.dt_sec == other.dt_sec
        assert np.all(self.scales == other.scales)
        assert np.all(self.noise_threshold_conversion == other.noise_threshold_conversion)

    def _concat(self, other, attr, axis, increment=None, increment_ind=None):
        if ((self_attr := getattr(self, attr, None)) is not None) and \
                ((other_attr := getattr(other, attr, None)) is not None):
            if increment is not None:
                if increment_ind is not None:
                    if other_attr.dtype.name == 'object':
                        for ind, _ in np.ndenumerate(other_attr):
                            other_attr[ind][increment_ind] += increment
                    else:
                        other_attr[increment_ind] += increment
                else:
                    other_attr += increment

            delattr(self, attr)
            delattr(other, attr)
            if (other_attr.dtype.name == 'object') and (self_attr.dtype.name == 'object'):
                concatenated = np.empty_like(other_attr, dtype=object)
                for ind, i_other in np.ndenumerate(other_attr):
                    concatenated[ind] = np.concatenate((self_attr[ind], i_other), axis=axis)
            else:
                concatenated = np.concatenate((self_attr, other_attr), axis=axis)
            del self_attr, other_attr
            setattr(self, attr, concatenated)
        else:
            pass

    @staticmethod
    def concatenate(*objects):
        obj0 = objects[0]
        for obj in objects[1:]:
            obj0.append(obj)
            del obj
        return obj0

    def filter_convert_attributes_to_dict(self):
        # Remove `None`
        mdict = {name: attr for name, attr in self.dict().items() if (attr is not None)}

        # Convert DataFrame to dict for `mat` format
        if (catalog := mdict.get('cluster_catalogs', None)) is not None:
            for ind, cat_ind in np.ndenumerate(catalog):
                catalog[ind] = cat_ind.to_dict(orient='tight')
            mdict['cluster_catalogs'] = catalog
        return mdict

    @staticmethod
    def convert_dict_to_attributes(mdict):
        # Convert dict to DataFrame
        if (catalog := mdict.get('cluster_catalogs', None)) is not None:
            for ind, cat_ind in np.ndenumerate(catalog):
                cat_dict = mat_structure_to_tight_dataframe_dict(cat_ind)
                catalog[ind] = pd.DataFrame.from_dict(cat_dict, orient='tight')

        return mdict

    def save(self, filepath, compress=False, header_info=None):
        self.header_info = header_info
        mdict = self.filter_convert_attributes_to_dict()
        savemat(filepath, mdict, do_compression=compress)
        del mdict

    @classmethod
    def load(cls, filepath):
        mdict = loadmat(filepath, simplify_cells=True)
        mdict = cls.convert_dict_to_attributes(mdict)
        return cls(**mdict)