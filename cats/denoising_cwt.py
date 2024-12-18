import numpy as np
from typing import Any, Union, Literal, List, Tuple, Callable
from pydantic import BaseModel, Field
from tqdm.notebook import tqdm
import obspy
import holoviews as hv

from cats.baseclass import CATSResult
from cats.denoising import CATSDenoisingResult
from cats.core.timefrequency import CWTOperator
from cats.core.clustering import Clustering, bbox_peaks, ClusterCatalogs
from cats.core.date import BEDATE_trimming, divide_into_groups
from cats.core.env_variables import get_min_bedate_block_size, get_max_memory_available_for_cats
from cats.core.utils import (get_interval_division, cast_to_bool_dict, del_vals_by_keys, StatusKeeper,
                             aggregate_array_by_axis_and_func)
from cats.core.utils import format_interval_by_limits, give_index_slice_by_limits
from cats.io.utils import load_pickle, save_pickle
from cats.io import convert_stream_to_dict


class CATSDenoiserCWT(BaseModel, extra='allow'):
    """
        CATS denoising operator which implements:
            1) CWT transform
            2) Noise estimation via B-E-DATE and spectrogram trimming
            3) Clustering trimmed spectrogram
            4) Inverse CWT transform
        """
    # main TF params
    dt_sec: float
    wavelet_type: Union[str, tuple[str, dict]] = ('morlet', {'mu': 5})
    scales_type: Union[Literal['log', 'log-piecewise', 'linear', 'log:maximal'],
                       tuple[float],
                       list[float]] = 'log'
    nvoices: int = 16
    cwt_kwargs: dict = None

    # main B-E-DATE params
    minSNR: float = 5.5
    stationary_frame_sec: float = None
    # main Clustering params
    cluster_size_t_sec: float = 0.2
    cluster_size_scale_octaves: float = 2.0
    cluster_distance_t_sec: float = None
    cluster_distance_scale_octaves: float = None

    # Extra CWT params
    bandpass_scale_octaves: Union[tuple[float, float], Any] = None

    # Extra B-E-DATE params
    bedate_scale_grouping_octaves: float = None
    date_Q: float = Field(0.95, gt=0.0, lt=1.0)
    date_Nmin_percentile: float = Field(0.25, gt=0.0, lt=0.5)
    date_original_mode: bool = False

    # Extra clustering params
    background_weight: float = 0.0
    cluster_catalogs_funcs: Union[List[Callable], None] = None
    cluster_feature_distributions: Union[List[str], None] = None
    cluster_catalogs_filter: Union[Callable, None] = None
    cluster_catalogs_opts: Union[dict, None] = None
    clustering_multitrace: bool = False
    cluster_size_trace: int = Field(1, ge=1)
    cluster_distance_trace: int = Field(1, ge=1)

    aggr_clustering_axis: Union[int, Tuple[int], None] = None
    aggr_clustering_func: Union[str, None] = None

    # Misc
    name: str = "CATSDenoiserCWT"

    def __init__(self, **kwargs):
        """
            Arguments:
                dt_sec: int : data sampling rate in seconds
                wavelet_type: str | tuple[str, dict] : mother wavelet
                scales_type: str : type of scales distribution. 'Log' uses pow of `2`
                nvoices: int : number of wavelet per octave, step `2**(1 / nv)`
                cwt_kwargs: dict : additional kwargs for `ssqueezepy.cwt`
                minSNR: float : minimum expected SNR of signal
                stationary_frame_sec: float : length of stationary window in seconds where noise is stationary
                cluster_size_t_sec: float : minimum size of clusters in time, seconds
                cluster_size_scale_octaves: float : minimum size of clusters in scales, in octaves
                cluster_distance_t_sec: float : minimum distance between clusters in time, seconds
                cluster_distance_scale_octaves: float : minimum distance between clusters in scales, in octaves
                bandpass_scales: tuple[float, float] : bandpass of scales (log2(scale_1), log2(scale_2))
                bedate_scales_grouping: float : grouping multiple scales step, log2(scale)
                date_Q: float : see `cats.CATSDenoiser`
                date_Nmin_percentile: float : see `cats.CATSDenoiser`
                date_original_mode: bool : see `cats.CATSDenoiser`
                background_weight: float : weight for the removed background noise to include back.
                cluster_catalogs: bool :  see `cats.CATSDenoiser`
                clustering_multitrace: bool : see `cats.CATSDenoiser`
                cluster_size_trace: int :  see `cats.CATSDenoiser`
                cluster_distance_trace: int : see `cats.CATSDenoiser`
                name: int : name of the denoiser
        """
        super().__init__(**kwargs)
        self._set_params()
        self._init_params = kwargs

    def _set_params(self):
        self.cwt_kwargs = self.cwt_kwargs or {}

        # Setting CWT operator
        self.CWT = CWTOperator(dt_sec=self.dt_sec, wavelet=self.wavelet_type, scales=self.scales_type,
                               nv=self.nvoices, **self.cwt_kwargs)

        # DATE params
        self.stationary_frame_sec = v if (v := self.stationary_frame_sec) is not None else 0.0
        self.stationary_frame_len = max(round(self.stationary_frame_sec / self.dt_sec), get_min_bedate_block_size())
        self.stationary_frame_sec = self.stationary_frame_len * self.dt_sec

        # Bandpass by zeroing
        self.bandpass_scale_octaves = format_interval_by_limits(self.bandpass_scale_octaves, (0, None))
        if (self.bandpass_scale_octaves[0] is not None) and (self.bandpass_scale_octaves[1] is not None):
            bandwidth = self.bandpass_scale_octaves[1] - self.bandpass_scale_octaves[0]
            assert bandwidth > (css := self.cluster_size_scale_octaves), \
                f"Scale bandwidth `{bandwidth}` must be bigger than min scale cluster size `{css}`"
        self.bandpass_scale_slice = give_index_slice_by_limits(self.bandpass_scale_octaves, 1 / self.nvoices)

        # Clustering params
        self.cluster_size_t_len = max(round(self.cluster_size_t_sec / self.dt_sec), 1)
        self.cluster_size_scale_len = max(round(self.cluster_size_scale_octaves * self.nvoices), 1)
        self.cluster_size_trace_len = self.cluster_size_trace

        self.cluster_distance_t_sec = self.cluster_distance_t_sec or self.dt_sec
        self.cluster_distance_t_len = max(round(self.cluster_distance_t_sec / self.dt_sec), 1)
        self.cluster_distance_scale_octaves = self.cluster_distance_scale_octaves or 1 / self.nvoices
        self.cluster_distance_scale_len = max(round(self.cluster_distance_scale_octaves * self.nvoices), 1)
        self.cluster_distance_trace_len = self.cluster_distance_trace

        self.time_edge = 2

        self.min_duration_len = max(self.cluster_size_t_len, 1)  # `-1` to include bounds (0, 1, 0)
        self.min_separation_len = max(self.cluster_distance_t_len + 1, 2)  # `+1` to exclude bounds (1, 0, 1)
        self.min_duration_sec = self.min_duration_len * self.dt_sec
        self.min_separation_sec = self.min_separation_len * self.dt_sec

        # Catalog params
        self.cluster_catalogs_funcs = self.cluster_catalogs_funcs or [bbox_peaks]
        self.cluster_feature_distributions = (self.cluster_feature_distributions or
                                              ['spectrogram', 'spectrogram_SNR'])
        self.cluster_catalogs_opts = self.cluster_catalogs_opts or {}

        # Grouping scales for BEDATE
        self.bedate_scale_grouping_octaves = self.bedate_scale_grouping_octaves or 1 / self.nvoices
        self.bedate_scale_grouping_len = max(round(self.bedate_scale_grouping_octaves * self.nvoices), 1)

    @property
    def all_params(self):
        """ All params from __init__ for the instance
        """
        return {kw: getattr(self, kw, None) for kw in self.model_fields.keys()}

    @property
    def main_params(self):
        """ Params from __init__ that are not None for the instance
        """
        return {kw: val for kw, val in self.all_params.items() if val is not None}

    @property
    def init_params(self):
        """ Params passed to __init__ for the instance. Reset updates it.
        """
        return self._init_params

    def reset_params(self, **params):
        """ Updates params of the instance.
        """
        kwargs = self.init_params
        kwargs.update(params)

        self.__init__(**kwargs)

    def save(self, filename):
        save_pickle(self, filename)

    @classmethod
    def load(cls, filename):
        loaded = load_pickle(filename)
        if isinstance(loaded, dict):
            return cls(**loaded)
        else:
            return loaded

    @classmethod
    def from_result(cls, CATSDenoiserCWTResult):
        return cls(**CATSDenoiserCWTResult.main_params)

    def apply_CWT(self, result_container, /):
        result_container['coefficients'] = self.CWT * result_container['signal']
        result_container['spectrogram'] = np.abs(result_container['coefficients'])
        N = result_container['signal'].shape[-1]
        result_container['scales'] = self.CWT.get_scales(N)
        result_container['frequencies'] = self.CWT.get_frequencies(N)

    def apply_BEDATE(self, result_container, /):

        # grouping scales
        scales_groups_indexes = divide_into_groups(num_freqs=len(result_container['scales']),
                                                   group_step=self.bedate_scale_grouping_len,
                                                   is_octaves_step=False, fft_base=False)
        result_container["scales_groups_indexes"] = scales_groups_indexes

        # BEDATE
        result_container['time_frames'] = get_interval_division(N=result_container['spectrogram'].shape[-1],
                                                                L=self.stationary_frame_len)
        dim = 1 + np.iscomplexobj(result_container["coefficients"])  # checks if coefs are complex for dims

        SNR, T, STD, THR = BEDATE_trimming(result_container['spectrogram'],
                                      scales_groups_indexes, self.bandpass_scale_slice,
                                      result_container['time_frames'], self.minSNR, self.time_edge,
                                      Q=self.date_Q, Nmin_percentile=self.date_Nmin_percentile,
                                      original_mode=self.date_original_mode, fft_base=False, dim=dim)

        result_container['spectrogram_SNR'] = SNR
        result_container['spectrogram_trim_mask'] = T
        result_container['noise_std'] = STD
        result_container['noise_threshold_conversion'] = THR

        # Aggregate trimmed spectrogram over `aggr_clustering_axis` by `aggr_clustering_func`
        # Maybe useful if multi-component data are used (3-C or receiver groups)
        result_container['spectrogram_trim_mask_aggr'] = aggregate_array_by_axis_and_func(T,
                                                                                          self.aggr_clustering_axis,
                                                                                          func='any', min_last_dims=2)

    def apply_Clustering(self, result_container, /):
        mc = self.clustering_multitrace
        q = (self.cluster_distance_trace_len,) * mc + (self.cluster_distance_scale_len, self.cluster_distance_t_len)
        s = (self.cluster_size_trace_len,) * mc + (self.cluster_size_scale_len, self.cluster_size_t_len)
        log_freq_cluster = (-1.0, -1.0)

        result_container['spectrogram_cluster_ID'] = Clustering(result_container['spectrogram_trim_mask_aggr'],
                                                                q=q, s=s, freq_octaves=log_freq_cluster)

    def apply_ClusterCatalogs(self, result_container, /):
        opts = self.cluster_catalogs_opts or {}

        feature_funcs = self.cluster_catalogs_funcs

        opts.setdefault("trace_dim_names", None)

        cluster_ID = result_container.get('spectrogram_cluster_ID', None)
        update_cluster_ID = opts.pop('update_cluster_ID', False)

        vals_keys = self.cluster_feature_distributions
        assert isinstance(vals_keys, (list, tuple)) and len(vals_keys) > 0

        values_dict = {}
        for name in vals_keys:
            val = result_container.get(name, None)  # take feature distribution from 'result_container'
            if val is not None:
                values_dict[name] = val

        if (cluster_ID is not None) and (len(values_dict) > 0):
            tf_time = np.arange(cluster_ID.shape[-1]) * self.dt_sec
            frequencies = result_container['frequencies']

            catalog = ClusterCatalogs(values_dict=values_dict,
                                      CID=cluster_ID,
                                      freq=frequencies,
                                      time=tf_time,
                                      feature_funcs=feature_funcs,
                                      aggr_clustering_axis=self.aggr_clustering_axis,
                                      **opts)

            catalog = CATSResult.filter_cluster_catalogs(catalog,
                                                         self.cluster_catalogs_filter,
                                                         cluster_ID if update_cluster_ID else None)
        else:
            catalog = None

        result_container['cluster_catalogs'] = catalog

    def apply_ICWT(self, result_container, /):
        weights = result_container['spectrogram_cluster_ID'] > 0
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
        del_vals_by_keys(result, full_info, ['spectrogram_SNR', 'spectrogram_trim_mask', 'noise_std',
                                             'noise_threshold_conversion'])

        # Clustering
        self.apply_func(func_name='apply_Clustering', result_container=result, status_keeper=history,
                        process_name='Clustering')

        # Cluster catalog
        self.cluster_catalogs_opts.setdefault('update_cluster_ID', True)
        self.apply_func(func_name='apply_ClusterCatalogs', result_container=result, status_keeper=history,
                        process_name='Cluster catalog')

        del_vals_by_keys(result, full_info, ['spectrogram', 'spectrogram_trim_mask_aggr'])

        # Inverse STFT
        self.apply_func(func_name='apply_ICWT', result_container=result, status_keeper=history,
                        process_name='Inverse CWT')

        del_vals_by_keys(result, full_info, ['coefficients', 'spectrogram_cluster_ID'])

        history.print_total_time()

        from_full_info = {kw: result.get(kw, None) for kw in full_info}

        result = CATSDenoisingCWTResult(dt_sec=self.dt_sec,
                                        time_npts=x.shape[-1],
                                        time_frames=result['time_frames'] * self.dt_sec,
                                        minSNR=self.minSNR,
                                        history=history,
                                        main_params=self.main_params,
                                        **from_full_info)

        return result

    def denoise(self, x: np.ndarray,
                /,
                verbose: bool = False,
                full_info: Union[bool, str, List[str]] = False):

        data, stats = self.convert_input_data(x)

        n_chunks = self.split_data_by_memory(data, full_info=full_info, to_file=False)
        single_chunk = n_chunks > 1
        data_chunks = np.array_split(data, n_chunks, axis=-1)
        results = []
        for dc in tqdm(data_chunks, desc='Data chunks', display=single_chunk):
            results.append(self._denoise(dc, verbose=verbose, full_info=full_info))

        result = CATSDenoisingCWTResult.concatenate(*results)
        setattr(result, "stats", stats)
        return result

    def apply(self, x: Union[np.ndarray, obspy.Stream],
              /,
              verbose: bool = False,
              full_info: Union[bool, str, List[str]] = False):
        """ Alias of `.denoise`. For compatibility with CATSBase """
        return self.denoise(x, verbose=verbose, full_info=full_info)

    def __mul__(self, x):
        return self.apply(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.apply(x, verbose=True, full_info='qc')

    def __matmul__(self, x):
        return self.apply(x, verbose=True, full_info=True)

    def convert_input_data(self, x):
        if isinstance(x, obspy.Stream):
            x_dict = convert_stream_to_dict(x)
            data, stats = x_dict['data'], x_dict['stats']
            dt_sec_data = stats.flat[0].delta
            if self.dt_sec != dt_sec_data:  # update sampling rate if different from data
                self.reset_params(dt_sec=dt_sec_data)
        else:
            data, stats = x, None
        return data, stats

    @staticmethod
    def get_all_keys():
        return ["signal",
                "coefficients",
                "spectrogram",
                "noise_std",
                "noise_threshold_conversion",
                "spectrogram_SNR",
                "spectrogram_trim_mask",
                "spectrogram_trim_mask_aggr",
                "spectrogram_cluster_ID",
                "cluster_catalogs",
                "scales",
                "frequencies",
                "scales_groups_indexes",
                "signal_denoised"]

    @classmethod
    def get_default_keys(cls):
        return ["noise_std", "noise_threshold_conversion", 'signal_denoised', "cluster_catalogs",]

    @classmethod
    def get_qc_keys(cls):
        return ["signal", "spectrogram", "spectrogram_trim_mask", "spectrogram_cluster_ID",
                "noise_std", "noise_threshold_conversion", "cluster_catalogs",
                "signal_denoised", "scales", "frequencies"]

    @classmethod
    def get_cluster_catalog_keys(cls):
        return ["spectrogram", "spectrogram_cluster_ID"]

    def parse_info_dict(self, full_info):

        if isinstance(full_info, str):
            if full_info in ['plot', 'plotting', 'plt', 'qc', 'main']:  # only those needed for plotting step-by-step
                full_info = self.get_qc_keys()
            elif full_info in ['cluster', 'cluster_catalog', 'cc', 'features']:
                full_info = self.get_cluster_catalog_keys()
            else:
                raise KeyError(f"Unknown {full_info = } code")
        elif isinstance(full_info, bool):
            full_info = self.get_all_keys() * full_info
        elif isinstance(full_info, list):
            pass
        else:
            raise ValueError(f"Invalid {type(full_info) = }")

        # add necessary, repetitions will be deleted
        full_info += self.get_default_keys()  # defaults
        full_info += self.cluster_feature_distributions  # for cluster catalogs

        # Parse
        full_info = cast_to_bool_dict(full_info, self.get_all_keys())

        return full_info

    def _memory_usage(self, x):
        time_len = x.shape[-1]
        scale_len = len(self.CWT.get_scales(time_len))
        ft_shape = x.shape[:-1] + (scale_len, time_len)
        ft_size = np.prod(np.float64(ft_shape))

        scale_groups_len = len(divide_into_groups(num_freqs=scale_len, group_step=self.bedate_scale_grouping_len,
                                                  is_octaves_step=False, fft_base=False))
        time_frames_len = len(get_interval_division(N=time_len, L=self.stationary_frame_len))
        bedate_shape = x.shape[:-1] + (scale_groups_len, time_frames_len)
        bedate_size = np.prod(np.float64(bedate_shape))

        cluster_sc = 2.0  # clustering requires < 2 copies of spectrogram (max estimate)
        if self.aggr_clustering_axis is not None:
            cluster_sc /= x.shape[self.aggr_clustering_axis]  # reduced by aggregation (if applied)

        x_bytes = x.nbytes
        precision_order = x_bytes / x.size

        memory_usage_bytes = {
            "scales":                       8. * scale_len,                               # scale
            "frequencies":                  8. * scale_len,                               # freqs
            "time_frames":                  8. * bedate_shape[-1],                        # float64 always
            "signal":                       1. * x_bytes,                                 # float / int
            "coefficients":                 2. * precision_order * ft_size,               # complex
            "spectrogram":                  1. * precision_order * ft_size,               # float
            "noise_threshold_conversion":   8. * bedate_shape[-2],                        # float
            "noise_std":                    1. * precision_order * bedate_size,           # float
            "spectrogram_SNR":              1. * precision_order * ft_size,               # float
            "spectrogram_trim_mask":        1. * ft_size,                                 # bool
            "spectrogram_trim_mask_aggr":   1. * ft_size * cluster_sc,                    # bool
            "spectrogram_cluster_ID":       4. * ft_size,                                 # uint32 always
        }

        used_together = [('coefficients', 'spectrogram'),  # STFT step
                         ('coefficients', 'spectrogram', 'noise_threshold_conversion', 'noise_std',
                          'spectrogram_SNR', "spectrogram_trim_mask", "spectrogram_trim_mask_aggr"),  # BEDATE step
                         ('coefficients', 'spectrogram_trim_mask_aggr', 'spectrogram_cluster_ID'),  # Clustering
                         ('coefficients', 'signal_denoised')]  # Inverse STFT

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

    def memory_usage_estimate(self, x, /, full_info=False):
        memory_usage_bytes, _ = self._memory_usage(x)
        full_info = self.parse_info_dict(full_info)

        memory_usage_bytes_new = {
                "signal_denoised":         memory_usage_bytes['signal'],        # float
        }
        memory_usage_bytes.update(memory_usage_bytes_new)

        used_together = [('coefficients', 'spectrogram'),
                         ('coefficients', 'spectrogram', 'noise_threshold_conversion',
                          'noise_std', 'spectrogram_trim_mask'),
                         ('coefficients', 'spectrogram_trim_mask_aggr', 'spectrogram_cluster_ID'),
                         ('coefficients', 'signal_denoised')]

        base_info = ["signal", "frequencies", "scales", "time_frames"]

        return self.memory_info(memory_usage_bytes, used_together, base_info, full_info)

    def split_data_by_memory(self, x, /, full_info, to_file):
        return self.memory_chunks(self.memory_usage_estimate(x, full_info=full_info), to_file)


class CATSDenoisingCWTResult(CATSDenoisingResult):
    scales: Any = None
    scales_groups_indexes: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tf_dt_sec = self.tf_dt_sec or self.dt_sec
        self.tf_time_npts = self.tf_time_npts or self.time_npts

    def plot(self,
             ind: Tuple[int] = None,
             time_interval_sec: Tuple[float] = None,
             SNR_spectrograms: bool = False,
             show_frequency=True,
             weighted_coefficients: bool = False,
             intervals: bool = False,
             picks: bool = False,
             interactive: bool = False,
             ):
        freqs = self.frequencies if show_frequency else self.scales
        plot_output = super(CATSDenoisingResult, self).plot(ind=ind,
                                                            time_interval_sec=time_interval_sec,
                                                            SNR_spectrograms=SNR_spectrograms,
                                                            interactive=interactive,
                                                            frequencies=freqs)
        fig, opts, inds_slices, time_interval_sec = plot_output

        fig = self._plot(weighted_coefficients=weighted_coefficients,
                         intervals=intervals,
                         picks=picks,
                         fig=fig,
                         opts=opts,
                         inds_slices=inds_slices,
                         time_interval_sec=time_interval_sec)

        if not show_frequency:
            mpl_opts_spects = hv.opts.QuadMesh(invert_yaxis=True, backend='matplotlib')  # for scales
            bkh_opts_spects = hv.opts.QuadMesh(invert_yaxis=True, backend='bokeh')  # for scales
            fig = fig.opts(mpl_opts_spects, bkh_opts_spects).redim(Frequency=hv.Dimension("Scale"))
        return fig
