"""
    Implements baseclasses for Cluster Analysis of Trimmed Spectrograms (CATS)

        CATSBase : baseclass to perform CATS
        CATSResult : baseclass for keeping the results of CATS
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import holoviews as hv
from typing import Any, Union, Callable, List
from pydantic import BaseModel, Field
import obspy
from typing import Tuple
from tqdm.notebook import tqdm

from .core.utils import replace_int_by_slice
from .core.timefrequency import STFTOperator
from .core.clustering import Clustering, ClusterCatalogs, concatenate_cluster_catalogs, index_cluster_catalog
from .core.clustering import bbox_peaks, update_cluster_ID_by_catalog
from .core.date import BEDATE_trimming, divide_into_groups
from .core.env_variables import get_min_bedate_block_size, get_max_memory_available_for_cats
from .core.utils import (get_interval_division, format_index_by_dimensions, cast_to_bool_dict,
                         intervals_intersection_inds, aggregate_array_by_axis_and_func, make_default_index_if_outrange)
from .core.utils import format_interval_by_limits, give_index_slice_by_limits, make_default_index_on_axis
from .core.utils import format_index_by_dimensions_new, give_nonzero_limits, mat_structure_to_tight_dataframe_dict
from .core.plottingutils import (plot_traces, switch_plotting_backend, detrend_func, save_all_segments_to_file,
                                 populate_index)
from .io.utils import convert_stream_to_dict, save_pickle, load_pickle
from .core.phaseseparation import NewMapFromClusters, PhaseSeparator, event_id_recorder
from .core.projection import ProjectCatalogs


# ------------------------ BASE CLASSES ------------------------ #


class CATSBase(BaseModel, extra='allow', arbitrary_types_allowed=True):
    """
        Base class for CATS based on STFT. Implements unpacking of the main parameters.
        Implements 4 main steps:
            1) STFT
            2) Noise estimation via B-E-DATE
            3) Trimming spectrogram based minSNR
            4) Clustering trimmed spectrogram
        """

    dt_sec: float

    # main STFT params
    stft_window_type: str = 'hann'
    stft_window_sec: float = 0.5
    stft_overlap: float = Field(0.75, ge=0.0, lt=1.0)

    # main B-E-DATE params
    minSNR: float = 8.0
    stationary_frame_sec: Union[float, None] = None

    # main Clustering params
    cluster_size_t_sec: float = 0.2
    cluster_size_f_Hz: float = 15.0
    cluster_size_f_octaves: Union[float, None] = None
    cluster_distance_t_sec: Union[float, None] = None
    cluster_distance_f_Hz: Union[float, None] = None
    cluster_distance_f_octaves: Union[float, None] = None

    # multi-component clustering
    aggr_clustering_axis: Union[int, Tuple[int], None] = None
    full_aggregated_mask: Union[bool, None] = False

    # main cluster catalog params
    cluster_catalogs_funcs: Union[List[Callable], None] = None
    cluster_feature_distributions: Union[List[str], None] = None
    cluster_catalogs_opts: Union[dict, None] = None
    cluster_catalogs_filter: Union[Callable, None] = None

    # Phase separation
    phase_separation: Union[PhaseSeparator, Callable, None] = None

    # mCATS
    clustering_multitrace: Union[bool, None] = False
    cluster_size_trace: int = Field(1, ge=1)
    cluster_distance_trace: int = Field(1, ge=1)

    # Extra STFT params
    freq_bandpass_Hz: Union[tuple[float, float], Any, None] = None
    stft_backend: str = 'ssqueezepy'
    stft_kwargs: Union[dict, None] = None
    stft_nfft: int = -1

    # Extra B-E-DATE params
    bedate_freq_grouping_Hz: Union[float, None] = None
    bedate_freq_grouping_octaves: Union[float, None] = None
    date_Q: float = 0.95
    date_Nmin_percentile: float = Field(0.25, gt=0.0, lt=0.5)
    date_original_mode: bool = False

    # Misc
    name: str = "CATS"

    def __init__(self, **kwargs):
        """
            Arguments:
                dt_sec : float : sampling time in seconds.

                stft_window_type: str : type of weighting window in STFT (e.g. `hann`, `hamming`, 'ones). Default 'hann'

                stft_window_sec : float : length of weighting window in seconds.

                stft_overlap : float [0, 1) : overlapping of STFT windows (e.g. `0.5` means 50% overlap).

                stft_nfft : int : zero-padding for each individual STFT window, recommended a power of 2 (e.g. 256).

                minSNR : float : minimum Signal-to-Noise Ratio (SNR) for B-E-DATE algorithm.

                stationary_frame_sec : float : length of time frame in seconds wherein noise is stationary. \
Length will be adjusted to have at least 256 elements in one frame. Default, 0.0 which leads to minimum possible

                bedate_freq_grouping_Hz : float : frequency grouping width in Hz. Default min possible is set

                bedate_freq_grouping_octaves : float : frequency grouping width in octaves (2^octaves). \
                Default None, not applied. If given, then `bedate_freq_grouping_Hz` is ignored.

                cluster_size_t_sec : float : minimum cluster size in time, in seconds. \
Can be estimated as length of the strongest phases.

                cluster_size_f_Hz : float : minimum cluster size in frequency, in hertz, i.e. minimum frequency width.

                cluster_distance_t_sec : float : neighborhood distance in time for clustering, in seconds. \
Minimum separation time between two different events.

                cluster_distance_f_Hz : float : neighborhood distance in frequency for clustering, in hertz. \
Minimum separation frequency width between two different events.

                cluster_size_f_octaves : float : cluster frequency width in octaves (2^octaves). \
                If not None, then it will be applied instead of `cluster_size_f_Hz`. Default None.

                cluster_distance_f_octaves : float : clustering distance in octaves (2^octaves). \
                If not None, then it will be applied instead of `cluster_distance_f_Hz`. Default None.

                aggr_clustering_axis : int or tuple[int] or None : axis to aggregate clustering mask by. Supposed
                to be components of multi-component data (e.g. 3-C or receiver groups).

                full_aggregated_mask : bool : whether to use the full aggregated mask (True) or only the trimmed
                elements masked by aggregated mask (False). Default, False.

                cluster_catalogs_opts : dict : options for calculating catalogs

                cluster_catalogs_filter : callable : function to filter cluster catalogs

                phase_separation : callable : function to separate phases in each event

                freq_bandpass_Hz : tuple[float, float] : bandpass frequency range, in hertz, i.e. everything out of \
the range is zero (e.g. (f_min, f_max)).

                clustering_multitrace : bool : whether to use multitrace clustering (Location x Frequency x Time). \
Increase accuracy for multiple arrays of receivers on regular grid. Performs cross-station association.

                cluster_size_trace : int : minimum cluster size for traces, minimum number of traces in one cluster.

                cluster_distance_trace : int : neighborhood distance across multiple traces for clustering.

                date_Q : float : probability that sorted elements after certain `Nmin` have amplitude higher \
than standard deviation. Used in Bienaymé–Chebyshev inequality to get `Nmin` to minimize cost of DATE. Default `0.95`

                date_Nmin_percentile : float : percentile of data to use as `Nmin` (see above `Q`), supersedes `date_Q`

                date_original_mode : bool : `True` means to use the original implementation of DATE algorithm. \
Original implementation assumes that if no outliers are found then standard deviation is estimated from `Nmin` \
to not overestimate the noise, but leads to underestimation. `False` uses our adaptation: if no outliers \
are found, then no outliers will be present in the trimmed spectrogram.

                stft_backend : str : backend for STFT operator ['scipy', 'ssqueezepy', 'ssqueezepy_gpu']. \
The fastest CPU version is 'ssqueezepy', which is default.

                stft_kwargs : dict : additional keyword arguments for STFT operator (see `cats.STFTOperator`).

                name : str : name of the object
        """
        super().__init__(**kwargs)
        self._init_params = kwargs  # this first because if `reset_params` raises error, the instance gets broken
        self._set_params()

    def _set_params(self):
        # Setting STFT
        stft_kwargs = {} if self.stft_kwargs is None else self.stft_kwargs
        self.STFT = STFTOperator(window_specs=(self.stft_window_type, self.stft_window_sec), overlap=self.stft_overlap,
                                 dt_sec=self.dt_sec, nfft=self.stft_nfft, backend=self.stft_backend, **stft_kwargs)
        self.stft_overlap_len = self.STFT.noverlap
        self.stft_overlap_sec = (self.stft_overlap_len - 1) * self.dt_sec

        self.stft_window = self.STFT.window
        self.stft_window_len = len(self.stft_window)
        self.stft_window_sec = (self.stft_window_len - 1) * self.dt_sec
        self.stft_frequency = self.STFT.f
        self.stft_df = self.STFT.df
        self.stft_nfft = self.STFT.nfft
        self.stft_hop_len = self.STFT.hop
        self.stft_hop_sec = self.stft_hop_len * self.dt_sec
        self.tf_dt_sec = self.stft_hop_sec

        # DATE params
        self.stationary_frame_sec = v if (v := self.stationary_frame_sec) is not None else 0.0
        self.stationary_frame_len = max(round(self.stationary_frame_sec / self.stft_hop_sec),
                                        get_min_bedate_block_size())
        self.stationary_frame_sec = self.stationary_frame_len * self.stft_hop_sec

        # Bandpass by zeroing
        self.freq_bandpass_Hz = format_interval_by_limits(self.freq_bandpass_Hz,
                                                          (self.stft_frequency[0], self.stft_frequency[-1]))
        self.freq_bandpass_Hz = (min(self.freq_bandpass_Hz), max(self.freq_bandpass_Hz))
        self.freq_bandwidth_Hz = self.freq_bandpass_Hz[1] - self.freq_bandpass_Hz[0]
        self.freq_bandpass_slice = give_index_slice_by_limits(self.freq_bandpass_Hz, self.STFT.df)

        # Clustering params
        self.cluster_size_t_len = max(round(self.cluster_size_t_sec / self.stft_hop_sec), 1)
        self.cluster_size_f_len = max(round(self.cluster_size_f_Hz / self.STFT.df), 1)
        self.cluster_size_trace_len = self.cluster_size_trace

        self.cluster_size_f_octaves = self.cluster_size_f_octaves or -1.0
        self.cluster_distance_f_octaves = self.cluster_distance_f_octaves or -1.0

        self.cluster_distance_t_sec = v if (v := self.cluster_distance_t_sec) is not None else self.stft_hop_sec
        self.cluster_distance_f_Hz = v if (v := self.cluster_distance_f_Hz) is not None else self.stft_df
        self.cluster_distance_t_len = max(round(self.cluster_distance_t_sec / self.stft_hop_sec), 1)
        self.cluster_distance_f_len = max(round(self.cluster_distance_f_Hz / self.stft_df), 1)
        self.cluster_distance_trace_len = self.cluster_distance_trace

        if self.cluster_size_f_octaves <= 0.0:
            assert (fbw := self.freq_bandwidth_Hz) > (csf := self.cluster_size_f_Hz), \
                f"Frequency bandpass width `{fbw}` must be bigger than min frequency cluster size `{csf}`"

        self.time_edge = int(self.stft_window_len // 2 / self.stft_hop_len)

        self.min_duration_len = max(self.cluster_size_t_len, 1)  # `-1` to include bounds (0, 1, 0)
        self.min_separation_len = max(self.cluster_distance_t_len + 1, 2)  # `+1` to exclude bounds (1, 0, 1)
        self.min_duration_sec = self.min_duration_len * self.stft_hop_sec
        self.min_separation_sec = self.min_separation_len * self.stft_hop_sec

        # Catalog params
        if self.cluster_catalogs_funcs is None:
            self.cluster_catalogs_funcs = [event_id_recorder, bbox_peaks]
        else:
            self.cluster_catalogs_funcs = [event_id_recorder] + self.cluster_catalogs_funcs

        self.cluster_feature_distributions = (self.cluster_feature_distributions or ['spectrogram', 'spectrogram_SNR'])
        self.cluster_feature_distributions = self.cluster_feature_distributions + ['spectrogram_event_ID']

        self.cluster_catalogs_opts = self.cluster_catalogs_opts or {}

        # BEDATE grouping
        self.bedate_freq_grouping_Hz = self.bedate_freq_grouping_Hz or self.stft_df
        self.bedate_freq_grouping_octaves = self.bedate_freq_grouping_octaves or 0.0

        self.freq_group_step = max(int(round(self.bedate_freq_grouping_Hz / self.stft_df)), 1)
        self.is_octaves_step = self.bedate_freq_grouping_octaves > 0.0
        self.freq_group_step = self.bedate_freq_grouping_octaves if self.is_octaves_step else self.freq_group_step

        self.frequency_groups_indexes = divide_into_groups(num_freqs=len(self.stft_frequency),
                                                           group_step=self.freq_group_step,
                                                           is_octaves_step=self.is_octaves_step,
                                                           fft_base=True)

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

    def save(self, filename, as_dict=False):
        obj = self.init_params if as_dict else self
        save_pickle(obj, filename)

    @classmethod
    def load(cls, filename, by_param_set='main_params'):
        """
            Loads CATS object from a file.

            Arguments:
                filename : str : path to the file with CATS object.
                by_param_set : str : which set of parameters to use for loading,
                    'all_params', 'main_params', 'init_params'. If None, then full original object is loaded.
        """
        loaded = load_pickle(filename)
        if isinstance(loaded, dict):
            return cls(**loaded)
        else:
            assert by_param_set in ['all_params', 'main_params', 'init_params', None], \
                f"Unknown {by_param_set = } code. [None, 'all_params', 'main_params', 'init_params'] are allowed."

            if isinstance(by_param_set, str):
                return cls(**getattr(loaded, by_param_set))
            else:
                return loaded

    @classmethod
    def from_result(cls, CATSResult):
        return cls(**CATSResult.main_params)

    def apply_STFT(self, result_container, /):
        result_container['coefficients'] = self.STFT * result_container['signal']
        result_container['spectrogram'] = np.abs(result_container['coefficients'])

    def apply_BEDATE(self, result_container, /):
        result_container['time_frames'] = get_interval_division(N=result_container['spectrogram'].shape[-1],
                                                                L=self.stationary_frame_len)

        SNR, T, STD, THR = BEDATE_trimming(result_container['spectrogram'],
                                           self.frequency_groups_indexes, self.freq_bandpass_slice,
                                           result_container['time_frames'], self.minSNR, self.time_edge,
                                           Q=self.date_Q, Nmin_percentile=self.date_Nmin_percentile,
                                           original_mode=self.date_original_mode, fft_base=True, dim=2)

        result_container['spectrogram_SNR'] = SNR
        result_container['spectrogram_trim_mask'] = T
        result_container['noise_std'] = STD
        result_container['noise_threshold_conversion'] = THR

        # Aggregate trimmed spectrogram over `aggr_clustering_axis` by `aggr_clustering_func`
        # Useful if multi-component data are used (3-C or receiver groups)
        result_container['spectrogram_trim_mask_aggr'] = aggregate_array_by_axis_and_func(T,
                                                                                          self.aggr_clustering_axis,
                                                                                          func='any', min_last_dims=2)

    def apply_Clustering(self, result_container, /):
        mc = self.clustering_multitrace
        q = (self.cluster_distance_trace_len,) * mc + (self.cluster_distance_f_len, self.cluster_distance_t_len)
        s = (self.cluster_size_trace_len,) * mc + (self.cluster_size_f_len, self.cluster_size_t_len)
        freq_octaves = (self.cluster_size_f_octaves, self.cluster_distance_f_octaves)

        result_container['spectrogram_cluster_ID'] = Clustering(result_container['spectrogram_trim_mask_aggr'],
                                                                q=q, s=s, freq_octaves=freq_octaves)
        result_container['spectrogram_event_ID'] = result_container['spectrogram_cluster_ID']

    def apply_PhaseSeparation(self, result_container, /, tf_time, frequencies):

        cluster_ID = result_container.get('spectrogram_cluster_ID', None)

        d_name = self.phase_separation.on_distribution_name
        self.phase_separation.set_dt_sec(self.tf_dt_sec)
        distribution = result_container.get(d_name, None)
        values_dict = {d_name: distribution}

        if (cluster_ID is not None) and (distribution is not None):
            Maps = NewMapFromClusters(values_dict=values_dict,
                                      CID=cluster_ID,
                                      freq=frequencies,
                                      time=tf_time,
                                      map_funcs=[self.phase_separation],
                                      out_dtypes=[cluster_ID.dtype],
                                      out_is_aggr=[True],
                                      aggr_clustering_axis=self.aggr_clustering_axis)

            result_container['spectrogram_cluster_ID'] = Maps[0]  # Phase IDs

    def apply_ClusterCatalogs(self, result_container, /, tf_time, frequencies):
        opts = self.cluster_catalogs_opts or {}

        feature_funcs = self.cluster_catalogs_funcs

        opts.setdefault("trace_dim_names", None)
        cluster_ID = result_container.get('spectrogram_cluster_ID', None)
        update_cluster_ID = opts.pop('update_cluster_ID', False)

        vals_keys = self.cluster_feature_distributions + ['spectrogram_event_ID']
        assert isinstance(vals_keys, (list, tuple)) and len(vals_keys) > 0

        values_dict = {}
        for name in vals_keys:
            val = result_container.get(name, None)  # take feature distribution from 'result_container'
            if val is not None:
                values_dict[name] = val

                if (not self.full_aggregated_mask) and (val.shape == result_container['spectrogram_trim_mask'].shape):
                    values_dict[name] = val * result_container['spectrogram_trim_mask']

        if (cluster_ID is not None) and (len(values_dict) > 0):

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

    def apply_func(self, func_name, result_container, status_keeper, process_name=None, **kwargs):
        process_name = process_name or func_name
        with status_keeper(current_process=process_name):
            getattr(self, func_name)(result_container, **kwargs)

    @classmethod
    def get_all_keys(cls):
        return ["signal",
                "coefficients",
                "spectrogram",
                "noise_std",
                "noise_threshold_conversion",
                "spectrogram_SNR",
                "spectrogram_trim_mask",
                "spectrogram_trim_mask_aggr",
                "spectrogram_event_ID",
                "spectrogram_cluster_ID",
                "cluster_catalogs"]

    @classmethod
    def get_qc_keys(cls):
        return ["signal", "spectrogram", "spectrogram_trim_mask", "noise_std", "noise_threshold_conversion",
                "spectrogram_event_ID", "spectrogram_cluster_ID", "cluster_catalogs"]

    @classmethod
    def get_cluster_catalog_keys(cls):
        return ["spectrogram_event_ID", "spectrogram_cluster_ID"]

    @classmethod
    def get_default_keys(cls):
        return ["noise_std", "noise_threshold_conversion"]

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

        stft_time_len = len(self.STFT.forward_time_axis(x.shape[-1]))
        stft_shape = x.shape[:-1] + (len(self.stft_frequency), stft_time_len)
        stft_size = np.prod(np.float64(stft_shape))  # float is to prevent `int` overflow to negatives

        bedate_shape = x.shape[:-1] + (len(self.frequency_groups_indexes),
                                       len(get_interval_division(N=stft_time_len,
                                                                 L=self.stationary_frame_len)))
        bedate_size = np.prod(np.float64(bedate_shape))  # float is to prevent `int` overflow to negatives

        cluster_sc = 2.0  # clustering requires < 2 copies of spectrogram (max estimate)
        cluster_sc += 1.0  # for 'cluster ID' and 'event ID' copies
        if self.aggr_clustering_axis is not None:
            cluster_sc /= x.shape[self.aggr_clustering_axis]  # reduced by aggregation (if applied)

        x_bytes = x.nbytes
        precision_order = x_bytes / x.size

        memory_usage_bytes = {
            "frequencies":                  8. * stft_shape[-2],                            # float64 always
            "time_frames":                  8. * bedate_shape[-1],                          # float64 always
            "signal":                       1. * x_bytes,                                   # float64 / int64
            "coefficients":                 2. * precision_order * stft_size,               # complex64
            "spectrogram":                  1. * precision_order * stft_size,               # float64
            "noise_threshold_conversion":   8. * bedate_shape[-2],                          # float64
            "noise_std":                    1. * precision_order * bedate_size,             # float64
            "spectrogram_SNR":              1. * precision_order * stft_size,               # float64
            "spectrogram_trim_mask":        1. * stft_size,                                 # bool
            "spectrogram_trim_mask_aggr":   1. * stft_size * cluster_sc,                    # bool
            "spectrogram_cluster_ID":       4. * stft_size * cluster_sc,                    # int32 always
        }
        used_together = [('coefficients', 'spectrogram'),  # STFT step
                         ('spectrogram', 'noise_threshold_conversion', 'noise_std', 'spectrogram_SNR',
                          "spectrogram_trim_mask", "spectrogram_trim_mask_aggr"),  # BEDATE step
                         ('spectrogram_trim_mask_aggr', 'spectrogram_cluster_ID')  # Clustering
                         ]

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

    def convert_input_data(self, x):
        if isinstance(x, obspy.Stream):
            x_dict = convert_stream_to_dict(x)
            data, stats = x_dict['data'], x_dict['stats']
            dt_sec_data = stats.flat[0].delta
            if self.dt_sec != dt_sec_data:  # update sampling rate if different from data
                self.reset_params(dt_sec=dt_sec_data, stft_nfft=-1)
        else:
            data, stats = x, None
        return data, stats

    def __mul__(self, x):
        return self.apply(x, verbose=False, full_info=False)

    def __pow__(self, x):
        return self.apply(x, verbose=True, full_info='qc')

    def __matmul__(self, x):
        return self.apply(x, verbose=True, full_info=True)


class CATSResult(BaseModel):
    signal: Any = None
    coefficients: Any = None
    spectrogram: Any = None
    noise_threshold_conversion: Any = None
    noise_std: Any = None
    spectrogram_SNR: Any = None
    spectrogram_trim_mask: Any = None
    spectrogram_trim_mask_aggr: Any = None
    spectrogram_event_ID: Any = None
    spectrogram_cluster_ID: Any = None
    cluster_catalogs: Any = None
    detected_intervals: Any = None
    picked_features: Any = None
    dt_sec: float = None
    tf_dt_sec: float = None
    tf_t0_sec: float = 0.0
    time_npts: int = None
    tf_time_npts: int = None
    frequencies: Any = None
    time_frames: Any = None
    frequency_groups_indexes: Any = None
    minSNR: float = None
    history: Any = None
    main_params: dict = None
    stats: Any = None
    picks_stream: Any = None

    @staticmethod
    def base_time_func(npts, dt_sec, t0, time_interval_sec):
        if time_interval_sec is None:
            return t0 + np.arange(npts) * dt_sec
        else:
            time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (npts - 1) * dt_sec))
            time_slice = give_index_slice_by_limits(time_interval_sec, dt_sec)
            return t0 + np.arange(time_slice.start, time_slice.stop) * dt_sec

    def time(self, time_interval_sec=None):
        return CATSResult.base_time_func(self.time_npts, self.dt_sec, 0, time_interval_sec)

    def tf_time(self, time_interval_sec=None):
        return CATSResult.base_time_func(self.tf_time_npts, self.tf_dt_sec, 0, time_interval_sec)

    @classmethod
    def get_qc_keys(cls):
        return CATSBase.get_qc_keys()

    def check_qc_attributes(self):
        qc_keys = self.get_qc_keys()
        missed_keys = list(filter(lambda key: getattr(self, key, None) is None, qc_keys))
        if missed_keys:
            raise AttributeError(f"Result does not have enough saved attributes to plot. "
                                 f"Required:\n{qc_keys},\nmissing:\n{missed_keys}")

    def plot(self,
             ind=None,
             time_interval_sec=None,
             SNR_spectrograms=True,
             show_cluster_ID=False,
             show_aggregated=False,
             detrend_type='constant',
             aggr_func=np.max,
             interactive=False,
             frequencies=None):

        self.check_qc_attributes()  # check whether necessary attributes were saved for plotting

        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        a_dim = hv.Dimension('Amplitude')

        if frequencies is None:
            frequencies = self.frequencies

        ind = format_index_by_dimensions(ind=ind, shape=self.signal.shape[:-1], slice_dims=0, default_ind=0)

        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.time_npts - 1) * self.dt_sec))

        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        i_tf = give_index_slice_by_limits(time_interval_sec, self.tf_dt_sec)
        inds_time = ind + (i_time,)
        inds_tf = ind + (slice(None), i_tf)
        aggr_ind = make_default_index_if_outrange(ind, self.spectrogram_cluster_ID.shape[:-2], default_ind_value=0)
        inds_tf_cid = aggr_ind + (slice(None), i_tf)

        signal = self.signal[inds_time]
        if detrend_type:
            detrend_type = 'linear' if detrend_type == 'linear' else 'constant'
            signal = detrend_func(signal, axis=-1, type=detrend_type)

        PSD = self.spectrogram[inds_tf]
        SNR = self.spectrogram_SNR[inds_tf]
        T = self.spectrogram_trim_mask[inds_tf]
        CID = self.spectrogram_cluster_ID[inds_tf_cid]
        M = (CID > 0)
        if not self.main_params.get('full_aggregated_mask', False):
            M = M & T

        PSD_clims = give_nonzero_limits(PSD, initials=(1e-1, 1e1))
        PSD_vdim = hv.Dimension('Spectrogram')
        trim_mask = r'\cdot T(t,f)'
        cluster_mask = r'\cdot \tilde\mathcal{A}(t,f)'
        label_func = lambda name, var, mask: f'{name} spectrogram: $' f'|{var}(t,f)| {mask}$'

        # For aggregating
        aggr_axis = self.main_params.get('aggr_clustering_axis')
        aggr_ind2 = replace_int_by_slice(ind, aggr_axis) + (slice(None), i_tf)
        Vals = self.spectrogram_SNR if SNR_spectrograms else self.spectrogram
        Vals = Vals[aggr_ind2]
        if show_aggregated and (aggr_axis is not None):
            Vals = aggregate_array_by_axis_and_func(Vals, axis=0,  # aggr axis is 0 after indexing
                                                    func=aggr_func, min_last_dims=2)
        if Vals.ndim > 2:
            Vals = Vals[ind[aggr_axis]] if Vals.shape[0] > 1 else Vals[0]  # if not aggr func was applied
        C = Vals * M

        if SNR_spectrograms:
            TS = SNR * T
            label_trimmed = label_func('SNR', 'SNR', trim_mask)
            label_clustered = label_func('SNR', 'SNR', cluster_mask)
            TS_clims = give_nonzero_limits(C, initials=(1e-1, 1e1))
            TS_vdim = hv.Dimension('SNR')
        else:
            TS = PSD * T
            label_trimmed = label_func('', 'X', trim_mask)
            label_clustered = label_func('', 'X', cluster_mask)
            TS_clims = PSD_clims
            TS_vdim = PSD_vdim

        signal_opts = (hv.opts.Curve(xlabel='', backend='matplotlib'),
                       hv.opts.Curve(xlabel='', backend='bokeh'))
        PSD_opts = (hv.opts.QuadMesh(clim=PSD_clims, backend='matplotlib'),
                    hv.opts.QuadMesh(clim=PSD_clims, backend='bokeh'))
        TS_opts = (hv.opts.QuadMesh(clim=TS_clims, backend='matplotlib'),
                   hv.opts.QuadMesh(clim=TS_clims, backend='bokeh'))
        CID_opts = (hv.opts.QuadMesh(clim=(1, None), backend='matplotlib'),
                    hv.opts.QuadMesh(clim=(1, None), backend='bokeh'))

        time = self.time(time_interval_sec)
        tf_time = self.tf_time(time_interval_sec)

        fig0 = hv.Curve((time, signal), kdims=[t_dim], vdims=a_dim,
                        label='0. Input data: $x(t)$').opts(*signal_opts)
        fig1 = hv.QuadMesh((tf_time, frequencies, PSD), kdims=[t_dim, f_dim], vdims=PSD_vdim,
                           label='1. Amplitude spectrogram: $|X(t,f)|$').opts(*PSD_opts)
        fig2 = hv.QuadMesh((tf_time, frequencies, TS), kdims=[t_dim, f_dim], vdims=TS_vdim,
                           label=f'2. Trimmed {label_trimmed}').opts(*TS_opts)
        fig3 = hv.QuadMesh((tf_time, frequencies, C), kdims=[t_dim, f_dim], vdims=TS_vdim,
                           label=f'3. Clustered {label_clustered}').opts(*TS_opts)

        if show_cluster_ID:
            fig31 = [hv.QuadMesh((tf_time, frequencies, CID), kdims=[t_dim, f_dim],
                                 vdims=hv.Dimension("Cluster ID"),
                                 label=r'3.1. Cluster ID: $\tilde\mathcal{A}_{ID}(t,f)$').opts(*CID_opts)]
        else:
            fig31 = []

        cluster_catalogs = getattr(self, 'cluster_catalogs', None)
        catalog = index_cluster_catalog(cluster_catalogs, aggr_ind).copy() if (cluster_catalogs is not None) else None

        fmin = frequencies[frequencies != 0].min()  # to handle 0 freq with Log axis

        if interactive and (catalog is not None):
            vdims = list(catalog.columns)[:21]  # before 'F1' & 'F2' (important before!), first 27 features
            # remove zeros for proper visualization of Frequency in Log scale
            f_unit = 'Hz' if "Frequency_start_Hz" in catalog.columns else 'octave'
            f_conversion = lambda x: (x or fmin) if f_unit == 'Hz' else 2**x
            catalog[["F1", 'F2']] = catalog[[f"Frequency_start_{f_unit}",
                                             f"Frequency_end_{f_unit}"]].map(f_conversion)
            if len(catalog) > 0:
                intervals = catalog[["Time_start_sec", "Time_end_sec"]].values
                catalog = catalog[intervals_intersection_inds(intervals, time_interval_sec)]
            kdims = ["Time_start_sec", "F1", "Time_end_sec", "F2"]

            cat_fig = hv.Rectangles(catalog, kdims=kdims, vdims=vdims, label='Catalog')
            bkh_cat_opts = hv.opts.Rectangles(line_width=1, fill_color='', line_color='k', line_alpha=0.04,
                                              hover_fill_color='black', hover_fill_alpha=0.1,
                                              hover_line_alpha=0.15, muted_fill_color='', muted_line_color='',
                                              hover_tooltips=vdims, tools=['hover'], backend='bokeh')
            cat_fig = cat_fig.opts(bkh_cat_opts)

            bkh_overlay_opts = hv.opts.Overlay(legend_opts={"click_policy": "hide"}, backend='bokeh')
            fig3 = (fig3 * cat_fig).relabel(fig3.label).opts(bkh_overlay_opts)

        # General params and Matplotlib
        fontsize = dict(labels=15, title=16, ticks=14, legend=10, cticks=14)
        figsize = 225
        aspect = 5
        cmap = 'viridis'
        xlim = time_interval_sec
        ylim = (fmin, None)
        layout_title = str(self.stats[ind].starttime) if (self.stats is not None) else ''
        # cymin = np.nanmin(fig0.data[a_dim.name])
        # cymax = np.nanmax(fig0.data[a_dim.name])
        # cylim = (cymin - abs(cymin) * 0.1, cymax + abs(cymax) * 0.1)
        cylim = (None, None)

        # bokeh params
        width = int(figsize * 0.9 * aspect * 0.8)
        height = int(figsize * 0.9)
        tools = ['hover']

        # Matplotlib backend params
        backend = 'matplotlib'
        mpl_curve_opts = hv.opts.Curve(aspect=aspect, fig_size=figsize, fontsize=fontsize, xlim=xlim, ylim=cylim,
                                       show_frame=True, backend=backend)
        mpl_spectr_opts = hv.opts.QuadMesh(cmap=cmap, colorbar=True, logy=True, norm='log', xlim=xlim, ylim=ylim,
                                           xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize,
                                           clipping_colors={'min': 'transparent'}, cbar_width=0.02, backend=backend)
        mpl_layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.35, title=layout_title,
                                         aspect_weight=0, sublabel_format='', backend=backend)

        # Bokeh backend params
        backend = 'bokeh'
        bkh_curve_opts = hv.opts.Curve(width=width, height=height, fontsize=fontsize, xlim=xlim, ylim=cylim,
                                       tools=tools, backend=backend)
        bkh_spectr_opts = hv.opts.QuadMesh(cmap=cmap, colorbar=True, logy=True, logz=True, xlim=xlim, ylim=ylim,
                                           xlabel='', clabel='', width=width, height=height, fontsize=fontsize,
                                           tools=tools, clipping_colors={'min': 'transparent'}, backend=backend)
        bkh_layout_opts = hv.opts.Layout(shared_axes=True, title=layout_title, merge_tools=True, backend=backend)

        # Finalising output
        inds_slices = (ind, i_time, i_tf)
        opts = (mpl_curve_opts, bkh_curve_opts,
                mpl_spectr_opts, bkh_spectr_opts,
                mpl_layout_opts, bkh_layout_opts)
        figs = [fig0, fig1, fig2, fig3] + fig31
        figs = hv.Layout(figs).opts(*opts)
        if fig31:
            fig31 = figs[-1].opts(hv.opts.QuadMesh(norm=None, backend='matplotlib'),
                                  hv.opts.QuadMesh(logz=False, backend='bokeh'))
        output = (figs, opts, inds_slices, time_interval_sec)

        switch_plotting_backend(interactive, fig_mpl='png', fig_bokeh='auto')  # switch MPL or BKH

        return output

    def plot_multi(self,
                   inds,
                   share_time_labels=True,
                   share_vertical_axes=True,
                   share_spectrogram_cmaps=True,
                   hspace=None,
                   vspace=None,
                   **kwargs):

        figs = [self.plot(ind, **kwargs) for ind in inds]
        nrows = len(figs[0])
        ncols = len(figs)

        if share_time_labels:
            for i, figs_i in enumerate(zip(*figs)):
                if i < nrows - 1:
                    for fi in figs_i:
                        fi = fi.opts(xaxis='bare')

        if share_vertical_axes:
            for i, fig_i in enumerate(figs):
                if i > 0:
                    fig_i = fig_i.opts(hv.opts.QuadMesh(yaxis='bare'),
                                       hv.opts.Curve(yaxis='bare'))
        if share_spectrogram_cmaps:
            for i, figs_i in enumerate(zip(*figs)):
                if (0 < i < nrows - 1):
                    clims = [fi.opts.get().kwargs['clim'] for fi in figs_i]
                    if all([(cl is not None) for i, cl in np.ndenumerate(clims)]):
                        clim = (np.min(clims), np.max(clims))
                        for fi in figs_i:
                            fi = fi.opts(clim=clim)

            for i, fig_i in enumerate(figs):
                if i < ncols - 1:
                    fig_i = fig_i.opts(hv.opts.QuadMesh(colorbar=False))

        layout_kwargs = figs[0].opts.get().kwargs
        layout_kwargs['title'] = f"Traces:  {inds}".replace('[', '').replace(']', '')
        layout_kwargs['title'] = layout_kwargs['title'].replace('),', ') ')
        layout_kwargs['fontsize'] = dict(title=22)
        fig = hv.Layout(figs).cols(nrows).opts(**layout_kwargs)

        vspace = vspace or 0.05 + 0.02 * (not share_time_labels)
        hspace = hspace or 0.025 + 0.1 * sum([not share_vertical_axes, not share_spectrogram_cmaps])

        fig = fig.opts(shared_axes=share_vertical_axes, vspace=vspace, hspace=hspace, transpose=True)

        return fig

    def get_intervals_picks(self, ind, catalog_picks):
        # Get proper dimensions associated with `ind`
        aggr_ax = self.main_params.get('aggr_clustering_axis')
        aggr_ind = make_default_index_on_axis(ind, aggr_ax, 0)

        shape = self.noise_std.shape[:-2]  # assume that noise_std is always present
        shape = (1,) if len(shape) == 0 else shape
        shape = make_default_index_on_axis(shape, aggr_ax, 1)  # to handle aggregation axis

        # Check attributes
        intv_none = (getattr(self, 'detected_intervals') is None)
        pick_none = (getattr(self, 'picked_features') is None)
        is_cluster_catalogs = (getattr(self, 'cluster_catalogs') is not None)

        if is_cluster_catalogs and (intv_none or pick_none):
            # If cluster catalogs are present, but intervals or picks are not, then calculate them
            detected_intervals, picked_features = ProjectCatalogs(shape,
                                                                  self.cluster_catalogs,
                                                                  dt_sec=self.tf_dt_sec,
                                                                  min_separation_sec=0,
                                                                  min_duration_sec=0,
                                                                  features_cols=catalog_picks)

            for i in np.ndindex(shape):
                picked_features[i] = picked_features[i].reshape(-1, 1)

            if intv_none:
                setattr(self, 'detected_intervals', detected_intervals)
            if pick_none:
                setattr(self, 'picked_features', picked_features)

        picked_onsets = np.empty(shape, dtype=object)
        for i in np.ndindex(shape):
            picked_onsets[i] = self.picked_features[i][:, 0]

        detected_intervals = self.detected_intervals[aggr_ind + (...,)]  # `(..., )` is to always keep `dtype=object`
        picked_onsets = picked_onsets[aggr_ind + (...,)]
        # Adjust if arrays are 0-dim
        zero_dim = detected_intervals.ndim == 0
        detected_intervals = detected_intervals.reshape(1) if zero_dim else detected_intervals
        picked_onsets = picked_onsets.reshape(1) if zero_dim else picked_onsets

        return detected_intervals, picked_onsets

    def plot_traces(self,
                    signal,
                    ind: Tuple[int] = None,
                    time_interval_sec: Tuple[float] = None,
                    intervals: bool = False,
                    picks: bool = False,
                    station_loc: np.ndarray = None,
                    gain: int = 0.5,
                    clip: bool = False,
                    detrend_type: str = 'constant',
                    each_station: int = 1,
                    amplitude_scale: float = None,
                    per_station_scale: bool = False,
                    component_labels: list[str] = None,
                    station_labels: list[str] = None,
                    interactive: bool = False,
                    allow_picking: bool = False,
                    **kwargs):
        ind = () if ind is None else ind
        ind = format_index_by_dimensions_new(ind=ind, ndim=signal.ndim - 1, default_ind=0)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.time_npts - 1) * self.dt_sec))
        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        traces = signal[ind + (i_time, )]

        arrival_cols = list(filter(lambda x: "arrival_sec" in x, self.cluster_catalogs.columns))
        picks_cols = ['Time_peak_sec'] + arrival_cols

        detected_intervals, picked_onsets = self.get_intervals_picks(ind, catalog_picks=picks_cols)
        detected_intervals = detected_intervals if intervals else None
        picked_onsets = picked_onsets if picks else None

        fig = plot_traces(traces,
                          self.time(time_interval_sec),
                          intervals=detected_intervals,
                          picks=picked_onsets,
                          station_loc=station_loc,
                          time_interval_sec=time_interval_sec,
                          gain=gain,
                          detrend_type=detrend_type,
                          amplitude_scale=amplitude_scale,
                          per_station_scale=per_station_scale,
                          clip=clip,
                          each_station=each_station,
                          interactive=interactive,
                          allow_picking=allow_picking,
                          component_labels=component_labels,
                          station_labels=station_labels,
                          **kwargs)

        stats = self.stats[ind] if (self.stats is not None) else None
        stats = stats.flat[0] if isinstance(stats, np.ndarray) else stats
        layout_title = str(stats.starttime) if (stats is not None) else ''

        if allow_picking:
            self.picks_stream = fig[1]  # object storing picks
            fig = fig[0]  # figure

        fig = fig.opts(title=layout_title)
        return fig

    def plot_traces_all_segments_to_file(self,
                                         folder_path,
                                         ind=None,
                                         segment_separation_sec=None,
                                         segment_extend_time_sec=None,
                                         window_interval_sec=None,
                                         filename=None,
                                         image_format='png',
                                         dpi=100,
                                         gc_every_iterations=5,
                                         **plot_traces_kwargs):

        if not isinstance(ind, list):
            ind = [ind]

        for ind_i in tqdm(ind, display=len(ind) > 1):
            save_all_segments_to_file(self,
                                      plot_function_name='plot_traces',
                                      ind=ind_i,
                                      folder_path=folder_path,
                                      segment_separation_sec=segment_separation_sec,
                                      segment_extend_time_sec=segment_extend_time_sec,
                                      window_interval_sec=window_interval_sec,
                                      filename=filename,
                                      image_format=image_format,
                                      dpi=dpi,
                                      gc_every_iterations=gc_every_iterations,
                                      **plot_traces_kwargs)

    def plot_all_segments_to_file(self,
                                  folder_path,
                                  ind=None,
                                  segment_separation_sec=None,
                                  segment_extend_time_sec=None,
                                  window_interval_sec=None,
                                  filename=None,
                                  image_format='png',
                                  dpi=100,
                                  gc_every_iterations=5,
                                  **plot_kwargs):

        shape = self.noise_std.shape[:-2]
        ind = populate_index(ind, shape)

        for ind_i in tqdm(ind, display=len(ind) > 1):
            save_all_segments_to_file(self,
                                      plot_function_name='plot',
                                      ind=ind_i,
                                      folder_path=folder_path,
                                      segment_separation_sec=segment_separation_sec,
                                      segment_extend_time_sec=segment_extend_time_sec,
                                      window_interval_sec=window_interval_sec,
                                      filename=filename,
                                      image_format=image_format,
                                      dpi=dpi,
                                      gc_every_iterations=gc_every_iterations,
                                      **plot_kwargs)

    def plot_multi_all_segments_to_file(self,
                                        folder_path,
                                        multi_axis,
                                        ind=None,
                                        segment_separation_sec=None,
                                        segment_extend_time_sec=None,
                                        window_interval_sec=None,
                                        filename=None,
                                        image_format='png',
                                        dpi=100,
                                        gc_every_iterations=5,
                                        **plot_multi_kwargs):

        shape = self.noise_std.shape[:-2]
        ind = populate_index(ind, shape, multi_axis)

        for ind_i in tqdm(ind, display=len(ind) > 1):
            save_all_segments_to_file(self,
                                      plot_function_name='plot_multi',
                                      ind=ind_i,
                                      folder_path=folder_path,
                                      segment_separation_sec=segment_separation_sec,
                                      segment_extend_time_sec=segment_extend_time_sec,
                                      window_interval_sec=window_interval_sec,
                                      filename=filename,
                                      image_format=image_format,
                                      dpi=dpi,
                                      gc_every_iterations=gc_every_iterations,
                                      **plot_multi_kwargs)

    def get_cluster_catalogs(self,
                             values_keys: List[str] = None,
                             feature_funcs: List[Callable] = None,
                             trace_dim_names: List[str] = None):
        """
            Calculates statistics/features of cluster in each trace

            Arguments:
                values_keys : list[str] : names of attributes / values in T-F domain to calculate features on
                                    (e.g. 'spectrogram', 'spectrogram_trim_mask_aggr')
                feature_funcs : list[callable] : Funcs of signature
                         `func(freq, time, values_dict, inds) -> dict[str, float]`,
                         where `freq`, `time`, `values` are 1D arrays, `inds = [freq_inds, time_inds]`.
                         Must return dict with the calculated features.
                trace_dim_names : list[str] : names of trace dimensions for DataFrame MultiIndex
        """
        assert (self.spectrogram_cluster_ID is not None), ("CATS result must contain `spectrogram_cluster_ID` "
                                                           "(use `full_info`)")

        if (values_keys is None) or (len(values_keys) == 0):
            values_keys = ['spectrogram', 'spectrogram_SNR']

        values_dict = {name: val for name in values_keys if (val := getattr(self, name)) is not None}

        catalog = ClusterCatalogs(values_dict=values_dict,
                                  CID=self.spectrogram_cluster_ID,
                                  freq=self.frequencies,
                                  time=self.tf_time(),
                                  feature_funcs=feature_funcs,
                                  aggr_clustering_axis=self.main_params['aggr_clustering_axis'],
                                  frequency_unit='Hz',
                                  time_unit='sec',
                                  trace_dim_names=trace_dim_names)

        if isinstance(self.cluster_catalogs, pd.DataFrame):
            merge_cols = self.cluster_catalogs.index.names + ['Cluster_ID']
            self.cluster_catalogs = pd.merge(self.cluster_catalogs, catalog, how='outer',
                                             on=merge_cols, suffixes=("_old", "_new"))
        else:
            self.cluster_catalogs = catalog

    @staticmethod
    def filter_cluster_catalogs(catalog, cluster_catalogs_filter, cluster_ID=None):
        if cluster_catalogs_filter is not None:
            keep_rows = cluster_catalogs_filter(catalog)
            if cluster_ID is not None:
                update_cluster_ID_by_catalog(cluster_ID, catalog, keep_rows)
            catalog = catalog[keep_rows]

        return catalog

    def filter_and_update_result(self, cluster_catalogs_filter):
        self.cluster_catalogs = self.filter_cluster_catalogs(self.cluster_catalogs,
                                                             cluster_catalogs_filter,
                                                             self.spectrogram_cluster_ID)

        # this can be followed by inverse transform or by extraction of the detection intervals

    @property
    def _concat_attr_list(self):
        return ["signal",
                "coefficients",
                "spectrogram",
                "noise_std",
                "spectrogram_SNR",
                "spectrogram_trim_mask",
                "spectrogram_trim_mask_aggr",
                "spectrogram_event_ID",
                "spectrogram_cluster_ID",
                "signal_denoised",
                "time_frames"]

    @property
    def _update_funcs_before_concat(self):
        return [self._update_cluster_id, self._update_time_frames]

    def _update_cluster_id(self, other):
        attrs = ["spectrogram_event_ID", "spectrogram_cluster_ID"]
        for attr in attrs:
            self_attr = getattr(self, attr, None)
            other_attr = getattr(other, attr, None)
            if (self_attr is not None) and (other_attr is not None):
                shape = other_attr.shape[:-2]
                for ind in np.ndindex(shape):
                    other_attr[ind][other_attr[ind] > 0] += self_attr[ind].max()
                setattr(other, attr, other_attr)  # other's IDs will be updated

    def _update_time_frames(self, other):
        attr = "time_frames"
        self_attr = getattr(self, attr, None)
        other_attr = getattr(self, attr, None)
        if (self_attr is not None) and (other_attr is not None):
            other_attr += self.get_t0_for_append()
            setattr(other, attr, other_attr)

    def get_t0_for_append(self):
        return (self.tf_time_npts + 1) * self.tf_dt_sec

    def append(self, other):

        # update items before concatenation
        for update_func in self._update_funcs_before_concat:
            update_func(other=other)

        # concatenation
        for name in self._concat_attr_list:
            self._concat(other=other, attr=name, axis=-1)

        t0 = self.get_t0_for_append()  # must be before updating `time_npts`
        self.cluster_catalogs = concatenate_cluster_catalogs(self.cluster_catalogs, other.cluster_catalogs, t0)

        self.time_npts += other.time_npts
        self.tf_time_npts += other.tf_time_npts

        self.history.merge(other.history)

        assert self.minSNR == other.minSNR
        assert self.dt_sec == other.dt_sec
        assert self.tf_dt_sec == other.tf_dt_sec
        assert np.all(self.frequencies == other.frequencies)
        assert np.all(self.noise_threshold_conversion == other.noise_threshold_conversion)

    def _concat(self, other, attr, axis):
        self_attr = getattr(self, attr, None)
        other_attr = getattr(other, attr, None)
        if (self_attr is not None) and (other_attr is not None):

            if (other_attr.dtype.name == 'object') and (self_attr.dtype.name == 'object'):
                axis_obj = axis - 1  # NOTE: assume that the last axis is for: [invt1, intv2] or [onset, likelihood]
                concatenated = np.empty_like(other_attr, dtype=object)
                for ind, i_other in np.ndenumerate(other_attr):
                    concatenated[ind] = np.concatenate((self_attr[ind], i_other), axis=axis_obj)
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

    def save(self, filepath, compress=False, stats=None):
        if self.stats is None:
            self.stats = stats
        else:
            self.stats.update(stats)
        if compress:
            print("Compress is legacy parameter, and is not used")

        save_pickle(self, filepath)

        # Legacy code for MAT format, I change it to pickle as MAT can't handle "None"s
        # mdict = self.filter_convert_attributes_to_dict()
        # savemat(filepath, mdict, do_compression=compress)
        # del mdict

    @classmethod
    def load(cls, filepath):
        splits = str(filepath).split('.')

        if (len(splits) > 1) and (splits[-1].casefold() == 'mat'):
            # 1. To support legacy MAT format
            mdict = loadmat(filepath, simplify_cells=True)
            result_object = cls(**cls.convert_dict_to_attributes(mdict))
        else:
            # 2. Load Pickle file
            result_object = load_pickle(filepath)
        return result_object

    # Legacy methods for saving to/from '.mat'
    def filter_convert_attributes_to_dict(self):
        # Remove `None`
        mdict = {name: attr for name, attr in self.model_dump().items() if (attr is not None)}
        catalog = mdict.get('cluster_catalogs', None)

        # Convert DataFrame to dict for `mat` format
        if catalog is not None:
            mdict['cluster_catalogs'] = catalog.to_dict(orient='tight')
        return mdict

    @staticmethod
    def convert_dict_to_attributes(mdict):
        # Convert dict to DataFrame
        if (catalog := mdict.get('cluster_catalogs', None)) is not None:
            mdict['cluster_catalogs'] = pd.DataFrame.from_dict(mat_structure_to_tight_dataframe_dict(catalog),
                                                               orient='tight')

        return mdict
