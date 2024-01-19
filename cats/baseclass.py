"""
    Implements baseclasses for Cluster Analysis of Trimmed Spectrograms (CATS)

        CATSBase : baseclass to perform CATS
        CATSResult : baseclass for keeping the results of CATS
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import holoviews as hv
from typing import Any, Union
from pydantic import BaseModel, Field, Extra

from .core.timefrequency import STFTOperator
from .core.clustering import Clustering, ClusterCatalogs, concatenate_arrays_of_cluster_catalogs
from .core.date import BEDATE_trimming, group_frequency, bandpass_frequency_groups
from .core.env_variables import get_min_bedate_block_size, get_max_memory_available_for_cats
from .core.utils import get_interval_division, format_index_by_dimensions, cast_to_bool_dict
from .core.utils import format_interval_by_limits, give_index_slice_by_limits
from .core.utils import give_nonzero_limits, mat_structure_to_tight_dataframe_dict


# ------------------------ BASE CLASSES ------------------------ #

class CATSBase(BaseModel, extra=Extra.allow):
    """
        Base class for CATS based on STFT. Implements unpacking of the main parameters.
        Implements 4 main steps:
            1) STFT
            2) Noise estimation via B-E-DATE
            3) Trimming spectrogram based minSNR
            4) Clustering trimmed spectrogram
        """
    # main STFT params
    dt_sec: float
    stft_window_type: str = 'hann'
    stft_window_sec: float = 0.5
    stft_overlap: float = Field(0.75, ge=0.0, lt=1.0)
    # main B-E-DATE params
    minSNR: float = 5.5
    stationary_frame_sec: float = None
    # main Clustering params
    cluster_size_t_sec: float = 0.2
    cluster_size_f_Hz: float = 15.0
    cluster_distance_t_sec: float = None
    cluster_distance_f_Hz: float = None

    # Extra STFT params
    freq_bandpass_Hz: Union[tuple[float, float], Any] = None
    stft_backend: str = 'ssqueezepy'
    stft_kwargs: dict = {}
    stft_nfft: int = -1

    # Extra B-E-DATE params
    bedate_freq_grouping_Hz: float = None
    bedate_log_freq_grouping: float = None
    date_Q: float = 0.95
    date_detection_mode: bool = True

    # Extra clustering params
    cluster_minSNR: float = 0.0
    cluster_fullness: float = Field(0, ge=0.0, le=1.0)
    cluster_size_f_logHz: float = None
    cluster_distance_f_logHz: float = None
    cluster_catalogs: bool = True
    clustering_multitrace: bool = False
    cluster_size_trace: int = Field(1, ge=1)
    cluster_distance_trace: int = Field(1, ge=1)

    # Misc
    reference_datetime: str = None  # datetime.datetime.isoformat
    name: str = "CATS"

    def __init__(self, **kwargs):
        """
            Arguments:
                dt_sec : float : sampling time in seconds.

                stft_window_type: str : type of weighting window in STFT (e.g. `hann`, `hamming`, 'ones). Default 'hann'

                stft_window_sec : float : length of weighting window in seconds.

                stft_overlap : float [0, 1) : overlapping of STFT windows (e.g. `0.5` means 50% overlap).

                stft_nfft : int : zero-padding for each individual STFT window, recommended a power of 2 (e.g. 256).

                minSNR : float : minimum Signal-to-Noise Ratio (SNR) for B-E-DATE algorithm. Recommended `4.0`.

                stationary_frame_sec : float : length of time frame in seconds wherein noise is stationary. \
Length will be adjusted to have at least 256 elements in one frame. Default, 0.0 which leads to minimum possible

                bedate_freq_grouping_Hz : float : frequency grouping width in Hz. Default min possible is set

                bedate_log_freq_grouping : float : frequency grouping width in log10 Hz. Default None, not applied. \
If given, then `bedate_freq_grouping_Hz` is ignored.

                cluster_size_t_sec : float : minimum cluster size in time, in seconds. \
Can be estimated as length of the strongest phases.

                cluster_size_f_Hz : float : minimum cluster size in frequency, in hertz, i.e. minimum frequency width.

                cluster_distance_t_sec : float : neighborhood distance in time for clustering, in seconds. \
Minimum separation time between two different events.

                cluster_distance_f_Hz : float : neighborhood distance in frequency for clustering, in hertz. \
Minimum separation frequency width between two different events.

                cluster_minSNR : float [xi(minSNR), inf] : minimum average cluster energy, if None, `minSNR` is used.

                cluster_fullness : float (0, 1] : minimum cluster fullness from its minimum size defined by `size` and \
`distance` params

                cluster_size_f_logHz : float : cluster frequency width in [log10 Hz] scale. If not None, then it will be \
applied instead of `cluster_size_f_Hz`. Default None.

                cluster_distance_f_logHz : float : clustering distance in [log10 Hz] scale. If not None, then it will be \
applied instead of `cluster_distance_f_Hz`. Default None.

                cluster_catalogs : bool : whether to calculate catalogs of clusters statistics. Default False

                freq_bandpass_Hz : tuple[float, float] : bandpass frequency range, in hertz, i.e. everything out of \
the range is zero (e.g. (f_min, f_max)).

                clustering_multitrace : bool : whether to use multitrace clustering (Location x Frequency x Time). \
Increase accuracy for multiple arrays of receivers on regular grid. Performs cross-station association.

                cluster_size_trace : int : minimum cluster size for traces, minimum number of traces in one cluster.

                cluster_distance_trace : int : neighborhood distance across multiple traces for clustering.

                date_Q : float : probability that sorted elements after certain `Nmin` have amplitude higher \
than standard deviation. Used in Bienaymé–Chebyshev inequality to get `Nmin` to minimize cost of DATE. Default `0.95`

                date_detection_mode : bool : `True` means NOT to use original implementation of DATE algorithm. \
Original implementation assumes that if no outliers are found then standard deviation is estimated from `Nmin` \
to not overestimate the noise. `True` implies that noise can be overestimated. It is beneficial if no outliers \
are found, then no outliers will be present in the trimmed spectrogram.

                stft_backend : str : backend for STFT operator ['scipy', 'ssqueezepy', 'ssqueezepy_gpu']. \
The fastest CPU version is 'ssqueezepy', which is default.

                stft_kwargs : dict : additional keyword arguments for STFT operator (see `cats.STFTOperator`).

                reference_datetime : str : reference datetime is `datetime.datetime.isoformat`

                name : str : name of the object
        """
        super().__init__(**kwargs)
        self._set_params()

    def _set_params(self):
        # Setting STFT
        self.STFT = STFTOperator(window_specs=(self.stft_window_type, self.stft_window_sec), overlap=self.stft_overlap,
                                 dt_sec=self.dt_sec, nfft=self.stft_nfft, backend=self.stft_backend, **self.stft_kwargs)
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

        # DATE params
        self.stationary_frame_sec = v if (v := self.stationary_frame_sec) is not None else 0.0
        self.stationary_frame_len = max(round(self.stationary_frame_sec / self.stft_hop_sec),
                                        get_min_bedate_block_size())
        self.stationary_frame_sec = self.stationary_frame_len * self.stft_hop_sec

        # Clustering params
        self.freq_bandpass_Hz = format_interval_by_limits(self.freq_bandpass_Hz,
                                                          (self.stft_frequency[0], self.stft_frequency[-1]))
        self.freq_bandpass_Hz = (min(self.freq_bandpass_Hz), max(self.freq_bandpass_Hz))
        self.freq_bandwidth_Hz = self.freq_bandpass_Hz[1] - self.freq_bandpass_Hz[0]
        assert (fbw := self.freq_bandwidth_Hz) > (csf := self.cluster_size_f_Hz), \
            f"Frequency bandpass width `{fbw}` must be bigger than min frequency cluster size `{csf}`"
        self.freq_bandpass_slice = give_index_slice_by_limits(self.freq_bandpass_Hz, self.STFT.df)
        self.cluster_size_t_len = max(round(self.cluster_size_t_sec / self.stft_hop_sec), 1)
        self.cluster_size_f_len = max(round(self.cluster_size_f_Hz / self.STFT.df), 1)
        self.cluster_size_trace_len = self.cluster_size_trace

        self.cluster_size_f_logHz = self.cluster_size_f_logHz or 0.0
        self.cluster_distance_f_logHz = self.cluster_distance_f_logHz or 0.0

        self.cluster_distance_t_sec = v if (v := self.cluster_distance_t_sec) is not None else self.stft_hop_sec
        self.cluster_distance_f_Hz = v if (v := self.cluster_distance_f_Hz) is not None else self.stft_df
        self.cluster_distance_t_len = max(round(self.cluster_distance_t_sec / self.stft_hop_sec), 1)
        self.cluster_distance_f_len = max(round(self.cluster_distance_f_Hz / self.stft_df), 1)
        self.cluster_distance_trace_len = self.cluster_distance_trace
        # self.cluster_minSNR = self.cluster_minSNR if (self.cluster_minSNR is not None) else self.minSNR

        self.time_edge = int(self.stft_window_len // 2 / self.stft_hop_len)

        self.min_duration_len = max(self.cluster_size_t_len, 1)  # `-1` to include bounds (0, 1, 0)
        self.min_separation_len = max(self.cluster_distance_t_len + 1, 2)  # `+1` to exclude bounds (1, 0, 1)
        self.min_duration_sec = self.min_duration_len * self.stft_hop_sec
        self.min_separation_sec = self.min_separation_len * self.stft_hop_sec

        self.bedate_freq_grouping_Hz = self.bedate_freq_grouping_Hz or self.stft_df
        self.bedate_log_freq_grouping = self.bedate_log_freq_grouping or 0.0

        self.bedate_frequency_grouping = (self.bedate_log_freq_grouping or self.bedate_freq_grouping_Hz,
                                          self.bedate_log_freq_grouping > 0.0)
        self.frequency_groups_index, self.frequency_groups = group_frequency(self.stft_frequency,
                                                                             *self.bedate_frequency_grouping)
        self.bandpassed_frequency_groups_slice = bandpass_frequency_groups(self.frequency_groups_index,
                                                                           self.freq_bandpass_slice)

    def export_main_params(self):
        return {kw: val for kw in type(self).__fields__.keys() if (val := getattr(self, kw, None)) is not None}

    @classmethod
    def from_result(cls, CATSResult):
        return cls(**CATSResult.main_params)

    def reset_params(self, **params):
        """
            Updates the instance with changed parameters.
        """
        kwargs = self.export_main_params()
        kwargs.update(params)
        self.__init__(**kwargs)

    def apply_STFT(self, result_container, /):
        result_container['coefficients'] = self.STFT * result_container['signal']
        result_container['spectrogram'] = np.abs(result_container['coefficients'])

    def apply_BEDATE(self, result_container, /):
        result_container['time_frames'] = get_interval_division(N=result_container['spectrogram'].shape[-1],
                                                                L=self.stationary_frame_len)

        T, STD, THR = BEDATE_trimming(result_container['spectrogram'], self.frequency_groups_index,
                                      self.bandpassed_frequency_groups_slice, self.freq_bandpass_slice,
                                      result_container['time_frames'], self.minSNR, self.time_edge, self.date_Q,
                                      not self.date_detection_mode)

        result_container['spectrogram_SNR_trimmed'] = T
        result_container['noise_std'] = STD
        result_container['noise_threshold_conversion'] = THR

    def apply_Clustering(self, result_container, /):
        mc = self.clustering_multitrace
        q = (self.cluster_distance_trace_len,) * mc + (self.cluster_distance_f_len, self.cluster_distance_t_len)
        s = (self.cluster_size_trace_len,) * mc + (self.cluster_size_f_len, self.cluster_size_t_len)
        alpha = self.cluster_fullness
        log_freq_cluster = (self.cluster_size_f_logHz, self.cluster_distance_f_logHz)

        result_container['spectrogram_SNR_clustered'] = np.zeros_like(result_container['spectrogram_SNR_trimmed'])
        result_container['spectrogram_cluster_ID'] = np.zeros(result_container['spectrogram_SNR_trimmed'].shape,
                                                              dtype=np.uint32)

        result_container['spectrogram_SNR_clustered'], result_container['spectrogram_cluster_ID'] = \
            Clustering(result_container['spectrogram_SNR_trimmed'], q=q, s=s,
                       minSNR=self.cluster_minSNR, alpha=alpha, log_freq_cluster=log_freq_cluster)

        if self.cluster_catalogs:
            result_container['cluster_catalogs'] = ClusterCatalogs(result_container['spectrogram_SNR_clustered'],
                                                                   result_container['spectrogram_cluster_ID'],
                                                                   self.stft_frequency, self.stft_hop_sec)
        else:
            result_container['cluster_catalogs'] = None

    def apply_func(self, func_name, result_container, status_keeper, process_name=None, **kwargs):
        process_name = process_name or func_name
        with status_keeper(current_process=process_name):
            getattr(self, func_name)(result_container, **kwargs)

    @staticmethod
    def _info_keys():
        return ["signal",
                "coefficients",
                "spectrogram",
                "noise_std",
                "noise_threshold_conversion",
                "spectrogram_SNR_trimmed",
                "spectrogram_SNR_clustered",
                "spectrogram_cluster_ID"]

    def _parse_info_dict(self, full_info):
        return cast_to_bool_dict(full_info, self._info_keys())

    def _memory_usage(self, x):

        stft_time_len = len(self.STFT.forward_time_axis(x.shape[-1]))
        stft_shape = x.shape[:-1] + (len(self.stft_frequency), stft_time_len)
        stft_size = np.prod(stft_shape)

        bedate_shape = x.shape[:-1] + (len(self.frequency_groups),
                                       len(get_interval_division(N=stft_time_len,
                                                                 L=self.stationary_frame_len)))
        bedate_size = np.prod(bedate_shape)

        x_bytes = x.nbytes
        precision_order = x_bytes / x.size

        memory_usage_bytes = {
            "stft_frequency":              8. * stft_shape[-2],                 # float64 always
            "time_frames":                 8. * bedate_shape[-1],               # float64 always
            "signal":                      1. * x_bytes,                        # float / int
            "coefficients":                2. * precision_order * stft_size,    # complex
            "spectrogram":                 1. * precision_order * stft_size,    # float
            "noise_threshold_conversion":  8. * bedate_shape[-2],               # float
            "noise_std":                   1. * precision_order * bedate_size,  # float
            "spectrogram_SNR_trimmed":     1. * precision_order * stft_size,    # float
            "spectrogram_SNR_clustered":   1. * precision_order * stft_size,    # float
            "spectrogram_cluster_ID":      4. * stft_size,                      # uint32 always
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


class CATSResult(BaseModel):
    signal: Any = None
    coefficients: Any = None
    spectrogram: Any = None
    noise_threshold_conversion: Any = None
    noise_std: Any = None
    spectrogram_SNR_trimmed: Any = None
    spectrogram_SNR_clustered: Any = None
    spectrogram_cluster_ID: Any = None
    cluster_catalogs: Any = None
    dt_sec: float = None
    stft_dt_sec: float = None
    stft_t0_sec: float = None
    npts: int = None
    stft_npts: int = None
    stft_frequency: Any = None
    time_frames: Any = None
    frequency_groups: Any = None
    minSNR: float = None
    threshold: float = None
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
        return CATSResult.base_time_func(self.npts, self.dt_sec, 0, time_interval_sec)

    def stft_time(self, time_interval_sec=None):
        return CATSResult.base_time_func(self.stft_npts, self.stft_dt_sec, 0, time_interval_sec)

    def plot(self, ind=None, time_interval_sec=None):
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        a_dim = hv.Dimension('Amplitude')

        ind = format_index_by_dimensions(ind=ind, shape=self.signal.shape[:-1], slice_dims=0, default_ind=0)

        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.npts - 1) * self.dt_sec))

        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        i_stft = give_index_slice_by_limits(time_interval_sec, self.stft_dt_sec)
        inds_time = ind + (i_time,)
        inds_stft = ind + (slice(None), i_stft)

        PSD = self.spectrogram[inds_stft]
        SNR = self.spectrogram_SNR_trimmed[inds_stft]
        C = self.spectrogram_SNR_clustered[inds_stft]

        PSD_clims = give_nonzero_limits(PSD, initials=(1e-1, 1e1))
        SNR_clims = give_nonzero_limits(SNR, initials=(1e-1, 1e1))

        time = self.time(time_interval_sec)
        stft_time = self.stft_time(time_interval_sec)

        fig0 = hv.Curve((time, self.signal[inds_time]), kdims=[t_dim], vdims=a_dim,
                        label='0. Input data: $x(t)$').opts(xlabel='', linewidth=0.5)
        fig1 = hv.Image((stft_time, self.stft_frequency, PSD), kdims=[t_dim, f_dim],
                        label='1. Amplitude spectrogram: $|X(t,f)|$').opts(clim=PSD_clims, clabel='Amplitude')
        fig2 = hv.Image((stft_time, self.stft_frequency, SNR), kdims=[t_dim, f_dim],
                        label='2. Trimmed SNR spectrogram: $T(t,f)$').opts(clim=SNR_clims)
        fig3 = hv.Image((stft_time, self.stft_frequency, C), kdims=[t_dim, f_dim],
                        label='3. Clustered SNR spectrogram: $\mathcal{L}(t,f)$').opts(clim=SNR_clims)

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250
        cmap = 'viridis'
        xlim = time_interval_sec
        ylim = (max(1e-1, self.stft_frequency[1] / 2), None)
        spectr_opts = hv.opts.Image(cmap=cmap, colorbar=True,  logy=True, logz=True, xlim=xlim, ylim=ylim,
                                    xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize)
        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=xlim)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        inds_slices = (ind, i_time, i_stft)
        figs = (fig0 + fig1 + fig2 + fig3)
        return figs, (layout_opts, spectr_opts, curve_opts), inds_slices, time_interval_sec

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

        t0 = self.time_frames[-1, -1] + self.stft_dt_sec

        self._concat(other, "time_frames", 0, t0)
        self.cluster_catalogs = concatenate_arrays_of_cluster_catalogs(self.cluster_catalogs,
                                                                       other.cluster_catalogs, t0)

        self.npts += other.npts
        self.stft_npts += other.stft_npts

        self.history.merge(other.history)

        if self.main_params['reference_datetime'] is not None:
            self.main_params['reference_datetime'] = min(self.main_params['reference_datetime'],
                                                         other.main_params['reference_datetime'])

        assert self.minSNR == other.minSNR
        assert self.dt_sec == other.dt_sec
        assert self.stft_dt_sec == other.stft_dt_sec
        assert np.all(self.stft_frequency == other.stft_frequency)
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
