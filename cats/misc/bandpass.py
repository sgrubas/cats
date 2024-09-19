import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.io import loadmat, savemat
from pydantic import BaseModel, Field
from cats.core.utils import (StatusKeeper, format_index_by_dimensions, format_interval_by_limits,
                             give_index_slice_by_limits)
from cats.io.utils import save_pickle, load_pickle
from typing import Any
from cats.baseclass import CATSResult
import holoviews as hv


class BandpassDenoiser(BaseModel, extra="allow"):
    dt_sec: float = Field(1.0, ge=0.0)
    f1_Hz: float = Field(1.0, ge=0.0)
    f2_Hz: float = Field(25.0, ge=0.0)
    order: int = 5

    name: str = 'Bandpass'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_params()

    def _set_params(self):
        self.fs = 1.0 / self.dt_sec
        self.filter_sos = butter(self.order, [self.f1_Hz, self.f2_Hz],
                                 btype='bandpass', output='sos', fs=self.fs)

    @property
    def main_params(self):
        return {kw: val for kw in self.model_fields.keys()
                if (val := getattr(self, kw, None)) is not None}

    def reset_params(self, **params):
        """
            Updates the instance with changed parameters.
        """
        kwargs = self.main_params
        kwargs.update(params)
        self.__init__(**kwargs)

    def _denoise(self, x, verbose=False):
        result = {"signal": x}
        history = StatusKeeper(verbose=verbose)
        with history(current_process='Bandpass filtering:'):
            result['signal_denoised'] = sosfiltfilt(self.filter_sos, x, axis=-1)

        history.print_total_time()

        return BandpassDenoisingResult(dt_sec=self.dt_sec,
                                       npts=x.shape[-1],
                                       history=history,
                                       main_params=self.main_params,
                                       **result)

    def denoise(self, x, verbose=False):
        return self._denoise(x, verbose=verbose)

    def __mul__(self, x):
        return self._denoise(x, verbose=False)

    def __pow__(self, x):
        return self._denoise(x, verbose=True)

    def __matmul__(self, x):
        return self.__pow__(x)

    def save(self, filename):
        save_pickle(self.main_params, filename)

    @classmethod
    def load(cls, filename):
        loaded = load_pickle(filename)
        if isinstance(loaded, cls):
            loaded = loaded.main_params
        return cls(**loaded)


class BandpassDenoisingResult(BaseModel):
    dt_sec: float = None
    npts: int = None
    history: Any = None
    main_params: dict = None
    signal: Any = None
    signal_denoised: Any = None
    header_info: dict = None

    def time(self, time_interval_sec=None):
        return CATSResult.base_time_func(self.npts, self.dt_sec, 0, time_interval_sec)

    def plot(self, ind=None, time_interval_sec=None):
        t_dim = hv.Dimension('Time', unit='s')
        a_dim = hv.Dimension('Amplitude')

        ind = format_index_by_dimensions(ind=ind, shape=self.signal.shape[:-1], slice_dims=0, default_ind=0)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.npts - 1) * self.dt_sec))
        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        inds_time = ind + (i_time,)

        time = self.time(time_interval_sec)

        fig0 = hv.Curve((time, self.signal[inds_time]), kdims=[t_dim], vdims=a_dim,
                        label='0. Input data: $x(t)$').opts(xlabel='', linewidth=0.5)
        fig1 = hv.Curve((time, self.signal_denoised[inds_time]), kdims=[t_dim], vdims=a_dim,
                        label='1. Denoised data: $\\tilde{s}(t)$').opts(xlabel='', linewidth=0.5)

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250
        xlim = time_interval_sec
        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=xlim)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')

        figs = (fig0 + fig1)
        opts = (layout_opts, curve_opts)
        return figs.opts(*opts).cols(1)

    def append(self, other):
        for name in ["signal", "signal_denoised"]:
            self._concat(other, name, -1)
        self.npts += other.time_npts
        self.history.merge(other.history)
        assert self.dt_sec == other.dt_sec

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
        return mdict

    def save(self, filepath, compress=False, header_info=None):
        self.header_info = header_info
        mdict = self.filter_convert_attributes_to_dict()
        savemat(filepath, mdict, do_compression=compress)
        del mdict

    @classmethod
    def load(cls, filepath):
        mdict = loadmat(filepath, simplify_cells=True)
        return cls(**mdict)
