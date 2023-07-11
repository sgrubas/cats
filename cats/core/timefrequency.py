"""
    Operators for computing Time-Frequency representations, which are implemented for convenient usage of Forward and
    Inverse transforms from one object. Overrides operators `*` for forward transform, and `/` for inverse.

        STFT operator : Short-Time Fourier Transform
        CWT operator : Short-Time Fourier Transform
"""

import numpy as np
from scipy import signal
import ssqueezepy as ssq
import os
from .utils import ReshapeArraysDecorator, update_object_params
from pydantic import BaseModel, Field, Extra
from typing import Union, Literal, Tuple, List


############################################################
#                      STFT transform                      #
############################################################


class STFTOperator(BaseModel, extra=Extra.allow):
    """
        STFT operator
    """
    window_specs: Union[int, float, Tuple[str, float]]
    overlap: float = Field(0.75, ge=0.0, lt=1.0)
    dt_sec: float = 1.0
    backend: Literal["scipy", "ssqueezepy", "ssqueezepy_gpu"] = "ssqueezepy"
    nfft: int = None
    padtype: Literal["reflect", "symmetric", "replicate", "wrap", "zero"] = 'reflect'
    scipy_padded: bool = False
    scipy_detrend: bool = False
    ssqueezepy_modulated: bool = False
    ssqueezepy_win_exp: int = 1

    def __init__(self, **kwargs):
        """
            Operator for Short-Term Fourier Transform (STFT) (forward & inverse)

            Arguments:
                window_specs: int / tuple(str, float) : weighting window for STFT
                overlap: float : [0, 1) : overlapping of sliding windows
                dt : float : sampling time (in seconds)
                backend : str : backend for computations (`scipy` or `ssqueezepy` or `ssqueezepy_gpu`)
                nfft : int : how many frequency bins for each individual window.
                             If `nfft > len(window)`, then zero-padding is performed.
                             if `nfft < 0`, then `nfft` is rounded to the closest `2**ceil(log2(len(window)))`
                             Default `nfft = len(window)`
                padtype : str : type of padding signal at both ends ('reflect', 'symmetric', 'replicate', 'wrap', 'zero').
                                Default 'reflect'

                scipy_padded : bool : applied for Forward STFT. Default False.
                             Used when `backend = 'scipy'` it can be (see `scipy.signal.stft`)
                scipy_detrend : bool : applied for Forward STFT. Default False.
                             Used when `backend = 'scipy'` it can be (see `scipy.signal.stft`)
                ssqueezepy_modulated : bool : applied for Forward & Inverse STFT. Default False.
                             Used when `backend = 'ssqueezepy'` it can be (see `ssqueezepy.stft` & `ssqueezepy.istft`)
                ssqueezepy_win_exp : int : applied for Inverse STFT. Default `1`.
                             Used when `backend = 'ssqueezepy'` it can be (see `ssqueezepy.stft` & `ssqueezepy.istft`)

        """
        super().__init__(**kwargs)
        self._set_params()

    def _set_params(self):
        self.window = self.define_stft_window(self.window_specs)
        self.nperseg = len(self.window)
        self.noverlap = int(self.overlap * self.nperseg)
        self.hop = (self.nperseg - self.noverlap)
        self.fs = 1 / self.dt_sec

        self.nfft = self.nperseg if (self.nfft is None) else self.nfft
        if self.nfft < 0:
            self.nfft = 2**int(np.ceil(np.log2(self.nperseg)))
        else:
            self.nfft = max(self.nfft, self.nperseg)
        self.nfft = self.nfft + self.nfft % 2

        self.padedge = self.nperseg // 2
        self.f = np.linspace(0, 0.5 * self.fs, self.nfft // 2 + 1)
        self.df = self.f[1] - self.f[0]

        # STFT Transform settings
        self._set_STFT_kwargs()

    def define_stft_window(self, window):
        if isinstance(window, tuple):
            wtype, length = window
        elif isinstance(window, (int, float)):
            wtype, length = 'ones', window
        else:
            raise ValueError(f"Unknown type of window {type(window)}")

        nperseg = round(length / self.dt_sec) + 1

        if ('one' in wtype) or (len(wtype) == 0) or (wtype is None):
            window = np.ones(nperseg)
        else:
            window = signal.get_window(wtype, nperseg)
        return window

    def _set_STFT_kwargs(self):
        kwargs = {"padded": self.scipy_padded, "detrend": self.scipy_detrend,
                  "modulated": self.ssqueezepy_modulated, "win_exp": self.ssqueezepy_win_exp}
        forw_kw, inv_kw = {}, {}

        # Common parameters
        forw_kw['window'] = inv_kw['window'] = self.window

        if self.backend == 'scipy':
            forw_kw['fs'] = inv_kw['fs'] = self.fs
            forw_kw['nfft'] = inv_kw['nfft'] = self.nfft
            forw_kw['nperseg'] = inv_kw['nperseg'] = self.nperseg
            forw_kw['noverlap'] = inv_kw['noverlap'] = self.noverlap
            forw_kw['return_onesided'] = inv_kw['input_onesided'] = True
            forw_kw['padded'] = kwargs.get('padded', False)
            forw_kw['detrend'] = kwargs.get('detrend', False)
            forw_kw['boundary'], inv_kw['boundary'] = None, True
            forw_kw['axis'], inv_kw['time_axis'], inv_kw['freq_axis'] = -1, -1, -2

        elif 'ssq' in self.backend:
            forw_kw['n_fft'] = inv_kw['n_fft'] = self.nfft
            forw_kw['win_len'] = inv_kw['win_len'] = self.nperseg
            forw_kw['hop_len'] = inv_kw['hop_len'] = self.nperseg - self.noverlap
            forw_kw['modulated'] = inv_kw['modulated'] = kwargs.get('modulated', False)
            inv_kw['win_exp'] = kwargs.get('win_exp', 1)
            forw_kw['padtype'], forw_kw['derivative'] = self.padtype,   False
            forw_kw['t'], forw_kw['fs'] = None, self.fs

        self.forw_kw = forw_kw
        self.inv_kw = inv_kw

    def reset_params(self, **params):
        update_object_params(self, **params)
        
    def padsignal(self, X):
        """
            Performs padding of `X` to the last axis with `window_length // 2` to the both ends

            X : numpy array : shape (..., N), `N` time samples
        """
        compatible_padmodes = {"replicate ": "edge", "zero": "constant"}
        padmode = compatible_padmodes.get(self.padtype, self.padtype)
        N = X.shape[-1]
        if self.backend == 'scipy':
            n1 = self.padedge
            n2 = n1 + self.hop - (N - self.nperseg % 2) % self.hop
        else:
            n1 = 0
            n2 = self.hop - (N - 1) % self.hop
        return np.pad(X, [(0, 0), (n1, n2)], mode=padmode)

    def _forward_backend(self, X):
        """
            Performs Forward STFT

            X : numpy array : shape (M, N), where `M` signals, `N` time samples
        """
        Y = self.padsignal(X)
        if self.backend == 'scipy':
            f, t, C = signal.stft(Y, **self.forw_kw)

        else:
            gpu_status = 'gpu' in self.backend
            os.environ['SSQ_GPU'] = '1' if gpu_status else '0'
        
            C = ssq.stft(Y, **self.forw_kw)
            if not isinstance(C, np.ndarray):
                C = C.cpu().numpy()

        return C

    def _inverse_backend(self, C):
        """
            Performs Inverse STFT

            C : numpy array : shape (M, Nf, Nt), where `M` spectrograms, `Nf` frequencies, `Nt` time frames
        """
        N = self.inverse_time_samples(C.shape[-1])
        if self.backend == 'scipy':
            t, X = signal.istft(C, **self.inv_kw)
            if N is not None: 
                X = X[:N]
        else:
            gpu_status = 'gpu' in self.backend
            os.environ['SSQ_GPU'] = '1' if gpu_status else '0'

            X = []
            for ci in C:
                xi = ssq.istft(ci, N=N, **self.inv_kw)
                if not isinstance(C, np.ndarray):
                    xi = xi.cpu().numpy()
                X.append(xi)
            X = np.stack(X, axis=0)

        return X

    @ReshapeArraysDecorator(dim=2, input_num=1, methodfunc=True, output_num=1, first_shape=True)
    def forward(self, X, /):
        """
            Performs Forward STFT

            X : numpy array : shape (..., N), where `N` time samples. 
                              `X` can have any dimesional structure for any number of signals, 
                              but the last axis `N` must be `time` axis
        """
        return self._forward_backend(X)

    @ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=True, output_num=1, first_shape=True)
    def inverse(self, C, /):
        """
            Performs Inverse STFT

            C : numpy array : shape (..., Nf, Nt), where `Nf` frequencies, `Nt` time frames. 
                              `C` can have any dimensional structure for any number of spectrograms,
                              but the last two axes `Nf` and `Nt` must be `frequency` and `time` axes respectively.
        """
        return self._inverse_backend(C)

    def forward_time_axis(self, N):
        bounds = self.padedge * 2 if (self.padtype is not None) else 0
        tail = (N + bounds - self.nperseg) % self.hop
        tail = self.hop - tail if tail > 0 else tail
        n_hops = (N + bounds + tail - self.nperseg) // self.hop + 1
        t = self.dt_sec * (np.arange(n_hops) * self.hop + (bounds // 2) * (bounds < 1))

        # from `squeezepy` where `x` is padded and extended
        # time = np.arange(nperseg / 2, x.shape[-1] - nperseg / 2 + 1,
        #                  nperseg - noverlap) / float(fs)
        # if boundary is not None:
        #     time -= (nperseg/2) / fs
        return t
    
    def inverse_time_samples(self, Nt):
        bounds = self.padedge * 2 if (self.padtype is not None) else 0
        N = (Nt - 1) * self.hop - bounds + self.nperseg + self.hop
        return N
    
    def __mul__(self, X):
        return self.forward(X)
    
    def __truediv__(self, C):
        return self.inverse(C)


class CWTOperator(BaseModel, extra=Extra.allow):
    """
        CWT operator
    """
    dt_sec: float
    wavelet: Union[str, Tuple[str, dict]] = ('morlet', {'mu': 5})
    scales: Union[Literal['log', 'log-piecewise', 'linear', 'log:maximal'], Tuple[float], List[float]] = 'log-piecewise'
    nv: int = 32  # >= 16
    l1_norm: bool = True
    derivative: bool = False
    padtype: Literal['reflect', 'symmetric', 'replicate', 'wrap', 'zero'] = 'reflect'
    rpadded: bool = False
    vectorized: bool = True
    astensor: bool = True
    cache_wavelet: bool = None
    order: Union[int, Tuple[int]] = 0
    average: bool = None
    nan_checks: bool = True,
    patience: Union[int, Tuple[int, int]] = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_params()

    def _set_params(self):
        self.fs = 1 / self.dt_sec
        # t:,
        pass

    @ReshapeArraysDecorator(dim=2, input_num=1, methodfunc=True, output_num=1, first_shape=True)
    def forward(self, X):
        # ssq.cwt()
        pass

    @ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=True, output_num=1, first_shape=True)
    def inverse(self, C):
        # ssq.icwt()
        pass

    def __mul__(self, X):
        return self.forward(X)

    def __truediv__(self, C):
        return self.inverse(C)