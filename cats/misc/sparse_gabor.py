import numpy as np

from scipy import signal
from scipy.sparse.linalg import lsqr as LSQR
from tqdm.auto import tqdm as std_tqdm
from tqdm.notebook import tqdm
import ssqueezepy as ssq
import os

from pylops import LinearOperator
from pylops.optimization.sparsity import FISTA

from pydantic import BaseModel, Field, Extra
from typing import Any
import holoviews as hv
from cats.core.utils import (format_index_by_dimensions, format_interval_by_limits, StatusKeeper,
                             give_index_slice_by_limits, give_nonzero_limits, complex_abs_square)
from cats.baseclass import CATSResult
from cats.io.utils import save_pickle, load_pickle


class SparseGaborDenoiser(BaseModel, extra="allow"):
    dt_sec: float
    shape: Any

    # STFT params
    stft_window_type: str = 'hann'
    stft_window_sec: float = 0.5
    stft_overlap: float = Field(0.75, ge=0.0, lt=1.0)
    stft_backend: str = 'ssq'
    stft_nfft: int = None
    freq_bandpass_Hz: tuple[float, float] = None
    stft_kwargs: dict = {}

    # FISTA params
    eps: float = 0.1
    eps_factor: float = 1.0
    niter: int = 10
    tol: float = 1e-5
    fista_kw: dict = {}
    background_weight: float = 0.0

    # misc
    name: str = "SparseGabor"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_params()

    def _set_params(self):
        self.window = define_stft_window((self.stft_window_type, self.stft_window_sec), self.dt_sec)
        self.operator = WSTFT_LOp(self.shape, window=self.window, overlap=self.stft_overlap,
                                  time_axis=-1, dt=self.dt_sec, target=self.stft_backend,
                                  nfft=self.stft_nfft, dtype=None, background_weight=self.background_weight,
                                  stft_kwargs=self.stft_kwargs)

        self.time_edge = int(len(self.window) // 2 / self.operator.hop)

    def reset_params(self, **params):
        kwargs = self.main_params
        kwargs.update(params)
        self.__init__(**kwargs)

    @property
    def main_params(self):
        # extract only params in __init__
        return {kw: val for kw in self.model_fields.keys()
                if (val := getattr(self, kw, None)) is not None}

    def denoise(self, x, verbose=False):
        history = StatusKeeper(verbose=verbose)

        with history(current_process='FISTA'):
            Df, Cmn, Cfista = self.operator.Sparse(x, niter=self.niter, eps=self.eps * self.eps_factor,
                                                   tol=self.tol, pbar=False, **self.fista_kw)

        history.print_total_time()

        transpose_coefs = lambda arr: np.transpose(arr, tuple(range(2, 2 + len(x.shape[:-1]))) + (0, 1))

        result = {'signal': x, 'signal_denoised': Df,
                  'coefficients': transpose_coefs(Cmn),
                  'coefficients_sparse': transpose_coefs(Cfista)}

        result['coefficients'][..., :self.time_edge] = 0.0 + 0.0j
        result['coefficients_sparse'][..., :self.time_edge] = 0.0 + 0.0j

        result['coefficients'][..., -self.time_edge - 1:] = 0.0 + 0.0j
        result['coefficients_sparse'][..., -self.time_edge - 1:] = 0.0 + 0.0j

        kwargs = {**result,
                  "dt_sec": self.dt_sec,
                  "tf_dt_sec": self.operator.hop * self.dt_sec,
                  "tf_t0_sec": self.operator.t[0],
                  "time_npts": x.shape[-1],
                  "tf_time_npts": self.operator.tshape,
                  "frequencies": self.operator.f,
                  "history": history,
                  "main_params": self.main_params}

        return SparseGaborDenoisingResult(**kwargs)

    def __mul__(self, x):
        return self.denoise(x, verbose=False)

    def __pow__(self, x):
        return self.denoise(x, verbose=True)

    def __matmul__(self, x):
        return self.denoise(x, verbose=True)

    def save(self, filename):
        save_pickle(self.main_params, filename)

    @classmethod
    def load(cls, filename):
        loaded = load_pickle(filename)
        if isinstance(loaded, cls):
            loaded = loaded.main_params
        return cls(**loaded)


class SparseGaborDenoisingResult(BaseModel, extra="allow"):
    signal: Any = None
    coefficients: Any = None
    coefficients_sparse: Any = None
    signal_denoised: Any = None

    dt_sec: float = None
    stft_dt_sec: float = None
    stft_t0_sec: float = None
    npts: int = None
    stft_npts: int = None
    stft_frequency: Any = None

    history: Any = None
    main_params: dict = None

    def time(self, time_interval_sec=None):
        return CATSResult.base_time_func(self.npts, self.dt_sec, 0, time_interval_sec)

    def stft_time(self, time_interval_sec=None):
        return CATSResult.base_time_func(self.stft_npts, self.stft_dt_sec, 0, time_interval_sec)

    def plot(self, ind=None, time_interval_sec=None):

        if ind is None:
            ind = (0,) * (self.signal.ndim - 1)
        t_dim = hv.Dimension('Time', unit='s')
        f_dim = hv.Dimension('Frequency', unit='Hz')
        A_dim = hv.Dimension('Amplitude')

        ind = format_index_by_dimensions(ind=ind, shape=self.signal.shape[:-1], slice_dims=0, default_ind=0)
        time_interval_sec = format_interval_by_limits(time_interval_sec, (0, (self.npts - 1) * self.dt_sec))

        i_time = give_index_slice_by_limits(time_interval_sec, self.dt_sec)
        i_stft = give_index_slice_by_limits(time_interval_sec, self.stft_dt_sec)

        inds_time = ind + (i_time,)
        inds_Stft = ind + (slice(None), i_stft)

        PSD = complex_abs_square(self.coefficients[inds_Stft])
        PSDC = complex_abs_square(self.coefficients_sparse[inds_Stft])

        PSD_clims = give_nonzero_limits(PSD, initials=(1e-1, 1e1))
        PSDC_clims = give_nonzero_limits(PSDC, initials=(1e-1, 1e1))

        time = self.time(time_interval_sec)
        stft_time = self.stft_time(time_interval_sec)

        fig0 = hv.Curve((time, self.signal[inds_time]), kdims=[t_dim], vdims=A_dim,
                        label='0. Input data: $x(t)$').opts(xlabel='', linewidth=0.5)
        fig1 = hv.Image((stft_time, self.stft_frequency, PSD), kdims=[t_dim, f_dim],
                        label='1. Power spectrogram: $|X(t,f)|^2$').opts(clim=PSD_clims, clabel='Power')
        fig2 = hv.Image((stft_time, self.stft_frequency, PSDC), kdims=[t_dim, f_dim],
                        label='2. Sparse Gabor power spectrogram: $|G(t,f)|^2$').opts(clim=PSDC_clims)
        fig3 = hv.Curve((time, self.signal_denoised[inds_time]),
                        kdims=[t_dim], vdims=A_dim,
                        label='3. Denoised data: $\\tilde{s}(t)$').opts(linewidth=0.5)

        fontsize = dict(labels=15, title=16, ticks=14)
        figsize = 250
        cmap = 'viridis'
        xlim = time_interval_sec
        ylim = (max(1e-1, self.stft_frequency[1]), None)
        spectr_opts = hv.opts.Image(cmap=cmap, colorbar=True,  logy=True, logz=True, xlim=xlim, ylim=ylim,
                                    xlabel='', clabel='', aspect=2, fig_size=figsize, fontsize=fontsize)
        curve_opts = hv.opts.Curve(aspect=5, fig_size=figsize, fontsize=fontsize, xlim=xlim, show_legend=False)
        layout_opts = hv.opts.Layout(fig_size=figsize, shared_axes=True, vspace=0.4,
                                     aspect_weight=0, sublabel_format='')
        figs = [fig0, fig1, fig2, fig3]
        fig = hv.Layout(figs).cols(1).opts(layout_opts, curve_opts, spectr_opts)
        return fig


# ------------------- Sparse STFT transform -------------------


class WSTFT_LOp(LinearOperator):
    
    def __init__(self, shape, window, overlap, time_axis, dt,
                 target='scipy', nfft=None, stft_kwargs=None,
                 background_weight=0.0, dtype=None):
        """
            Creates operator Short-Term Fourier Transform (STFT) (forward & inverse)

            Arguments:
                shape: tuple of ints : shape of reference data
                window: 1D array : weighting window for STFT
                overlap: float : % of overlapping of sliding windows
                time_axis: int : position of time axis in reference array
                dt : float : sampling frequency (in seconds)
                target : str : backend for computations (`scipy` or `ssq`)
                stft_kwargs : dict : arguments for forward STFT (see `scipy.stft` or `ssq.stft`)
                dtype : data type, by default np.float32

        """

        # Reshaping
        self.shaper = NDshaper(np.empty(shape), axis=time_axis, axplace=0)
        self.data_shape = self.shaper.transhape
        self.N = self.data_shape[0]
        self.window = window
        self.nperseg = len(window)
        self.overlap = overlap 
        self.noverlap = int(self.overlap * self.nperseg)
        self.hop = (self.nperseg - self.noverlap)
        self.dt = dt
        self.fs = 1 / dt    
        
        # STFT Transform settings
        self.target = target
        self.backends = ['scipy', 'ssq', 'ssq_gpu']
        self.nfft = nfft
        self.set_nfft()

        self.fshape = int(max(self.nperseg, self.nfft) / 2) + 1
        self.f = np.linspace(0, 0.5 * self.fs, self.fshape)
        self.t = forward_time_axis(self.N, self.target, self.nperseg, self.hop, self.dt)
        self.tshape = len(self.t)

        self.stft_kwargs = stft_kwargs
        self.set_STFT_kwargs()
        self.background_weight = background_weight

        super().__init__(dtype=np.dtype(dtype), shape=(self.N, self.fshape * self.tshape))
        self.explicit = False

        # Masking for optimization
        self.mask = None
    
    def set_STFT_kwargs(self):

        forw_kw, inv_kw = {}, {}

        # Common parameters
        forw_kw['window'] = inv_kw['window'] = self.window

        if self.target == 'scipy':
            forw_kw['fs'] = inv_kw['fs'] = self.fs
            forw_kw['nfft'] = inv_kw['nfft'] = self.nfft
            forw_kw['nperseg'] = inv_kw['nperseg'] = self.nperseg
            forw_kw['noverlap'] = inv_kw['noverlap'] = self.noverlap
            forw_kw['return_onesided'] = inv_kw['input_onesided'] = True
            forw_kw['padded'] = self.stft_kwargs.get('padded', False)
            forw_kw['detrend'] = self.stft_kwargs.get('detrend', False)
            forw_kw['boundary'], inv_kw['boundary'] = None, True
            forw_kw['axis'], inv_kw['time_axis'], inv_kw['freq_axis'] = 0, 1, 0

        elif 'ssq' in self.target:
            forw_kw['n_fft'] = inv_kw['n_fft'] = self.nfft
            forw_kw['win_len'] = inv_kw['win_len'] = self.nperseg
            forw_kw['hop_len'] = inv_kw['hop_len'] = self.nperseg - self.noverlap
            forw_kw['modulated'] = inv_kw['modulated'] = self.stft_kwargs.get('modulated', False)
            inv_kw['win_exp'] = 1
            forw_kw['padtype'], forw_kw['derivative'] = self.stft_kwargs.get('padtype', 'reflect'), False
            forw_kw['t'], forw_kw['fs'] = None, self.fs

        self.forw_kw = forw_kw
        self.inv_kw = inv_kw

    def set_nfft(self):
        self.nfft = self.nperseg if (self.nfft is None) else self.nfft  # None means nperseg
        if self.nfft < 0:
            self.nfft = 2 ** int(np.ceil(np.log2(self.nperseg)))  # closest biggest power of 2
        else:
            self.nfft = max(self.nfft, self.nperseg)  # biggest of two
        self.nfft = self.nfft + self.nfft % 2  # Always make "even" number

    def _forward_backend(self, x):
        
        if not self.target in self.backends:
            raise KeyError(f"Unknown `target` = `{self.target}`, must be one of {self.backends}")

        if self.target == 'scipy':
            f, t, C = signal.stft(x, **self.forw_kw)
            C = C.swapaxes(-1, 1)

        elif 'ssq' in self.target:
            gpu_status = 'gpu' in self.target
            os.environ['SSQ_GPU'] = '1' if gpu_status else '0'
        
            C = ssq.stft(x.T, **self.forw_kw)
            if gpu_status: 
                C = C.cpu().numpy()
            C = np.moveaxis(C, 0, -1)

        return C

    def _inverse_backend(self, x):
        
        if not self.target in self.backends:
            raise KeyError(f"Unknown `target` = `{self.target}`, must be one of {self.backends}")

        if self.target == 'scipy':
        
            t, C = signal.istft(x, **self.inv_kw)
            C = C[:self.N]

        elif 'ssq' in self.target:
            gpu_status = 'gpu' in self.target
            os.environ['SSQ_GPU'] = '1' if gpu_status else '0'

            if x.ndim == 2:
                C = ssq.istft(x, **self.inv_kw)
            else:
                C = np.zeros(self.shaper.outshape)
                for i in range(x.shape[-1]):
                    # ci = ssq.istft(x[..., i], **self.inv_kw)
                    # C[:, i] = ci.cpu().numpy() if gpu_status else ci
                    C[:, i] = ssq.istft(x[..., i], N=self.N, **self.inv_kw)
        else:
            raise ValueError(f"Unknown target backend {self.target}")

        return C

    def _matvec(self, C):
        if self.mask is not None:
            C = C * self.mask
        return self._inverse_backend(self.Reshape(C))
    
    def _rmatvec(self, x):
        return self.Flatten(self._forward_backend(x))
    
    def _matmat(self, C):
        return self._matvec(C)
        
    def _rmatmat(self, x):
        return self._rmatvec(x)

    def matvec(self, x):
        return self._matvec(x)

    def rmatvec(self, x):
        return self._rmatvec(x)

    def matmat(self, x):
        return self._matmat(x)

    def rmatmat(self, x):
        return self._rmatmat(x)

    def _adjoint(self):
        return CustomAdjointLinearOperator(self)

    def Inverse(self, C):
        Xf = self._inverse_backend(C.reshape(*C.shape[:2], -1))
        return self.shaper.unshape(Xf)

    def Forward(self, x):
        D = self.shaper.flatten(self.shaper.transpose(x))
        Cmn = self._forward_backend(D)
        return self.shaper.unflatten(Cmn)

    def Reshape(self, C):       
        return C.reshape(self.fshape, self.tshape, *C.shape[1:])
    
    def Flatten(self, Cmn):
        return Cmn.reshape(-1, *Cmn.shape[2:])

    def setup_FISTA(self, Y, X, eps=1e-1, niter=15, pbar=True, alpha=None, callbacks=None, **kwargs):
        # kwargs.setdefault('show', pbar)
        # callback = TqdmExt(total=niter, disable=not pbar, desc='FISTA')

        self.fista = FISTA(self, callbacks=callbacks)
        self.fista.setup(Y, X, eps=eps, niter=niter, show=pbar, alpha=alpha, **kwargs)

    def Sparse(self, X, eps=1e-1, niter=15, pbar=True, alpha=None, **fista_kw):
        
        # Preparation
        D = self.shaper.transpose(X)
        D = self.shaper.flatten(D)
        
        Cmn = self._forward_backend(D)
        C0 = self.Flatten(Cmn)
        # C0 = np.zeros_like(C0)
        Cmn = self.shaper.unflatten(Cmn)

        self.setup_FISTA(D, C0, eps=eps, niter=niter, pbar=pbar, alpha=alpha, **fista_kw)
        xfista = self.fista.run(C0)

        if self.background_weight:

            inds = ~(np.abs(xfista) > 0.0)
            xfista[inds] = C0[inds] * self.background_weight

        Cfista = self.shaper.unflatten(xfista)
        Cfista = self.Reshape(Cfista)

        Xf = self * xfista
        Xf = self.shaper.unflatten(Xf)
        Df = self.shaper.untranspose(Xf)
        
        return Df, Cmn, Cfista

    def _debias(self, D, C0, **lsqr_kw):
        lsqr_kw.setdefault('iter_lim', 15)
        lsqr_kw.setdefault('damp', 0.0)
        lsqr_kw.setdefault('show', False)
        pbar = lsqr_kw.pop('pbar', False)

        Cfls = np.empty_like(C0)
        for i, ci in enumerate(tqdm(C0.T, disable=not pbar, desc='LSQR')):
            self.mask = abs(ci) > 0
            x, *info = LSQR(self, D[:, i], x0=ci, **lsqr_kw)
            Cfls[:, i] = x * self.mask

        self.mask = None
        return Cfls

    def Debias(self, X, Csparse, **lsqr_kw):

        # Preparation
        D = self.shaper.transpose(X)
        D = self.shaper.flatten(D)
        C = self.Flatten(Csparse)

        Cfls = self._debias(D, C, **lsqr_kw)
        
        Xfls = self.shaper.unflatten(self * Cfls)
        Dfls = self.shaper.untranspose(Xfls)
        
        Cfls = self.shaper.unflatten(Cfls)
        Cfls = self.Reshape(Cfls)

        return Dfls, Cfls

    def DebiasedSparse(self, X, pbar=True, update_coefs=False, fista_kw={}, lsqr_kw={}):
        
        # Preparation
        D = self.shaper.transpose(X)
        D = self.shaper.flatten(D)
        
        Cmn = self._forward_backend(D)
        C0 = self.Flatten(Cmn)
        Cmn = self.shaper.unflatten(Cmn)
        
        xfista, *f_info = self._sparse(D, C0, pbar=pbar, **fista_kw)
        if update_coefs:
            xfista = self.H * (self * xfista)
        xfls = self._debias(D, xfista, pbar=pbar, **lsqr_kw)
        
        Cfista = self.shaper.unflatten(xfista)
        Cfista = self.Reshape(Cfista)

        Cfls = self.shaper.unflatten(xfls)
        Cfls = self.Reshape(Cfls)
        
        return Cmn, Cfista, Cfls


class CustomAdjointLinearOperator(WSTFT_LOp):
    """Adjoint of arbitrary Linear Operator"""
    def __init__(self, A):
        super().__init__(shape=A.shaper.baseshape,
                         window=A.window,
                         overlap=A.overlap,
                         time_axis=A.shaper.axis,
                         dt=A.dt,
                         target=A.target,
                         nfft=A.nfft,
                         stft_kwargs=A.stft_kwargs,
                         dtype=A.dtype)

        self.shape = (A.shape[1], A.shape[0])
        self.A = A
        self.args = (A,)

    def _matvec(self, x):
        return self.A._rmatvec(x)

    def _rmatvec(self, x):
        return self.A._matvec(x)

    def _matmat(self, x):
        return self.A._rmatmat(x)

    def _rmatmat(self, x):
        return self.A._matmat(x)


class NDshaper:
    def __init__(self, A, axis, axplace=0):
        self.ndim = A.ndim
        self.baseshape = A.shape
        self.axis = self._pos_axis(axis)
        self.axplace = self._pos_axis(axplace)

        self.len = A.shape[axis]
        self.restshape = self.baseshape[:self.axis] + self.baseshape[self.axis + 1:]
        self.transhape = np.moveaxis(A, self.axis, self.axplace).shape
        self.width = np.prod(self.restshape)
        self.flat_permit = self._check_flattening()
        if self.flat_permit == 1:
            self.outshape = (self.len, self.width)
        elif self.flat_permit == 2: 
            self.outshape = (self.width, self.len)
        else:
            self.outshape = None

    def transpose(self, A):
        assert A.shape == self.baseshape
        return np.moveaxis(A, self.axis, self.axplace)
    
    def untranspose(self, A):
        assert A.shape == self.transhape
        return np.moveaxis(A, self.axplace, self.axis)
    
    def flatten(self, A):
        assert self.flat_permit > 0
        assert A.shape == self.transhape
        return A.reshape(*self.outshape)
    
    def unflatten(self, A):
        assert self.flat_permit > 0
        axis = -(self.flat_permit == 1)
        assert A.shape[axis] == self.width
        if self.flat_permit == 1:
            return A.reshape(*A.shape[:axis], *self.restshape)
        else:
            return A.reshape(*self.restshape, *A.shape[axis+1:])
    
    def reshape(self, A):
        assert A.shape == self.baseshape
        return self.flatten(self.transpose(A))
    
    def unshape(self, A):
        assert A.shape == self.outshape
        return self.untranspose(self.unflatten(A))

    def _check_flattening(self):
        if self.axplace == 0:
            return 1
        elif (self.axplace == len(self.baseshape) - 1) or (self.axplace == -1): 
            return 2
        else:
            return 0

    def _pos_axis(self, a):
        return a if a >= 0 else self.ndim + a


def get_time_freq_shape(N, nperseg, noverlap, boundary, padded, n_fft):

    # Frequency axis
    L = max(nperseg, n_fft)
    fshape = int(L / 2) + 1

    # Time axis
    hop = nperseg - noverlap
    bounds = 0; pad = 0
    if boundary:
        bounds += int(nperseg / 2) * 2
    if padded and boundary and (y := (N + bounds - nperseg) % hop) != 0:
        pad = hop - y
    elif padded and (not boundary):
        pad += int(nperseg / 2) * 2
    tshape = int((N + bounds + pad - nperseg) / hop) + 1

    return fshape, tshape


class TqdmExt(std_tqdm):
    def update(self, n=1):
        displayed = super(TqdmExt, self).update(n)
        return displayed

    def __call__(self, x):
        return self.update()


# -- new -- #


def forward_time_axis(N, backend, nperseg, hop, dt_sec):
    if backend == 'scipy':
        N_padded = N - nperseg % 2
        N_padded += ((N - nperseg % 2) % hop) % nperseg
    else:
        N_padded = N - 1
    Nt = N_padded // hop + 1
    time = np.arange(Nt) * dt_sec * hop
    return time


def inverse_time_samples(Nt, nperseg, hop):
    return (Nt - 1) * hop + nperseg % 2 + hop


def define_stft_window(window, dt_sec):
    if isinstance(window, tuple):
        wtype, length = window
    elif isinstance(window, (int, float)):
        wtype, length = 'ones', window
    else:
        raise ValueError(f"Unknown type of window {type(window)}")

    nperseg = round(length / dt_sec) + 1

    if ('one' in wtype) or (len(wtype) == 0) or (wtype is None):
        window = np.ones(nperseg)
    else:
        window = signal.get_window(wtype, nperseg)

    return window
