import numpy as np
from scipy import signal
import ssqueezepy as ssq
import os
from .utils import ReshapeInputArray


############################################################
#                      STFT transform                      #
############################################################


class STFT_Operator: 
    def __init__(self, window, overlap=0.5, dt=1, backend='scipy', nfft=None, padtype='reflect', **kwargs):
        """
            Operator for Short-Term Fourier Transform (STFT) (forward & inverse)

            Arguments:
                window: np.ndarray / int : weighting window for STFT
                overlap: float : [0, 1) : overlapping of sliding windows
                dt : float : sampling time (in seconds)
                backend : str : backend for computations (`scipy` or `ssqueezepy` or `ssqueezepy_gpu`)
                nfft : int : how many frequency bins for each individual window. 
                             If `nfft > len(window)`, then zero-padding is performed. 
                             if `nfft < 0`, then `nfft` is rounded to the closest `2**p > len(window)`
                             Default `nfft = len(window)`
                padtype : str : type of padding signal at both ends ('reflect', 'symmetric', 'replicate', 'wrap', 'zero').
                                Default 'reflect'
                kwargs : dict : optional keyword arguments:
                        For `backend = 'scipy'` it can be (see `scipy.signal.stft`):
                            padded = True / False (Forward STFT). Default False
                            detrend = True / False (Forward STFT). Default False
                        For `backend = 'ssqueezepy'` it can be (see `ssqueezepy.stft` & `ssqueezepy.istft`):
                            modulated = True / False (Forward & Inverse STFT). Default False
                            win_exp = integer (Inverse STFT). Default `1`

        """
        self.dt         =   dt
        self.window     =   self.define_stft_window(window)
        self.nperseg    =   len(self.window)
        self.overlap    =   overlap 
        self.noverlap   =   int(self.overlap * self.nperseg)
        self.hop        =   (self.nperseg - self.noverlap)
        self.fs         =   1 / dt
        self.nfft       =   self.nperseg if (nfft is None) else nfft
        if self.nfft < 0:
            self.nfft   =   2**int(np.ceil(np.log2(self.nperseg)))
        else:
            self.nfft   =   max(self.nfft, self.nperseg)
        self.nfft       =   self.nfft + self.nfft % 2
        self.padtype    =   padtype
        self.padedge    =   self.nperseg // 2 
        self.f          =   np.linspace(0, 0.5 * self.fs, self.nfft // 2 + 1)
        self.df         =   self.f[1] - self.f[0]

        # STFT Transform settings
        self._backends  =   ['scipy', 'ssqueezepy', 'ssqueezepy_gpu']
        if backend not in self._backends:
            KeyError(f"Unknown `backend` = `{backend}`, must be one of {self._backends}")
        self.backend    =   backend
        self.kwargs     =   kwargs
        self._set_STFT_kwargs(kwargs)

    def define_stft_window(self, window):
        if isinstance(window, (int, float)):
            window = np.ones(int(window / self.dt))
        elif isinstance(window, np.ndarray):
            window = window
        elif isinstance(window, tuple):
            assert (len(window) == 2) and isinstance(window[1], (int, float))
            sec_perseg = window[1]
            n_perseg   = int(sec_perseg / self.dt)
            window = signal.get_window(window[0], n_perseg)
        return window

    def _set_STFT_kwargs(self, kwargs):
        forw_kw = {}
        inv_kw = {}

        # Common parameters
        forw_kw['window'] = inv_kw['window'] = self.window

        if self.backend == 'scipy':
            forw_kw['fs']               =   inv_kw['fs']                =   self.fs
            forw_kw['nfft']             =   inv_kw['nfft']              =   self.nfft
            forw_kw['nperseg']          =   inv_kw['nperseg']           =   self.nperseg
            forw_kw['noverlap']         =   inv_kw['noverlap']          =   self.noverlap
            forw_kw['return_onesided']  =   inv_kw['input_onesided']    =   True
            forw_kw['padded']           =   kwargs.get('padded', False)
            forw_kw['detrend']          =   kwargs.get('detrend', False)
            forw_kw['boundary'],    inv_kw['boundary']                  =   None,   True
            forw_kw['axis'],        inv_kw['time_axis'],    inv_kw['freq_axis'] =   -1,     -1,     -2

        elif 'ssq' in self.backend:
            forw_kw['n_fft']        =   inv_kw['n_fft']     =   self.nfft
            forw_kw['win_len']      =   inv_kw['win_len']   =   self.nperseg
            forw_kw['hop_len']      =   inv_kw['hop_len']   =   self.nperseg - self.noverlap
            forw_kw['modulated']    =   inv_kw['modulated'] =   kwargs.get('modulated', False)
            inv_kw['win_exp']       =   kwargs.get('win_exp', 1)
            forw_kw['padtype'], forw_kw['derivative']       =   self.padtype,   False
            forw_kw['t'],       forw_kw['fs']               =   None,           self.fs

        self.forw_kw = forw_kw
        self.inv_kw = inv_kw

    def _reset_backend(self, new_backend):
        self.__init__(window=self.window, overlap=self.overlap, 
                      dt=self.dt, backend=new_backend, nfft=self.nfft, 
                      padtype=self.padtype, **self.kwargs)
        
    def padsignal(self, X):
        """
            Performs padding of `X` to the last axis with `window_length // 2` to the both ends

            X : numpy array : shape (..., N), `N` time samples
        """
        N = X.shape[-1]
        if self.backend == 'scipy':
            n1 = self.padedge
            n2 = n1 + self.hop - (N - self.nperseg % 2) % self.hop
        else:
            n1 = 0
            n2 = self.hop - (N - 1) % self.hop
        return np.pad(X, [(0, 0), (n1, n2)], mode=self.padtype)

    def _forward_backend(self, X):
        """
            Performs Forward STFT

            X : numpy array : shape (M, N), where `M` signals, `N` time samples
        """
        Y = self.padsignal(X)
        if self.backend == 'scipy':
            f, t, C = signal.stft(Y, **self.forw_kw)

        elif 'ssq' in self.backend:
            gpu_status = 'gpu' in self.backend
            os.environ['SSQ_GPU'] = '1' if gpu_status else '0'
        
            C = ssq.stft(Y, **self.forw_kw)
            if gpu_status: 
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

        elif 'ssq' in self.backend:
            gpu_status = 'gpu' in self.backend
            os.environ['SSQ_GPU'] = '1' if gpu_status else '0'

            X = []
            for ci in C:
                xi = ssq.istft(ci, N=N, **self.inv_kw)
                if gpu_status:
                    xi = xi.cpu().numpy()
                X.append(xi)
            X = np.stack(X, axis=0)

        return X

    @ReshapeInputArray(dim=2, num=1, methodfunc=True)
    def forward(self, X):
        """
            Performs Forward STFT

            X : numpy array : shape (..., N), where `N` time samples. 
                              `X` can have any dimesional structure for any number of signals, 
                              but the last axis `N` must be `time` axis
        """
        return self._forward_backend(X)

    @ReshapeInputArray(dim=3, num=1, methodfunc=True)
    def inverse(self, C):
        """
            Performs Inverse STFT

            C : numpy array : shape (..., Nf, Nt), where `Nf` frequencies, `Nt` time frames. 
                              `C` can have any dimensional structure for any number of spectrograms,
                              but the last two axes `Nf` and `Nt` must be `frequency` and `time` axes respectively.
            N : int : output length of the original signal. Default None, 
                      will compute default output for `scipy.istft` and `ssqueezepy.istft`.
        """
        return self._inverse_backend(C)

    def forward_time_axis(self, N):
        bounds = self.padedge * 2 if (self.padtype is not None) else 0
        tail = (N + bounds - self.nperseg) % self.hop
        tail = self.hop - tail if tail > 0 else tail
        n_hops = (N + bounds + tail - self.nperseg) // self.hop + 1
        t = self.dt * (np.arange(n_hops) * self.hop + (bounds // 2) * (bounds < 1))
        return t
    
    def inverse_time_samples(self, Nt):
        bounds = self.padedge * 2 if (self.padtype is not None) else 0
        N = (Nt - 1) * self.hop - bounds + self.nperseg + self.hop
        return N
    
    def __mul__(self, X):
        return self.forward(X)
    
    def __truediv__(self, C):
        return self.inverse(C)