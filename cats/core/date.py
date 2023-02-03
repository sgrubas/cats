import numpy as np
import numba as nb
from .utils import ReshapeInputArray
from scipy import special, optimize


########################  B-E-DATE   ########################


@nb.njit(["f8(f8[:], i8, f8, b1)",
          "f8(f4[:], i8, f8, b1)"])
def _DATE(Y, Nmin, xi_lamb, original_mode):
    """
        Estimates thresholding function for outlier trimming via DATE algorithm [1]

        Arguments:
            Y : np.ndarray (N,) : sequence of random numbers
            Nmin : int : minimum number of trimmed values
            xi_lamb : float : xi(rho) / lambda (see DATE)
            original_mode : boolean : whether to use original implementation [1].
                            If `True`, noise is estimated on `Nmin` in the worst case.
                            If `False`, i.e. we may overestimate noise,
                            but it is good for detection with high minSNR (`xi_lamb`)
        Returns:
            eta : float : threshold value for outlier trimming

        References:
            [1] Pastor, D., & Socheleau, F. X. (2012). Robust estimation of noise standard deviation in presence of
            signals with unknown distributions and occurrences. IEEE transactions on Signal Processing, 60(4), 1545-1555.
    """
    N = len(Y)
    Y_sort = np.sort(Y)
    
    M = M0 = sum(Y_sort[:Nmin - 1])
    eta0 = (M0 + Y_sort[Nmin - 1]) / Nmin * xi_lamb
    found = False
    for ni in range(Nmin - 1, N - 1):
        M += Y_sort[ni]
        M_s = M / (ni + 1)
        eta = M_s * xi_lamb
        found = (Y_sort[ni] <= eta < Y_sort[ni + 1])
        if found: break
    if original_mode:
        eta = eta if found else eta0  # if not found in loop, then minimum
    return eta


@nb.njit(["f8[:, :](f8[:, :], i8[:, :], i8[:], f8[:], b1)",
          "f8[:, :](f4[:, :], i8[:, :], i8[:], f8[:], b1)"], parallel=True)
def _BEDATE(PSD, frames, Nmins, xi_lamb, original_mode):
    """
        Performs Block-Extended-DATE algorithm [1]

        Arguments:
            PSD : np.ndarray (M, N) : array of random sequences (norm of complex STFT coefficients)
            frames : np.ndarray (n,) : intervals of time frames (blocks) which are processed independently
            Nmins : np.ndarray (n,) : minimum number of trimmed values for each block
            xi_lamb : np.ndarray (M,) : xi(rho) / lambda for each sequence
            original_mode : boolean : whether to use original implementation [1].
                            If `True`, noise is estimated on `Nmin` in the worst case.
                            If `False`, i.e. we may overestimate noise,
                            but it is good for detection with high minSNR (`xi_lamb`)

        Returns:
            Eta : np.ndarray (M, n) : threshold values for each sequence `M` and each time frame `n`

        References:
            [1] Mai, V. K., Pastor, D., Aïssa-El-Bey, A., & Le-Bidan, R. (2015). Robust estimation of non-stationary
            noise power spectrum for speech enhancement. IEEE/ACM Transactions on Audio, Speech,
            and Language Processing, 23(4), 670-682.
    """
    M, N = PSD.shape
    n = len(frames)
    Eta = np.zeros((M, n))
    for i in nb.prange(M):
        ci, xi_lamb_i = np.abs(PSD[i]), xi_lamb[i]
        for j in range(n):
            j1, j2 = frames[j]
            Eta[i, j] = _DATE(ci[j1 : j2], Nmins[j], xi_lamb_i, original_mode)
    return Eta


@ReshapeInputArray(dim=2, num=1, methodfunc=False)
def _BEDATE_API(PSD, frames, Nmins, xi_lamb, original_mode):
    return _BEDATE(PSD, frames, Nmins, xi_lamb, original_mode)


def BEDATE(PSD, frames=None, minSNR=4.0, Q=0.95, original_mode=False):
    """
        Performs Block-Extended-DATE algorithm 
        
        Arguments:

            PSD : np.ndarray (..., Nf, N) : norm of complex STFT coefficients, with `Nf` frequencies and N time samples
                                            `Nf=0` is zero-frequency, `Nf=Nf` is Nyquist frequency  
            minSNR : float : expected minimum signal-to-noise ratio 
            frames : np.ndarray (n, 2) / None : independent time frames (stationary windows) in index form.
                      If `None`, the default np.array([[0, N]]), i.e. 1 time frame
            Q : float [0, 1) : probability value to compute appropriate `Nmin`. Recommended value `0.95`
            original_mode : boolean : whether to use original implementation [1].
                            If `True`, noise is estimated on `Nmin` in the worst case.
                            Default `False`, i.e. we may overestimate noise,
                            but it is good for detection with high minSNR (`xi_lamb`)

        Return:
            Eta : np.ndarray (..., Nf, n) : thresholding function

        References:
            [1] Mai, V. K., Pastor, D., Aïssa-El-Bey, A., & Le-Bidan, R. (2015). Robust estimation of non-stationary
            noise power spectrum for speech enhancement. IEEE/ACM Transactions on Audio, Speech,
            and Language Processing, 23(4), 670-682.
    """
    preshape, N = PSD.shape[:-1], PSD.shape[-1]
    d = np.full(preshape, 2)
    d[..., 0] = 1; d[..., -1] = 1 # zero and Nyq frequencies are 1-dim samples (imag = 0)

    # Calculating constants depending on dimension and `minSNR`
    rho = minSNR if (minSNR is not None) else np.sqrt(2 * np.log(N * 2)) # `N * d` for `d = 2`
    xi_lamb = _Xi_Lambda(d, rho, d_unique=[1, 2]).reshape(-1)
    frames = np.array([[0, N]]) if frames is None else frames
    Nmins = np.array([_Nmin(abs(i2 - i1), Q) for i1, i2 in frames])

    # The B-E-DATE
    Eta = _BEDATE_API(PSD, frames, Nmins, xi_lamb, original_mode)
    
    return Eta

######################### CONSTANTS #########################


def _Nmin(N, Q=None):
    maxQ = 1 - N / 4 / (N / 2 - 1)**2
    if Q is None: 
        Q = maxQ
    else:
        Q = Q if Q <= maxQ else maxQ
    Nmin = np.clip(int(0.5 * (N - np.sqrt(N / (1 - Q)))), 2, N)
    return Nmin


def _Lambda(d):
    return np.sqrt(2) * special.gamma((d + 1) / 2) / special.gamma(d / 2)


def _xi_loss(xi, d, rho):
    return (special.hyp0f1(d / 2, rho**2 * xi**2 / 4) - np.exp(rho**2 / 2))**2


def _xi(d, rho):
    if d == 1:
        return np.arccosh(np.exp(rho**2 / 2)) / rho
    else:
        return abs(optimize.minimize_scalar(_xi_loss, args=(d, rho)).x)


def _Xi(d, rho, d_unique=None):
    if np.isscalar(d):
        return _xi(d, rho)
    else:
        d_unique = d_unique if d_unique else np.unique(d).ravel()
        xsi_kw = {di : _xi(di, rho) for di in d_unique}
        
        xsi = np.empty_like(d, dtype=float)
        for ind, di in np.ndenumerate(d):
            xsi[ind] = xsi_kw[di]
        return xsi


def _Xi_Lambda(d, rho, d_unique=None):
    if np.isscalar(d):
        return _xi(d, rho) / _Lambda(d)
    else:
        d_unique = d_unique if d_unique else np.unique(d)
        xi_lamd_kw = {di : _xi(di, rho) / _Lambda(di) for di in d_unique}

        xi_lamd = np.empty_like(d, dtype=float)
        for ind, di in np.ndenumerate(d):
            xi_lamd[ind] = xi_lamd_kw[di]
        return xi_lamd


def EtaToSigma(eta, rho):
    """
        Converts thresholding function `eta` to standard deviation `sigma` assuming usage in Time-Frequency domain

        Arguments:
            eta : np.ndarray (..., Nf, n) : thresholding function where `Nf` is number of frequencies,
                                             `n` is number of time frames.
            rho : float : used minimum SNR for thresholding function `eta`.

        Returns:
            sigma : np.ndarray (..., Nf, n) : converted standard deviation
    """
    d = np.full(eta.shape[:-1], 2)
    d[..., 0] = 1
    d[..., -1] = 1      # zero and Nyq frequencies are 1-dim samples (imag = 0)
    xi = _Xi(d, rho, d_unique=[1, 2])
    return eta / xi[..., None]
