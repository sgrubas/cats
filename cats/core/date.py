import numpy as np
import numba as nb
from .utils import ReshapeArraysDecorator, get_interval_division, get_logarithmic_interval_division
from scipy import special, optimize


# ---------------------- B-E-DATE ---------------------- #


@nb.njit("i8(i8, f8)", cache=True)
def _Nmin(N, Q):
    maxQ = 1 - N / 4 / (N / 2 - 1) ** 2
    Q = min(Q, maxQ)
    Nmin = int(0.5 * (N - np.sqrt(N / (1 - Q))))
    Nmin = min(max(2, Nmin), N)
    return Nmin


@nb.njit(["f8(f8[:], f8, f8, f8, b1)",
          "f4(f4[:], f8, f8, f8, b1)"], cache=True)
def _DATE(Y, xi, lamb, Q, original_mode):
    N = len(Y)
    Nmin = _Nmin(N, Q)

    Y_sort = np.sort(Y)
    M = M0 = sum(Y_sort[:Nmin - 1])
    std = std0 = (M0 + Y_sort[Nmin - 1]) / Nmin / lamb
    found = False
    for ni in range(Nmin - 1, N - 1):
        M += Y_sort[ni]
        M_s = M / (ni + 1)
        std = M_s / lamb  # threshold height (std * xi)
        found = (Y_sort[ni] <= std * xi < Y_sort[ni + 1])
        if found:
            break
    if original_mode:
        std = std if found else std0  # if not found in the loop, then minimum
    return std


@nb.njit(["f8[:, :, :](f8[:, :, :], i8[:, :], i8[:, :], f8[:], f8[:], f8, b1)",
          "f4[:, :, :](f4[:, :, :], i8[:, :], i8[:, :], f8[:], f8[:], f8, b1)"], parallel=True, cache=True)
def _BEDATE(PSD, time_frames, freq_groups, xi, lamb, Q, original_mode):
    K = PSD.shape[0]
    m, n = len(freq_groups), len(time_frames)
    Sgm = np.zeros((K, m, n), dtype=PSD.dtype)
    for i in nb.prange(m):  # iter over frequencies
        i1, i2 = freq_groups[i]
        xi_i = xi[i]
        lamb_i = lamb[i]
        for k in nb.prange(K):  # iter over spectrograms
            psdl = PSD[k]
            for j in nb.prange(n):  # iter over time frames
                j1, j2 = time_frames[j]
                psd = psdl[i1: i2 + 1, j1: j2 + 1].ravel()
                Sgm[k, i, j] = _DATE(psd, xi_i, lamb_i, Q, original_mode)
    return Sgm


@ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=False, output_num=1, first_shape=True)
def _BEDATE_API(PSD, /, time_frames, freq_groups, xi_lamb, Q, original_mode):
    return _BEDATE(PSD, time_frames, freq_groups, xi_lamb, Q, original_mode)


def BEDATE(PSD, time_frames=None, freq_groups=None, minSNR=4.0, Q=0.95,
           original_mode=False, zero_Nyq_freqs=(True, True)):
    """
        Performs Block-Extended-DATE algorithm 
        
        Arguments:

            PSD : np.ndarray (..., Nf, N) : norm of complex STFT coefficients, with `Nf` frequencies and N time samples
                                            `Nf=0` is zero-frequency, `Nf=Nf` is Nyquist frequency  
            time_frames : np.ndarray (n, 2) / None : independent time frames (stationary windows) in index form.
                      If `None`, the default np.array([[0, N]]), i.e. 1 time frame
            freq_groups : np.ndarray (m, 2) / None : frequency groups.
                      If None, each frequency bin is processed separately
            minSNR : float : expected minimum signal-to-noise ratio
            Q : float [0, 1) : probability value to compute appropriate `Nmin`. Recommended value `0.95`
            original_mode : boolean : whether to use original implementation [1].
                            If `True`, noise is estimated on `Nmin` in the worst case.
                            Default `False`, i.e. we may overestimate noise,
                            but it is good for detection with high minSNR (`xi_lamb`)
            zero_Nyq_freqs : tuple(bool, bool) : Do 'freq_groups' contain `zero` and `Nyquist` frequencies?

        Return:
            Sgm : np.ndarray (..., m, n) : standard deviation

        References:
            [1] Mai, V. K., Pastor, D., Aïssa-El-Bey, A., & Le-Bidan, R. (2015). Robust estimation of non-stationary
            noise power spectrum for speech enhancement. IEEE/ACM Transactions on Audio, Speech,
            and Language Processing, 23(4), 670-682.
    """

    time_frames = time_frames if time_frames is not None else np.array([[0, PSD.shape[-1] - 1]])
    freq_groups = freq_groups if freq_groups is not None else np.stack([PSD.shape[-2]]*2, axis=-1)

    # Calculating constants depending on dimension and `minSNR`
    rho = _minSNR(minSNR, N=np.diff(time_frames, axis=-1).mean(), d=2)
    freq_dimensions = np.full(len(freq_groups), 2)
    if zero_Nyq_freqs[0]:
        freq_dimensions[0] = 1
    if zero_Nyq_freqs[1]:
        freq_dimensions[-1] = 1

    Lambda = _Lambda(freq_dimensions)
    Xi = _Xi(freq_dimensions, rho, d_unique=[1, 2])

    # The B-E-DATE
    Sgm = _BEDATE_API(PSD, time_frames, freq_groups, Xi, Lambda, Q, original_mode)

    return Sgm


########################################################################################


@nb.njit(["UniTuple(f8[:, :, :], 2)(f8[:, :, :], i8[:, :], i8[:, :], f8[:], f8[:], f8, b1)",
          "UniTuple(f4[:, :, :], 2)(f4[:, :, :], i8[:, :], i8[:, :], f8[:], f8[:], f8, b1)"],
         parallel=True, cache=True)
def _BEDATE_trimming(PSD, time_frames, freq_groups, xi, lamb, Q, original_mode):
    K = PSD.shape[0]
    m, n = len(freq_groups), len(time_frames)
    Sgm = np.zeros((K, m, n), dtype=PSD.dtype)
    SNR = np.zeros_like(PSD)
    for i in nb.prange(m):  # iter over frequencies
        i1, i2 = freq_groups[i]
        xi_i = xi[i]
        lamb_i = lamb[i]
        for k in nb.prange(K):  # iter over spectrograms
            psdl = PSD[k]
            for j in nb.prange(n):  # iter over time frames
                j1, j2 = time_frames[j]

                psd = psdl[i1: i2 + 1, j1: j2 + 1]
                Sgm[k, i, j] = sgm = _DATE(psd.ravel(), xi_i, lamb_i, Q, original_mode)

                snr = (psd > sgm * xi_i) * psd / (sgm + 1e-8)
                SNR[k, i1: i2 + 1, j1: j2 + 1] = snr
    return SNR, Sgm


@ReshapeArraysDecorator(dim=3, input_num=1, methodfunc=False, output_num=2, first_shape=True)
def BEDATE_trimming(PSD, /, frequency_groups_index, bandpassed_frequency_groups_slice, bandpass_slice,
                    time_frames, minSNR, time_edge, Q=0.95, original_mode=False, fft_bounds=True):

    m, n = len(frequency_groups_index), len(time_frames)
    groups_slice = bandpassed_frequency_groups_slice

    # constants for B-E-DATE
    freq_dimensions = np.full(m, 2)
    if fft_bounds:
        f_min, f_max = frequency_groups_index[0, 1], frequency_groups_index[-1, 0]
        if f_min == 0:  # zero frequency is 1-dim (real number)
            freq_dimensions[0] = 1
        if f_max == PSD.shape[-2] - 1:  # Nyquist's frequency is 1-dim (real number)
            freq_dimensions[-1] = 1
    Lambda = _Lambda(freq_dimensions)
    Xi = _Xi(freq_dimensions, minSNR, d_unique=[1, 2])

    # The B-E-DATE trimming
    bedate_slice = (..., groups_slice, slice(None))
    noise_std = np.zeros(PSD.shape[:-2] + (m, n), dtype=PSD.dtype)
    spectrogram_SNR_trimmed, noise_std[bedate_slice] = _BEDATE_trimming(PSD, time_frames,
                                                                        frequency_groups_index[groups_slice],
                                                                        Xi[groups_slice],
                                                                        Lambda[groups_slice],
                                                                        Q, original_mode)

    # Removing everything out of the bandpass range
    spectrogram_SNR_trimmed[..., :bandpass_slice.start, :] = 0.0
    spectrogram_SNR_trimmed[..., bandpass_slice.stop:, :] = 0.0

    # Removing spiky high-energy edges (edge effects)
    spectrogram_SNR_trimmed[..., :time_edge] = 0.0
    spectrogram_SNR_trimmed[..., -time_edge - 1:] = 0.0

    return spectrogram_SNR_trimmed, noise_std, Xi


# ------------------------ CONSTANTS ------------------------ #

def _minSNR(rho, N, d):
    return rho or np.sqrt(2 * np.log(N * d))  # `N * d` for `d = 2`


def _Lambda(d):
    return np.sqrt(2) * special.gamma((d + 1) / 2) / special.gamma(d / 2)


def lambda_func(d):
    return _Lambda(d)


def _xi_loss(xi, d, rho):
    return (special.hyp0f1(d / 2, rho**2 * xi**2 / 4) - np.exp(rho**2 / 2))**2


def _xi(d, rho):
    if d == 1:
        return np.arccosh(np.exp(rho**2 / 2)) / rho
    else:
        return abs(optimize.minimize_scalar(_xi_loss, args=(d, rho)).x)


def xi_func(d, rho):
    return _xi(d, rho)


def _Xi(d, rho, d_unique=None):
    if np.isscalar(d):
        return _xi(d, rho)
    else:
        d_unique = d_unique if d_unique else np.unique(d).ravel()
        xsi_kw = {di: _xi(di, rho) for di in d_unique}
        
        xsi = np.empty_like(d, dtype=float)
        for ind, di in np.ndenumerate(d):
            xsi[ind] = xsi_kw[di]
        return xsi


def _Xi_Lambda(d, rho, d_unique=None):
    if np.isscalar(d):
        return _xi(d, rho) / _Lambda(d)
    else:
        d_unique = d_unique if d_unique else np.unique(d)
        xi_lamd_kw = {di: _xi(di, rho) / _Lambda(di) for di in d_unique}

        xi_lamd = np.empty_like(d, dtype=float)
        for ind, di in np.ndenumerate(d):
            xi_lamd[ind] = xi_lamd_kw[di]
        return xi_lamd


def EtaToSigma(eta, rho):
    """
        Converts thresholding function `eta` to standard deviation `sigma` assuming usage with STFT

        Arguments:
            eta : np.ndarray (..., Nf, n) : thresholding function where `Nf` is number of frequencies,
                                             `n` is number of time frames.
            rho : float : used minimum SNR for thresholding function `eta`.

        Returns:
            sigma : np.ndarray (..., Nf, n) : converted standard deviation
    """
    d = np.full(eta.shape[:-1], 2, dtype=int)
    d[..., 0] = 1
    d[..., -1] = 1      # zero and Nyq frequencies are 1-dim samples (imag = 0)
    xi = _Xi(d, rho, d_unique=[1, 2]).astype(eta.dtype)
    return eta / xi[..., None]


# UTILS #

def group_frequency(frequency, freq_step, log_step=False):
    Nf = len(frequency)
    df = frequency[1] - frequency[0]
    if log_step:
        grouped_index = get_logarithmic_interval_division(Nf, freq_step, 10)
    else:
        Ndf = max(round(freq_step / df), 1)
        grouped_index = get_interval_division(Nf, Ndf)

    # Zero frequency is separated due to dimensionality
    if (not log_step) and (grouped_index[0, 1] != 0):
        grouped_index[0, 0] = 1
        grouped_index = np.r_[[[0, 0]], grouped_index]

    # Nyquist's frequency is separated due to dimensionality
    if grouped_index[-1, 0] != Nf - 1:
        grouped_index[-1, 1] = Nf - 2
        grouped_index = np.r_[grouped_index, [[Nf - 1, Nf - 1]]]

    grouped_frequency = frequency[grouped_index]
    return grouped_index, grouped_frequency


def bandpass_frequency_groups(frequency_groups_index, bandpass_index_slice):
    if isinstance(bandpass_index_slice, slice):
        f1 = bandpass_index_slice.start or frequency_groups_index.min()
        f2 = bandpass_index_slice.stop
        f2 = f2 - 1 if f2 else frequency_groups_index.max()
    else:
        f1, f2 = bandpass_index_slice
    groups_inds = np.argwhere((f1 <= frequency_groups_index[:, 1]) & (frequency_groups_index[:, 0] <= f2))
    groups_slice = slice(groups_inds.min(), groups_inds.max() + 1)
    return groups_slice
