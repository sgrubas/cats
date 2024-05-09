from typing import Callable, Any
from pydantic import BaseModel, Extra
from cats.core.date import xi_func, lambda_func, get_interval_division, _BEDATE_trimming
from cats.core.utils import ReshapeArraysDecorator, StatusKeeper, save_pickle, load_pickle
import numpy as np
import numba as nb
from scipy.signal import hilbert, find_peaks
from scipy.linalg import LinAlgError

# TODO:
#   - S picks from estimated instantaneous period of the strongest and most prominent phase?


class NewPicker(BaseModel, extra=Extra.allow):
    dt_sec: float

    # DATE
    minSNR: float = 5.0

    # Peak finding
    rel_peak_height: float = 0.75
    min_peak_width: float = 0.005
    peak_eps: float = 0.7
    peak_prominence_eps: float = 0.5
    peak_dist_to_S: float = 1.0
    peak_dist_to_P: float = 1.0

    correct_picks: bool = True
    corr_eps: float = 1.0
    svd_rcond: float = 1e3
    reference_S_phase: bool = None

    S_comps_slice: Any = slice(None, 2)
    P_comps_slice: Any = slice(-1, None)
    S_aggregation: Callable = np.max
    P_aggregation: Callable = np.max

    name: str = "NewPicker"

    @ReshapeArraysDecorator(dim=3, methodfunc=True, output_num=0)
    def _pick(self, x):

        result_container = {}

        # ----- Hilbert Transform ----- #
        x_HT = hilbert(x, axis=-1)
        envelope = abs(x_HT)
        result_container['envelope'] = envelope

        # ----- BEDATE ----- #

        N = envelope.shape[-1]
        M = envelope.shape[1]
        _dims = np.full(M, 2)  # dims 2 as analytic signal (Real, Imag)
        _xi = xi_func(_dims, self.minSNR)
        _lambda = lambda_func(_dims)
        _frames = get_interval_division(N, N)
        _groups = get_interval_division(M, 1)

        P_comp = self.P_aggregation(envelope[self.P_comps_slice], axis=0)
        S_comp = self.S_aggregation(envelope[self.S_comps_slice], axis=0)

        result_container['P_envelope'] = P_comp
        result_container['S_envelope'] = S_comp

        PS_comps = np.stack([P_comp, S_comp], axis=0)

        bedated, sgms = _BEDATE_trimming(PS_comps,
                                         time_frames=_frames,
                                         freq_groups=_groups,
                                         xi=_xi, lamb=_lambda,
                                         Nmin=int(0.25 * N), original_mode=False)
        result_container['P_envelope_trimmed'] = P_comp_trimmed = bedated[0]
        result_container['S_envelope_trimmed'] = S_comp_trimmed = bedated[1]
        result_container['P_envelope_noise_std'] = sgms[0]
        result_container['S_envelope_noise_std'] = sgms[1]

        # ----- S arrivals ----- #
        S_arrivals, S_peak_phase, S_peak_sizes = self.get_biggest_peak(S_comp_trimmed, self.peak_dist_to_S, None)

        # ----- P arrivals ----- #
        max_inds = (S_arrivals / self.dt_sec).astype(np.int32)  # P arrivals are always before S arrival
        P_arrivals, P_peak_phase, P_peak_sizes = self.get_biggest_peak(P_comp_trimmed, self.peak_dist_to_P, max_inds)

        # ----- Correction by strongest phase ----- #
        if self.correct_picks:  # 'Bad' picks will be replaced with linear trend from `valid picks
            if self.reference_S_phase is None:
                strong_phase_ind = np.argmax([P_peak_sizes.mean(), S_peak_sizes.mean()])  # 0 - P, 1 - S
            else:
                strong_phase_ind = int(self.reference_S_phase)
            P_arrivals, S_arrivals, anom_picks = correct_picks_by_strongest(P_arrivals, S_arrivals,
                                                                            strong_phase_ind,
                                                                            self.corr_eps, self.svd_rcond)
            result_container['anomalous_picks'] = anom_picks

        result_container['picks'] = np.stack([P_arrivals, S_arrivals], axis=-1)

        return result_container

    def get_biggest_peak(self, envelope, peak_dist, max_limit_inds=None):
        shape = envelope.shape[:-1]

        peak_sizes = np.zeros(shape)
        peak_phase = np.zeros(shape)
        arrivals = np.zeros(shape)

        for ind in np.ndindex(shape):
            env_ind = envelope[ind]
            if max_limit_inds is not None:
                env_ind = env_ind[: max_limit_inds[ind]]

            max_peak = np.max(env_ind, initial=0.0)

            peaks, props = find_peaks(env_ind,
                                      height=max_peak * self.peak_eps,
                                      prominence=max_peak * self.peak_prominence_eps,
                                      width=self.min_peak_width / self.dt_sec,
                                      rel_height=self.rel_peak_height)
            try:
                # heights = props['peak_heights']
                # peaks_feature = props['peak_heights'] * props['widths'] * props['prominences']

                peak_ind = 0  # the first prominent peak

                peak_sizes[ind] = props['peak_heights'][peak_ind]
                peak_phase[ind] = peaks[0] * self.dt_sec
                peak_width = props['widths'][peak_ind]
                # peak_width = peak_dist * (peaks[max_ind] - props['left_ips'][max_ind])
                arrivals[ind] = (peaks[0] - peak_dist * peak_width) * self.dt_sec
            except (ValueError, IndexError):
                peak_phase[ind] = 0.0
                arrivals[ind] = 0.0

        return arrivals, peak_phase, peak_sizes

    def pick(self, x, verbose):
        """
            x: (N_comps, N_traces, N_samples):
        """
        history = StatusKeeper(verbose=verbose)
        with history("Picking"):
            result_container = self._pick(x)

        return PickingResult(**result_container, history=history)

    def __mul__(self, x):
        return self.pick(x, verbose=False)

    def __pow__(self, x):
        return self.pick(x, verbose=True)

    def __matmul__(self, x):
        return self.pick(x, verbose=True)

    def export_main_params(self):
        return {kw: val for kw in type(self).__fields__.keys() if (val := getattr(self, kw, None)) is not None}

    def reset_params(self, **params):
        kwargs = self.export_main_params()
        kwargs.update(params)
        self.__init__(**kwargs)

    def save(self, filename):
        save_pickle(self.export_main_params(), filename)

    @classmethod
    def load(cls, filename):
        loaded = load_pickle(filename)
        if isinstance(loaded, cls):
            loaded = loaded.export_main_params()
        return cls(**loaded)


class PickingResult(BaseModel, extra=Extra.allow):
    envelope: Any = None
    trimmed_envelope: Any = None
    picks: Any = None
    history: Any = None


@nb.njit("b1[:](f8[:], f8[:], f8)", cache=True)
def find_anomalous_picks(reference, picks, eps):
    status = np.full(picks.shape, False, dtype=np.bool_)
    last_valid_pick = picks[0]
    last_valid_ref = reference[0]
    for i in range(1, len(picks)):
        pick_i, ref_i = picks[i], reference[i]
        d_pick = pick_i - last_valid_pick
        d_ref = ref_i - last_valid_ref

        if abs(d_pick - d_ref) > eps * abs(d_ref):
            curr = True
        else:
            last_valid_pick = pick_i
            last_valid_ref = ref_i
            curr = False

        status[i] = curr
    return status


def correct_picks_by_strongest(P_picks: np.ndarray,
                               S_picks: np.ndarray,
                               strong_ind: int = 1,
                               dstrong_eps: float = 1.0,
                               rcond: float = None):
    shape = P_picks.shape
    if strong_ind == 0:
        strong_phase = P_picks
        weak_phase = S_picks
    else:
        strong_phase = S_picks
        weak_phase = P_picks

    _strong_phase = strong_phase.ravel()
    _weak_phase = weak_phase.ravel()

    # Sort by strongest phase
    argsort = np.argsort(_strong_phase)
    sort_back = np.argsort(argsort)

    # Find picks which are not in the trend,
    # i.e. have bigger change in values than `diff(strong_phase_sort)`
    anom_ids = find_anomalous_picks(_strong_phase[argsort], _weak_phase[argsort], dstrong_eps)[sort_back]

    weak_appr = _weak_phase
    good_ids = ~anom_ids

    # Replace bad picks with the linear trend (S_phase, picks)
    if 1 < np.count_nonzero(good_ids) < len(_weak_phase):
        try:
            p = np.polyfit(_strong_phase[good_ids], weak_phase[good_ids], deg=1, rcond=rcond)
            weak_appr[anom_ids] = p[1] + _strong_phase[anom_ids] * p[0]
        except LinAlgError:
            pass

    weak_appr = weak_appr.reshape(shape)
    anom_ids = anom_ids.reshape(shape)

    if strong_ind == 0:
        S_picks = weak_appr
        weak_ind = 1
    else:
        P_picks = weak_appr
        weak_ind = 0

    anom_picks = np.full(shape + (2,), False)
    anom_picks[..., weak_ind] = anom_ids

    return P_picks, S_picks, anom_picks


# ----- old P-picking ----- #
# eff_period = (Strong_phase - S_picks).mean()  # in samples
# eff_separation = int(self.min_P_separation * eff_period / self.dt_sec)
# eff_duration = int(self.min_P_duration * eff_period / self.dt_sec)
#
# dets, intervals = FilterDetection(P_comp_trimmed > 0,
#                                   min_separation=eff_separation,
#                                   min_duration=eff_duration)
#
# # P_picks = Strong_phase - 2 * eff_period  # 'initial' estimate for P picks
# P_picks = np.zeros_like(Strong_phase)
# for ind in np.ndindex(shape):
#     intv = intervals[ind]
#     if len(intv) > 0:
#         P_picks[ind] = intv[0, 0] * self.dt_sec  # take the first break
