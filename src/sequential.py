"""
Sequentially Valid E-Process Engines for Corporate Credit Monitoring
Master's Thesis: Monitoring Credit Ratings with E-Variables
Supervised by: Prof. Dr. Johanna Ziegel, Prof. Dr. Damir Filipovic, Joshua Hayes

--------------------------------------------------------------------------
The validity contract
--------------------------------------------------------------------------
Every engine here compounds e-variables into a wealth process

        M_t = prod_{s <= t} e_s,    e_s = e(Z_s; lambda_{s-1}),

where (i) Z_s is *exactly* Uniform(0,1) under H_0 (see `src.betting`), and
(ii) the bet size lambda_{s-1} is F_{s-1}-measurable ("predictable"), i.e. it
may look at anything known strictly before s but never at Z_s itself.

Under (i)+(ii),  E[M_t | F_{t-1}] = M_{t-1} * E[e_t | F_{t-1}] = M_{t-1},
so M is a non-negative martingale with M_0 = 1 and Ville's inequality gives

        P_{H_0}( sup_t M_t >= 1/alpha ) <= alpha.

That inequality is the *only* thing that makes the alarm threshold 1/alpha
mean "level alpha". Any modification that breaks the supermartingale property
also voids the threshold's interpretation. Two such modifications were present
in earlier revisions of this codebase and are documented, with their exact
consequences, in `ANYTIME_VALIDITY.md`:

  * `M_t <- max(1, M_{t-1} * e_t)`  (a CUSUM-style floor). A floored process is
    a strict *sub*martingale: E[max(1, M_{t-1} e_t)] >= max(1, M_{t-1}) by
    Jensen, with strict inequality whenever P(M_{t-1} e_t < 1) > 0. Ville's
    inequality does not apply and the observed false-alarm rate is unbounded.
    Provided here only behind `floor_at_one=True`, which forces
    `anytime_valid=False` and is refused unless explicitly acknowledged.

  * `e = 1 + 2(Z - 0.75)/0.25` for Z > 0.75, else 1. This has
    E[e] = 0.75 + int_{0.75}^{1}(1 + 8(z - 0.75))dz = 1.25, not 1, so the
    "null" wealth grows like 1.25^t (211x over 24 steps) and crosses the
    alarm threshold of 10 by drift alone in roughly 11 steps.

--------------------------------------------------------------------------
Scope of the guarantee: restarts and multiplicity
--------------------------------------------------------------------------
All engines restart (M <- 1) at realized rating changes and after an alarm.
A restart begins a *fresh* level-alpha test: Ville controls the probability
that a given run ever alarms, so alpha bounds the per-run, not per-bond or
per-panel, false-alarm probability. With R restarts the per-bond bound is
Bonferroni-style R*alpha, and across a panel of B bonds the expected number of
false alarms is bounded by alpha * (total runs), not by alpha. Reported panel
alarm counts must be read against that denominator; `n_runs` is returned by
every engine for exactly this purpose.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .betting import (deadzone_e, grenander_e, kelly_lambda,
                      mixture_power_e, power_e)

__all__ = [
    "DeadzoneEProcess",
    "MixtureRestartEProcess",
    "RollingWindowEProcess",
    "AsymmetricLeakyEProcess",
    "OptimalHybridEProcess",
    "TierVelocityEProcess",
    "InnovationGatedEProcess",
]


# ==========================================================================
# Shared machinery
# ==========================================================================
def _ordered_groups(df: pd.DataFrame, id_col: str, date_col: str):
    """
    Return {id: positional indices in strict chronological order}.

    The sequential loops below are only meaningful if each bond's rows are
    visited in time order. `groupby(...).indices` preserves *row* order, not
    date order, so an unsorted frame silently produces a scrambled e-process.
    Sorting is therefore enforced here rather than assumed of the caller.
    """
    if date_col not in df.columns:
        raise KeyError(f"date column {date_col!r} is required to order the panel")
    if id_col not in df.columns:
        raise KeyError(f"id column {id_col!r} is required to group the panel")
    if not df[date_col].is_monotonic_increasing:
        pass  # global monotonicity is neither required nor sufficient
    grouped = df.groupby(id_col, sort=False).indices
    dates = df[date_col].values
    ordered = {}
    for key, idxs in grouped.items():
        idxs = np.asarray(idxs)
        d = dates[idxs]
        if np.any(np.diff(d.astype("datetime64[ns]").astype("int64")) < 0):
            idxs = idxs[np.argsort(d, kind="stable")]
        if len(np.unique(d)) != len(d):
            warnings.warn(
                f"{id_col}={key!r} has duplicate timestamps; the e-process "
                "treats them as distinct sequential observations.",
                RuntimeWarning,
                stacklevel=2,
            )
        ordered[key] = idxs
    return ordered


def _compound(
    df: pd.DataFrame,
    e_step: np.ndarray,
    threshold: float,
    is_change: np.ndarray,
    cooldown_months: int,
    id_col: str,
    date_col: str,
    floor_at_one: bool = False,
    decay: np.ndarray = None,
    persistence_k: int = 1,
    gate_mask: np.ndarray = None,
):
    """
    Compound e-values per bond with restarts, and record alarms.

    `decay` is an optional predictable multiplicative factor rho_{t-1} in (0,1].
    Deflating wealth is always safe: E[rho_{t-1} M_{t-1} e_t | F_{t-1}]
    = rho_{t-1} M_{t-1} <= M_{t-1}, so M stays a non-negative supermartingale
    and Ville's inequality is preserved (the test only becomes conservative).

    `floor_at_one` reproduces the invalid CUSUM floor for comparison only.
    """
    n = len(df)
    M_vals = np.ones(n)
    alarms = np.zeros(n, dtype=bool)
    n_runs = 0

    for _, idxs in _ordered_groups(df, id_col, date_col).items():
        M = 1.0
        n_runs += 1
        last_alarm_k = -(10**9)
        consec = 0
        for k, idx in enumerate(idxs):
            if is_change[idx]:
                M = 1.0
                consec = 0
                n_runs += 1

            if decay is not None:
                M *= decay[idx]
            M = M * e_step[idx]
            if floor_at_one:
                M = max(1.0, M)
            M_vals[idx] = M

            crossed = M >= threshold
            if gate_mask is not None:
                crossed = crossed and bool(gate_mask[idx])
            consec = consec + 1 if crossed else 0

            if consec >= persistence_k and (k - last_alarm_k) >= cooldown_months:
                alarms[idx] = True
                last_alarm_k = k
                M = 1.0
                consec = 0
                n_runs += 1

    return M_vals, alarms, n_runs


class _BaseEProcess:
    """Common configuration, validation bookkeeping and result assembly."""

    #: Set False by subclasses/settings that forfeit Ville's inequality.
    anytime_valid = True

    def __init__(self, alpha: float = 0.10, tail: str = "two-sided"):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0,1), got {alpha}")
        self.alpha = alpha
        self.threshold = 1.0 / alpha
        self.tail = tail

    def compute_single_step_e(self, z_scores, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def _finish(self, df, e_step, M_vals, alarms, n_runs):
        df = df.copy()
        df["e_step"] = e_step
        df["M_t"] = M_vals
        df["is_alarm"] = alarms
        df.attrs["n_runs"] = n_runs
        df.attrs["alpha"] = self.alpha
        df.attrs["threshold"] = self.threshold
        df.attrs["anytime_valid"] = self.anytime_valid
        df.attrs["expected_false_alarm_bound"] = self.alpha * n_runs
        return df


# ==========================================================================
# 1. Waghmare-Ziegel deadzone martingale
# ==========================================================================
class DeadzoneEProcess(_BaseEProcess):
    """
    Anytime-valid deadzone e-martingale on the PIT.

    Formerly ``WaghmareZiegelEProcess``, which was a misnomer: this fixed
    two-step bet appears in neither Waghmare & Ziegel (proper scoring rules,
    used in Section 4.3) nor Arnold, Henzi & Ziegel (sequential calibration,
    whose method estimates the density instead -- see GrenanderEProcess).
    Its properties are thesis Lemma B.3.

        phi(u) = -1 if u <= delta else delta/(1-delta),   u = orient(Z)
        e_t    = 1 + lam * phi(Z_t)

    E[e_t] = 1 exactly for every lam in [0,1] and delta in (0,1); see
    `src.betting.deadzone_e` for the certificate.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        delta: float = 0.75,
        lam: float = 0.95,
        tail: str = "two-sided",
    ):
        super().__init__(alpha, tail)
        if not 0.0 <= lam <= 1.0:
            raise ValueError(
                f"lam must lie in [0,1] so that e >= 1-lam >= 0; got {lam}. "
                "Clipping a negative e would break E[e]=1."
            )
        self.delta = delta
        self.lam = lam

    def compute_single_step_e(self, z_scores: np.ndarray) -> np.ndarray:
        return deadzone_e(z_scores, self.lam, self.delta, self.tail)

    def run_sequential_test(
        self,
        df: pd.DataFrame,
        z_col: str = "pit",
        rating_change_col: str = "is_rating_change",
        cooldown_months: int = 12,
        id_col: str = "isin",
        date_col: str = "dates",
    ) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        e_step = self.compute_single_step_e(df[z_col].values)
        is_change = df[rating_change_col].values.astype(bool)
        M, alarms, runs = _compound(
            df, e_step, self.threshold, is_change, cooldown_months, id_col, date_col
        )
        return self._finish(df, e_step, M, alarms, runs)


# ==========================================================================
# 2. Mixture-of-restarts e-process  (valid finite-memory alternative)
# ==========================================================================
class MixtureRestartEProcess(_BaseEProcess):
    """
    Anytime-valid changepoint-mixture e-detector with geometric forgetting.

    Motivation. A long quiet stretch drives the plain product towards 0 and
    effectively anaesthetises the monitor: a deterioration starting in year 10
    must first undo ten years of accumulated losses. A truncated rolling window
    fixes that but is not an e-process (see RollingWindowEProcess). The valid
    construction mixes, over every possible changepoint s, the e-process that
    starts betting at s:

        M_t = sum_{s <= t} w_s * prod_{r=s}^{t} e_r   +   sum_{s > t} w_s,
        w_s = (1 - p) p^(s-1),   sum_{s >= 1} w_s = 1.

    The second sum is essential and is what an earlier draft of this class got
    wrong. A component that has not started yet is worth exactly 1, not 0: it is
    the constant process 1 until s, then compounds. Carrying that unspent mass
    keeps M_0 = 1 and makes every component a non-negative martingale in its own
    right, so the convex combination is one too:

        E[M_t | F_{t-1}] = (A_{t-1} + w_t) + p^t = A_{t-1} + p^(t-1) = M_{t-1},

    using E[e_t | F_{t-1}] = 1 and w_t + p^t = p^(t-1). Dropping the unspent mass
    instead lets a newly born component enter at 1 and inflate the average; on an
    iid Uniform(0,1) null panel that pushed the false-alarm rate to 0.334 against
    a nominal alpha = 0.10.

    Memory is geometric rather than truncated: the prior weight on a changepoint
    L periods back decays like p^L, giving a mean start-lag of 1/(1-p). The
    `horizon` argument sets p = 1 - 1/horizon, so `horizon` is that mean lag in
    months. Everything runs in O(1) per step via

        A_t = e_t * (A_{t-1} + w_t),    M_t = A_t + p^t.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        horizon: int = 24,
        delta: float = 0.75,
        lam: float = 0.95,
        tail: str = "two-sided",
    ):
        super().__init__(alpha, tail)
        if horizon < 2:
            raise ValueError("horizon must be >= 2 to define a geometric prior")
        self.horizon = horizon
        self.p = 1.0 - 1.0 / horizon
        self.delta = delta
        self.lam = lam

    def compute_single_step_e(self, z_scores: np.ndarray) -> np.ndarray:
        return deadzone_e(z_scores, self.lam, self.delta, self.tail)

    def _e_step(self, df, z_col, id_col, date_col) -> np.ndarray:
        """Hook: subclasses may make the bet size depend on each bond's past."""
        return self.compute_single_step_e(df[z_col].values)

    def run_sequential_test(
        self,
        df: pd.DataFrame,
        z_col: str = "pit",
        rating_change_col: str = "is_rating_change",
        cooldown_months: int = 12,
        id_col: str = "isin",
        date_col: str = "dates",
    ) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        e_step = self._e_step(df, z_col, id_col, date_col)
        is_change = df[rating_change_col].values.astype(bool)
        p = self.p

        M_vals = np.ones(len(df))
        alarms = np.zeros(len(df), dtype=bool)
        n_runs = 0

        for _, idxs in _ordered_groups(df, id_col, date_col).items():
            A, unspent = 0.0, 1.0   # A_{-1} = 0, sum of unstarted weights = 1
            n_runs += 1
            last_alarm_k = -(10**9)
            for k, idx in enumerate(idxs):
                if is_change[idx]:
                    A, unspent = 0.0, 1.0
                    n_runs += 1
                w = unspent * (1.0 - p)      # weight of a changepoint at this step
                unspent -= w                 # remaining unstarted mass = p^(k+1)
                A = e_step[idx] * (A + w)
                M = A + unspent
                M_vals[idx] = M

                if M >= self.threshold and (k - last_alarm_k) >= cooldown_months:
                    alarms[idx] = True
                    last_alarm_k = k
                    A, unspent = 0.0, 1.0
                    n_runs += 1

        return self._finish(df, e_step, M_vals, alarms, n_runs)


# ==========================================================================
# 3. Rolling-window compounding  (NOT anytime valid -- kept for comparison)
# ==========================================================================
class RollingWindowEProcess(_BaseEProcess):
    """
    Truncated-product "e-process":  M_t = prod_{s = t-W+1}^{t} e_s.

    WARNING -- this is NOT an e-process and 1/alpha is NOT a level-alpha
    threshold for it. Dropping the term e_{t-W} from the product multiplies the
    wealth by 1/e_{t-W}; whenever e_{t-W} < 1 the process jumps *upward* on
    information already in F_{t-1}, so

        E[M_t | F_{t-1}] = M_{t-1} / e_{t-W} > M_{t-1}

    on that event. It is a strict submartingale and Ville's inequality fails.
    Empirically the false-alarm rate under a simulated Uniform(0,1) null runs
    several times the nominal alpha (see `tests_and_plots/validate_math.py`).

    Retained only so the thesis can report the size distortion it induces.
    Use `MixtureRestartEProcess` for a valid finite-memory process.
    """

    anytime_valid = False

    def __init__(
        self,
        alpha: float = 0.10,
        window_size: int = 24,
        delta: float = 0.75,
        lam: float = 0.95,
        tail: str = "two-sided",
        acknowledge_invalid: bool = False,
    ):
        super().__init__(alpha, tail)
        if not acknowledge_invalid:
            warnings.warn(
                "RollingWindowEProcess is not anytime-valid: the truncated "
                "product is a submartingale, so the 1/alpha threshold carries "
                "no type-I error guarantee. Pass acknowledge_invalid=True to "
                "silence this, and do not report its alarms as level-alpha.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.window_size = window_size
        self.delta = delta
        self.lam = lam

    def compute_single_step_e(self, z_scores: np.ndarray) -> np.ndarray:
        return deadzone_e(z_scores, self.lam, self.delta, self.tail)

    def run_sequential_test(
        self,
        df: pd.DataFrame,
        z_col: str = "pit",
        rating_change_col: str = "is_rating_change",
        cooldown_months: int = 12,
        id_col: str = "isin",
        date_col: str = "dates",
    ) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        e_step = self.compute_single_step_e(df[z_col].values)
        is_change = df[rating_change_col].values.astype(bool)

        M_vals = np.ones(len(df))
        alarms = np.zeros(len(df), dtype=bool)
        n_runs = 0

        for _, idxs in _ordered_groups(df, id_col, date_col).items():
            e_arr = e_step[idxs]
            seg_start = 0
            last_alarm_k = -(10**9)
            n_runs += 1
            log_e = np.log(np.maximum(1e-300, e_arr))
            for k, idx in enumerate(idxs):
                if is_change[idx]:
                    seg_start = k
                    n_runs += 1
                w_start = max(seg_start, k - self.window_size + 1)
                M = float(np.exp(log_e[w_start : k + 1].sum()))
                M_vals[idx] = M
                if M >= self.threshold and (k - last_alarm_k) >= cooldown_months:
                    alarms[idx] = True
                    last_alarm_k = k

        return self._finish(df, e_step, M_vals, alarms, n_runs)


# ==========================================================================
# 4. Asymmetric leaky e-martingale (valid: predictable deflation only)
# ==========================================================================
class AsymmetricLeakyEProcess(_BaseEProcess):
    """
    Deadzone e-martingale with predictable recovery deflation.

        M_t = rho_{t-1} * M_{t-1} * e_t,
        rho_{t-1} = rho  if DD_{t-1} <= 0 (credit recovering), else 1.

    Certificate.  rho_{t-1} in (0,1] is F_{t-1}-measurable, so
    E[M_t | F_{t-1}] = rho_{t-1} M_{t-1} <= M_{t-1}: a non-negative
    supermartingale. Ville's inequality still holds, the test merely becomes
    conservative. This is the correct way to express "forget stale evidence" --
    it deflates wealth rather than inflating it.

    (The previous revision documented this decay but never applied it, and left
    `rho` and `dd_col` unused, making the class a duplicate of Waghmare-Ziegel.)
    """

    def __init__(
        self,
        alpha: float = 0.10,
        delta: float = 0.75,
        lam: float = 0.95,
        rho: float = 0.80,
        tail: str = "two-sided",
    ):
        super().__init__(alpha, tail)
        if not 0.0 < rho <= 1.0:
            raise ValueError(
                f"rho must lie in (0,1]; rho > 1 would inflate wealth on "
                f"F_(t-1) information and void Ville's inequality. Got {rho}."
            )
        self.delta = delta
        self.lam = lam
        self.rho = rho

    def compute_single_step_e(self, z_scores: np.ndarray) -> np.ndarray:
        return deadzone_e(z_scores, self.lam, self.delta, self.tail)

    def run_sequential_test(
        self,
        df: pd.DataFrame,
        z_col: str = "pit",
        dd_col: str = "directional_deviation",
        rating_change_col: str = "is_rating_change",
        cooldown_months: int = 12,
        id_col: str = "isin",
        date_col: str = "dates",
    ) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        e_step = self.compute_single_step_e(df[z_col].values)
        is_change = df[rating_change_col].values.astype(bool)

        # rho must be predictable: it reads DD at t-1, never at t.
        dd_prev = df.groupby(id_col)[dd_col].shift(1)
        decay = np.where(dd_prev.fillna(1.0).values <= 0.0, self.rho, 1.0)

        M, alarms, runs = _compound(
            df,
            e_step,
            self.threshold,
            is_change,
            cooldown_months,
            id_col,
            date_col,
            decay=decay,
        )
        return self._finish(df, e_step, M, alarms, runs)


# ==========================================================================
# 5. Optimal hybrid surveillance pipeline
# ==========================================================================
class OptimalHybridEProcess(_BaseEProcess):
    """
    Gated surveillance pipeline: predictable severity gates on the bet size,
    a valid mixture-of-restarts memory, and a K-of-K persistence confirmation.

    Only the *bet size* responds to the gates; the null randomness always comes
    from the PIT. Persistence and cooldown filter alarms after the fact and can
    only reduce the alarm set, so they preserve the type-I error bound.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        dd_gate: float = 0.60,
        p_down_gate: float = 0.35,
        horizon: int = 24,
        persistence_k: int = 2,
        delta: float = 0.75,
        lam_max: float = 0.95,
        tail: str = "two-sided",
    ):
        super().__init__(alpha, tail)
        if not 0.0 <= lam_max <= 1.0:
            raise ValueError(f"lam_max must lie in [0,1], got {lam_max}")
        self.dd_gate = dd_gate
        self.p_down_gate = p_down_gate
        self.horizon = horizon
        self.persistence_k = persistence_k
        self.delta = delta
        self.lam_max = lam_max

    def compute_single_step_e(self, z_scores, dd_prev, p_down_prev) -> np.ndarray:
        lam_t = np.where(
            (dd_prev >= self.dd_gate) & (p_down_prev >= self.p_down_gate),
            self.lam_max,
            0.0,
        )
        # lam = 0  =>  e == 1 exactly: no bet, no information drawn from Z_t.
        return deadzone_e(z_scores, lam_t, self.delta, self.tail)

    def run_sequential_test(
        self,
        df: pd.DataFrame,
        z_col: str = "pit",
        dd_col: str = "directional_deviation",
        p_down_col: str = "downgrade_tail_prob",
        rating_change_col: str = "is_rating_change",
        cooldown_months: int = 12,
        id_col: str = "isin",
        date_col: str = "dates",
    ) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        dd_prev = df.groupby(id_col)[dd_col].shift(1).fillna(0.0).values
        p_down_prev = df.groupby(id_col)[p_down_col].shift(1).fillna(0.0).values
        e_step = self.compute_single_step_e(df[z_col].values, dd_prev, p_down_prev)
        is_change = df[rating_change_col].values.astype(bool)

        M, alarms, runs = _compound(
            df,
            e_step,
            self.threshold,
            is_change,
            cooldown_months,
            id_col,
            date_col,
            persistence_k=self.persistence_k,
        )
        return self._finish(df, e_step, M, alarms, runs)


# ==========================================================================
# 6. Tier + velocity gated e-process
# ==========================================================================
class TierVelocityEProcess(_BaseEProcess):
    """
    Tier-specific, velocity-gated e-martingale.

        lambda_{i,t-1} = 0 unless   h_{i,t-1}      >= h_min(tier_{i,t-1})
                                and dh6m_{i,t-1}   >= velocity_floor
        lambda_{i,t-1} = min(lam_max, lam_base + lam_scale * h_{i,t-1})

        e_{i,t} = 1 + lambda_{i,t-1} * phi_delta(Z_{i,t})

    Every gate input is lagged one period, so lambda is F_{t-1}-measurable and
    E[e_t | F_{t-1}] = 1 exactly. The gates change *power*, never *validity*:
    a gated-off month contributes e = 1 and leaves wealth untouched.

    CAVEAT (data snooping).  The tier gates and lambda coefficients below were
    selected by inspecting downgrade outcomes on the same panel they are
    evaluated on. Ville's inequality is unaffected -- it holds pathwise for any
    predictable bet -- but the reported *precision/recall* are in-sample and
    optimistically biased. Honest out-of-sample numbers require selecting these
    constants on a holdout period; see `ANYTIME_VALIDITY.md`.
    """

    DEFAULT_TIER_GATES = {0: 0.25, 1: 0.25, 2: 0.35, 3: 0.50, 4: 0.30, 5: 0.30}

    def __init__(
        self,
        alpha: float = 0.10,
        tier_gates: dict = None,
        velocity_floor: float = -0.05,
        delta: float = 0.75,
        lam_base: float = 0.2,
        lam_scale: float = 1.0,
        lam_max: float = 0.95,
        tail: str = "two-sided",
    ):
        super().__init__(alpha, tail)
        if not 0.0 <= lam_max <= 1.0:
            raise ValueError(f"lam_max must lie in [0,1], got {lam_max}")
        if lam_base < 0.0 or lam_scale < 0.0:
            raise ValueError("lam_base and lam_scale must be non-negative")
        self.tier_gates = dict(tier_gates or self.DEFAULT_TIER_GATES)
        self.velocity_floor = velocity_floor
        self.delta = delta
        self.lam_base = lam_base
        self.lam_scale = lam_scale
        self.lam_max = lam_max

    def compute_predictable_lambda(self, hazard_prev, hazard_d6m_prev, tier_prev):
        """F_{t-1}-measurable bet size. Vectorised; NaN tiers fall back to BBB."""
        hazard_prev = np.asarray(hazard_prev, dtype=float)
        hazard_d6m_prev = np.asarray(hazard_d6m_prev, dtype=float)
        tier_prev = np.asarray(tier_prev, dtype=float)

        tiers = np.where(np.isnan(tier_prev), 3, tier_prev).astype(int)
        default_gate = 0.35
        gate = np.full(len(tiers), default_gate)
        for tier_value, gate_value in self.tier_gates.items():
            gate[tiers == int(tier_value)] = gate_value

        active = (hazard_prev >= gate) & (hazard_d6m_prev >= self.velocity_floor)
        lam = self.lam_base + self.lam_scale * np.maximum(0.0, hazard_prev)
        return np.where(active, np.minimum(self.lam_max, lam), 0.0)

    def compute_single_step_e(self, z_scores, lambdas) -> np.ndarray:
        return deadzone_e(z_scores, lambdas, self.delta, self.tail)

    def run_sequential_test(
        self,
        df: pd.DataFrame,
        z_col: str = "pit",
        hazard_col: str = "hazard_score_24m",
        hazard_d6m_col: str = "hazard_d6m",
        prev_enc_col: str = "prev_enc_y",
        rating_change_col: str = "is_rating_change",
        cooldown_months: int = 12,
        id_col: str = "isin",
        date_col: str = "dates",
    ) -> pd.DataFrame:
        df = df.reset_index(drop=True)

        if hazard_d6m_col not in df.columns:
            df = df.copy()
            df[hazard_d6m_col] = (
                df.groupby(id_col)[hazard_col].diff(6).fillna(0.0)
            )

        h_prev = df.groupby(id_col)[hazard_col].shift(1).fillna(0.0).values
        h_d6m_prev = df.groupby(id_col)[hazard_d6m_col].shift(1).fillna(0.0).values
        tier_prev = df.groupby(id_col)[prev_enc_col].shift(1).values

        lambdas = self.compute_predictable_lambda(h_prev, h_d6m_prev, tier_prev)
        e_step = self.compute_single_step_e(df[z_col].values, lambdas)
        is_change = df[rating_change_col].values.astype(bool)

        M, alarms, runs = _compound(
            df, e_step, self.threshold, is_change, cooldown_months, id_col, date_col
        )
        out = self._finish(df, e_step, M, alarms, runs)
        out["lambda_bet"] = lambdas
        return out


# ==========================================================================
# 7. Innovation bet, level gate
# ==========================================================================
class InnovationGatedEProcess(_BaseEProcess):
    """
    Bet on the innovation; gate on the level.

    The level of a risk score and its innovation carry different information:

        level       "this bond is risky"
        innovation  "something about this bond just changed"

    Only the innovation may drive Z_t. The level is badly autocorrelated -- 0.854
    at lag 1 on this panel -- so a PIT built from it is uniform in each
    cross-section but NOT uniform conditional on the bond's own past, and
    E[e_t | F_{t-1}] = 1 fails. Using it as the randomness source compounds a
    dozen near-copies as if they were a dozen independent bets.

    But the level at t-1 is F_{t-1}-measurable, so it may set the PREDICTABLE bet
    size without touching validity:

        lambda_{i,t-1} = 0                                    if L_{i,t-1} < g
        lambda_{i,t-1} = min(lam_max, lam_base + lam_scale * (L_{i,t-1} - g)/(1 - g))
        e_{i,t}        = 1 + lambda_{i,t-1} * phi_delta(Z_{i,t})

    where L is the level rank (uniform on [0,1], so `level_gate` reads directly
    as a quantile) and Z is the innovation rank PIT. Since E[phi_delta(Z)] = 0
    for any F_{t-1}-measurable lambda, E[e_t | F_{t-1}] = 1 exactly and Ville's
    inequality is untouched. The gate changes POWER, never VALIDITY: a gated-off
    month contributes e = 1 and leaves wealth alone.

    This recovers the level's signal legitimately -- as a statement about when to
    bet, not about which way the coin landed.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        level_gate: float = 0.70,
        lam_base: float = 0.20,
        lam_scale: float = 0.75,
        lam_max: float = 0.95,
        delta: float = 0.75,
        tail: str = "upper",
    ):
        super().__init__(alpha, tail)
        if not 0.0 <= level_gate < 1.0:
            raise ValueError(f"level_gate must lie in [0,1), got {level_gate}")
        if not 0.0 <= lam_max <= 1.0:
            raise ValueError(f"lam_max must lie in [0,1], got {lam_max}")
        if lam_base < 0.0 or lam_scale < 0.0:
            raise ValueError("lam_base and lam_scale must be non-negative")
        self.level_gate = level_gate
        self.lam_base = lam_base
        self.lam_scale = lam_scale
        self.lam_max = lam_max
        self.delta = delta

    def compute_predictable_lambda(self, level_prev: np.ndarray) -> np.ndarray:
        """F_{t-1}-measurable bet size from the lagged level rank."""
        L = np.asarray(level_prev, dtype=float)
        g = self.level_gate
        active = np.isfinite(L) & (L >= g)
        excess = np.where(active, (L - g) / max(1e-12, 1.0 - g), 0.0)
        lam = self.lam_base + self.lam_scale * excess
        return np.where(active, np.minimum(self.lam_max, lam), 0.0)

    def compute_single_step_e(self, z_scores, lambdas) -> np.ndarray:
        return deadzone_e(z_scores, lambdas, self.delta, self.tail)

    def run_sequential_test(
        self,
        df: pd.DataFrame,
        z_col: str = "pit",
        level_col: str = "level_rank",
        rating_change_col: str = "is_rating_change",
        cooldown_months: int = 12,
        id_col: str = "isin",
        date_col: str = "dates",
    ) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        if level_col not in df.columns:
            raise KeyError(
                f"{level_col!r} is required: it is the level rank that sets the "
                "predictable bet size. Build it with randomized_rank_pit on the "
                "score LEVEL, while z_col holds the innovation PIT."
            )
        level_prev = df.groupby(id_col)[level_col].shift(1).to_numpy()
        lambdas = self.compute_predictable_lambda(level_prev)
        e_step = self.compute_single_step_e(df[z_col].values, lambdas)
        is_change = df[rating_change_col].values.astype(bool)

        M, alarms, runs = _compound(
            df, e_step, self.threshold, is_change, cooldown_months, id_col, date_col
        )
        out = self._finish(df, e_step, M, alarms, runs)
        out["lambda_bet"] = lambdas
        return out


# ==========================================================================
# 8. Arnold-Henzi-Ziegel, out of the box: sequential Grenander estimation
# ==========================================================================
class GrenanderEProcess(_BaseEProcess):
    """The AHZ method as published: no fixed bet, learn the density.

    At each step the non-decreasing density MLE is refitted to the bond's own
    past PIT values and evaluated at the new one,

        e_t = f_{t-1}(Z_t),      f_{t-1} = Grenander MLE on Z_1..Z_{t-1},

    optionally shrunk towards the fair bet as in Arnold et al.  Validity is
    immediate and needs no tuning: f_{t-1} is a density and is
    F_{t-1}-measurable, so E[e_t | F_{t-1}] = 1 whatever shape it has.

    This is the natural benchmark for the fixed dead-zone bet of
    ``DeadzoneEProcess``.  It is strictly more flexible -- it can learn
    any monotone shape rather than a two-step one -- but it must pay for that
    flexibility out of the same short series it is monitoring, and a bond
    whose whole life is 53 months does not supply much to learn from.  The
    ``burn_in`` argument makes this explicit: the engine abstains (e = 1)
    until it has seen that many observations of the bond.
    """

    def __init__(self, alpha: float = 0.10, tail: str = "two-sided",
                 burn_in: int = 6, regularise: bool = True):
        super().__init__(alpha, tail)
        self.burn_in = int(burn_in)
        self.regularise = bool(regularise)

    def compute_single_step_e(self, z_scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError("the Grenander bet is inherently sequential")

    def _e_for_group(self, z: np.ndarray) -> np.ndarray:
        out = np.ones(len(z), dtype=float)
        for t in range(len(z)):
            if t < self.burn_in:
                continue
            if self.tail == "two-sided":
                up = grenander_e(z[t], z[:t], "upper", self.regularise)
                dn = grenander_e(z[t], z[:t], "lower", self.regularise)
                out[t] = 0.5 * (up + dn)     # averaging preserves E[e] <= 1
            else:
                out[t] = grenander_e(z[t], z[:t], self.tail, self.regularise)
        return out

    def run_sequential_test(
        self,
        df: pd.DataFrame,
        z_col: str = "pit",
        rating_change_col: str = "is_rating_change",
        cooldown_months: int = 12,
        id_col: str = "isin",
        date_col: str = "dates",
    ) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        z = df[z_col].to_numpy(dtype=float)
        e_step = np.ones(len(df), dtype=float)
        for idx in _ordered_groups(df, id_col, date_col).values():
            e_step[idx] = self._e_for_group(z[idx])
        is_change = df[rating_change_col].values.astype(bool)
        M, alarms, runs = _compound(
            df, e_step, self.threshold, is_change, cooldown_months, id_col, date_col
        )
        return self._finish(df, e_step, M, alarms, runs)


# ==========================================================================
# 9. Plug-in Kelly: the dead-zone bet with the stake of Lemma B.3(iv)
# ==========================================================================
class AdaptiveKellyEProcess(_BaseEProcess):
    """Dead-zone shape, growth-optimal stake, estimated from the past.

    Lemma B.3(iv) gives the stake that maximises expected log-growth,

        lambda* = ( q - (1-q)(1-delta)/delta )_+ ,     q = P(Z > delta),

    which is exactly 0 under the null.  ``q`` is unknown, so it is estimated
    from the bond's own past PITs with a Laplace-smoothed frequency

        q_hat_t = (#{s < t : Z_s > delta} + a) / (t + a + b),

    with a = 1, b = 1/(1-delta) chosen so that q_hat starts at the null value
    1-delta and hence lambda_hat starts at exactly 0.  q_hat_t is
    F_{t-1}-measurable, so E[e_t | F_{t-1}] = 1 holds exactly -- the bet size
    may depend on the past in any way at all.

    This keeps the finite-sample exactness of the fixed bet while removing
    its one genuinely arbitrary constant.  It cannot learn a shape, only a
    stake, which is what distinguishes it from ``GrenanderEProcess``.
    """

    def __init__(self, alpha: float = 0.10, delta: float = 0.75,
                 tail: str = "two-sided", lam_cap: float = 0.95):
        super().__init__(alpha, tail)
        self.delta = float(delta)
        self.lam_cap = float(lam_cap)

    def compute_single_step_e(self, z_scores: np.ndarray) -> np.ndarray:
        raise NotImplementedError("the plug-in stake is inherently sequential")

    def _lam_for_group(self, z: np.ndarray) -> np.ndarray:
        d = self.delta
        a, b = 1.0, 1.0 / (1.0 - d)          # prior mean a/(a+b) = 1-delta
        lam = np.zeros(len(z), dtype=float)
        hits = 0.0
        for t in range(len(z)):
            q_hat = (hits + a) / (t + a + b)
            lam[t] = min(kelly_lambda(q_hat, d), self.lam_cap)
            if np.isfinite(z[t]):
                u = z[t] if self.tail != "lower" else 1.0 - z[t]
                if self.tail == "two-sided":
                    u = max(z[t], 1.0 - z[t])
                hits += float(u > d)
        return lam

    def run_sequential_test(
        self,
        df: pd.DataFrame,
        z_col: str = "pit",
        rating_change_col: str = "is_rating_change",
        cooldown_months: int = 12,
        id_col: str = "isin",
        date_col: str = "dates",
    ) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        z = df[z_col].to_numpy(dtype=float)
        lam = np.zeros(len(df), dtype=float)
        for idx in _ordered_groups(df, id_col, date_col).values():
            lam[idx] = self._lam_for_group(z[idx])
        e_step = deadzone_e(z, lam, self.delta, self.tail)
        is_change = df[rating_change_col].values.astype(bool)
        M, alarms, runs = _compound(
            df, e_step, self.threshold, is_change, cooldown_months, id_col, date_col
        )
        return self._finish(df, e_step, M, alarms, runs)


# ==========================================================================
# 10. Both improvements at once: growth-optimal stake + geometric memory
# ==========================================================================
class KellyMixtureEProcess(MixtureRestartEProcess):
    """Plug-in Kelly stake compounded through the changepoint mixture.

    The two improvements of Sections 9 and 2 are orthogonal: one decides how
    much to stake at each step, the other how the stakes are combined over
    time.  Both are predictable and both preserve validity on their own, so
    composing them does too.  This engine is what the chapter recommends when
    neither the arbitrary constant nor the unbounded memory is acceptable.
    """

    def __init__(self, alpha: float = 0.10, horizon: int = 24,
                 delta: float = 0.75, tail: str = "two-sided",
                 lam_cap: float = 0.95):
        super().__init__(alpha, horizon=horizon, delta=delta, lam=lam_cap,
                         tail=tail)
        self._kelly = AdaptiveKellyEProcess(alpha=alpha, delta=delta,
                                            tail=tail, lam_cap=lam_cap)

    def _e_step(self, df, z_col, id_col, date_col) -> np.ndarray:
        z = df[z_col].to_numpy(dtype=float)
        lam = np.zeros(len(df), dtype=float)
        for idx in _ordered_groups(df, id_col, date_col).values():
            lam[idx] = self._kelly._lam_for_group(z[idx])
        return deadzone_e(z, lam, self.delta, self.tail)
