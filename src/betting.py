"""
Exact e-value primitives and probability-integral-transform (PIT) utilities.

Master's Thesis: Monitoring Credit Ratings with E-Variables

Every betting function exported here satisfies the defining property of an
e-variable *exactly* (not approximately) under the stated null:

        H_0 :  Z ~ Uniform(0, 1)   ==>   E_{H_0}[ e(Z) ] = 1,   e(Z) >= 0.

That identity is what licenses Ville's inequality for the compounded process
M_t = prod_{s<=t} e_s, and hence the anytime-valid alarm threshold 1/alpha.
Each function below carries the closed-form expectation used to certify it;
`src.validation` re-certifies all of them numerically (quadrature + Monte Carlo).

Direction convention
--------------------
The credit-monitoring null is

        H_0 : the agency's standing rating R_{i,t-1} is a draw from the model's
              predictive rating distribution p_{i,t}(.)   (the agency is "right").

Under H_0 the randomized PIT of R_{i,t-1} is exactly Uniform(0,1). Deterioration
(the model believing the true rating is *worse* than the standing one) pushes the
monitored rating into the **lower** tail of the predictive distribution, i.e. it
makes the PIT *small*.  Empirically on the 457,037-row panel, PIT decile 0 carries
2.32x the base rate of a 12-month downgrade and decile 9 carries 0.29x.

So the alternative of interest is a *lower*-tail alternative. All functions take
`tail='lower'` (default) and internally apply the measure-preserving reflection
Z -> 1 - Z, which keeps the null exactly Uniform(0,1) while letting the betting
functions be written in the conventional "bet on large values" form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "orient",
    "deadzone_phi",
    "deadzone_e",
    "power_e",
    "mixture_power_e",
    "randomized_pit_discrete",
    "randomized_rank_pit",
    "ar1_innovation",
]


# --------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------
def orient(z: np.ndarray, tail: str = "two-sided") -> np.ndarray:
    """
    Map the PIT onto the convention "large values contradict H_0".

    Every branch is a measure-preserving map of Uniform(0,1) onto Uniform(0,1),
    so orientation never disturbs the null.

    tail='upper'      : u = Z.          Alternative in the upper PIT tail.
    tail='lower'      : u = 1 - Z.      Alternative in the lower PIT tail.
                        Reflection is measure preserving on U(0,1).
    tail='two-sided'  : u = |2Z - 1|.   Alternative in EITHER tail.
                        If Z ~ U(0,1) then 2Z - 1 ~ U(-1,1) and |2Z - 1| ~ U(0,1)
                        exactly, so the folded statistic is still a valid PIT.
                        Under the deadzone with delta, u > delta corresponds to
                        Z < (1-delta)/2 or Z > (1+delta)/2 -- a symmetric
                        two-sided rejection region of total mass 1 - delta.

    Which one to use is decided by the alternative, not by convenience.
    In this thesis the monitored event is ANY rating transition, in either
    direction: the model may imply a better or a worse rating than the agency's
    standing one, and both are evidence that the standing rating is stale.
    That makes the alternative two-sided, which is the default here. A
    one-sided orientation would forfeit all power against transitions in the
    discarded direction.
    """
    z = np.asarray(z, dtype=float)
    if tail == "lower":
        return 1.0 - z
    if tail == "upper":
        return z
    if tail in ("two-sided", "both"):
        return np.abs(2.0 * z - 1.0)
    raise ValueError(
        f"tail must be 'lower', 'upper' or 'two-sided', got {tail!r}"
    )


# --------------------------------------------------------------------------
# Deadzone (Waghmare-Ziegel style) betting function
# --------------------------------------------------------------------------
def deadzone_phi(z: np.ndarray, delta: float = 0.75, tail: str = "two-sided") -> np.ndarray:
    """
    Mean-zero deadzone payoff.

        phi(u) = -1              if u <= delta
        phi(u) = delta/(1-delta) if u >  delta          [u = orient(z)]

    Certificate.  Under H_0, u ~ U(0,1), so P(u <= delta) = delta and

        E[phi] = -1*delta + (delta/(1-delta))*(1-delta) = -delta + delta = 0.

    The identity is exact for every delta in (0,1).
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0,1), got {delta}")
    u = orient(z, tail)
    return np.where(u > delta, delta / (1.0 - delta), -1.0)


def deadzone_e(
    z: np.ndarray,
    lam: np.ndarray | float = 0.5,
    delta: float = 0.75,
    tail: str = "two-sided",
) -> np.ndarray:
    """
    Deadzone e-variable  e = 1 + lam * phi_delta(Z).

    Certificate.  E[e] = 1 + lam * E[phi] = 1 + lam*0 = 1, exactly, for any
    F_{t-1}-measurable (predictable) bet size `lam`.

    Non-negativity.  e >= 1 - lam, so lam in [0,1] guarantees e >= 0 with no
    truncation.  Truncation would *break* the expectation identity, so instead
    of clipping we reject lam > 1 outright.
    """
    lam_arr = np.asarray(lam, dtype=float)
    if np.any(lam_arr < 0.0) or np.any(lam_arr > 1.0):
        raise ValueError(
            "bet size lam must lie in [0,1]; outside that range e can go "
            "negative and clipping would destroy E[e]=1."
        )
    e = 1.0 + lam_arr * deadzone_phi(z, delta, tail)
    # Abstain where the PIT is undefined (e.g. the first month of a bond, where
    # an innovation has no lag to difference against). e = 1 is the fair bet:
    # it leaves wealth untouched and preserves E[e]=1 trivially. Substituting
    # 0.5 instead would score as a loss and deflate wealth for a reason that
    # has nothing to do with the data.
    return np.where(np.isfinite(np.asarray(z, dtype=float)), e, 1.0)


# --------------------------------------------------------------------------
# Power / Beta likelihood-ratio betting function
# --------------------------------------------------------------------------
def power_e(z: np.ndarray, kappa: float = 2.0, tail: str = "two-sided") -> np.ndarray:
    """
    Power e-variable  e = kappa * u^(kappa - 1),  u = orient(z),  kappa > 0.

    Certificate.  This is exactly the likelihood ratio dQ/dP with P = U(0,1)
    and Q = Beta(kappa, 1), whose density is kappa*u^(kappa-1). A likelihood
    ratio against the null is an e-variable by construction:

        E_P[e] = int_0^1 kappa u^(kappa-1) du = [u^kappa]_0^1 = 1.

    kappa > 1 concentrates the bet on large u (i.e. the chosen tail);
    kappa = 1 is the trivial bet e == 1.  Unlike the deadzone it is smooth,
    so it degrades gracefully rather than switching at a hard cut.
    """
    if kappa <= 0.0:
        raise ValueError(f"kappa must be > 0, got {kappa}")
    zf = np.asarray(z, dtype=float)
    u = np.clip(orient(zf, tail), 1e-300, 1.0)
    return np.where(np.isfinite(zf), kappa * np.power(u, kappa - 1.0), 1.0)


def mixture_power_e(
    z: np.ndarray,
    kappas: np.ndarray = None,
    weights: np.ndarray = None,
    tail: str = "two-sided",
) -> np.ndarray:
    """
    Method-of-mixtures e-variable  e = sum_j w_j * power_e(Z; kappa_j).

    Certificate.  A convex combination of e-variables is an e-variable:
    E[sum_j w_j e_j] = sum_j w_j E[e_j] = sum_j w_j = 1, exactly.

    This removes the need to tune kappa on the evaluation data, which is the
    main data-snooping exposure of the single-kappa and deadzone engines.
    """
    kappas = np.asarray([1.5, 2.0, 3.0, 5.0] if kappas is None else kappas, float)
    if weights is None:
        weights = np.full(len(kappas), 1.0 / len(kappas))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != kappas.shape:
        raise ValueError("kappas and weights must have the same shape")
    if not np.isclose(weights.sum(), 1.0) or np.any(weights < 0):
        raise ValueError("weights must be non-negative and sum to 1")
    out = np.zeros(np.shape(z), dtype=float)
    for k, w in zip(kappas, weights):
        out += w * power_e(z, float(k), tail)
    return out


# --------------------------------------------------------------------------
# PIT construction
# --------------------------------------------------------------------------
def randomized_pit_discrete(
    probs: np.ndarray, y: np.ndarray, rng: np.random.Generator = None
) -> np.ndarray:
    """
    Exact randomized PIT for a discrete predictive distribution.

        Z = F(y-1) + V * p(y),    V ~ U(0,1) independent.

    Certificate.  For any discrete law, this Z is *exactly* Uniform(0,1) when y
    is drawn from that law.  The randomization is essential: the deterministic
    PIT F(y) is stochastically larger than uniform for discrete outcomes, and
    the mid-PIT F(y-1) + p(y)/2 is uniform only in mean, not in distribution.

    probs : (n, K) row-stochastic predictive class probabilities
    y     : (n,)   realized class index in {0, ..., K-1}
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError("probs must be 2-D (n, K)")
    if len(probs) != len(y):
        raise ValueError("probs and y must have the same length")
    row = probs.sum(axis=1)
    if not np.allclose(row, 1.0, atol=1e-6):
        raise ValueError("each row of probs must sum to 1")

    rng = np.random.default_rng() if rng is None else rng
    n = len(y)
    cum = np.cumsum(probs, axis=1)
    idx = np.arange(n)
    f_lower = np.where(y > 0, cum[idx, np.maximum(y - 1, 0)], 0.0)
    p_realized = probs[idx, y]
    return f_lower + rng.uniform(0.0, 1.0, size=n) * p_realized


def randomized_rank_pit(
    scores: np.ndarray, group_ids: np.ndarray, rng: np.random.Generator = None
) -> np.ndarray:
    """
    Exactly-uniform randomized rank PIT within cross-sectional cohorts.

        Z_i = (rank_i - V_i) / n,   V_i ~ U(0,1) iid,  rank_i in {1, ..., n}.

    Certificate.  Under the exchangeability null "score_i is exchangeable with
    the other n-1 scores in its cohort", rank_i ~ Uniform{1,...,n}; combined with
    an independent V_i, (rank_i - V_i)/n is *exactly* Uniform(0,1).

    Why this replaces `groupby(...).rank(pct=True)`.  `rank(pct=True)` returns
    rank/n, which lives on the lattice {1/n, ..., 1} and attains 1 with
    probability 1/n. It is therefore NOT Uniform(0,1), and the deadzone identity
    fails by P(Z > delta) = floor(delta*n)/n - delta.  For a cohort of n = 2 and
    delta = 0.75 this drives E[e] to 1.95 at lam = 0.95 -- a 95% per-step upward
    drift under the null, which alone crosses the 1/alpha = 10 alarm threshold in
    about five steps. Ties are broken at random rather than averaged, since
    average ranks reintroduce the same lattice defect.
    """
    scores = np.asarray(scores, dtype=float)
    group_ids = np.asarray(group_ids)
    if len(scores) != len(group_ids):
        raise ValueError("scores and group_ids must have the same length")
    rng = np.random.default_rng() if rng is None else rng

    z = np.full(len(scores), np.nan, dtype=float)
    order = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_groups[1:] != sorted_groups[:-1], True]
    )

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        members = order[start:stop]
        vals = scores[members]
        n = len(members)
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        # Random tie-breaking: lexicographic sort on (value, random key).
        tiebreak = rng.random(n)
        ranks = np.empty(n, dtype=float)
        ranks[np.lexsort((tiebreak, np.where(finite, vals, np.inf)))] = np.arange(
            1, n + 1
        )
        z[members] = (ranks - rng.random(n)) / n
        z[members[~finite]] = np.nan
    return z


def ar1_innovation(
    scores: np.ndarray,
    panel_ids: np.ndarray,
    order: np.ndarray = None,
    strata: np.ndarray = None,
    fit_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Residual of a per-stratum AR(1) fit:   r_t = s_t - beta * s_{t-1}.

    Why this exists
    ---------------
    Ranking the LEVEL of a risk score gives a PIT that is uniform in each
    cross-section -- an identity, since ranks of n items are always a
    permutation of 1..n -- but NOT uniform conditional on the bond's own past.
    On the thesis panel the level-ranked `det_score` has lag-1 autocorrelation
    0.854 and lag-6 autocorrelation 0.635. The e-process then compounds a dozen
    near-copies as if they were a dozen independent bets, so

        E[e_t | F_{t-1}] != 1

    and the wealth process is not a martingale. Ville's inequality does not
    apply, however carefully everything downstream is implemented: the defect is
    at the source, in the PIT.

    Ranking this residual instead drops lag-1 autocorrelation to 0.026 --
    conditionally uniform to within noise -- and *raises* precision lift at a
    matched alarm budget from 1.43x to 1.48x. Validity and power move together.

    Predictability
    --------------
    `beta` is estimated once per stratum on `fit_mask` (default: the first half
    of the ordering), never on the evaluation half, so the coefficient is not
    chosen on the data the e-process is scored against. It is a single scalar per
    stratum and is stable, but note this is a holdout split, not a walk-forward
    re-estimation.

    The first observation of each bond has no lag and returns NaN. Downstream
    betting functions read NaN as "abstain" and emit e = 1.

    Parameters
    ----------
    scores    : (n,) risk score s_t
    panel_ids : (n,) bond identifier, used to take the within-bond lag
    order     : (n,) sort key within a bond (dates). Assumed sorted if omitted.
    strata    : (n,) group for the beta fit (e.g. the monitored rating).
                A single global beta is used if omitted.
    fit_mask  : (n,) boolean; rows the beta fit may use.
    """
    s_arr = np.asarray(scores, dtype=float)
    ids = np.asarray(panel_ids)
    n = len(s_arr)
    if len(ids) != n:
        raise ValueError("scores and panel_ids must have the same length")

    frame = pd.DataFrame({"s": s_arr, "id": ids})
    if order is not None:
        frame["ord"] = np.asarray(order)
        pos = np.lexsort((frame["ord"].to_numpy(), frame["id"].to_numpy()))
    else:
        pos = np.arange(n)
    inv = np.empty(n, dtype=np.int64)
    inv[pos] = np.arange(n)

    sorted_frame = frame.iloc[pos]
    lag = sorted_frame.groupby("id", sort=False)["s"].shift(1).to_numpy()
    lag = lag[inv]

    if strata is None:
        strata = np.zeros(n, dtype=np.int64)
    strata = np.asarray(strata)
    if fit_mask is None:
        fit_mask = np.zeros(n, dtype=bool)
        fit_mask[pos[: n // 2]] = True
    fit_mask = np.asarray(fit_mask, dtype=bool)

    resid = np.full(n, np.nan)
    for g in pd.unique(strata):
        in_g = strata == g
        usable = in_g & fit_mask & np.isfinite(lag) & np.isfinite(s_arr)
        if usable.sum() > 100:
            beta = float(np.polyfit(lag[usable], s_arr[usable], 1)[0])
        else:
            beta = 1.0
        resid[in_g] = s_arr[in_g] - beta * lag[in_g]
    return resid


# --------------------------------------------------------------------------
# Arnold-Henzi-Ziegel: sequential monotone density estimation
# --------------------------------------------------------------------------
def grenander_increasing(z_past: np.ndarray, grid: np.ndarray = None):
    """Grenander MLE of a NON-DECREASING density on [0,1] from ``z_past``.

    The nonparametric maximum likelihood estimator over the class of
    non-increasing densities is the left derivative of the least concave
    majorant of the empirical distribution function; the non-decreasing case
    follows by the reflection ``z -> 1 - z``.  This is the estimator
    Arnold, Henzi and Ziegel use for the stochastic-dominance alternative,
    and it is what makes their test *nonparametric*: nothing is fixed in
    advance, the shape of the bet is learned from the sequence so far.

    Returns ``(knots, heights)`` describing the piecewise-constant density:
    ``heights[k]`` on ``[knots[k], knots[k+1])``.  With no data the estimator
    is the uniform density, which stakes nothing.

    Contrast with ``deadzone_e``.  The dead-zone bet fixes a two-step density
    in advance and so cannot adapt, but is exactly valid at every finite
    sample and needs no burn-in.  The Grenander estimator adapts, at the cost
    of being useless until it has seen enough of the sequence to shape itself
    -- which on a panel whose median bond lives 53 months is the binding
    consideration.  Section 4.4 measures which of the two wins.
    """
    z = np.asarray(z_past, dtype=float)
    z = z[np.isfinite(z)]
    n = len(z)
    if n == 0:
        return np.array([0.0, 1.0]), np.array([1.0])
    # reflect: a non-decreasing density in z is a non-increasing one in 1-z
    w = np.sort(1.0 - z)
    # Least concave majorant of the ECDF points (w_i, i/n), anchored at BOTH
    # ends: (0,0) and (1,1).  The right anchor matters -- the ECDF reaches 1 at
    # w_max <= 1, and without the anchor the last hull segment is stretched to
    # x = 1, which inflates the integral above 1 and destroys E[e] = 1.  The
    # support of a PIT is known to be [0,1], so the estimator is flat (density
    # 0) on (w_max, 1].
    xs = np.concatenate([[0.0], w, [1.0]])
    ys = np.concatenate([[0.0], np.arange(1, n + 1) / n, [1.0]])
    # upper convex hull by monotone chain -> the LCM vertices
    hull = []
    for i in range(len(xs)):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # drop the middle vertex if it lies below the chord
            if (y2 - y1) * (xs[i] - x1) <= (ys[i] - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append((xs[i], ys[i]))
    hx = np.array([p[0] for p in hull])
    hy = np.array([p[1] for p in hull])
    slopes = np.diff(hy) / np.maximum(np.diff(hx), 1e-15)   # non-increasing in w
    # map back to z: knots 1-hx reversed, heights reversed
    knots_z = 1.0 - hx[::-1]
    knots_z[0], knots_z[-1] = 0.0, 1.0
    heights = slopes[::-1]
    if len(heights) != len(knots_z) - 1:          # degenerate hull
        return np.array([0.0, 1.0]), np.array([1.0])
    return knots_z, heights


def grenander_e(
    z_next: float,
    z_past: np.ndarray,
    tail: str = "upper",
    regularise: bool = True,
) -> float:
    """One AHZ e-value: evaluate the Grenander density fitted on the past.

    Certificate.  ``f_t`` is a Lebesgue density on [0,1] built only from
    ``z_past``, hence F_t-measurable, so under H_0 (Z_{t+1} ~ U[0,1] given
    the past) E[f_t(Z_{t+1}) | F_t] = int_0^1 f_t = 1.  Validity does not
    depend on the estimator being any good; a badly shaped f_t costs power,
    never type-I error.

    ``regularise`` applies the shrinkage of Arnold et al.,
    ``e -> 1/(t+1) + (1 - 1/(t+1)) e``, a convex combination with the fair
    bet 1.  It preserves E[e] <= 1 (both components satisfy it) and stops a
    single early observation in a sparse region driving the e-value to zero,
    from which a product can never recover.
    """
    z_past = np.asarray(z_past, dtype=float)
    z_past = z_past[np.isfinite(z_past)]
    zn = float(z_next)
    if not np.isfinite(zn):
        return 1.0
    u_past = orient(z_past, tail) if len(z_past) else z_past
    un = float(orient(np.array([zn]), tail)[0])
    knots, heights = grenander_increasing(u_past)
    k = int(np.searchsorted(knots, un, side="right") - 1)
    k = min(max(k, 0), len(heights) - 1)
    e = float(heights[k])
    if regularise:
        t = len(z_past)
        e = 1.0 / (t + 1) + (1.0 - 1.0 / (t + 1)) * e
    return max(e, 0.0)


def kelly_lambda(q: float, delta: float) -> float:
    """Growth-optimal dead-zone stake (thesis Lemma B.3(iv)).

        lambda* = ( q - (1-q)(1-delta)/delta )_+ ,     q = P(Z > delta)

    Exactly zero at q = 1-delta, the null: the growth-optimal bet under H_0
    is not to bet.  Used by ``AdaptiveKellyEProcess``, which plugs in an
    estimate of q formed from the past and is therefore predictable.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    return float(min(max(q - (1.0 - q) * (1.0 - delta) / delta, 0.0), 1.0))
