"""Gamma marginals with a shared lower support floor (thesis Sections 3.3.2, 3.4).

The floor is computed once per (date, class) from the *pooled window history* —
strictly F_{t-1}-measurable — and shared across blocks, so that no block can
disagree with another about where the support of the target ends (per-block
floors let a single observation below one block's endpoint cost an unbounded
number of nats for reasons unrelated to exchangeability).  Log-densities are
additionally capped below at ``LOGF_CAP`` per observation: a current-month
observation below the historical floor then loses a bounded, predictable number
of nats instead of annihilating the entire e-process.  The capped score is
still an F_{t-1}-measurable non-negative function, so validity is unaffected.
"""

import numpy as np
from scipy.stats import gamma

LOGF_CAP = -30.0
U_EPS = 1e-9


def shared_loc_floor(pooled_window_x: np.ndarray) -> float:
    """Lower support floor from the pooled class history (past data only)."""
    x = np.asarray(pooled_window_x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("no finite historical observations for the loc floor")
    span = float(np.max(x) - np.min(x))
    buffer = max(1e-6, 1e-3 * span)
    return float(np.min(x)) - buffer


def fit_gamma(x: np.ndarray, loc_floor: float) -> tuple:
    """Fit Gamma(a, loc, scale) with the location pinned at the shared floor."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or float(np.min(x)) == float(np.max(x)):
        centre = float(x[0]) if x.size else loc_floor + 1.0
        return (1.0, loc_floor, max(abs(centre - loc_floor), 1e-6))
    loc = min(loc_floor, float(np.min(x)) - 1e-9)
    try:
        a, loc_fit, scale = gamma.fit(x, floc=loc)
    except Exception:
        a, loc_fit, scale = np.nan, loc, np.nan
    if not np.isfinite(a) or not np.isfinite(scale) or scale <= 0 or a <= 0:
        return (1.0, loc, max(float(np.std(x)), 1e-6))
    return (float(a), float(loc_fit), float(scale))


def gamma_logpdf_capped(x: np.ndarray, params: tuple, cap: float = LOGF_CAP) -> np.ndarray:
    a, loc, scale = params
    out = gamma.logpdf(np.asarray(x, dtype=float), a=a, loc=loc, scale=scale)
    return np.maximum(np.nan_to_num(out, nan=cap, neginf=cap), cap)


def gamma_cdf_clipped(x: np.ndarray, params: tuple, eps: float = U_EPS) -> np.ndarray:
    a, loc, scale = params
    u = gamma.cdf(np.asarray(x, dtype=float), a=a, loc=loc, scale=scale)
    return np.clip(np.nan_to_num(u, nan=0.5), eps, 1.0 - eps)
