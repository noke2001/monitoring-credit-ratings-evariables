"""
Numerical certification of the anytime-validity claims.

Nothing here is decorative: each check corresponds to one link in the chain

    exact PIT  ->  E[e] = 1  ->  M is a non-negative martingale  ->  Ville

and a broken link is reported as a FAIL rather than a warning. The suite is
designed to fail loudly on the specific defects that were present in earlier
revisions, so that a regression cannot pass silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, stats

from .betting import deadzone_e, mixture_power_e, power_e, randomized_pit_discrete

__all__ = [
    "certify_e_value",
    "certify_pit_uniformity",
    "simulate_null_panel",
    "ville_false_alarm_rate",
    "legacy_ramp_e",
]

TOL = 1e-9


def legacy_ramp_e(z: np.ndarray, delta: float = 0.75, slope: float = 2.0) -> np.ndarray:
    """
    The betting function used by the previous driver scripts:

        e(z) = 1 + slope*(z - delta)/(1 - delta)   for z > delta,  else 1.

    Reproduced verbatim so the suite can demonstrate that E[e] = 1.25 != 1.
    Do not use for inference.
    """
    z = np.asarray(z, dtype=float)
    return np.where(z > delta, 1.0 + slope * (z - delta) / (1.0 - delta), 1.0)


def certify_e_value(e_fn, name: str, n_mc: int = 2_000_000, seed: int = 0) -> dict:
    """
    Certify E_{Z~U(0,1)}[e(Z)] <= 1 by adaptive quadrature and Monte Carlo.

    Quadrature is authoritative (it is exact to ~1e-10 for these piecewise-smooth
    integrands); Monte Carlo is reported as an independent cross-check.
    """
    quad, quad_err = integrate.quad(
        lambda z: float(e_fn(np.array([z]))[0]),
        0.0,
        1.0,
        points=[0.25, 0.5, 0.75],
        limit=400,
    )
    rng = np.random.default_rng(seed)
    draws = e_fn(rng.uniform(0.0, 1.0, n_mc))
    mc = float(np.mean(draws))
    mc_se = float(np.std(draws, ddof=1) / np.sqrt(n_mc))
    nonneg = bool(np.all(draws >= -TOL))
    return {
        "name": name,
        "E_quad": quad,
        "quad_err": quad_err,
        "E_mc": mc,
        "mc_se": mc_se,
        "nonnegative": nonneg,
        "passes": bool(quad <= 1.0 + 1e-8 and nonneg),
    }


def certify_pit_uniformity(z: np.ndarray, name: str, alpha: float = 0.01) -> dict:
    """
    Kolmogorov-Smirnov test of H_0: Z ~ Uniform(0,1), plus the quantity that
    actually matters downstream -- the deadzone identity E[phi] = 0, equivalently
    P(Z > delta) = 1 - delta.
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    ks = stats.kstest(z, "uniform")
    tail_err = {d: float(np.mean(z > d) - (1.0 - d)) for d in (0.5, 0.75, 0.9)}
    return {
        "name": name,
        "n": int(z.size),
        "mean": float(z.mean()),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "tail_error": tail_err,
        "passes": bool(ks.pvalue > alpha),
    }


def simulate_null_panel(
    n_bonds: int = 2000, n_months: int = 120, seed: int = 0
) -> pd.DataFrame:
    """
    A panel drawn exactly from H_0: PIT iid Uniform(0,1), no rating changes.

    Any alarm raised on this panel is by construction a false alarm, so the
    empirical alarm rate is a direct estimate of the type-I error.
    """
    rng = np.random.default_rng(seed)
    n = n_bonds * n_months
    dates = pd.date_range("2000-01-31", periods=n_months, freq="ME")
    return pd.DataFrame(
        {
            "isin": np.repeat([f"B{i:06d}" for i in range(n_bonds)], n_months),
            "dates": np.tile(dates, n_bonds),
            "pit": rng.uniform(0.0, 1.0, n),
            "is_rating_change": np.zeros(n, dtype=bool),
            "is_downgrade": np.zeros(n, dtype=int),
            "directional_deviation": rng.normal(0.0, 1.0, n),
            "downgrade_tail_prob": rng.uniform(0.0, 1.0, n),
            "hazard_score_24m": rng.uniform(0.0, 1.0, n),
            "level_rank": rng.uniform(0.0, 1.0, n),
            "prev_enc_y": rng.integers(0, 6, n),
        }
    )


def ville_false_alarm_rate(result: pd.DataFrame, alpha: float, name: str) -> dict:
    """
    Fraction of bonds that ever alarm on a pure-null panel.

    Ville's inequality asserts this is <= alpha for a genuine e-process with a
    single run per bond. The pass band uses a one-sided binomial tolerance of
    3 standard errors to absorb Monte Carlo noise.
    """
    per_bond = result.groupby("isin")["is_alarm"].any()
    rate = float(per_bond.mean())
    n = int(per_bond.size)
    se = float(np.sqrt(max(alpha * (1 - alpha), 1e-12) / n))
    return {
        "name": name,
        "alpha": alpha,
        "false_alarm_rate": rate,
        "n_bonds": n,
        "tolerance": alpha + 3 * se,
        "max_M": float(result["M_t"].max()),
        "passes": bool(rate <= alpha + 3 * se),
    }
