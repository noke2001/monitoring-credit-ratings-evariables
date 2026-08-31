"""Candidate summary functions $\\Gamma$ — thesis Chapter 4, Section 4.3.

A supervised model returns a predictive distribution $P_{t,i,\\cdot}$ over the
rating scale.  Section 4.1 needs a *scalar*, so something must condense that
vector against the rating $r_i$ the bond actually carries.  Five constructions
are used in this thesis; they are collected here so they can be compared on one
panel rather than argued about.

All five are oriented so that **large means downgrade candidate**, which is the
convention the cohort PIT of Section 4.2 and the one-sided E-values of Section
4.4 assume.  Where the natural definition runs the other way it is negated, and
the docstring says so.

    directional_deviation   E[enc(R)] - enc(r)        the thesis default
    transition_risk_ratio   P(R > r) / P(R < r)       ratio of tail masses
      (deadzone=th)         neutral when P(r) > th    the modified TRR
    brier                   the quadratic score       proper
    ranked_probability      RPS                       proper, ordinal
    log_score               -log P(r)                 proper, local

The last three are *proper scoring rules* in the sense of
\\cite{WaghmareZiegel2026}: the expected score is optimised when the quoted
distribution is the forecaster's true belief.  Used as a summary they measure
surprise rather than direction --- a bond scores badly whether it is about to
be upgraded or downgraded --- so they are naturally one-sided, and Section 4.4
tests them against $H_1 = H_{ST}$ alone rather than the two-sided compound.
The first two are directional and need the two-sided construction.
"""

import numpy as np

EPS = 1e-12


def _check(P, k):
    P = np.asarray(P, dtype=float)
    k = np.asarray(k)
    if P.ndim != 2:
        raise ValueError("P must be (n_rows, n_classes)")
    if len(k) != len(P):
        raise ValueError("P and k must have the same number of rows")
    valid = np.isfinite(k)
    ki = np.clip(np.where(valid, k, 0).astype(int), 0, P.shape[1] - 1)
    return P, ki, valid


def directional_deviation(P, k, enc=None):
    """$\\Gamma = \\sum_r P(r)\\,\\mathrm{enc}(r) - \\mathrm{enc}(r_i)$.

    The thesis default.  With ``enc`` increasing in credit risk this is
    positive when the model reads the bond as worse than its label, so it is
    already oriented for downgrade.  Interpretable and cheap, but ad hoc: it
    collapses the whole distribution onto its mean, so a bond with a bimodal
    forecast (likely to move, direction unclear) scores the same as one the
    model is confident about.
    """
    P, ki, valid = _check(P, k)
    enc = np.arange(P.shape[1], dtype=float) if enc is None else np.asarray(enc, float)
    out = P @ enc - enc[ki]
    return np.where(valid, out, np.nan)


def transition_risk_ratio(P, k, log=True, deadzone=None):
    """$\\mathrm{TRR} = P(R > r_i) / P(R < r_i)$, downgrade mass over upgrade mass.

    Ignores $P(R = r_i)$ entirely, which is the dominant mass in a monthly
    panel, so it reads the *shape of the tails* rather than the height of the
    mode.  That makes it far more responsive than the deviation, and far
    noisier: the denominator is tiny for high-grade bonds and the raw ratio is
    heavy-tailed.  ``log=True`` returns $\\log \\mathrm{TRR}$, which is the form
    worth ranking; the cohort PIT is invariant to the choice, since the log is
    increasing.

    Note this is the *ratio*, not the odds $P(\\mathrm{down})/(1-P(\\mathrm{down}))$
    whose denominator is $P(\\mathrm{stay}) + P(\\mathrm{up})$.

    ``deadzone`` implements the **modified TRR**: when the model puts more than
    ``deadzone`` of its mass on the carried rating, the ratio is replaced by its
    neutral value (1, or 0 in logs).  The motivation is that for a bond the
    model is sure will stay put, the ratio is computed from two vanishing tail
    masses and might be dominated by numerical noise rather than by credit,
    which would push a placid bond to an extreme rank for no reason.  Whether
    that actually happens is an empirical question, and on this panel it does
    not: see the sweep in ``scripts/audit_tabpfn.py``.  The dead-zone does buy
    a substantially less persistent PIT, so it is a real point on the
    power-versus-validity frontier rather than a strict improvement.

    The neutral value ties every dead-zoned bond at the *centre* of the
    ranking, which is the intent: the cohort PIT then places them among
    themselves at random (Lemma B.2) rather than at either edge.
    """
    P, ki, valid = _check(P, k)
    grid = np.arange(P.shape[1])[None, :]
    p_dn = np.where(grid > ki[:, None], P, 0.0).sum(axis=1)
    p_up = np.where(grid < ki[:, None], P, 0.0).sum(axis=1)
    r = p_dn / np.maximum(EPS, p_up)
    r = np.log(np.maximum(EPS, r)) if log else r
    if deadzone is not None:
        p_stay = P[np.arange(len(P)), ki]
        r = np.where(p_stay > deadzone, 0.0 if log else 1.0, r)
    return np.where(valid, r, np.nan)


def brier(P, k):
    """Quadratic score as a *loss*: $\\sum_j p_j^2 - 2p_{r_i}$.

    The reward form is $S_Q(p, y) = 2p_y - \\sum_j p_j^2$; it is negated here so
    that large means badly forecast, hence downgrade candidate.  Proper, and
    insensitive to the ordering of the classes --- it treats a AAA/B confusion
    exactly like a BBB/BB one, which is why the ranked probability score is
    usually preferable on an ordinal scale.
    """
    P, ki, valid = _check(P, k)
    out = (P ** 2).sum(axis=1) - 2.0 * P[np.arange(len(P)), ki]
    return np.where(valid, out, np.nan)


def ranked_probability(P, k):
    """Ranked probability score, $\\sum_{m} \\bigl(F_m(P) - F_m(\\delta_{r_i})\\bigr)^2$.

    Proper, and the only one of the three that respects the *ordering* of the
    rating scale: being wrong by one notch costs less than being wrong by five.
    On an ordinal target this is the natural choice, and it is the scoring rule
    the thesis reports.
    """
    P, ki, valid = _check(P, k)
    F = np.cumsum(P, axis=1)
    G = (np.arange(P.shape[1])[None, :] >= ki[:, None]).astype(float)
    out = ((F - G) ** 2)[:, :-1].sum(axis=1)
    return np.where(valid, out, np.nan)


def log_score(P, k, floor=1e-8):
    """$-\\log P(r_i)$.  Proper and *local*: it uses only the probability the

    model put on the realised class and ignores how the rest of the mass is
    arranged.  That locality is the point of the rule and its weakness here ---
    a bond whose mass has shifted one notch down scores the same as one whose
    mass has shifted five notches up.  ``floor`` keeps it finite.
    """
    P, ki, valid = _check(P, k)
    out = -np.log(np.maximum(floor, P[np.arange(len(P)), ki]))
    return np.where(valid, out, np.nan)


#: name -> (function, is the alternative one-sided?)
SUMMARIES = {
    "directional_deviation": (directional_deviation, False),
    "log_TRR": (transition_risk_ratio, False),
    "log_TRR_dz_0.9999": (
        lambda P, k: transition_risk_ratio(P, k, deadzone=0.9999), False),
    "log_TRR_dz_0.9": (
        lambda P, k: transition_risk_ratio(P, k, deadzone=0.9), False),
    "brier_loss": (brier, True),
    "ranked_probability": (ranked_probability, True),
    "log_score": (log_score, True),
}


def all_summaries(P, k, enc=None) -> dict:
    """Every candidate $\\Gamma$ on one predictive matrix, keyed by name."""
    out = {}
    for name, (fn, _) in SUMMARIES.items():
        out[name] = fn(P, k, enc=enc) if fn is directional_deviation else fn(P, k)
    return out
