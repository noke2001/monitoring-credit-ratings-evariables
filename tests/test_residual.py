"""Certification of the innovation target of thesis Section 3.8.2.

Four things have to hold, and the last is the one that matters:

  1. the location-scale map is fitted correctly, and degrades to a constant map
     rather than to nonsense when the window is thin;
  2. the map is F_{t-1}-measurable — perturbing the current cross-section
     leaves every fitted coefficient untouched;
  3. on a panel whose *level* is persistent but whose *innovation* is
     exchangeable, the residual target really does remove the dependence the
     level test lives on (rank correlation ~ 0, where the level's is ~ phi);
  4. on that same panel E[E_t] <= 1, so the e-value is valid for the
     hypothesis H_0^res that the innovation is exchangeable;
  5. and the window split earns its place: with one window fitting both the map
     and the candidate, a break in the relation is invisible, and with the
     split it is not.

Point 4 is a check of the whole chain: replacing the target does not disturb
Lemma 3.3, because the map is a fixed F_{t-1}-measurable function.  Point 5 is
the reason Section 3.8.2 cannot be implemented quite as literally as it is
written.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate import build_candidate, evaluate_date  # noqa: E402
from src.evalue import mc_permutation_log_evalue  # noqa: E402
from src.residual import (  # noqa: E402
    LOG_HALF_NORMAL_BIAS, cross_section_scale, fit_binned_map,
    fit_loc_scale_map, residualize_item, split_window,
)


# --------------------------------------------------------------------------
# 1. the map itself
# --------------------------------------------------------------------------
def test_location_slope_recovered():
    rng = np.random.default_rng(0)
    z = rng.uniform(-2, 2, size=20_000)
    x = 0.4 + 1.7 * z + rng.normal(scale=0.5, size=z.size)
    m = fit_loc_scale_map(z, x, scale_model="const")
    assert abs(m.b - 1.7) < 0.02, m.b
    assert abs(m.a - 0.4) < 0.02, m.a
    assert abs(np.exp(m.c) - 0.5) < 0.02, np.exp(m.c)


def test_scale_slope_recovered():
    """log sigma(z) = c + d z is recovered from the log|r| regression, bias
    correction included."""
    rng = np.random.default_rng(1)
    z = rng.uniform(-1, 1, size=40_000)
    sigma = np.exp(-0.5 + 0.8 * z)
    x = 1.0 * z + sigma * rng.normal(size=z.size)
    m = fit_loc_scale_map(z, x, scale_model="affine")
    assert abs(m.d - 0.8) < 0.05, m.d
    assert abs(m.c - (-0.5)) < 0.05, m.c
    # and the standardised residual is unit-scale across the whole z range
    r = m.residual(x, z)
    for lo, hi in ((-1, -0.5), (-0.25, 0.25), (0.5, 1.0)):
        sel = (z >= lo) & (z < hi)
        assert abs(np.std(r[sel]) - 1.0) < 0.08, (lo, hi, np.std(r[sel]))


def test_thin_window_degrades_to_constant_map():
    rng = np.random.default_rng(2)
    z = rng.uniform(0, 1, size=5)
    x = 3.0 + 2.0 * z
    m = fit_loc_scale_map(z, x, min_obs=30)
    assert m.fallback and m.b == 0.0 and m.d == 0.0
    assert np.all(np.isfinite(m.residual(x, z)))


def test_scale_is_clipped():
    """A slope fitted on a short window cannot drive the denominator to zero."""
    rng = np.random.default_rng(3)
    z = rng.uniform(0, 1, size=200)
    x = np.where(z < 0.5, rng.normal(scale=1e-6, size=200), rng.normal(size=200))
    m = fit_loc_scale_map(z, x, scale_model="affine")
    s = m.scale(np.linspace(-5, 5, 101))
    assert s.min() >= 0.2 * m.s0 - 1e-12 and s.max() <= 5.0 * m.s0 + 1e-12


def test_bias_constant():
    """The constant added back to the log|r| regression is -E[log|N(0,1)|]."""
    rng = np.random.default_rng(4)
    g = np.abs(rng.normal(size=2_000_000))
    assert abs(np.mean(np.log(g)) + LOG_HALF_NORMAL_BIAS) < 2e-3


# --------------------------------------------------------------------------
# three generators, each a null for a different target
# --------------------------------------------------------------------------
def _panel(kind, rng, n=40, months=24, phi=0.9, rho=0.4, b0=4.0):
    """y_null       X_t = 6 + b0 Y + exchangeable noise, no idiosyncratic
                    persistence.  The LEVEL null is maximally false (the
                    cross-section is ordered by Y at every date); the
                    innovation around the Y map is exchangeable.
       self_null    a random walk started ordered by Y.  The level null is
                    false through persistence AND through the Y ordering --- the
                    panel's own configuration; the innovation around the own-lag
                    map is exchangeable.
       fixed_effect AR(1) around heterogeneous entity means.  Neither pooled
                    affine map is exact, so neither innovation null holds --
                    the counterexample of Section 3.8.2.
    """
    y = np.sort(rng.uniform(0.0, 1.0, size=n))
    cov = (1 - rho) * np.eye(n) + rho * np.ones((n, n))
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(n))
    X = np.empty((months, n))
    if kind == "y_null":
        for t in range(months):
            X[t] = 6.0 + b0 * y + L @ rng.normal(size=n)
        truth = ("y", 6.0, b0)
    elif kind == "self_null":
        X[0] = 6.0 + b0 * y + L @ rng.normal(size=n)   # ordered by Y at t = 0
        for t in range(1, months):
            X[t] = X[t - 1] + L @ rng.normal(size=n)   # exchangeable increments
        truth = ("self", 0.0, 1.0)
    elif kind == "fixed_effect":
        mu = 6.0 + b0 * y
        X[0] = mu + L @ rng.normal(size=n)
        for t in range(1, months):
            X[t] = (1 - phi) * mu + phi * X[t - 1] + L @ rng.normal(size=n)
        truth = None
    else:
        raise ValueError(kind)
    # mean-correct exactly as Section 2.4.2 does on the panel, so the synthetic
    # cross-sections enter the monitor in the same form as the bond ones
    X = X - X.mean(axis=1, keepdims=True)
    Y = y[None, :] + 1e-4 * rng.normal(size=(months, n))
    Y = Y - Y.mean(axis=1, keepdims=True)
    return X, Y, np.array([f"b{i}" for i in range(n)]), truth


def _item(X, Y, ids, t, K=12):
    window = list(range(t - K, t))
    return {
        "date": t, "degenerate": False,
        "x": X[t].copy(), "y_lag": Y[t - 1].copy(), "x_lag": X[t - 1].copy(),
        "bond_ids": ids,
        "window_x_by_bond": {ids[i]: {m: float(X[m, i]) for m in window}
                             for i in range(ids.size)},
        "window_ylag_by_bond": {ids[i]: {m: float(Y[m - 1, i]) for m in window}
                                for i in range(ids.size)},
        "window_xlag_by_bond": {ids[i]: {m: float(X[m - 1, i]) for m in window}
                                for i in range(ids.size)},
        "pooled_window_x": X[window].ravel(),
        "window_month_list": window,
    }


def _oracle(item, truth):
    """Residualise with the generator's own coefficients: the innovation is then
    exactly exchangeable, so only Lemma 3.3 is under test."""
    mode, a, b = truth
    z_cur = item["y_lag"] if mode == "y" else item["x_lag"]
    z_by = (item["window_ylag_by_bond"] if mode == "y"
            else item["window_xlag_by_bond"])
    out = dict(item)
    out["x"] = item["x"] - (a + b * np.asarray(z_cur, dtype=float))
    res, pooled = {}, []
    for bond, months in item["window_x_by_bond"].items():
        zb = z_by.get(bond, {})
        inner = {w: xv - (a + b * zb[w]) for w, xv in months.items() if w in zb}
        if inner:
            res[bond] = inner
            pooled.extend(inner.values())
    out["window_x_by_bond"] = res
    out["pooled_window_x"] = np.asarray(pooled, dtype=float)
    return out


# --------------------------------------------------------------------------
# 2. measurability
# --------------------------------------------------------------------------
def test_map_does_not_see_the_current_cross_section():
    rng = np.random.default_rng(5)
    X, Y, ids, _ = _panel("fixed_effect", rng)
    for mode in ("y", "self"):
        base = residualize_item(_item(X, Y, ids, 20), mode=mode)
        it = _item(X, Y, ids, 20)
        it["x"] = it["x"] + 17.0 + rng.normal(size=it["x"].size)   # shock at t
        pert = residualize_item(it, mode=mode)
        for f in ("a", "b", "c", "d", "s0"):
            assert getattr(base["map"], f) == getattr(pert["map"], f), (mode, f)
        # the window residuals — hence f_b and theta_b — are unchanged too
        assert np.allclose(np.sort(base["pooled_window_x"]),
                           np.sort(pert["pooled_window_x"]))


def test_window_and_current_share_one_map():
    """Both go through the SAME fitted map, so the candidate's marginals are on
    the scale of the thing they score."""
    rng = np.random.default_rng(6)
    X, Y, ids, _ = _panel("fixed_effect", rng)
    out = residualize_item(_item(X, Y, ids, 20), mode="self")
    assert np.allclose(out["x"], out["map"].residual(X[20], X[19]))


# --------------------------------------------------------------------------
# 3. the dependence the level test lives on is gone
# --------------------------------------------------------------------------
def _median_rho(kind, mode, seed, conditioner="y"):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    X, Y, ids, _ = _panel(kind, rng)
    out = []
    for t in range(12, X.shape[0]):
        it = _item(X, Y, ids, t)
        z = Y[t - 1] if conditioner == "y" else X[t - 1]
        x = it["x"] if mode is None else residualize_item(it, mode=mode)["x"]
        out.append(spearmanr(z, x).statistic)
    return float(np.median(out))


def test_y_map_removes_the_lagged_y_ordering():
    assert _median_rho("y_null", None, 7) > 0.70
    assert abs(_median_rho("y_null", "y", 7)) < 0.15


def test_self_map_removes_the_own_lag_ordering():
    assert _median_rho("self_null", None, 8, conditioner="self") > 0.85
    assert abs(_median_rho("self_null", "self", 8, conditioner="self")) < 0.15


def test_y_map_does_not_remove_persistence_orthogonal_to_y():
    """The limitation Section 3.8.2 has to state: the Y map removes only the
    part of the persistence that Y explains."""
    assert abs(_median_rho("self_null", "y", 9, conditioner="self")) > 0.4


# --------------------------------------------------------------------------
# 4. Lemma 3.3 survives the change of target
# --------------------------------------------------------------------------
def _calibration(kind, mode, estimator, reps=40, n_perms=99, seed=11):
    """Mean log E_t and mean normalised permutation rank.

    The rank is the sharp statistic: R_t is uniform on {1, .., N+1} under the
    null, so R_t/(N+1) has mean 1/2 and variance 1/12 whatever the candidate
    does.  E[E_t] itself is capped at N+1 and far too skewed to read off a few
    hundred draws (the same caveat scripts/run_synthetic.py records).
    """
    log_e, ranks = [], []
    for rep in range(reps):
        rng = np.random.default_rng(seed + rep)
        X, Y, ids, truth = _panel(kind, rng, months=18)
        for t in range(12, 18):
            it = _item(X, Y, ids, t)
            if estimator == "oracle":
                it = _oracle(it, truth)
            elif estimator == "fitted":
                it = residualize_item(it, mode=mode)
                if it.get("degenerate"):
                    continue
            cand = build_candidate(it["x"], it["y_lag"], it["bond_ids"],
                                   it["window_x_by_bond"], it["pooled_window_x"],
                                   it["window_month_list"], max_block_size=10)
            if cand is None:
                continue
            le, r = mc_permutation_log_evalue(
                *evaluate_date(cand, it["x"], n_perms, rng))
            log_e.append(le)
            ranks.append(r / (n_perms + 1))
    u = np.asarray(ranks)
    return float(np.mean(log_e)), float(u.mean()), float(u.std(ddof=1) / np.sqrt(u.size))


def test_oracle_map_is_calibrated():
    """With the true map the innovation is exactly exchangeable: the rank is
    uniform and the process does not grow."""
    for kind, mode in (("y_null", "y"), ("self_null", "self")):
        mean_log_e, mean_u, se_u = _calibration(kind, mode, "oracle")
        assert mean_log_e <= 0.05, (kind, mean_log_e)
        assert abs(mean_u - 0.5) <= 3.5 * se_u, (kind, mean_u, se_u)


def test_fitted_map_is_near_calibrated_and_far_below_the_level():
    """The fitted map costs a little — its estimation error is an
    F_{t-1}-measurable, entity-specific offset — but two orders of magnitude
    less than testing the level."""
    for kind, mode in (("y_null", "y"), ("self_null", "self")):
        fit_log_e, fit_u, _ = _calibration(kind, mode, "fitted")
        lvl_log_e, lvl_u, _ = _calibration(kind, None, "level")
        assert fit_log_e < 0.5, (kind, fit_log_e)
        assert fit_log_e < 0.1 * lvl_log_e, (kind, fit_log_e, lvl_log_e)
        assert fit_u > 0.35 and lvl_u < fit_u, (kind, fit_u, lvl_u)


def test_pointwise_cap_still_binds():
    rng = np.random.default_rng(9)
    X, Y, ids, _ = _panel("fixed_effect", rng)
    it = residualize_item(_item(X, Y, ids, 20), mode="self")
    cand = build_candidate(it["x"], it["y_lag"], it["bond_ids"],
                           it["window_x_by_bond"], it["pooled_window_x"],
                           it["window_month_list"], max_block_size=10)
    log_e, _ = mc_permutation_log_evalue(*evaluate_date(cand, it["x"], 99, rng))
    assert log_e <= np.log(100.0) + 1e-12, log_e


def test_degenerate_cohort_is_reported_not_crashed():
    assert residualize_item({"date": 1, "degenerate": True}, mode="y")["degenerate"]
    rng = np.random.default_rng(10)
    X, Y, ids, _ = _panel("self_null", rng, n=6, months=20)
    it = _item(X, Y, ids, 13)
    it["window_xlag_by_bond"] = {}                 # no history to fit on
    assert residualize_item(it, mode="self").get("degenerate")


# --------------------------------------------------------------------------
# 5. the window split
# --------------------------------------------------------------------------
def test_split_window_partitions_and_keeps_order():
    months = list(range(100, 124))
    ref, cand = split_window(months, 0.5)
    assert ref + cand == months and len(ref) == len(cand) == 12
    assert split_window(months, 0.0) == (months, months)      # literal 3.8.2
    ref, cand = split_window(months, 0.99)                    # never empty
    assert len(cand) >= 1 and len(ref) >= 1


def test_split_confines_the_candidate_to_the_recent_half():
    rng = np.random.default_rng(20)
    X, Y, ids, _ = _panel("y_null", rng, months=40)
    it = residualize_item(_item(X, Y, ids, 30, K=24), mode="y",
                          reference_share=0.5)
    assert it["ref_months"] == list(range(6, 18))
    assert it["window_month_list"] == list(range(18, 30))
    assert all(set(v) <= set(range(18, 30))
               for v in it["window_x_by_bond"].values())
    assert it["map"].n_obs <= 12 * ids.size


def _break_panel(rng, n=40, months=40, brk=28, b0=4.0, rho=0.4):
    """Stable relation X_t = b0 Y + noise, whose slope halves at ``brk``."""
    y = np.sort(rng.uniform(0.0, 1.0, size=n))
    L = np.linalg.cholesky((1 - rho) * np.eye(n) + rho * np.ones((n, n))
                           + 1e-9 * np.eye(n))
    X = np.empty((months, n))
    for t in range(months):
        b = b0 if t < brk else 0.5 * b0
        X[t] = 6.0 + b * y + L @ rng.normal(size=n)
    X = X - X.mean(axis=1, keepdims=True)
    Y = y[None, :] + 1e-4 * rng.normal(size=(months, n))
    Y = Y - Y.mean(axis=1, keepdims=True)
    return X, Y, np.array([f"b{i}" for i in range(n)])


def _evidence_after_break(share, reps=6, n_perms=99, K=24, brk=28):
    """Mean log E_t over the dates that follow the break."""
    vals = []
    for rep in range(reps):
        rng = np.random.default_rng(30 + rep)
        X, Y, ids = _break_panel(rng, brk=brk)
        for t in range(brk + 2, X.shape[0]):
            it = residualize_item(_item(X, Y, ids, t, K=K), mode="y",
                                  reference_share=share)
            if it.get("degenerate"):
                continue
            cand = build_candidate(it["x"], it["y_lag"], it["bond_ids"],
                                   it["window_x_by_bond"], it["pooled_window_x"],
                                   it["window_month_list"], max_block_size=10)
            if cand is None:
                continue
            vals.append(mc_permutation_log_evalue(
                *evaluate_date(cand, it["x"], n_perms, rng))[0])
    return float(np.mean(vals))


def test_the_split_is_what_gives_the_monitor_power():
    """One window fitting both jobs is far weaker against a break in the
    relation than the split is.  This is the whole reason for ``split_window``.

    The unsplit version is not powerless once the window straddles the break —
    the map is then a compromise between the two regimes, so its residuals do
    carry some block structure — but it is diluted by the pre-break months it
    is still averaging over, and the gap is large.
    """
    literal = _evidence_after_break(0.0)
    split = _evidence_after_break(0.5)
    assert split > 2.0, split
    assert split > 2.0 * literal, (split, literal)


# --------------------------------------------------------------------------
# 6. the binned map: does it absorb curvature the affine one leaves behind?
# --------------------------------------------------------------------------
def test_binned_map_recovers_a_curved_relation():
    rng = np.random.default_rng(40)
    z = rng.uniform(-1.0, 1.0, size=20_000)
    x = 2.0 * z ** 2 + rng.normal(scale=0.3, size=z.size)      # pure curvature
    aff = fit_loc_scale_map(z, x, scale_model="const")
    binned = fit_binned_map(z, x, scale_model="const")
    # the affine fit leaves a systematic residual at the edges of the range
    for lo, hi in ((-1.0, -0.7), (-0.15, 0.15), (0.7, 1.0)):
        sel = (z >= lo) & (z < hi)
        assert abs(np.mean(aff.residual(x, z)[sel])) > 0.5, (lo, hi)
        assert abs(np.mean(binned.residual(x, z)[sel])) < 0.15, (lo, hi)


def test_binned_map_is_still_measurable_and_bounded():
    rng = np.random.default_rng(41)
    X, Y, ids, _ = _panel("fixed_effect", rng)
    base = residualize_item(_item(X, Y, ids, 20), mode="y", map_form="binned")
    it = _item(X, Y, ids, 20)
    it["x"] = it["x"] + 5.0
    pert = residualize_item(it, mode="y", map_form="binned")
    assert np.allclose(base["map"].means, pert["map"].means)
    assert np.allclose(base["map"].sds, pert["map"].sds)
    s = base["map"].scale(np.linspace(-10, 10, 101))
    assert s.min() > 0 and np.all(np.isfinite(s))


def test_binned_map_falls_back_on_a_thin_reference():
    rng = np.random.default_rng(42)
    m = fit_binned_map(rng.uniform(size=10), rng.normal(size=10))
    assert m.fallback
    assert np.all(np.isfinite(m.residual(np.zeros(3), np.zeros(3))))


# --------------------------------------------------------------------------
# 7. cross-sectional standardisation is free, because it is symmetric
# --------------------------------------------------------------------------
def test_cross_section_scale_is_permutation_invariant():
    rng = np.random.default_rng(50)
    v = rng.normal(size=60)
    s0 = cross_section_scale(v)
    for _ in range(20):
        assert cross_section_scale(rng.permutation(v)) == s0


def test_cross_scale_commutes_with_relabelling():
    """The whole justification in one line: standardising by a symmetric
    statistic and then permuting is the same as permuting and then
    standardising, so an exchangeable cross-section stays exchangeable."""
    rng = np.random.default_rng(51)
    u = rng.normal(size=40)
    for _ in range(20):
        p = rng.permutation(40)
        assert np.allclose((u / cross_section_scale(u))[p],
                           u[p] / cross_section_scale(u[p]))


def test_cross_scale_normalises_a_dispersion_shock():
    """A month whose dispersion is ten times the window's is put back on the
    window's footing; without it the candidate has to model the shock."""
    rng = np.random.default_rng(52)
    X, Y, ids, _ = _panel("y_null", rng, months=40)
    X[30] = X[30] * 10.0
    X[30] -= X[30].mean()
    plain = residualize_item(_item(X, Y, ids, 30, K=24), mode="y",
                             reference_share=0.5)
    scaled = residualize_item(_item(X, Y, ids, 30, K=24), mode="y",
                              reference_share=0.5, cross_scale="mad")
    assert np.std(plain["x"]) > 5.0, np.std(plain["x"])
    assert 0.5 < np.std(scaled["x"]) < 2.0, np.std(scaled["x"])
    # and the RANKS, which are what the permutation test sees, are untouched
    assert np.array_equal(np.argsort(plain["x"]), np.argsort(scaled["x"]))


def test_cross_scale_keeps_the_e_value_valid():
    for kind, mode in (("y_null", "y"), ("self_null", "self")):
        log_e, ranks = [], []
        for rep in range(30):
            rng = np.random.default_rng(60 + rep)
            X, Y, ids, _ = _panel(kind, rng, months=18)
            for t in range(12, 18):
                it = residualize_item(_item(X, Y, ids, t), mode=mode,
                                      cross_scale="mad")
                if it.get("degenerate"):
                    continue
                cand = build_candidate(it["x"], it["y_lag"], it["bond_ids"],
                                       it["window_x_by_bond"],
                                       it["pooled_window_x"],
                                       it["window_month_list"],
                                       max_block_size=10)
                if cand is None:
                    continue
                le, r = mc_permutation_log_evalue(
                    *evaluate_date(cand, it["x"], 99, rng))
                log_e.append(le)
                ranks.append(r / 100.0)
        u = np.asarray(ranks)
        se = float(u.std(ddof=1) / np.sqrt(u.size))
        assert float(np.mean(log_e)) <= 0.05, (kind, np.mean(log_e))
        assert abs(u.mean() - 0.5) <= 3.5 * se, (kind, u.mean(), se)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all residual-target tests passed")
