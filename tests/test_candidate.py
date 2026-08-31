"""Tests of the block candidate: destination-PIT correctness, measurability,
and the degenerate-cohort convention (thesis Sections 3.3.1, 3.4.3).

The destination-vs-origin test reproduces the direction of thesis Table 3.2:
scoring permuted arrangements under the transform of the block a value came
FROM (origin) favours the identity systematically on exchangeable data, while
scoring at the DESTINATION holds E[E_t] near one.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate import build_candidate, evaluate_date  # noqa: E402
from src.evalue import mc_permutation_log_evalue  # noqa: E402
from src.marginals import gamma_cdf_clipped, gamma_logpdf_capped  # noqa: E402
from src.frank_copula import frank_logpdf  # noqa: E402


def _null_panel(rng, n=24, months=12, block_sep=1.5):
    """Exchangeable current cross-section; the WINDOW carries block-separated
    marginals (blocks by lagged Y) and a common monthly factor, so the fitted
    thetas are positive — the configuration that arms the origin-PIT defect."""
    bond_ids = np.array([f"b{i}" for i in range(n)])
    y_lag = np.linspace(0, 1, n) + rng.normal(scale=1e-3, size=n)
    window_x_by_bond = {}
    month_list = list(range(100, 100 + months))
    common = {m: rng.normal() for m in month_list}       # within-month dependence
    for i, b in enumerate(bond_ids):
        shift = block_sep if i >= n // 2 else 0.0        # two historical populations
        window_x_by_bond[b] = {
            m: float(6.0 + shift + common[m] + 0.6 * rng.normal()) for m in month_list
        }
    pooled = np.array([v for w in window_x_by_bond.values() for v in w.values()])
    # current month: identical law for every position => exchangeable null
    x_current = 6.0 + 0.5 * block_sep + rng.normal() + 0.6 * rng.normal(size=n)
    return x_current, y_lag, bond_ids, window_x_by_bond, pooled, month_list


def _origin_pit_log_evalue(cand, x, n_perms, rng):
    """The defective estimator: pseudo-observations computed once under each
    value's ORIGINAL block, then permuted across blocks (marginals dropped),
    as in the legacy notebooks."""
    n = cand.n
    U = np.empty(n)
    block_of_position = np.empty(n, dtype=int)
    for b, (pos, gp) in enumerate(zip(cand.block_positions, cand.gamma_params)):
        U[pos] = gamma_cdf_clipped(x[pos], gp)
        block_of_position[pos] = b
    def score(arr):
        s = 0.0
        for b, pos in enumerate(cand.block_positions):
            s += float(frank_logpdf(U[arr[pos]][None, :], cand.thetas[b])[0])
        return s
    ident = np.arange(n)
    ll_orig = score(ident)
    ll_perms = np.array([score(rng.permutation(n)) for _ in range(n_perms)])
    return mc_permutation_log_evalue(ll_orig, ll_perms)[0]


def test_destination_pit_holds_size_origin_pit_does_not():
    rng = np.random.default_rng(7)
    N, reps = 19, 250
    e_dest, e_orig = [], []
    for _ in range(reps):
        x, y, ids, w, pooled, months = _null_panel(rng)
        cand = build_candidate(x, y, ids, w, pooled, months, max_block_size=12)
        assert cand is not None
        ll_orig, ll_perms = evaluate_date(cand, x, N, rng)
        e_dest.append(np.exp(mc_permutation_log_evalue(ll_orig, ll_perms)[0]))
        e_orig.append(np.exp(_origin_pit_log_evalue(cand, x, N, rng)))
    e_dest, e_orig = np.array(e_dest), np.array(e_orig)
    se_d = e_dest.std() / np.sqrt(reps)
    assert e_dest.mean() <= 1.0 + 3 * se_d, (e_dest.mean(), se_d)
    # the origin convention manufactures evidence out of the historical block
    # separation on data whose current cross-section is exchangeable
    assert e_orig.mean() > 2.0, e_orig.mean()


def test_candidate_is_window_measurable():
    """Perturbing the current cross-section must not change the fitted
    candidate (blocks, marginals, thetas) — only the evaluation."""
    rng = np.random.default_rng(11)
    x, y, ids, w, pooled, months = _null_panel(rng)
    c1 = build_candidate(x, y, ids, w, pooled, months)
    c2 = build_candidate(x + rng.normal(scale=5.0, size=x.size), y, ids, w, pooled, months)
    assert all(np.array_equal(p1, p2) for p1, p2 in
               zip(c1.block_positions, c2.block_positions))
    assert np.allclose(c1.thetas, c2.thetas)
    assert all(np.allclose(g1, g2) for g1, g2 in zip(c1.gamma_params, c2.gamma_params))


def test_degenerate_cohort_declines_to_bet():
    rng = np.random.default_rng(13)
    x, y, ids, w, pooled, months = _null_panel(rng, n=3)
    assert build_candidate(x, y, ids, w, pooled, months) is None


def test_marginal_channel_carries_power():
    """A current cross-section whose block structure matches the window's
    separated marginals should be strongly favoured over its relabelings."""
    rng = np.random.default_rng(17)
    vals = []
    for _ in range(40):
        x, y, ids, w, pooled, months = _null_panel(rng, block_sep=2.0)
        n = x.size
        # re-impose the historical block structure on the current month
        x_alt = 6.0 + 0.6 * rng.normal(size=n)
        x_alt[np.argsort(y)[n // 2:]] += 2.0
        cand = build_candidate(x_alt, y, ids, w, pooled, months, max_block_size=12)
        ll_orig, ll_perms = evaluate_date(cand, x_alt, 19, rng)
        vals.append(mc_permutation_log_evalue(ll_orig, ll_perms)[0])
    assert np.median(vals) > np.log(10.0), np.median(vals)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all candidate tests passed")
