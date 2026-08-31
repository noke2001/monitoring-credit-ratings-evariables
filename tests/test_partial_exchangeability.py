"""Tests for the two SIC designs.

(a) categorical blocks: the partition is the label grouping itself.
(b) partial exchangeability: permutations restricted to within-group, which
    requires the candidate to be partitioned on something OTHER than the
    grouping variable -- otherwise q_t is invariant under the group action and
    E_t == 1 identically.  That degeneracy is silent and produces a
    perfectly flat E-process, so it is worth a test.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate import build_candidate, evaluate_date, sample_permutation  # noqa: E402
from src.evalue import mc_permutation_log_evalue  # noqa: E402


def _panel(rng, n=24, months=12, sep=1.5, n_sectors=3):
    ids = np.array([f"b{i}" for i in range(n)])
    y_lag = np.linspace(0, 1, n) + rng.normal(scale=1e-3, size=n)
    sectors = np.array([f"S{i % n_sectors}" for i in range(n)], dtype=object)
    ml = list(range(100, 100 + months))
    common = {m: rng.normal() for m in ml}
    wxb = {b: {m: float(6.0 + (sep if i >= n // 2 else 0.0) + common[m]
                        + 0.6 * rng.normal()) for m in ml}
           for i, b in enumerate(ids)}
    pooled = np.array([v for w in wxb.values() for v in w.values()])
    x = 6.0 + 0.5 * sep + rng.normal() + 0.6 * rng.normal(size=n)
    return x, y_lag, sectors, ids, wxb, pooled, ml


def test_within_group_permutation_preserves_groups():
    rng = np.random.default_rng(0)
    groups = np.array([f"S{i % 4}" for i in range(40)], dtype=object)
    for _ in range(50):
        perm = sample_permutation(40, rng, groups)
        assert sorted(perm) == list(range(40))          # a genuine permutation
        assert np.all(groups[perm] == groups)           # never leaves its group


def test_partitioning_on_the_grouping_variable_is_degenerate():
    """The trap: blocks = sectors AND permutations within sectors => E_t == 1."""
    rng = np.random.default_rng(1)
    x, y, sec, ids, wxb, pooled, ml = _panel(rng)
    cand = build_candidate(x, y, ids, wxb, pooled, ml, block_labels=sec)
    assert cand is not None
    ll_orig, ll_perms = evaluate_date(cand, x, 40, rng, groups=sec)
    assert np.allclose(ll_perms, ll_orig, atol=1e-9), \
        "scores must be identical when the group cannot move a value between blocks"
    log_e, _ = mc_permutation_log_evalue(ll_orig, ll_perms)
    assert abs(log_e) < 1e-9, log_e


def test_partial_exchangeability_is_non_degenerate_with_a_different_partition():
    """Blocks from lagged Y, permutations within sector: q_t does vary."""
    rng = np.random.default_rng(2)
    x, y, sec, ids, wxb, pooled, ml = _panel(rng)
    cand = build_candidate(x, y, ids, wxb, pooled, ml, max_block_size=8)
    ll_orig, ll_perms = evaluate_date(cand, x, 40, rng, groups=sec)
    assert np.std(ll_perms) > 1e-6, "within-sector relabelling must move q_t"


def test_partial_exchangeability_holds_size():
    """Under a within-group-exchangeable null, E[E_t] <= 1."""
    rng = np.random.default_rng(3)
    vals = []
    for _ in range(250):
        x, y, sec, ids, wxb, pooled, ml = _panel(rng)
        cand = build_candidate(x, y, ids, wxb, pooled, ml, max_block_size=8)
        if cand is None:
            continue
        ll_orig, ll_perms = evaluate_date(cand, x, 19, rng, groups=sec)
        vals.append(np.exp(mc_permutation_log_evalue(ll_orig, ll_perms)[0]))
    e = np.array(vals)
    se = e.std(ddof=1) / np.sqrt(e.size)
    assert e.mean() <= 1.0 + 3 * se, (e.mean(), se)


def test_categorical_blocks_group_by_label():
    rng = np.random.default_rng(4)
    x, y, sec, ids, wxb, pooled, ml = _panel(rng, n=30, n_sectors=3)
    cand = build_candidate(x, y, ids, wxb, pooled, ml, block_labels=sec)
    assert cand is not None
    assert len(cand.block_positions) == 3
    for pos in cand.block_positions:
        assert len({sec[i] for i in pos}) == 1        # one sector per block


def test_unlabelled_positions_are_pooled_not_dropped():
    rng = np.random.default_rng(5)
    x, y, sec, ids, wxb, pooled, ml = _panel(rng, n=30, n_sectors=2)
    sec = sec.copy()
    sec[:6] = None                                     # unmapped SIC codes
    cand = build_candidate(x, y, ids, wxb, pooled, ml, block_labels=sec)
    assert cand is not None
    covered = np.concatenate(cand.block_positions)
    assert len(covered) == len(set(covered.tolist())) == 30, "every position blocked once"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all partial-exchangeability tests passed")
