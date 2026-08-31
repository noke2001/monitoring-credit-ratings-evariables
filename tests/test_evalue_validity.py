"""Validity of the Monte Carlo permutation e-value (thesis Lemma 3.14).

These tests simulate the null and verify the guarantees the thesis claims:
  * E[E_t | null] <= 1 for the identity-adjoined estimator (Lemma 3.14),
  * the legacy estimator (identity omitted) violates the bound (Section 3.4.3),
  * the pointwise cap E_t <= N+1,
  * uniformity of the permutation rank R_t (Proposition 3.5),
  * exactness of the closed-form legacy -> corrected transform.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evalue import (  # noqa: E402
    mc_permutation_log_evalue, legacy_log_evalue, legacy_to_corrected,
)


def _heavy_tailed_score(x: np.ndarray) -> float:
    """A deliberately spiky non-symmetric score: rewards alignment between
    position and value rank, exponentiated so the permutation distribution of
    the score is heavy-tailed (the regime where the legacy bias explodes)."""
    n = x.size
    weights = np.linspace(-2.0, 2.0, n)
    return float(np.exp(4.0 * np.corrcoef(weights, x)[0, 1]) * n)


def _simulate(n=12, N=19, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    log_e, log_e_legacy, ranks = [], [], []
    for _ in range(reps):
        x = rng.normal(size=n)                    # i.i.d. => exchangeable null
        ll_orig = np.log(_heavy_tailed_score(x))
        ll_perms = np.array([np.log(_heavy_tailed_score(x[rng.permutation(n)]))
                             for _ in range(N)])
        le, r = mc_permutation_log_evalue(ll_orig, ll_perms)
        log_e.append(le)
        log_e_legacy.append(legacy_log_evalue(ll_orig, ll_perms))
        ranks.append(r)
    return np.array(log_e), np.array(log_e_legacy), np.array(ranks), N


def test_corrected_mean_at_most_one():
    log_e, _, _, N = _simulate()
    e = np.exp(log_e)
    mean, se = e.mean(), e.std() / np.sqrt(e.size)
    assert mean <= 1.0 + 3 * se, (mean, se)
    assert np.all(e <= N + 1 + 1e-9)              # pointwise cap of Lemma 3.14


def test_legacy_mean_exceeds_one():
    _, log_e_legacy, _, _ = _simulate()
    e = np.exp(log_e_legacy)
    se = e.std() / np.sqrt(e.size)
    assert e.mean() > 1.0 + 3 * se, (e.mean(), se)


def test_rank_uniformity():
    _, _, ranks, N = _simulate()
    counts = np.bincount(ranks, minlength=N + 2)[1:]
    expected = ranks.size / (N + 1)
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    # chi-square with N degrees of freedom: mean N, sd sqrt(2N); allow 5 sd
    assert chi2 < N + 5 * np.sqrt(2 * N), chi2


def test_transform_is_exact():
    rng = np.random.default_rng(2)
    for _ in range(200):
        M = int(rng.integers(5, 2000))
        ll_perms = rng.normal(scale=rng.uniform(0.5, 60.0), size=M)
        ll_orig = float(rng.normal(scale=60.0))
        direct, _ = mc_permutation_log_evalue(ll_orig, ll_perms)
        via_transform = legacy_to_corrected(legacy_log_evalue(ll_orig, ll_perms), M)
        assert abs(direct - float(via_transform)) < 1e-9


def test_transform_extremes():
    # a legacy log-e-value of +283 (observed in the August notebooks) maps to
    # just under the cap log(M+1); strongly negative values are barely moved
    assert abs(float(legacy_to_corrected(283.0, 500)) - np.log(501.0)) < 1e-6
    assert abs(float(legacy_to_corrected(-53.0, 500)) - (-53.0 + np.log(501.0 / 500.0))) < 1e-9


def test_ville_false_alarm_rate():
    """Product over 24 dates, threshold 1/alpha: alarms under the null must be
    rare (Ville: P <= alpha). Uses a fresh candidate draw each date."""
    rng = np.random.default_rng(3)
    alpha, n, N, T, reps = 0.05, 10, 19, 24, 400
    alarms = 0
    for _ in range(reps):
        log_m = 0.0
        for _t in range(T):
            x = rng.normal(size=n)
            ll_orig = np.log(_heavy_tailed_score(x))
            ll_perms = np.array([np.log(_heavy_tailed_score(x[rng.permutation(n)]))
                                 for _ in range(N)])
            le, _ = mc_permutation_log_evalue(ll_orig, ll_perms)
            log_m += le
            if log_m >= np.log(1 / alpha):
                alarms += 1
                break
    rate = alarms / reps
    assert rate <= alpha + 3 * np.sqrt(alpha * (1 - alpha) / reps), rate


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all e-value validity tests passed")
