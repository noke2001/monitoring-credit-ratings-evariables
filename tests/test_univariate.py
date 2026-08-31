"""Certification of the Section 4.2 PIT construction.

Every claim the thesis text makes about eq. (4.1)/(4.3) is checked here
against either a closed form or a Monte Carlo draw from the null.  Exits
non-zero on failure.

    python tests/test_univariate.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.univariate import (auc, counterfactual_cohort, kde_cdf_pit,  # noqa: E402
                            pit_persistence, rank_pit, transition_labels)

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def _panel(n_bonds=60, n_months=40, seed=0, phi=0.0):
    """A pure-null panel: one cohort per month, bonds exchangeable within it.

    ``phi`` injects AR(1) persistence into each bond's latent level so the
    persistence diagnostic can be tested against a known answer.
    """
    rng = np.random.default_rng(seed)
    lat = np.zeros((n_bonds, n_months))
    lat[:, 0] = rng.normal(size=n_bonds)
    for t in range(1, n_months):
        lat[:, t] = phi * lat[:, t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(size=n_bonds)
    rows = []
    for t in range(n_months):
        for i in range(n_bonds):
            rows.append({"dates": 200001 + (t // 12) * 100 + (t % 12),
                         "isin": f"B{i:04d}", "nrtg": 5, "rtg": "A",
                         "x": float(lat[i, t])})
    df = pl.DataFrame(rows)
    return df.with_columns((((pl.col("dates") // 100) * 12) +
                            (pl.col("dates") % 100)).alias("midx")
                           ).sort(["isin", "dates"])


print("1. the lattice: eq. (4.3) is uniform on {1/n,...,1}, not on [0,1]")
for n in (5, 20, 82):
    df = _panel(n_bonds=n, n_months=200, seed=1)
    z = rank_pit(df, "x")["Z_x"].to_numpy()
    lattice = np.allclose(np.unique(np.round(z * n)), np.arange(1, n + 1))
    check(f"n={n}: support is the lattice", lattice)
    check(f"n={n}: E[Z] = (n+1)/2n = {(n+1)/(2*n):.4f}",
          abs(z.mean() - (n + 1) / (2 * n)) < 3e-3, f"observed {z.mean():.4f}")
    check(f"n={n}: P(Z=1) = 1/n = {1/n:.4f}",
          abs((z >= 1 - 1e-12).mean() - 1 / n) < 3e-3,
          f"observed {(z >= 1 - 1e-12).mean():.4f}")

print("\n2. the randomised PIT is exactly U([0,1])")
for n in (5, 20, 82):
    df = _panel(n_bonds=n, n_months=400, seed=2)
    z = rank_pit(df, "x", randomize=True, seed=7)["Z_x"].to_numpy()
    from scipy.stats import kstest
    p = kstest(z, "uniform").pvalue
    check(f"n={n}: KS vs U([0,1]) not rejected", p > 0.01, f"p={p:.3f}")
    check(f"n={n}: E[Z] = 1/2", abs(z.mean() - 0.5) < 5e-3, f"observed {z.mean():.4f}")

print("\n2b. Lemma B.x, the conditional step: uniformity given a FIXED multiset")
# This is the step the proof turns on; everything else is a tower-property
# argument.  Fixing the multiset is the sharpest possible check -- no averaging
# over multisets can rescue a formula that fails here.  The three competing
# constructions are included to show the tie multiplier tau is load-bearing.


def _pit_variants(gamma, u):
    n = len(gamma)
    lo = (gamma[:, None] < gamma[None, :]).sum(0)       # r^-
    hi = (gamma[:, None] <= gamma[None, :]).sum(0)      # r^+
    return {"randomised": (lo + u * (hi - lo)) / n,     # the shipped formula
            "unrandomised": hi / n,
            "midrank": (lo + hi) / (2 * n),
            "no tie factor": (lo + u) / n}


def _kolmogorov(z):
    z = np.sort(z); m = len(z); k = np.arange(1, m + 1)
    return max(np.abs(k / m - z).max(), np.abs(z - (k - 1) / m).max())


_MULTISETS = {
    "no ties, n=7": np.arange(7.0),
    "unequal ties, n=10": np.array([0., 0, 0, 0, 1, 1, 1, 2, 3, 3]),
    "single value, n=6": np.full(6, 5.0),
    "singleton, n=1": np.array([4.0]),
    "skewed, n=12": np.array([0.] * 9 + [1., 2., 3.]),
}
_REPS = 200_000
_rng = np.random.default_rng(11)
_tol = 4 * 1.36 / np.sqrt(_REPS)                        # ~4 sigma of MC noise
for _name, _base in _MULTISETS.items():
    _n = len(_base)
    _acc = {k: np.empty(_REPS) for k in
            _pit_variants(np.array([0., 1]), np.array([.5, .5]))}
    for _b in range(_REPS):
        _p = _pit_variants(_rng.permutation(_base), _rng.random(_n))
        for _k in _acc:
            _acc[_k][_b] = _p[_k][0]
    _d = {k: _kolmogorov(v) for k, v in _acc.items()}
    _tied = len(np.unique(_base)) < _n
    check(f"{_name}: randomised PIT is U([0,1])", _d["randomised"] < _tol,
          f"Kolmogorov {_d['randomised']:.4f} < {_tol:.4f}")
    # The deterministic constructions fail on every multiset.  Dropping the
    # tie multiplier tau is harmless when tau == 1 everywhere, and fatal
    # otherwise -- which is exactly the claim that tau is load-bearing.
    # A construction supported on a lattice of spacing 1/n cannot be closer to
    # U([0,1]) than the half-spacing 1/(2n); that is the exact lower bound the
    # deterministic rivals attain in the tie-free case (0.143 = 1/n for the
    # unrandomised PIT at n=7, 0.071 = 1/2n for the midrank).
    _rivals = ["unrandomised", "midrank"] + (["no tie factor"] if _tied else [])
    check(f"{_name}: {'all three' if _tied else 'both deterministic'} "
          f"rival{'s' if _tied else ''} fail",
          min(_d[k] for k in _rivals) > 1 / (2 * _n) - _tol,
          " | ".join(f"{k} {_d[k]:.3f}" for k in _rivals) +
          f"  (lattice bound 1/2n = {1/(2*_n):.3f})")
    if not _tied:
        check(f"{_name}: with no ties the tie factor is inert",
              _d["no tie factor"] < _tol,
              f"Kolmogorov {_d['no tie factor']:.4f}")

print("\n2c. Lemma B.x Steps 2-3: the closed form, checked exactly (not by MC)")
# P(Z <= z | multiset) = (1/n) sum_k min(max(nz - c_k, 0), m_k).  Step 3 claims
# this telescopes to z for every multiset and every z.  That is an algebraic
# identity, so it can be checked to floating-point rather than sampled.
_rng2 = np.random.default_rng(23)
_worst = 0.0
for _trial in range(3000):
    _K = int(_rng2.integers(1, 9))
    _m = _rng2.integers(1, 12, size=_K)                 # multiplicities
    _n = int(_m.sum())
    _c = np.concatenate([[0], np.cumsum(_m)[:-1]])      # counts strictly below
    _zs = np.r_[0.0, 1.0, _rng2.random(40), np.cumsum(_m) / _n]   # incl. knots
    for _z in _zs:
        _lhs = np.minimum(np.maximum(_n * _z - _c, 0.0), _m).sum() / _n
        _worst = max(_worst, abs(_lhs - _z))
check("closed form equals z for 3,000 random multisets x 42 knots",
      _worst < 1e-12, f"max |P(Z<=z) - z| = {_worst:.2e}")

print("\n3. ties: the randomised PIT stays uniform on a discrete covariate")
df = _panel(n_bonds=50, n_months=400, seed=3)
df = df.with_columns((pl.col("x").round(0)).alias("x"))       # heavy ties
z = rank_pit(df, "x", randomize=True, seed=11)["Z_x"].to_numpy()
from scipy.stats import kstest
check("KS vs U([0,1]) not rejected under ties",
      kstest(z, "uniform").pvalue > 0.01,
      f"p={kstest(z, 'uniform').pvalue:.3f}")

print("\n4. the cross-sectional histogram is flat as an identity, not a test")
rng = np.random.default_rng(4)
df = _panel(n_bonds=40, n_months=30, seed=4)
# break exchangeability as hard as possible: one bond is shifted by +100
df = df.with_columns(pl.when(pl.col("isin") == "B0000")
                     .then(pl.col("x") + 100.0).otherwise(pl.col("x")).alias("x"))
z = rank_pit(df, "x")
per_date = z.group_by("dates").agg(pl.col("Z_x").mean().alias("m"))["m"].to_numpy()
check("pooled per-date mean PIT is (n+1)/2n regardless of the violation",
      np.allclose(per_date, 41 / 80), f"observed {per_date[0]:.4f}")
one = z.filter(pl.col("isin") == "B0000")["Z_x"].to_numpy()
check("the single bond's own PIT does detect it", np.allclose(one, 1.0))

print("\n5. rank invariance to strictly increasing transforms")
df = _panel(n_bonds=40, n_months=20, seed=5)
a = rank_pit(df, "x")["Z_x"].to_numpy()
df2 = df.with_columns((pl.col("x") * 3.0 + 7.0).alias("x"))
b = rank_pit(df2, "x")["Z_x"].to_numpy()
check("affine rescale leaves the PIT unchanged", np.array_equal(a, b))
df3 = df.with_columns((pl.col("x") - pl.col("x").mean().over(["dates", "nrtg"])).alias("x"))
c = rank_pit(df3, "x")["Z_x"].to_numpy()
check("Chapter 2 mean-correction leaves the PIT unchanged", np.array_equal(a, c))
df4 = df.with_columns(pl.col("x").exp().alias("x"))
d = rank_pit(df4, "x")["Z_x"].to_numpy()
check("any strictly increasing map leaves the PIT unchanged", np.array_equal(a, d))

print("\n6. persistence diagnostic recovers a known AR(1)")
for phi in (0.0, 0.5, 0.9):
    df = _panel(n_bonds=80, n_months=180, seed=6, phi=phi)
    r = pit_persistence(rank_pit(df, "x"), "x")
    # the rank PIT of a Gaussian AR(1) has ACF(1) = (6/pi) arcsin(phi/2)
    expect = (6 / np.pi) * np.arcsin(phi / 2)
    check(f"phi={phi}: ACF(1) ~ (6/pi)arcsin(phi/2) = {expect:.3f}",
          abs(r[1] - expect) < 0.04, f"observed {r[1]:.3f}")

print("\n7. gaps are not paired across")
df = _panel(n_bonds=30, n_months=24, seed=7, phi=0.9)
full = pit_persistence(rank_pit(df, "x"), "x")["n_pairs"]
holed = df.filter(pl.col("midx") % 4 != 0)          # punch out every 4th month
holed_pairs = pit_persistence(rank_pit(holed, "x"), "x")["n_pairs"]
check("removing every 4th month removes those lag-1 pairs",
      holed_pairs < full * 0.75, f"{holed_pairs} vs {full}")

print("\n8. transition labels")
rows = [{"dates": 200001 + m, "isin": "A", "nrtg": 5 if m < 6 else 6, "rtg": "A"}
        for m in range(12)]
t = pl.DataFrame(rows).with_columns(
    (((pl.col("dates") // 100) * 12) + (pl.col("dates") % 100)).alias("midx"))
dn, up, obs = transition_labels(t, horizon=3)
check("downgrade flagged exactly in the 3 months before the move",
      list(np.where(dn)[0]) == [3, 4, 5], f"flagged {list(np.where(dn)[0])}")
check("no upgrade flagged", not up.any())
check("last row unobservable at H=3", not obs[-1])

print("\n9. AUC handles ties and matches a closed form")
sc = np.array([1.0, 2, 3, 4]); lb = np.array([False, False, True, True])
check("perfect separation -> 1.0", abs(auc(sc, lb) - 1.0) < 1e-12)
sc = np.array([1.0, 1, 1, 1])
check("all tied -> 0.5", abs(auc(sc, lb) - 0.5) < 1e-12)

print("\n10. the counterfactual cohort is the *most recent* previous rating")
rows = [{"dates": 200001 + m, "isin": "A", "rtg": "A",
         "nrtg": 4 if m < 3 else (5 if m < 7 else 6)} for m in range(10)]
t = pl.DataFrame(rows)
cf = counterfactual_cohort(t, "A")["prev_nrtg"].to_list()
check("undefined before the first transition", cf[:3] == [None, None, None])
check("= 4 between the first and second transition", cf[3:7] == [4, 4, 4, 4])
check("= 5 after the second, NOT 4", cf[7:] == [5, 5, 5],
      "so 'had it not changed rating' means 'had the *last* change not happened'")

print("\n11. the KDE CDF is not eq. (4.3)")
rng = np.random.default_rng(12)
v = rng.lognormal(0, 1.5, 200)                      # heavy tail, like me/sales
ecdf = np.array([np.mean(v <= x) for x in v])
kde = kde_cdf_pit(v, v)
check("KDE and ECDF differ materially on a heavy tail",
      np.abs(ecdf - kde).max() > 0.05,
      f"max |diff| = {np.abs(ecdf - kde).max():.3f}")

print()
if FAIL:
    print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
    sys.exit(1)
print("all univariate PIT checks passed")
