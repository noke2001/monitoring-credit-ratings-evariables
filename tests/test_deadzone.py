"""Certification of the dead-zone betting function — thesis Lemma B.3.

Arnold, Henzi and Ziegel characterise *which* densities give a valid E-value
for a stochastic-dominance alternative (any non-decreasing one), but they
estimate that density sequentially and do not analyse a fixed parametric bet.
The dead-zone family this thesis uses is such a bet, and Lemma B.3 supplies
the four properties Section 4.4 relies on:

    (i)   exact validity, E[e] = 1 under the null, for every (delta, lambda)
    (ii)  the bounds 1 - lambda <= e <= 1 + lambda*delta/(1-delta)
    (iii) admissibility: f_delta is a non-decreasing Lebesgue density, so the
          bet is valid for the whole composite H_ST, not just one alternative
    (iv)  the Kelly-optimal stake lambda* = q - (1-q)(1-delta)/delta, which is
          exactly zero under the null and positive precisely when q > 1-delta

    python tests/test_deadzone.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.betting import deadzone_e, deadzone_phi  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


DELTAS = (0.05, 0.25, 0.5, 0.75, 0.9, 0.99)
LAMS = (0.0, 0.1, 0.5, 0.95, 1.0)


def kelly(q, d):
    """Lemma B.3(iv): the growth-optimal stake, clipped to [0,1]."""
    return min(max(q - (1.0 - q) * (1.0 - d) / d, 0.0), 1.0)


print("1. (i) exact validity — closed form over the (delta, lambda) grid")
worst = 0.0
for d in DELTAS:
    for lam in LAMS:
        Ee = d * (1 - lam) + (1 - d) * (1 + lam * d / (1 - d))
        worst = max(worst, abs(Ee - 1.0))
check("E[e] = 1 for every (delta, lambda)", worst < 1e-12,
      f"max |E[e]-1| = {worst:.2e}")

print("\n2. (i) the same identity through the shipped code, by Monte Carlo")
rng = np.random.default_rng(0)
z = rng.random(2_000_000)
for d, lam in ((0.75, 0.5), (0.75, 0.95), (0.9, 0.5), (0.25, 1.0)):
    e = deadzone_e(z, lam=lam, delta=d, tail="upper")
    se = e.std() / np.sqrt(len(e))
    check(f"delta={d}, lam={lam}: E[e] = 1 within 3 s.e.",
          abs(e.mean() - 1.0) < 3 * se, f"{e.mean():.6f} +- {3*se:.6f}")

print("\n3. (ii) the bounds are attained, and e >= 0 requires lambda <= 1")
for d, lam in ((0.75, 0.5), (0.9, 0.95), (0.25, 1.0)):
    e = deadzone_e(z, lam=lam, delta=d, tail="upper")
    lo, hi = 1 - lam, 1 + lam * d / (1 - d)
    check(f"delta={d}, lam={lam}: e in [{lo:.4f}, {hi:.4f}]",
          e.min() >= lo - 1e-12 and e.max() <= hi + 1e-12,
          f"observed [{e.min():.4f}, {e.max():.4f}]")
try:
    deadzone_e(z[:10], lam=1.5, delta=0.75)
    check("lambda > 1 is rejected", False)
except ValueError:
    check("lambda > 1 is rejected", True, "e could go negative; clipping would break E[e]=1")

print("\n4. (iii) f_delta is a non-decreasing Lebesgue density")
for d in (0.25, 0.75, 0.9):
    zz = np.linspace(0.0, 1.0, 1_000_001)
    f = np.where(zz > d, 1.0 / (1.0 - d), 0.0)
    integral = np.trapezoid(f, zz)
    check(f"delta={d}: integrates to 1 and is non-decreasing",
          abs(integral - 1.0) < 1e-5 and bool(np.all(np.diff(f) >= 0)),
          f"integral {integral:.6f}")
# phi is the centred version: e = 1 + lam*phi, so E[phi] must be 0
for d in DELTAS:
    Ephi = d * (-1.0) + (1 - d) * (d / (1 - d))
    check(f"delta={d}: E[phi] = 0", abs(Ephi) < 1e-12, f"{Ephi:+.2e}")

print("\n5. (iv) the Kelly stake matches the numerical optimum")
from scipy.optimize import minimize_scalar  # noqa: E402
worst = 0.0
for d in (0.5, 0.75, 0.9):
    for mult in (1.0, 1.2, 2.0, 3.5):
        q = min(0.98, mult * (1 - d))
        a = 1.0 / (1.0 - d)

        def neg_growth(L, q=q, a=a):
            return -(q * np.log(1 + L * (a - 1))
                     + (1 - q) * np.log(max(1e-300, 1 - L)))
        r = minimize_scalar(neg_growth, bounds=(0.0, 1 - 1e-9),
                            method="bounded", options={"xatol": 1e-13})
        worst = max(worst, abs(kelly(q, d) - r.x))
check("closed form equals argmax of E[log e] over 12 (delta, q) pairs",
      worst < 1e-6, f"max |lam* - numeric| = {worst:.2e}")

print("\n6. (iv) the stake vanishes exactly at the null and only there")
for d in (0.5, 0.75, 0.9):
    q0 = 1 - d
    check(f"delta={d}: lam*(q = 1-delta) = 0 exactly",
          abs(kelly(q0, d)) < 1e-15)
    check(f"delta={d}: lam* > 0 for q just above 1-delta",
          kelly(q0 + 1e-4, d) > 0)
    check(f"delta={d}: lam* = 0 (clipped) for q below 1-delta",
          kelly(q0 - 1e-4, d) == 0.0,
          "a bet against the alternative would be inadmissible")

print("\n7. (iv) growth is positive under the alternative, zero under the null")
for d in (0.5, 0.75, 0.9):
    a = 1.0 / (1.0 - d)
    for q, expect_pos in ((1 - d, False), (1.5 * (1 - d), True)):
        L = kelly(q, d)
        g = q * np.log(1 + L * (a - 1)) + (1 - q) * np.log(1 - L) if L > 0 else 0.0
        check(f"delta={d}, q={q:.3f}: growth {'>' if expect_pos else '='} 0",
              (g > 1e-6) if expect_pos else (abs(g) < 1e-12), f"E[log e] = {g:+.5f}")

print("\n8. predictability — a lambda that depends on the past keeps E[e]=1")
rng2 = np.random.default_rng(7)
z2 = rng2.random(500_000)
lam_pred = np.roll(z2, 1)          # F_{t-1}-measurable: last month's PIT
lam_pred[0] = 0.5
e = deadzone_e(z2, lam=lam_pred, delta=0.75, tail="upper")
se = e.std() / np.sqrt(len(e))
check("E[e] = 1 with a predictable, data-dependent stake",
      abs(e.mean() - 1.0) < 3 * se, f"{e.mean():.6f} +- {3*se:.6f}")
lam_bad = z2                        # NOT predictable: this month's PIT
e_bad = deadzone_e(z2, lam=lam_bad, delta=0.75, tail="upper")
check("a NON-predictable stake breaks it, as it must",
      e_bad.mean() > 1.0 + 10 * (e_bad.std() / np.sqrt(len(e_bad))),
      f"E[e] = {e_bad.mean():.4f} > 1")

print()
if FAIL:
    print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
    sys.exit(1)
print("all dead-zone (Lemma B.3) checks passed")
