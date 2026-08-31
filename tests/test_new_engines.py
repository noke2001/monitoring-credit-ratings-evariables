"""Validity of the two engines added in Section 4.4: AHZ Grenander and plug-in Kelly.

The dead-zone bet is exactly valid by Lemma B.3.  These two are not covered by
that lemma -- one learns the density, the other learns the stake -- so each
needs its own certificate:

    Grenander     f_{t-1} is a Lebesgue density built only from the past, so
                  E[f_{t-1}(Z_t) | F_{t-1}] = int_0^1 f_{t-1} = 1 whatever its
                  shape.  The check below is that the estimator really is a
                  density (integrates to exactly 1) at every sample size.
    plug-in Kelly lambda_hat_t is F_{t-1}-measurable, so Lemma B.3(i) applies
                  verbatim.  The check is that it is genuinely predictable and
                  that it starts at exactly zero.

    python tests/test_new_engines.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.betting import grenander_e, grenander_increasing, kelly_lambda  # noqa: E402
from src.sequential import (AdaptiveKellyEProcess, GrenanderEProcess,  # noqa: E402
                            KellyMixtureEProcess)
from src.validation import simulate_null_panel, ville_false_alarm_rate  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


rng = np.random.default_rng(0)

print("1. the Grenander estimate is a density -- integrates to EXACTLY 1")
worst = 0.0
for n in (1, 2, 3, 5, 20, 100, 1000, 5000):
    for _ in range(25):
        k, h = grenander_increasing(rng.random(n))
        worst = max(worst, abs(float(np.sum(np.diff(k) * h)) - 1.0))
check("integral = 1 at every sample size", worst < 1e-12,
      f"max |integral - 1| = {worst:.2e}")
check("empty past gives the uniform density (stake nothing)",
      np.allclose(grenander_increasing(np.array([]))[1], [1.0]))

print("\n2. it is non-decreasing, and finds a genuinely increasing density")
for n in (50, 500, 5000):
    k, h = grenander_increasing(rng.random(n))
    check(f"n={n}: non-decreasing on uniform data", bool(np.all(np.diff(h) >= -1e-12)))
k, h = grenander_increasing(rng.beta(2, 1, 4000))
check("Beta(2,1): the fitted density slopes up", h[-1] > 3 * max(h[0], 1e-9),
      f"{h[0]:.3f} -> {h[-1]:.3f}")

print("\n3. E[e] = 1 under the null, by Monte Carlo through the estimator")
for t in (10, 40, 120):
    es = []
    for _ in range(4000):
        past = rng.random(t)
        es.append(grenander_e(rng.random(), past, "upper", regularise=False))
    es = np.array(es); se = es.std() / np.sqrt(len(es))
    check(f"past length {t}: E[e] = 1 within 3 s.e.", abs(es.mean() - 1) < 3 * se,
          f"{es.mean():.4f} +- {3*se:.4f}")

print("\n4. the shrinkage is a convex combination with the fair bet")
past = rng.random(30)
raw = grenander_e(0.99, past, "upper", regularise=False)
reg = grenander_e(0.99, past, "upper", regularise=True)
check("regularised value lies between 1 and the raw one",
      min(1.0, raw) - 1e-12 <= reg <= max(1.0, raw) + 1e-12,
      f"raw {raw:.4f} -> reg {reg:.4f}")
check("shrinkage cannot produce a negative e", reg >= 0.0)

print("\n5. the plug-in stake is predictable and starts at exactly zero")
eng = AdaptiveKellyEProcess(alpha=0.10, delta=0.75, tail="upper")
# Own generator: this block must not depend on how much randomness the blocks
# above happened to consume.
z = np.random.default_rng(11).random(60)
lam = eng._lam_for_group(z)
check("lambda_0 = 0 exactly (no data, so no bet)", lam[0] == 0.0)
# Perturb ONE observation, and make sure the perturbation actually crosses
# delta -- otherwise the hit count is unchanged and the test proves nothing.
z2 = z.copy(); z2[30] = 0.01 if z[30] > 0.75 else 0.99
lam2 = eng._lam_for_group(z2)
check("lambda_t depends only on the past, never on z_t",
      np.array_equal(lam[:31], lam2[:31]) and not np.array_equal(lam[31:], lam2[31:]),
      "changing z_30 leaves lambda_0..lambda_30 untouched")
check("lambda stays inside [0, cap]", lam.min() >= 0.0 and lam.max() <= 0.95)
check("kelly_lambda agrees with Lemma B.3(iv)",
      abs(kelly_lambda(0.5, 0.75) - (0.5 - 0.5 * (0.25 / 0.75))) < 1e-12)

print("\n6. Ville's inequality on a 3,000 x 120 pure-null panel (alpha = 0.10)")
panel = simulate_null_panel(n_bonds=3000, n_months=120, seed=0)
for name, e in [
    ("Grenander burn=6", GrenanderEProcess(0.10, tail="upper", burn_in=6)),
    ("Grenander burn=12", GrenanderEProcess(0.10, tail="upper", burn_in=12)),
    ("plug-in Kelly", AdaptiveKellyEProcess(0.10, delta=0.75, tail="upper")),
    ("Kelly x mixture", KellyMixtureEProcess(0.10, horizon=24, tail="upper")),
]:
    res = e.run_sequential_test(panel, z_col="pit", cooldown_months=12)
    r = ville_false_alarm_rate(res, 0.10, name)
    check(f"{name}: false-alarm rate <= alpha",
          r["false_alarm_rate"] <= 0.10,
          f"{r['false_alarm_rate']:.4f}   mean e_t {res['e_step'].mean():.4f}")

print()
if FAIL:
    print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
    sys.exit(1)
print("all new-engine checks passed")
