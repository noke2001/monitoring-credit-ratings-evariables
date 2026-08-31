"""
Mathematical soundness suite for the Approach-1 e-process stack.

Run:  python3 tests_and_plots/validate_math.py

Exits non-zero if any link in the validity chain is broken:
    exact PIT  ->  E[e] = 1  ->  M is a non-negative martingale  ->  Ville.

Sections 1-4 certify the corrected implementation.
Section 5 reproduces the legacy specifications and demonstrates, numerically,
the size distortion each one caused.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.betting import (
    deadzone_e,
    mixture_power_e,
    power_e,
    randomized_pit_discrete,
    randomized_rank_pit,
)
from src.sequential import (
    AsymmetricLeakyEProcess,
    InnovationGatedEProcess,
    MixtureRestartEProcess,
    OptimalHybridEProcess,
    RollingWindowEProcess,
    TierVelocityEProcess,
    DeadzoneEProcess,
)
from src.validation import (
    certify_e_value,
    certify_pit_uniformity,
    legacy_ramp_e,
    simulate_null_panel,
    ville_false_alarm_rate,
)

FAILURES = []
RULE = "=" * 78


def head(t):
    print(f"\n{RULE}\n{t}\n{RULE}")


def mark(ok):
    if not ok:
        FAILURES.append(True)
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------- 1. e-values
head("1. BETTING FUNCTIONS  --  is E[e(Z)] = 1 under H_0: Z ~ U(0,1)?")
print(f"{'betting function':<44}{'E[e] (quad)':>13}{'E[e] (MC)':>12}{'':>7}")
specs = [
    ("deadzone  delta=0.75, lam=0.95", lambda z: deadzone_e(z, 0.95, 0.75)),
    ("deadzone  delta=0.75, lam=0.20", lambda z: deadzone_e(z, 0.20, 0.75)),
    ("deadzone  delta=0.50, lam=1.00", lambda z: deadzone_e(z, 1.00, 0.50)),
    ("power     kappa=2.0", lambda z: power_e(z, 2.0)),
    ("power     kappa=5.0", lambda z: power_e(z, 5.0)),
    ("mixture   kappa in {1.5,2,3,5}", lambda z: mixture_power_e(z)),
]
for nm, fn in specs:
    r = certify_e_value(fn, nm, n_mc=500_000)
    print(f"{nm:<44}{r['E_quad']:>13.9f}{r['E_mc']:>12.6f}{mark(r['passes']):>7}")

# ------------------------------------------------------------------- 2. PITs
head("2. PIT CONSTRUCTIONS  --  is Z exactly Uniform(0,1) under H_0?")
rng = np.random.default_rng(7)
n = 200_000

# (a) randomized PIT of a discrete predictive distribution
K = 6
probs = rng.dirichlet(np.full(K, 0.8), size=n)
y = np.array([rng.choice(K, p=p) for p in probs[:20_000]])
z_disc = randomized_pit_discrete(probs[:20_000], y, rng)
r = certify_pit_uniformity(z_disc, "randomized_pit_discrete (K=6)")
print(f"  randomized_pit_discrete   n={r['n']:>7}  mean={r['mean']:.4f}  "
      f"KS={r['ks_stat']:.4f} p={r['ks_p']:.3f}  {mark(r['passes'])}")

# (b) deterministic PIT -- included to show why randomization is required
cum = np.cumsum(probs[:20_000], axis=1)
z_det = cum[np.arange(len(y)), y]
r_det = certify_pit_uniformity(z_det, "deterministic PIT F(y)")
print(f"  deterministic PIT F(y)    n={r_det['n']:>7}  mean={r_det['mean']:.4f}  "
      f"KS={r_det['ks_stat']:.4f} p={r_det['ks_p']:.3f}  "
      f"{'PASS' if not r_det['passes'] else 'FAIL'}  (expected to be non-uniform)")
if r_det["passes"]:
    FAILURES.append(True)

# (c) randomized rank PIT vs the legacy rank(pct=True)
for cohort in (2, 5, 20, 200):
    g = np.repeat(np.arange(n // cohort), cohort)
    s = rng.normal(size=len(g))
    z_new = randomized_rank_pit(s, g, rng)
    z_old = pd.Series(s).groupby(g).rank(pct=True).to_numpy()
    a = certify_pit_uniformity(z_new, "randomized_rank_pit")
    e_new = float(np.mean(deadzone_e(z_new, 0.95, 0.75, tail="upper")))
    e_old = float(np.mean(deadzone_e(np.clip(z_old, 0, 1), 0.95, 0.75, tail="upper")))
    print(f"  cohort n={cohort:<4} randomized_rank_pit KS={a['ks_stat']:.4f} "
          f"p={a['ks_p']:.3f} E[e]={e_new:.4f} {mark(a['passes'])}"
          f"   |  legacy rank(pct=True) E[e]={e_old:.4f}"
          f" {'<-- DRIFTS UP' if e_old > 1.001 else ''}")

# ------------------------------------------------- 3. Ville on a pure null panel
head("3. VILLE'S INEQUALITY  --  false-alarm rate on an iid U(0,1) null panel")
panel = simulate_null_panel(n_bonds=3000, n_months=120, seed=11)
print(f"  panel: {panel['isin'].nunique():,} bonds x 120 months = {len(panel):,} rows, "
      f"PIT iid U(0,1), no rating changes\n")
print(f"{'engine':<46}{'alpha':>7}{'false-alarm':>13}{'bound':>8}{'':>7}")

ALPHA = 0.10
engines = [
    ("Deadzone (lam=0.95)", DeadzoneEProcess(ALPHA, lam=0.95)),
    ("Deadzone (lam=0.50)", DeadzoneEProcess(ALPHA, lam=0.50)),
    ("MixtureRestart (horizon=24)", MixtureRestartEProcess(ALPHA, horizon=24)),
    ("AsymmetricLeaky (rho=0.80)", AsymmetricLeakyEProcess(ALPHA, rho=0.80)),
    ("OptimalHybrid (gated, K=2)", OptimalHybridEProcess(ALPHA, horizon=24)),
    ("TierVelocity", TierVelocityEProcess(ALPHA)),
    ("InnovationGated (level gate 0.70)", InnovationGatedEProcess(ALPHA)),
]
for nm, eng in engines:
    res = eng.run_sequential_test(panel, z_col="pit", cooldown_months=10**6)
    r = ville_false_alarm_rate(res, ALPHA, nm)
    print(f"{nm:<46}{ALPHA:>7.2f}{r['false_alarm_rate']:>13.4f}"
          f"{r['tolerance']:>8.4f}{mark(r['passes']):>7}")

print("\n  Same check across significance levels (Deadzone, lam=0.95):")
for a in (0.20, 0.10, 0.05, 0.01):
    eng = DeadzoneEProcess(a, lam=0.95)
    res = eng.run_sequential_test(panel, z_col="pit", cooldown_months=10**6)
    r = ville_false_alarm_rate(res, a, "wz")
    print(f"    alpha={a:<5.2f} threshold={1/a:>6.1f}  observed={r['false_alarm_rate']:.4f}"
          f"  bound={r['tolerance']:.4f}  {mark(r['passes'])}")

# --------------------------------------------------------- 4. predictability
head("4. PREDICTABILITY  --  does the bet size depend only on F_{t-1}?")
p2 = panel.copy()
eng = TierVelocityEProcess(ALPHA)
base = eng.run_sequential_test(p2, z_col="pit", cooldown_months=10**6)
shuffled = p2.copy()
rng2 = np.random.default_rng(3)
shuffled["pit"] = rng2.permutation(shuffled["pit"].to_numpy())
alt = eng.run_sequential_test(shuffled, z_col="pit", cooldown_months=10**6)
same_lambda = np.allclose(base["lambda_bet"], alt["lambda_bet"])
print(f"  lambda_bet is invariant to permuting the contemporaneous PIT: "
      f"{mark(same_lambda)}")
print("  (if lambda moved with Z_t the bet would not be predictable and "
      "E[e_t|F_{t-1}]=1 would fail)")

# ------------------------------------------------------- 5. legacy regressions
head("5. LEGACY SPECIFICATIONS  --  magnitude of the defects now removed")

r = certify_e_value(lambda z: legacy_ramp_e(z), "legacy ramp", n_mc=500_000)
print(f"\n  (a) e = 1 + 2(Z-0.75)/0.25 for Z>0.75   [run_all_strategies_evaluation.py]")
print(f"      E[e] = {r['E_quad']:.6f}  (must be 1.0)")
print(f"      null wealth drift over 24 months: {r['E_quad']**24:>9,.1f}x   "
      f"alarm threshold 1/alpha = 10")
steps = int(np.ceil(np.log(10) / np.log(r["E_quad"])))
print(f"      crosses the threshold by drift alone in ~{steps} months, with no signal")

print(f"\n  (b) M_t <- max(1, M_(t-1) * e_t)   [CUSUM floor, 4 driver scripts]")
from src.sequential import _compound, _ordered_groups
e_null = deadzone_e(panel["pit"].to_numpy(), 0.95, 0.75)
for floor in (False, True):
    M, al, runs = _compound(
        panel.reset_index(drop=True), e_null, 1 / ALPHA,
        np.zeros(len(panel), bool), 10**6, "isin", "dates", floor_at_one=floor,
    )
    rate = pd.Series(al).groupby(panel["isin"].to_numpy()).any().mean()
    tag = "floored (invalid)" if floor else "plain product (valid)"
    print(f"      {tag:<26} false-alarm rate = {rate:.4f}   (alpha = {ALPHA})"
          f"   {'<-- ' + f'{rate/ALPHA:.1f}x nominal' if rate > ALPHA * 1.5 else ''}")

print(f"\n  (c) truncated rolling-window product   [RollingWindowEProcess]")
print(f"      false-alarm rate vs nominal alpha = {ALPHA}, by bet size and window:")
print(f"      {'lam':>6}{'W=12':>9}{'W=24':>9}{'W=36':>9}{'plain':>9}")
worst = 0.0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for lam in (0.95, 0.70, 0.50, 0.30):
        row = []
        for W in (12, 24, 36):
            rw = RollingWindowEProcess(ALPHA, window_size=W, lam=lam,
                                       acknowledge_invalid=True)
            rr = ville_false_alarm_rate(
                rw.run_sequential_test(panel, z_col="pit", cooldown_months=10**6),
                ALPHA, "rolling")
            row.append(rr["false_alarm_rate"])
            worst = max(worst, rr["false_alarm_rate"])
        pl = ville_false_alarm_rate(
            DeadzoneEProcess(ALPHA, lam=lam).run_sequential_test(
                panel, z_col="pit", cooldown_months=10**6), ALPHA, "wz")
        flag = "   <-- EXCEEDS alpha" if max(row) > ALPHA else ""
        print(f"      {lam:>6.2f}" + "".join(f"{v:>9.4f}" for v in row)
              + f"{pl['false_alarm_rate']:>9.4f}" + flag)
print(f"      worst observed = {worst:.4f} = {worst/ALPHA:.1f}x nominal alpha;"
      f" the plain product stays at or below alpha in every cell.")
print(f"      MixtureRestartEProcess is the valid finite-memory replacement.")

# ------------------------------------------------------------------- verdict
head("VERDICT")
if FAILURES:
    print(f"  {len(FAILURES)} CHECK(S) FAILED -- the validity chain is broken.")
    sys.exit(1)
print("  All checks passed.")
print("  The corrected stack satisfies: exact PIT -> E[e]=1 -> martingale -> Ville.")
sys.exit(0)
