"""
The null the e-process actually needs is CONDITIONAL uniformity.

    python tests_and_plots/test_conditional_pit.py

Exchangeability within a (date, rating) cohort delivers

        Z_t ~ Uniform(0,1)   marginally, at each date,

which is what a rank transform guarantees. The e-process needs strictly more:

        Z_t | F_{t-1} ~ Uniform(0,1),

because E[e_t | F_{t-1}] = 1 is what makes M a martingale. The two coincide only
if Z_t is independent of the bond's own past. On this panel the rank PIT of
det_score has lag-1 autocorrelation 0.85 and is still at 0.52 twelve months out:
consecutive months are near-copies, so E[e_t | F_{t-1}] != 1 and the martingale
property fails at the source. The process then compounds twelve nearly identical
observations as if they were twelve independent bets, which manufactures
evidence out of persistence rather than out of deterioration.

The repair is to rank an INNOVATION -- the part of the score that was not
already predictable at t-1 -- rather than its level. Variants tested:

  level        rank of s_t                       (the original Z_det)
  diff-1       rank of s_t - s_{t-1}
  diff-6       rank of s_t - s_{t-6}
  resid-AR1    rank of s_t - beta*s_{t-1}, beta fitted per rating on past data
  demeaned     rank of s_t - mean(s_{t-12..t-1})

Each is scored on (i) how close its lag-1 autocorrelation is to zero, which is
the validity question, and (ii) precision lift at a matched alarm budget, which
is the power question.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eprocess"))

from compare_pit_constructions import load
from diagnose_f1_drop import random_alarms, windowed_metric
from src.sequential import DeadzoneEProcess
from src.betting import randomized_rank_pit

RULE = "=" * 110
ALPHAS = (0.60, 0.45, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 5e-3, 1e-3, 1e-4, 1e-6)
BUDGET = 4000


def acf1(df, col, lag=1):
    a = df[col].to_numpy()
    b = df.groupby("isin")[col].shift(lag).to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def sweep_lift(df, z, tail, budget=BUDGET):
    d = df.copy(); d["pit"] = z
    rows = []
    for a in ALPHAS:
        eng = DeadzoneEProcess(a, delta=0.75, lam=0.50, tail=tail)
        res = eng.run_sequential_test(d, z_col="pit", cooldown_months=12)
        n = int(res["is_alarm"].sum())
        if n == 0:
            continue
        m = windowed_metric(res, 24)
        c = windowed_metric(random_alarms(res, n, seed=5), 24,
                            alarm_col="rand_alarm")
        rows.append((n, m["P"], m["R"], m["F1"], c["P"]))
    if not rows:
        return None
    s = pd.DataFrame(rows, columns=["n", "P", "R", "F1", "cP"]).sort_values("n")
    if budget < s["n"].min() or budget > s["n"].max():
        return dict(span=(int(s["n"].min()), int(s["n"].max())), P=np.nan,
                    R=np.nan, F1=np.nan, lift=np.nan)
    P = float(np.interp(budget, s["n"], s["P"]))
    R = float(np.interp(budget, s["n"], s["R"]))
    F1 = float(np.interp(budget, s["n"], s["F1"]))
    cP = float(np.interp(budget, s["n"], s["cP"]))
    return dict(span=(int(s["n"].min()), int(s["n"].max())), P=P, R=R, F1=F1,
                lift=P / max(1e-9, cP))


def main():
    t0 = time.time()
    print(RULE)
    print("  CONDITIONAL UNIFORMITY: does ranking an INNOVATION restore the "
          "martingale property?")
    print(RULE)
    df = load()
    rng = np.random.default_rng(42)
    g = df.groupby("isin")["det_score"]

    df["s_level"] = df["det_score"]
    df["s_diff1"] = g.diff(1)
    df["s_diff6"] = g.diff(6)
    df["s_demean"] = df["det_score"] - g.transform(
        lambda s: s.shift(1).rolling(12, min_periods=3).mean())
    # AR(1) residual, coefficient fitted per rating on the first half only
    lag1 = g.shift(1)
    half = df["dates"] <= df["dates"].median()
    beta = {}
    for r, sub in df.assign(lag1=lag1).groupby("prev_enc_y"):
        # fit on the first half only, so the coefficient is not chosen on the
        # same data the e-process is later evaluated on
        ok = np.isfinite(sub["lag1"]) & half.loc[sub.index]
        x = sub.loc[ok, "lag1"]
        y = sub.loc[ok, "det_score"]
        beta[r] = float(np.polyfit(x, y, 1)[0]) if len(x) > 100 else 1.0
    b_arr = df["prev_enc_y"].map(beta).fillna(1.0).to_numpy()
    df["s_ar1"] = df["det_score"].to_numpy() - b_arr * lag1.to_numpy()

    cohort = pd.MultiIndex.from_frame(df[["dates", "prev_enc_y"]]).to_numpy()

    variants = []
    for name, col in [("level      (original Z_det)", "s_level"),
                      ("diff-1", "s_diff1"),
                      ("diff-6", "s_diff6"),
                      ("resid-AR1", "s_ar1"),
                      ("demeaned (12m trailing)", "s_demean")]:
        v = df[col].to_numpy().astype(float)
        v = np.where(np.isfinite(v), v, np.nan)
        z = pd.Series(randomized_rank_pit(v, cohort, rng)).fillna(0.5).to_numpy()
        zc = f"Z_{col}"
        df[zc] = z
        variants.append((name, zc))

    print(f"\n{'variant':<30}{'lag-1 ACF':>11}{'lag-6 ACF':>11}"
          f"{'alarm span':>18}{'P24':>8}{'R24':>8}{'F1':>8}{'lift':>7}")
    print("-" * len(RULE))
    for name, zc in variants:
        a1, a6 = acf1(df, zc, 1), acf1(df, zc, 6)
        r = sweep_lift(df, df[zc].to_numpy(), "upper")
        span = f"{r['span'][0]:,}–{r['span'][1]:,}"
        if np.isnan(r["P"]):
            print(f"{name:<30}{a1:>11.3f}{a6:>11.3f}{span:>18}"
                  f"{'—':>8}{'—':>8}{'—':>8}{'—':>7}")
        else:
            print(f"{name:<30}{a1:>11.3f}{a6:>11.3f}{span:>18}"
                  f"{r['P']:>8.2f}{r['R']:>8.2f}{r['F1']:>8.4f}{r['lift']:>7.2f}")

    # reference: the model PIT
    df["Z_model"] = df["randomized_pit"].fillna(0.5)
    a1, a6 = acf1(df, "Z_model", 1), acf1(df, "Z_model", 6)
    r = sweep_lift(df, df["Z_model"].to_numpy(), "two-sided")
    span = f"{r['span'][0]:,}–{r['span'][1]:,}"
    print(f"{'model PIT (two-sided)':<30}{a1:>11.3f}{a6:>11.3f}{span:>18}"
          f"{r['P']:>8.2f}{r['R']:>8.2f}{r['F1']:>8.4f}{r['lift']:>7.2f}")

    print(f"\n{RULE}\n  READING\n{RULE}")
    print("  'lag-1 ACF' is the validity column: H_0 requires 0. A level-ranked")
    print("  score at 0.85 is not conditionally uniform, so E[e_t|F_(t-1)] != 1")
    print("  and the compounded process is not a martingale, whatever the")
    print(f"  cross-sectional histogram looks like.")
    print("  'lift' is the power column: precision at a matched "
          f"{BUDGET:,}-alarm budget,")
    print("  divided by the precision random alarms achieve at that budget.")
    print(f"\n  total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
