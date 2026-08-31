"""
Exact attribution of the F1 change from 0.63 to 0.19.

    python tests_and_plots/decompose_f1_change.py

The 0.6265 in the withdrawn report came from verify_and_plot_approaches.py on
the GBDT panel, combining five things that were changed together:

  1. a hazard model trained WITHOUT a label-horizon embargo (leaked),
  2. Z = groupby(...).rank(pct=True)          (not Uniform(0,1)),
  3. e = 1 + 2(Z-0.75)/0.25                   (E[e] = 1.25, not 1),
  4. M <- max(1, M*e)                          (submartingale),
  5. an unbounded "next transition" metric     (no time limit either side).

This script rebuilds the original configuration and then flips ONE item at a
time, so each rung's contribution is isolated. Every rung also reports the
CHANCE baseline for its own metric -- the score that count-matched random
alarms achieve -- because a raw F1 is meaningless without it.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eprocess"))

from diagnose_f1_drop import old_regime_metric, random_alarms, windowed_metric
from src.betting import deadzone_e, randomized_rank_pit

RULE = "=" * 104
PANEL = _panel("multi_horizon_survival_inference_results.csv")

FEATS = ['spread', 'yield', 'duration', 'volatility', 'VaR', 'D2D', 'debt_ebitda',
         'market_lev', 'ebitda_debt', 'mom6', 'age', 'ret_6_1', 'ni_me', 'at_be',
         'gp_at', 'turn_vol', 'be_me', 'coupon', 'moddurtn', 'skew', 'retexc',
         'spr_to_d2d', 'vixbeta', 'oper_lvg', 'sales_at', 'at_me', 'eq_dur',
         'rvol_21d', 'spread_d6m', 'spread_d12m', 'D2D_d6m', 'D2D_d12m',
         'volatility_d6m', 'debt_ebitda_d6m']


def train_hazard(df, feats, embargo_months):
    """Annual walk-forward hazard, with or without a label-horizon embargo."""
    out = np.zeros(len(df))
    dates = sorted(df["dates"].unique())
    off = pd.DateOffset(months=embargo_months)
    model, last_fit = None, None
    for d in dates:
        d = pd.Timestamp(d)
        if last_fit is None or (d - last_fit).days >= 365:
            tr = df["dates"] <= (d - off) if embargo_months else df["dates"] < d
            y = df.loc[tr, "change_in_24m"].to_numpy()
            if tr.sum() > 5000 and len(np.unique(y)) > 1:
                model = HistGradientBoostingClassifier(
                    max_iter=100, learning_rate=0.05, max_leaf_nodes=31,
                    random_state=42)
                model.fit(df.loc[tr, feats].to_numpy(), y)
                last_fit = d
        te = (df["dates"] == d).to_numpy()
        if model is not None and te.any():
            out[te] = model.predict_proba(df.loc[te, feats].to_numpy())[:, 1]
    return out


def run_eprocess(df, z, use_ramp, use_floor, alpha=0.10, cooldown=12):
    """Compound e-values; `use_ramp`/`use_floor` reproduce the original code."""
    if use_ramp:
        e = np.where(z > 0.75, 1.0 + 2.0 * (z - 0.75) / 0.25, 1.0)
    else:
        e = deadzone_e(z, 0.95, 0.75, tail="upper")
    thresh = 1.0 / alpha
    is_chg = df["is_rating_change"].to_numpy().astype(bool)
    alarms = np.zeros(len(df), dtype=bool)
    for _, idxs in df.groupby("isin", sort=False).indices.items():
        idxs = np.asarray(idxs)
        idxs = idxs[np.argsort(df["dates"].to_numpy()[idxs], kind="stable")]
        M, last = 1.0, -(10**9)
        for k, i in enumerate(idxs):
            if is_chg[i]:
                M = 1.0
            M = M * e[i]
            if use_floor:
                M = max(1.0, M)
            if M >= thresh and (k - last) >= cooldown:
                alarms[i] = True
                last = k
                M = 1.0
    return alarms


def score(df, alarms, metric):
    d = df.copy()
    d["is_alarm"] = alarms
    r = random_alarms(d, int(alarms.sum()), seed=7)
    if metric == "old":
        m = old_regime_metric(d)
        c = old_regime_metric(r, alarm_col="rand_alarm")
    else:
        m = windowed_metric(d, 24)
        c = windowed_metric(r, 24, alarm_col="rand_alarm")
    return m, c


def main():
    t0 = time.time()
    print(RULE)
    print("  ATTRIBUTION LADDER -- each rung flips exactly ONE item")
    print(RULE)

    keep = (["isin", "dates", "enc_y", "prev_enc_y", "is_rating_change",
             "is_downgrade", "change_in_24m"] + FEATS)
    df = pd.read_csv(PANEL, usecols=lambda c: c in keep, low_memory=False)
    df["dates"] = pd.to_datetime(df["dates"])
    df = df.sort_values(["isin", "dates"]).reset_index(drop=True)
    feats = [f for f in FEATS if f in df.columns]
    df[feats] = df[feats].astype("float32").fillna(0.0)
    for c in ("is_rating_change", "is_downgrade", "change_in_24m"):
        df[c] = df[c].astype(float).fillna(0).astype(int)
    print(f"  panel: {len(df):,} rows, {df['isin'].nunique():,} bonds, "
          f"{len(feats)} features\n")

    print("  training hazard twice (leaked / embargoed) ...", flush=True)
    h_leak = train_hazard(df, feats, 0)
    h_emb = train_hazard(df, feats, 24)
    print(f"  done ({time.time()-t0:.0f}s)\n")

    rng = np.random.default_rng(42)
    cohort = pd.MultiIndex.from_frame(df[["dates", "prev_enc_y"]]).to_numpy()

    def pit_old(h):
        return pd.Series(h).groupby(cohort).rank(pct=True).fillna(0.5).to_numpy()

    def pit_new(h):
        return pd.Series(randomized_rank_pit(h, cohort, rng)).fillna(0.5).to_numpy()

    # rung: (label, hazard, pit_fn, ramp, floor, metric)
    rungs = [
        ("0. Original configuration",            h_leak, pit_old, True,  True,  "old"),
        ("1. + remove the max(1,.) floor",       h_leak, pit_old, True,  False, "old"),
        ("2. + fix the e-value (E[e]=1)",        h_leak, pit_old, False, False, "old"),
        ("3. + fix the PIT (exactly uniform)",   h_leak, pit_new, False, False, "old"),
        ("4. + embargo the hazard labels",       h_emb,  pit_new, False, False, "old"),
        ("5. + windowed 24m metric  [FINAL]",    h_emb,  pit_new, False, False, "new"),
    ]

    print(f"{'rung':<40}{'alarms':>8}{'P (%)':>9}{'R (%)':>9}{'F1':>9}"
          f"{'chance F1':>11}{'ratio':>8}")
    print("-" * len(RULE))
    prev_f1 = None
    for label, h, pit_fn, ramp, floor, metric in rungs:
        z = pit_fn(h)
        al = run_eprocess(df, z, ramp, floor)
        m, c = score(df, al, metric)
        ratio = m["F1"] / c["F1"] if c["F1"] > 0 else float("nan")
        delta = "" if prev_f1 is None else f"   ({m['F1']-prev_f1:+.4f})"
        print(f"{label:<40}{int(al.sum()):>8,}{m['P']:>9.2f}{m['R']:>9.2f}"
              f"{m['F1']:>9.4f}{c['F1']:>11.4f}{ratio:>8.2f}{delta}")
        prev_f1 = m["F1"]

    print(f"\n{RULE}\n  READING\n{RULE}")
    print("  'ratio' is F1 divided by the chance F1 for that same metric. It is")
    print("  the only column comparable ACROSS rungs, because the metric changes")
    print("  at rung 5 and the two metrics have very different chance levels.")
    print(f"\n  total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
