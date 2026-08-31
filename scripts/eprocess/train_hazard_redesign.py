"""
Redesigned transition-hazard model.

    python tests_and_plots/train_hazard_redesign.py

The previous hazard scored AUC 0.528 out-of-sample once the label leak was
embargoed away. Three design faults explain most of that, and none is fixed by
more trees:

  1. TARGET. `change_in_24m` collapses twenty-four very different events into one
     indicator -- a transition next month and a transition in twenty-three months
     get the same label. Replaced by a DISCRETE-TIME HAZARD,
     h_t = P(transition at t+1 | no transition through t), which is the natural
     object for this problem and is what a survival model actually estimates.
     It also shrinks the label embargo from 24 months to 1, recovering ~23 months
     of training data at every refit.

  2. FEATURES ANCHORED TO THE WRONG ORIGIN. Trailing 6m/12m deltas answer "what
     changed recently". The agency's decision depends on "what has changed SINCE
     WE LAST LOOKED", and the standing rating on this panel has a median age of
     22 months. Added: rating age, and the change in each key fundamental
     measured from the date the standing rating was set.

  3. STALENESS ENTERING IMPLICITLY. Rating age is a legitimate and strongly
     predictive covariate. Leaving it out did not remove its influence, it just
     let it act through correlated features in an uncontrolled way. It is now an
     explicit feature.

Reported against the old design on identical folds and identical evaluation rows.
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RULE = "=" * 92
PANEL = _panel("multi_horizon_survival_inference_results.csv")
OUT = _panel("hazard_redesign_scores.csv")

BASE = ['spread', 'yield', 'duration', 'volatility', 'VaR', 'D2D', 'debt_ebitda',
        'market_lev', 'ebitda_debt', 'mom6', 'age', 'ret_6_1', 'ni_me', 'at_be',
        'gp_at', 'turn_vol', 'be_me', 'coupon', 'moddurtn', 'skew', 'retexc',
        'spr_to_d2d', 'vixbeta', 'oper_lvg', 'sales_at', 'at_me', 'eq_dur',
        'rvol_21d']
DELTAS = ['spread_d6m', 'spread_d12m', 'D2D_d6m', 'D2D_d12m', 'volatility_d6m',
          'debt_ebitda_d6m']
# fundamentals re-measured from the date the standing rating was set
ANCHOR = ['spread', 'D2D', 'debt_ebitda', 'volatility', 'market_lev', 'yield']


def load():
    keep = (["isin", "dates", "enc_y", "prev_enc_y", "is_rating_change",
             "change_in_24m"] + BASE + DELTAS)
    df = pd.read_csv(PANEL, usecols=lambda c: c in keep, low_memory=False)
    df["dates"] = pd.to_datetime(df["dates"])
    df = df.sort_values(["isin", "dates"]).reset_index(drop=True)
    df["is_rating_change"] = df["is_rating_change"].astype(float).fillna(0).astype(int)
    df["change_in_24m"] = df["change_in_24m"].astype(float).fillna(0).astype(int)

    # --- fault 1: the proper target -------------------------------------
    # h_t = 1 if a transition occurs at t+1. Rows are naturally "at risk"
    # because a transition resets the segment.
    df["transition_next"] = (
        df.groupby("isin")["is_rating_change"].shift(-1).fillna(0).astype(int)
    )

    # --- fault 2 & 3: rating age and rating-anchored deltas --------------
    seg = df.groupby("isin")["is_rating_change"].cumsum()
    df["rating_age"] = df.groupby(["isin", seg]).cumcount()
    for c in ANCHOR:
        if c in df.columns:
            at_set = df.groupby(["isin", seg])[c].transform("first")
            df[f"{c}_since_rating"] = df[c] - at_set
            df[f"{c}_since_rating_rel"] = (df[c] - at_set) / (at_set.abs() + 1e-6)
    return df, seg


def walk_forward(df, feats, target, embargo_months, label):
    """Annual refit, expanding window, label-horizon embargo."""
    X = df[feats].astype("float32").fillna(0.0).to_numpy()
    y = df[target].to_numpy()
    out = np.full(len(df), np.nan)
    off = pd.DateOffset(months=embargo_months)
    model, last_fit = None, None
    for d in sorted(df["dates"].unique()):
        d = pd.Timestamp(d)
        if last_fit is None or (d - last_fit).days >= 365:
            tr = (df["dates"] <= (d - off)).to_numpy()
            if tr.sum() > 5000 and len(np.unique(y[tr])) > 1:
                model = lgb.LGBMClassifier(
                    n_estimators=250, learning_rate=0.05, num_leaves=31,
                    min_child_samples=40, subsample=0.85, subsample_freq=1,
                    colsample_bytree=0.85, verbose=-1, n_jobs=4, random_state=42)
                model.fit(X[tr], y[tr])
                last_fit = d
        te = (df["dates"] == d).to_numpy()
        if model is not None and te.any():
            out[te] = model.predict_proba(X[te])[:, 1]
    return out


def main():
    t0 = time.time()
    print(RULE); print("  HAZARD REDESIGN"); print(RULE)
    df, seg = load()
    old_feats = [f for f in BASE + DELTAS if f in df.columns]
    new_feats = old_feats + ["rating_age", "prev_enc_y"] + [
        c for c in df.columns if c.endswith(("_since_rating", "_since_rating_rel"))]
    print(f"  panel {len(df):,} rows | old design {len(old_feats)} features | "
          f"new design {len(new_feats)} features")
    print(f"  1-month transition rate: {100*df['transition_next'].mean():.3f}%   "
          f"24-month: {100*df['change_in_24m'].mean():.2f}%\n")

    print("  fitting ...", flush=True)
    df["h_old"] = walk_forward(df, old_feats, "change_in_24m", 24, "old")
    df["h_new"] = walk_forward(df, new_feats, "transition_next", 1, "new")
    print(f"  done ({time.time()-t0:.0f}s)\n")

    # score both on identical rows, against both targets
    ok = np.isfinite(df["h_old"]) & np.isfinite(df["h_new"])
    print(f"{'design':<44}{'AUC vs 24m':>13}{'AUC vs next-month':>20}")
    print("-" * len(RULE))
    for col, name in (("h_old", "old: 24m indicator, level features"),
                      ("h_new", "new: 1m hazard, rating-anchored features")):
        a24 = roc_auc_score(df.loc[ok, "change_in_24m"], df.loc[ok, col])
        a1 = roc_auc_score(df.loc[ok, "transition_next"], df.loc[ok, col])
        print(f"{name:<44}{a24:>13.4f}{a1:>20.4f}")

    # per-year AUC, the honest view
    print(f"\n  AUC vs 24m transition, by year:")
    print(f"    {'year':<8}{'old':>9}{'new':>9}{'delta':>9}")
    yr = df.loc[ok].assign(y=df.loc[ok, "dates"].dt.year)
    for year, g in yr.groupby("y"):
        if g["change_in_24m"].nunique() < 2 or len(g) < 500:
            continue
        a_o = roc_auc_score(g["change_in_24m"], g["h_old"])
        a_n = roc_auc_score(g["change_in_24m"], g["h_new"])
        print(f"    {year:<8}{a_o:>9.4f}{a_n:>9.4f}{a_n-a_o:>+9.4f}")

    keep = ["isin", "dates", "enc_y", "prev_enc_y", "is_rating_change",
            "rating_age", "h_old", "h_new"]
    df[keep].to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT.relative_to(ROOT)}   ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
