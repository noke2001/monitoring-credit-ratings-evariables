"""
Which PIT should the e-process run on?

    python tests_and_plots/compare_pit_constructions.py

The null under test is EXCHANGEABILITY WITHIN A COHORT: at each date, a bond's
score is exchangeable with those of its peers. Under that null its rank is
Uniform{1..n}, so Z = (rank - V)/n is exactly Uniform(0,1), and the e-process
asks whether that holds SEQUENTIALLY for a fixed bond over time.

Two distinct nulls live in this codebase and they are not interchangeable:

  MODEL-CALIBRATION null   Z = randomized PIT of R_{t-1} under the model's own
                           predictive distribution. Non-uniformity means the
                           model is miscalibrated. Testable from the marginal
                           histogram.

  EXCHANGEABILITY null     Z = randomized rank of a risk score within a
                           (date, rating) cohort. Uniform CROSS-SECTIONALLY BY
                           CONSTRUCTION -- the marginal histogram is flat no
                           matter what, and carries no information. Violations
                           are visible only in the TIME SERIES of one bond's
                           ranks: a bond persistently in the upper tail
                           contradicts the null while every cross-section stays
                           perfectly uniform.

The second is the stated null of this thesis, so the rank PIT is the right
object and a flat marginal histogram is expected, not a disappointment.

THE STALENESS CONFOUND. The standing rating R_{t-1} was set at some earlier
date s; on this panel the median standing rating is 22 months old and the 90th
percentile is 64 months. Ratings are deliberately through-the-cycle and sticky,
so disagreement between fundamentals and the standing rating grows with the AGE
of that rating whether or not a transition is imminent. Ranking within
(date, rating) compares a bond carrying a five-year-old rating against peers
rated last quarter, and the stale ones sit near the top of the ranking
persistently -- which is exactly the signature the e-process is built to fire
on. Adding rating age to the cohort key removes that confound: the null becomes
exchangeability among bonds with the same rating AND the same rating age.

Variants are scored on F1 at 24 months and, more importantly, on the RATIO to
the chance F1 of count-matched random alarms.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import pathlib

# --- panel location -----------------------------------------------------------
# The inference panels are data, not code, and are not part of this repository.
# Look in results/ first (where scripts/vae/train_semisupervised.py writes), then
# fall back to EPROCESS_PANELS if it is set.
def _panel(name):
    import os
    for cand in (ROOT / "results" / name,
                 pathlib.Path(os.environ.get("EPROCESS_PANELS", "")) / name):
        if str(cand) != name and cand.exists():
            return cand
    return ROOT / "results" / name

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eprocess"))

from diagnose_f1_drop import random_alarms, windowed_metric
from src.sequential import DeadzoneEProcess
from src.betting import randomized_rank_pit

RULE = "=" * 112
SRC = _panel("semi_supervised_fresh_inference_results.csv")


def load():
    df = pd.read_csv(SRC, low_memory=False)
    df["dates"] = pd.to_datetime(df["dates"])
    df = df.sort_values(["isin", "dates"]).reset_index(drop=True)
    prev = df.groupby("isin")["enc_y"].shift(1)
    df["is_rating_change"] = ((df["enc_y"] != prev) & prev.notna()).astype(int)

    # age of the standing rating, in months since the last rating action
    seg = df.groupby("isin")["is_rating_change"].cumsum()
    df["rating_age"] = df.groupby(["isin", seg]).cumcount()
    df["age_bucket"] = pd.cut(df["rating_age"], [-1, 11, 23, 47, 10**6],
                              labels=[0, 1, 2, 3]).astype(int)

    # forward target for the supervised score, and the deterioration score
    df["change_in_24m"] = df.groupby("isin")["is_rating_change"].transform(
        lambda s: s.iloc[::-1].rolling(24, min_periods=1).max().iloc[::-1].shift(-1)
    ).fillna(0).astype(int)
    rating_cols = [c for c in df.columns if c.startswith("prob_class_")]
    P = df[rating_cols].to_numpy()
    k = np.clip(df["prev_enc_y"].fillna(0).astype(int).to_numpy(), 0, P.shape[1] - 1)
    df["det_score"] = 1.0 - P[np.arange(len(df)), k]
    return df


def semi_supervised_score(df):
    """
    Revives the semi-supervised design: VAE latent + reconstruction anomaly +
    predictive class mass + directional deviation -> LightGBM risk score.
    Walk-forward, annual refit, 24-month label embargo.
    """
    import lightgbm as lgb
    feats = ([f"latent_z_{k}" for k in range(4)] +
             ["anomaly_score", "downgrade_tail_prob", "directional_deviation",
              "det_score", "rating_age", "prev_enc_y"] +
             [c for c in df.columns if c.startswith("prob_class_")])
    feats = [f for f in feats if f in df.columns]
    X = df[feats].astype("float32").fillna(0.0).to_numpy()
    y = df["change_in_24m"].to_numpy()
    out = np.zeros(len(df))
    off = pd.DateOffset(months=24)
    model, last_fit = None, None
    for d in sorted(df["dates"].unique()):
        d = pd.Timestamp(d)
        if last_fit is None or (d - last_fit).days >= 365:
            tr = (df["dates"] <= (d - off)).to_numpy()
            if tr.sum() > 5000 and len(np.unique(y[tr])) > 1:
                model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05,
                                           num_leaves=31, min_child_samples=30,
                                           verbose=-1, n_jobs=4, random_state=42)
                model.fit(X[tr], y[tr])
                last_fit = d
        te = (df["dates"] == d).to_numpy()
        if model is not None and te.any():
            out[te] = model.predict_proba(X[te])[:, 1]
    return out, feats


def evaluate(df, z, tail, label, alpha=0.20, lam=0.50):
    eng = DeadzoneEProcess(alpha, delta=0.75, lam=lam, tail=tail)
    d = df.copy()
    d["pit"] = z
    res = eng.run_sequential_test(d, z_col="pit", cooldown_months=12)
    n = int(res["is_alarm"].sum())
    if n == 0:
        return dict(Variant=label, Tail=tail, Alarms=0, P=0, R=0, F1=0,
                    Chance=0, Ratio=0)
    m = windowed_metric(res, 24)
    c = windowed_metric(random_alarms(res, n, seed=5), 24, alarm_col="rand_alarm")
    return dict(Variant=label, Tail=tail, Alarms=n, P=m["P"], R=m["R"],
                F1=m["F1"], Chance=c["F1"],
                Ratio=m["F1"] / c["F1"] if c["F1"] > 0 else np.nan)


def main():
    t0 = time.time()
    print(RULE)
    print("  PIT CONSTRUCTION COMPARISON — same engine, same alpha, only the "
          "PIT changes")
    print(RULE)
    df = load()
    rng = np.random.default_rng(42)
    print(f"  panel {len(df):,} rows | {df['isin'].nunique():,} bonds | "
          f"{int(df['is_rating_change'].sum()):,} transitions")
    print(f"  standing-rating age: median {df['rating_age'].median():.0f}m, "
          f"p90 {df['rating_age'].quantile(.9):.0f}m\n")

    print("  fitting the semi-supervised LightGBM risk score "
          "(latent + anomaly + probs -> P(change in 24m)) ...", flush=True)
    ss_score, feats = semi_supervised_score(df)
    df["ss_score"] = ss_score
    print(f"  {len(feats)} features, done ({time.time()-t0:.0f}s)\n")

    def rank_pit(score, keys):
        cohort = pd.MultiIndex.from_frame(df[keys]).to_numpy()
        return pd.Series(randomized_rank_pit(df[score].to_numpy(), cohort, rng)
                         ).fillna(0.5).to_numpy()

    variants = [
        ("A. model PIT of R_(t-1), two-sided",
         df["randomized_pit"].fillna(0.5).to_numpy(), "two-sided"),
        ("B. model PIT of R_(t-1), lower tail",
         df["randomized_pit"].fillna(0.5).to_numpy(), "lower"),
        ("C. rank PIT of det_score | (date, rating)",
         rank_pit("det_score", ["dates", "prev_enc_y"]), "upper"),
        ("D. rank PIT of det_score | (date, rating, age)",
         rank_pit("det_score", ["dates", "prev_enc_y", "age_bucket"]), "upper"),
        ("E. rank PIT of semi-sup score | (date, rating)",
         rank_pit("ss_score", ["dates", "prev_enc_y"]), "upper"),
        ("F. rank PIT of semi-sup score | (date, rating, age)",
         rank_pit("ss_score", ["dates", "prev_enc_y", "age_bucket"]), "upper"),
    ]

    print(f"{'variant':<52}{'tail':>11}{'alarms':>8}{'P24':>8}{'R24':>8}"
          f"{'F1_24':>9}{'chance':>9}{'ratio':>8}")
    print("-" * len(RULE))
    rows = []
    for label, z, tail in variants:
        r = evaluate(df, z, tail, label)
        rows.append(r)
        print(f"{r['Variant']:<52}{r['Tail']:>11}{r['Alarms']:>8,}"
              f"{r['P']:>8.2f}{r['R']:>8.2f}{r['F1']:>9.4f}"
              f"{r['Chance']:>9.4f}{r['Ratio']:>8.2f}")

    out = pd.DataFrame(rows)
    dest = ROOT / "results" / "pit_construction_comparison.csv"
    out.to_csv(dest, index=False)
    best = out.loc[out["Ratio"].idxmax()]
    print(f"\n  Best by ratio-to-chance: {best['Variant']}  "
          f"({best['Ratio']:.2f}x, F1={best['F1']:.4f})")
    print(f"\nSaved -> {dest.relative_to(ROOT)}   ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
