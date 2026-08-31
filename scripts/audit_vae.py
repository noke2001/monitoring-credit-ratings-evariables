"""Audit of the Chapter 4 §4.3.2–§4.3.3 VAE panels.

The semi-supervised pipeline emits several candidate scores from one forward
pass, and they are not equally useful.  This script measures each of them on
the same rows, so §4.3.2 (what the *unsupervised* signal alone can do) and
§4.3.3 (what supervision adds) rest on a like-for-like comparison rather than
on separate runs.

Two panels can be compared side by side with ``--baseline``, which is how the
imputation and delta-gap fixes were checked: the pre-fix panel filled every
missing feature with 0.0 on the raw scale and differenced by row rather than by
month.

    python scripts/audit_vae.py
    python scripts/audit_vae.py --baseline "../approach1 newer/semi_supervised_fresh_inference_results.csv"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.univariate import auc  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
DEFAULT = BASE / "results" / "semi_supervised_fresh_inference_results.csv"

#: score -> (label, does LARGE mean downgrade candidate?)
SCORES = {
    "anomaly_score":        ("VAE reconstruction error (unsupervised)", True),
    "directional_deviation": ("directional deviation (supervised)", True),
    "downgrade_tail_prob":  ("downgrade tail mass (supervised)", True),
    "randomized_pit":       ("randomised PIT of the monitored rating", None),
}


def emit(fh, s=""):
    print(s)
    fh.write(s + "\n")


def load(path):
    d = pl.read_csv(path, infer_schema_length=50000, ignore_errors=True)
    d = d.with_columns(
        (pl.col("dates").str.slice(0, 4).cast(pl.Int64) * 12
         + pl.col("dates").str.slice(5, 2).cast(pl.Int64)).alias("midx"))
    return d.sort(["isin", "midx"])


def acf(d, col, lag=1):
    """Lag-k autocorrelation within a bond, contiguous months only."""
    v = d[col].to_numpy().astype(float)
    isin = d["isin"].to_numpy(); m = d["midx"].to_numpy()
    ok = (isin[lag:] == isin[:-lag]) & (m[lag:] - m[:-lag] == lag)
    a, b = v[:-lag][ok], v[lag:][ok]
    g = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[g], b[g])[0, 1]) if g.sum() > 30 else np.nan


def cohort_pit(d, col):
    """Ascending rank within (date, monitored rating): large = downgrade."""
    z = (d.with_columns(
        (pl.col(col).rank("average").over(["dates", "prev_enc_y"])
         / pl.col(col).count().over(["dates", "prev_enc_y"])).alias("_z")))
    return z["_z"].to_numpy().astype(float)


def report(fh, d, title, rng, perm=20):
    emit(fh, "\n" + "=" * 76)
    emit(fh, title)
    emit(fh, "=" * 76)
    lab = d["future_downgrade_12m"].to_numpy()
    lab = (np.asarray(lab, bool) if lab.dtype != object
           else np.array([str(x).lower() in ("true", "1") for x in lab]))
    emit(fh, f"  {d.height:,} rows | {d['isin'].n_unique():,} bonds | "
             f"downgrade-within-12m base rate {lab.mean():.2%}")
    emit(fh, f"\n{'score':<42}{'ACF(1)':>9}{'AUC raw':>9}{'AUC PIT':>9}{'null':>15}")
    out = {}
    for col, (nice, _) in SCORES.items():
        if col not in d.columns:
            continue
        v = d[col].to_numpy().astype(float)
        z = cohort_pit(d, col)
        a_raw, a_pit = auc(v, lab), auc(z, lab)
        null = np.array([auc(rng.permutation(z), lab) for _ in range(perm)])
        emit(fh, f"{nice:<42}{acf(d, col):>9.3f}{a_raw:>9.3f}{a_pit:>9.3f}"
                 f"{f'{null.mean():.3f}+-{null.std():.3f}':>15}")
        out[col] = (a_raw, a_pit)
    emit(fh, "\n  AUC raw = the score itself; AUC PIT = its within-cohort rank.")
    emit(fh, "  Chance is 0.500; a value below 0.5 means the score points the")
    emit(fh, "  other way, which for the randomised PIT is expected (it is a")
    emit(fh, "  calibration statistic, not a directional one).")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(DEFAULT))
    ap.add_argument("--baseline", default=None,
                    help="a second panel to compare against (e.g. the pre-fix one)")
    ap.add_argument("--perm", type=int, default=20)
    args = ap.parse_args()

    (BASE / "results").mkdir(exist_ok=True)
    fh = open(BASE / "results" / "vae_audit.txt", "w")
    rng = np.random.default_rng(0)

    new = load(args.panel)
    res_new = report(fh, new, "CORRECTED PANEL  (train-window median imputation, "
                              "month-aligned deltas)", rng, args.perm)

    if args.baseline and Path(args.baseline).exists():
        old = load(args.baseline)
        res_old = report(fh, old, "BASELINE PANEL  (zero-fill imputation, "
                                  "row-based deltas)", rng, args.perm)
        emit(fh, "\n" + "=" * 76)
        emit(fh, "DIFFERENCE  (corrected - baseline)")
        emit(fh, "=" * 76)
        emit(fh, f"{'score':<42}{'d AUC raw':>11}{'d AUC PIT':>11}")
        for col in res_new:
            if col in res_old:
                emit(fh, f"{SCORES[col][0]:<42}"
                         f"{res_new[col][0] - res_old[col][0]:>+11.3f}"
                         f"{res_new[col][1] - res_old[col][1]:>+11.3f}")

    fh.close()
    print("\nwrote results/vae_audit.txt")


if __name__ == "__main__":
    main()
