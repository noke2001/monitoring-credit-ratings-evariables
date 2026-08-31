"""Model-free companion to the E-process: does last month's covariate order
this month's cross-section?

Under H_0 the within-class rank correlation between Y_{t-1} and X_t must be
zero in expectation at every date (a conditional-exchangeability consequence
that involves no copulas, no permutations and no e-values).  Measuring it
directly says how false the null is, and splitting it by NBER recession status
says when.  This is the diagnostic Section 3.6 reads its mechanism off, because
unlike log Ê_t it is not censored by the per-month ceiling.

Usage:
    python scripts/rank_diagnostic.py                                  # X=yield
    python scripts/rank_diagnostic.py --target spread \\
        --partition-covs coupon duration
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.panel import load_panel, RATING_CLASSES  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
CSV_DEFAULT = BASE.parent / "CorpBond_Reconciling" / "corp_jkp_mergedv2.csv"
NBER = [(200712, 200906), (202002, 202004)]


def in_recession(d):
    return any(a <= d <= b for a, b in NBER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_DEFAULT))
    ap.add_argument("--target", default="yield")
    ap.add_argument("--partition-covs", nargs="*", default=["duration", "spread"])
    ap.add_argument("--min-obs", type=int, default=8)
    args = ap.parse_args()

    df = load_panel(args.csv, target=args.target,
                    partition_covs=tuple(args.partition_covs))
    tgt = f"{args.target}_mc"

    lines = []
    out = lines.append
    out(f"Median within-class Spearman rho( Y_(t-1), X_t ),  X = {args.target} "
        f"(mean-corrected)")
    out("Under H_0 this is 0 in expectation at every date.")
    out("Recession months = official NBER (2007-12--2009-06, 2020-02--2020-04)")
    out("")

    falls = total = 0
    for cov in args.partition_covs:
        out(f"===== Y = {cov} =====")
        out(f"  {'class':6s} {'median':>8s} {'|rho|>0.3':>10s} "
            f"{'expansion':>10s} {'recession':>10s} {'n dates':>8s}")
        for cls in RATING_CLASSES:
            sub = (df.filter(pl.col("rtg") == cls)
                     .select(["dates", tgt, f"{cov}_mc_lag"]).drop_nulls())
            rhos, exp, rec = [], [], []
            for (d,), g in sub.partition_by("dates", as_dict=True).items():
                if len(g) < args.min_obs:
                    continue
                r = spearmanr(g.get_column(f"{cov}_mc_lag").to_numpy(),
                              g.get_column(tgt).to_numpy()).statistic
                if not np.isfinite(r):
                    continue
                rhos.append(r)
                (rec if in_recession(d) else exp).append(r)
            if not rhos:
                out(f"  {cls:6s} {'--':>8s}")
                continue
            me, mr = np.median(exp), np.median(rec) if rec else np.nan
            total += 1
            falls += int(np.isfinite(mr) and mr < me)
            out(f"  {cls:6s} {np.median(rhos):8.3f} "
                f"{np.mean(np.abs(rhos) > 0.3):10.0%} {me:10.3f} {mr:10.3f} "
                f"{len(rhos):8d}")
        out("")
    out(f"dependence falls from expansion to recession in {falls} of {total} "
        f"class x covariate cells")

    text = "\n".join(lines)
    print(text)
    path = BASE / "results" / f"rank_correlation_{args.target}.txt"
    path.write_text(text + "\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
