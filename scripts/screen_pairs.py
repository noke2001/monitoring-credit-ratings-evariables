"""Which (X, Y) pair would make the innovation monitor fire in stress?

Sections 3.6 and 3.8 both end on the same complaint: the monitor accrues
evidence fastest when markets are calm and loses ground in recessions, because
what a repricing destroys is precisely the relation the candidate was fitted on.
That is a property of the PAIR, not of the machinery, and the panel carries
thirty-odd covariates of which the thesis has used four.  This script screens
them, model-free.

The screen, per (X, Y) pair and per rating class, over the evaluated dates:

  rho_lvl      median within-class Spearman rho(Y_{t-1}, X_t).  Large means the
               level is ordered, which Section 3.6 shows is uninformative on its
               own.
  rho_res      the same after the lagged-Y location-scale map of Section 3.8 has
               been applied, using the same window split the monitor uses.
  share        fraction of dates with |rho_res| > 0.3.
  acf1         lag-one autocorrelation of the rho_res SERIES.  Low means the
               residual ordering is transient; high means it is a slowly
               reversing structure the monitor would accrue on month after month
               for reasons unconnected to any event.
  stress       median |rho_res| inside NBER recession months minus the same
               outside them.  THIS IS THE SCREEN.  Positive means the historical
               relation breaks down harder in stress than in calm --- exactly
               the configuration in which an innovation monitor built on it
               would fire when it matters, instead of stalling as every pair
               reported in Sections 3.6 and 3.8 does.

Nothing here computes an e-value, fits a copula or draws a permutation, so a
pair that screens well still has to be run through the monitor before anything
can be claimed for it.  The point of the screen is to decide which ones are
worth that.

Usage:
    python scripts/screen_pairs.py                      # the full grid
    python scripts/screen_pairs.py --targets spread --classes A BBB
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.panel import load_panel, iter_class_dates, RATING_CLASSES  # noqa: E402
from src.residual import residualize_item  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
CSV_DEFAULT = BASE.parent / "CorpBond_Reconciling" / "corp_jkp_mergedv2.csv"
NBER = [(200712, 200906), (202002, 202004)]

# Targets the thesis already reports, plus the one new one worth a look:
# spr_to_d2d is the spread scaled by a Merton distance-to-default, i.e. a
# compensation-per-unit-of-fundamental-risk measure rather than a price.
TARGETS = ["yield", "spread", "spr_to_d2d"]

# Every covariate on the panel with usable coverage that could plausibly order a
# rating class.  Grouped only for the printout.
PARTITIONS = {
    "term structure": ["duration", "coupon", "age"],
    "price / risk premium": ["spread", "yield", "spr_to_d2d", "mom6",
                             "mom6mspread", "mom6xrtg", "retexc"],
    "market risk": ["volatility", "VaR", "skew", "vixbeta", "rvol_21d"],
    "liquidity / size": ["dolvol", "amtout", "turn_vol", "me"],
    "fundamentals": ["D2D", "market_lev", "debt_ebitda",
                     "at_me", "be_me", "eq_dur"],
}
ALL_COVS = sorted({c for v in PARTITIONS.values() for c in v} | set(TARGETS))


def in_recession(d):
    return any(a <= d <= b for a, b in NBER)


def _rho(a, b, min_obs=8):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_obs:
        return np.nan
    r = spearmanr(a[ok], b[ok]).statistic
    return float(r) if np.isfinite(r) else np.nan


def screen_pair(df, target, cov, classes, window_months, reference_share,
                scale_model):
    """Return one row per rating class."""
    rows = []
    for cls in classes:
        lvl, res, dates = [], [], []
        for item in iter_class_dates(df, cls, target, cov,
                                     window_months=window_months):
            if item is None or item.get("degenerate"):
                continue
            r_l = _rho(item["y_lag"], item["x"])
            out = residualize_item(item, mode="y", scale_model=scale_model,
                                   reference_share=reference_share)
            r_r = np.nan if out.get("degenerate") else _rho(out["y_lag"], out["x"])
            if not (np.isfinite(r_l) or np.isfinite(r_r)):
                continue
            lvl.append(r_l)
            res.append(r_r)
            dates.append(item["date"])
        if len(dates) < 24:
            continue
        lvl, res = np.asarray(lvl), np.asarray(res)
        d = np.asarray(dates)
        rec = np.array([in_recession(x) for x in d])
        fin = np.isfinite(res)

        def med_abs(v, mask):
            m = mask & np.isfinite(v)
            return float(np.median(np.abs(v[m]))) if m.any() else np.nan

        r = res[fin]
        acf = (float(np.corrcoef(r[:-1], r[1:])[0, 1]) if r.size > 3 else np.nan)
        rows.append({
            "target": target, "y_cov": cov, "rtg": cls, "n_dates": len(dates),
            "rho_lvl": float(np.nanmedian(lvl)),
            "abs_rho_lvl": med_abs(lvl, np.ones_like(rec)),
            "rho_res": float(np.nanmedian(res)),
            "abs_rho_res": med_abs(res, np.ones_like(rec)),
            "share_res_gt_03": float(np.mean(np.abs(r) > 0.3)) if r.size else np.nan,
            "acf1_res": acf,
            "stress_lift_res": med_abs(res, rec) - med_abs(res, ~rec),
            "stress_lift_lvl": med_abs(lvl, rec) - med_abs(lvl, ~rec),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_DEFAULT))
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--partition-covs", nargs="*", default=None)
    ap.add_argument("--classes", nargs="*", default=RATING_CLASSES)
    ap.add_argument("--window-months", type=int, default=24)
    ap.add_argument("--reference-share", type=float, default=0.25)
    ap.add_argument("--scale-model", choices=["none", "const", "affine"],
                    default="affine")
    args = ap.parse_args()
    covs = args.partition_covs or [c for c in ALL_COVS]

    print(f"Loading panel with {len(ALL_COVS)} covariates ...", flush=True)
    # one scan: every candidate is carried as a partition covariate, so every
    # one of them acquires its _mc and _mc_lag columns and any of them can play
    # either role below
    df = load_panel(args.csv, target="yield",
                    partition_covs=tuple(c for c in ALL_COVS if c != "yield"))
    print(f"  {len(df):,} (date, bond, class) rows\n", flush=True)

    rows = []
    for target in args.targets:
        if f"{target}_mc" not in df.columns:
            print(f"skipping target {target}: not on the panel")
            continue
        for cov in covs:
            if cov == target or f"{cov}_mc_lag" not in df.columns:
                continue
            t0 = time.time()
            got = screen_pair(df, target, cov, args.classes,
                              args.window_months, args.reference_share,
                              args.scale_model)
            rows.extend(got)
            if got:
                st = np.nanmean([g["stress_lift_res"] for g in got])
                print(f"  X={target:11s} Y={cov:12s} {len(got)} classes, "
                      f"mean stress lift {st:+.3f}   ({time.time()-t0:.0f}s)",
                      flush=True)

    RESULTS = BASE / "results"
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "pair_screen.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    _report(rows, args, RESULTS / "pair_screen.txt")
    print(f"\nWrote {out}")


def _report(rows, args, path):
    """Average over classes and rank by the screen."""
    lines = []

    def w(s=""):
        lines.append(s)
        print(s)

    w(f"Pair screen — K = {args.window_months}, reference share "
      f"{args.reference_share:.0%}, scale model {args.scale_model}")
    w("stress = median |rho(Y_(t-1), X^res_t)| in NBER recession months minus "
      "the same outside them.")
    w("Positive means the historical relation breaks down HARDER in stress, "
      "which is the")
    w("configuration an innovation monitor needs in order to fire when it "
      "matters.")
    w("acf1 is the lag-one autocorrelation of that same series: low is "
      "transient structure,")
    w("high is a slowly reversing one the monitor would accrue on regardless "
      "of any event.")
    w("")

    agg = {}
    for r in rows:
        k = (r["target"], r["y_cov"])
        agg.setdefault(k, []).append(r)

    def mean(k, v):
        a = [x[v] for x in agg[k] if np.isfinite(x[v])]
        return float(np.mean(a)) if a else np.nan

    table = []
    for k in agg:
        table.append({
            "target": k[0], "y_cov": k[1], "n_classes": len(agg[k]),
            "abs_rho_lvl": mean(k, "abs_rho_lvl"),
            "abs_rho_res": mean(k, "abs_rho_res"),
            "share": mean(k, "share_res_gt_03"),
            "acf1": mean(k, "acf1_res"),
            "stress_res": mean(k, "stress_lift_res"),
            "stress_lvl": mean(k, "stress_lift_lvl"),
        })
    table.sort(key=lambda r: (-r["stress_res"] if np.isfinite(r["stress_res"])
                              else 1e9))
    w(f"  {'X':12s} {'Y':13s} {'cls':>3s} {'|rho| lvl':>10s} {'|rho| res':>10s} "
      f"{'share>.3':>9s} {'acf1':>6s} {'stress res':>11s} {'stress lvl':>11s}")
    for r in table:
        w(f"  {r['target']:12s} {r['y_cov']:13s} {r['n_classes']:3d} "
          f"{r['abs_rho_lvl']:10.3f} {r['abs_rho_res']:10.3f} "
          f"{r['share']:9.2f} {r['acf1']:6.2f} {r['stress_res']:+11.3f} "
          f"{r['stress_lvl']:+11.3f}")
    w("")
    w("The four pairs Sections 3.6 and 3.8 report, for reference:")
    for k in (("yield", "duration"), ("yield", "spread"),
              ("spread", "coupon"), ("spread", "duration")):
        m = [r for r in table if (r["target"], r["y_cov"]) == k]
        if m:
            r = m[0]
            w(f"  X={k[0]:11s} Y={k[1]:12s} stress {r['stress_res']:+.3f}, "
              f"acf1 {r['acf1']:.2f}")
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
