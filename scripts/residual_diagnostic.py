"""Model-free companion to the innovation monitor (thesis Section 3.8.2).

`rank_diagnostic.py` measures how false the LEVEL null is: the median
within-class rank correlation between Y_{t-1} and X_t reaches 0.92 on this
panel, which is why the level e-process rejects at essentially every date and
says nothing about credit.

This script measures the same quantity after the target has been replaced by
the innovation, using exactly the objects the monitor sees — the same window,
the same location-scale map, the same cohort.  If the change of target does
what Section 3.8.2 claims, the correlation the level test lives on should
collapse towards zero, and what remains should be time-varying rather than
structural.

It also records the fitted map itself: the location slope b_t and the log-scale
slope d_t, per class and date, so the question "did the relation between Y and
X move?" can be read off directly without any e-values.

Usage:
    python scripts/residual_diagnostic.py                        # X=yield
    python scripts/residual_diagnostic.py --target spread \\
        --partition-covs coupon duration
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.panel import load_panel, iter_class_dates, RATING_CLASSES  # noqa: E402
from src.residual import residualize_item  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
CSV_DEFAULT = BASE.parent / "CorpBond_Reconciling" / "corp_jkp_mergedv2.csv"
NBER = [(200712, 200906), (202002, 202004)]


def in_recession(d):
    return any(a <= d <= b for a, b in NBER)


def _rho(a, b):
    if a.size < 8:
        return np.nan
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else np.nan


def collect(df, cls, target, cov, scale_model, window_months, reference_share,
            map_form="affine", cross_scale="none"):
    """One row per evaluated date, for the level and both innovation targets."""
    rows = []
    for item in iter_class_dates(df, cls, target, cov,
                                 window_months=window_months):
        if item is None or item.get("degenerate"):
            continue
        row = {"date": item["date"], "n": int(item["x"].size),
               "rho_level_y": _rho(item["y_lag"], item["x"]),
               "rho_level_self": _rho(item["x_lag"], item["x"])}
        for mode in ("y", "self"):
            out = residualize_item(item, mode=mode, scale_model=scale_model,
                                   reference_share=reference_share,
                                   map_form=map_form,
                                   cross_scale=cross_scale)
            if out.get("degenerate"):
                row[f"rho_res{mode}_y"] = np.nan
                row[f"rho_res{mode}_self"] = np.nan
                row[f"slope_{mode}"] = np.nan
                row[f"log_slope_{mode}"] = np.nan
                continue
            row[f"rho_res{mode}_y"] = _rho(out["y_lag"], out["x"])
            row[f"rho_res{mode}_self"] = _rho(out["x_lag"], out["x"])
            row[f"slope_{mode}"] = out["map"].b
            row[f"log_slope_{mode}"] = out["map"].d
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_DEFAULT))
    ap.add_argument("--target", default="yield")
    ap.add_argument("--partition-covs", nargs="*", default=["duration", "spread"])
    ap.add_argument("--scale-model", choices=["none", "const", "affine"],
                    default="affine")
    ap.add_argument("--classes", nargs="*", default=RATING_CLASSES)
    ap.add_argument("--map-form", choices=["affine", "binned"], default="affine")
    ap.add_argument("--cross-scale", choices=["none", "mad"], default="none")
    ap.add_argument("--window-months", type=int, default=24)
    ap.add_argument("--reference-share", type=float, default=0.25,
                    help="share of the window's oldest months that fits the map; "
                         "0 reproduces Section 3.8.2 read literally")
    args = ap.parse_args()

    df = load_panel(args.csv, target=args.target,
                    partition_covs=tuple(args.partition_covs))
    (BASE / "results").mkdir(exist_ok=True)

    lines = []
    out = lines.append
    out(f"Median within-class Spearman rho, X = {args.target}, "
        f"scale model = {args.scale_model}, K = {args.window_months}, "
        f"reference share = {args.reference_share:.0%}")
    out("Columns: the conditioning variable of the rank correlation.")
    out("  level        rho( Z_(t-1) , X^mc_t )        --- what Section 3.6 tests")
    out("  res-y        rho( Z_(t-1) , X^res_t )       --- map fitted on lagged Y")
    out("  res-self     rho( Z_(t-1) , X^res_t )       --- map fitted on own lag")
    out("Under the corresponding null each is 0 in expectation at every date.")
    out("")

    all_rows = {}
    for cov in args.partition_covs:
        out(f"===== Y = {cov} =====")
        out(f"  {'class':6s} | {'Z = Y_(t-1)':^33s} | {'Z = X_(t-1)':^33s} | "
            f"{'map slope b':>20s}")
        out(f"  {'':6s} | {'level':>10s} {'res-y':>10s} {'res-self':>10s} | "
            f"{'level':>10s} {'res-y':>10s} {'res-self':>10s} | "
            f"{'median':>9s} {'expan':>4s}/{'rec':<4s}")
        for cls in args.classes:
            rows = collect(df, cls, args.target, cov, args.scale_model,
                           args.window_months, args.reference_share,
                           args.map_form, args.cross_scale)
            if not rows:
                out(f"  {cls:6s} |  (no evaluated dates)")
                continue
            all_rows[(cov, cls)] = rows

            def med(k, sel=None):
                v = [r[k] for r in rows
                     if np.isfinite(r.get(k, np.nan))
                     and (sel is None or sel(r["date"]))]
                return float(np.median(v)) if v else np.nan

            out(f"  {cls:6s} | {med('rho_level_y'):10.3f} "
                f"{med('rho_resy_y'):10.3f} {med('rho_resself_y'):10.3f} | "
                f"{med('rho_level_self'):10.3f} {med('rho_resy_self'):10.3f} "
                f"{med('rho_resself_self'):10.3f} | "
                f"{med('slope_y'):9.3f} "
                f"{med('slope_y', lambda d: not in_recession(d)):4.2f}/"
                f"{med('slope_y', lambda d: in_recession(d)):<4.2f}")
        out("")

    # A median near zero is NOT the same as a series near zero.  The lagged-Y
    # map centres the correlation but leaves long, slowly-reversing excursions
    # of either sign, which is exactly what the e-process accumulates on; the
    # own-lag map leaves excursions that are short-lived.  Two numbers separate
    # the cases: how often the correlation is large, and how persistent it is.
    out("Beyond the median: how often is rho( Y_(t-1), . ) large, and how "
        "persistent?")
    for cov in args.partition_covs:
        out(f"===== Y = {cov} =====")
        out(f"  {'class':6s} | {'share |rho| > 0.3':^32s} | "
            f"{'lag-1 autocorrelation of rho_t':^32s}")
        out(f"  {'':6s} | {'level':>10s} {'res-y':>10s} {'res-self':>10s} | "
            f"{'level':>10s} {'res-y':>10s} {'res-self':>10s}")
        for cls in args.classes:
            rows = all_rows.get((cov, cls))
            if not rows:
                continue
            big, acf = [], []
            for k in ("rho_level_y", "rho_resy_y", "rho_resself_y"):
                a = np.array([r[k] for r in rows])
                a = a[np.isfinite(a)]
                big.append(float(np.mean(np.abs(a) > 0.3)) if a.size else np.nan)
                acf.append(float(np.corrcoef(a[:-1], a[1:])[0, 1])
                           if a.size > 3 else np.nan)
            out(f"  {cls:6s} | " + " ".join(f"{v:10.2f}" for v in big) + " | "
                + " ".join(f"{v:10.2f}" for v in acf))
        out("")

    # how much of the level dependence survives the change of target
    out("Reduction in |median rho( Y_(t-1), X_t )| relative to the level:")
    for cov in args.partition_covs:
        for cls in args.classes:
            rows = all_rows.get((cov, cls))
            if not rows:
                continue
            lvl = np.median([r["rho_level_y"] for r in rows
                             if np.isfinite(r["rho_level_y"])])
            for mode in ("y", "self"):
                k = f"rho_res{mode}_y"
                r_ = np.median([r[k] for r in rows if np.isfinite(r[k])])
                out(f"  Y={cov:9s} {cls:5s} {mode:5s}: "
                    f"{abs(lvl):.3f} -> {abs(r_):.3f}  "
                    f"({100 * (1 - abs(r_) / abs(lvl)):5.1f}% removed)")
        out("")

    text = "\n".join(lines)
    print(text)
    stem = f"residual_diagnostic_{args.target}"
    if args.map_form != "affine":
        stem += f"_{args.map_form}"
    if args.cross_scale != "none":
        stem += f"_cs{args.cross_scale}"
    if args.reference_share == 0.0:
        stem += "_nosplit"
    path = BASE / "results" / f"{stem}.txt"
    path.write_text(text + "\n")

    csv_path = BASE / "results" / f"{stem}.csv"
    fields = ["y_cov", "rtg", "date", "n", "rho_level_y", "rho_level_self",
              "rho_resy_y", "rho_resy_self", "rho_resself_y", "rho_resself_self",
              "slope_y", "log_slope_y", "slope_self", "log_slope_self"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (cov, cls), rows in all_rows.items():
            for r in rows:
                w.writerow({"y_cov": cov, "rtg": cls,
                            **{k: r.get(k) for k in fields[2:]}})
    print(f"\nWrote {path} and {csv_path}")


if __name__ == "__main__":
    main()
