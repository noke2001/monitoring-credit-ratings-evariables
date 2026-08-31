"""Legacy vs corrected, on identical data — the figure that explains the change.

Reads the per-date CSVs written by ``run_empirical.py`` in both candidate modes
and draws three accumulated processes per class:

  1. corrected candidate + identity adjoined   — the statistic of the thesis
  2. corrected candidate + identity omitted    — isolates the Monte Carlo defect
  3. legacy candidate  + identity omitted      — what the old notebooks plotted

Curves 1 and 2 differ only in the denominator of the e-value, computed from the
same permutation draws, so their gap is exactly the cost of the missing
identity. Curves 2 and 3 differ only in the candidate, so their gap is exactly
the cost of the legacy candidate's defects. The two effects run in opposite
directions, which is why the old figures looked neither valid nor alarming.

Usage:  python scripts/make_comparison_figure.py [--y-cov duration]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path(__file__).resolve().parents[1]
CLASSES = ["AAA", "AA", "A", "BBB", "BB", "B"]


def read_csv(path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    return {
        "date": np.array([int(r["date"]) for r in rows]),
        "log_M": np.array([float(r["log_M"]) for r in rows]),
        "log_M_naive": np.array([float(r["log_M_naive"]) for r in rows]),
        "log_E": np.array([float(r["log_E"]) for r in rows]),
        "log_E_naive": np.array([float(r["log_E_naive"]) for r in rows]),
    }


def to_float_dates(d):
    return (d // 100) + ((d % 100) - 0.5) / 12.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y-cov", default="duration")
    ap.add_argument("--target", default="yield")
    ap.add_argument("--n-perms", type=int, default=1000)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    any_data = False
    for j, cls in enumerate(CLASSES):
        ax = axes.ravel()[j]
        corr = read_csv(BASE / "results" / f"empirical_{args.target}_{args.y_cov}_{cls}.csv")
        leg = read_csv(BASE / "results" / f"empirical_legacycand_{args.target}_{args.y_cov}_{cls}.csv")
        if corr is None:
            ax.set_visible(False)
            continue
        any_data = True
        t = to_float_dates(corr["date"])
        ax.plot(t, corr["log_M_naive"], lw=1.2, color="tab:gray", ls="--",
                label="corrected $q$, identity omitted (invalid)")
        if leg is not None:
            tl = to_float_dates(leg["date"])
            ax.plot(tl, leg["log_M_naive"], lw=1.2, color="tab:orange",
                    label="legacy $q$, identity omitted (old figures)")
        ax.plot(t, corr["log_M"], lw=1.8, color="firebrick",
                label=r"corrected $q$, identity adjoined (thesis)")
        ax.axhline(np.log(20), ls=":", c="k", lw=0.8)
        ax.set_yscale("symlog", linthresh=100)
        ax.set_title(f"({chr(97 + j)}) {cls}", fontsize=10, fontweight="bold")
        ax.set_ylabel(r"$\log M_k$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")
    if not any_data:
        raise SystemExit("no result CSVs found — run run_empirical.py first")
    fig.suptitle(f"Identical data, three statistics — $Y$ = {args.y_cov}, "
                 f"$N$ = {args.n_perms}  (symlog scale)", fontsize=12)
    fig.tight_layout()
    out = BASE / "plots" / f"legacy_vs_corrected_{args.target}_{args.y_cov}.png"
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")

    # per-class summary table
    print(f"\n{'class':6s} {'corrected':>12s} {'naive MC':>14s} "
          f"{'legacy q + naive':>18s} {'max monthly (naive)':>20s}")
    for cls in CLASSES:
        corr = read_csv(BASE / "results" / f"empirical_{args.target}_{args.y_cov}_{cls}.csv")
        leg = read_csv(BASE / "results" / f"empirical_legacycand_{args.target}_{args.y_cov}_{cls}.csv")
        if corr is None:
            continue
        legval = f"{leg['log_M_naive'][-1]:18.1f}" if leg is not None else f"{'-':>18s}"
        print(f"{cls:6s} {corr['log_M'][-1]:12.1f} {corr['log_M_naive'][-1]:14.1f} "
              f"{legval} {corr['log_E_naive'].max():20.2f}")
    print(f"\nvalid per-month ceiling log(N+1) = {np.log(args.n_perms + 1):.2f}")


if __name__ == "__main__":
    main()
