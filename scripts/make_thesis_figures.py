"""Regenerate the empirical figures as vector PDFs for inclusion in the thesis.

Reads the per-date CSVs already written by ``run_empirical.py`` — no
recomputation — and emits both .pdf (for \\includegraphics) and .png (for
quick viewing) into plots/.  Fonts and line widths are set for a figure
occupying \\textwidth in an 11pt report class.

Usage:  python scripts/make_thesis_figures.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path(__file__).resolve().parents[1]
CLASSES = ["AAA", "AA", "A", "BBB", "BB", "B"]
# Official NBER recession months (peak month through trough month):
#   Great Recession          2007-12 -- 2009-06  (18 months)
#   COVID-19 recession       2020-02 -- 2020-04  (2 months)
CRISIS = [(200712, 200906), (202002, 202004)]
CRISIS_LABEL = "NBER recessions (2007-12--2009-06, 2020-02--2020-04)"
N_PERMS = 1000


def read(path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    out = {k: np.array([float(r[k]) for r in rows])
           for k in ("log_M", "log_E", "log_M_naive", "log_E_naive")}
    out["date"] = np.array([int(r["date"]) for r in rows])
    return out


def tfloat(d):
    return (d // 100) + ((d % 100) - 0.5) / 12.0


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    return plt


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(BASE / "plots" / f"{stem}.{ext}")
    print(f"  wrote plots/{stem}.pdf and .png")


def figure_eprocess(y_cov, plt, target="yield"):
    """Six classes: monthly log Ê_t above, accumulated log M_k below."""
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 6.1),
                             gridspec_kw={"height_ratios": [1, 1.5, 1, 1.5],
                                          "hspace": 0.55, "wspace": 0.40})
    cap = np.log(N_PERMS + 1)
    for j, cls in enumerate(CLASSES):
        d = read(BASE / "results" / f"empirical_{target}_{y_cov}_{cls}.csv")
        if d is None:
            continue
        r0, c = 2 * (j // 3), j % 3
        ax1, ax2 = axes[r0][c], axes[r0 + 1][c]
        t = tfloat(d["date"])
        for (a, b) in CRISIS:
            for ax in (ax1, ax2):
                ax.axvspan(tfloat(np.array([a]))[0], tfloat(np.array([b]))[0],
                           color="0.85", alpha=0.6, lw=0, zorder=0)
        ax1.plot(t, d["log_E"], lw=0.6, color="C0", zorder=3)
        ax1.axhline(0, ls="--", c="k", lw=0.7, zorder=2)
        ax1.axhline(cap, ls=":", c="0.4", lw=0.7, zorder=2)
        ax1.set_title(f"({chr(97 + j)}) {cls}", fontweight="bold", pad=3)
        ax1.set_ylabel(r"$\log \hat E_t$", labelpad=1)
        ax1.tick_params(labelbottom=False)
        # symlog keeps the ceiling legible while still showing the deep
        # single-month losses (e.g. AA in Sept 2008, BB in March 2020),
        # which a linear scale would either clip or squash flat
        ax1.set_yscale("symlog", linthresh=10, linscale=0.7)
        lo = min(-12.0, 1.6 * float(d["log_E"].min()))
        ax1.set_ylim(lo, cap * 2.2)
        # keep tick labels narrow: wide ones push the next column's ylabel
        # into this panel. The ceiling is annotated inline instead.
        pairs = [(v, lab) for v, lab in
                 ((-100.0, "$-100$"), (-10.0, "$-10$"), (0.0, "$0$")) if v >= lo]
        ax1.set_yticks([v for v, _ in pairs])
        ax1.set_yticklabels([lab for _, lab in pairs])
        ax1.annotate(r"$\log(N{+}1)$", xy=(0.02, cap), xycoords=("axes fraction", "data"),
                     ha="left", va="bottom", fontsize=5.5, color="0.35")
        ax2.plot(t, d["log_M"], lw=1.2, color="C3", zorder=3)
        ax2.axhline(np.log(20), ls="--", c="k", lw=0.7, zorder=2)
        ax2.set_ylabel(r"$\log M_k$", labelpad=1)
    # say what the shading is, on the figure itself rather than only in the caption
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(facecolor="0.85", label=CRISIS_LABEL),
                        Line2D([], [], ls="--", c="k", lw=0.7,
                               label=r"no evidence ($\log \hat E_t=0$) / threshold $1/\alpha=20$"),
                        Line2D([], [], ls=":", c="0.4", lw=0.7,
                               label=r"per-month ceiling $\log(N{+}1)$")],
               loc="lower center", ncol=1, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.035))
    fig.suptitle(rf"$X=\mathtt{{{target}}}$, $Y=\mathtt{{{y_cov}}}$ (lagged), "
                 rf"both mean-corrected, $N={N_PERMS}$", y=0.995)
    save(fig, f"fig_eprocess_{target}_{y_cov}")
    plt.close(fig)


def figure_comparison(y_cov, plt, target="yield"):
    """Three statistics on identical data."""
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.42})
    for j, cls in enumerate(CLASSES):
        corr = read(BASE / "results" / f"empirical_{target}_{y_cov}_{cls}.csv")
        leg = read(BASE / "results" / f"empirical_legacycand_{target}_{y_cov}_{cls}.csv")
        if corr is None:
            continue
        ax = axes.ravel()[j]
        t = tfloat(corr["date"])
        ax.plot(t, corr["log_M_naive"], lw=1.0, ls="--", color="C3",
                label="corrected $q$, identity omitted (invalid)")
        if leg is not None:
            ax.plot(tfloat(leg["date"]), leg["log_M_naive"], lw=1.0, color="C2",
                    label="legacy $q$, identity omitted (old figures)")
        ax.plot(t, corr["log_M"], lw=1.5, color="C0",
                label="corrected $q$, identity adjoined (thesis)")
        ax.axhline(np.log(20), ls=":", c="k", lw=0.7)
        ax.set_yscale("symlog", linthresh=100)
        ax.set_title(f"({chr(97 + j)}) {cls}", fontweight="bold")
        ax.set_ylabel(r"$\log M_k$", labelpad=1)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    # thesis line first; the legacy series is absent unless a legacy-candidate
    # run exists for this target, so order by label rather than by position
    order = sorted(range(len(labels)), key=lambda i: "adjoined" not in labels[i])
    if handles:
        fig.legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="lower center", ncol=min(3, len(order)), frameon=False,
                   fontsize=6.5, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(rf"Identical data, three statistics — $Y=\mathtt{{{y_cov}}}$, "
                 rf"$N={N_PERMS}$ (symlog scale)", y=1.0)
    save(fig, f"fig_legacy_vs_corrected_{target}_{y_cov}")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="yield")
    ap.add_argument("--partition-covs", nargs="*", default=["duration", "spread"])
    args = ap.parse_args()
    plt = style()
    (BASE / "plots").mkdir(exist_ok=True)
    for y_cov in args.partition_covs:
        if read(BASE / "results" / f"empirical_{args.target}_{y_cov}_AAA.csv") is None:
            print(f"skipping {y_cov}: no results found")
            continue
        print(f"{args.target} / {y_cov}:")
        figure_eprocess(y_cov, plt, args.target)
        figure_comparison(y_cov, plt, args.target)
    print("\nInclude in LaTeX with e.g.\n"
          r"  \includegraphics[width=\textwidth]{figures/fig_eprocess_duration.pdf}")


if __name__ == "__main__":
    main()
