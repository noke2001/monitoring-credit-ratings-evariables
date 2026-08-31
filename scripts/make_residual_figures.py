"""Figures for the innovation target (thesis Section 3.8.2), as vector PDFs.

Reads the CSVs already written by ``run_empirical.py --target-mode ...``,
``residual_diagnostic.py`` and ``run_residual_synthetic.py`` — no
recomputation — and emits .pdf and .png into plots/, in the house style of
``make_thesis_figures.py``.

  fig_innovation_eprocess_<target>_<Y>   the level against both innovations,
                                         one panel per rating class
  fig_innovation_rho_<target>_<Y>        the model-free companion: the rank
                                         correlation the level test lives on,
                                         before and after the change of target
  fig_innovation_synthetic               the five validation regimes, redrawn
                                         from the saved traces

Usage:  python scripts/make_residual_figures.py [--target yield]
                                                [--partition-covs duration spread]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path(__file__).resolve().parents[1]
CLASSES = ["AAA", "AA", "A", "BBB", "BB", "B"]
CRISIS = [(200712, 200906), (202002, 202004)]
CRISIS_LABEL = "NBER recessions (2007-12--2009-06, 2020-02--2020-04)"

# level, innovation around the lagged-Y map, innovation around the own-lag map
SERIES = [
    ("", "level $X^{\\mathrm{mc}}$", "0.55", 1.0),
    ("_resy", "innovation, lagged-$Y$ map", "C0", 1.4),
    ("_resself", "innovation, own-lag map", "C3", 1.4),
]
# the same two maps with the symmetric cross-sectional standardisation added
SERIES_CS = [
    ("", "level $X^{\\mathrm{mc}}$", "0.55", 1.0),
    ("_resy_csmad", "lagged-$Y$ map, standardised", "C0", 1.4),
    ("_resself_csmad", "own-lag map, standardised", "C3", 1.4),
]


def tfloat(d):
    d = np.asarray(d)
    return (d // 100) + ((d % 100) - 0.5) / 12.0


def read(path, keys=("log_M", "log_E")):
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    out = {}
    for k in keys:
        if k in rows[0]:
            out[k] = np.array([float(r[k]) if r[k] not in ("", None) else np.nan
                               for r in rows])
    out["date"] = np.array([int(r["date"]) for r in rows])
    return out


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
    (BASE / "plots").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(BASE / "plots" / f"{stem}.{ext}")
    print(f"  wrote plots/{stem}.pdf and .png")


def _shade(ax):
    for a, b in CRISIS:
        ax.axvspan(tfloat([a])[0], tfloat([b])[0], color="0.85", alpha=0.6,
                   lw=0, zorder=0)


# --------------------------------------------------------------------------
def figure_eprocess(target, y_cov, plt, series=None, stem_suffix="",
                    title_suffix=""):
    """One panel per class: log M_k under the level and under both maps.

    A symlog vertical axis is unavoidable here.  The level process ends three
    orders of magnitude above the innovation ones, and the whole point of the
    figure is that they are not the same kind of object; a linear axis would
    show one line and five flat ones.
    """
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.32})
    drawn = False
    for j, cls in enumerate(CLASSES):
        ax = axes.ravel()[j]
        _shade(ax)
        for tag, label, color, lw in (series or SERIES):
            d = read(BASE / "results"
                     / f"empirical{tag}_{target}_{y_cov}_{cls}.csv")
            if d is None:
                continue
            drawn = True
            ax.plot(tfloat(d["date"]), d["log_M"], lw=lw, color=color,
                    label=label, zorder=3)
        ax.axhline(np.log(20), ls="--", c="k", lw=0.7, zorder=2)
        ax.axhline(0.0, ls="-", c="0.8", lw=0.6, zorder=1)
        ax.set_yscale("symlog", linthresh=10, linscale=0.6)
        ax.set_title(f"({chr(97 + j)}) {cls}", fontweight="bold", pad=3)
        ax.set_ylabel(r"$\log M_k$", labelpad=1)
    if not drawn:
        plt.close(fig)
        return False
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles + [Patch(facecolor="0.85", label=CRISIS_LABEL),
                          Line2D([], [], ls="--", c="k", lw=0.7,
                                 label=r"threshold $1/\alpha = 20$")],
               labels + [CRISIS_LABEL, r"threshold $1/\alpha = 20$"],
               loc="lower center", ncol=2, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(rf"$X=\mathtt{{{target}}}$, $Y=\mathtt{{{y_cov}}}$ (lagged): "
                 r"testing the level against testing the innovation "
                 rf"(symlog scale){title_suffix}", y=1.0)
    save(fig, f"fig_innovation_eprocess{stem_suffix}_{target}_{y_cov}")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
def figure_rho(target, y_cov, plt):
    """The mechanism, without any of the machinery.

    Under the level null the rank correlation between Y_{t-1} and X_t is zero
    in expectation at every date; on this panel it sits near 0.9, which is the
    whole content of the level rejection.  After the change of target the same
    correlation is what the innovation null constrains, and it is this series —
    not the censored e-process — that says whether the change of target did
    what it was supposed to.
    """
    path = BASE / "results" / f"residual_diagnostic_{target}.csv"
    if not path.exists():
        return False
    rows = [r for r in csv.DictReader(open(path)) if r["y_cov"] == y_cov]
    if not rows:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.32})
    for j, cls in enumerate(CLASSES):
        sub = [r for r in rows if r["rtg"] == cls]
        ax = axes.ravel()[j]
        _shade(ax)
        if sub:
            t = tfloat([int(r["date"]) for r in sub])
            for key, label, color, lw in (
                    ("rho_level_y", r"level $X^{\mathrm{mc}}_t$", "0.55", 1.0),
                    ("rho_resy_y", r"innovation, lagged-$Y$ map", "C0", 1.1),
                    ("rho_resself_y", r"innovation, own-lag map", "C3", 1.1)):
                v = np.array([float(r[key]) if r[key] not in ("", "nan")
                              else np.nan for r in sub])
                ax.plot(t, v, lw=lw, color=color, label=label, zorder=3)
        ax.axhline(0.0, ls="--", c="k", lw=0.7, zorder=2)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"({chr(97 + j)}) {cls}", fontweight="bold", pad=3)
        ax.set_ylabel(r"$\rho(Y_{t-1},\, \cdot_t)$", labelpad=1)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                   fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(r"Within-class Spearman $\rho$ between $Y_{t-1}$ and the "
                 rf"target — $X=\mathtt{{{target}}}$, $Y=\mathtt{{{y_cov}}}$; "
                 r"the null requires $0$", y=1.0)
    save(fig, f"fig_innovation_rho_{target}_{y_cov}")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
def figure_synthetic(plt, suffix=""):
    path = BASE / "results" / f"residual_synthetic_traces{suffix}.csv"
    if not path.exists():
        return False
    rows = list(csv.DictReader(open(path)))
    regimes, seen = [], set()
    for r in rows:
        if r["regime"] not in seen:
            seen.add(r["regime"])
            regimes.append(r["regime"])
    colors = {"level": "0.55", "y": "C0", "self": "C3"}
    labels = {"level": r"level $X^{\mathrm{mc}}$",
              "y": r"innovation, lagged-$Y$ map",
              "self": r"innovation, own-lag map"}
    titles = {"stable_relation": "(a) stable relation\nnull for the $Y$ map",
              "relation_break": "(b) relation break\nat $t=36$",
              "dispersion_break": "(c) dispersion break\nat $t=36$",
              "random_walk": "(d) random walk\nnull for the own-lag map",
              "drift_subset": "(e) subset drifts\nfrom $t=36$"}
    fig, axes = plt.subplots(1, len(regimes), figsize=(7.0, 2.6), sharey=True,
                             gridspec_kw={"wspace": 0.18, "top": 0.74})
    for ax, regime in zip(np.atleast_1d(axes), regimes):
        for target in ("level", "y", "self"):
            sel = [r for r in rows
                   if r["regime"] == regime and r["target"] == target]
            if not sel:
                continue
            months = np.array([int(r["month"]) for r in sel])
            # traces written before the median summary was added carry a single
            # replication under "log_M"; fall back to it rather than failing
            key = "log_M_median" if "log_M_median" in sel[0] else "log_M"
            ax.plot(months, [float(r[key]) for r in sel],
                    lw=1.3, color=colors[target], label=labels[target])
            if "log_M_p10" in sel[0]:
                ax.fill_between(months, [float(r["log_M_p10"]) for r in sel],
                                [float(r["log_M_p90"]) for r in sel],
                                color=colors[target], alpha=0.12, lw=0)
        ax.axhline(np.log(20), ls="--", c="k", lw=0.7)
        ax.axhline(0.0, ls="-", c="0.8", lw=0.6)
        if regime in ("relation_break", "dispersion_break", "drift_subset"):
            ax.axvline(36, ls=":", c="0.3", lw=0.9)
        ax.set_yscale("symlog", linthresh=10, linscale=0.6)
        ax.set_title(titles.get(regime, regime), pad=4, fontsize=7.5)
        ax.set_xlabel("month")
    np.atleast_1d(axes)[0].set_ylabel(r"$\log M_k$")
    handles, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=3, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("The level rejects in every regime; the innovation rejects "
                 "only where the relation moved\n"
                 "median over 10 replications, 10--90% band", y=1.02,
                 fontsize=9)
    save(fig, f"fig_innovation_synthetic{suffix}")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="yield")
    ap.add_argument("--partition-covs", nargs="*",
                    default=["duration", "spread"])
    args = ap.parse_args()
    plt = style()
    for y_cov in args.partition_covs:
        print(f"{args.target} / {y_cov}:")
        if not figure_eprocess(args.target, y_cov, plt):
            print("  no e-process results found")
        figure_eprocess(args.target, y_cov, plt, series=SERIES_CS,
                        stem_suffix="_csmad",
                        title_suffix=", cross-sections standardised")
        if not figure_rho(args.target, y_cov, plt):
            print("  no residual_diagnostic results found")
    print("synthetic:")
    for suffix in ("", "_nosplit"):
        if not figure_synthetic(plt, suffix):
            print(f"  no traces{suffix} found")
    print("\nInclude in LaTeX with e.g.\n"
          r"  \includegraphics[width=\textwidth]"
          r"{figures/fig_innovation_eprocess_yield_duration.pdf}")


if __name__ == "__main__":
    main()
