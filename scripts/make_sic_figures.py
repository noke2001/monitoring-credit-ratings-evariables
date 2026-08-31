"""Figures for the two SIC designs of Section 3.7.

Produces two vector PDFs from the saved CSVs (no recomputation):

  fig_sic_comparison       one panel per class, three accumulated processes:
                           the Section 3.6 baseline (full symmetric group,
                           duration blocks), design (a) sector blocks, and
                           design (b) within-sector permutations.  This is the
                           figure that shows (b) tracking the baseline almost
                           exactly -- conceding sector structure buys nothing --
                           while (a) separates the classes.

  fig_eprocess_sic_sectorblocks
                           the usual two-panel-per-class layout for design (a),
                           where AAA never crosses the threshold at all.

Usage:  python scripts/make_sic_figures.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path(__file__).resolve().parents[1]
CLASSES = ["AAA", "AA", "A", "BBB", "BB", "B"]
CRISIS = [(200712, 200906), (202002, 202004)]
CRISIS_LABEL = "NBER recessions (2007-12--2009-06, 2020-02--2020-04)"
N_PERMS = 1000


def read(path):
    if not Path(path).exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    out = {"date": np.array([int(r["date"]) for r in rows])}
    for k in ("log_M", "log_E"):
        out[k] = np.array([float(r[k]) for r in rows])
    for k in ("sectors", "B", "n"):
        if k in rows[0]:
            out[k] = np.array([float(r[k] or 0) for r in rows])
    out["rank"] = [r.get("rank") for r in rows]
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


def figure_comparison(plt):
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.34})
    series = [
        ("empirical_yield_duration_{}.csv", "C7",
         r"baseline: full $\mathcal{S}_{n_t}$, duration blocks", "-", 1.0),
        ("sic_sectorblocks_yield_duration_{}.csv", "C0",
         "(a) SIC blocks, full group", "-", 1.5),
        ("sic_partial_yield_duration_{}.csv", "C1",
         "(b) duration blocks, within-SIC group", "--", 1.3),
    ]
    for j, cls in enumerate(CLASSES):
        ax = axes.ravel()[j]
        for (a, b) in CRISIS:
            ax.axvspan(tfloat(np.array([a]))[0], tfloat(np.array([b]))[0],
                       color="0.85", alpha=0.6, lw=0, zorder=0)
        for pat, col, lab, ls, lw in series:
            d = read(BASE / "results" / pat.format(cls))
            if d is None:
                continue
            ax.plot(tfloat(d["date"]), d["log_M"], lw=lw, ls=ls, color=col,
                    label=lab, zorder=3)
        ax.axhline(np.log(20), ls=":", c="k", lw=0.7, zorder=2)
        ax.axhline(0.0, c="0.6", lw=0.5, zorder=1)
        ax.set_title(f"({chr(97 + j)}) {cls}", fontweight="bold")
        ax.set_ylabel(r"$\log M_k$", labelpad=1)
    handles, labels = axes.ravel()[1].get_legend_handles_labels()
    handles.append(Patch(facecolor="0.85", label=CRISIS_LABEL))
    labels.append(CRISIS_LABEL)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(r"Does conceding sector structure rescue the null?  "
                 rf"$X=\mathtt{{yield}}$, $N={N_PERMS}$", y=1.0)
    save(fig, "fig_sic_comparison")
    plt.close(fig)


def figure_eprocess(plt, tag="sic_sectorblocks_yield_duration",
                    stem="fig_eprocess_sic_sectorblocks",
                    title=r"(a) SIC divisions as the candidate's partition, "
                          r"full symmetric group"):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 6.1),
                             gridspec_kw={"height_ratios": [1, 1.5, 1, 1.5],
                                          "hspace": 0.55, "wspace": 0.40})
    cap = np.log(N_PERMS + 1)
    for j, cls in enumerate(CLASSES):
        d = read(BASE / "results" / f"{tag}_{cls}.csv")
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
        nsec = int(np.median(d.get("sectors", np.zeros(1))))
        ax1.set_title(f"({chr(97 + j)}) {cls}  ({nsec} sectors)",
                      fontweight="bold", pad=3)
        ax1.set_ylabel(r"$\log \hat E_t$", labelpad=1)
        ax1.tick_params(labelbottom=False)
        ax1.set_yscale("symlog", linthresh=10, linscale=0.7)
        lo = min(-12.0, 1.6 * float(d["log_E"].min()))
        ax1.set_ylim(lo, cap * 2.2)
        pairs = [(v, lab) for v, lab in
                 ((-100.0, "$-100$"), (-10.0, "$-10$"), (0.0, "$0$")) if v >= lo]
        ax1.set_yticks([v for v, _ in pairs])
        ax1.set_yticklabels([lab for _, lab in pairs])
        ax1.annotate(r"$\log(N{+}1)$", xy=(0.02, cap),
                     xycoords=("axes fraction", "data"), ha="left", va="bottom",
                     fontsize=5.5, color="0.35")
        ax2.plot(t, d["log_M"], lw=1.2, color="C3", zorder=3)
        ax2.axhline(np.log(20), ls="--", c="k", lw=0.7, zorder=2)
        ax2.axhline(0.0, c="0.6", lw=0.5, zorder=1)
        ax2.set_ylabel(r"$\log M_k$", labelpad=1)
    fig.legend(handles=[Patch(facecolor="0.85", label=CRISIS_LABEL),
                        Line2D([], [], ls="--", c="k", lw=0.7,
                               label=r"no evidence / threshold $1/\alpha=20$"),
                        Line2D([], [], ls=":", c="0.4", lw=0.7,
                               label=r"per-month ceiling $\log(N{+}1)$")],
               loc="lower center", ncol=1, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.035))
    fig.suptitle(title, y=0.995)
    save(fig, stem)
    plt.close(fig)


def main():
    plt = style()
    (BASE / "plots").mkdir(exist_ok=True)
    if read(BASE / "results" / "sic_sectorblocks_yield_duration_AAA.csv") is None:
        raise SystemExit("no SIC results found — run scripts/run_sic.py first")
    print("SIC figures:")
    figure_comparison(plt)
    figure_eprocess(plt)
    figure_eprocess(plt, tag="sic_partial_yield_duration",
                    stem="fig_eprocess_sic_partial",
                    title=r"(b) partial exchangeability: duration blocks, "
                          r"permutations within SIC division")


if __name__ == "__main__":
    main()
