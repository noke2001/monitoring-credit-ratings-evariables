"""Publication figures for Chapter 4, in the house style of make_thesis_figures.py.

Emits vector PDFs sized for \\textwidth in an 11pt report class, plus PNGs for
quick viewing.  Everything is regenerated from the saved panels and the
benchmark CSV -- no model is refitted.

    fig_case_studies      one bond per confusion cell: the rating, the PIT, the
                          single-step e-values and the compounded e-process
    fig_pr_frontier       precision vs recall for every engine and every alpha,
                          against F1 contours and the random-alarm baseline
    fig_gamma_ladder      what each family of summary function achieves
    fig_pit_choice        validity (ACF) against power (lift): why the
                          innovation is ranked and not the level

Usage
    conda activate bond
    export EPROCESS_PANELS=/path/to/panels        # if not in results/
    python scripts/make_ch4_figures.py
    python scripts/make_ch4_figures.py --only pr_frontier
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts" / "eprocess"))

CLASSES = ["AAA", "AA", "A", "BBB", "BB", "B"]

# One palette for the whole chapter.  Colour-blind safe (Okabe-Ito derived),
# and every series stays distinguishable in greyscale by linestyle or marker.
C_RATING = "#1f4e79"
C_PROC = "#c0392b"
C_EVAL = "#95a5a6"
C_PIT = "#2e7d32"
C_ALARM = "#c0392b"
C_CHANGE = "#7b5aa6"
C_THRESH = "#333333"

# A more saturated set for the categorical figures, where the point is to tell
# families apart at a glance rather than to sit quietly behind a time series.
G_PINK = "#d81b60"
G_BLUE = "#1565c0"
G_GREEN = "#2e9e3e"
G_PURPLE = "#8e24aa"
G_AMBER = "#ef6c00"
G_TEAL = "#00838f"
G_GREY = "#78909c"


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
    print(f"  wrote plots/{stem}.pdf / .png")


# ------------------------------------------------------------------ figure 1
def fig_case_studies(args):
    """The rating, the PIT, the e-values and the e-process, one bond per cell.

    This is the figure that shows what the monitor actually does month by
    month, and in particular what a *valid* e-process looks like when nothing
    is happening: it decays, without bound, because every quiet month is a
    small loss on the bet.  That decay is the honest behaviour and is exactly
    what the invalid ``max(1, .)`` floor used to hide.
    """
    import matplotlib.dates as mdates
    plt = style()
    import plot_case_studies as pcs                         # noqa: E402

    df = pcs.build(args.panel, args.alpha, args.engine, args.lam,
                   args.delta, args.pit) if hasattr(pcs, "build") else None
    if df is None:                       # the module does its work in main()
        df = _build_case_panel(args)

    buckets = _pure_classify(df, args.horizon)
    titles = {
        "TP": "true positive — the alarm precedes a transition",
        "FP": "false positive — the alarm fires, no transition follows",
        "FN": "false negative — the transition arrives unannounced",
        "TN": "true negative — a quiescent bond, correctly left alone",
    }
    thresh = 1.0 / args.alpha

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 8.6))
    handles = {}
    for ax, cell in zip(axes, ("TP", "FP", "FN", "TN")):
        isin = _pick(df, buckets[cell])
        if isin is None:
            ax.axis("off"); continue
        b = df[df["isin"] == isin].sort_values("dates")
        ax_p, ax_z = _draw_case(ax, b, thresh, mdates)
        ax.set_title(f"({'ABCD'[list('TP FP FN TN'.split()).index(cell)]}) "
                     f"{titles[cell]}  —  {isin}", loc="left")
        for a in (ax, ax_p, ax_z):
            for h, lab in zip(*a.get_legend_handles_labels()):
                handles.setdefault(lab, h)
    axes[-1].set_xlabel("year")
    fig.legend(handles.values(), handles.keys(), loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(_case_title(args, thresh), y=0.999)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    save(fig, "fig_case_studies")




def _case_title(args, thresh):
    """Name the engine on the figure: which E-process this is matters as much
    as the panel it ran on, and the earlier version said only the latter."""
    if args.engine == "wz":
        eng = (f"Deadzone E-process ($\\delta={args.delta:g}$, "
               f"$\\lambda={args.lam:g}$)")
    else:
        eng = "Tier-velocity gated E-process"
    return (f"{eng} on the {args.panel.upper()} panel\n"
            f"PIT = {args.pit},  $\\alpha={args.alpha:g}$  (threshold "
            f"${thresh:.0f}$),  event = any rating transition within "
            f"{args.horizon} months")


def _pure_classify(df, H):
    """Bonds that are *unambiguous* members of one confusion cell.

    ``plot_case_studies.classify`` assigns a bond to TP as soon as one of its
    alarms anticipates one of its transitions, and to FP when none do.  A bond
    can therefore be labelled TP while also carrying unanticipated alarms, or
    FP while carrying transitions the monitor simply never fired on -- which
    makes it a poor illustration of the cell it is supposed to represent.
    Here each cell demands the whole history agree:

        TP   at least one alarm and one transition, EVERY alarm anticipates a
             transition within H, and EVERY transition is preceded by one
        FP   at least one alarm and NO transitions at all
        FN   at least one transition and NO alarms at all
        TN   neither alarms nor transitions
    """
    from metrics_shim import months_between as _m
    out = {"TP": [], "FP": [], "FN": [], "TN": []}
    for isin, g in df.groupby("isin", sort=False):
        a = g.loc[g["is_alarm"], "dates"].to_numpy()
        c = g.loc[g["is_rating_change"] == 1, "dates"].to_numpy()
        if len(a) and len(c):
            all_a = all(any(0 < _m(cc, aa) <= H for cc in c) for aa in a)
            all_c = all(any(0 < _m(cc, aa) <= H for aa in a) for cc in c)
            if all_a and all_c:
                out["TP"].append(isin)
        elif len(a) and not len(c):
            out["FP"].append(isin)
        elif len(c) and not len(a):
            out["FN"].append(isin)
        elif not len(a) and not len(c):
            out["TN"].append(isin)
    return out


def _pick(df, isins, min_months=72):
    """The longest, clearest representative of a cell."""
    if not isins:
        return None
    g = df[df["isin"].isin(isins)].groupby("isin")
    st = pd.DataFrame({"n": g.size(), "peak": g["M_t"].max()})
    long = st[st["n"] >= min_months]
    st = long if not long.empty else st
    return st.sort_values(["n", "peak"], ascending=[False, False]).index[0]


def _build_case_panel(args):
    """Re-run the engine over the panel; mirrors plot_case_studies.main()."""
    from benchmark_eprocesses import build_pit, load_gbdt, load_vae   # noqa: E402
    from src.sequential import TierVelocityEProcess, DeadzoneEProcess
    df = load_vae() if args.panel == "vae" else load_gbdt()
    df["pit"], tail = build_pit(df, "det_score", args.pit)
    if args.engine == "wz":
        eng = DeadzoneEProcess(alpha=args.alpha, lam=args.lam,
                                     delta=args.delta, tail=tail)
    else:
        eng = TierVelocityEProcess(alpha=args.alpha, tail=tail)
    res = eng.run_sequential_test(df, z_col="pit", cooldown_months=12)
    return res


def _draw_case(ax, b, thresh, mdates):
    ax_p = ax.twinx()
    ax_z = ax.twinx()
    ax_z.spines["right"].set_position(("outward", 34))
    for a in (ax_p, ax_z):
        a.grid(False)
        a.spines["right"].set_visible(True)

    ax.step(b["dates"], b["enc_y"], where="post", color=C_RATING, lw=1.3,
            label="agency rating", zorder=3)
    lo, hi = int(b["enc_y"].min()), int(b["enc_y"].max())
    lo, hi = max(0, lo - 1), min(len(CLASSES) - 1, hi + 1)
    ax.set_yticks(range(lo, hi + 1))
    ax.set_yticklabels([CLASSES[i] for i in range(lo, hi + 1)])
    ax.set_ylim(hi + 0.4, lo - 0.4)
    ax.set_ylabel("rating", color=C_RATING)
    ax.tick_params(axis="y", labelcolor=C_RATING)

    ax_p.bar(b["dates"], b["e_step"] - 1.0, bottom=1.0, width=20,
             color=C_EVAL, alpha=0.55, linewidth=0, zorder=1,
             label=r"single-step $E_t$")
    ax_p.plot(b["dates"], b["M_t"], color=C_PROC, lw=1.0, zorder=4,
              label=r"E-process $M_t$")
    ax_p.axhline(thresh, color=C_THRESH, ls="--", lw=0.8, zorder=2,
                 label=rf"threshold $1/\alpha$")
    ax_p.axhline(1.0, color="0.6", ls=":", lw=0.6, zorder=1)
    ax_p.set_yscale("log")
    ax_p.set_ylim(1e-3, max(thresh * 3.0, float(b["M_t"].max()) * 3.0))
    ax_p.set_ylabel(r"$M_t$, $E_t$", labelpad=0, fontsize=7)

    ax_z.scatter(b["dates"], b["pit"], s=4, marker="D", color=C_PIT,
                 alpha=0.55, linewidths=0, zorder=2, label=r"PIT $Z_t$")
    ax_z.set_ylim(-0.02, 1.02)
    ax_z.set_ylabel(r"$Z_t$", color=C_PIT, labelpad=0, fontsize=7)
    ax_z.tick_params(axis="y", labelcolor=C_PIT)

    for i, d in enumerate(b.loc[b["is_rating_change"] == 1, "dates"]):
        ax.axvline(d, color=C_CHANGE, lw=1.0, alpha=0.55, zorder=2,
                   label="rating change" if i == 0 else None)
    for i, d in enumerate(b.loc[b["is_alarm"], "dates"]):
        ax.axvline(d, color=C_ALARM, lw=0.8, ls="-.", alpha=0.8, zorder=2,
                   label="alarm" if i == 0 else None)

    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_zorder(ax_p.get_zorder() + 1)
    ax.patch.set_visible(False)
    return ax_p, ax_z


# ------------------------------------------------------------------ figure 2
def fig_pr_frontier(_args):
    """Precision against recall, every engine at every alpha.

    The point of plotting rather than tabulating: what the benchmark produces
    is a *frontier*, and which point on it to occupy is an operational choice.
    A table invites the reader to look for the single best row, which does not
    exist.  F1 contours make the trade-off legible, and the base rate is drawn
    as the line a random alarm would sit on.
    """
    plt = style()
    f = BASE / "results" / "eprocess_benchmark_results.csv"
    d = pd.read_csv(f)
    d = d[d["Panel"].str.lower() == "vae"] if "Panel" in d else d
    base = float(d["Base_24m (%)"].iloc[0])

    valid = d[d["anytime_valid"]]
    invalid = d[~d["anytime_valid"]]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    # F1 iso-contours
    rr = np.linspace(0.5, 60, 400)
    for f1 in (0.05, 0.10, 0.15, 0.20, 0.25):
        pp = np.where(2 * rr - f1 * 100 > 0, f1 * 100 * rr / (2 * rr - f1 * 100), np.nan)
        ok = (pp > 0) & (pp < 60)
        ax.plot(rr[ok], pp[ok], color="0.8", ls=":", lw=0.6, zorder=0)
        # label where the contour is still inside the axes, not at its end
        j = np.where(ok & (rr < 25) & (pp < 40))[0]
        if len(j):
            k = j[len(j) // 3]
            ax.annotate(f"$F_1$={f1:.2f}", (rr[k], pp[k]), fontsize=6,
                        color="0.5", ha="left", va="bottom", zorder=0,
                        rotation=-38, rotation_mode="anchor")

    ax.axhline(base, color=C_CHANGE, lw=0.9, ls="--", zorder=1,
               label=f"random alarm (base rate {base:.1f}%)")

    # one marker shape per engine family, one shade per alpha
    fams = {"Deadzone": ("o", G_BLUE), "Mixture-restart": ("s", G_GREEN),
            "Innovation bet": ("^", G_AMBER), "Asym. leaky": ("v", G_PURPLE),
            "Optimal hybrid": ("D", G_PINK), "Tier+velocity": ("P", G_TEAL),
            "AHZ Grenander": ("*", "#5d4037"), "Plug-in Kelly": ("h", "#c2185b"),
            "Kelly x mixture": ("8", "#7cb342")}
    seen = set()
    for _, r in valid.iterrows():
        fam = next((k for k in fams if r["Engine"].startswith(k)), None)
        if fam is None:
            continue
        m, c = fams[fam]
        sz = {0.20: 14, 0.10: 26, 0.05: 40, 0.01: 58}.get(round(r["alpha"], 2), 26)
        ax.scatter(r["R_24m (%)"], r["P_24m (%)"], marker=m, s=sz, color=c,
                   alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3,
                   label=fam if fam not in seen else None)
        seen.add(fam)
    for _, r in invalid.iterrows():
        ax.scatter(r["R_24m (%)"], r["P_24m (%)"], marker="X", s=34,
                   facecolors="none", edgecolors="#c0392b", linewidths=1.0,
                   zorder=3, label="rolling window (INVALID)"
                   if "rolling window (INVALID)" not in seen else None)
        seen.add("rolling window (INVALID)")

    ax.set_xlabel("recall at 24 months (%)")
    ax.set_ylabel("precision at 24 months (%)")
    ax.set_xlim(0, 26); ax.set_ylim(0, 42)
    ax.set_title("Alarm quality is a frontier, not a single operating point",
                 loc="left")
    ax.annotate("marker size grows as $\\alpha$ falls\n(0.20, 0.10, 0.05, 0.01)",
                (0.985, 0.04), xycoords="axes fraction", fontsize=6,
                color="0.4", ha="right")
    ax.legend(loc="upper right", frameon=False, ncol=2, handletextpad=0.4,
              columnspacing=1.0, fontsize=6.2)
    fig.tight_layout()
    save(fig, "fig_pr_frontier")


# ------------------------------------------------------------------ figure 3
def fig_gamma_ladder(_args):
    """What each family of summary function buys, on one axis.

    Three tables in Section 4.3 report the same quantity for different
    families; a reader comparing them has to hold six numbers in their head.
    One dot plot, ordered, does it in a glance.
    """
    plt = style()
    rows = [
        ("no model", "best single covariate (spread)", 0.634, G_GREY),
        ("unsupervised", "VAE reconstruction error", 0.564, G_PURPLE),
        ("supervised", "deviation, fresh-rating context", 0.678, G_BLUE),
        ("supervised", "deviation, transition-only context", 0.673, G_BLUE),
        ("supervised", "deviation, level target $K{=}12$", 0.656, G_BLUE),
        ("supervised", "drift target (best of four)", 0.625, G_BLUE),
        ("supervised", "forward-rating target", 0.552, G_BLUE),
        ("semi-supervised", "downgrade tail mass", 0.745, G_PINK),
        ("semi-supervised", "directional deviation", 0.738, G_PINK),
    ]
    rows = sorted(rows, key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(rows))
    for i, (fam, lab, v, c) in enumerate(rows):
        ax.plot([0.5, v], [i, i], color=c, lw=1.4, alpha=0.55, zorder=1)
        ax.scatter(v, i, s=46, color=c, zorder=3, edgecolors="white", linewidths=0.7)
        ax.annotate(f"{v:.3f}", (v, i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7, color=c)
    ax.axvline(0.5, color="0.5", lw=0.9, ls="--", zorder=0)
    ax.annotate("chance", (0.5, len(rows) - 0.4), xytext=(3, 0),
                textcoords="offset points", fontsize=6.5, color="0.45")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}" for _, lab, _, _ in rows])
    ax.set_xlim(0.49, 0.79)
    ax.set_xlabel("AUC for a rating migration within 12 months")
    ax.set_title("Withholding the label costs more than any architecture recovers",
                 loc="left")
    for fam, c in (("unsupervised", G_PURPLE), ("supervised", G_BLUE),
                   ("semi-supervised", G_PINK), ("no model", G_GREY)):
        ax.scatter([], [], color=c, s=34, label=fam)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    save(fig, "fig_gamma_ladder")


# ------------------------------------------------------------------ figure 4
def fig_pit_choice(_args):
    """Validity against power: the one plot that justifies ranking the innovation.

    The x-axis is the quantity the null requires to be zero, the y-axis is what
    we actually want.  The shaded band is the region where the martingale
    argument survives.  Ranking the level sits far outside it *and* is weaker,
    which is the whole argument in one picture.
    """
    plt = style()
    pts = [
        ("score level, ranked", 0.852, 1.38, G_PINK, False),
        ("score AR(1) innovation, ranked", 0.005, 1.50, G_GREEN, True),
        ("model PIT, level", 0.319, 1.67, G_AMBER, False),
        ("model PIT, AR(1) innovation", -0.034, 1.13, G_BLUE, True),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.axvspan(-0.10, 0.10, color=G_GREEN, alpha=0.09, zorder=0)
    ax.annotate("conditionally uniform\n(martingale survives)", (0.0, 1.735),
                ha="center", va="top", fontsize=6.5, color=G_GREEN)
    ax.axhline(1.0, color="0.5", lw=0.9, ls="--", zorder=1)
    ax.annotate("no lift over a random alarm", (0.88, 1.012), ha="right",
                va="bottom", fontsize=6.5, color="0.45")
    # Explicit label placement: these four points sit close to the axes and to
    # the annotation arrow, so automatic centring runs them off the frame.
    place = {
        "score level, ranked":            (-8, 10, "right", "bottom"),
        "score AR(1) innovation, ranked": (10, -2, "left", "center"),
        "model PIT, level":               (0, 10, "center", "bottom"),
        "model PIT, AR(1) innovation":    (10, 2, "left", "center"),
    }
    for lab, acf, lift, c, ok in pts:
        ax.scatter(acf, lift, s=56, color=c, zorder=4, edgecolors="white",
                   linewidths=0.7, marker="o" if ok else "X")
        dx, dy, ha, va = place[lab]
        ax.annotate(lab, (acf, lift), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=6.8, color=c, zorder=5)
    # The arrow runs BELOW both endpoints so it cannot cross either label.
    # Shallow arc, caption clear beneath it: a deeper curve runs through its
    # own label, which is what the first draft of this figure did.
    ax.annotate("", xy=(0.055, 1.478), xytext=(0.80, 1.352),
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9,
                                connectionstyle="arc3,rad=-0.10"), zorder=2)
    ax.annotate("removing the persistence\nraises power as well",
                (0.44, 1.30), ha="center", va="top", fontsize=6.5,
                color="0.45", zorder=3)
    ax.set_xlabel(r"lag-1 autocorrelation of the PIT  (the null requires $0$)")
    ax.set_ylabel("precision lift at a matched alarm budget")
    ax.set_xlim(-0.12, 0.92); ax.set_ylim(1.0, 1.80)
    ax.set_title("Validity and power are not in conflict here — except for one variant",
                 loc="left")
    ax.scatter([], [], marker="o", color="0.35", s=52, label="usable")
    ax.scatter([], [], marker="X", color="0.35", s=52, label="not conditionally uniform")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, "fig_pit_choice")


FIGS = {"case_studies": fig_case_studies, "pr_frontier": fig_pr_frontier,
        "gamma_ladder": fig_gamma_ladder, "pit_choice": fig_pit_choice}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FIGS), default=None)
    ap.add_argument("--panel", choices=["vae", "gbdt"], default="vae")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--engine", choices=["wz", "tier"], default="wz")
    ap.add_argument("--lam", type=float, default=0.50)
    ap.add_argument("--delta", type=float, default=0.75)
    ap.add_argument("--pit", choices=["ar1", "level", "model"], default="ar1")
    args = ap.parse_args()
    for name, fn in FIGS.items():
        if args.only and name != args.only:
            continue
        print(f"{name} ...")
        fn(args)


if __name__ == "__main__":
    main()
