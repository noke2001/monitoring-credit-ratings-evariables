"""
Confusion-matrix case studies: PIT scores, single-step e-values and the
compounded e-process for one representative bond of each outcome class.

    python tests_and_plots/plot_case_studies.py [--panel vae|gbdt] [--alpha 0.10]

Classification at horizon H (default 24 months), on the same definitions the
benchmark scores:

  TP  an alarm is followed by a rating transition within H months
  FP  the bond alarms, but no alarm is followed by a transition within H months
  FN  the bond transitions, but no alarm precedes any transition within H months
  TN  the bond neither alarms nor transitions

"Transition" means ANY rating change, in either direction -- that is the event
this monitor is built to anticipate.

Each panel carries three aligned series:
  * agency rating (step, left axis, inverted so worse ratings sit lower)
  * e-process M_t and single-step e-values e_t (log right axis, baseline 1.0)
  * PIT score Z_t (far right axis)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.sequential import TierVelocityEProcess, DeadzoneEProcess
from src.metrics import _m

CLASSES = ["AAA", "AA", "A", "BBB", "BB", "B"]
C_RATING, C_PROC, C_EVAL, C_PIT = "#1f4e79", "#d81b7a", "#f7b6d2", "#33a02c"
C_ALARM, C_CHANGE, C_THRESH = "#e41a1c", "#5b2c8d", "#333333"


def classify(df: pd.DataFrame, H: int) -> dict:
    """Bucket every bond into TP / FP / FN / TN at horizon H."""
    out = {"TP": [], "FP": [], "FN": [], "TN": []}
    for isin, g in df.groupby("isin", sort=False):
        a_dates = g.loc[g["is_alarm"], "dates"].to_numpy()
        c_dates = g.loc[g["is_rating_change"] == 1, "dates"].to_numpy()
        anticipated = [
            a for a in a_dates if any(0 < _m(c, a) <= H for c in c_dates)
        ]
        caught = [
            c for c in c_dates if any(0 < _m(c, a) <= H for a in a_dates)
        ]
        if len(a_dates) and anticipated:
            out["TP"].append(isin)
        elif len(a_dates) and not anticipated:
            out["FP"].append(isin)
        elif len(c_dates) and not caught:
            out["FN"].append(isin)
        elif not len(a_dates) and not len(c_dates):
            out["TN"].append(isin)
    return out


def pick(df: pd.DataFrame, isins: list, min_months: int = 72) -> str:
    """Longest-history, most legible representative of a class."""
    if not isins:
        return None
    g = df[df["isin"].isin(isins)].groupby("isin")
    stats = pd.DataFrame({"n": g.size(), "peak": g["M_t"].max()})
    stats = stats[stats["n"] >= min_months]
    if stats.empty:
        stats = pd.DataFrame({"n": g.size(), "peak": g["M_t"].max()})
    # prefer a long series; among those, the clearest signal
    stats = stats.sort_values(["n", "peak"], ascending=[False, False])
    return stats.index[0]


def draw(ax, b: pd.DataFrame, title: str, thresh: float, alpha: float):
    ax_p = ax.twinx()                        # e-process / e-values, log
    ax_z = ax.twinx()                        # PIT
    ax_z.spines["right"].set_position(("outward", 52))

    # --- agency rating -----------------------------------------------------
    ax.step(b["dates"], b["enc_y"], where="post", color=C_RATING, lw=2.6,
            label="Agency rating", zorder=3)
    lo, hi = int(b["enc_y"].min()), int(b["enc_y"].max())
    lo, hi = max(0, lo - 1), min(len(CLASSES) - 1, hi + 1)
    ax.set_yticks(range(lo, hi + 1))
    ax.set_yticklabels([CLASSES[i] for i in range(lo, hi + 1)])
    ax.set_ylim(hi + 0.4, lo - 0.4)          # inverted: worse ratings lower
    ax.set_ylabel("Agency rating", color=C_RATING, fontweight="bold", fontsize=9)
    ax.tick_params(axis="y", labelcolor=C_RATING, labelsize=8)

    # --- single-step e-values as bars around the fair-bet baseline 1.0 -----
    ax_p.bar(b["dates"], b["e_step"] - 1.0, bottom=1.0, width=18,
             color=C_EVAL, alpha=.55, linewidth=0, zorder=1,
             label=r"single-step $e_t$")
    # --- compounded e-process ---------------------------------------------
    ax_p.plot(b["dates"], b["M_t"], color=C_PROC, lw=1.9, zorder=4,
              label=r"e-process $M_t$")
    ax_p.axhline(thresh, color=C_THRESH, ls="--", lw=1.2, zorder=2,
                 label=rf"threshold $1/\alpha={thresh:.0f}$")
    ax_p.axhline(1.0, color="#999999", ls=":", lw=.9, zorder=1)
    ax_p.set_yscale("log")
    # Display floor. During quiet stretches a valid e-process decays without
    # bound (that decay is the honest behaviour -- it is exactly what the
    # max(1,.) floor used to hide). Letting the axis follow it to 1e-6 would
    # compress the region around the alarm threshold, where the reading
    # happens, so the AXIS is clipped while the process itself is untouched.
    FLOOR = 1e-3
    top = max(thresh * 3.0, float(b["M_t"].max()) * 3.0)
    ax_p.set_ylim(FLOOR, top)
    ax_p.set_ylabel(r"$M_t$ / $e_t$  (log, clipped at $10^{-3}$)", fontsize=9)
    ax_p.tick_params(axis="y", labelsize=8)

    # --- PIT ---------------------------------------------------------------
    ax_z.scatter(b["dates"], b["pit"], s=16, marker="D", color=C_PIT,
                 alpha=.6, zorder=2, label=r"PIT $Z_t$")
    ax_z.set_ylim(-0.02, 1.02)
    ax_z.set_ylabel(r"PIT $Z_t$", color=C_PIT, fontsize=9)
    ax_z.tick_params(axis="y", labelcolor=C_PIT, labelsize=8)

    # --- events ------------------------------------------------------------
    for d in b.loc[b["is_rating_change"] == 1, "dates"]:
        ax.axvline(d, color=C_CHANGE, lw=1.6, alpha=.55, zorder=2)
    al = b.loc[b["is_alarm"], "dates"]
    for i, d in enumerate(al):
        ax.axvline(d, color=C_ALARM, lw=1.3, ls="-.", alpha=.75, zorder=2,
                   label="alarm" if i == 0 else None)

    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=.22, zorder=0)
    ax.set_zorder(ax_p.get_zorder() + 1)
    ax.patch.set_visible(False)
    return ax_p, ax_z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["vae", "gbdt"], default="vae")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--engine", choices=["wz", "tier"], default="tier")
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--delta", type=float, default=0.75)
    ap.add_argument("--pit", choices=["ar1", "level", "model"], default="ar1")
    args = ap.parse_args()

    from benchmark_eprocesses import build_pit, load_gbdt, load_vae
    df = load_vae() if args.panel == "vae" else load_gbdt()
    # Ship PIT: rank of the AR(1) innovation of det_score within a
    # (date, monitored-rating) cohort -- the only variant that is uniform
    # conditional on the bond's own past (lag-1 ACF 0.019 vs 0.854 for levels).
    df["pit"], PIT_TAIL = build_pit(df, "det_score", args.pit)

    if args.engine == "tier":
        eng = TierVelocityEProcess(args.alpha, tail=PIT_TAIL)
        kw = {"hazard_col": "hazard"}
        eng_name = "TierVelocityEProcess"
    else:
        eng = DeadzoneEProcess(args.alpha, delta=args.delta,
                                     lam=args.lam, tail=PIT_TAIL)
        kw = {}
        eng_name = (f"DeadzoneEProcess (delta={args.delta}, "
                    f"lambda={args.lam})")

    res = eng.run_sequential_test(df, z_col="pit", cooldown_months=12, **kw)
    buckets = classify(res, args.horizon)
    print("Bond counts by outcome class "
          f"(H={args.horizon}m, alpha={args.alpha}):")
    for k in ("TP", "FP", "FN", "TN"):
        print(f"  {k}: {len(buckets[k]):,}")

    titles = {
        "TP": "(A) True positive — alarm precedes a transition within {H} months",
        "FP": "(B) False positive — alarm fires, no transition within {H} months",
        "FN": "(C) False negative — transition occurs with no preceding alarm",
        "TN": "(D) True negative — quiescent bond, no alarm and no transition",
    }
    fig, axes = plt.subplots(4, 1, figsize=(15.5, 19.0))
    plt.subplots_adjust(hspace=.42, right=.865, left=.06, top=.885, bottom=.07)

    handles = None
    for ax, key in zip(axes, ["TP", "FP", "FN", "TN"]):
        isin = pick(res, buckets[key])
        if isin is None:
            ax.text(.5, .5, f"no {key} bond found", ha="center",
                    transform=ax.transAxes)
            continue
        b = res[res["isin"] == isin].sort_values("dates")
        t = titles[key].format(H=args.horizon) + f"   —   {isin}"
        ax_p, ax_z = draw(ax, b, t, eng.threshold, args.alpha)
        if handles is None:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_p.get_legend_handles_labels()
            h3, l3 = ax_z.get_legend_handles_labels()
            handles = (h1 + h2 + h3, l1 + l2 + l3)
        print(f"  {key}: {isin}  ({len(b)} months, peak M_t={b['M_t'].max():.2f})")

    fig.legend(*handles, loc="lower center", ncol=6, frameon=False,
               fontsize=10, bbox_to_anchor=(.5, .012))
    fig.text(.5, .960,
             "Sequential monitoring case studies — PIT scores, single-step "
             "e-values and e-process",
             ha="center", va="center", fontsize=15, fontweight="bold")
    fig.text(.5, .940,
             f"{eng_name} on the {args.panel.upper()} panel   ·   "
             f"PIT = {args.pit}   ·   "
             r"$\alpha$=" f"{args.alpha} (threshold {1/args.alpha:.0f})   ·   "
             f"event = any rating transition within {args.horizon} months",
             ha="center", va="center", fontsize=10.5, color="#444444")

    for d in (ROOT / "plots",):
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"case_studies_confusion_{args.panel}.png"
        fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
        print(f"saved -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
