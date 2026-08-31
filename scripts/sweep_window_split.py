"""How the reference/candidate split trades size against power (Section 3.8.2).

`run_residual_synthetic.py` fixes K = 24 with the older half fitting the map.
That is a choice, and this script is what justifies it rather than asserting it.

The trade-off has two sides and they pull against each other:

  * a longer REFERENCE window W_ref estimates the map more precisely, so less of
    its own estimation error leaks into the residual (Remark, Section 3.8.1) ---
    but it is also staler, so a relation that drifts is described by a map
    averaged over a longer stretch of history that no longer applies;
  * a longer CANDIDATE window W_cand fits the marginals and copula parameters of
    q_t on more data, so the candidate is sharper --- but it is slower to tilt
    after a break, because the break has to displace a larger average before the
    blocks separate.

Neither effect is visible from a single setting, so we sweep both. For each
(K, reference share) the script reports, on panels whose relation-stability is
known by construction:

  size    the rejection rate on the regime that is the target's own null
  power   the rejection rate of a process RESTARTED at the break, so evidence
          banked before it cannot flatter the number
  delay   median months from the break to that crossing
  growth  mean log E_t before and after the break, which is the uncensored
          version of the same thing and does not saturate at log(N+1)

The break is placed K + 24 months in and the panel run for 36 months past it, so
every setting sees the same number of evaluated dates on each side of the break
and the comparison across K is not confounded by sample length.

Usage:  python scripts/sweep_window_split.py [--reps 10] [--n-perms 199]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.candidate import build_candidate, evaluate_date  # noqa: E402
from src.evalue import EProcess, mc_permutation_log_evalue  # noqa: E402
from src.residual import residualize_item  # noqa: E402
from run_residual_synthetic import _item, simulate_panel  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
PLOTS, RESULTS = BASE / "plots", BASE / "results"

PRE, POST = 24, 36          # evaluated dates each side of the break, every K
# (regime, target, is the regime this target's null?)
CELLS = [("stable_relation", "y", True), ("relation_break", "y", False),
         ("random_walk", "self", True), ("drift_subset", "self", False)]


def run(regime, target, K, share, n_perms, rep, alpha, break_month,
        t_total, max_block_size, drift):
    rng_panel = np.random.default_rng(2000 + rep)
    X, Y, ids = simulate_panel(regime, rng_panel, t_total=t_total, drift=drift,
                               break_month=break_month)
    proc = EProcess(alpha=alpha)
    for t in range(K + 1, t_total):
        rng = np.random.default_rng(10_000 * rep + t)
        item = _item(X, Y, ids, t, K)
        item = residualize_item(item, mode=target, reference_share=share)
        if item.get("degenerate"):
            proc.update(t, 0.0, None)
            continue
        cand = build_candidate(item["x"], item["y_lag"], item["bond_ids"],
                               item["window_x_by_bond"], item["pooled_window_x"],
                               item["window_month_list"],
                               max_block_size=max_block_size)
        if cand is None:
            proc.update(t, 0.0, None)
            continue
        log_e, rank = mc_permutation_log_evalue(
            *evaluate_date(cand, item["x"], n_perms, rng))
        proc.update(t, log_e, rank)
    return proc


def _restarted(proc, month, thr):
    """(crossed?, delay) for the process restarted at ``month``."""
    dates = np.asarray(proc.dates)
    sel = dates >= month
    if not sel.any():
        return False, np.nan
    lm = np.cumsum(np.asarray(proc.log_e)[sel])
    hit = np.nonzero(lm >= thr)[0]
    return (True, float(dates[sel][hit[0]] - month)) if hit.size else (False, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-perms", type=int, default=199)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--windows", nargs="*", type=int, default=[12, 18, 24, 36])
    ap.add_argument("--shares", nargs="*", type=float,
                    default=[0.0, 0.25, 0.33, 0.5, 0.67, 0.75])
    ap.add_argument("--max-block-size", type=int, default=10)
    ap.add_argument("--drift", type=float, default=0.8)
    args = ap.parse_args()
    thr = float(np.log(1.0 / args.alpha))

    rows = []
    for regime, target, is_null in CELLS:
        print(f"\n===== {regime} / {target} map "
              f"({'NULL' if is_null else 'alternative'}) =====")
        print(f"  {'K':>3s} {'ref':>5s} {'W_ref':>6s} {'W_cand':>7s} | "
              f"{'reject':>7s} {'rej|break':>10s} {'delay':>6s} | "
              f"{'mean logE pre':>14s} {'post':>8s}")
        for K in args.windows:
            brk = K + PRE
            t_total = brk + POST
            for share in args.shares:
                st = {k: [] for k in ("rej", "rej_b", "delay", "pre", "post")}
                for rep in range(args.reps):
                    proc = run(regime, target, K, share, args.n_perms, rep,
                               args.alpha, brk, t_total, args.max_block_size,
                               args.drift)
                    st["rej"].append(proc.first_alarm() is not None)
                    c, d = _restarted(proc, brk, thr)
                    st["rej_b"].append(c)
                    st["delay"].append(d)
                    le = np.asarray(proc.log_e)
                    dt = np.asarray(proc.dates)
                    st["pre"].append(float(le[dt < brk].mean()))
                    st["post"].append(float(le[dt >= brk].mean()))
                n_ref = max(1, int(round(share * K))) if share > 0 else K
                n_cand = K - n_ref if share > 0 else K
                delay = (float(np.nanmedian(st["delay"]))
                         if np.any(np.isfinite(st["delay"])) else float("nan"))
                row = {"regime": regime, "target": target,
                       "is_null_for_target": is_null, "K": K,
                       "reference_share": share, "n_ref": n_ref,
                       "n_cand": n_cand,
                       "rejection_rate": float(np.mean(st["rej"])),
                       "rejection_rate_after_break": float(np.mean(st["rej_b"])),
                       "median_delay": delay,
                       "mean_log_e_pre": float(np.mean(st["pre"])),
                       "mean_log_e_post": float(np.mean(st["post"]))}
                rows.append(row)
                print(f"  {K:3d} {share:5.2f} {n_ref:6d} {n_cand:7d} | "
                      f"{row['rejection_rate']:7.2f} "
                      f"{row['rejection_rate_after_break']:10.2f} "
                      f"{delay:6.1f} | {row['mean_log_e_pre']:14.3f} "
                      f"{row['mean_log_e_post']:8.3f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "window_split_sweep.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    _figure(rows, args)
    print(f"\nWrote {out} and {PLOTS / 'fig_window_split_sweep.pdf'}")


def _figure(rows, args):
    """Size on the left of each pair, growth on the right.

    Power is plotted as the mean monthly log E_t after the break, not as a
    rejection rate: the rejection rate saturates at one over most of the grid
    and so distinguishes nothing, while the growth rate is uncensored and is
    what actually determines how fast a real monitor would cross.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    panels = [(("stable_relation", "y"), "size", "(a) size, $Y$ map\nstable relation"),
              (("relation_break", "y"), "growth", "(b) power, $Y$ map\nrelation break"),
              (("random_walk", "self"), "size", "(c) size, own-lag map\nrandom walk"),
              (("drift_subset", "self"), "growth", "(d) power, own-lag map\nsubset drifts")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5),
                             gridspec_kw={"wspace": 0.40, "top": 0.70})
    cmap = plt.get_cmap("viridis")
    Ks = sorted({r["K"] for r in rows})
    for ax, (key, kind, title) in zip(axes, panels):
        sel = [r for r in rows if (r["regime"], r["target"]) == key]
        for i, K in enumerate(Ks):
            sub = sorted((r for r in sel if r["K"] == K),
                         key=lambda r: r["reference_share"])
            y = [r["rejection_rate"] if kind == "size" else r["mean_log_e_post"]
                 for r in sub]
            ax.plot([r["reference_share"] for r in sub], y, "o-", ms=3, lw=1.1,
                    color=cmap(i / max(1, len(Ks) - 1)), label=f"$K={K}$")
        ax.axhline(args.alpha if kind == "size" else 0.0, ls="--", c="k", lw=0.7)
        ax.set_xlabel("reference share")
        ax.set_ylabel("rejection rate" if kind == "size"
                      else r"mean $\log \hat E_t$ after", labelpad=1)
        if kind == "size":
            ax.set_ylim(-0.03, 0.45)
        ax.set_title(title, pad=4, fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Give the map the minimum it needs and the rest to the "
                 "candidate\n"
                 f"({args.reps} replications, $N={args.n_perms}$; "
                 "share $0$ is one window doing both jobs)", y=1.02, fontsize=9)
    PLOTS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"fig_window_split_sweep.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
