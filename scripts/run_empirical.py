"""Corrected empirical monitor over the corporate bond panel (thesis Section 3.6).

For each rating class and each partitioning covariate Y in {duration, spread},
runs the corrected pipeline (lagged-Y blocks, destination-PIT, marginal factors
retained, identity-adjoined Monte Carlo) and writes:

  results/empirical_<Y>_<class>.csv   per-date logE, rank, n_t, B, logM
  plots/empirical_<Y>.png             6-class figure (monthly logE + logM)
  plots/empirical_averaged.png        the averaged-across-Y process, Eq. (3.21)
  results/empirical_summary.txt       the numbers for the [TODO]s of Section 3.6

Optional notch-level runs for the speculative grades (Section 3.6.4):
  --notches 14 15 16

Usage:  python scripts/run_empirical.py [--n-perms 1000] [--classes AAA AA ...]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate import build_candidate, evaluate_date  # noqa: E402
from src.evalue import (  # noqa: E402
    EProcess, mc_permutation_log_evalue, legacy_log_evalue,
)
from src.panel import load_panel, iter_class_dates, RATING_CLASSES  # noqa: E402
from src.residual import residualize_item, MODES  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
CSV_DEFAULT = BASE.parent / "CorpBond_Reconciling" / "corp_jkp_mergedv2.csv"
# Official NBER recession months (peak month through trough month):
#   Great Recession     2007-12 -- 2009-06  (18 months)
#   COVID-19 recession  2020-02 -- 2020-04  (2 months)
CRISIS_WINDOWS = [(200712, 200906), (202002, 202004)]


def run_class(df, rating, target, y_cov, n_perms, alpha=0.05, candidate_mode="corrected",
              target_mode="level", scale_model="affine", window_months=12,
              reference_share=0.0, map_form="affine",
              cross_scale="none"):
    """Monitor one class.  Both the corrected and the naive (identity-omitted)
    estimators are computed from the SAME permutation draws, so their
    difference isolates the Monte Carlo defect exactly.

    ``candidate_mode="legacy"`` additionally reproduces the legacy candidate:
    contemporaneous-Y blocks, origin-PIT scoring and no marginal factors.

    ``target_mode`` selects the object being tested (Section 3.8.2): "level"
    is the mean-corrected target of Sections 3.6-3.7, "y" and "self" replace it
    by the innovation around a location-scale map fitted on the window.  Only
    the target changes; the partition is on the lagged Y throughout.

    ``reference_share`` splits the window: that share of its oldest months
    estimates the map, the rest supplies the candidate (``src.residual``
    explains why one window cannot do both jobs).  It is ignored when
    ``target_mode="level"``.
    """
    legacy_cand = candidate_mode == "legacy"
    proc = EProcess(alpha=alpha)
    for item in iter_class_dates(df, rating, target, y_cov,
                                 window_months=window_months):
        if item is None:
            continue
        if target_mode != "level" and not item.get("degenerate"):
            item = residualize_item(item, mode=target_mode, scale_model=scale_model,
                                    reference_share=reference_share,
                                    map_form=map_form, cross_scale=cross_scale)
        if item.get("degenerate"):
            proc.update(item["date"], 0.0, None, n=0, B=0, log_e_legacy=0.0)
            continue
        cand = build_candidate(
            item["x"], item["y_cur"] if legacy_cand else item["y_lag"],
            item["bond_ids"], item["window_x_by_bond"], item["pooled_window_x"],
            item["window_month_list"],
            pit_mode="origin" if legacy_cand else "destination",
            include_marginals=not legacy_cand,
        )
        if cand is None:
            proc.update(item["date"], 0.0, None, n=item["x"].size, B=0, log_e_legacy=0.0)
            continue
        rng = np.random.default_rng(42 + item["window_month_list"][-1])
        ll_orig, ll_perms = evaluate_date(cand, item["x"], n_perms, rng)
        log_e, rank = mc_permutation_log_evalue(ll_orig, ll_perms)
        m_hat = item.get("map")
        proc.update(item["date"], log_e, rank,
                    n=item["x"].size, B=len(cand.block_positions),
                    slope=(None if m_hat is None else m_hat.b),
                    log_slope=(None if m_hat is None else m_hat.d),
                    log_e_legacy=legacy_log_evalue(ll_orig, ll_perms))
    return proc


def _dates_to_float(dates):
    d = np.asarray(dates)
    return (d // 100) + ((d % 100) - 0.5) / 12.0


def save_csv(proc, path):
    import csv
    log_m = proc.log_m
    log_m_legacy = np.cumsum([m.get("log_e_legacy", 0.0) for m in proc.meta])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "n", "B", "log_E", "rank", "log_M",
                    "log_E_naive", "log_M_naive", "map_slope", "map_log_slope"])
        for i, d in enumerate(proc.dates):
            meta = proc.meta[i]
            sl, ls = meta.get("slope"), meta.get("log_slope")
            w.writerow([d, meta.get("n"), meta.get("B"),
                        f"{proc.log_e[i]:.6f}", proc.ranks[i], f"{log_m[i]:.6f}",
                        f"{meta.get('log_e_legacy', 0.0):.6f}",
                        f"{log_m_legacy[i]:.6f}",
                        "" if sl is None else f"{sl:.6f}",
                        "" if ls is None else f"{ls:.6f}"])


def plot_grid(processes, title, path, n_perms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(processes.keys())
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.25,
                          height_ratios=[1, 1.6, 1, 1.6])
    for j, cls in enumerate(classes):
        proc = processes[cls]
        row0 = 2 * (j // 3)
        col = j % 3
        t = _dates_to_float(proc.dates)
        ax1 = fig.add_subplot(gs[row0, col])
        ax2 = fig.add_subplot(gs[row0 + 1, col], sharex=ax1)
        ax1.plot(t, proc.log_e, lw=0.7, color="steelblue")
        ax1.axhline(0, c="k", lw=0.5)
        ax1.axhline(np.log(n_perms + 1), ls=":", c="gray", lw=0.7)
        ax1.set_title(f"({chr(97 + j)}) {cls}", fontsize=10, fontweight="bold")
        ax1.set_ylabel(r"$\log \hat E_t$", fontsize=8)
        ax1.tick_params(labelbottom=False, labelsize=7)
        ax2.plot(t, proc.log_m, lw=1.4, color="firebrick")
        ax2.axhline(np.log(20), ls="--", c="k", lw=0.7)
        ax2.set_ylabel(r"$\log M_k$", fontsize=8)
        ax2.tick_params(labelsize=7)
        for (w0, w1) in CRISIS_WINDOWS:
            for ax in (ax1, ax2):
                ax.axvspan(_dates_to_float(np.array([w0]))[0],
                           _dates_to_float(np.array([w1]))[0],
                           color="orange", alpha=0.12)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize(processes_by_y, df, target, out_path, n_perms):
    """Compute the quantities the thesis Section 3.6 [TODO]s ask for."""
    import polars as pl
    lines = []

    def w(s=""):
        lines.append(s)
        print(s)

    for y_cov, procs in processes_by_y.items():
        w(f"===== Y = {y_cov} =====")
        terminals = {}
        for cls, proc in procs.items():
            log_m = proc.log_m
            terminals[cls] = float(log_m[-1])
            ceiling_share = float(np.mean([r == 1 for r in proc.ranks if r is not None]))
            crossed = proc.first_alarm()
            in_windows = 0.0
            for (w0, w1) in CRISIS_WINDOWS:
                mask = [(d >= w0) and (d <= w1) for d in proc.dates]
                in_windows += float(np.sum(np.asarray(proc.log_e)[mask]))
            share = in_windows / log_m[-1] if log_m[-1] > 0 else np.nan
            first_str = ("never" if crossed is None
                         else str(proc.dates[crossed]))
            naive_terminal = float(np.sum([m.get("log_e_legacy", 0.0) for m in proc.meta]))
            naive_max_step = max([m.get("log_e_legacy", 0.0) for m in proc.meta] or [0.0])
            w(f"  {cls:5s} terminal logM = {log_m[-1]:9.1f} | max logM = "
              f"{log_m.max():9.1f} | first crossing of log20: {first_str:8s} | "
              f"share of terminal logM inside crisis windows = {share:6.2f} | "
              f"share of dates at ceiling (R_t=1) = {ceiling_share:.2f}")
            w(f"        naive (identity omitted): terminal logM = {naive_terminal:12.1f}"
              f" | largest monthly increment = {naive_max_step:9.2f}"
              f" (valid cap log(N+1) = {np.log(n_perms + 1):.2f})")
        order = sorted(terminals, key=terminals.get, reverse=True)
        w(f"  class ordering by terminal evidence: {' > '.join(order)}")
        w(f"  min terminal = {min(terminals.values()):.1f} "
          f"({min(terminals, key=terminals.get)}), "
          f"max terminal = {max(terminals.values()):.1f} "
          f"({max(terminals, key=terminals.get)})")
        w()

    # correlation of monthly logE with the change in class-mean spread (3.6.3)
    ref = "spread" if "spread" in df.columns else target
    spread_mean = (df.group_by(["dates", "rtg"]).agg(pl.col(ref).mean())
                     .sort(["rtg", "dates"]))
    w(f"corr(log E_t, delta class-mean {ref}), per class:")
    for y_cov, procs in processes_by_y.items():
        for cls, proc in procs.items():
            sm = spread_mean.filter(pl.col("rtg") == cls)
            d2v = dict(zip(sm.get_column("dates").to_list(),
                           sm.get_column(ref).to_list()))
            dates = list(proc.dates)
            pairs = []
            for i in range(1, len(dates)):
                if dates[i] in d2v and dates[i - 1] in d2v:
                    pairs.append((proc.log_e[i], d2v[dates[i]] - d2v[dates[i - 1]]))
            if len(pairs) > 10:
                a, b = np.array(pairs).T
                w(f"  Y={y_cov:9s} {cls:5s} corr = {np.corrcoef(a, b)[0, 1]:+.3f}")
    w()

    # pooled within-(date,class) correlations of the mean-corrected target with
    # each partitioning covariate (3.6.1)
    for c in processes_by_y:
        sub = df.drop_nulls(subset=[f"{target}_mc", f"{c}_mc"])
        corr = sub.select(pl.corr(f"{target}_mc", f"{c}_mc")).item()
        w(f"pooled correlation of {target}_mc with {c}_mc: {corr:+.3f}")

    Path(out_path).write_text("\n".join(lines))
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_DEFAULT))
    ap.add_argument("--target", default="yield")
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument("--classes", nargs="*", default=RATING_CLASSES)
    ap.add_argument("--partition-covs", nargs="*", default=["duration", "spread"])
    ap.add_argument("--notches", nargs="*", type=int, default=[])
    ap.add_argument("--candidate", choices=["corrected", "legacy"], default="corrected",
                    help="'legacy' reproduces the old candidate (contemporaneous-Y "
                         "blocks, origin-PIT, marginal factors deleted)")
    ap.add_argument("--target-mode", choices=["level", *MODES], default="level",
                    help="Section 3.8.2: 'level' tests the mean-corrected target, "
                         "'y' the innovation around the lagged-Y map, 'self' the "
                         "innovation around the bond's own previous value")
    ap.add_argument("--scale-model", choices=["none", "const", "affine"],
                    default="affine",
                    help="the scale half of m_hat; 'affine' is the location-scale "
                         "map of Section 3.8.2, 'const' divides by one number, "
                         "'none' subtracts the location only")
    ap.add_argument("--map-form", choices=["affine", "binned"], default="affine",
                    help="'binned' replaces the affine map by a nonparametric "
                         "one fitted at the reference window's own quantiles, "
                         "which answers whether a rejection is curvature in the "
                         "Y-X relation rather than a change in it")
    ap.add_argument("--cross-scale", choices=["none", "mad"], default="none",
                    help="'mad' divides every cross-section by its own robust "
                         "scale.  This is a symmetric function of the "
                         "cross-section, so it does not disturb the null; it "
                         "relieves the candidate of having to model the overall "
                         "dispersion of a month")
    ap.add_argument("--window-months", type=int, default=None,
                    help="estimation window; defaults to 12 for the level target "
                         "and 24 for the innovation targets, which need a "
                         "reference half and a candidate half")
    ap.add_argument("--reference-share", type=float, default=None,
                    help="share of the window's OLDEST months used to fit the "
                         "map; the rest fits the candidate.  Defaults to 0.25 for "
                         "the innovation targets, which scripts/sweep_window_split.py "
                         "selects: the map has four parameters and the candidate a "
                         "Gamma and a Frank theta per block, so the data is worth "
                         "far more to the second.  0 is one window doing both jobs "
                         "(Section 3.8.2 read literally)")
    args = ap.parse_args()
    if args.window_months is None:
        args.window_months = 12 if args.target_mode == "level" else 24
    if args.reference_share is None:
        args.reference_share = 0.0 if args.target_mode == "level" else 0.25
    tag = "" if args.candidate == "corrected" else "_legacycand"
    if args.target_mode != "level":
        tag += f"_res{args.target_mode}"
        if args.scale_model != "affine":
            tag += f"_{args.scale_model}"
        if args.map_form != "affine":
            tag += f"_{args.map_form}"
        if args.cross_scale != "none":
            tag += f"_cs{args.cross_scale}"
        if args.reference_share == 0.0:
            tag += "_nosplit"        # Section 3.8.2 read literally
        elif abs(args.reference_share - 0.25) > 1e-9:
            tag += f"_ref{int(round(100 * args.reference_share))}"
        if args.window_months != 24:
            tag += f"_K{args.window_months}"
    # the target is part of the identity of a run, not just the partition
    stem = f"empirical{tag}_{args.target}"

    print("Loading panel ...")
    df = load_panel(args.csv, target=args.target,
                    partition_covs=tuple(args.partition_covs))
    (BASE / "results").mkdir(exist_ok=True)
    (BASE / "plots").mkdir(exist_ok=True)

    processes_by_y = {}
    for y_cov in args.partition_covs:
        procs = {}
        for cls in args.classes:
            t0 = time.time()
            proc = run_class(df, cls, args.target, y_cov, args.n_perms,
                             candidate_mode=args.candidate,
                             target_mode=args.target_mode,
                             scale_model=args.scale_model,
                             window_months=args.window_months,
                             reference_share=args.reference_share,
                             map_form=args.map_form,
                             cross_scale=args.cross_scale)
            procs[cls] = proc
            save_csv(proc, BASE / "results" / f"{stem}_{y_cov}_{cls}.csv")
            naive = np.cumsum([m.get("log_e_legacy", 0.0) for m in proc.meta])[-1]
            print(f"[{y_cov} | {cls:4s}] {len(proc.dates)} dates, terminal logM = "
                  f"{proc.log_m[-1]:9.1f} (naive MC: {naive:11.1f})"
                  f"   ({time.time() - t0:.0f}s)", flush=True)
        processes_by_y[y_cov] = procs
        xdesc = {"level": f"X = {args.target} (mean-corrected)",
                 "y": f"X = {args.target} innovation around the lagged-{y_cov} map",
                 "self": f"X = {args.target} innovation around its own lag",
                 }[args.target_mode]
        if args.target_mode != "level":
            xdesc += (f", K = {args.window_months} "
                      f"({args.reference_share:.0%} reference)")
        plot_grid(procs,
                  f"{xdesc}, Y = {y_cov} "
                  f"({'contemporaneous, legacy' if args.candidate == 'legacy' else 'lagged'}, "
                  f"mean-corrected), N = {args.n_perms}",
                  BASE / "plots" / f"{stem}_{y_cov}.png", args.n_perms)

    # averaged-across-partitions process, Eq. (3.21): E_bar_t = mean_k E_t^(k)
    if len(processes_by_y) > 1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)
        for j, cls in enumerate(args.classes):
            series = []
            for y_cov, procs in processes_by_y.items():
                p = procs[cls]
                series.append(dict(zip(p.dates, p.log_e)))
            common = sorted(set.intersection(*[set(s) for s in series]))
            log_avg = [float(np.log(np.mean([np.exp(min(s[d], 700)) for s in series])))
                       for d in common]
            ax = axes.ravel()[j]
            ax.plot(_dates_to_float(common), np.cumsum(log_avg), lw=1.4, color="darkgreen")
            ax.axhline(np.log(20), ls="--", c="k", lw=0.7)
            ax.set_title(f"({chr(97 + j)}) {cls}", fontsize=10, fontweight="bold")
            ax.set_ylabel(r"$\log \bar M_k$", fontsize=8)
        fig.suptitle("Averaged process across partitioning covariates, Eq. (3.21)")
        fig.tight_layout()
        fig.savefig(BASE / "plots" / f"{stem}_averaged.png", dpi=200)
        plt.close(fig)

    summarize(processes_by_y, df, args.target,
              BASE / "results" / f"{stem}_summary.txt", args.n_perms)

    # notch-level runs for the speculative grades (Section 3.6.4): own
    # mean-correction cell per notch via rating_col="nrtg"
    if args.notches:
        df_notch = load_panel(args.csv, target=args.target, rating_col="nrtg")
        y_cov = args.partition_covs[-1]
        for notch in args.notches:
            proc = run_class(df_notch, str(notch), args.target, y_cov, args.n_perms,
                             target_mode=args.target_mode,
                             scale_model=args.scale_model,
                             window_months=args.window_months,
                             reference_share=args.reference_share,
                             map_form=args.map_form,
                             cross_scale=args.cross_scale)
            save_csv(proc, BASE / "results" / f"{stem}_{y_cov}_nrtg{notch}.csv")
            terminal = proc.log_m[-1] if len(proc.dates) else float("nan")
            print(f"[notch {notch} | Y={y_cov}] {len(proc.dates)} dates, "
                  f"terminal logM = {terminal:9.1f}")


if __name__ == "__main__":
    main()
