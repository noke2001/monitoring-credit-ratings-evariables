"""Synthetic validation of the innovation target (thesis Section 3.8.2).

Section 3.5 validated the monitor against panels whose *exchangeability* status
was known.  The change of target changes the hypothesis, so it needs its own
validation, against panels whose *stability of the relation* is known.

Five regimes, each n = 60 entities over T = 72 months with a K = 24 window, so
47 evaluated dates.  In every one of them the LEVEL null (3.1) is false by
construction — that is the point of Section 3.8.1 — and the regimes differ only
in whether the innovation null is false as well.

The window is split: the older half estimates the map, the recent half fits the
candidate.  This is not cosmetic.  With one window doing both jobs — Section
3.8.2 read literally — the least-squares fit leaves the window residuals
homogeneous across blocks by construction, q_t is then nearly exchangeable by
Lemma 3.14, and the monitor has no power against any of the alternatives below;
``--reference-share 0`` reproduces that and the table shows what it costs.

  stable_relation   X_t = a + b Y_{t-1} + exchangeable noise, b fixed.
                    The level is ordered; the relation never moves.  The y-map
                    innovation is exchangeable: this is the NULL for the
                    monitor of Section 3.8.2, and the one the level test
                    cannot distinguish from any of the others below.
  relation_break    as above, but b halves at month 36.  The historical map
                    stops describing the cross-section; nothing else changes.
  dispersion_break  b fixed, but from month 36 the noise scale becomes ordered
                    by Y (long entities get three times the dispersion).  The
                    location map still holds; only the scale half breaks.
  random_walk       X_t = X_{t-1} + exchangeable noise, Y sticky and aligned
                    with the initial level.  The level is maximally persistent
                    and maximally ordered; the increments are exchangeable, so
                    this is the NULL for the own-lag map.
  drift_subset      random walk until month 36, after which the top quarter by
                    Y receives a persistent positive drift of ``--drift``
                    innovation standard deviations (default 0.8): a subset of
                    the cohort ceases to belong, with the rest untouched.  The
                    script also sweeps the drift size, since detection delay
                    against this alternative is entirely a question of how far
                    the subset has moved.

Each regime is run under all three targets (level, y-map innovation, own-lag
innovation) so the table reads as a design matrix: the level column rejects
everywhere and is therefore uninformative; the innovation columns are supposed
to reject only where the corresponding relation actually moved.

Usage:  python scripts/run_residual_synthetic.py [--reps 10] [--n-perms 1000]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate import build_candidate, evaluate_date  # noqa: E402
from src.evalue import EProcess, mc_permutation_log_evalue  # noqa: E402
from src.residual import residualize_item  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
PLOTS = BASE / "plots"
RESULTS = BASE / "results"

BREAK = 36                      # month at which the three break regimes change
WINDOW = 24                     # K: the older half fits the map, the recent the candidate
REFERENCE_SHARE = 0.25   # selected by scripts/sweep_window_split.py
DRIFT_SWEEP = (0.2, 0.4, 0.8, 1.6)
REGIMES = ["stable_relation", "relation_break", "dispersion_break",
           "random_walk", "drift_subset"]
# which target each regime is a null for (None = an alternative for both)
NULL_FOR = {"stable_relation": "y", "random_walk": "self",
            "relation_break": None, "dispersion_break": None,
            "drift_subset": None}
TARGETS = ["level", "y", "self"]


def _equicorr(n, rho):
    return (1 - rho) * np.eye(n) + rho * np.ones((n, n))


def simulate_panel(regime, rng, n=60, t_total=72, rho=0.4, b0=4.0, drift=0.8,
                   break_month=None):
    """Return (X, Y, ids).  Y is sticky, as in Section 3.5.

    ``break_month`` moves the break, which ``scripts/sweep_window_split.py``
    needs: comparing window lengths is only fair if every setting sees the
    same number of evaluated dates on each side of it."""
    brk = BREAK if break_month is None else break_month
    y = np.sort(rng.uniform(0.0, 1.0, size=n))
    L = np.linalg.cholesky(_equicorr(n, rho) + 1e-9 * np.eye(n))
    X = np.empty((t_total, n))

    if regime in ("stable_relation", "relation_break", "dispersion_break"):
        for t in range(t_total):
            b = b0
            s = np.ones(n)
            if regime == "relation_break" and t >= brk:
                b = 0.5 * b0
            if regime == "dispersion_break" and t >= brk:
                s = 1.0 + 2.0 * y                  # scale becomes Y-ordered
            X[t] = 6.0 + b * y + s * (L @ rng.normal(size=n))
    elif regime in ("random_walk", "drift_subset"):
        X[0] = 6.0 + b0 * y + L @ rng.normal(size=n)
        drift_vec = np.where(y >= np.quantile(y, 0.75), drift, 0.0)
        for t in range(1, t_total):
            step = L @ rng.normal(size=n)
            if regime == "drift_subset" and t >= brk:
                step = step + drift_vec
            X[t] = X[t - 1] + step
    else:
        raise ValueError(regime)

    # mean-correct exactly as Section 2.4.2 does on the panel, so the synthetic
    # cross-sections enter the monitor in the same form as the bond ones
    X = X - X.mean(axis=1, keepdims=True)
    Y = y[None, :] + 1e-4 * rng.normal(size=(t_total, n))
    Y = Y - Y.mean(axis=1, keepdims=True)
    ids = np.array([f"e{i}" for i in range(n)])
    return X, Y, ids


def _item(X, Y, ids, t, K):
    window = list(range(t - K, t))
    return {
        "date": t,
        "degenerate": False,
        "x": X[t].copy(),
        "y_lag": Y[t - 1].copy(),
        "x_lag": X[t - 1].copy(),
        "bond_ids": ids,
        "window_x_by_bond": {ids[i]: {m: float(X[m, i]) for m in window}
                             for i in range(ids.size)},
        "window_ylag_by_bond": {ids[i]: {m: float(Y[m - 1, i]) for m in window}
                                for i in range(ids.size)},
        "window_xlag_by_bond": {ids[i]: {m: float(X[m - 1, i]) for m in window}
                                for i in range(ids.size)},
        "pooled_window_x": X[window].ravel(),
        "window_month_list": window,
    }


def run_monitor(X, Y, ids, target, n_perms, alpha=0.05, K=WINDOW, seed=0,
                max_block_size=10, scale_model="affine",
                reference_share=REFERENCE_SHARE, map_form="affine"):
    """One pass of the monitor under one target.  The partition is on the
    lagged Y in every case; only the object being permuted changes."""
    t_total = X.shape[0]
    proc = EProcess(alpha=alpha)
    for t in range(K + 1, t_total):        # +1: the own-lag map needs month t-K-1
        rng = np.random.default_rng(10_000 * seed + t)
        item = _item(X, Y, ids, t, K)
        if target != "level":
            item = residualize_item(item, mode=target, scale_model=scale_model,
                                    reference_share=reference_share,
                                    map_form=map_form)
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
        ll_orig, ll_perms = evaluate_date(cand, item["x"], n_perms, rng)
        log_e, rank = mc_permutation_log_evalue(ll_orig, ll_perms)
        proc.update(t, log_e, rank)
    return proc


def _first_alarm_after(proc, month):
    """First date >= ``month`` at which the process, restarted there, crosses
    1/alpha: the detection delay of a break, uncontaminated by evidence banked
    before it."""
    dates = np.asarray(proc.dates)
    sel = dates >= month
    if not sel.any():
        return np.nan
    lm = np.cumsum(np.asarray(proc.log_e)[sel])
    hit = np.nonzero(lm >= proc.threshold)[0]
    return float(dates[sel][hit[0]] - month) if hit.size else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-block-size", type=int, default=10)
    ap.add_argument("--scale-model", choices=["none", "const", "affine"],
                    default="affine")
    ap.add_argument("--map-form", choices=["affine", "binned"], default="affine")
    ap.add_argument("--window-months", type=int, default=WINDOW)
    ap.add_argument("--reference-share", type=float, default=REFERENCE_SHARE,
                    help="0 reproduces Section 3.8.2 read literally, with one "
                         "window fitting both the map and the candidate")
    ap.add_argument("--drift", type=float, default=0.8,
                    help="size of the drift_subset shift, in innovation sds")
    ap.add_argument("--no-drift-sweep", action="store_true")
    args = ap.parse_args()

    rows, example = [], {}
    hdr = (f"{'regime':17s} {'target':6s} {'null?':6s} {'rej':>4s} "
           f"{'med maxlogM':>12s} {'med termlogM':>13s} {'meanE':>7s} "
           f"{'ceil%':>6s} {'rej>br':>7s} {'delay':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for regime in REGIMES:
        for target in TARGETS:
            st = {k: [] for k in ("rej", "maxlogM", "term", "meanE", "ceil",
                                  "delay")}
            for rep in range(args.reps):
                rng = np.random.default_rng(2000 + rep)
                X, Y, ids = simulate_panel(regime, rng, drift=args.drift)
                proc = run_monitor(X, Y, ids, target, args.n_perms,
                                   alpha=args.alpha, seed=rep,
                                   K=args.window_months,
                                   max_block_size=args.max_block_size,
                                   scale_model=args.scale_model,
                                   reference_share=args.reference_share,
                                   map_form=args.map_form)
                lm = proc.log_m
                st["maxlogM"].append(float(lm.max()))
                st["term"].append(float(lm[-1]))
                st["rej"].append(proc.first_alarm() is not None)
                st["meanE"].append(float(np.mean(np.exp(np.clip(proc.log_e,
                                                                -700, 700)))))
                st["ceil"].append(float(np.mean([r == 1 for r in proc.ranks
                                                 if r is not None])))
                st["delay"].append(_first_alarm_after(proc, BREAK))
                example.setdefault((regime, target), []).append(proc)
            is_null = NULL_FOR[regime] == target
            # a median delay taken over the reps that DID cross says nothing
            # about how many did; report both, and read the delay only when the
            # post-break rejection rate is high
            rej_after = float(np.mean([np.isfinite(d) for d in st["delay"]]))
            row = {"regime": regime, "target": target, "null_for_target": is_null,
                   "rejection_rate": float(np.mean(st["rej"])),
                   "rejection_rate_after_break": rej_after,
                   "median_max_logM": float(np.median(st["maxlogM"])),
                   "median_terminal_logM": float(np.median(st["term"])),
                   "mean_E_t": float(np.mean(st["meanE"])),
                   "share_dates_at_ceiling": float(np.mean(st["ceil"])),
                   "median_delay_after_break": float(np.nanmedian(st["delay"]))
                   if np.any(np.isfinite(st["delay"])) else float("nan")}
            rows.append(row)
            print(f"{regime:17s} {target:6s} {str(is_null):6s} "
                  f"{row['rejection_rate']:4.2f} {row['median_max_logM']:12.1f} "
                  f"{row['median_terminal_logM']:13.1f} {row['mean_E_t']:7.2f} "
                  f"{row['share_dates_at_ceiling']:6.2f} {rej_after:7.2f} "
                  f"{row['median_delay_after_break']:6.1f}", flush=True)
        print()

    RESULTS.mkdir(exist_ok=True)
    suffix = ("" if args.reference_share > 0 else "_nosplit")
    if args.map_form != "affine":
        suffix = f"_{args.map_form}{suffix}"
    out_csv = RESULTS / f"residual_synthetic{suffix}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # the figure shows the MEDIAN across replications with a 10--90% band,
    # not one hand-picked run: an individual trace of a monitor this noisy is
    # not representative of the table above it
    with open(RESULTS / f"residual_synthetic_traces{suffix}.csv", "w",
              newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["regime", "target", "month", "log_M_median",
                     "log_M_p10", "log_M_p90", "log_E_median"])
        for (regime, target), procs in example.items():
            months = procs[0].dates
            lm = np.array([p.log_m for p in procs])
            le = np.array([p.log_e for p in procs])
            for i, m in enumerate(months):
                wr.writerow([regime, target, m,
                             f"{np.median(lm[:, i]):.6f}",
                             f"{np.percentile(lm[:, i], 10):.6f}",
                             f"{np.percentile(lm[:, i], 90):.6f}",
                             f"{np.median(le[:, i]):.6f}"])

    if not args.no_drift_sweep:
        _drift_sweep(args, suffix)
    _figure(example, args, suffix)
    print(f"\nWrote {out_csv}, "
          f"{RESULTS / f'residual_synthetic_traces{suffix}.csv'} and "
          f"{PLOTS / f'residual_synthetic{suffix}.pdf'}")


def _drift_sweep(args, suffix=""):
    """How far must a subset move before the own-lag monitor names it?

    Detection delay against ``drift_subset`` is a pure question of effect size,
    so quoting one number for one drift would be quoting a tuning choice.  This
    sweeps it.  Delay is measured from the break with the process restarted
    there, so evidence banked earlier cannot flatter it."""
    print(f"drift_subset, own-lag map: delay vs effect size "
          f"({args.reps} reps, N = {args.n_perms})")
    print(f"  {'drift (sd)':>11s} {'rej>br':>7s} {'median delay':>13s} "
          f"{'mean logE after break':>22s}")
    rows = []
    for drift in DRIFT_SWEEP:
        rej, delays, post = [], [], []
        for rep in range(args.reps):
            rng = np.random.default_rng(2000 + rep)
            X, Y, ids = simulate_panel("drift_subset", rng, drift=drift)
            proc = run_monitor(X, Y, ids, "self", args.n_perms, alpha=args.alpha,
                               seed=rep, K=args.window_months,
                               max_block_size=args.max_block_size,
                               scale_model=args.scale_model,
                               reference_share=args.reference_share,
                               map_form=args.map_form)
            d = _first_alarm_after(proc, BREAK)
            rej.append(np.isfinite(d))
            delays.append(d)
            after = [e for t, e in zip(proc.dates, proc.log_e) if t >= BREAK]
            post.append(float(np.mean(after)) if after else np.nan)
        med = (float(np.nanmedian(delays)) if np.any(np.isfinite(delays))
               else float("nan"))
        rows.append({"drift_sd": drift,
                     "rejection_rate_after_break": float(np.mean(rej)),
                     "median_delay": med,
                     "mean_log_e_after_break": float(np.nanmean(post))})
        print(f"  {drift:11.2f} {np.mean(rej):5.2f} {med:13.1f} "
              f"{np.nanmean(post):22.3f}", flush=True)
    with open(RESULTS / f"residual_synthetic_drift_sweep{suffix}.csv", "w",
              newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    print()


def _figure(example, args, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"level": "0.55", "y": "tab:blue", "self": "tab:red"}
    labels = {"level": r"level $X^{mc}$",
              "y": r"innovation, lagged-$Y$ map",
              "self": r"innovation, own-lag map"}
    titles = {"stable_relation": "stable relation\n(null for the $Y$ map)",
              "relation_break": f"relation break at $t={BREAK}$",
              "dispersion_break": f"dispersion break at $t={BREAK}$",
              "random_walk": "random walk\n(null for the own-lag map)",
              "drift_subset": (f"subset drifts from $t={BREAK}$\n"
                               f"$({args.drift:g}$ sd$)$")}

    fig, axes = plt.subplots(1, len(REGIMES), figsize=(4.0 * len(REGIMES), 3.6),
                             sharex=True)
    for ax, regime in zip(np.atleast_1d(axes), REGIMES):
        for target in TARGETS:
            procs = example[(regime, target)]
            months = np.asarray(procs[0].dates)
            lm = np.array([p.log_m for p in procs])
            ax.plot(months, np.median(lm, axis=0), lw=1.5,
                    color=colors[target], label=labels[target])
            ax.fill_between(months, np.percentile(lm, 10, axis=0),
                            np.percentile(lm, 90, axis=0),
                            color=colors[target], alpha=0.12, lw=0)
        ax.axhline(np.log(1.0 / args.alpha), ls="--", c="k", lw=0.8)
        ax.axhline(0.0, ls="-", c="0.85", lw=0.8)
        if regime in ("relation_break", "dispersion_break", "drift_subset"):
            ax.axvline(BREAK, ls=":", c="0.3", lw=1.0)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(titles[regime], fontsize=9)
        ax.set_xlabel("month")
    np.atleast_1d(axes)[0].set_ylabel(r"$\log M_k$  (symlog)")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=7)
    split = (f"map on the older {args.reference_share:.0%} of a "
             f"{args.window_months}-month window"
             if args.reference_share > 0 else
             "one window fits both the map and the candidate "
             "(Section 3.8.2 read literally)")
    fig.suptitle("The level rejects everywhere; the innovation rejects only "
                 f"where the relation moved\n{split}", fontsize=11)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(PLOTS / f"residual_synthetic{suffix}.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
