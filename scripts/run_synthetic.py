"""Synthetic validation of the corrected pipeline (thesis Section 3.5).

Four regimes with known exchangeability status, plus the legacy generator whose
"null" carried an accidental 0.5*Y location gradient (Section 3.5.2).  For each
regime we run the full corrected monitor and report rejection rate at alpha,
median max_k log M_k and median first-alarm date, reproducing Table 3.3.

Reading the ``meanE`` columns.  These average E_t over the evaluated dates and
then over replications.  E_t is bounded by N+1 = 1001 but extremely
right-skewed, so with only a few hundred draws this sample mean is a poor and
upward-biased-looking estimator of E[E_t]: a single date at the ceiling
contributes ~20 to a 48-date average.  It is a rough diagnostic only, and a
value above 1 on the null row is not evidence of a calibration failure --- the
rejection rate and the downward drift of log M_k are the informative columns
here, and ``scripts/audit_calibration.py`` (400 replications, with standard
errors) is what actually certifies E[E_t] <= 1.

Usage:  python scripts/run_synthetic.py  [--reps 10] [--n-perms 1000]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate import build_candidate, evaluate_date  # noqa: E402
from src.evalue import (  # noqa: E402
    EProcess, mc_permutation_log_evalue, legacy_log_evalue,
)

PLOTS = Path(__file__).resolve().parents[1] / "plots"
RESULTS = Path(__file__).resolve().parents[1] / "results"


def _cov_equicorr(d, rho, sd=1.0):
    return sd**2 * ((1 - rho) * np.eye(d) + rho * np.ones((d, d)))


def _cov_ar(d, phi):
    idx = np.arange(d)
    return phi ** np.abs(idx[:, None] - idx[None, :])


def _cov_tridiag(d, off=0.4):
    c = np.eye(d)
    c[np.arange(d - 1), np.arange(1, d)] = off
    c[np.arange(1, d), np.arange(d - 1)] = off
    return c


def _cov_star(d, rho=0.3):
    c = np.eye(d)
    c[0, 1:] = rho
    c[1:, 0] = rho
    return c


def _block_cov(blocks):
    from scipy.linalg import block_diag
    return block_diag(*blocks)


def make_generator(regime: str, n: int = 60):
    """Return (cov, means) for one monthly cross-section draw."""
    h = n // 2
    if regime == "exchangeable":
        return _cov_equicorr(n, 0.5), np.zeros(n)
    if regime == "simple":
        return _block_cov([_cov_equicorr(h, 0.85), np.eye(n - h)]), np.zeros(n)
    if regime == "complex":
        d = n // 6
        blocks = [_cov_equicorr(d, 0.6), _cov_ar(d, 0.8), _cov_ar(d, -0.6),
                  _cov_tridiag(d), _cov_star(d), np.eye(n - 5 * d)]
        return _block_cov(blocks), np.zeros(n)
    if regime == "marginal_shift":
        means = np.concatenate([np.zeros(h), 0.8 * np.ones(n - h)])
        return _cov_equicorr(n, 0.5), means
    raise ValueError(regime)


def simulate_panel(regime: str, rng, n=60, t_total=60):
    """Sticky partitioning covariate aligned with the structural blocks; the
    target is Gamma-flavoured noise on top of the block structure."""
    y_base = np.sort(rng.uniform(0, 1, size=n))         # aligned with block index
    if regime == "legacy_null":
        # the mislabelled null of Section 3.5.2: 0.5 * Y location gradient
        y_base = np.concatenate([rng.uniform(0.01, 0.03, n // 2),
                                 rng.uniform(0.06, 0.10, n - n // 2)])
        y_base = np.sort(y_base)
        cov, means = _cov_equicorr(n, 0.5, sd=0.014), 0.03 + 0.5 * y_base
    else:
        cov, means = make_generator(regime, n)
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(n))
    X = np.empty((t_total, n))
    gam = rng.gamma(4.0, 0.5, size=(t_total, n)) - 2.0  # centred gamma noise flavour
    # keep the extra noise proportional to the regime's own scale, so the
    # legacy generator's 2.5-sd location gradient is not drowned out
    gam_scale = 0.3 * float(np.sqrt(np.median(np.diag(cov))))
    for t in range(t_total):
        X[t] = 6.0 + means + L @ rng.normal(size=n) + gam_scale * gam[t]
    Y = y_base[None, :] + 1e-4 * rng.normal(size=(t_total, n))
    ids = np.array([f"e{i}" for i in range(n)])
    return X, Y, ids


def run_monitor(X, Y, ids, n_perms, alpha=0.05, K=12, seed=0, max_block_size=10):
    """Both estimators are computed from the same permutation draws, so their
    difference isolates the effect of omitting the identity permutation.

    ``max_block_size`` fixes the resolution of the candidate's partition. It
    matters: the complex regime carries its signal entirely in the copula
    channel, so a partition whose blocks straddle the generator's structural
    boundaries averages the contrast away. At n = 60 the default gives six
    blocks, matching the six-block regimes; see the sweep in the docstring of
    ``scripts/`` or re-run with --max-block-size to reproduce it.
    """
    t_total, n = X.shape
    proc = EProcess(alpha=alpha)
    for t in range(K, t_total):
        rng = np.random.default_rng(10_000 * seed + t)
        window = list(range(t - K, t))
        wxb = {ids[i]: {m: float(X[m, i]) for m in window} for i in range(n)}
        pooled = X[window].ravel()
        cand = build_candidate(X[t], Y[t - 1], ids, wxb, pooled, window,
                               max_block_size=max_block_size)
        if cand is None:
            proc.update(t, 0.0, None, log_e_legacy=0.0)
            continue
        ll_orig, ll_perms = evaluate_date(cand, X[t], n_perms, rng)
        log_e, rank = mc_permutation_log_evalue(ll_orig, ll_perms)
        proc.update(t, log_e, rank, log_e_legacy=legacy_log_evalue(ll_orig, ll_perms))
    return proc


def null_diagnostic(X, Y, K=12):
    """Model-free check that a generator's cross-sections really are
    exchangeable given the lagged partitioning covariate: the median Spearman
    rank correlation between Y_{t-1} and X_t must be ~0 under the null
    (Lemma 3.12 of the thesis; requires no copulas and no permutations)."""
    from scipy.stats import spearmanr
    rhos = [spearmanr(Y[t - 1], X[t]).statistic for t in range(K, X.shape[0])]
    return float(np.median(rhos)), float(np.median(np.abs(rhos)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-block-size", type=int, default=10,
                    help="candidate partition resolution; 10 gives six blocks at n=60")
    ap.add_argument("--plot-legacy-null", action="store_true",
                    help="include the mislabelled legacy generator in the figure "
                         "(it is always kept in the results table)")
    args = ap.parse_args()

    regimes = ["exchangeable", "simple", "complex", "marginal_shift", "legacy_null"]
    rows = []
    example = {}
    print(f"{'regime':16s} {'exch?':6s} {'med rho':>8s} | corrected: "
          f"{'rej':>4s} {'medmaxlogM':>11s} {'meanE':>7s} | naive: "
          f"{'rej':>4s} {'medmaxlogM':>11s} {'meanE':>9s}")
    for regime in regimes:
        st = {k: [] for k in ("reject", "maxlogM", "first", "meanE",
                              "reject_n", "maxlogM_n", "meanE_n", "rho")}
        for rep in range(args.reps):
            rng = np.random.default_rng(1000 + rep)
            X, Y, ids = simulate_panel(regime, rng)
            st["rho"].append(null_diagnostic(X, Y)[0])
            proc = run_monitor(X, Y, ids, args.n_perms, alpha=args.alpha, seed=rep,
                               max_block_size=args.max_block_size)
            log_m = proc.log_m
            log_e_n = np.array([m.get("log_e_legacy", 0.0) for m in proc.meta])
            log_m_n = np.cumsum(log_e_n)
            thr = np.log(1.0 / args.alpha)
            st["maxlogM"].append(float(log_m.max()))
            st["maxlogM_n"].append(float(log_m_n.max()))
            fa = proc.first_alarm()
            st["reject"].append(fa is not None)
            st["reject_n"].append(bool(np.any(log_m_n >= thr)))
            st["first"].append(np.nan if fa is None else fa + 1)
            # mean of E_t itself: must be <= 1 under the null (Lemma 3.14)
            st["meanE"].append(float(np.mean(np.exp(np.clip(proc.log_e, -700, 700)))))
            st["meanE_n"].append(float(np.mean(np.exp(np.clip(log_e_n, -700, 700)))))
            if rep == 0:
                example[regime] = proc
        exch = regime in ("exchangeable",)
        rows.append({
            "regime": regime,
            "truly_exchangeable": exch,
            "median_lagged_rank_corr": float(np.median(st["rho"])),
            "rejection_rate": float(np.mean(st["reject"])),
            "median_max_logM": float(np.median(st["maxlogM"])),
            "mean_E_t": float(np.mean(st["meanE"])),
            "median_first_alarm": float(np.nanmedian(st["first"])),
            "naive_rejection_rate": float(np.mean(st["reject_n"])),
            "naive_median_max_logM": float(np.median(st["maxlogM_n"])),
            "naive_mean_E_t": float(np.mean(st["meanE_n"])),
        })
        r = rows[-1]
        print(f"{regime:16s} {str(exch):6s} {r['median_lagged_rank_corr']:8.3f} | "
              f"           {r['rejection_rate']:4.2f} {r['median_max_logM']:11.2f} "
              f"{r['mean_E_t']:7.2f} |        {r['naive_rejection_rate']:4.2f} "
              f"{r['naive_median_max_logM']:11.2f} {r['naive_mean_E_t']:9.2f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "synthetic_validation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = {"exchangeable": "tab:blue", "simple": "tab:orange",
              "complex": "tab:green", "marginal_shift": "tab:red",
              "legacy_null": "tab:gray"}
    labels = {"exchangeable": "exchangeable (null)", "simple": "simple non-exch.",
              "complex": "complex non-exch.", "marginal_shift": "marginal shift",
              "legacy_null": "legacy 'null' (mislabelled)"}
    shown = [r for r in example
             if r != "legacy_null" or args.plot_legacy_null]
    for regime in shown:
        proc = example[regime]
        t = np.asarray(proc.dates)
        axes[0].plot(t, proc.log_e, lw=1.0, alpha=0.85, color=colors[regime])
        axes[1].plot(t, proc.log_m, lw=1.6, color=colors[regime], label=labels[regime])
    # zero line: above it a month is evidence AGAINST exchangeability, below it
    # the candidate has lost ground on that month
    axes[0].axhline(0.0, ls="--", c="k", lw=0.9)
    axes[0].annotate("no evidence", xy=(0.995, 0.0), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7, color="0.35")
    axes[0].axhline(np.log(args.n_perms + 1), ls=":", c="0.4", lw=0.8)
    axes[0].annotate(r"$\log(N{+}1)$", xy=(0.995, np.log(args.n_perms + 1)),
                     xycoords=("axes fraction", "data"), ha="right", va="top",
                     fontsize=7, color="0.35")
    axes[0].set_ylabel(r"$\log \hat E_t$")
    axes[0].set_title("Synthetic validation, corrected estimator "
                      f"($N={args.n_perms}$, identity adjoined)")
    axes[1].axhline(0.0, ls="-", c="0.75", lw=0.8)
    # the threshold is on the E-process scale, so it sits at height log(1/alpha)
    axes[1].axhline(np.log(1.0 / args.alpha), ls="--", c="k", lw=0.9)
    axes[1].annotate(rf"$1/\alpha={1/args.alpha:.0f}$",
                     xy=(0.995, np.log(1.0 / args.alpha)),
                     xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7)
    axes[1].set_ylabel(r"$\log M_k$")
    axes[1].set_xlabel("month")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):                    # pdf for \includegraphics
        fig.savefig(PLOTS / f"synthetic_validation.{ext}", dpi=200,
                    bbox_inches="tight")
    # keep the plotted traces so the figure can be redrawn without re-running
    with open(RESULTS / "synthetic_traces.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["regime", "month", "log_E", "log_M"])
        for regime, proc in example.items():
            lm = proc.log_m
            for i, m in enumerate(proc.dates):
                wr.writerow([regime, m, f"{proc.log_e[i]:.6f}", f"{lm[i]:.6f}"])
    print(f"\nWrote {RESULTS / 'synthetic_validation.csv'}, "
          f"{RESULTS / 'synthetic_traces.csv'} and "
          f"{PLOTS / 'synthetic_validation.pdf'}")


if __name__ == "__main__":
    main()
