"""Calibration audit: what omitting the identity permutation actually costs.

Simulates a genuine null — the current cross-section is exchangeable — while
the candidate is *armed*: its blocks are fitted on a history whose blocks are
separated, exactly the configuration of the bond panel, where the permutation
distribution of q_t is heavy-tailed and the Jensen gap is largest.

Reports, for the corrected (identity adjoined) and naive (identity omitted)
estimators computed from the SAME permutation draws:

  * E[E_t] under the null              (must be <= 1 for a valid e-variable)
  * P(E_t >= 1/alpha) at a single date (must be <= alpha)
  * Ville false-alarm rate over T dates (must be <= alpha)
  * the largest single-date log E_t     (corrected: capped at log(N+1))

Usage:  python scripts/audit_calibration.py [--reps 400] [--n-perms 199]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate import build_candidate, evaluate_date  # noqa: E402
from src.evalue import mc_permutation_log_evalue, legacy_log_evalue  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def null_panel(rng, n=24, months=12, block_sep=1.5):
    """Exchangeable current cross-section, block-separated history."""
    ids = np.array([f"b{i}" for i in range(n)])
    y_lag = np.linspace(0, 1, n) + rng.normal(scale=1e-3, size=n)
    month_list = list(range(100, 100 + months))
    common = {m: rng.normal() for m in month_list}
    wxb = {}
    for i, b in enumerate(ids):
        shift = block_sep if i >= n // 2 else 0.0
        wxb[b] = {m: float(6.0 + shift + common[m] + 0.6 * rng.normal())
                  for m in month_list}
    pooled = np.array([v for w in wxb.values() for v in w.values()])
    # identical law at every position => exchangeable given the past
    x = 6.0 + 0.5 * block_sep + rng.normal() + 0.6 * rng.normal(size=n)
    return x, y_lag, ids, wxb, pooled, month_list


def one_date(rng, n_perms, block_sep):
    x, y, ids, wxb, pooled, months = null_panel(rng, block_sep=block_sep)
    cand = build_candidate(x, y, ids, wxb, pooled, months, max_block_size=12)
    if cand is None:
        return 0.0, 0.0
    ll_orig, ll_perms = evaluate_date(cand, x, n_perms, rng)
    return (mc_permutation_log_evalue(ll_orig, ll_perms)[0],
            legacy_log_evalue(ll_orig, ll_perms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--n-perms", type=int, default=199)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--dates", type=int, default=24)
    args = ap.parse_args()

    lines = []

    def w(s=""):
        lines.append(s)
        print(s, flush=True)

    w(f"Calibration audit — genuine null, armed candidate")
    w(f"N = {args.n_perms} permutations, {args.reps} replications, "
      f"alpha = {args.alpha}, cap log(N+1) = {np.log(args.n_perms + 1):.3f}")
    w()

    for block_sep in (0.5, 1.5, 3.0):
        rng = np.random.default_rng(20260822)
        log_c, log_n = [], []
        for _ in range(args.reps):
            a, b = one_date(rng, args.n_perms, block_sep)
            log_c.append(a)
            log_n.append(b)
        e_c = np.exp(np.clip(log_c, -700, 700))
        e_n = np.exp(np.clip(log_n, -700, 700))
        se_c = e_c.std(ddof=1) / np.sqrt(e_c.size)
        se_n = e_n.std(ddof=1) / np.sqrt(e_n.size)
        w(f"--- block separation {block_sep:.1f} sd ---")
        w(f"  corrected  E[E_t] = {e_c.mean():8.3f} +/- {se_c:.3f} | "
          f"P(E_t >= 1/alpha) = {np.mean(e_c >= 1 / args.alpha):.3f} | "
          f"max log E_t = {max(log_c):7.3f}")
        w(f"  naive      E[E_t] = {e_n.mean():8.3f} +/- {se_n:.3f} | "
          f"P(E_t >= 1/alpha) = {np.mean(e_n >= 1 / args.alpha):.3f} | "
          f"max log E_t = {max(log_n):7.3f}")
        w()

    # Ville false-alarm rate over a monitoring run of `dates` dates
    w(f"--- Ville false-alarm rate over {args.dates} dates, block separation 1.5 sd ---")
    rng = np.random.default_rng(777)
    thr = np.log(1.0 / args.alpha)
    alarms_c = alarms_n = 0
    runs = max(50, args.reps // 4)
    for _ in range(runs):
        m_c = m_n = 0.0
        hit_c = hit_n = False
        for _t in range(args.dates):
            a, b = one_date(rng, args.n_perms, 1.5)
            m_c += a
            m_n += b
            hit_c |= m_c >= thr
            hit_n |= m_n >= thr
        alarms_c += hit_c
        alarms_n += hit_n
    w(f"  corrected: {alarms_c / runs:.3f}   (nominal {args.alpha})")
    w(f"  naive:     {alarms_n / runs:.3f}   (nominal {args.alpha})")
    w(f"  runs = {runs}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "calibration_audit.txt").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS / 'calibration_audit.txt'}")


if __name__ == "__main__":
    main()
