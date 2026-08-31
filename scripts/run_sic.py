"""Two SIC-based designs (thesis Section 3.7).

(a) --design sector-blocks
    The null is unchanged --- the cross-section is exchangeable within the
    rating class --- and the SIC division supplies the candidate's partition.
    A rejection says sector membership predicts a bond's position, i.e. the
    rating label is not a sufficient summary of the class. Because SIC is
    static it is F_0-measurable, so no lagging is required.

(b) --design partial
    The null is weakened to Gamma-exchangeability: only permutations *within*
    a SIC division are considered, so the hypothesis reads "within a rating
    class and a sector, the identity of a bond is statistically irrelevant".
    The candidate must then be partitioned on a *different* covariate --- we
    use lagged duration --- since a candidate partitioned on sector would be
    invariant under the group and give E_t == 1 identically.

Both use the identity-adjoined estimator, so validity is unaffected: the proof
uses only that the permutations form a group and that q_t is measurable with
respect to the conditioning.

Usage:
    python scripts/run_sic.py --design sector-blocks --target yield
    python scripts/run_sic.py --design partial --target yield --partition-cov duration
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate import build_candidate, evaluate_date  # noqa: E402
from src.evalue import EProcess, mc_permutation_log_evalue  # noqa: E402
from src.panel import load_panel, iter_class_dates, RATING_CLASSES  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
CSV_DEFAULT = BASE.parent / "CorpBond_Reconciling" / "corp_jkp_mergedv2.csv"
NBER = [(200712, 200906), (202002, 202004)]


def in_recession(d):
    return any(a <= d <= b for a, b in NBER)


def run_class(df, rating, target, y_cov, n_perms, design, alpha=0.05):
    proc = EProcess(alpha=alpha)
    for item in iter_class_dates(df, rating, target, y_cov):
        if item is None:
            continue
        if item.get("degenerate"):
            proc.update(item["date"], 0.0, None, n=0, B=0, sectors=0)
            continue
        sector = item["sector"]
        if design == "sector-blocks":
            cand = build_candidate(
                item["x"], item["y_lag"], item["bond_ids"],
                item["window_x_by_bond"], item["pooled_window_x"],
                item["window_month_list"], block_labels=sector)
            groups = None
        else:                                   # partial exchangeability
            cand = build_candidate(
                item["x"], item["y_lag"], item["bond_ids"],
                item["window_x_by_bond"], item["pooled_window_x"],
                item["window_month_list"])
            # unlabelled bonds form their own group; they are never mixed with
            # a real sector, which keeps the group well defined
            groups = np.array([s if s is not None else "__unmapped__"
                               for s in sector], dtype=object)
        if cand is None:
            proc.update(item["date"], 0.0, None, n=item["x"].size, B=0, sectors=0)
            continue
        rng = np.random.default_rng(42 + item["window_month_list"][-1])
        ll_orig, ll_perms = evaluate_date(cand, item["x"], n_perms, rng, groups=groups)
        log_e, rank = mc_permutation_log_evalue(ll_orig, ll_perms)
        n_sec = len({s for s in sector if s is not None})
        proc.update(item["date"], log_e, rank, n=item["x"].size,
                    B=len(cand.block_positions), sectors=n_sec)
    return proc


def save_csv(proc, path):
    log_m = proc.log_m
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "n", "B", "sectors", "log_E", "rank", "log_M"])
        for i, d in enumerate(proc.dates):
            m = proc.meta[i]
            w.writerow([d, m.get("n"), m.get("B"), m.get("sectors"),
                        f"{proc.log_e[i]:.6f}", proc.ranks[i], f"{log_m[i]:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_DEFAULT))
    ap.add_argument("--design", choices=["sector-blocks", "partial"], required=True)
    ap.add_argument("--target", default="yield")
    ap.add_argument("--partition-cov", default="duration",
                    help="candidate partition; used by BOTH designs for the "
                         "lag machinery, and as the actual partition in 'partial'")
    ap.add_argument("--n-perms", type=int, default=1000)
    ap.add_argument("--classes", nargs="*", default=RATING_CLASSES)
    args = ap.parse_args()

    df = load_panel(args.csv, target=args.target,
                    partition_covs=(args.partition_cov,))
    (BASE / "results").mkdir(exist_ok=True)
    tag = f"sic_{args.design.replace('-', '')}_{args.target}_{args.partition_cov}"

    lines = []
    def w(s=""):
        lines.append(s)
        print(s, flush=True)

    w(f"design = {args.design} | X = {args.target} | "
      f"partition = {'SIC division' if args.design == 'sector-blocks' else args.partition_cov}"
      f" | group = {'full symmetric' if args.design == 'sector-blocks' else 'within SIC division'}"
      f" | N = {args.n_perms}")
    w(f"{'class':6s} {'dates':>6s} {'medn':>6s} {'medB':>5s} {'medSec':>7s} "
      f"{'terminal':>9s} {'peak':>9s} {'ceil%':>6s} {'NBERshare':>10s} {'first':>8s}")
    for cls in args.classes:
        t0 = time.time()
        proc = run_class(df, cls, args.target, args.partition_cov,
                         args.n_perms, args.design)
        save_csv(proc, BASE / "results" / f"{tag}_{cls}.csv")
        lm, le = proc.log_m, np.asarray(proc.log_e)
        d = np.asarray(proc.dates)
        m = np.array([in_recession(x) for x in d])
        ceil = np.mean([r == 1 for r in proc.ranks if r is not None]) if any(
            r is not None for r in proc.ranks) else float("nan")
        fa = proc.first_alarm()
        share = le[m].sum() / lm[-1] if lm[-1] != 0 else float("nan")
        w(f"{cls:6s} {len(d):6d} "
          f"{np.median([x.get('n', 0) for x in proc.meta]):6.0f} "
          f"{np.median([x.get('B', 0) for x in proc.meta]):5.0f} "
          f"{np.median([x.get('sectors', 0) for x in proc.meta]):7.0f} "
          f"{lm[-1]:9.1f} {lm.max():9.1f} {ceil:6.2f} {share:+10.2f} "
          f"{(str(d[fa]) if fa is not None else 'never'):>8s}   ({time.time()-t0:.0f}s)")
    (BASE / "results" / f"{tag}_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nWrote results/{tag}_*.csv and {tag}_summary.txt")


if __name__ == "__main__":
    main()
