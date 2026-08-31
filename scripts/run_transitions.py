"""Rating-migration statistics for thesis Chapter 2.

Section 2.3.5 already states how *rarely* ratings move.  This adds the shape of
the movement when it happens: the direction, the size of the jump, and the
twelve-month migration matrix that is the standard summary object in the credit
literature.

Everything is computed on contiguous month pairs only -- the panel is
unbalanced, and a naive shift would pair observations across gaps in a bond's
history and count a gap as a migration.

    conda activate bond          # or copula
    python scripts/run_transitions.py
    python scripts/run_transitions.py --horizon 24
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE.parent / "CorpBond_Reconciling" / "corp_jkp_mergedv2.csv"
LETTERS = ["AAA", "AA", "A", "BBB", "BB", "B"]
RANK = {c: i for i, c in enumerate(LETTERS)}


def emit(fh, s=""):
    print(s)
    fh.write(s + "\n")


def load(csv_path):
    d = (pl.scan_csv(csv_path, infer_schema_length=20000, ignore_errors=True)
         .select(["dates", "isin", "rtg", "nrtg"])
         .filter(pl.col("isin").is_not_null() & pl.col("rtg").is_in(LETTERS)
                 & pl.col("nrtg").is_not_null() & pl.col("nrtg").is_not_nan())
         .collect()
         .with_columns(pl.col("nrtg").cast(pl.Int64),
                       (((pl.col("dates") // 100) * 12)
                        + (pl.col("dates") % 100)).alias("midx"))
         .sort(["isin", "midx"]))
    # one row per (bond, month)
    return d.unique(subset=["isin", "midx"], keep="last").sort(["isin", "midx"])


def pairs(d, lag):
    """Indices (i, j) of rows exactly `lag` contiguous months apart, same bond."""
    isin = d["isin"].to_numpy()
    m = d["midx"].to_numpy()
    ok = (isin[lag:] == isin[:-lag]) & (m[lag:] - m[:-lag] == lag)
    idx = np.flatnonzero(ok)
    return idx, idx + lag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--horizon", type=int, default=12)
    args = ap.parse_args()

    d = load(args.csv)
    (BASE / "results").mkdir(exist_ok=True)
    fh = open(BASE / "results" / "transition_summary.txt", "w")

    n_bonds = d["isin"].n_unique()
    emit(fh, f"panel: {d.height:,} bond-months | {n_bonds:,} bonds | "
             f"{d['dates'].min()}--{d['dates'].max()}")

    letter = np.array([RANK[c] for c in d["rtg"].to_numpy()])
    notch = d["nrtg"].to_numpy()

    # ---------------------------------------------------------- T1 how often
    i, j = pairs(d, 1)
    emit(fh, "\nT1  month-to-month migration")
    emit(fh, "-" * 60)
    emit(fh, f"{'':<34}{'letter':>12}{'notch':>12}")
    for name, v in (("contiguous month pairs", None),
                    ("migrations", None), ("  downgrades", None),
                    ("  upgrades", None), ("monthly migration rate", None)):
        pass
    rows = []
    for label, v in (("letter", letter), ("notch", notch)):
        moved = v[j] != v[i]
        rows.append({
            "pairs": len(i), "moved": int(moved.sum()),
            "down": int((v[j] > v[i]).sum()), "up": int((v[j] < v[i]).sum()),
            "rate": moved.mean(),
        })
    L, N = rows
    emit(fh, f"{'contiguous month pairs':<34}{L['pairs']:>12,}{N['pairs']:>12,}")
    emit(fh, f"{'migrations':<34}{L['moved']:>12,}{N['moved']:>12,}")
    emit(fh, f"{'  of which downgrades':<34}{L['down']:>12,}{N['down']:>12,}")
    emit(fh, f"{'  of which upgrades':<34}{L['up']:>12,}{N['up']:>12,}")
    emit(fh, f"{'monthly migration rate':<34}{L['rate']:>11.2%}{N['rate']:>12.2%}")

    for label, v in (("letter", letter), ("notch", notch)):
        s = pl.DataFrame({"isin": d["isin"], "v": v}).group_by("isin").agg(
            pl.col("v").n_unique().alias("k"))
        ever = (s["k"] > 1).mean()
        emit(fh, f"{'bonds that ever migrate (' + label + ')':<34}{ever:>11.1%}")

    # ------------------------------------------------------- T2 size of jump
    dv = (notch[j] - notch[i])
    moves = dv[dv != 0]
    emit(fh, "\nT2  size of a notch migration")
    emit(fh, "-" * 60)
    for k in sorted(set(np.abs(moves))):
        share = (np.abs(moves) == k).mean()
        if share >= 0.001:
            emit(fh, f"  |delta nrtg| = {k}:  {int((np.abs(moves)==k).sum()):>5,}"
                     f"  ({share:>6.2%})")
    emit(fh, f"  |delta nrtg| <= 2 covers {np.mean(np.abs(moves) <= 2):.2%}")
    li, lj = letter[i], letter[j]
    sel = notch[i] != notch[j]
    emit(fh, f"  notch moves that also change the letter class: "
             f"{np.mean(li[sel] != lj[sel]):.2%}")

    # ------------------------------------------------ T3 migration matrix
    H = args.horizon
    i2, j2 = pairs(d, H)
    emit(fh, f"\nT3  {H}-month letter migration matrix (row %, "
             f"{len(i2):,} bond pairs)")
    emit(fh, "-" * 72)
    emit(fh, f"{'from \\ to':<10}" + "".join(f"{c:>9}" for c in LETTERS)
             + f"{'n':>10}{'unchanged':>11}")
    M = np.zeros((6, 6))
    for a in range(6):
        m = letter[i2] == a
        if m.sum() == 0:
            continue
        for b in range(6):
            M[a, b] = (letter[j2][m] == b).mean() * 100
        emit(fh, f"{LETTERS[a]:<10}" + "".join(
            f"{M[a, b]:>9.2f}" if M[a, b] >= 0.005 else f"{'--':>9}"
            for b in range(6)) + f"{int(m.sum()):>10,}{M[a, a]:>10.1f}%")

    # censoring: how many bonds simply stop being observed
    isin = d["isin"].to_numpy(); mm = d["midx"].to_numpy()
    have_future = np.zeros(d.height, bool)
    have_future[i2] = True
    last_month = pl.DataFrame({"isin": d["isin"], "m": mm}).group_by("isin").agg(
        pl.col("m").max().alias("last"))
    lm = dict(zip(last_month["isin"].to_list(), last_month["last"].to_list()))
    still_alive = np.array([lm[s] >= t + H for s, t in zip(isin, mm)])
    emit(fh, f"\n  rows with no observation {H} months later: "
             f"{(~have_future).mean():.1%}")
    emit(fh, f"  of which the bond has left the panel entirely: "
             f"{np.mean(~still_alive[~have_future]):.1%}")
    emit(fh, "  (maturity, call, or loss of coverage -- these are censored, not"
             "\n   'no migration', and Chapter 4 excludes them rather than"
             " scoring them)")

    fh.close()
    print("\nwrote results/transition_summary.txt")


if __name__ == "__main__":
    main()
