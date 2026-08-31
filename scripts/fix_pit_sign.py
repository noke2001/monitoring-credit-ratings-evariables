"""Re-emit the saved TabPFN fits with the PIT sign corrected.

The drivers have been fixed at source (``rank(..., ascending=True)``), so any
run from now on is already correct.  Files produced before the fix carry the
inverted column; this script writes a corrected sidecar next to each one rather
than rewriting hundreds of megabytes in place.

Each sidecar holds ``isin, date, directional_deviation, pit, pit_stored`` — the
key, the score, the corrected PIT and the original, so the change is auditable.

    python scripts/fix_pit_sign.py                # write the sidecars
    python scripts/fix_pit_sign.py --check        # report, write nothing
    python scripts/fix_pit_sign.py --in-place     # rewrite the fits themselves
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tabpfn_panel import PROVENANCE, load_fit, scan_integrity  # noqa: E402

FITS = Path("/Users/philip/Library/CloudStorage/OneDrive-Personal/Desktop/"
            "ETH/Master_Thesis/R_code_agent/tabpfnfit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default=str(FITS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()
    fits = Path(args.fits)

    print(f"{'file':<46}{'rows':>9}{'corr(new,old)':>15}{'action':>22}")
    for fname in PROVENANCE:
        src = fits / fname
        if not src.exists():
            print(f"{fname:<46}{'MISSING':>9}")
            continue
        d = load_fit(src)
        if "pit" not in d.columns:
            print(f"{fname:<46}{d.height:>9,}{'-':>15}{'no PIT column':>22}")
            continue
        new = d["pit"].to_numpy().astype(float)
        if "pit_stored" in d.columns:
            old = d["pit_stored"].to_numpy().astype(float)
            m = np.isfinite(new) & np.isfinite(old)
            r = float(np.corrcoef(new[m], old[m])[0, 1])
        else:
            old, r = None, np.nan

        if args.check:
            action = "check only"
        elif args.in_place:
            keep = [c for c in d.columns if c != "pit_stored"]
            d.select(keep).write_csv(src)
            action = "rewritten in place"
        else:
            cols = ["isin", "date", "directional_deviation", "pit"]
            if old is not None:
                cols.append("pit_stored")
            out = src.with_name(src.stem + "_pit_corrected.csv")
            d.select(cols).write_csv(out)
            action = "sidecar written"
        rs = "-" if not np.isfinite(r) else f"{r:+.4f}"
        print(f"{fname:<46}{d.height:>9,}{rs:>15}{action:>22}")

    print("\ncorr(new, old) near -1 is the expected result: the correction is a")
    print("reversal of the rank, so the two columns are near-perfect opposites.")
    print("Values are not exactly -1 because the stored column used the default")
    print("'average' tie handling on a descending sort of a score with ties.")

    if not args.check:
        print("\nintegrity of the probability block (unchanged by this script):")
        for fname in PROVENANCE:
            if not (fits / fname).exists():
                continue
            r = scan_integrity(fits / fname)
            if not r["probs_trustworthy"]:
                print(f"  {fname}: argmax agreement "
                      f"{r['argmax_agreement']:.2%} -- probabilities NOT usable")


if __name__ == "__main__":
    main()
