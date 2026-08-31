"""Audit of the legacy notebooks' e-values under the exact identity fix.

Parses the per-date ``Log E`` values printed into a legacy notebook's stored
outputs and applies the closed-form correction of ``evalue.legacy_to_corrected``
(E_new = (M+1) E_old / (M + E_old)), which is exact because the corrected and
legacy estimators share the same sampled permutations.

NOTE — this repairs ONLY the missing-identity defect.  The legacy candidate
also used origin-PIT, contemporaneous-Y bucketing, and dropped the marginal
factors (thesis Sections 3.4.3-3.4.4), none of which a post-hoc transform can
undo.  The corrected empirical results come from ``run_empirical.py``; this
script exists to document how much of the legacy picture the identity bug alone
accounts for.

Usage:  python scripts/correct_legacy_outputs.py ../approach3_R_avg_attempt1.ipynb --M 500
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evalue import legacy_to_corrected  # noqa: E402

PLOTS = Path(__file__).resolve().parents[1] / "plots"


def parse_notebook_outputs(path: str):
    """Extract (section headers, per-date log E) series from printed outputs."""
    nb = json.load(open(path))
    text = ""
    for c in nb.get("cells", []):
        for o in c.get("outputs", []):
            if "text" in o:
                text += "".join(o["text"])
    runs, current_label, cur = [], "run", []
    for line in text.splitlines():
        m = re.search(r"Processing Rating Class: (\S+)", line)
        if m:
            if cur:
                runs.append((current_label, cur))
            current_label, cur = m.group(1), []
        m = re.search(r"Processing Date: (\d{4}-\d{2})-\d{2}.*?Log E: (-?\d+\.\d+)", line)
        if m:
            cur.append((m.group(1), float(m.group(2))))
    if cur:
        runs.append((current_label, cur))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--M", type=int, default=500,
                    help="number of Monte Carlo permutations used by the notebook")
    args = ap.parse_args()

    runs = parse_notebook_outputs(args.notebook)
    if not runs:
        raise SystemExit("no printed 'Log E' values found in notebook outputs")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(runs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows),
                             squeeze=False)
    for j, (label, series) in enumerate(runs):
        dates = [d for d, _ in series]
        log_e_old = np.array([v for _, v in series])
        log_e_new = legacy_to_corrected(log_e_old, args.M)
        ax = axes[j // ncols][j % ncols]
        x = np.arange(len(dates))
        ax.plot(x, np.cumsum(log_e_old), lw=1.2, color="gray",
                label="legacy (invalid)")
        ax.plot(x, np.cumsum(log_e_new), lw=1.4, color="firebrick",
                label="identity fix applied")
        ax.axhline(np.log(20), ls="--", c="k", lw=0.7)
        ax.set_title(label, fontsize=10)
        step = max(1, len(dates) // 6)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(dates[::step], fontsize=7, rotation=45)
        if j == 0:
            ax.legend(fontsize=8)
        tot_old, tot_new = np.sum(log_e_old), float(np.sum(log_e_new))
        print(f"{label:8s} terminal logM: legacy {tot_old:10.1f} -> corrected "
              f"{tot_new:10.1f} (delta {tot_new - tot_old:+.1f}); "
              f"per-step cap log(M+1) = {np.log(args.M + 1):.2f}, "
              f"legacy max step = {log_e_old.max():.2f}")
    fig.suptitle(f"Exact post-hoc identity fix, M = {args.M} "
                 f"(legacy candidate unchanged)", fontsize=11)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    out = PLOTS / f"legacy_identity_fix_{Path(args.notebook).stem}.png"
    fig.savefig(out, dpi=200)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
