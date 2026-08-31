"""
Final PIT selection: validity (lag-1 ACF) against power (lift at matched budget).

    python tests_and_plots/select_final_pit.py

Two scores can feed the e-process on the VAE panel:
  det_score        1 - P(monitored rating)   -- a cross-sectional disagreement score
  randomized_pit   PIT of R_{t-1} under the model's predictive distribution

and each can be used as a LEVEL or as an AR(1) INNOVATION. The innovation
treatment is what makes the PIT uniform CONDITIONAL on the bond's own past,
which is the requirement E[e_t|F_{t-1}] = 1 imposes. This table decides which
combination to ship by reading both columns together: a high lift on a series
with lag-1 ACF near 0.85 is not a real 1.5x, because the process is compounding
correlated observations as if they were independent.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eprocess"))

from compare_pit_constructions import load
from test_conditional_pit import acf1, sweep_lift
from src.betting import ar1_innovation, randomized_rank_pit

RULE = "=" * 108


def main():
    t0 = time.time()
    print(RULE)
    print("  FINAL PIT SELECTION -- validity (ACF) vs power (lift @ 4,000 alarms)")
    print(RULE)
    df = load()
    rng = np.random.default_rng(42)
    cohort = pd.MultiIndex.from_frame(df[["dates", "prev_enc_y"]]).to_numpy()
    strata = df["prev_enc_y"].fillna(3).astype(int).to_numpy()
    ids, order = df["isin"].to_numpy(), df["dates"].to_numpy()

    def rank_of(v):
        return randomized_rank_pit(np.asarray(v, dtype=float), cohort, rng)

    def innov(col):
        return ar1_innovation(df[col].to_numpy(), ids, order=order, strata=strata)

    df["Z1"] = pd.Series(rank_of(df["det_score"])).fillna(0.5).to_numpy()
    df["Z2"] = rank_of(innov("det_score"))
    df["Z3"] = df["randomized_pit"].fillna(0.5).to_numpy()
    df["Z4"] = rank_of(innov("randomized_pit"))

    variants = [
        ("1. det_score LEVEL rank        (old Z_det)", "Z1", "upper"),
        ("2. det_score AR(1) innovation rank",         "Z2", "upper"),
        ("3. model PIT level             (two-sided)", "Z3", "two-sided"),
        ("4. model PIT AR(1) innovation  (two-sided)", "Z4", "two-sided"),
        ("5. model PIT AR(1) innovation  (upper)",     "Z4", "upper"),
    ]

    print(f"\n{'variant':<46}{'lag-1 ACF':>11}{'lag-6 ACF':>11}"
          f"{'alarm span':>16}{'P24':>8}{'R24':>8}{'F1':>8}{'lift':>7}")
    print("-" * len(RULE))
    rows = []
    for name, zc, tail in variants:
        a1, a6 = acf1(df, zc, 1), acf1(df, zc, 6)
        r = sweep_lift(df, df[zc].to_numpy(), tail)
        span = f"{r['span'][0]:,}-{r['span'][1]:,}"
        vals = ("—", "—", "—", "—") if np.isnan(r["P"]) else (
            f"{r['P']:.2f}", f"{r['R']:.2f}", f"{r['F1']:.4f}", f"{r['lift']:.2f}")
        print(f"{name:<46}{a1:>11.3f}{a6:>11.3f}{span:>16}"
              f"{vals[0]:>8}{vals[1]:>8}{vals[2]:>8}{vals[3]:>7}")
        rows.append(dict(Variant=name, Tail=tail, ACF1=a1, ACF6=a6,
                         P24=r["P"], R24=r["R"], F1=r["F1"], Lift=r["lift"]))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "final_pit_selection.csv", index=False)
    ok = out[out["ACF1"].abs() < 0.10].dropna(subset=["Lift"])
    if len(ok):
        b = ok.loc[ok["Lift"].idxmax()]
        print(f"\n  Best among conditionally-uniform variants (|ACF1| < 0.10):")
        print(f"    {b['Variant'].strip()}  ->  lift {b['Lift']:.2f}x, "
              f"F1 {b['F1']:.4f}, ACF1 {b['ACF1']:+.3f}")
    print(f"\n  total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
