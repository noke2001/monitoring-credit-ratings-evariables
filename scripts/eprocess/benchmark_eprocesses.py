"""
E-Process benchmark suite for Approach 1.

    python tests_and_plots/benchmark_eprocesses.py [--panel vae|gbdt|both]

WHAT IS BEING DETECTED
----------------------
The monitored event is ANY rating transition, in either direction. The thesis
calls this a "downgrade", but the objective is to flag a bond before the agency
moves its rating at all. Consequences carried through this whole suite:

  * Evaluation targets `is_rating_change`, not `is_downgrade`. Scoring against
    downgrades alone would count every correctly anticipated upgrade as a false
    alarm.
  * The alternative is TWO-SIDED. The model may imply a better or a worse rating
    than the agency's standing one, and either is evidence the standing rating is
    stale. Engines therefore fold the PIT through u = |2Z - 1|, which is exactly
    Uniform(0,1) under H_0, and bet on large u. A one-sided orientation forfeits
    all power against transitions in the discarded direction.

SCORING
-------
For a horizon H (months), precision and recall share one event definition:

    P_H  = #{alarms a : a transition falls in (a, a+H]} / #alarms
    R_H  = #{transitions c : an alarm falls in [c-H, c)} / #transitions
    F1_H = harmonic mean

Both strict: an alarm in the same month as the transition is not an early
warning and counts for neither.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import pathlib

# --- panel location -----------------------------------------------------------
# The inference panels are data, not code, and are not part of this repository.
# Look in results/ first (where scripts/vae/train_semisupervised.py writes), then
# fall back to EPROCESS_PANELS if it is set.
def _panel(name):
    import os
    for cand in (ROOT / "results" / name,
                 pathlib.Path(os.environ.get("EPROCESS_PANELS", "")) / name):
        if str(cand) != name and cand.exists():
            return cand
    return ROOT / "results" / name

sys.path.insert(0, str(ROOT))

from src.sequential import (
    GrenanderEProcess,
    AdaptiveKellyEProcess,
    KellyMixtureEProcess,
    AsymmetricLeakyEProcess,
    InnovationGatedEProcess,
    MixtureRestartEProcess,
    OptimalHybridEProcess,
    RollingWindowEProcess,
    TierVelocityEProcess,
    DeadzoneEProcess,
)
from src.betting import ar1_innovation, randomized_rank_pit
from src.metrics import evaluate_lookforward_alarms

RULE = "=" * 118
ALPHAS = (0.20, 0.10, 0.05, 0.01)


# ---------------------------------------------------------------- panels
def build_pit(df: pd.DataFrame, score_col: str, kind: str, seed: int = 42):
    """
    Construct the PIT the e-process runs on, and say which tail contradicts H_0.

    kind='ar1'    DEFAULT. Rank of the AR(1) residual s_t - beta*s_{t-1} within
                  a (date, monitored-rating) cohort. This is the only variant
                  that is uniform CONDITIONAL on the bond's own past, which is
                  what E[e_t|F_{t-1}] = 1 requires. Level ranks carry lag-1
                  autocorrelation 0.854 on this panel; the AR(1) residual carries
                  0.026. A large innovation contradicts H_0, so tail='upper'.
    kind='level'  Rank of the score level -- the original Z_det. Retained for
                  the comparison table only; not conditionally uniform.
    kind='model'  Randomized PIT of R_{t-1} under the model's own predictive
                  distribution. A different null (calibration, not
                  exchangeability) and two-sided.
    """
    rng = np.random.default_rng(seed)
    if kind == "model":
        return df["randomized_pit"].fillna(0.5).to_numpy(), "two-sided"

    cohort = pd.MultiIndex.from_frame(df[["dates", "prev_enc_y"]]).to_numpy()
    if kind == "level":
        z = randomized_rank_pit(df[score_col].to_numpy(), cohort, rng)
        return pd.Series(z).fillna(0.5).to_numpy(), "upper"
    if kind == "ar1":
        resid = ar1_innovation(
            df[score_col].to_numpy(),
            df["isin"].to_numpy(),
            order=df["dates"].to_numpy(),
            strata=df["prev_enc_y"].fillna(3).astype(int).to_numpy(),
        )
        # NaN residuals (a bond's first month) stay NaN: the engines abstain,
        # emitting e = 1 rather than scoring a spurious loss.
        return randomized_rank_pit(resid, cohort, rng), "upper"
    raise ValueError(f"unknown pit kind {kind!r}")


def load_vae() -> pd.DataFrame:
    """VAE panel: uses the exact randomized PIT of the monitored rating."""
    f = _panel("semi_supervised_fresh_inference_results.csv")
    cols = (["isin", "dates", "enc_y", "prev_enc_y", "randomized_pit",
             "downgrade_tail_prob", "directional_deviation"] +
            [f"prob_class_{c}" for c in
             ("AAA", "AA", "A", "BBB", "BB", "B")])
    df = pd.read_csv(f, usecols=cols, low_memory=False)
    df["dates"] = pd.to_datetime(df["dates"])
    df = df.sort_values(["isin", "dates"]).reset_index(drop=True)
    prev = df.groupby("isin")["enc_y"].shift(1)
    df["is_rating_change"] = ((df["enc_y"] != prev) & prev.notna()).astype(int)
    df["hazard"] = df["downgrade_tail_prob"]
    # deterioration score: predictive mass NOT on the monitored rating
    P = df[[c for c in df.columns if c.startswith("prob_class_")]].to_numpy()
    k = np.clip(df["prev_enc_y"].fillna(0).astype(int).to_numpy(), 0, P.shape[1] - 1)
    df["det_score"] = 1.0 - P[np.arange(len(df)), k]
    return df.reset_index(drop=True)


def load_gbdt() -> pd.DataFrame:
    """GBDT hazard panel: exactly-uniform randomized rank PIT of the score."""
    f = _panel("multi_horizon_survival_inference_results.csv")
    cols = ["isin", "dates", "enc_y", "prev_enc_y", "is_rating_change",
            "det_score", "hazard_score_24m", "directional_deviation"]
    df = pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
    df["dates"] = pd.to_datetime(df["dates"])
    df = df.sort_values(["isin", "dates"]).reset_index(drop=True)
    df["is_rating_change"] = df["is_rating_change"].astype(float).fillna(0).astype(int)
    df["hazard"] = df["hazard_score_24m"]

    # Prefer the redesigned hazard when it exists: a 1-month discrete-time
    # hazard on rating-anchored features (next-month AUC 0.5715 vs 0.5134 for
    # the 24-month-indicator design). See train_hazard_redesign.py.
    redesign = _panel("hazard_redesign_scores.csv")
    if redesign.exists():
        h = pd.read_csv(redesign, usecols=["isin", "dates", "h_new"],
                        low_memory=False)
        h["dates"] = pd.to_datetime(h["dates"])
        df = df.merge(h, on=["isin", "dates"], how="left")
        if df["h_new"].notna().mean() > 0.9:
            df["det_score"] = df["h_new"].fillna(df["det_score"])
            df["hazard"] = df["h_new"].fillna(df["hazard"])
            print("  [gbdt] using redesigned 1-month hazard as the score")

    # OptimalHybrid gates on (DD, p_down); on this panel the hazard score plays
    # the role of the tail probability.
    df["downgrade_tail_prob"] = df["hazard_score_24m"]
    if "directional_deviation" not in df.columns:
        df["directional_deviation"] = 0.0
    df["directional_deviation"] = df["directional_deviation"].fillna(0.0)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- suite
def engine_grid(alpha: float, tail: str = "upper"):
    """(label, engine, extra run kwargs). `tail` comes from the PIT construction."""
    g = [
        (f"Deadzone d=.75 lam=.95",  DeadzoneEProcess(alpha, delta=.75, lam=.95, tail=tail), {}),
        (f"Deadzone d=.75 lam=.50",  DeadzoneEProcess(alpha, delta=.75, lam=.50, tail=tail), {}),
        (f"Deadzone d=.90 lam=.95",  DeadzoneEProcess(alpha, delta=.90, lam=.95, tail=tail), {}),
        # Arnold-Henzi-Ziegel out of the box: no fixed bet, the monotone
        # density is estimated from the bond's own past at every step.
        (f"AHZ Grenander burn=6",       GrenanderEProcess(alpha, tail=tail, burn_in=6), {}),
        (f"AHZ Grenander burn=12",      GrenanderEProcess(alpha, tail=tail, burn_in=12), {}),
        (f"AHZ Grenander burn=24",      GrenanderEProcess(alpha, tail=tail, burn_in=24), {}),
        # Dead-zone shape, growth-optimal stake plugged in from the past.
        (f"Plug-in Kelly d=.75",        AdaptiveKellyEProcess(alpha, delta=.75, tail=tail), {}),
        (f"Plug-in Kelly d=.90",        AdaptiveKellyEProcess(alpha, delta=.90, tail=tail), {}),
        (f"Kelly x mixture h=24",       KellyMixtureEProcess(alpha, horizon=24, delta=.75, tail=tail), {}),
        (f"Kelly x mixture h=12",       KellyMixtureEProcess(alpha, horizon=12, delta=.75, tail=tail), {}),
        (f"Mixture-restart h=12",       MixtureRestartEProcess(alpha, horizon=12, tail=tail), {}),
        (f"Mixture-restart h=24",       MixtureRestartEProcess(alpha, horizon=24, tail=tail), {}),
        (f"Mixture-restart h=60",       MixtureRestartEProcess(alpha, horizon=60, tail=tail), {}),
        (f"Asym. leaky rho=.80",        AsymmetricLeakyEProcess(alpha, rho=.80, tail=tail), {}),
        (f"Optimal hybrid gated K=2",   OptimalHybridEProcess(alpha, persistence_k=2, tail=tail), {}),
        (f"Tier+velocity gated",        TierVelocityEProcess(alpha, tail=tail), {"hazard_col": "hazard"}),
        (f"Innovation bet + level gate .70", InnovationGatedEProcess(alpha, level_gate=.70, tail=tail), {}),
        (f"Innovation bet + level gate .85", InnovationGatedEProcess(alpha, level_gate=.85, tail=tail), {}),
        (f"Innovation bet + level gate .50", InnovationGatedEProcess(alpha, level_gate=.50, tail=tail), {}),
    ]
    rw = RollingWindowEProcess(alpha, window_size=24, tail=tail,
                               acknowledge_invalid=True)
    g.append((f"Rolling window W=24 [INVALID]", rw, {}))
    return g


def run_panel(df: pd.DataFrame, panel: str, cooldown: int = 12,
              pit_kind: str = "ar1") -> pd.DataFrame:
    base = {}
    for H in (12, 24):
        fwd = df.groupby("isin")["is_rating_change"].transform(
            lambda s_: s_.iloc[::-1].rolling(H, min_periods=1).max()
                          .iloc[::-1].shift(-1)
        ).fillna(0)
        base[H] = float(fwd.mean())
    df = df.copy()
    df["pit"], tail = build_pit(df, "det_score", pit_kind)
    # Level rank: the PREDICTABLE gate input for InnovationGatedEProcess.
    # Uniform on [0,1], so `level_gate` reads directly as a quantile.
    df["level_rank"] = pd.Series(randomized_rank_pit(
        df["det_score"].to_numpy(),
        pd.MultiIndex.from_frame(df[["dates", "prev_enc_y"]]).to_numpy(),
        np.random.default_rng(7))).fillna(0.5).to_numpy()
    print(f"\n{RULE}\n  PANEL: {panel.upper()}   PIT: {pit_kind} (tail={tail})   "
          f"{len(df):,} bond-months | {df['isin'].nunique():,} bonds | "
          f"{df['dates'].min():%Y-%m}..{df['dates'].max():%Y-%m} | "
          f"{int(df['is_rating_change'].sum()):,} transitions")
    print(f"  Base rate of a transition within 12m: {100*base[12]:.2f}%   "
          f"within 24m: {100*base[24]:.2f}%   "
          f"(a random alarm attains exactly this precision)\n{RULE}")
    hdr = (f"{'alpha':>6}  {'engine':<32}{'alarms':>8}{'runs':>8}"
           f"{'P12':>7}{'R12':>7}{'F1_12':>8}"
           f"{'P24':>7}{'R24':>7}{'F1_24':>8}{'lift24':>7}{'lead24':>8}  valid")
    print(hdr); print("-" * len(hdr))
    rows = []
    for alpha in ALPHAS:
        for label, eng, kw in engine_grid(alpha, tail):
            res = eng.run_sequential_test(df, z_col="pit",
                                          cooldown_months=cooldown, **kw)
            m = evaluate_lookforward_alarms(res, horizons=(12, 24),
                                            event_col="is_rating_change",
                                            censoring="exclude")
            m.update({"Panel": panel, "PIT": pit_kind, "Alpha": alpha,
                      "Engine": label,
                      "Anytime_Valid": eng.anytime_valid,
                      "Base_12m (%)": round(100 * base[12], 2),
                      "Base_24m (%)": round(100 * base[24], 2),
                      "Lift_12m": round(m.get("P_12m (%)", 0) /
                                        max(1e-9, 100 * base[12]), 2),
                      "Lift_24m": round(m.get("P_24m (%)", 0) /
                                        max(1e-9, 100 * base[24]), 2)})
            rows.append(m)
            print(f"{alpha:>6.2f}  {label:<32}{m['Total_Alarms']:>8,}"
                  f"{m.get('n_runs',0):>8,}"
                  f"{m.get('P_12m (%)',0):>7.2f}{m.get('R_12m (%)',0):>7.2f}"
                  f"{m.get('F1_12m',0):>8.4f}"
                  f"{m.get('P_24m (%)',0):>7.2f}{m.get('R_24m (%)',0):>7.2f}"
                  f"{m.get('F1_24m',0):>8.4f}"
                  f"{m.get('P_24m (%)',0)/max(1e-9,100*base[24]):>7.2f}"
                  f"{m.get('Lead_24m (m)',0):>8.1f}"
                  f"  {'yes' if eng.anytime_valid else 'NO'}")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["vae", "gbdt", "both"], default="both")
    ap.add_argument("--cooldown", type=int, default=12)
    ap.add_argument("--pit", choices=["ar1", "level", "model"], default="ar1")
    args = ap.parse_args()

    t0 = time.time()
    print(RULE)
    print("  APPROACH 1 -- E-PROCESS BENCHMARK SUITE")
    print("  Event = ANY rating transition | two-sided PIT | F1 over matched "
          "precision/recall")
    print(RULE)

    out = []
    if args.panel in ("vae", "both"):
        out.append(run_panel(load_vae(), "vae", args.cooldown, args.pit))
    if args.panel in ("gbdt", "both"):
        f = _panel("multi_horizon_survival_inference_results.csv")
        if f.exists():
            out.append(run_panel(load_gbdt(), "gbdt", args.cooldown, args.pit))
        else:
            print(f"\n  [skip] {f.name} not found -- run "
                  f"train_multi_horizon_pipeline.py first.")

    res = pd.concat(out, ignore_index=True)
    front = ["Panel", "PIT", "Engine", "Alpha", "Anytime_Valid", "Total_Alarms",
             "Total_Events", "n_runs"]
    res = res[[c for c in front if c in res] +
              [c for c in res.columns if c not in front]]
    dest = ROOT / "results" / "eprocess_benchmark_results.csv"
    res.to_csv(dest, index=False)

    # ---- headline: best valid configuration per panel and horizon ----
    print(f"\n{RULE}\n  BEST ANYTIME-VALID CONFIGURATION BY F1\n{RULE}")
    valid = res[res["Anytime_Valid"]]
    for panel in valid["Panel"].unique():
        sub = valid[valid["Panel"] == panel]
        for H in (12, 24):
            b = sub.loc[sub[f"F1_{H}m"].idxmax()]
            print(f"  {panel:<5} H={H:>2}m  {b['Engine']:<32} alpha={b['Alpha']:.2f}  "
                  f"F1={b[f'F1_{H}m']:.4f}  P={b[f'P_{H}m (%)']:.2f}%  "
                  f"R={b[f'R_{H}m (%)']:.2f}%  alarms={int(b['Total_Alarms']):,}")
    inv = res[~res["Anytime_Valid"]]
    if len(inv):
        bi = inv.loc[inv["F1_24m"].idxmax()]
        print(f"\n  For contrast, the best NON-valid entry "
              f"({bi['Engine']}, alpha={bi['Alpha']:.2f}) reaches F1_24m="
              f"{bi['F1_24m']:.4f}. It carries no type-I error guarantee and "
              f"must not be reported as a level-alpha procedure.")

    print(f"\nSaved -> {dest.relative_to(ROOT)}   ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
