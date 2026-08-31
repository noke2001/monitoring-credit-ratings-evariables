"""
Fair power comparison of PIT constructions, plus uniformity diagnostics that
actually show the violation.

    python tests_and_plots/pit_power_and_diagnostics.py

WHY A MATCHED ALARM BUDGET. Comparing variants at a common alpha is not a fair
test: alpha fixes the false-alarm rate PER RUN under H_0, but the variants have
very different power, so they emit wildly different alarm counts (2,036 vs 9,831
at alpha=0.20). F1 rewards recall, and recall is cheap when you alarm on five
times as many months. The honest comparison sweeps alpha per variant and reads
precision off at the SAME alarm budget.

WHY THE MARGINAL HISTOGRAM SHOWS NOTHING. For a rank PIT, uniformity within each
cross-section is an identity, not a hypothesis: ranks of n items are always a
permutation of 1..n. The marginal histogram is therefore flat by construction and
cannot reveal anything. The exchangeability null is a statement about the TIME
SERIES of one bond's ranks, so the diagnostics below condition on time --
months-to-next-transition, and rank autocorrelation -- which is where the
violation actually lives.
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eprocess"))

from compare_pit_constructions import load, semi_supervised_score
from diagnose_f1_drop import random_alarms, windowed_metric
from src.sequential import DeadzoneEProcess
from src.betting import randomized_rank_pit

RULE = "=" * 104
ALPHAS = (0.60, 0.45, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
BUDGETS = (2000, 4000, 8000)


def sweep(df, z, tail, label):
    d = df.copy(); d["pit"] = z
    rows = []
    for a in ALPHAS:
        eng = DeadzoneEProcess(a, delta=0.75, lam=0.50, tail=tail)
        res = eng.run_sequential_test(d, z_col="pit", cooldown_months=12)
        n = int(res["is_alarm"].sum())
        if n == 0:
            continue
        m = windowed_metric(res, 24)
        c = windowed_metric(random_alarms(res, n, seed=5), 24,
                            alarm_col="rand_alarm")
        rows.append(dict(Variant=label, Alpha=a, Alarms=n, P=m["P"], R=m["R"],
                         F1=m["F1"], Chance_P=c["P"], Chance_F1=c["F1"]))
    return pd.DataFrame(rows)


def at_budget(sw, budget):
    """Linear interpolation of precision/recall/F1 at a fixed alarm count."""
    s = sw.sort_values("Alarms")
    if budget < s["Alarms"].min() or budget > s["Alarms"].max():
        return None
    return {k: float(np.interp(budget, s["Alarms"], s[k]))
            for k in ("P", "R", "F1", "Chance_P", "Chance_F1")}


def main():
    t0 = time.time()
    print(RULE); print("  PIT POWER AT MATCHED ALARM BUDGETS"); print(RULE)
    df = load()
    rng = np.random.default_rng(42)
    df["ss_score"], _ = semi_supervised_score(df)

    def rank_pit(col, keys):
        cohort = pd.MultiIndex.from_frame(df[keys]).to_numpy()
        return pd.Series(randomized_rank_pit(df[col].to_numpy(), cohort, rng)
                         ).fillna(0.5).to_numpy()

    variants = [
        ("A. model PIT, two-sided",
         df["randomized_pit"].fillna(0.5).to_numpy(), "two-sided"),
        ("B. model PIT, lower tail",
         df["randomized_pit"].fillna(0.5).to_numpy(), "lower"),
        ("C. rank PIT det_score | (date,rtg)   [old Z_det]",
         rank_pit("det_score", ["dates", "prev_enc_y"]), "upper"),
        ("D. rank PIT det_score | (date,rtg,age)",
         rank_pit("det_score", ["dates", "prev_enc_y", "age_bucket"]), "upper"),
        ("E. rank PIT semi-sup | (date,rtg)",
         rank_pit("ss_score", ["dates", "prev_enc_y"]), "upper"),
        ("F. rank PIT semi-sup | (date,rtg,age)",
         rank_pit("ss_score", ["dates", "prev_enc_y", "age_bucket"]), "upper"),
    ]

    sweeps = {}
    for label, z, tail in variants:
        sweeps[label] = sweep(df, z, tail, label)
        print(f"  swept {label}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{RULE}\n  PRECISION AT A MATCHED ALARM BUDGET (24-month window)\n{RULE}")
    hdr = f"{'variant':<50}" + "".join(f"{'n='+str(b):>22}" for b in BUDGETS)
    print(hdr)
    print(f"{'':<50}" + "".join(f"{'P24':>8}{'chance':>8}{'lift':>6}"
                                for _ in BUDGETS))
    print("-" * len(hdr))
    best = (None, 0.0)
    for label in sweeps:
        line = f"{label:<50}"
        lifts = []
        for b in BUDGETS:
            v = at_budget(sweeps[label], b)
            if v is None:
                line += f"{'—':>22}"
            else:
                lift = v["P"] / max(1e-9, v["Chance_P"])
                lifts.append(lift)
                line += f"{v['P']:>8.2f}{v['Chance_P']:>8.2f}{lift:>6.2f}"
        print(line)
        if lifts and np.mean(lifts) > best[1]:
            best = (label, float(np.mean(lifts)))
    print(f"\n  Strongest signal per alarm spent: {best[0]}  "
          f"(mean lift {best[1]:.2f}x)")

    allsw = pd.concat(sweeps.values(), ignore_index=True)
    allsw.to_csv(ROOT / "results" / "pit_power_sweep.csv", index=False)

    # ================= diagnostics =================
    cohort = pd.MultiIndex.from_frame(df[["dates", "prev_enc_y"]]).to_numpy()
    df["Z_rank"] = pd.Series(randomized_rank_pit(
        df["det_score"].to_numpy(), cohort, rng)).fillna(0.5).to_numpy()
    df["Z_model"] = df["randomized_pit"].fillna(0.5)

    # months until the next rating transition
    ttn = np.full(len(df), np.nan)
    dates = df["dates"].to_numpy()
    ids = df["isin"].to_numpy()
    chg = np.flatnonzero(df["is_rating_change"].to_numpy() == 1)
    by = {}
    for q in chg:
        by.setdefault(ids[q], []).append(q)
    for i in range(len(df)):
        nxt = [q for q in by.get(ids[i], ()) if dates[q] > dates[i]]
        if nxt:
            a, b = pd.Timestamp(dates[nxt[0]]), pd.Timestamp(dates[i])
            ttn[i] = (a.year - b.year) * 12 + (a.month - b.month)
    df["months_to_next"] = ttn

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    plt.subplots_adjust(hspace=.34, wspace=.24, top=.885, bottom=.075,
                        left=.07, right=.97)

    # (A) marginal histograms — the point is that the rank one is flat
    ax = axes[0, 0]
    ax.hist(df["Z_rank"], bins=25, range=(0, 1), density=True, alpha=.75,
            color="#2b3a9c", label="rank PIT (exchangeability null)")
    ax.hist(df["Z_model"], bins=25, range=(0, 1), density=True, alpha=.55,
            color="#d81b7a", label="model PIT (calibration null)")
    ax.axhline(1.0, color="#333", ls="--", lw=1.2, label="Uniform(0,1)")
    ax.set_title("(A) Marginal PIT — uninformative for the rank construction",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("PIT"); ax.set_ylabel("density"); ax.set_ylim(0, 1.6)
    ax.legend(fontsize=8.5); ax.grid(alpha=.25)

    # (B) the real diagnostic: PIT conditional on time-to-transition
    ax = axes[0, 1]
    bins = [(0, 6), (7, 12), (13, 24), (25, 48), (49, 10**6)]
    labels = ["0–6", "7–12", "13–24", "25–48", "49+"]
    for col, colr, nm in (("Z_rank", "#2b3a9c", "rank PIT"),
                          ("Z_model", "#d81b7a", "model PIT")):
        means, errs = [], []
        for lo, hi in bins:
            m = (df["months_to_next"] >= lo) & (df["months_to_next"] <= hi)
            v = df.loc[m, col]
            means.append(v.mean())
            errs.append(v.std() / np.sqrt(max(1, len(v))))
        ax.errorbar(range(len(bins)), means, yerr=np.array(errs) * 1.96,
                    marker="o", lw=2, capsize=4, color=colr, label=nm)
    ax.axhline(0.5, color="#333", ls="--", lw=1.2, label="Uniform mean = 0.5")
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(labels)
    ax.set_xlabel("months until the next rating transition")
    ax.set_ylabel("mean PIT")
    ax.set_title("(B) PIT vs time-to-transition — where the violation lives",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(alpha=.25)
    ax.invert_xaxis()

    # (C) persistence: lag-1..12 autocorrelation of a bond's PIT
    ax = axes[1, 0]
    for col, colr, nm in (("Z_rank", "#2b3a9c", "rank PIT"),
                          ("Z_model", "#d81b7a", "model PIT")):
        acf = []
        g = df.groupby("isin")[col]
        for L in range(1, 13):
            a = df[col].to_numpy()
            b = g.shift(L).to_numpy()
            ok = np.isfinite(a) & np.isfinite(b)
            acf.append(np.corrcoef(a[ok], b[ok])[0, 1])
        ax.plot(range(1, 13), acf, marker="o", lw=2, color=colr, label=nm)
    ax.axhline(0.0, color="#333", ls="--", lw=1.2, label="i.i.d. under H0")
    ax.set_xlabel("lag (months)"); ax.set_ylabel("autocorrelation")
    ax.set_title("(C) Serial dependence of a bond's PIT — H0 says zero",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(alpha=.25)

    # (D) staleness confound: PIT vs age of the standing rating
    ax = axes[1, 1]
    ab = [(0, 11), (12, 23), (24, 47), (48, 95), (96, 10**6)]
    al = ["0–11", "12–23", "24–47", "48–95", "96+"]
    for col, colr, nm in (("Z_rank", "#2b3a9c", "rank PIT"),
                          ("Z_model", "#d81b7a", "model PIT")):
        means, errs = [], []
        for lo, hi in ab:
            v = df.loc[(df["rating_age"] >= lo) & (df["rating_age"] <= hi), col]
            means.append(v.mean()); errs.append(v.std() / np.sqrt(max(1, len(v))))
        ax.errorbar(range(len(ab)), means, yerr=np.array(errs) * 1.96,
                    marker="s", lw=2, capsize=4, color=colr, label=nm)
    ax.axhline(0.5, color="#333", ls="--", lw=1.2)
    ax.set_xticks(range(len(ab))); ax.set_xticklabels(al)
    ax.set_xlabel("age of the standing rating (months)")
    ax.set_ylabel("mean PIT")
    ax.set_title("(D) Staleness confound — does age alone move the PIT?",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(alpha=.25)

    fig.suptitle("PIT uniformity diagnostics — the exchangeability null is a "
                 "statement about time, not about the marginal histogram",
                 fontsize=13.5, fontweight="bold", y=.965)
    for d in (ROOT / "plots",):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "pit_uniformity_diagnostics.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
    print(f"\nsaved -> Reports/plots/pit_uniformity_diagnostics.png")

    # ---- PR frontier ----
    fig2, ax = plt.subplots(figsize=(9.5, 7))
    colors = ["#2b3a9c", "#7a8ce8", "#d81b7a", "#f08cbb", "#186b4e", "#6cc79b"]
    for (label, _, _), c in zip(variants, colors):
        s = sweeps[label].sort_values("R")
        ax.plot(s["R"], s["P"], marker="o", lw=1.9, ms=5, color=c, label=label)
    base = 100 * df.groupby("isin")["is_rating_change"].transform(
        lambda x: x.iloc[::-1].rolling(24, min_periods=1).max().iloc[::-1].shift(-1)
    ).fillna(0).mean()
    ax.axhline(base, color="#333", ls="--", lw=1.4,
               label=f"chance precision ({base:.1f}%)")
    ax.set_xlabel("Recall at 24 months (%)"); ax.set_ylabel("Precision at 24 months (%)")
    ax.set_title("Precision–recall frontier by PIT construction\n"
                 r"(WZ deadzone $\delta$=0.75 $\lambda$=0.50, $\alpha$ swept "
                 "0.005–0.60)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(alpha=.25)
    for d in (ROOT / "plots",):
        fig2.savefig(d / "pit_precision_recall_frontier.png", dpi=150,
                     bbox_inches="tight", facecolor="white")
    print(f"saved -> Reports/plots/pit_precision_recall_frontier.png")
    print(f"\ntotal {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
