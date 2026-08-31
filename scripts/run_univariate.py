"""Section 4.2 — the univariate exploration, reproduced and measured.

Reproduces Figure 4.1 (the single-bond PIT trajectories) from eq. (4.3)
rather than from the KDE-smoothed CDF the legacy notebook plots, and emits
the four diagnostic tables the section quotes:

  T1  cohort sizes and the PIT lattice
  T2  ECDF (eq. 4.3) vs the KDE-smoothed CDF actually plotted
  T3  persistence of the PIT — the conditional-uniformity verdict
  T4  separability of the rating class, and early-warning AUC

Usage
    conda activate copula          # or bond; numpy 1.x and 2.x both work
    python scripts/run_univariate.py
    python scripts/run_univariate.py --isin US171232AF85 --horizon 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.univariate import (DEFAULT_COVARIATES, REDUNDANT_AT_NOTCH_LEVEL,  # noqa: E402
                            auc, counterfactual_cohort, kde_cdf_pit,
                            load_bond_panel, pit_persistence, rank_pit,
                            rating_path, transition_labels)

BASE = Path(__file__).resolve().parents[1]
DEFAULT_CSV = BASE.parent / "CorpBond_Reconciling" / "corp_jkp_mergedv2.csv"
# Section 4.2 also reports duration and D2D, which Figure 4.1 omits.
EXTRA = ("duration", "D2D")
# The 16-notch scale of the panel, grouped into the six letter classes.
LETTER_OF = {1: "AAA", **{k: "AA" for k in (2, 3, 4)},
             **{k: "A" for k in (5, 6, 7)}, **{k: "BBB" for k in (8, 9, 10)},
             **{k: "BB" for k in (11, 12, 13)}, **{k: "B" for k in (14, 15, 16)}}


def table(fh, title, header, rows, widths):
    line = f"\n{title}\n" + "-" * len(title) + "\n"
    line += "".join(f"{h:>{w}}" if i else f"{h:<{w}}"
                    for i, (h, w) in enumerate(zip(header, widths))) + "\n"
    for r in rows:
        line += "".join(f"{c:>{w}}" if i else f"{c:<{w}}"
                        for i, (c, w) in enumerate(zip(r, widths))) + "\n"
    print(line, end="")
    fh.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--isin", default="US171232AF85")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--kde-dates", type=int, default=24,
                    help="random dates sampled for the ECDF-vs-KDE comparison")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    covs = list(DEFAULT_COVARIATES) + list(EXTRA)
    df = load_bond_panel(args.csv, covariates=tuple(covs))
    for c in covs:
        df = rank_pit(df, c, cohort=("dates", "nrtg"))

    (BASE / "results").mkdir(exist_ok=True)
    out = open(BASE / "results" / "univariate_summary.txt", "w")
    out.write(f"Section 4.2 diagnostics\npanel: {df.height} rows, "
              f"{df['isin'].n_unique()} bonds, {df['dates'].n_unique()} dates, "
              f"{df['dates'].min()}--{df['dates'].max()}\n")
    print(f"panel: {df.height} rows, {df['isin'].n_unique()} bonds, "
          f"{df['dates'].n_unique()} dates")

    # -------------------------------------------------------------- T0 redundancy
    msg = "\nredundancy check (identity of the rank PIT, notch cohorts)\n"
    for a, b in REDUNDANT_AT_NOTCH_LEVEL.items():
        za, zb = df[f"Z_{a}"].to_numpy(), df[f"Z_{b}"].to_numpy()
        m = np.isfinite(za) & np.isfinite(zb)
        share = float(np.isclose(za[m], zb[m]).mean())
        ratio = (df[a].to_numpy() / df[b].to_numpy())
        g = np.isfinite(ratio)
        exact = float(np.isclose(ratio[g], df["nrtg"].to_numpy()[g]).mean())
        msg += (f"  {a} = {b} * nrtg on {exact:.2%} of rows;  "
                f"Z_{a} == Z_{b} on {share:.2%} of rows\n")
    print(msg, end=""); out.write(msg)

    # ------------------------------------------------------------------- T1 lattice
    sz = df.group_by(["dates", "nrtg"]).len()["len"].to_numpy()
    q = np.percentile(sz, [5, 25, 50, 75, 95])
    zs = df["Z_spread"].drop_nulls().to_numpy()
    rows = [
        ("median cohort size n_t", f"{int(q[2])}"),
        ("IQR", f"[{int(q[1])}, {int(q[3])}]"),
        ("5th / 95th pct", f"{int(q[0])} / {int(q[4])}"),
        ("share of cohorts with n < 20", f"{(sz < 20).mean():.2%}"),
        ("E[Z] under H0 = mean (n+1)/2n", f"{np.mean((sz + 1) / (2 * sz)):.4f}"),
        ("observed mean Z (spread)", f"{zs.mean():.4f}"),
        ("P(Z = 1) observed", f"{(zs >= 1 - 1e-12).mean():.4f}"),
    ]
    table(out, "T1  cohort sizes and the PIT lattice", ("quantity", "value"),
          rows, (34, 16))

    # ---------------------------------------------------------------- T2 ECDF/KDE
    rng = np.random.default_rng(args.seed)
    dates_all = df["dates"].unique().sort().to_list()
    samp = rng.choice(dates_all, size=min(args.kde_dates, len(dates_all)),
                      replace=False)
    rows = []
    for c in DEFAULT_COVARIATES:
        diffs = []
        for dt in samp:
            for g in (3, 6, 9, 12, 15):
                v = df.filter((pl.col("dates") == dt) & (pl.col("nrtg") == g))[c]
                v = v.drop_nulls().to_numpy()
                v = v[np.isfinite(v)]
                if len(v) < 10 or np.std(v) < 1e-8:
                    continue
                ecdf = np.array([np.mean(v <= x) for x in v])
                kd = kde_cdf_pit(v, v)
                if np.isfinite(kd).all():
                    diffs.append(np.abs(ecdf - kd))
        if diffs:
            a = np.concatenate(diffs)
            rows.append((c, f"{a.mean():.4f}", f"{np.percentile(a, 95):.4f}",
                         f"{a.max():.4f}"))
    table(out, "T2  ECDF (eq. 4.3) vs the KDE-smoothed CDF the notebook plots",
          ("covariate", "mean |d|", "p95 |d|", "max |d|"), rows, (14, 10, 10, 10))

    # -------------------------------------------------------------- T3 persistence
    rows = []
    for c in covs:
        if c in REDUNDANT_AT_NOTCH_LEVEL:
            continue
        r = pit_persistence(df, c)
        hl = f"{r['half_life']:.0f}" if np.isfinite(r["half_life"]) else "-"
        rows.append((c, f"{r[1]:.3f}", f"{r[3]:.3f}", f"{r[6]:.3f}",
                     f"{r[12]:.3f}", hl))
    rows.sort(key=lambda r: -float(r[1]))
    table(out, "T3  persistence of the cohort PIT (lag-k ACF, contiguous months)",
          ("covariate", "ACF(1)", "ACF(3)", "ACF(6)", "ACF(12)", "half-life"),
          rows, (14, 9, 9, 9, 9, 11))

    # ------------------------------------------------- T4a what a migration is
    isin = df["isin"].to_numpy(); mi = df["midx"].to_numpy()
    g = df["nrtg"].to_numpy()
    adj_ok = (isin[1:] == isin[:-1]) & (mi[1:] - mi[:-1] == 1)
    delta = (g[1:] - g[:-1])[adj_ok]
    moves = delta[delta != 0]
    a_, b_ = g[:-1][adj_ok], g[1:][adj_ok]
    sel = a_ != b_
    crosses = np.mean([LETTER_OF.get(int(x)) != LETTER_OF.get(int(y))
                       for x, y in zip(a_[sel], b_[sel])])
    nb = df.group_by("isin").agg(pl.col("nrtg").n_unique().alias("k"))
    rows = [
        ("month-to-month notch moves", f"{len(moves):,}"),
        ("as a share of contiguous pairs", f"{len(moves)/adj_ok.sum():.2%}"),
        ("|delta nrtg| = 1", f"{np.mean(np.abs(moves) == 1):.2%}"),
        ("|delta nrtg| <= 2", f"{np.mean(np.abs(moves) <= 2):.2%}"),
        ("also crosses a letter boundary", f"{crosses:.2%}"),
        ("bonds with >= 1 notch move", f"{(nb['k'] > 1).mean():.1%}"),
    ]
    table(out, "T4a  what a rating migration actually is",
          ("quantity", "value"), rows, (34, 16))

    # ------------------------------------------------------ T4 separability / power
    dn, up, obs = transition_labels(df, horizon=args.horizon)
    base = (f"\nH = {args.horizon} months.  observed rows {obs.sum():,} of "
            f"{len(obs):,};  base rate: downgrade {dn[obs].mean():.2%}, "
            f"upgrade {up[obs].mean():.2%}, any {(dn | up)[obs].mean():.2%}\n")
    print(base, end=""); out.write(base)

    from scipy.stats import spearmanr
    rows = []
    for c in covs:
        if c in REDUNDANT_AT_NOTCH_LEVEL:
            continue
        rhos, adj, ext = [], [], []
        for dt in dates_all[::6]:
            sub = df.filter(pl.col("dates") == dt)
            x = sub[c].to_numpy().astype(float)
            g = sub["nrtg"].to_numpy().astype(float)
            m = np.isfinite(x) & np.isfinite(g)
            if m.sum() < 50:
                continue
            rhos.append(spearmanr(x[m], g[m]).statistic)
            for k in range(2, 16):
                a_, b_ = x[m][g[m] == k], x[m][g[m] == k + 1]
                if len(a_) >= 10 and len(b_) >= 10:
                    s = np.concatenate([a_, b_])
                    lab = np.r_[np.zeros(len(a_), bool), np.ones(len(b_), bool)]
                    adj.append(max(auc(s, lab), 1 - auc(s, lab)))
            a_, b_ = x[m][g[m] <= 2], x[m][g[m] >= 14]
            if len(a_) >= 10 and len(b_) >= 10:
                s = np.concatenate([a_, b_])
                lab = np.r_[np.zeros(len(a_), bool), np.ones(len(b_), bool)]
                ext.append(max(auc(s, lab), 1 - auc(s, lab)))
        z = df[f"Z_{c}"].to_numpy()
        rows.append((c, f"{np.nanmean(rhos):+.3f}", f"{np.nanmean(adj):.3f}",
                     f"{np.nanmean(ext):.3f}",
                     f"{auc(z[obs], dn[obs]):.3f}", f"{auc(z[obs], up[obs]):.3f}",
                     f"{auc(np.abs(z[obs] - 0.5), (dn | up)[obs]):.3f}"))
    table(out, "T4  what one covariate can and cannot resolve",
          ("covariate", "rho(x,nrtg)", "AUC adj", "AUC far",
           "AUC down", "AUC up", "AUC any"),
          rows, (14, 13, 9, 9, 10, 9, 9))
    out.write("\nAUC adj  = adjacent notches k vs k+1, chance 0.5\n"
              "AUC far  = {AAA,AA} vs {B}, chance 0.5\n"
              "AUC down/up/any = the PIT as an early-warning score at horizon H\n")

    # ------------------------------------------------------------------- Figure 4.1
    make_figure(df, args.isin)
    out.close()
    print(f"\nwrote results/univariate_summary.txt")


def make_figure(df, isin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })
    print(f"\nrating path of {isin}:")
    print(rating_path(df, isin))

    b = counterfactual_cohort(df, isin, "nrtg").sort("dates")
    # the letter-class counterfactual, same construction
    b = b.with_columns(
        pl.when((pl.col("rtg") != pl.col("rtg").shift(1)) &
                pl.col("rtg").shift(1).is_not_null())
        .then(pl.col("rtg").shift(1)).otherwise(None)
        .fill_null(strategy="forward").alias("prev_rtg"))

    covs = list(DEFAULT_COVARIATES)
    dates = b["dates"].to_list()
    tf = np.array([(d // 100) + ((d % 100) - 0.5) / 12 for d in dates])

    # One grey level per notch the bond visits, ordered by credit quality:
    # lightest is the best rating it holds, darkest the worst.  A monotone
    # ramp keeps the ordering readable, prints unchanged in black and white,
    # and stays far enough below the saturation of the trajectory colours that
    # the lines remain unambiguously in the foreground.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    BAND_LIGHTEST, BAND_DARKEST = 0.96, 0.84

    nr = b["nrtg"].to_list()
    notches = sorted(set(nr))
    notch_colour = {
        g: str(BAND_LIGHTEST if len(notches) == 1 else
               BAND_LIGHTEST - (BAND_LIGHTEST - BAND_DARKEST)
               * k / (len(notches) - 1))
        for k, g in enumerate(notches)}

    identity = {}
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.0), sharex=True)
    for ax, c in zip(axes.ravel(), covs):
        series = {k: [] for k in ("nrtg", "rtg", "prev_nrtg", "prev_rtg")}
        raw = []
        for row in b.iter_rows(named=True):
            x = row[c]
            raw.append(x)
            peers = df.filter(pl.col("dates") == row["dates"])
            for key, col, val in (("nrtg", "nrtg", row["nrtg"]),
                                  ("rtg", "rtg", row["rtg"]),
                                  ("prev_nrtg", "nrtg", row["prev_nrtg"]),
                                  ("prev_rtg", "rtg", row["prev_rtg"])):
                if val is None or x is None or not np.isfinite(x):
                    series[key].append(np.nan); continue
                v = peers.filter(pl.col(col) == val)[c].drop_nulls().to_numpy()
                v = v[np.isfinite(v)]
                series[key].append(np.mean(v <= x) if len(v) else np.nan)
        identity[c] = (np.asarray(series["nrtg"], float),
                       np.asarray(series["rtg"], float))
        ax.plot(tf, series["nrtg"], color="tab:blue", lw=1.0, label=r"$Z_t$ (notch)")
        ax.plot(tf, series["rtg"], color="tab:orange", lw=1.0, label=r"$Z_t$ (letter)")
        ax.plot(tf, series["prev_nrtg"], color="tab:green", lw=0.8, ls=":",
                label="counterfactual notch")
        ax.plot(tf, series["prev_rtg"], color="tab:red", lw=0.8, ls=":",
                label="counterfactual letter")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(c)
        ax2 = ax.twinx()
        ax2.plot(tf, raw, color="tab:purple", lw=0.8, ls="--", alpha=0.7)
        ax2.tick_params(axis="y", labelsize=6, colors="tab:purple")
        ax2.grid(False)
        # shade the notch regime.  Spans run to the *next* regime's first month
        # so the bands are contiguous rather than leaving one-month gaps.
        s = 0
        for i in range(1, len(nr) + 1):
            if i == len(nr) or nr[i] != nr[s]:
                ax.axvspan(tf[s], tf[i] if i < len(nr) else tf[-1],
                           facecolor=notch_colour[nr[s]], edgecolor="none",
                           zorder=0)
                if i < len(nr):
                    # Mark the transition date itself.  It reads the exact
                    # month off the axis, and it keeps the regimes legible if
                    # the thesis is printed in black and white, where the
                    # bands are deliberately equal in lightness.
                    ax.axvline(tf[i], color="0.35", lw=0.5, ls=(0, (3, 2)),
                               zorder=1)
                s = i
    for ax in axes.ravel()[len(covs):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("year")
    for ax in axes.ravel():
        ax.set_xticks([2003, 2006, 2009])
        ax.set_xticklabels(["2003", "2006", "2009"])
    h, lab = axes[0, 0].get_legend_handles_labels()
    # the raw covariate lives on the twin axis and was missing from the legend
    h.append(Line2D([], [], color="tab:purple", lw=0.8, ls="--", alpha=0.7))
    lab.append("raw covariate (right axis)")
    h.append(Line2D([], [], color="0.35", lw=0.5, ls=(0, (3, 2))))
    lab.append("rating change")
    for g in notches:
        h.append(Patch(facecolor=notch_colour[g], edgecolor="0.6", lw=0.4))
        lab.append(f"notch {g} ({LETTER_OF[g]})")
    fig.legend(h, lab, loc="lower right", bbox_to_anchor=(0.98, 0.05), ncol=1,
               frameon=False, handlelength=1.8, labelspacing=0.35)
    fig.suptitle(f"Cohort PIT of eq. (4.3), bond {isin}", y=0.995)
    fig.tight_layout()
    (BASE / "plots").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(BASE / "plots" / f"fig_univariate_pit.{ext}")
    print("wrote plots/fig_univariate_pit.pdf / .png")

    # the redundancy claim, on this bond's own trajectories
    a_n, a_l = identity["spread"]
    b_n, b_l = identity["mom6xrtg"]
    m = np.isfinite(a_n) & np.isfinite(b_n)
    ml = np.isfinite(a_l) & np.isfinite(b_l)
    print(f"  on this bond: notch PIT spread == mom6xrtg on "
          f"{np.isclose(a_n[m], b_n[m]).mean():.0%} of months; "
          f"letter PIT on {np.isclose(a_l[ml], b_l[ml]).mean():.0%}")


if __name__ == "__main__":
    main()
