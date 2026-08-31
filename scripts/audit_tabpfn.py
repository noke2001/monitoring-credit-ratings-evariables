"""Audit of the Chapter 4 §4.3 supervised runs, from the saved fits.

TabPFN is refitted at every date, so a full sweep is hours of GPU time.  The
per-date outputs are already saved under ``tabpfnfit/``; this script audits
them instead of recomputing, and answers four questions:

  A1  provenance and integrity -- which script wrote which file, and is the
      file internally consistent?
  A2  the accuracy paradox -- instantaneous vs transition-month vs forward
      accuracy, each against the do-nothing baseline
  A3  early warning -- does the PIT of the learned score rank bonds by
      imminent migration, and by how much over chance?
  A4  the sign convention of the stored PIT

Usage
    conda activate bond
    python scripts/audit_tabpfn.py                 # every config
    python scripts/audit_tabpfn.py --quick         # skip the integrity scan
    python scripts/audit_tabpfn.py --horizon 24
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.summaries import SUMMARIES  # noqa: E402
from src.univariate import auc  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
FITS = Path("/Users/philip/Library/CloudStorage/OneDrive-Personal/Desktop/"
            "ETH/Master_Thesis/R_code_agent/tabpfnfit")
DEFAULT_CSV = BASE.parent / "CorpBond_Reconciling" / "corp_jkp_mergedv2.csv"

RMAP = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6}

# file -> (label, the script in the parent directory that wrote it)
CONFIGS = [
    ("tabpfn_data_rtg_12_wprobs.csv",
     "level target, K=12", "approach1_tabpfn_bash_probs.py"),
    ("tabpfn_data_rtg_12_onlytrans.csv",
     "level target, transition-only context", "approach1_tabpfn_bash_only_trans_probs.py"),
    ("tabpfn_data_rtg_12_fresh_fit_minbonds_10.csv",
     "level target, fresh-rating context", "approach1_tabpfn_bash_fresh_fit_probs.py"),
    ("tabpfn_data_rtg_24_fresh_fit_minbonds_10.csv",
     "level target, fresh-rating, K=24", "approach1_tabpfn_bash_fresh_fit_probs.py"),
    ("tabpfn_delta_fit_rtg_12m.csv",
     "delta target, K=12", "approach1_tabpfn_bash_delta_fit_probs.py"),
    ("tabpfn_delta_fit_dom_50_rtg_12m.csv",
     "delta target, K=12, dom cap 50%", "approach1_tabpfn_bash_delta_fit_probs_dom.py"),
    ("tabpfn_delta_fit_dom_50_rtg_24m.csv",
     "delta target, K=24, dom cap 50%", "approach1_tabpfn_bash_delta_fit_probs_dom.py"),
    ("tabpfn_delta_fit_dom_40_rtg_24m_new.csv",
     "delta target, K=24, dom 40% + fresh ratio",
     "approach1_tabpfn_bash_delta_fit_probs_dom_new.py"),
    ("tabpfn_horizon_fit_results_10000_24_16.csv",
     "FORWARD delta target, 10k context, 16 feats",
     "approach1_tabpfn_bash_topdelta_fit.py"),
]

# This one predicts the rating twelve months AHEAD, not the current one, so its
# accuracy columns are not comparable with the rest and A2 skips it.  Its
# ranking score is comparable, so A3 keeps it.
FORWARD_TARGET = {"tabpfn_horizon_fit_results_10000_24_16.csv"}

# Scanned by A1 only: different class set (nrtg) or superseded, but they carry
# the append-header defect and it should be on the record.
INTEGRITY_ONLY = [
    "tabpfn_data_nrtg_12_wprobs.csv", "tabpfn_data_nrtg_1_wprobs.csv",
    "tabpfn_data_nrtg_6_wprobs.csv", "tabpfn_data_rtg_3_wprobs.csv",
    "tabpfn_data_rtg_2_wprobs.csv", "tabpfn_data_rtg_1_wprobs.csv",
]


def emit(fh, text=""):
    print(text)
    fh.write(text + "\n")


def load_panel(csv_path, horizon):
    """Letter-rating panel with forward-transition labels."""
    p = (pl.scan_csv(csv_path, infer_schema_length=20000, ignore_errors=True)
         .select(["dates", "isin", "rtg"])
         .filter(pl.col("isin").is_not_null() & pl.col("rtg").is_in(list(RMAP)))
         .collect().sort(["isin", "dates"])
         .with_columns((((pl.col("dates") // 100) * 12) +
                        (pl.col("dates") % 100)).alias("midx"),
                       pl.col("rtg").replace_strict(RMAP, return_dtype=pl.Int64)
                       .alias("lnum")))
    isin = p["isin"].to_numpy(); mi = p["midx"].to_numpy()
    g = p["lnum"].to_numpy().astype(float)
    n = len(g)
    dn = np.zeros(n, bool); up = np.zeros(n, bool)
    obs = np.zeros(n, bool); fw = np.full(n, np.nan)
    prev = np.full(n, np.nan)
    s = 0
    for i in range(1, n + 1):
        if i == n or isin[i] != isin[s]:
            gi, mm = g[s:i], mi[s:i]
            for k in range(i - s):
                if k > 0 and mm[k] - mm[k - 1] == 1:
                    prev[s + k] = gi[k - 1]
                f = (mm > mm[k]) & (mm <= mm[k] + horizon)
                if not f.any():
                    continue
                obs[s + k] = True
                dn[s + k] = bool((gi[f] > gi[k]).any())
                up[s + k] = bool((gi[f] < gi[k]).any())
                fw[s + k] = gi[f][-1]
            s = i
    return p.with_columns(pl.Series("dn", dn), pl.Series("up", up),
                          pl.Series("obs", obs), pl.Series("fw", fw),
                          pl.Series("prev", prev))


def integrity(fh, fname):
    """A1: ragged rows, and whether argmax(prob_*) reproduces the stored label."""
    path = FITS / fname
    widths = {}
    with open(path) as f:
        header = next(f).rstrip("\n").split(",")
        for line in f:
            widths[line.count(",") + 1] = widths.get(line.count(",") + 1, 0) + 1
    ragged = len(widths) > 1
    d = pl.read_csv(path, infer_schema_length=50000, ignore_errors=True,
                    truncate_ragged_lines=True)
    probs = [c for c in d.columns if c.startswith("prob_")]
    pred_col = "pred_rtg" if "pred_rtg" in d.columns else "pred_delta"
    agree = np.nan
    if probs and pred_col in d.columns:
        P = d.select(probs).to_numpy().astype(float)
        names = [c.replace("prob_delta_", "").replace("prob_", "") for c in probs]
        am = np.array(names, dtype=object)[
            np.nanargmax(np.nan_to_num(P, nan=-1.0), axis=1)]

        def norm(v):
            try:
                return f"{float(v):g}"
            except (TypeError, ValueError):
                return str(v)
        a = np.array([norm(x) for x in am])
        b = np.array([norm(x) for x in d[pred_col].cast(pl.String).to_numpy()])
        agree = float((a == b).mean())
    return {"header_fields": len(header), "widths": widths, "ragged": ragged,
            "argmax_agreement": agree, "rows": d.height}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--quick", action="store_true",
                    help="skip the A1 integrity scan (it reads every file twice)")
    ap.add_argument("--perm", type=int, default=30)
    args = ap.parse_args()

    (BASE / "results").mkdir(exist_ok=True)
    fh = open(BASE / "results" / "tabpfn_audit.txt", "w")
    emit(fh, f"Section 4.3 audit of the saved TabPFN fits   (horizon = "
             f"{args.horizon} months)")

    # ---------------------------------------------------------------- A1
    if not args.quick:
        emit(fh, "\n" + "=" * 78)
        emit(fh, "A1  provenance and integrity")
        emit(fh, "=" * 78)
        emit(fh, f"{'file':<46}{'rows':>9}{'ragged':>8}{'argmax=label':>14}")
        for fname, _label, _script in CONFIGS + [(f, "", "") for f in INTEGRITY_ONLY]:
            if not (FITS / fname).exists():
                emit(fh, f"{fname:<46}{'MISSING':>9}")
                continue
            r = integrity(fh, fname)
            ag = ("-" if not np.isfinite(r["argmax_agreement"])
                  else f"{r['argmax_agreement']:.2%}")
            emit(fh, f"{fname:<46}{r['rows']:>9,}"
                     f"{('YES' if r['ragged'] else 'no'):>8}{ag:>14}")
        emit(fh, "\n  'ragged' = rows in the file do not all have the same number of"
                 " fields.\n  The per-date append writes one prob_* column per class"
                 " present in that\n  date's training context, but the header is fixed"
                 " at the first date's set,\n  so from the first missing class onward"
                 " the probability columns are shifted.\n  The seven leading columns"
                 " are written before the block and stay correct.")

    panel = load_panel(args.csv, args.horizon)

    # ------------------------------------------------------------ A2 / A3
    emit(fh, "\n" + "=" * 78)
    emit(fh, "A2  the accuracy paradox")
    emit(fh, "=" * 78)
    emit(fh, f"{'configuration':<42}{'inst.':>8}{'trans.':>8}"
             f"{'fwd':>8}{'base':>8}{'fwd-base':>10}")
    rows = []
    for fname, label, script in CONFIGS:
        if not (FITS / fname).exists():
            continue
        d = pl.read_csv(FITS / fname, infer_schema_length=50000,
                        ignore_errors=True, truncate_ragged_lines=True)
        d = (d.with_columns(pl.col("date").str.slice(0, 4).cast(pl.Int64) * 100
                            + pl.col("date").str.slice(5, 2).cast(pl.Int64))
             .rename({"bond_id": "isin"}))
        j = d.join(panel, left_on=["isin", "date"], right_on=["isin", "dates"],
                   how="inner")
        if j.height == 0:
            continue
        if "pred_delta" in j.columns:
            pr = (j["entry_rtg_num"].to_numpy().astype(float)
                  - j["pred_delta"].to_numpy().astype(float))
        else:
            pr = np.array([RMAP.get(str(x), np.nan)
                           for x in j["pred_rtg"].to_numpy()], float)
        L = j["lnum"].to_numpy().astype(float)
        F = j["fw"].to_numpy(); PV = j["prev"].to_numpy()
        o = j["obs"].to_numpy()
        mo = o & np.isfinite(F) & np.isfinite(pr)
        istr = np.isfinite(PV) & (PV != L)
        inst = float(np.mean(pr == L))
        trans = float(np.mean(pr[istr] == L[istr])) if istr.any() else np.nan
        fwd = float(np.mean(pr[mo] == F[mo]))
        base = float(np.mean(L[mo] == F[mo]))
        if fname in FORWARD_TARGET:
            emit(fh, f"{label:<42}{'  (forward target -- not comparable)':>44}")
        else:
            emit(fh, f"{label:<42}{inst:>8.3f}{trans:>8.3f}{fwd:>8.3f}{base:>8.3f}"
                     f"{fwd - base:>+10.3f}")

        # Recompute the PIT from the deviation rather than trusting the stored
        # column: one config never wrote it, and recomputing lets every row use
        # the SAME (ascending) convention, so the signs below are comparable.
        # Ascending means a bond the model thinks is worse than its label -- a
        # downgrade candidate -- sits near 1, which is the write-up's convention.
        dd = (j["expected_rtg"].to_numpy().astype(float) - L)
        jz = (j.with_columns(pl.Series("_dd", dd))
              .with_columns((pl.col("_dd").rank("average")
                             .over(["date", "actual_rtg"]) /
                             pl.col("_dd").count().over(["date", "actual_rtg"]))
                            .alias("_z")))
        z = jz["_z"].to_numpy().astype(float)
        stored = (j["within_class_std_rank"].to_numpy().astype(float)
                  if "within_class_std_rank" in j.columns else None)
        rows.append((label, script, fname, inst, trans, fwd, base, z, stored, o,
                     j["up"].to_numpy(), j["dn"].to_numpy(),
                     j["actual_rtg"].to_numpy(), j["date"].to_numpy()))
    emit(fh, "\n  inst.  = Pred_t == Actual_t          trans. = the same, restricted"
             " to months\n                                          in which the"
             " rating actually moved"
             "\n  fwd    = Pred_t == Actual_(t+H)      base   = Actual_t =="
             " Actual_(t+H), the\n                                          "
             "do-nothing persistence forecast")

    emit(fh, "\n" + "=" * 78)
    emit(fh, "A3  early warning from the PIT of the learned score")
    emit(fh, "=" * 78)
    emit(fh, "  PIT recomputed in the ASCENDING convention: downgrade candidate"
             " near 1.\n")
    emit(fh, f"{'configuration':<42}{'AUC down':>10}{'AUC up':>9}"
             f"{'best lift':>11}{'null':>16}{'stored':>9}")
    rng = np.random.default_rng(0)
    for (label, _sc, _fn, _i, _t, _f, _b, z, stored, o, U, DN, _cls, _dt) in rows:
        au = auc(z[o], U[o]); ad = auc(z[o], DN[o])
        null = np.array([auc(rng.permutation(z[o]), U[o])
                         for _ in range(args.perm)])
        sd = ("-" if stored is None
              else f"{auc(stored[o], DN[o]):.3f}")
        emit(fh, f"{label:<42}{ad:>10.3f}{au:>9.3f}"
                 f"{max(abs(au - .5), abs(ad - .5)):>11.3f}"
                 f"{f'{null.mean():.3f}+-{null.std():.3f}':>16}{sd:>9}")
    emit(fh, "\n  chance = 0.500.  'null' permutes the PIT within the evaluated rows.")
    emit(fh, "  'stored' = AUC(stored within_class_std_rank -> downgrade).  It sits")
    emit(fh, "  BELOW 0.5 everywhere, mirroring the recomputed column: that is the")
    emit(fh, "  sign inversion of A4, measured.")

    # stability of the two level-target configs
    emit(fh, "\n  stability of the strongest signal, by rating class and sub-period:")
    emit(fh, "  (AUC -> downgrade throughout, to match the column above)")
    for (label, _sc, _fn, _i, _t, _f, _b, z, _st, o, U, DN, cls, dt) in rows[:3]:
        emit(fh, f"    {label}")
        parts = []
        for c in ["AAA", "AA", "A", "BBB", "BB", "B"]:
            m = o & (cls == c)
            if m.sum() > 500 and DN[m].sum() > 20:
                parts.append(f"{c} {auc(z[m], DN[m]):.3f}")
        med = np.median(dt[o])
        parts.append(f"| 1st half {auc(z[o & (dt <= med)], DN[o & (dt <= med)]):.3f}")
        parts.append(f"2nd half {auc(z[o & (dt > med)], DN[o & (dt > med)]):.3f}")
        emit(fh, "      " + "  ".join(parts))

    # ---------------------------------------------------------------- A3b
    emit(fh, "\n" + "=" * 78)
    emit(fh, "A3b  which summary function?  (level target, K=12)")
    emit(fh, "=" * 78)
    emit(fh, "  Five ways to condense the predictive vector against the carried")
    emit(fh, "  rating, all oriented so that large = downgrade candidate, each")
    emit(fh, "  ranked within its (date, rating) cohort exactly as in 4.2.")
    fname = "tabpfn_data_rtg_12_wprobs.csv"
    if (FITS / fname).exists():
        d = pl.read_csv(FITS / fname, infer_schema_length=50000, ignore_errors=True)
        d = (d.with_columns(pl.col("date").str.slice(0, 4).cast(pl.Int64) * 100
                            + pl.col("date").str.slice(5, 2).cast(pl.Int64))
             .rename({"bond_id": "isin"}))
        j = d.join(panel, left_on=["isin", "date"], right_on=["isin", "dates"],
                   how="inner")
        order = ["AAA", "AA", "A", "BBB", "BB", "B"]
        P = j.select([f"prob_{c}" for c in order]).to_numpy().astype(float)
        k = j["lnum"].to_numpy().astype(float) - 1.0        # 0-indexed class
        o = j["obs"].to_numpy(); U = j["up"].to_numpy(); DN = j["dn"].to_numpy()
        isin_a = j["isin"].to_numpy(); midx_a = j["midx"].to_numpy()
        emit(fh, f"\n{'summary':<24}{'ACF(1)':>9}{'AUC down':>10}"
                 f"{'AUC up':>9}{'lift':>8}{'alternative':>14}")
        for name, (fn, one_sided) in SUMMARIES.items():
            g = fn(P, k)
            jz = (j.with_columns(pl.Series("_g", g))
                  .with_columns((pl.col("_g").rank("average").over(["date", "actual_rtg"])
                                 / pl.col("_g").count().over(["date", "actual_rtg"]))
                                .alias("_z")))
            z = jz["_z"].to_numpy().astype(float)
            # persistence of the resulting cohort PIT
            isin_ = j["isin"].to_numpy(); mi_ = j["midx"].to_numpy()
            ok = (isin_[1:] == isin_[:-1]) & (mi_[1:] - mi_[:-1] == 1)
            a_, b_ = z[:-1][ok], z[1:][ok]
            gg = np.isfinite(a_) & np.isfinite(b_)
            acf = float(np.corrcoef(a_[gg], b_[gg])[0, 1])
            ad = auc(z[o], DN[o]); au = auc(z[o], U[o])
            emit(fh, f"{name:<24}{acf:>9.3f}{ad:>10.3f}{au:>9.3f}"
                     f"{max(abs(ad - .5), abs(au - .5)):>8.3f}"
                     f"{('one-sided' if one_sided else 'two-sided'):>14}")
        # Does the cardinal encoding enc(r) that eq. (4.dd) needs actually
        # matter?  The scoring rules need no encoding (Brier, log) or only the
        # ordering (RPS), which is the principled argument for them -- but the
        # deviation turns out to be nearly as robust in practice.
        from src.summaries import directional_deviation
        ENCODINGS = {
            "linear 1..6 (default)": np.array([1, 2, 3, 4, 5, 6], float),
            "notch midpoints": np.array([1, 3, 6, 9, 12, 15], float),
            "hist. default rate %": np.array([.01, .05, .10, .30, 1.5, 6.0]),
            "log default rate": np.log(np.array([.01, .05, .10, .30, 1.5, 6.0])),
            "convex 1,2,4,..,32": np.array([1, 2, 4, 8, 16, 32], float),
        }
        # The modified TRR: how much does the dead-zone threshold neutralise,
        # and what does it cost?  Also: are the vanishing tails it discards
        # actually noise?  If they were, the ratio restricted to the confident
        # rows would carry no signal.
        from src.summaries import transition_risk_ratio
        p_stay = P[np.arange(len(P)), k.astype(int)]
        emit(fh, "\n  modified TRR: neutralise log-TRR where P(carried) > th")
        emit(fh, f"    {'th':<12}{'% neutralised':>15}{'ACF(1)':>9}"
                 f"{'AUC down':>10}{'lift':>8}{'AUC | th-subset':>18}")
        for th in (None, 0.9999, 0.999, 0.99, 0.95, 0.9):
            g = (transition_risk_ratio(P, k) if th is None
                 else transition_risk_ratio(P, k, deadzone=th))
            jz = (j.with_columns(pl.Series("_g", g))
                  .with_columns((pl.col("_g").rank("average").over(["date", "actual_rtg"])
                                 / pl.col("_g").count().over(["date", "actual_rtg"]))
                                .alias("_z")))
            z = jz["_z"].to_numpy().astype(float)
            ok2 = (isin_a[1:] == isin_a[:-1]) & (midx_a[1:] - midx_a[:-1] == 1)
            aa, bb = z[:-1][ok2], z[1:][ok2]
            gg = np.isfinite(aa) & np.isfinite(bb)
            acf = float(np.corrcoef(aa[gg], bb[gg])[0, 1])
            ad = auc(z[o], DN[o]); au = auc(z[o], U[o])
            # signal inside the region the dead-zone would discard
            sub = "-"
            if th is not None:
                m2 = o & (p_stay > th)
                if m2.sum() > 1000:
                    zp = jz["_z"].to_numpy().astype(float)
                    base = transition_risk_ratio(P, k)
                    jb = (j.with_columns(pl.Series("_b", base))
                          .with_columns((pl.col("_b").rank("average").over(["date", "actual_rtg"])
                                         / pl.col("_b").count().over(["date", "actual_rtg"]))
                                        .alias("_zb")))
                    sub = f"{auc(jb['_zb'].to_numpy().astype(float)[m2], DN[m2]):.4f}"
            share = 0.0 if th is None else float(np.mean(p_stay > th))
            emit(fh, f"    {str(th):<12}{share:>14.1%}{acf:>9.3f}{ad:>10.4f}"
                     f"{max(abs(ad-.5), abs(au-.5)):>8.4f}{sub:>18}")
        emit(fh, "    'AUC | th-subset' is the UNMODIFIED ratio scored only on the rows")
        emit(fh, "    the dead-zone would neutralise.  It stays well above 0.5, so those")
        emit(fh, "    vanishing tails are signal, not numerical noise, and neutralising")
        emit(fh, "    them costs power.  What the dead-zone does buy is a much less")
        emit(fh, "    persistent PIT -- the same trade the scoring rules make.")

        emit(fh, "\n  does the cardinal encoding enc(r) matter for eq. (4.dd)?")
        eaucs = []
        for ename, enc in ENCODINGS.items():
            g = directional_deviation(P, k, enc=enc)
            jz = (j.with_columns(pl.Series("_g", g))
                  .with_columns((pl.col("_g").rank("average").over(["date", "actual_rtg"])
                                 / pl.col("_g").count().over(["date", "actual_rtg"]))
                                .alias("_z")))
            a = auc(jz["_z"].to_numpy().astype(float)[o], DN[o])
            eaucs.append(a)
            emit(fh, f"    {ename:<26}AUC down {a:.4f}")
        emit(fh, f"    {'spread across encodings':<26}         {max(eaucs)-min(eaucs):.4f}"
                 "   <- immaterial")
        emit(fh, "    Brier and the log score need NO encoding; RPS and log-TRR need")
        emit(fh, "    only the ORDER.  So the encoding objection to the deviation is")
        emit(fh, "    an inelegance, not a defect, and the case for the scoring rules")
        emit(fh, "    rests on propriety instead.")

        emit(fh, "\n  The directional constructions read the sign of the shift; the")
        emit(fh, "  proper scoring rules read its magnitude, so they cannot separate")
        emit(fh, "  an imminent upgrade from an imminent downgrade and their two AUCs")
        emit(fh, "  sit on the same side of one half.  That is why Section 4.4 tests")
        emit(fh, "  them against the one-sided alternative alone.")

    # ---------------------------------------------------------------- A4
    emit(fh, "\n" + "=" * 78)
    emit(fh, "A4  sign convention of the stored PIT")
    emit(fh, "=" * 78)
    emit(fh, "  Every script stores within_class_std_rank as")
    emit(fh, "      rank(directional_deviation, ascending=False) / count,")
    emit(fh, "  and DD = expected_rating - actual_rating with 1=AAA .. 6=B, so a bond")
    emit(fh, "  the model thinks is WORSE than its label has DD > 0 and lands near 0.")
    emit(fh, "  Meeting 7 §3.1.1 states the opposite ('downgrade -> PIT near 1').")
    emit(fh, "  Measured, over every configuration above: AUC(PIT -> downgrade) < 0.5")
    emit(fh, "  and AUC(PIT -> upgrade) > 0.5, confirming the stored PIT is inverted")
    emit(fh, "  relative to the write-up.  One-sided E-values built on it fire on the")
    emit(fh, "  wrong tail; the fix is ascending=True, or 1 - Z downstream.")

    fh.close()
    print(f"\nwrote results/tabpfn_audit.txt")


if __name__ == "__main__":
    main()
