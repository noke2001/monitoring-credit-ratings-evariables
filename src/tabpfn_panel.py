"""Reading the saved TabPFN fits — thesis Chapter 4, Section 4.3.

TabPFN is refitted at every date, so a sweep is hours of GPU time and the
per-date outputs are appended to CSVs under ``tabpfnfit/``.  Two defects in
those files are corrected here rather than in nine downstream scripts.

**The sign of the stored PIT.**  Every driver wrote

    within_class_std_rank = rank(directional_deviation, ascending=False) / n

with ``directional_deviation = expected_rating - actual_rating`` on the scale
1 = AAA .. 6 = B.  A bond the model rates *worse* than its label therefore has
DD > 0 and landed near **0**, while the write-up and the one-sided E-values of
Section 4.4 require it near **1**.  Measured over every configuration, the
stored column gives AUC(PIT -> downgrade) in [0.324, 0.382], uniformly below
one half.  ``load_fit`` recomputes it ascending; the drivers have been fixed at
source, so files produced from now on already agree.

**The append-header defect.**  Each date writes one ``prob_*`` column per class
present in *that date's* training context, but ``to_csv(mode='a')`` fixes the
header at the first date's set.  Where a later date carries a class the first
did not, the row has an extra field and every probability from that class
onward is shifted.  ``scan_integrity`` measures it; the leading columns are
written before the block and are unaffected, so scores and ranks survive even
where the probability vector does not.  Six ``*_wprobs.csv`` files are affected,
worst ``tabpfn_data_nrtg_12_wprobs.csv`` at 8.5% argmax agreement.  The
``delta_fit``/``horizon_fit`` drivers reindex to a fixed column set and are
clean.
"""

from pathlib import Path

import numpy as np
import polars as pl

RATING_MAP = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6}

#: file -> (label, driver that wrote it, whether the target is the FORWARD rating)
PROVENANCE = {
    "tabpfn_data_rtg_12_wprobs.csv":
        ("level target, K=12", "tabpfn/run_level.py", False),
    "tabpfn_data_rtg_12_onlytrans.csv":
        ("level target, transition-only context", "tabpfn/run_level_onlytrans.py", False),
    "tabpfn_data_rtg_12_fresh_fit_minbonds_10.csv":
        ("level target, fresh-rating context", "tabpfn/run_level_freshfit.py", False),
    "tabpfn_data_rtg_24_fresh_fit_minbonds_10.csv":
        ("level target, fresh-rating, K=24", "tabpfn/run_level_freshfit.py", False),
    "tabpfn_delta_fit_rtg_12m.csv":
        ("delta target, K=12", "tabpfn/run_delta.py", False),
    "tabpfn_delta_fit_dom_50_rtg_12m.csv":
        ("delta target, K=12, dom cap 50%", "tabpfn/run_delta_dom.py", False),
    "tabpfn_delta_fit_dom_50_rtg_24m.csv":
        ("delta target, K=24, dom cap 50%", "tabpfn/run_delta_dom.py", False),
    "tabpfn_delta_fit_dom_40_rtg_24m_new.csv":
        ("delta target, K=24, dom 40% + fresh ratio", "tabpfn/run_delta_dom_new.py", False),
    "tabpfn_horizon_fit_results_10000_24_16.csv":
        ("FORWARD delta target, 10k context, 16 feats", "tabpfn/run_topdelta.py", True),
}


def scan_integrity(path) -> dict:
    """Field-count histogram and argmax agreement for one saved fit.

    ``argmax_agreement`` is the share of rows on which ``argmax(prob_*)``
    reproduces the stored predicted label.  Anything below 1.0 means the
    probability block is misaligned on those rows.
    """
    path = Path(path)
    widths: dict = {}
    with open(path) as fh:
        next(fh)
        for line in fh:
            k = line.count(",") + 1
            widths[k] = widths.get(k, 0) + 1
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
    return {"rows": d.height, "widths": widths, "ragged": len(widths) > 1,
            "argmax_agreement": agree,
            "probs_trustworthy": bool(np.isfinite(agree) and agree > 0.999)}


def load_fit(path, correct_sign: bool = True) -> pl.DataFrame:
    """Load one saved fit with ``date`` as YYYYMM and the PIT sign corrected.

    Adds ``pit`` — the within-(date, rating) ascending rank of the directional
    deviation, so a downgrade candidate sits near 1.  The original column is
    kept as ``pit_stored`` where it exists, so the two can be compared.
    """
    d = pl.read_csv(path, infer_schema_length=50000, ignore_errors=True,
                    truncate_ragged_lines=True)
    d = (d.with_columns(pl.col("date").str.slice(0, 4).cast(pl.Int64) * 100
                        + pl.col("date").str.slice(5, 2).cast(pl.Int64))
         .rename({"bond_id": "isin"}))
    if "within_class_std_rank" in d.columns:
        d = d.rename({"within_class_std_rank": "pit_stored"})
    if "directional_deviation" not in d.columns:
        # the forward-target driver stores neither DD nor the rank
        lnum = pl.col("actual_numeric") if "actual_numeric" in d.columns else None
        if lnum is None:
            return d
        d = d.with_columns((pl.col("expected_rtg") - lnum).alias("directional_deviation"))
    if not correct_sign:
        return d
    cohort = ["date", "actual_rtg"] if "actual_rtg" in d.columns else ["date"]
    return d.with_columns(
        (pl.col("directional_deviation").rank("average").over(cohort)
         / pl.col("directional_deviation").count().over(cohort)).alias("pit"))


def forward_labels(panel: pl.DataFrame, horizon: int = 12,
                   rating_col: str = "lnum"):
    """Per row: does the rating move within the next ``horizon`` months?

    Returns ``(downgrade, upgrade, observed, rating_at_end, previous_rating)``
    aligned with ``panel``, which must be sorted by ``isin, dates`` and carry a
    contiguous month index ``midx``.  ``observed`` is False where the bond has
    no observation inside the window, so those rows can be excluded rather than
    scored as "no transition".
    """
    isin = panel["isin"].to_numpy()
    mi = panel["midx"].to_numpy()
    g = panel[rating_col].to_numpy().astype(float)
    n = len(g)
    dn = np.zeros(n, bool); up = np.zeros(n, bool); obs = np.zeros(n, bool)
    fw = np.full(n, np.nan); prev = np.full(n, np.nan)
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
    return dn, up, obs, fw, prev


def load_rating_panel(csv_path) -> pl.DataFrame:
    """Letter-rating panel keyed by (isin, dates) with a contiguous month index."""
    return (pl.scan_csv(csv_path, infer_schema_length=20000, ignore_errors=True)
            .select(["dates", "isin", "rtg"])
            .filter(pl.col("isin").is_not_null()
                    & pl.col("rtg").is_in(list(RATING_MAP)))
            .collect().sort(["isin", "dates"])
            .with_columns((((pl.col("dates") // 100) * 12)
                           + (pl.col("dates") % 100)).alias("midx"),
                          pl.col("rtg").replace_strict(RATING_MAP,
                                                       return_dtype=pl.Int64)
                          .alias("lnum")))
