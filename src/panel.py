"""Bond panel loading and preparation (thesis Chapter 2 conventions).

- Aggregation to (date, company, class) level by mean, as in the legacy
  pipeline (multiple bonds of one issuer are averaged).
- Mean-correction within each (date, class) cell (thesis Section 2.4.2 /
  Lemma 3.26): x_mc = x - class cross-sectional mean at that date.
- Lagged partitioning covariate: the bond's value in the *preceding* month,
  as Section 3.1.3 requires; bonds without a preceding observation are not in
  the cohort ("monitored from its second observation onwards").
"""

import numpy as np
import polars as pl

RATING_CLASSES = ["AAA", "AA", "A", "BBB", "BB", "B"]

# Official SIC divisions.  Note these are NOT a leading-digit split:
# Manufacturing spans 2000-3999 and Services 7000-8999, while Mining/
# Construction and Wholesale/Retail each split a single leading digit.
# Codes 9997/9999 are placeholders and map to None (those bonds are dropped).
SIC_DIVISIONS = [
    (100, 999, "Agriculture"), (1000, 1499, "Mining"),
    (1500, 1799, "Construction"), (2000, 3999, "Manufacturing"),
    (4000, 4999, "TransportUtilities"), (5000, 5199, "Wholesale"),
    (5200, 5999, "Retail"), (6000, 6799, "FinanceInsuranceRE"),
    (7000, 8999, "Services"), (9100, 9729, "PublicAdmin"),
]


def _sic_division_expr():
    e = pl.when(pl.col("sic").is_between(SIC_DIVISIONS[0][0], SIC_DIVISIONS[0][1])) \
          .then(pl.lit(SIC_DIVISIONS[0][2]))
    for lo, hi, name in SIC_DIVISIONS[1:]:
        e = e.when(pl.col("sic").is_between(lo, hi)).then(pl.lit(name))
    return e.otherwise(None).alias("sector")


def load_panel(csv_path: str, target: str = "yield",
               partition_covs: tuple = ("duration", "spread"),
               rating_col: str = "rtg") -> pl.DataFrame:
    """Load and aggregate the panel.  ``rating_col`` = "rtg" groups by letter
    class; "nrtg" groups by integer notch (labels "1".."16"), so notch-level
    runs get their own mean-correction cell (Section 3.6.4)."""
    cols = ["dates", "isin", "rtg", "nrtg", "sic", target, *partition_covs]
    lf = (
        pl.scan_csv(csv_path, infer_schema_length=10000, ignore_errors=True)
        .select(cols)
        .filter(pl.col("isin").is_not_null() &
                pl.col(target).is_not_null() & pl.col(target).is_not_nan())
    )
    # SIC is static per bond, hence F_0-measurable: no lagging is needed and
    # the sector label can be used directly, either as a partition (design a)
    # or as the group restricting the permutations (design b).
    lf = lf.with_columns(
        # NaN must become null before casting: a strict cast is evaluated on
        # every row regardless of any guarding when/then
        pl.when(pl.col("sic").is_nan()).then(None).otherwise(pl.col("sic"))
          .cast(pl.Int64, strict=False).alias("sic")
    ).with_columns(_sic_division_expr())
    if rating_col == "rtg":
        lf = lf.filter(pl.col("rtg").is_in(RATING_CLASSES))
    else:
        lf = lf.filter(pl.col("nrtg").is_not_null() &
                       pl.col("nrtg").is_not_nan()).with_columns(
            pl.col("nrtg").cast(pl.Int32).cast(pl.String).alias("rtg"))
    # NaN is not null in polars, and mean() propagates it: a single NaN in a
    # (date, class) cell would otherwise make the whole cell's mean-correction
    # NaN and silently delete that class-date from every downstream statistic.
    # The four covariates of Sections 3.6-3.7 carry no NaNs once the target
    # filter above has run, so this changes none of those results; the sparser
    # covariates screened in scripts/screen_pairs.py carry a great many.
    lf = lf.with_columns([
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in [target, *partition_covs]
    ])
    df = (
        lf.group_by(["dates", "isin", "rtg"])
        .agg([pl.col(c).mean() for c in [target, *partition_covs]] +
             [pl.col("nrtg").median().alias("nrtg"),
              pl.col("sector").first().alias("sector")])
        .rename({"isin": "compsym"})   # downstream code uses "compsym" as the unit id
        .collect()
    )
    # mean-correction within (date, class)
    mc_exprs = []
    for c in [target, *partition_covs]:
        mc_exprs.append((pl.col(c) - pl.col(c).mean().over(["dates", "rtg"])).alias(f"{c}_mc"))
    df = df.with_columns(mc_exprs)
    # lagged (previous-month) covariates per company.  The target is lagged
    # too: Section 3.8.2 conditions the innovation on the bond's own previous
    # value, and both maps must be built from F_{t-1}-measurable columns.
    df = df.sort(["compsym", "dates"])
    month_idx = (pl.col("dates") // 100) * 12 + (pl.col("dates") % 100)
    df = df.with_columns(month_idx.alias("_midx"))
    for c in (target, *partition_covs):
        df = df.with_columns([
            pl.col(f"{c}_mc").shift(1).over("compsym").alias(f"{c}_mc_lag"),
            pl.col("_midx").shift(1).over("compsym").alias(f"_midx_prev_{c}"),
        ])
        # a lag is only valid if the previous row is exactly one month back
        df = df.with_columns(
            pl.when(pl.col(f"_midx_prev_{c}") == pl.col("_midx") - 1)
            .then(pl.col(f"{c}_mc_lag")).otherwise(None).alias(f"{c}_mc_lag")
        ).drop(f"_midx_prev_{c}")
    return df


def iter_class_dates(df: pl.DataFrame, rating: str, target: str, partition_cov: str,
                     window_months: int = 12, min_current: int = 4):
    """Yield, for each evaluable date t of a class, the ingredients of q_t.

    Yields dicts with: date, x (mean-corrected target), y_lag (lagged
    partitioning covariate), bond_ids, window_x_by_bond, pooled_window_x,
    window_month_list.
    """
    tgt = f"{target}_mc"
    xlag = f"{target}_mc_lag"
    ylag = f"{partition_cov}_mc_lag"
    ycur = f"{partition_cov}_mc"
    sub = df.filter(pl.col("rtg") == rating).select(
        ["dates", "compsym", tgt, xlag, ylag, ycur, "sector", "_midx"]
    ).drop_nulls(subset=[tgt])
    by_month = {k[0]: g for k, g in sub.partition_by("_midx", as_dict=True).items()}
    months = sorted(by_month)
    for m in months:
        window = [w for w in range(m - window_months, m) if w in by_month]
        if len(window) < window_months // 2:
            continue                                       # warm-up: not monitored
        date = int(by_month[m].get_column("dates")[0])
        cur = by_month[m].drop_nulls(subset=[ylag])
        if len(cur) < min_current:
            yield {"date": date, "degenerate": True}       # monitored, declines to bet
            continue
        window_x_by_bond: dict = {}
        # the two conditioning variables of Section 3.8.2, carried over the
        # window as well as the current date: month w's entry holds the value
        # observed at w-1, so both are strictly F_{t-1}-measurable.
        window_ylag_by_bond: dict = {}
        window_xlag_by_bond: dict = {}
        pooled = []
        for w in window:
            g = by_month[w]
            ids = g.get_column("compsym").to_list()
            xs = g.get_column(tgt).to_numpy()
            xls = g.get_column(xlag).to_numpy()
            yls = g.get_column(ylag).to_numpy()
            pooled.append(xs)
            for i, x, xl, yl in zip(ids, xs, xls, yls):
                if np.isfinite(x):
                    window_x_by_bond.setdefault(i, {})[w] = float(x)
                    if xl is not None and np.isfinite(xl):
                        window_xlag_by_bond.setdefault(i, {})[w] = float(xl)
                    if yl is not None and np.isfinite(yl):
                        window_ylag_by_bond.setdefault(i, {})[w] = float(yl)
        yield {
            "date": date,
            "degenerate": False,
            "x": cur.get_column(tgt).to_numpy().astype(float),
            "y_lag": cur.get_column(ylag).to_numpy().astype(float),
            # the bond's own previous target value (may be NaN where the bond
            # was absent last month); the innovation target of Section 3.8.2
            # drops those positions, which is a decision made at t-1.
            "x_lag": cur.get_column(xlag).to_numpy().astype(float),
            # contemporaneous Y, used ONLY to reproduce the legacy candidate
            "y_cur": cur.get_column(ycur).to_numpy().astype(float),
            # static SIC division label (may contain None for unmapped codes)
            "sector": np.array(cur.get_column("sector").to_list(), dtype=object),
            "bond_ids": np.array(cur.get_column("compsym").to_list()),
            "window_x_by_bond": window_x_by_bond,
            "window_ylag_by_bond": window_ylag_by_bond,
            "window_xlag_by_bond": window_xlag_by_bond,
            "pooled_window_x": np.concatenate(pooled),
            "window_month_list": window,
        }
