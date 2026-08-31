"""Univariate cohort PIT — thesis Chapter 4, Sections 4.1 and 4.2.

Chapter 3 asks whether a rating *class* has stopped being homogeneous.
Chapter 4 asks which *bond* has stopped belonging.  The object it monitors is

    Z_{i,t} = F_t( Gamma(x_{i,t}) ),   F_t = empirical CDF of the cohort   (4.1)

with the cohort I_{t,R} = { j : R_{j,t} = R } taken at the *same* date and the
*same* rating.  Section 4.2 is the special case Gamma = id, i.e. the PIT of a
single raw covariate, and is an exploration rather than a monitor: it fixes the
vocabulary and shows why a *learned* Gamma is needed.

Three properties of eq. (4.1) drive everything the section says, and each is
certified in ``tests/test_univariate.py``:

1.  The target is a member of its own cohort, so ``F_t`` is a self-including
    ECDF.  Its value lies on the lattice {1/n, 2/n, ..., 1}: under the null it
    is uniform *on that lattice*, not on [0,1].  E[Z] = (n+1)/(2n) > 1/2 and
    P(Z = 1) = 1/n > 0.  Both vanish as n grows but neither is zero, and an
    e-value construction that assumes a continuous PIT is invalid at the
    cohort sizes this panel actually has.  ``rank_pit(..., randomize=True)``
    returns the randomised version, which is exactly U([0,1]).

2.  Pooled over the cohort at a fixed date, the PITs are a permutation of
    {1/n, ..., 1} whatever the data.  The cross-sectional histogram is
    therefore flat as an algebraic identity and diagnoses nothing.  Only the
    *time series* of one bond's PIT carries information.

3.  Ranks are invariant to any strictly increasing transform applied to the
    whole cohort.  Mean-correction within (date, class) — Chapter 2's
    time-invariance device — is such a transform, so Chapter 4 may work with
    raw covariates and still be comparable to Chapter 3.  The same invariance
    is what makes ``mom6xrtg`` redundant at notch level (see
    ``REDUNDANT_AT_NOTCH_LEVEL``).

The hypothesis of eq. (4.2) is *conditional*: Z_{i,t} | F_{t-1} ~ U([0,1]) for
all t.  ``pit_persistence`` measures the autocorrelation that decides whether
that is tenable, and for every raw covariate in this panel it is not.
"""

import numpy as np
import polars as pl

# ``mom6xrtg`` is not a momentum variable: in this panel it equals
# ``spread * nrtg`` to machine precision on 100% of rows.  Within a notch
# cohort nrtg is constant, so mom6xrtg is a strictly increasing function of
# spread there and its rank PIT is *identical* to the spread PIT.  It carries
# independent information only at letter-class level, where the notch varies.
REDUNDANT_AT_NOTCH_LEVEL = {"mom6xrtg": "spread"}

DEFAULT_COVARIATES = ("coupon", "spread", "me", "yield",
                      "mom6xrtg", "sales", "spr_to_d2d")


def load_bond_panel(csv_path: str, covariates=DEFAULT_COVARIATES) -> pl.DataFrame:
    """Bond-level panel for Chapter 4.

    No aggregation and no mean-correction: the cohort rank is invariant to
    both an issuer-level average and a common location shift, and Chapter 4
    monitors a named security, so the ``isin`` must survive.  Adds ``midx``,
    a contiguous month index used to tell a real one-month lag from a gap.
    """
    cols = ["dates", "isin", "nrtg", "rtg", *covariates]
    df = (
        pl.scan_csv(csv_path, infer_schema_length=20000, ignore_errors=True)
        .select(sorted(set(cols), key=cols.index))
        .filter(pl.col("isin").is_not_null() &
                pl.col("nrtg").is_not_null() & pl.col("nrtg").is_not_nan())
        .with_columns(pl.col("nrtg").cast(pl.Int64))
        .with_columns((((pl.col("dates") // 100) * 12) +
                       (pl.col("dates") % 100)).alias("midx"))
        .collect()
        .sort(["isin", "dates"])
    )
    # NaN and null both mean "missing" here; polars ranks NaN as a value.
    return df.with_columns([
        pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in covariates if df.schema[c] == pl.Float64
    ])


def rank_pit(df: pl.DataFrame, covariate: str, cohort=("dates", "nrtg"),
             randomize: bool = False, seed: int = 0) -> pl.DataFrame:
    """Add ``Z_<covariate>``: the self-including cohort ECDF of eq. (4.1).

    ``randomize=True`` returns the randomised rank PIT

        Z = (r_below + U * (1 + n_tied)) / n,      U ~ U([0,1]) independent,

    which is exactly U([0,1]) under exchangeability of the cohort, for every n
    and in the presence of ties.  The unrandomised version is what the thesis
    figure plots and what eq. (4.3) defines; the randomised one is what any
    downstream e-value needs.
    """
    by = list(cohort)
    out = df.with_columns([
        pl.col(covariate).rank("max").over(by).alias("_rmax"),
        pl.col(covariate).rank("min").over(by).alias("_rmin"),
        pl.col(covariate).count().over(by).alias("_n"),
    ])
    if not randomize:
        out = out.with_columns(
            (pl.col("_rmax") / pl.col("_n")).alias(f"Z_{covariate}"))
    else:
        rng = np.random.default_rng(seed)
        u = rng.random(out.height)
        rmin = out["_rmin"].to_numpy().astype(float)
        rmax = out["_rmax"].to_numpy().astype(float)
        n = out["_n"].to_numpy().astype(float)
        with np.errstate(invalid="ignore"):
            z = (rmin - 1.0 + u * (rmax - rmin + 1.0)) / n
        out = out.with_columns(pl.Series(f"Z_{covariate}", z))
    return out.drop(["_rmax", "_rmin", "_n"])


def kde_cdf_pit(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """The *smoothed* CDF the legacy notebook plots when ``SMOOTH = True``.

    A Gaussian KDE with Scott bandwidth is fitted to the cohort and integrated
    to -inf.  This is not eq. (4.3): on heavy-tailed covariates (``me``,
    ``sales``) it departs from the ECDF by up to 0.36.  Provided so the
    difference can be measured rather than argued about.
    """
    from scipy.stats import gaussian_kde
    v = values[np.isfinite(values)]
    if v.size < 2 or np.std(v) < 1e-8:
        return np.full(targets.shape, np.nan)
    k = gaussian_kde(v)
    return np.array([k.integrate_box_1d(-np.inf, float(t)) if np.isfinite(t)
                     else np.nan for t in targets])


def pit_persistence(df: pl.DataFrame, covariate: str, lags=(1, 3, 6, 12)) -> dict:
    """Autocorrelation of one bond's PIT series, pooled over bonds.

    Only pairs separated by *exactly* ``lag`` contiguous months and belonging
    to the same ``isin`` enter the estimate; the panel is unbalanced and a
    naive shift would silently pair across gaps.

    The null of eq. (4.2) demands Z_t | F_{t-1} ~ U([0,1]).  A nonzero ACF(1)
    falsifies it directly: Z_{t-1} is F_{t-1}-measurable, so if it predicts
    Z_t the conditional law is not the uniform.
    """
    z = df[f"Z_{covariate}"].to_numpy().astype(float)
    isin = df["isin"].to_numpy()
    m = df["midx"].to_numpy()
    out = {}
    for lag in lags:
        ok = (isin[lag:] == isin[:-lag]) & (m[lag:] - m[:-lag] == lag)
        a, b = z[:-lag][ok], z[lag:][ok]
        g = np.isfinite(a) & np.isfinite(b)
        out[lag] = float(np.corrcoef(a[g], b[g])[0, 1]) if g.sum() > 30 else np.nan
    phi = out.get(1, np.nan)
    out["half_life"] = (float(np.log(0.5) / np.log(phi))
                        if np.isfinite(phi) and 0 < phi < 1 else np.nan)
    out["n_pairs"] = int(((isin[1:] == isin[:-1]) & (m[1:] - m[:-1] == 1)).sum())
    return out


def transition_labels(df: pl.DataFrame, horizon: int = 12):
    """Per row: does the notch move within the next ``horizon`` months?

    Returns ``(downgrade, upgrade, observed)`` boolean arrays aligned with
    ``df`` (which must be sorted by ``isin, dates``).  ``observed`` is False
    when the bond has no observation at all inside the window — at the panel
    edge, or after it matures — so those rows can be excluded rather than
    silently scored as "no transition".
    """
    isin = df["isin"].to_numpy()
    m = df["midx"].to_numpy()
    g = df["nrtg"].to_numpy().astype(float)
    n = len(g)
    dn = np.zeros(n, bool); up = np.zeros(n, bool); obs = np.zeros(n, bool)
    start = 0
    for i in range(1, n + 1):
        if i == n or isin[i] != isin[start]:
            gi, mi = g[start:i], m[start:i]
            for k in range(i - start):
                fut = (mi > mi[k]) & (mi <= mi[k] + horizon)
                if not fut.any():
                    continue
                obs[start + k] = True
                dn[start + k] = bool((gi[fut] > gi[k]).any())
                up[start + k] = bool((gi[fut] < gi[k]).any())
            start = i
    return dn, up, obs


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties (``coupon`` is discrete)."""
    from scipy.stats import rankdata
    m = np.isfinite(score)
    score, label = score[m], label[m].astype(bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(score)
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def rating_path(df: pl.DataFrame, isin: str) -> pl.DataFrame:
    """The compressed rating history of one bond: one row per transition."""
    b = df.filter(pl.col("isin") == isin).sort("dates")
    return b.filter(
        (pl.col("nrtg") != pl.col("nrtg").shift(1)) |
        pl.col("nrtg").shift(1).is_null()
    ).select(["dates", "nrtg", "rtg"])


def counterfactual_cohort(df: pl.DataFrame, isin: str, level: str = "nrtg"):
    """The 'pseudo' series of Figure 4.1: the previous rating, forward-filled.

    The legacy notebook labels this "what F_t would be had the bond not
    changed its rating".  It is more precisely the rating held immediately
    *before the most recent transition*: after a second transition the
    reference is the intermediate rating, not the original one, and before the
    first transition it is undefined (null, hence no line drawn).
    """
    b = df.filter(pl.col("isin") == isin).sort("dates")
    return b.with_columns(
        pl.when((pl.col(level) != pl.col(level).shift(1)) &
                pl.col(level).shift(1).is_not_null())
        .then(pl.col(level).shift(1)).otherwise(None)
        .fill_null(strategy="forward").alias(f"prev_{level}")
    )
