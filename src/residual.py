"""The innovation target of thesis Section 3.8.2 — residualising the level.

Motivation (Section 3.8.1).  Under the null (3.1) the conditioning is on the
past, and the past identifies which bond is which.  A characteristic that is
persistent over time and varied across bonds therefore makes (3.1) false for
structural reasons: on the panel the median within-class rank correlation
between the lagged partitioning covariate and the current target reaches 0.92,
so the level test rejects at essentially every date and learns nothing about
credit.  The informative hypothesis concerns the *innovation*.

The construction.  Fix an evaluation date t and its window W = {t-K, .., t-1}.
Let Z_{i,s} be a conditioning variable known at s-1 — either the lagged
partitioning covariate Y_{i,s-1} (``mode="y"``, the map of Section 3.8.2 as
written) or the bond's own previous target X^mc_{i,s-1} (``mode="self"``, the
cross-sectional analogue of the AR(1) innovation of Section 4.4.2).  Fit the
location-scale map

    mu(z)    = a + b z
    sigma(z) = exp(c + d z)                      (clipped to [lo, hi])

by least squares on the pooled pairs {(Z_{i,s}, X^mc_{i,s}) : s in W}, and set

    X^res_{i,s} := (X^mc_{i,s} - mu(Z_{i,s})) / sigma(Z_{i,s}).

The same fitted map is applied to every month of the window *and* to the
current cross-section, so the two are on one scale and the candidate's
marginals mean what they say.  The map depends on data up to t-1 only, hence
so does every ingredient of q_t, and Lemma 3.3 applies unchanged.

One departure from Section 3.8.2 as written.  That section says no part of the
machinery requires modification, and for validity that is exactly right.  For
power it is not.  If the map and the candidate are fitted on the *same* window,
the least-squares fit leaves that window's residuals homogeneous across blocks
by construction; q_t is then nearly exchangeable by Lemma 3.14, E_t is pinned
at one, and a break in the relation --- the alternative the whole change of
target was introduced for --- produces a strongly ordered current residual that
the candidate has no block structure to score.  ``split_window`` repairs this:
the OLDER months of the window fit the map (the reference period) and the RECENT
ones fit the candidate, both residualised through the reference map.  Under a
stable relation the recent residuals are still homogeneous and nothing accrues;
once the relation moves, the recent months tilt, the candidate learns the tilt,
and the current month is scored against it.  Both halves end strictly before t,
so the measurability argument is untouched.  ``reference_share=0`` reproduces
the literal reading, and the synthetic regimes of
``scripts/run_residual_synthetic.py`` measure what it costs.

What changes is the null, not the guarantee.  X^res_{i,t} is a bond-specific
transform of X_{i,t}, so exchangeability of the level does not imply
exchangeability of the innovation and vice versa: the hypothesis under test
becomes

    H_0^res : X^res(t) is exchangeable given F_{t-1}, for every t,

i.e. "the historical relation between Z and X still describes this month's
cross-section".  A rejection now says the ordering has *changed*, which is the
statement a monitoring system was wanted for.
"""

from dataclasses import dataclass

import numpy as np

# -E[log|N(0,1)|].  Regressing log|r| on z estimates log sigma(z) shifted down
# by this constant when the errors are Gaussian; adding it back makes the fit
# unbiased for the log scale.
LOG_HALF_NORMAL_BIAS = 0.6351814227307391

# sigma(z) is clipped to this multiple of the homoskedastic residual scale, so
# that a slope fitted on twelve months cannot send the denominator to zero at
# the edges of the covariate range.
SCALE_CLIP = (0.2, 5.0)

MODES = ("y", "self")
SCALE_MODELS = ("none", "const", "affine")
MAP_FORMS = ("affine", "binned")
CROSS_SCALES = ("none", "mad")

# number of quantile bins used by the "binned" map form
N_BINS = 8


@dataclass(frozen=True)
class LocScaleMap:
    """m_hat of Section 3.8.2: an affine location and a log-affine scale."""

    a: float
    b: float
    c: float
    d: float
    s0: float          # homoskedastic residual scale, used for the clip
    n_obs: int
    fallback: bool     # True when the window gave too little to fit on

    def loc(self, z: np.ndarray) -> np.ndarray:
        return self.a + self.b * np.asarray(z, dtype=float)

    def scale(self, z: np.ndarray) -> np.ndarray:
        s = np.exp(self.c + self.d * np.asarray(z, dtype=float))
        lo, hi = SCALE_CLIP[0] * self.s0, SCALE_CLIP[1] * self.s0
        return np.clip(s, lo, hi)

    def residual(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.loc(z)) / self.scale(z)


def fit_loc_scale_map(z: np.ndarray, x: np.ndarray, scale_model: str = "affine",
                      min_obs: int = 30) -> LocScaleMap:
    """Fit m_hat on the pooled window pairs (z, x).

    With too few pairs, or a degenerate conditioning variable, the map degrades
    gracefully to a constant one.  That still leaves it F_{t-1}-measurable, so
    the e-value stays valid; it only means the innovation is not being
    conditioned on anything that date, which the caller records.
    """
    if scale_model not in SCALE_MODELS:
        raise ValueError(f"scale_model must be one of {SCALE_MODELS}")
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(z) & np.isfinite(x)
    z, x = z[ok], x[ok]
    n = z.size
    if n < min_obs:
        s0 = float(np.std(x)) if n > 1 else 1.0
        s0 = s0 if s0 > 0 else 1.0
        a = float(np.mean(x)) if n else 0.0
        return LocScaleMap(a, 0.0, np.log(s0), 0.0, s0, n, True)

    zbar, xbar = float(np.mean(z)), float(np.mean(x))
    szz = float(np.sum((z - zbar) ** 2))
    b = float(np.sum((z - zbar) * (x - xbar)) / szz) if szz > 0 else 0.0
    a = xbar - b * zbar
    r = x - (a + b * z)
    s0 = float(np.std(r))
    if not np.isfinite(s0) or s0 <= 0:
        s0 = 1.0

    if scale_model == "none":
        return LocScaleMap(a, b, 0.0, 0.0, 1.0, n, False)
    if scale_model == "const" or szz <= 0:
        return LocScaleMap(a, b, np.log(s0), 0.0, s0, n, False)

    # log|r| = log sigma(z) - LOG_HALF_NORMAL_BIAS + noise
    lr = np.log(np.abs(r) + 1e-12 * s0)
    lbar = float(np.mean(lr))
    d = float(np.sum((z - zbar) * (lr - lbar)) / szz)
    c = lbar - d * zbar + LOG_HALF_NORMAL_BIAS
    if not np.isfinite(c) or not np.isfinite(d):
        return LocScaleMap(a, b, np.log(s0), 0.0, s0, n, False)
    return LocScaleMap(a, b, c, d, s0, n, False)


@dataclass(frozen=True)
class BinnedMap:
    """A nonparametric location-scale map: bin z at the reference window's own
    quantiles, take each bin's mean and standard deviation, and interpolate
    linearly between bin centres.

    Why it exists.  The affine map removes only the *linear* part of the
    relation between Z and X.  If the true relation is curved, a stable panel
    still leaves the quantile blocks of the candidate with systematically
    non-zero residual means, and the monitor accrues evidence month after month
    for a reason that has nothing to do with anything changing --- the same
    trap, one level down, that Section 3.8.1 describes for the level.  Swapping
    this map in and re-running answers "is the rejection curvature?" directly.
    Like the affine map it is a deterministic function of the reference window,
    so validity is untouched.
    """

    edges: np.ndarray          # bin centres in z, ascending
    means: np.ndarray          # location at each centre
    sds: np.ndarray            # scale at each centre
    s0: float
    n_obs: int
    fallback: bool

    # reported alongside the affine map so the two runs stay comparable
    @property
    def b(self) -> float:
        """Overall slope, for the diagnostics: rise over run across the bins."""
        if self.edges.size < 2:
            return 0.0
        span = float(self.edges[-1] - self.edges[0])
        return float((self.means[-1] - self.means[0]) / span) if span else 0.0

    @property
    def d(self) -> float:
        if self.edges.size < 2:
            return 0.0
        span = float(self.edges[-1] - self.edges[0])
        return (float(np.log(self.sds[-1] / self.sds[0]) / span)
                if span and self.sds[0] > 0 and self.sds[-1] > 0 else 0.0)

    def loc(self, z: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(z, dtype=float), self.edges, self.means)

    def scale(self, z: np.ndarray) -> np.ndarray:
        s = np.interp(np.asarray(z, dtype=float), self.edges, self.sds)
        return np.clip(s, SCALE_CLIP[0] * self.s0, SCALE_CLIP[1] * self.s0)

    def residual(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.loc(z)) / self.scale(z)


def fit_binned_map(z: np.ndarray, x: np.ndarray, n_bins: int = N_BINS,
                   min_obs: int = 30, scale_model: str = "affine") -> BinnedMap:
    """Fit the binned map on the pooled reference-window pairs."""
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(z) & np.isfinite(x)
    z, x = z[ok], x[ok]
    n = z.size
    s0 = float(np.std(x)) if n > 1 else 1.0
    s0 = s0 if s0 > 0 else 1.0
    if n < max(min_obs, 3 * n_bins):
        return BinnedMap(np.array([0.0]), np.array([float(np.mean(x)) if n else 0.0]),
                         np.array([s0]), s0, n, True)
    # equal-count bins at the reference window's own quantiles
    qs = np.quantile(z, np.linspace(0.0, 1.0, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(qs, z, side="right") - 1, 0, n_bins - 1)
    centres, means, sds = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < 3:
            continue
        centres.append(float(np.median(z[sel])))
        means.append(float(np.mean(x[sel])))
        sds.append(float(np.std(x[sel])))
    if len(centres) < 2:
        return BinnedMap(np.array([0.0]), np.array([float(np.mean(x))]),
                         np.array([s0]), s0, n, True)
    centres = np.asarray(centres)
    order = np.argsort(centres)
    means = np.asarray(means)[order]
    sds = np.asarray(sds)[order]
    if scale_model in ("none", "const"):
        sds = np.full(sds.shape, 1.0 if scale_model == "none" else s0)
    sds = np.maximum(sds, 1e-9)
    return BinnedMap(centres[order], means, sds,
                     1.0 if scale_model == "none" else s0, n, False)


def cross_section_scale(v: np.ndarray, floor_frac: float = 0.1) -> float:
    """A robust scale of one cross-section, used to standardise it.

    Why this is allowed.  Every other ingredient of the construction has to be
    F_{t-1}-measurable, and this one is not: it is computed from X(t) itself.
    It is nevertheless free, because it is a SYMMETRIC function of the
    cross-section.  If U is exchangeable given F_{t-1} and s is permutation
    invariant, then s(U_sigma) = s(U) for every sigma, so (U/s(U))_sigma =
    U_sigma/s(U_sigma) and the standardised vector is exchangeable too.  The
    null is unchanged and Lemma 3.3 still applies; what changes is that the
    candidate no longer has to model the overall dispersion of a month, only
    the shape of the cross-section within it.

    Why it matters here.  Measured on the panel, the residual's cross-sectional
    standard deviation is close to one in calm months by construction and three
    to seven times that in recession months.  A Gamma marginal fitted on the
    window cannot anticipate a scale break of that size, so the observed
    arrangement scores badly for a reason that has nothing to do with which bond
    is which -- which is the mechanical form of the "the candidate becomes
    misspecified in crises" complaint of Sections 3.6 and 3.8.

    The median absolute deviation is used rather than the standard deviation so
    that a handful of extreme bonds cannot set the scale for the cohort.
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 4:
        return 1.0
    mad = float(np.median(np.abs(v - np.median(v)))) * 1.4826
    sd = float(np.std(v))
    # a degenerate cohort (many tied values) would otherwise divide by ~0
    floor = max(floor_frac * sd, 1e-8)
    return max(mad, floor) if np.isfinite(mad) else max(sd, 1e-8)


def _window_pairs(item: dict, z_by_bond: dict, months_keep=None) -> tuple:
    """Pooled (z, x) pairs over the window, month w contributing Z_{i,w-1}."""
    keep = None if months_keep is None else set(months_keep)
    zs, xs = [], []
    for bond, months in item["window_x_by_bond"].items():
        zb = z_by_bond.get(bond)
        if not zb:
            continue
        for w, xv in months.items():
            if keep is not None and w not in keep:
                continue
            zv = zb.get(w)
            if zv is not None:
                zs.append(zv)
                xs.append(xv)
    return np.asarray(zs, dtype=float), np.asarray(xs, dtype=float)


def split_window(window_months: list, reference_share: float = 0.5) -> tuple:
    """Split the window into a reference half and a candidate half.

    Why this exists.  Taken literally, "replace the target and change nothing
    else" has no power against the alternative it was introduced for.  If the
    map is fitted on the same window the candidate's marginals are fitted on,
    then under a stable relation the window residuals are homogeneous across
    blocks *by construction of the least-squares fit* — so q_t is very nearly
    exchangeable (Lemma 3.14) and E_t is pinned at one.  When the relation then
    breaks, the residual of the current month is strongly block-ordered but the
    candidate has no block structure with which to score it, and the break goes
    undetected.  The synthetic ``relation_break`` regime measures exactly this.

    Splitting the window repairs it without leaving the framework.  The map is
    estimated on the OLDER months (the reference period) and the candidate's
    blocks, marginals and copula parameters on the RECENT ones, all residualised
    through that same reference map.  Under a stable relation the recent
    residuals are still homogeneous and E_t stays at one; once the relation
    moves, the recent residuals tilt, the candidate learns the tilt, and the
    current month is scored against it.  Both halves end strictly before t, so
    q_t remains F_{t-1}-measurable and Lemma 3.3 is untouched.

    A rejection then reads: "the cross-section has been departing from the
    reference relation, and this month continues it."
    """
    months = sorted(window_months)
    if not 0.0 < reference_share < 1.0:
        return months, months                      # no split: the literal 3.8.2
    k = max(1, int(round(reference_share * len(months))))
    k = min(k, len(months) - 1)
    return months[:k], months[k:]


def residualize_item(item: dict, mode: str = "y", scale_model: str = "affine",
                     min_obs: int = 30, min_current: int = 4,
                     min_hist_obs: int = 10,
                     reference_share: float = 0.0,
                     map_form: str = "affine",
                     cross_scale: str = "none") -> dict:
    """Return a copy of one cohort item with the innovation as its target.

    The candidate is still partitioned on the *lagged partitioning covariate*
    (``item["y_lag"]``), unchanged, whichever mode is used: Section 3.8.2 asks
    only that the target be replaced, and the machinery applies as is.

    ``reference_share`` in (0, 1) splits the window as ``split_window``
    describes: that share of the oldest months estimates the map, the rest
    supplies the candidate.  ``0.0`` (the default) is Section 3.8.2 read
    literally, with one window doing both jobs.

    ``cross_scale="mad"`` additionally divides every cross-section --- each
    window month and the current one --- by its own robust scale, which
    ``cross_section_scale`` explains is free of charge.

    Positions whose conditioning variable is missing are dropped from the
    cohort.  Whether Z_{i,t} exists is settled at t-1, so the composition of
    the cohort remains F_{t-1}-measurable as Section 3.1.2 assumes.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if item is None or item.get("degenerate"):
        return item
    ref_months, cand_months = split_window(item["window_month_list"],
                                           reference_share)

    if mode == "y":
        z_cur = np.asarray(item["y_lag"], dtype=float)
        z_by_bond = item["window_ylag_by_bond"]
    else:
        z_cur = np.asarray(item["x_lag"], dtype=float)
        z_by_bond = item["window_xlag_by_bond"]

    zw, xw = _window_pairs(item, z_by_bond, ref_months)
    if xw.size < min_hist_obs:
        return {"date": item["date"], "degenerate": True}
    if map_form not in MAP_FORMS:
        raise ValueError(f"map_form must be one of {MAP_FORMS}")
    m_hat = (fit_loc_scale_map(zw, xw, scale_model=scale_model, min_obs=min_obs)
             if map_form == "affine"
             else fit_binned_map(zw, xw, min_obs=min_obs,
                                 scale_model=scale_model))

    keep = np.flatnonzero(np.isfinite(z_cur) & np.isfinite(item["x"]))
    if keep.size < min_current:
        return {"date": item["date"], "degenerate": True}

    if cross_scale not in CROSS_SCALES:
        raise ValueError(f"cross_scale must be one of {CROSS_SCALES}")
    out = dict(item)
    out["x"] = m_hat.residual(item["x"][keep], z_cur[keep])
    if cross_scale == "mad":
        out["cross_scale_t"] = cross_section_scale(out["x"])
        out["x"] = out["x"] / out["cross_scale_t"]
    for key in ("y_lag", "x_lag", "y_cur", "sector", "bond_ids"):
        if key in item:
            out[key] = np.asarray(item[key])[keep]

    res_by_bond: dict = {}
    pooled = []
    cand_set = set(cand_months)
    for bond, months in item["window_x_by_bond"].items():
        zb = z_by_bond.get(bond)
        if not zb:
            continue
        inner = {}
        for w, xv in months.items():
            if w not in cand_set:
                continue
            zv = zb.get(w)
            if zv is None:
                continue
            rv = float(m_hat.residual(np.array([xv]), np.array([zv]))[0])
            if np.isfinite(rv):
                inner[w] = rv
                pooled.append(rv)
        if inner:
            res_by_bond[bond] = inner
    if cross_scale == "mad":
        # each window month is standardised by its OWN cross-section, so the
        # marginals the candidate fits describe shape rather than level of
        # dispersion, on the same footing as the current month above
        by_month: dict = {}
        for bond, months in res_by_bond.items():
            for w, v in months.items():
                by_month.setdefault(w, []).append(v)
        scales = {w: cross_section_scale(np.asarray(v))
                  for w, v in by_month.items()}
        pooled = []
        for bond, months in res_by_bond.items():
            for w in list(months):
                months[w] = months[w] / scales[w]
                pooled.append(months[w])
        out["cross_scales_window"] = scales
    if len(pooled) < min_hist_obs:
        return {"date": item["date"], "degenerate": True}

    out["window_x_by_bond"] = res_by_bond
    out["pooled_window_x"] = np.asarray(pooled, dtype=float)
    out["window_month_list"] = cand_months
    out["map"] = m_hat
    out["n_dropped"] = int(item["x"].size - keep.size)
    out["ref_months"] = ref_months
    return out


def target_label(mode: str) -> str:
    return {"y": "residual on lagged Y",
            "self": "residual on own lagged target"}[mode]
