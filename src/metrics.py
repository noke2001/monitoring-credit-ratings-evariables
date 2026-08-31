"""
Core Metrics & Evaluation Functions for Credit Rating Sequential Testing
Master's Thesis: Monitoring Credit Ratings with E-Variables

  - compute_DD                : Directional Deviation vs. the monitored baseline
  - compute_TRR               : Transition Risk Ratio  P(down)/P(up)
  - compute_conformal_bounds  : genuine split-conformal one-sided rating bound
  - evaluate_lookforward_alarms : episode-level precision/recall/F1 with matched
                                  event definitions across horizons

Convention: `enc_y` is an ordinal rating index with 0 = AAA and larger = worse,
so a downgrade is an *increase* in enc_y.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "compute_DD",
    "compute_TRR",
    "compute_conformal_bounds",
    "evaluate_lookforward_alarms",
]


def _require(df: pd.DataFrame, cols, who: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{who} requires column(s) {missing}, which are absent. "
            "Silently substituting a default here would report a metric that "
            "was never computed."
        )


def compute_DD(
    df: pd.DataFrame,
    expected_rating_col: str = "expected_rtg",
    monitored_rating_col: str = "prev_enc_y",
) -> pd.DataFrame:
    """
    Directional Deviation relative to the monitored rating baseline R_{i,t-1}:

        DD_{i,t} = E[R_{i,t} | x_{i,t}] - R_{i,t-1}

    Positive DD = the model's fundamentals imply a *worse* rating than the one
    the agency currently has standing, i.e. downgrade pressure.
    """
    _require(df, [expected_rating_col, monitored_rating_col], "compute_DD")
    df = df.copy()
    df["directional_deviation"] = df[expected_rating_col] - df[monitored_rating_col]
    return df


def compute_TRR(
    df: pd.DataFrame,
    monitored_col: str = "prev_enc_y",
    prob_prefix: str = "prob_class_",
    classes: list = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Transition Risk Ratio, as documented:

        TRR_{i,t} = P(R > R_{i,t-1}) / P(R < R_{i,t-1})

    i.e. downgrade mass over *upgrade* mass, computed from the full predictive
    class distribution.

    The previous revision computed  P(down) / (1 - P(down))  instead. That is
    the downgrade *odds*: its denominator is P(stay) + P(up), not P(up). The two
    coincide only when P(stay) = 0, which never holds here -- P(stay) is the
    dominant mass in a monthly rating panel -- so the old column understated the
    ratio by roughly a factor of 1/P(up) and was not the quantity named in the
    thesis. `TRR_odds` is retained under its correct name for continuity.
    """
    classes = classes or ["AAA", "AA", "A", "BBB", "BB", "B"]
    prob_cols = [f"{prob_prefix}{c}" for c in classes]
    _require(df, prob_cols + [monitored_col], "compute_TRR")

    df = df.copy()
    P = df[prob_cols].to_numpy(dtype=float)
    k = df[monitored_col].to_numpy()
    valid = np.isfinite(k)
    k_int = np.where(valid, k, 0).astype(int)
    k_int = np.clip(k_int, 0, len(classes) - 1)

    grid = np.arange(len(classes))[None, :]
    p_down = np.where(grid > k_int[:, None], P, 0.0).sum(axis=1)
    p_up = np.where(grid < k_int[:, None], P, 0.0).sum(axis=1)

    df["p_downgrade"] = np.where(valid, p_down, np.nan)
    df["p_upgrade"] = np.where(valid, p_up, np.nan)
    df["TRR"] = np.where(valid, p_down / np.maximum(eps, p_up), np.nan)
    if "downgrade_tail_prob" in df.columns:
        d = df["downgrade_tail_prob"].to_numpy(dtype=float)
        df["TRR_odds"] = d / np.maximum(eps, 1.0 - d)
    return df


def compute_conformal_bounds(
    df: pd.DataFrame,
    prob_prefix: str = "prob_class_",
    classes: list = None,
    monitored_col: str = "prev_enc_y",
    actual_col: str = "enc_y",
    date_col: str = "dates",
    coverage_level: float = 0.95,
    calibration_end=None,
) -> pd.DataFrame:
    """
    Genuine split-conformal one-sided bound on the true rating tier.

    Procedure (split conformal, time-ordered to avoid look-ahead):
      1. Calibration fold = all rows with date <= `calibration_end`
         (default: the median date, so the earlier half calibrates the later).
      2. Nonconformity score  s_i = 1 - p_i(y_i)  on the calibration fold.
      3. qhat = the ceil((n+1) * coverage_level)/n empirical quantile of s.
      4. Prediction set at a test point: C(x) = { c : p(c) >= 1 - qhat }.
      5. `conformal_lower_bound` = min(C(x)) -- the best rating not excluded.

    Guarantee: under exchangeability of calibration and test scores,
    P(y in C(x)) >= coverage_level, with the finite-sample correction (n+1)/n.

    The previous revision accepted `coverage_level`, derived
    `threshold = 1 - coverage_level`, then never used it -- it hard-coded a
    P(downgrade) >= 0.50 rule. That rule is a median comparison with no coverage
    guarantee whatsoever, so calling it a "95% conformal lower bound" was wrong
    at any coverage level, and the argument had no effect on the output.
    """
    classes = classes or ["AAA", "AA", "A", "BBB", "BB", "B"]
    prob_cols = [f"{prob_prefix}{c}" for c in classes]
    _require(df, prob_cols + [monitored_col, actual_col, date_col], "compute_conformal_bounds")
    if not 0.0 < coverage_level < 1.0:
        raise ValueError(f"coverage_level must lie in (0,1), got {coverage_level}")

    df = df.copy()
    P = df[prob_cols].to_numpy(dtype=float)
    dates = pd.to_datetime(df[date_col])
    if calibration_end is None:
        calibration_end = dates.median()
    calibration_end = pd.Timestamp(calibration_end)

    y = df[actual_col].to_numpy()
    ok = np.isfinite(y)
    y_int = np.clip(np.where(ok, y, 0).astype(int), 0, len(classes) - 1)
    p_true = P[np.arange(len(df)), y_int]

    cal = (dates <= calibration_end).to_numpy() & ok
    n_cal = int(cal.sum())
    if n_cal < 20:
        raise ValueError(
            f"only {n_cal} calibration points before {calibration_end.date()}; "
            "split conformal needs a non-trivial calibration fold."
        )
    scores = 1.0 - p_true[cal]
    # Finite-sample conformal quantile: ceil((n+1)*coverage)/n
    level = min(1.0, np.ceil((n_cal + 1) * coverage_level) / n_cal)
    qhat = float(np.quantile(scores, level, method="higher"))

    in_set = P >= (1.0 - qhat)
    # min index in the set = best (lowest-index) rating not excluded
    any_in = in_set.any(axis=1)
    lower = np.where(any_in, in_set.argmax(axis=1), 0)

    df["conformal_qhat"] = qhat
    df["conformal_coverage_target"] = coverage_level
    df["conformal_is_calibration"] = cal
    df["conformal_lower_bound"] = lower
    df["passes_conformal_downgrade"] = lower > df[monitored_col].to_numpy()
    return df


def _months_between(later: np.ndarray, earlier: np.ndarray) -> np.ndarray:
    """Whole-month difference, avoiding the 'H * 31 days' approximation."""
    l = pd.DatetimeIndex(later)
    e = pd.DatetimeIndex(earlier)
    return (l.year - e.year) * 12 + (l.month - e.month)


def _km_prob_by(times, events, H):
    """
    Kaplan-Meier estimate of P(event by H) from right-censored observations.

    `times` is min(time-to-event, time-to-censoring) and `events` is 1 when the
    event was observed. Returns 1 - S(H). Censored observations still contribute
    their survived interval, which is the whole point: an alarm with ten months
    of follow-up and no transition is evidence about the first ten months, not
    a false positive.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    order = np.argsort(times, kind="stable")
    times, events = times[order], events[order]
    n_at_risk = len(times)
    S = 1.0
    for i, (t, e) in enumerate(zip(times, events)):
        if t > H:
            break
        if e == 1 and n_at_risk - i > 0:
            S *= 1.0 - 1.0 / (n_at_risk - i)
    return 1.0 - S


def evaluate_lookforward_alarms(
    df: pd.DataFrame,
    horizons=(12, 24),
    alarm_col: str = "is_alarm",
    id_col: str = "isin",
    date_col: str = "dates",
    event_col: str = "is_rating_change",
    censoring: str = "exclude",
) -> dict:
    """
    Episode-level evaluation of alarms against realized rating transitions.

    THE MONITORED EVENT IS ANY RATING CHANGE, in either direction. The thesis
    uses "downgrade" loosely: the objective is to flag a bond before the agency
    moves its rating at all, so `event_col` defaults to `is_rating_change`.
    Restricting to downgrades would score every anticipated upgrade as a false
    alarm.

    For each horizon H, precision and recall share one event definition:

        P_H = #{alarms a : a transition falls in (a, a+H]} / #alarms
        R_H = #{transitions c : an alarm falls in [c-H, c)} / #transitions

    Both strict -- an alarm in the same month as the transition is not an early
    warning and counts for neither side.

    RIGHT-CENSORING (`censoring`)
    -----------------------------
    An alarm raised less than H months before a bond's LAST observation has no
    observable outcome. This is not only a panel-edge effect: 65% of bonds in
    this panel stop before 2020-09 (maturity, call, coverage), so at H = 24
    months 42.8% of rows have an incomplete window while the global panel edge
    accounts for only 19.7% of them. Truncating the last 24 months of the panel
    therefore leaves more than half the censoring unaddressed.

      'naive'    Score every alarm; unobservable outcomes count as failures.
                 This is the previous behaviour and it biases precision DOWN.
      'exclude'  DEFAULT. Score only alarms with a complete H-month window
                 within their own bond's follow-up, and only transitions with a
                 complete H-month lookback. Per-bond, so it handles early exits
                 as well as the panel edge. Unbiased if censoring is independent
                 of the transition process.
      'km'       Kaplan-Meier. Keeps every alarm and uses its partial follow-up:
                 an alarm censored at 10 months contributes the information that
                 no transition occurred in its first 10 months. Strictly more
                 efficient than 'exclude', and the right choice when censoring is
                 heavy, at the cost of assuming independent censoring.

    Recall is left-truncated symmetrically: a transition occurring less than H
    months into a bond's history could not have been anticipated over the full
    window, so under 'exclude' and 'km' it leaves the denominator.
    """
    if censoring not in ("naive", "exclude", "km"):
        raise ValueError(f"censoring must be naive|exclude|km, got {censoring!r}")
    _require(df, [alarm_col, id_col, date_col, event_col],
             "evaluate_lookforward_alarms")
    df = df.reset_index(drop=True)
    dates = pd.to_datetime(df[date_col])
    ids = df[id_col].to_numpy()

    last = dates.groupby(ids).transform("max")
    first = dates.groupby(ids).transform("min")
    followup = ((last.dt.year - dates.dt.year) * 12
                + (last.dt.month - dates.dt.month)).to_numpy()
    history = ((dates.dt.year - first.dt.year) * 12
               + (dates.dt.month - first.dt.month)).to_numpy()

    dates = dates.to_numpy()
    alarm_pos = np.flatnonzero(df[alarm_col].to_numpy().astype(bool))
    event_pos = np.flatnonzero(df[event_col].to_numpy().astype(float) == 1)

    out = {
        "Total_Alarms": int(len(alarm_pos)),
        "Total_Events": int(len(event_pos)),
        "Total_Bonds": int(pd.unique(ids).size),
        "Event_Definition": event_col,
        "Censoring": censoring,
    }
    for key in ("n_runs", "alpha", "anytime_valid", "expected_false_alarm_bound"):
        if key in df.attrs:
            out[key] = df.attrs[key]
    if len(alarm_pos) == 0 or len(event_pos) == 0:
        return out

    by_id_alarms: dict = {}
    for p_ in alarm_pos:
        by_id_alarms.setdefault(ids[p_], []).append(p_)
    by_id_events: dict = {}
    for q_ in event_pos:
        by_id_events.setdefault(ids[q_], []).append(q_)

    # time from each alarm to its next transition (inf if none observed)
    gap_next = {}
    for p_ in alarm_pos:
        gaps = [_m(dates[q_], dates[p_]) for q_ in by_id_events.get(ids[p_], ())]
        fwd = [g for g in gaps if g > 0]
        gap_next[p_] = min(fwd) if fwd else np.inf

    for H in horizons:
        # ---------------- precision ----------------
        if censoring == "km":
            t_obs = [min(gap_next[p_], followup[p_]) for p_ in alarm_pos]
            ev = [1 if gap_next[p_] <= followup[p_] else 0 for p_ in alarm_pos]
            prec = _km_prob_by(t_obs, ev, H)
            n_prec = len(alarm_pos)
        else:
            usable = ([p_ for p_ in alarm_pos if followup[p_] >= H]
                      if censoring == "exclude" else list(alarm_pos))
            n_prec = len(usable)
            prec = (sum(1 for p_ in usable if gap_next[p_] <= H) / n_prec
                    if n_prec else 0.0)

        # ---------------- recall ----------------
        ev_usable = ([q_ for q_ in event_pos if history[q_] >= H]
                     if censoring in ("exclude", "km") else list(event_pos))
        n_rec = len(ev_usable)
        tp_e, leads = 0, []
        for q_ in ev_usable:
            gaps = [_m(dates[q_], dates[p_]) for p_ in by_id_alarms.get(ids[q_], ())]
            hit = [g for g in gaps if 0 < g <= H]
            if hit:
                tp_e += 1
                leads.append(min(hit))
        rec = tp_e / n_rec if n_rec else 0.0

        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out[f"P_{H}m (%)"] = round(100 * prec, 2)
        out[f"R_{H}m (%)"] = round(100 * rec, 2)
        out[f"F1_{H}m"] = round(f1, 4)
        out[f"Lead_{H}m (m)"] = round(float(np.mean(leads)), 2) if leads else 0.0
        out[f"N_scored_alarms_{H}m"] = int(n_prec)
        out[f"N_scored_events_{H}m"] = int(n_rec)
    return out


def _m(later, earlier) -> int:
    """Whole months from `earlier` to `later` (scalar helper)."""
    l, e = pd.Timestamp(later), pd.Timestamp(earlier)
    return (l.year - e.year) * 12 + (l.month - e.month)
