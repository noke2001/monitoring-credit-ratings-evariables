"""Calibration audit for the innovation target (thesis Section 3.8.2).

The e-value is valid for whatever null it is pointed at.  Replacing the target
points it at a different one,

    H_0^res : (X^mc_{i,t} - m(Z_{i,t-1})) / s(Z_{i,t-1}) exchangeable | F_{t-1},

and the map that appears there is the *estimated* m_hat, not the true one.  That
matters, and this script measures how much.  Any error in m_hat leaves a
bond-specific, F_{t-1}-measurable offset in the residual, so a panel whose
relation is genuinely stable still fails H_0^res by a margin that shrinks with
the amount of data the map is fitted on.  Lemma 3.3 is untouched throughout:
what is being measured is not a broken guarantee but a null that is only
approximately true.

Two generators, each a null for exactly one map (the level null is false in
both, badly):

  y_null      X_t = a + b Y_{t-1} + exchangeable noise, b fixed.
  self_null   a random walk started ordered by Y: persistent and Y-ordered in
              level, exchangeable in increments, no entity fixed effect.

and for each, three estimators of the map:

  oracle      the true (a, b, s) — isolates the exact null;
  fitted      m_hat on the K-month window, i.e. what the monitor does;
  and a sweep over K, so the excess can be seen to shrink.

A third generator, ``fixed_effect``, is the honest counterexample: an AR(1)
around heterogeneous entity means.  There a single pooled slope cannot remove
the entity effect, so *neither* map yields an exchangeable innovation and both
monitors accrue evidence on a panel where nothing is changing.

Usage:  python scripts/audit_residual_calibration.py [--reps 400] [--n-perms 199]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidate import build_candidate, evaluate_date  # noqa: E402
from src.evalue import mc_permutation_log_evalue  # noqa: E402
from src.residual import residualize_item  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

GENERATORS = ("y_null", "self_null", "fixed_effect")
NULL_MODE = {"y_null": "y", "self_null": "self", "fixed_effect": None}


def _equicorr(n, rho):
    return (1 - rho) * np.eye(n) + rho * np.ones((n, n))


def simulate(generator, rng, n=40, months=14, rho=0.4, b0=4.0, phi=0.9):
    """Return (X, Y, ids, truth) with ``truth`` the exact location map."""
    y = np.sort(rng.uniform(0.0, 1.0, size=n))
    L = np.linalg.cholesky(_equicorr(n, rho) + 1e-9 * np.eye(n))
    X = np.empty((months, n))
    if generator == "y_null":
        for t in range(months):
            X[t] = 6.0 + b0 * y + L @ rng.normal(size=n)
        truth = ("y", 6.0, b0)
    elif generator == "self_null":
        # a random walk started ordered by Y: the level is both persistent and
        # Y-ordered, so the level null is false in the way the panel's is, and
        # the increments are exchangeable, so the own-lag null holds exactly.
        X[0] = 6.0 + b0 * y + L @ rng.normal(size=n)
        for t in range(1, months):
            X[t] = X[t - 1] + L @ rng.normal(size=n)
        truth = ("self", 0.0, 1.0)
    elif generator == "fixed_effect":
        mu = 6.0 + b0 * y
        X[0] = mu + L @ rng.normal(size=n)
        for t in range(1, months):
            X[t] = (1 - phi) * mu + phi * X[t - 1] + L @ rng.normal(size=n)
        truth = ("self", None, None)          # no exact pooled affine map exists
    else:
        raise ValueError(generator)
    # mean-correct exactly as Section 2.4.2 does on the panel, so the synthetic
    # cross-sections enter the monitor in the same form as the bond ones
    X = X - X.mean(axis=1, keepdims=True)
    Y = y[None, :] + 1e-4 * rng.normal(size=(months, n))
    Y = Y - Y.mean(axis=1, keepdims=True)
    return X, Y, np.array([f"b{i}" for i in range(n)]), truth


def make_item(X, Y, ids, t, K):
    window = list(range(t - K, t))
    return {
        "date": t, "degenerate": False,
        "x": X[t].copy(), "y_lag": Y[t - 1].copy(), "x_lag": X[t - 1].copy(),
        "bond_ids": ids,
        "window_x_by_bond": {ids[i]: {m: float(X[m, i]) for m in window}
                             for i in range(ids.size)},
        "window_ylag_by_bond": {ids[i]: {m: float(Y[m - 1, i]) for m in window}
                                for i in range(ids.size)},
        "window_xlag_by_bond": {ids[i]: {m: float(X[m - 1, i]) for m in window}
                                for i in range(ids.size)},
        "pooled_window_x": X[window].ravel(),
        "window_month_list": window,
    }


def oracle_residualize(item, truth):
    """Residualise with the generator's own coefficients, so the innovation is
    exactly exchangeable and only Lemma 3.3 is being tested."""
    mode, a, b = truth
    if a is None:
        return None
    z_cur = item["y_lag"] if mode == "y" else item["x_lag"]
    z_by = item["window_ylag_by_bond"] if mode == "y" else item["window_xlag_by_bond"]
    out = dict(item)
    out["x"] = item["x"] - (a + b * np.asarray(z_cur, dtype=float))
    res, pooled = {}, []
    for bond, months in item["window_x_by_bond"].items():
        zb = z_by.get(bond, {})
        inner = {w: xv - (a + b * zb[w]) for w, xv in months.items() if w in zb}
        if inner:
            res[bond] = inner
            pooled.extend(inner.values())
    out["window_x_by_bond"] = res
    out["pooled_window_x"] = np.asarray(pooled, dtype=float)
    return out


def one_date(rng, generator, mode, estimator, n_perms, K, n, scale_model):
    X, Y, ids, truth = simulate(generator, rng, n=n, months=K + 2)
    item = make_item(X, Y, ids, K + 1, K)
    if estimator == "oracle":
        item = oracle_residualize(item, truth)
        if item is None:
            return None
    elif estimator == "fitted":
        item = residualize_item(item, mode=mode, scale_model=scale_model)
        if item.get("degenerate"):
            return None
    cand = build_candidate(item["x"], item["y_lag"], item["bond_ids"],
                           item["window_x_by_bond"], item["pooled_window_x"],
                           item["window_month_list"], max_block_size=10)
    if cand is None:
        return None
    ll_orig, ll_perms = evaluate_date(cand, item["x"], n_perms, rng)
    return mc_permutation_log_evalue(ll_orig, ll_perms)


def _summary(log_e, ranks, n_perms, alpha):
    """E[E_t] is the quantity the definition names, but it is a hopeless
    estimator here: E_t is capped at N+1 and violently right-skewed, so a few
    hundred draws give it a standard error of the same order as its mean.  Two
    low-variance statistics of the same null do the work instead.

      E[log E_t]  is what the process actually accumulates per date, and is
                  <= 0 under the null by Jensen;
      E[R_t/(N+1)] is 1/2 under the null exactly, because the permutation rank
                  R_t is uniform on {1, .., N+1} (Eq. 3.9).  A value below 1/2
                  means the observed arrangement is systematically favoured.
    """
    log_e = np.asarray(log_e, dtype=float)
    e = np.exp(np.clip(log_e, -700, 700))
    u = np.asarray(ranks, dtype=float) / (n_perms + 1)
    return {
        "mean_log_e": float(log_e.mean()),
        "se_log_e": float(log_e.std(ddof=1) / np.sqrt(log_e.size)),
        "mean_u": float(u.mean()),
        "se_u": float(u.std(ddof=1) / np.sqrt(u.size)),
        "mean_e": float(e.mean()),
        "se_e": float(e.std(ddof=1) / np.sqrt(e.size)),
        "p_reject": float(np.mean(e >= 1 / alpha)),
        "max_log_e": float(log_e.max()),
    }


def _row(w, label, K, vals, n_perms, alpha):
    st = _summary([v[0] for v in vals], [v[1] for v in vals], n_perms, alpha)
    w(f"  {label:22s} {K:4d} {st['mean_log_e']:9.3f} {st['se_log_e']:6.3f} "
      f"{st['mean_u']:11.3f} {st['se_u']:6.3f} {st['mean_e']:8.2f} "
      f"{st['p_reject']:10.3f} {st['max_log_e']:9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--n-perms", type=int, default=199)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--windows", nargs="*", type=int, default=[12, 36, 120])
    ap.add_argument("--scale-model", choices=["none", "const", "affine"],
                    default="affine")
    args = ap.parse_args()

    lines = []

    def w(s=""):
        lines.append(s)
        print(s, flush=True)

    w("Calibration audit of the innovation target (Section 3.8.2)")
    w(f"n = {args.n} entities, N = {args.n_perms} permutations, "
      f"{args.reps} replications, alpha = {args.alpha}, "
      f"cap log(N+1) = {np.log(args.n_perms + 1):.3f}")
    w("An e-value is valid for the null it is pointed at.  With a FITTED map "
      "the null")
    w("is only approximately true, because the map's estimation error is itself "
      "an")
    w("F_(t-1)-measurable, entity-specific offset.  The oracle rows isolate the "
      "exact null.")
    w()

    for generator in GENERATORS:
        mode = NULL_MODE[generator]
        w(f"===== generator: {generator} "
          f"(null for the {mode or 'neither'} map) =====")
        w(f"  {'target':22s} {'K':>4s} {'E[logE]':>9s} {'+/-':>6s} "
          f"{'E[R/(N+1)]':>11s} {'+/-':>6s} {'E[E_t]':>8s} "
          f"{'P(E>=1/a)':>10s} {'max logE':>9s}")

        # the level target, for scale: how false the level null is here
        rng = np.random.default_rng(20260826)
        vals = [v for _ in range(args.reps)
                if (v := one_date(rng, generator, None, "level", args.n_perms,
                                  args.windows[0], args.n, args.scale_model))
                is not None]
        _row(w, "level X^mc", args.windows[0], vals, args.n_perms, args.alpha)

        for mode_ in ("y", "self"):
            for estimator in ("oracle", "fitted"):
                if estimator == "oracle" and mode_ != mode:
                    continue                      # no exact map for the other one
                Ks = args.windows if estimator == "fitted" else args.windows[:1]
                for K in Ks:
                    rng = np.random.default_rng(20260826)
                    vals = []
                    for _ in range(args.reps):
                        v = one_date(rng, generator, mode_, estimator,
                                     args.n_perms, K, args.n, args.scale_model)
                        if v is not None:
                            vals.append(v)
                    if not vals:
                        continue
                    _row(w, f"innovation {mode_}, {estimator}", K, vals,
                         args.n_perms, args.alpha)
        w()

    w("Reading it.  Read E[logE] and E[R/(N+1)] first; E[E_t] is reported for "
      "continuity")
    w("with scripts/audit_calibration.py but is too skewed to read at these "
      "sample sizes.")
    w("On the oracle row of its own generator the innovation e-value is "
      "calibrated:")
    w("E[logE] <= 0 and the rank is uniform, as Lemma 3.3 requires.  The fitted "
      "rows sit")
    w("above it by the price of estimating the map, which is an "
      "F_(t-1)-measurable,")
    w("entity-specific offset and therefore a genuine (if small) failure of "
      "H_0^res, not")
    w("a failure of the estimator.  On the fixed_effect generator neither map "
      "is exact")
    w("and both innovation ranks sit below 1/2 on a panel where nothing is "
      "changing: a")
    w("pooled affine map cannot remove an entity fixed effect combined with")
    w("autoregressive dynamics.  What saves that case here is only that the "
      "candidate")
    w("is too weak to convert the residual structure into growth -- E[logE] is "
      "still")
    w("negative -- which is a statement about power, not about validity.  "
      "Compare every")
    w("row against the level row of the same block, which is what Sections 3.6 "
      "and 3.7")
    w("monitor.")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "residual_calibration_audit.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
