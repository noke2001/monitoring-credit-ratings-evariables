"""Frank copula log-densities in pure numpy, evaluated in log space.

The d-dimensional Archimedean density is

    c_theta(u) = (psi^{-1})^{(d)}(S) * prod_i psi'(u_i),      S = sum_i psi(u_i),

with generator psi_theta(t) = -log[(1 - e^{-theta t}) / (1 - e^{-theta})].
Writing a = 1 - e^{-theta} and w(s) = a e^{-s}, the pseudo-inverse is
psi^{-1}(s) = (1/theta) Li_1(w(s)) and its d-th derivative is
(1/theta)(-1)^d Li_{1-d}(w(s)), so that

    c_theta(u) = theta^{d-1} * Li_{-(d-1)}(w(S)) * prod_i e^{-theta u_i}/(1 - e^{-theta u_i}).

Negative-integer-order polylogarithms have the closed form

    Li_{-n}(w) = sum_{k=0}^{n-1} A(n,k) w^{n-k} / (1-w)^{n+1},

with Eulerian numbers A(n,k), which we evaluate in log space.  For d = 2 the
expression reduces to the familiar bivariate density, which is also implemented
directly (valid for theta < 0 as well) and used to cross-check the general form.
"""

from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

_THETA_EPS = 1e-6  # |theta| below this is treated as independence


@lru_cache(maxsize=None)
def log_eulerian(n: int) -> tuple:
    """Log Eulerian numbers log A(n, k), k = 0..n-1, via the standard recursion."""
    if n < 1:
        raise ValueError("n must be >= 1")
    row = np.array([0.0])  # A(1, 0) = 1
    for m in range(2, n + 1):
        prev = row
        row = np.full(m, -np.inf)
        for k in range(m):
            terms = []
            if k <= m - 2:
                terms.append(np.log(k + 1.0) + prev[k])
            if 1 <= k <= m - 1 and k - 1 <= m - 2:
                terms.append(np.log(m - k * 1.0) + prev[k - 1])
            row[k] = logsumexp(terms) if terms else -np.inf
    return tuple(row)


def log_polylog_neg(n: int, log_w: np.ndarray) -> np.ndarray:
    """log Li_{-n}(w) for w in (0,1), given log w, n >= 1.  Positive on (0,1)."""
    log_w = np.asarray(log_w, dtype=float)
    log_A = np.array(log_eulerian(n))                       # (n,)
    k = np.arange(n)
    powers = (n - k)[None, :] * log_w[..., None]            # (..., n)
    num = logsumexp(log_A[None, :] + powers, axis=-1)
    log_one_minus_w = np.log1p(-np.exp(log_w))
    return num - (n + 1) * log_one_minus_w


def _log1mexp(x: np.ndarray) -> np.ndarray:
    """log(1 - e^{-x}) for x > 0, numerically stable."""
    x = np.asarray(x, dtype=float)
    return np.where(x > np.log(2.0), np.log1p(-np.exp(-x)), np.log(-np.expm1(-x)))


def frank_logpdf_bivariate(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """log c_theta(u, v) for the bivariate Frank copula, any theta != 0."""
    if abs(theta) < _THETA_EPS:
        return np.zeros(np.broadcast(np.asarray(u), np.asarray(v)).shape)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    a = -np.expm1(-theta)                                   # 1 - e^{-theta}
    gu = -np.expm1(-theta * u)
    gv = -np.expm1(-theta * v)
    denom = a - gu * gv                                     # > 0 for both signs of theta
    return np.log(theta * a) - theta * (u + v) - 2.0 * np.log(np.abs(denom)) \
        if theta < 0 else \
        np.log(theta) + np.log(a) - theta * (u + v) - 2.0 * np.log(denom)


def frank_logpdf(U: np.ndarray, theta: float) -> np.ndarray:
    """log c_theta over the last axis of U (shape (..., d)); theta > 0 for d >= 3.

    d = 1 returns 0 (a one-dimensional copula is the identity), d = 2 delegates
    to the closed bivariate form so negative theta is supported there.
    """
    U = np.asarray(U, dtype=float)
    d = U.shape[-1]
    if d == 1 or abs(theta) < _THETA_EPS:
        return np.zeros(U.shape[:-1])
    if d == 2:
        return frank_logpdf_bivariate(U[..., 0], U[..., 1], theta)
    if theta < 0:
        raise ValueError("Frank copula requires theta > 0 in dimension d >= 3")
    U = np.clip(U, 1e-12, 1.0 - 1e-12)
    log_g = _log1mexp(theta * U)                            # log(1 - e^{-theta u_i})
    log_a = _log1mexp(np.asarray(theta))                    # log(1 - e^{-theta})
    log_w = np.sum(log_g, axis=-1) - (d - 1) * log_a        # log w(S) in (0, log a)
    log_li = log_polylog_neg(d - 1, log_w)
    log_prod = np.sum(-theta * U - log_g, axis=-1)          # sum log e^{-theta u}/(1-e^{-theta u})
    return (d - 1) * np.log(theta) + log_li + log_prod


def fit_theta_mle(monthly_U: list, allow_negative: bool | None = None,
                  bounds: tuple = (1e-4, 30.0)) -> float:
    """MLE of theta from a window of monthly cross-sections (each a 1-d array of
    pseudo-observations of that month's block members).

    Months with fewer than two members carry no copula information and are
    skipped.  Negative dependence is only representable when every contributing
    month is bivariate; otherwise the search is restricted to theta > 0 and a
    genuinely negatively-dependent block sits at the independence boundary
    (thesis Section 3.4.1).
    """
    from scipy.optimize import minimize_scalar

    months = [np.clip(np.asarray(u, dtype=float), 1e-9, 1 - 1e-9)
              for u in monthly_U if len(u) >= 2]
    if not months:
        return 0.0
    if allow_negative is None:
        allow_negative = all(len(u) == 2 for u in months)

    def nll(theta: float) -> float:
        total = 0.0
        for u in months:
            val = frank_logpdf(u[None, :], theta)[0]
            if not np.isfinite(val):
                return np.inf
            total += val
        return -total

    res_pos = minimize_scalar(nll, bounds=bounds, method="bounded")
    best_theta, best_val = float(res_pos.x), float(res_pos.fun)
    if allow_negative:
        res_neg = minimize_scalar(nll, bounds=(-bounds[1], -bounds[0]), method="bounded")
        if float(res_neg.fun) < best_val:
            best_theta, best_val = float(res_neg.x), float(res_neg.fun)
    if best_val >= nll(bounds[0]) - 1e-12 and abs(best_theta - bounds[0]) < 1e-3:
        return 0.0                                          # independence boundary
    return best_theta
