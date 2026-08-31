"""Monte Carlo permutation e-values with the identity permutation adjoined.

Implements Eq. (3.8) / Lemma 3.14 of the thesis: with sigma_0 = id and
sigma_1..sigma_N i.i.d. uniform permutations,

    E_t = (N+1) q(X) / sum_{j=0}^{N} q(X_{sigma_j}),

which satisfies E[E_t | G_t] <= 1 exactly for every finite N (with equality when
the denominator is a.s. positive) and is bounded by N+1 pointwise.  Omitting
sigma_0 gives the "legacy" estimator whose expectation exceeds one by Jensen's
inequality; ``legacy_to_corrected`` maps its realized values onto the valid
estimator in closed form, since the two share the same sampled permutations.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp


def mc_permutation_log_evalue(ll_orig: float, ll_perms: np.ndarray) -> tuple:
    """Return (log E_t, rank R_t) from the observed and permuted log-scores.

    ``ll_perms`` are the log-scores of the N *sampled* permutations only; the
    identity is adjoined here.  R_t = #{j in 0..N : LL_j >= LL_orig} is the
    permutation rank of Eq. (3.9), uniform on {1, .., N+1} under the null
    (ties resolved conservatively against the observed arrangement).
    """
    ll_perms = np.asarray(ll_perms, dtype=float)
    n_total = ll_perms.size + 1
    log_denom = logsumexp(np.append(ll_perms, ll_orig)) - np.log(n_total)
    rank = 1 + int(np.sum(ll_perms >= ll_orig))
    return float(ll_orig - log_denom), rank


def legacy_log_evalue(ll_orig: float, ll_perms: np.ndarray) -> float:
    """The biased estimator (identity omitted) — kept only for comparisons."""
    ll_perms = np.asarray(ll_perms, dtype=float)
    return float(ll_orig - (logsumexp(ll_perms) - np.log(ll_perms.size)))


def legacy_to_corrected(log_E_legacy: np.ndarray, M: int) -> np.ndarray:
    """Exact post-hoc identity fix:  E_new = (M+1) E_old / (M + E_old).

    Derivation: the legacy denominator is D = (1/M) sum_j q(X_sigma_j), and the
    corrected one is (q(X) + M D)/(M+1); dividing numerator and denominator of
    the corrected ratio by D gives the transform.  Computed in log space so
    log-e-values of order +-300 do not overflow.
    """
    log_E = np.asarray(log_E_legacy, dtype=float)
    return np.log(M + 1.0) + log_E - np.logaddexp(np.log(float(M)), log_E)


@dataclass
class EProcess:
    """Accumulates log e-values into the test supermartingale M_k = prod E_t."""

    alpha: float = 0.05
    dates: list = field(default_factory=list)
    log_e: list = field(default_factory=list)
    ranks: list = field(default_factory=list)
    meta: list = field(default_factory=list)

    def update(self, date, log_e_t: float, rank: int | None = None, **meta) -> None:
        self.dates.append(date)
        self.log_e.append(float(log_e_t))
        self.ranks.append(rank)
        self.meta.append(meta)

    @property
    def log_m(self) -> np.ndarray:
        return np.cumsum(np.asarray(self.log_e, dtype=float))

    @property
    def threshold(self) -> float:
        return float(np.log(1.0 / self.alpha))

    def first_alarm(self):
        """Index of the first date at which M_k >= 1/alpha, or None."""
        crossed = np.nonzero(self.log_m >= self.threshold)[0]
        return int(crossed[0]) if crossed.size else None
