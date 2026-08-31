"""The corrected non-exchangeable candidate (thesis Section 3.3).

Construction, per date t:

  (i)   Partition positions [n_t] into B blocks of near-equal size by sorting on
        the *lagged* partitioning covariate Y_{i,t-1} (F_{t-1}-measurable) and
        splitting at its empirical quantiles.  Blocks belong to POSITIONS.
  (ii)  Per block, fit a Gamma marginal f_b (shared support floor from the
        pooled class window) and a Frank parameter theta_b by MLE over the K
        monthly cross-sections of the window, blocked exactly as the e-variable
        will block them.
  (iii) q_t(x) = prod_b [ c_{theta_b}(F_b(x_i), i in B_b) * prod_{i in B_b} f_b(x_i) ].

Scoring is at the DESTINATION: an arrangement sigma places value x_{sigma(i)}
at position i and is scored under the marginal/transform of position i's block.
Every ingredient is computed from the window alone, so q_t is a fixed,
F_{t-1}-measurable function on R^{n_t}, and the marginal factors are retained
— they carry most of the candidate's power (thesis Sections 3.4.3-3.4.4).
"""

from dataclasses import dataclass

import numpy as np

from .frank_copula import frank_logpdf, fit_theta_mle
from .marginals import fit_gamma, gamma_logpdf_capped, gamma_cdf_clipped, shared_loc_floor


@dataclass
class BlockCandidate:
    block_positions: list          # list of index arrays into [n_t] (positions)
    gamma_params: list             # per block
    thetas: list                   # per block
    n: int
    pit_mode: str = "destination"  # "origin" reproduces the legacy defect
    include_marginals: bool = True # False reproduces the legacy deletion

    def score_arrangements(self, x_values: np.ndarray, arrangements: np.ndarray) -> np.ndarray:
        """Log q_t of each arrangement (rows of ``arrangements`` map position i
        to the index of the value placed there).

        With ``pit_mode="destination"`` and ``include_marginals=True`` this is
        Eq. (3.20) as written.  The other settings exist only to reproduce the
        legacy pipeline's defects for the comparison of Section 3.4.
        """
        arrangements = np.atleast_2d(arrangements)
        scores = np.zeros(arrangements.shape[0])
        if self.pit_mode == "origin":
            # each value carries the pseudo-observation of the block it CAME
            # from, so relabelings are scored under mismatched transforms
            u_origin = np.empty(self.n)
            for pos, gp in zip(self.block_positions, self.gamma_params):
                u_origin[pos] = gamma_cdf_clipped(x_values[pos], gp)
        for pos, gp, theta in zip(self.block_positions, self.gamma_params, self.thetas):
            assigned = x_values[arrangements[:, pos]]              # (n_arr, d_b)
            if self.include_marginals:
                scores += gamma_logpdf_capped(assigned, gp).sum(axis=1)
            u = (u_origin[arrangements[:, pos]] if self.pit_mode == "origin"
                 else gamma_cdf_clipped(assigned, gp))
            scores += frank_logpdf(u, theta if theta > 0 or u.shape[1] == 2 else 0.0)
        return scores


def build_candidate(
    x_current: np.ndarray,
    y_lagged: np.ndarray,
    bond_ids: np.ndarray,
    window_x_by_bond: dict,
    pooled_window_x: np.ndarray,
    window_months: list,
    max_block_size: int = 15,
    min_occupancy: int = 2,
    min_hist_obs: int = 10,
    pit_mode: str = "destination",
    include_marginals: bool = True,
    block_labels: np.ndarray | None = None,
) -> BlockCandidate | None:
    """Build q_t for one date, or return None if the cohort is degenerate
    (in which case the caller records E_t = 1 — declining to bet, not skipping).

    ``window_x_by_bond`` maps bond id -> {month -> x value} over the window;
    ``pooled_window_x`` is the pooled class history used for the shared floor.

    ``block_labels`` supplies a categorical partition (e.g. SIC divisions) and
    takes precedence over the quantile split of ``y_lagged``.  Groups smaller
    than ``min_occupancy``, together with unlabelled positions, are merged into
    a single residual block.
    """
    n = x_current.size
    if n < 2 * min_occupancy:
        return None
    floor = shared_loc_floor(pooled_window_x)

    if block_labels is not None:
        # Categorical partition (design a): blocks are the label groups
        # themselves, not quantile cuts.  Labels are static SIC divisions, so
        # the partition is F_0-measurable; the merging rule below reads only
        # cohort composition, which Section 3.1.2 assumes F_{t-1}-measurable,
        # never the current target values.
        labels = np.asarray(block_labels, dtype=object)
        groups = [np.flatnonzero(labels == lab)
                  for lab in sorted({l for l in labels if l is not None},
                                    key=str)]
        big = [g for g in groups if g.size >= min_occupancy]
        small = [g for g in groups if g.size < min_occupancy]
        leftover = np.concatenate(
            [np.flatnonzero(np.array([l is None for l in labels]))] +
            [g for g in small]) if (small or any(l is None for l in labels)) \
            else np.array([], dtype=int)
        candidate_blocks = list(big)
        if leftover.size >= min_occupancy:
            candidate_blocks.append(np.sort(leftover))
        if len(candidate_blocks) < 2:
            return None
        return _fit_blocks(candidate_blocks, bond_ids, window_x_by_bond,
                           window_months, floor, min_hist_obs, n,
                           pit_mode, include_marginals)

    order = np.argsort(y_lagged, kind="stable")            # positions sorted by lagged Y
    b_max = min(n // min_occupancy, max(2, int(np.ceil(n / max_block_size))))

    for n_blocks in range(b_max, 1, -1):
        cand = _fit_blocks(np.array_split(order, n_blocks), bond_ids,
                           window_x_by_bond, window_months, floor,
                           min_hist_obs, n, pit_mode, include_marginals)
        if cand is not None:
            return cand
    return None


def _fit_blocks(blocks, bond_ids, window_x_by_bond, window_months, floor,
                min_hist_obs, n, pit_mode, include_marginals):
    """Fit a marginal and a copula parameter on each block, or return None if
    any block lacks enough history."""
    gamma_params, thetas = [], []
    for pos in blocks:
        members = bond_ids[pos]
        hist = np.concatenate([
            np.array([w[m] for m in window_months if m in w], dtype=float)
            for w in (window_x_by_bond.get(b, {}) for b in members)
        ]) if len(members) else np.array([])
        if hist.size < min_hist_obs:
            return None
        gp = fit_gamma(hist, floor)
        gamma_params.append(gp)
        monthly_U = []
        for m in window_months:
            vals = np.array([window_x_by_bond[b][m] for b in members
                             if b in window_x_by_bond and m in window_x_by_bond[b]])
            if vals.size >= 2:
                monthly_U.append(gamma_cdf_clipped(vals, gp))
        thetas.append(fit_theta_mle(monthly_U) if monthly_U else 0.0)
    return BlockCandidate([np.asarray(p) for p in blocks], gamma_params,
                          thetas, n, pit_mode, include_marginals)


def sample_permutation(n: int, rng: np.random.Generator,
                       groups: np.ndarray | None = None) -> np.ndarray:
    """Uniform draw from the full symmetric group, or from the subgroup that
    permutes only within the given groups.

    ``groups`` implements partial exchangeability (design b): the null then
    asserts only that the cross-section is exchangeable *within* each group ---
    for us, within a rating class and SIC division --- and conceding the
    between-group structure costs nothing in validity, since the proof of the
    e-value property uses only that the permutations form a group.
    """
    if groups is None:
        return rng.permutation(n)
    perm = np.empty(n, dtype=int)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        perm[idx] = rng.permutation(idx)
    return perm


def evaluate_date(candidate: BlockCandidate, x_current: np.ndarray,
                  n_perms: int, rng: np.random.Generator,
                  groups: np.ndarray | None = None) -> tuple:
    """Return (ll_orig, ll_perms) for the identity and N sampled arrangements.

    ``groups`` restricts the permutations to a subgroup (see
    ``sample_permutation``).  NOTE: with a restricted group the candidate must
    be partitioned on something *other* than the grouping variable, or q_t is
    invariant under the group action and E_t == 1 identically.
    """
    identity = np.arange(candidate.n)[None, :]
    perms = np.stack([sample_permutation(candidate.n, rng, groups)
                      for _ in range(n_perms)])
    ll = candidate.score_arrangements(x_current, np.vstack([identity, perms]))
    return float(ll[0]), ll[1:]
