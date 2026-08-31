"""Corrected exchangeability-monitoring pipeline (thesis Chapter 3, v5).

Modules:
    frank_copula  — Frank copula log-densities (numpy, log space) and theta MLE
    marginals     — Gamma marginals with shared support floor
    candidate     — the block candidate q_t with destination-PIT scoring
    evalue        — Monte Carlo permutation e-values (identity adjoined), e-process
    panel         — bond panel loading, mean-correction, lagged covariates
    residual      — §3.8.2: the location-scale map and the innovation
                    target (level -> residual); leaves q_t untouched
    univariate    — Chapter 4 §4.1-§4.2: the cohort PIT and its diagnostics
    tabpfn_panel  — Chapter 4 §4.3: reading the saved TabPFN fits
    models        — Chapter 4 §4.3: the VAE zoo (torch, imported lazily)
    summaries     — Chapter 4 §4.3: the five candidate summary functions
    betting       — Chapter 4 §4.4: exact e-value primitives and the PIT
                    constructions (note: Chapter 3's permutation e-values are
                    in `evalue`, singular -- different object, different null)
    sequential    — Chapter 4 §4.4: the E-process engines
    metrics       — Chapter 4 §4.4: censoring-aware alarm evaluation, DD, TRR
    validation    — Chapter 4 §4.4: the certification harness
"""

from . import frank_copula, marginals, candidate, evalue, panel, residual  # noqa: F401,E501
from . import univariate  # noqa: F401
from . import tabpfn_panel, summaries, betting, sequential, metrics, validation  # noqa: F401,E501
# models imports torch; leave it to the caller (`from src.models import ...`)
