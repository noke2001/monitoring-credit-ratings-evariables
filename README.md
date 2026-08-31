# Monitoring Credit Ratings with E-Variables — codebase

Code for my Master's Thesis *Monitoring Credit Ratings with
E-Variables* (2026). **This directory is the whole repository**: everything the
thesis cites is either here or can be generated from here, and nothing outside it is
needed to read the code.

| Chapter | Question | Code |
|---|---|---|
| **3** — Testing Exchangeability | has a rating *class* stopped being homogeneous? | `src/{frank_copula,marginals,candidate,evalue,panel}.py`, `scripts/run_*.py` |
| **3.8.2** — Testing the Innovation | has the *relation* the class was ordered by stopped holding? | `src/residual.py`, `scripts/{run_residual_synthetic,residual_diagnostic,audit_residual_calibration,make_residual_figures}.py` |
| **4** — Outlier Detection | which *bond* has stopped belonging? | `src/{univariate,tabpfn_panel,models}.py`, `scripts/{run_univariate,audit_tabpfn}.py`, `scripts/{tabpfn,vae}/` |

---

## Layout

```
src/                     importable library — no scripts, no side effects
  frank_copula.py        Ch3  Frank copula log-densities in pure numpy, any
                              dimension (negative-order polylogs), + theta MLE
  marginals.py           Ch3  Gamma marginals with the shared support floor
  candidate.py           Ch3  the block candidate q_t, destination-PIT scoring
  evalue.py              Ch3  MC permutation e-values (identity adjoined),
                              e-process, exact legacy->corrected transform
  panel.py               Ch3  panel loading, (date,class) mean-correction,
                              lagged covariates, per-date cohort iterator
  residual.py            §3.8.2  the innovation target: the location-scale map
                              m_hat (affine or binned), the reference/candidate
                              window split, the level -> residual rewrite, and
                              the symmetric cross-sectional standardisation.
                              q_t, the e-value and Lemma 3.3 are untouched;
                              only the object being permuted changes
  summaries.py           §4.3 the five candidate Gamma's: directional
                              deviation, log transition-risk-ratio, and the
                              Brier / ranked-probability / logarithmic scores
  univariate.py          §4.2 the cohort PIT of eq. (4.1), its randomised
                              (exactly uniform) variant, the persistence
                              diagnostic, transition labels
  tabpfn_panel.py        §4.3 reading the saved TabPFN fits: provenance table,
                              integrity scan, and the PIT sign correction
  models.py              §4.3 the VAE zoo — SemiSupervisedFTVAE, TabularVAE,
                              LSTMVAE.  Needs torch; import it directly.
  betting.py             §4.4 exact e-value primitives (deadzone_e, power_e,
                              mixture_power_e, grenander_e, kelly_lambda)
                              and the PIT constructions
                              (randomized_rank_pit, ar1_innovation).  Every
                              function carries a closed-form E[e]=1 certificate
                              in its docstring.  NB Chapter 3's permutation
                              e-values live in `evalue.py`, singular.
  sequential.py          §4.4 the E-process engines: seven fixed-bet ones
                              (DeadzoneEProcess is the baseline),
                              the AHZ Grenander (published, nonparametric),
                              plug-in Kelly, and their composition
  metrics.py             §4.4 censoring-aware alarm evaluation, DD, TRR
  validation.py          §4.4 the certification harness

scripts/                 drivers — every one writes, none is imported
  run_synthetic.py       §3.5  the four validation regimes + legacy generator
  run_empirical.py       §3.6  corrected monitor over the bond panel.
                               --target-mode {level,y,self} switches between
                               the level and the two innovation targets
  run_sic.py             §3.7  the two SIC designs
  rank_diagnostic.py     §3.6  model-free companion: rho(Y_{t-1}, X_t)
  run_residual_synthetic.py §3.8.2  five regimes whose *relation stability* is
                               known, run under all three targets, plus the
                               drift-size power sweep
  residual_diagnostic.py §3.8.2  rho(Y_{t-1}, X_t) before and after the change
                               of target, and the fitted map over time
  audit_residual_calibration.py §3.8.2  E[logE] and the permutation rank under
                               the innovation null, oracle map vs fitted map
  make_residual_figures.py §3.8.2  the figures, from the saved CSVs
  sweep_window_split.py  §3.8.2  size and power over (K, reference share);
                               this is what selects K = 24 and 25%, rather
                               than the split being asserted
  screen_pairs.py        §3.8.2  model-free screen of every (X, Y) pair on the
                               panel: how much ordering the map leaves, how
                               persistent it is, and whether it grows or
                               collapses in NBER recessions
  audit_calibration.py   §3.1  E[E_t] <= 1 and the Ville rate, corrected vs naive
  make_thesis_figures.py       vector PDFs for Ch3, from the saved CSVs
  make_comparison_figure.py    legacy vs corrected on identical data
  make_sic_figures.py          the Section 3.7 figures
  correct_legacy_outputs.py    exact identity-fix audit of the legacy notebooks
  run_univariate.py      §4.2  reproduces Figure 4.1 from eq. (4.3) and the
                               five diagnostic tables the section quotes (~9 s)
  audit_tabpfn.py        §4.3  audits the SAVED fits without refitting:
                               provenance + integrity, the accuracy paradox,
                               early-warning AUC, the PIT sign  (~15 s)
  fix_pit_sign.py        §4.3  re-emits the pre-fix saved fits with the PIT
                               corrected (sidecar by default, --in-place opt-in)
  audit_vae.py           §4.3  scores every candidate from the VAE panel on
                               identical rows; --baseline compares two panels
  make_ch4_figures.py    Ch4   the four publication figures, house style,
                               vector PDF + PNG.  --only <name> for one
  eprocess/              §4.4  the benchmark and PIT-selection drivers
    benchmark_eprocesses.py    13 engine configs x 4 alpha levels (~16 s)
    select_final_pit.py        validity (ACF) vs power, decides the PIT
    test_conditional_pit.py    innovation variants: diff-1, diff-6, AR(1)
    compare_pit_constructions.py  head-to-head incl. the model PIT
    pit_power_and_diagnostics.py  uniformity diagnostics + PR frontier
    plot_case_studies.py       TP / FP / FN / TN case studies
    diagnose_f1_drop.py, decompose_f1_change.py, quantify_label_leak.py,
    rating_change_statistics.py, train_hazard_redesign.py,
    train_multi_horizon_pipeline.py
  tabpfn/                §4.3  the eight supervised variants — see below
    sweep.sh                   drives one variant across every date
  vae/                   §4.3  the unsupervised / semi-supervised trainers
    train_semisupervised.py    FT-VAE + LightGBM, fresh-rating context (~157 s)
    train_semisupervised_v1.py.orig  pre-fix version, kept for comparison
    train_enhanced.py          earlier variant

tests/                   certification — run these first, they exit non-zero
  test_frank_copula.py         densities against mpmath and by integration
  test_evalue_validity.py      E[E_t] <= 1, the cap, rank uniformity, Ville
  test_candidate.py            destination vs origin PIT, measurability
  test_residual.py             §3.8.2: the map (location and scale slopes,
                               fallbacks, clipping), F_{t-1}-measurability,
                               that the innovation removes the dependence the
                               level test lives on, calibration under the
                               oracle and the fitted map, and that the window
                               split is what gives the monitor its power
  test_partial_exchangeability.py  within-group permutations, degeneracy trap
  test_univariate.py           47 checks on the §4.2 PIT, incl. Lemma B.2
  test_summaries.py            the five summaries against closed forms, and
                               the three scoring rules against propriety
  test_deadzone.py             §4.4: Lemma B.3 — exact validity of the
                               dead-zone bet, its range, admissibility, and
                               the closed-form Kelly stake against the
                               numerical optimum
  test_new_engines.py          §4.4: the AHZ Grenander (the estimate really
                               is a density, to 1e-12) and the plug-in Kelly
                               stake (predictable, starts at exactly 0), plus
                               Ville on a pure-null panel for both
  test_eprocess.py             §4.4: the whole validity chain — exact PIT,
                               E[e]=1, martingale, Ville — plus the measured
                               size of each legacy defect.  Run after any
                               change to src/betting.py or src/sequential.py

thesis_edits/            drop-in LaTeX for the thesis, with its own README
plots/  results/         generated figures and per-date CSVs
```

### What is deliberately *not* here

The raw panel (`CorpBond_Reconciling/corp_jkp_mergedv2.csv`, 349 MB) and the
saved TabPFN fits (`tabpfnfit/`, ~4 GB) are data, not code. Point at them with

```bash
export BOND_CSV=/path/to/corp_jkp_mergedv2.csv
export TABPFN_OUT=/path/to/tabpfnfit
```

Both default to sensible locations relative to this directory.

### Credentials

`scripts/tabpfn/*` call the TabPFN and HuggingFace APIs and read
`TABPFN_TOKEN` and `HF_TOKEN` **from the environment**. They are never written
to disk and the scripts refuse to start without them.

```bash
export TABPFN_TOKEN=...   # https://tabpfn.com -> API key
export HF_TOKEN=...       # https://huggingface.co/settings/tokens
```

---

## Environments — they are not interchangeable

The .yaml are available from within the codebase. Note that they were built for MacOS, meaining that Windows and Linux users ight have to tweak them to work.

```bash
conda activate copula   # Chapter 3          numpy 1.26, polars, scipy, mpmath
conda activate bond     # Chapter 4          numpy 2.x, torch, lightgbm, tabpfn
```

Chapter 3 is pure numpy/scipy — **no R or rpy2**: the Frank densities the legacy
notebooks took from the R `copula` package are computed natively and certified
against `mpmath` in `tests/test_frank_copula.py`. Chapter 4 needs torch and
lightgbm. `src/univariate.py` and `src/tabpfn_panel.py` run under either.
Specs in `../environment-copula.yml` and `../environment-bond.yml`.

---

## Running things

```bash
# certify first — all five exit non-zero on failure
conda activate copula
python tests/test_frank_copula.py
python tests/test_evalue_validity.py
python tests/test_candidate.py
python tests/test_partial_exchangeability.py
python tests/test_univariate.py
python tests/test_summaries.py
conda activate bond
python tests/test_deadzone.py               # ~5 s
python tests/test_new_engines.py             # ~90 s
python tests/test_eprocess.py               # ~7 s

# Chapter 3
python scripts/run_synthetic.py --reps 10 --n-perms 1000
python scripts/audit_calibration.py --reps 400
python scripts/run_empirical.py --n-perms 1000 --notches 14 15 16
python scripts/make_thesis_figures.py

# Chapter 3, Section 3.8.2 — the innovation target
python tests/test_residual.py                     # ~4 min
python scripts/run_residual_synthetic.py --reps 10 --n-perms 1000
python scripts/audit_residual_calibration.py --reps 400
python scripts/run_empirical.py --n-perms 1000 --target-mode y
python scripts/run_empirical.py --n-perms 1000 --target-mode self
python scripts/residual_diagnostic.py
python scripts/make_residual_figures.py
python scripts/sweep_window_split.py --reps 10 --n-perms 199   # ~2 h
python scripts/screen_pairs.py                                 # ~40 min

# Chapter 4 — §4.2 and the §4.3 audit need no model fitting
python scripts/run_univariate.py            # ~9 s
python scripts/audit_tabpfn.py              # ~15 s, reads the saved fits

# Chapter 4 — refitting (hours, GPU/MPS, needs the API tokens)
conda activate bond
./scripts/tabpfn/sweep.sh run_delta_dom_new.py 2003-08-01
python scripts/vae/train_semisupervised.py

# Chapter 4 §4.4 — needs the VAE panel in results/
python scripts/eprocess/benchmark_eprocesses.py --panel vae   # ~16 s
python scripts/eprocess/select_final_pit.py                   # ~15 s
```

The larger inference panels (`multi_horizon_survival_inference_results.csv`,
`hazard_redesign_scores.csv`) are data and are not in the repository; point at
them with `export EPROCESS_PANELS=/path/to/panels`.

---

## Chapter 4 §4.3 — the eight supervised variants

Each `scripts/tabpfn/run_*.py` handles **one date** and prints the next, so
`sweep.sh` can drive a full pass. They differ only in how the in-context
training set is built and what is predicted.

| Script | Target | Training context | Writes |
|---|---|---|---|
| `run_level_basic.py` | current rating | trailing K months, all rows | `tabpfn_data_{T}_{K}.csv` |
| `run_level.py` | current rating | as above, + class probabilities | `..._wprobs.csv` |
| `run_level_onlytrans.py` | current rating | **transition events only** | `..._onlytrans.csv` |
| `run_level_freshfit.py` | current rating | fresh ratings + quiescent anchors | `..._fresh_fit_minbonds_{N}.csv` |
| `run_delta.py` | Δ vs entry rating | trailing K months | `tabpfn_delta_fit_{T}_{K}m.csv` |
| `run_delta_dom.py` | Δ vs entry rating | + dominant-class cap | `tabpfn_delta_fit_dom_{R}_*.csv` |
| `run_delta_dom_new.py` | Δ vs entry rating | + fresh-entry ratio, minority backfill | `..._new.csv` |
| `run_topdelta.py` | Δ **12m ahead** | 10k rows, 16 LightGBM-selected feats | `tabpfn_horizon_fit_results_*.csv` |

`src/tabpfn_panel.PROVENANCE` maps every saved file back to its driver.

### Two defects in the pre-existing saved fits

Both are fixed at source; `src/tabpfn_panel.py` corrects the first on read so
the old files stay usable.

1. **The stored PIT was inverted.** Every driver wrote
   `rank(directional_deviation, ascending=False)`, which put *downgrade*
   candidates near 0 — the opposite of the convention the write-up and the
   one-sided E-values of §4.4 assume. Measured: AUC(stored PIT → downgrade)
   ∈ [0.324, 0.382] across all nine configurations, uniformly below ½.
   `load_fit()` recomputes it ascending as `pit` and keeps the original as
   `pit_stored`; `scripts/fix_pit_sign.py` re-emits corrected sidecars.
2. **Append-header misalignment in `*_wprobs.csv`.** Each date wrote one
   `prob_*` column per class in *that* date's context while `to_csv(mode='a')`
   froze the header at the first date's set, so probabilities shift from the
   first missing class onward. `scan_integrity()` measures it as the share of
   rows where `argmax(prob_*)` reproduces the stored label: 8.5% for
   `tabpfn_data_nrtg_12_wprobs.csv`, 9.7% for `nrtg_6`, 57.3% for `nrtg_1`,
   ~99.6% for the `rtg` ones. **The leading columns are written before the
   block and are unaffected**, so scores, deviations and ranks survive; only
   the probability *vector* is unusable on the `nrtg` files. The `delta_fit`
   and `horizon_fit` drivers reindex to a fixed column set and are clean.

---

## Status

Verified 2026-08-24: every test above passes under both environments, and
`run_univariate.py` / `audit_tabpfn.py` run end to end.

Verified 2026-08-26: `tests/test_residual.py` (14 checks) passes, and the
Section 3.8.2 drivers run end to end on the bond panel.

### Why the innovation monitor splits its window

Section 3.8.2 of the thesis says that replacing the target requires no change
to the machinery. For *validity* that is exactly right and `test_residual.py`
certifies it. For *power* it is not, and the failure is complete rather than
marginal. If `m_hat` and the candidate are fitted on the same window, least
squares leaves that window's residuals homogeneous across the candidate's
blocks by construction; `q_t` is then nearly exchangeable by Lemma 3.14, `E_t`
is pinned at one, and a break in the relation — the alternative the change of
target exists for — is very nearly invisible.

`src/residual.split_window` fits the map on the older half of a 24-month window
and the candidate on the recent half, both residualised through the same
reference map. On the `relation_break` regime, 10 replications, `N = 1000`:

| | rejection rate | median max `log M_k` |
|---|---|---|
| literal (one window, `K = 12`) | 0.00 | −1.1 |
| **split (`K = 24`, older half fits the map)** | **1.00** | **73.2** |

The literal version is not quite powerless — restarted at the break it does
cross in 8 of 10 — but it has spent the preceding two years giving ground, so
it never recovers to its own starting level, let alone `1/alpha`, inside the
sample. The split monitor crosses two months after the break.
`--reference-share 0 --window-months 12` reproduces the literal reading and
writes to `*_nosplit_K12` files, so the two sit side by side.

### Where the split should fall, and two things that came out of asking

`sweep_window_split.py` sweeps `K` against the reference share on panels whose
relation-stability is known. Three results, the same under both maps:

1. **A quarter, not a half.** Power peaks at a reference share of `0.25` at
   every `K`; `0.5` costs 0.4–0.9 nats a month against it. The map has four
   parameters and the candidate a Gamma and a Frank `theta` *per block*, so a
   month of history is worth far more to the second.
2. **`K = 24`, not more.** Power rises monotonically with `K`, but at `K = 36`
   the realised size on the null regime runs at 0.20–0.30 against a nominal
   0.05 — the map's own estimation error becoming visible to a candidate sharp
   enough to see it. The default is the largest `K` whose size is still nominal.
3. **No split is dominated.** Share `0` is worse than `0.25` on power at every
   `K` and worse on size at `K >= 24`.

Two findings came out of the same investigation and are worth flagging:

**Symmetric standardisation is free, and it fixes the recession pathology.**
`cross_section_scale` divides each cross-section by its own MAD. Because that is
a *permutation-invariant* function of `X(t)`, the standardised vector is
exchangeable whenever the original is, so the null is untouched — this is the
one ingredient allowed to look at the current cross-section. Measured: the
residual's dispersion is 3–7x its calm value in recession months while the
level's barely moves, and standardising moves the share of terminal evidence
accruing inside recessions from `-0.35` to `+0.11` under the own-lag map.
`--cross-scale mad`.

**A NaN-vs-null bug in `load_panel`, fixed.** Polars propagates NaN through
`mean()`, so one NaN in a `(date, class)` cell made that cell's mean-correction
NaN and silently deleted it from everything downstream. `yield`, `spread`, `duration` and `coupon` carry
**zero** NaNs after the target filter, so **no published result changes** — a
re-run with the patch reverted reproduces the current CSVs to the same 5e-6
(1e-8 relative, from `scipy` version drift since the files were first written,
not from these changes), and cells re-run within one session match
bit-for-bit. The sparser covariates `screen_pairs.py` needs carry a great many:
`D2D` had a 0.3% usable-lag rate before the fix and 92% after.

### A trap worth recording before §4.4 is migrated

Removing the persistence of a score and ranking it **do not commute**, and the
difference is large. On the VAE panel, with `det_score = 1 - P(monitored
rating)`:

| construction | ACF(1) |
|---|---|
| rank the level | 0.854 |
| **AR(1) residual of the score, then rank** (what the pipeline does) | **0.020** |
| rank the level, then take the AR(1) residual of the *rank* | −0.138 |

Residualising the rank over-differences, because the rank transform is
nonlinear and a coefficient fitted on the rank scale does not remove the
dependence it was fitted to. The order in `benchmark_eprocesses.build_pit` --
`ar1_innovation` on the score, `randomized_rank_pit` on the residual -- is the
correct one. Randomized versus average ranks makes no difference to the ACF
(both 0.8541), since it is a rank-order property.

Two smaller notes on the same layer: `src/evalues.py` quotes 0.026 for the
innovation ACF where the panel gives 0.0195, and `scripts/vae/*.py` define
their own `SemiSupervisedFTVAE` that differs from `src/models.py` (it
classifies from `mu` rather than the sampled `z` and clamps `logvar`). The
local definition is the one that produced the shipped panel.

**Migration complete.** Chapters 3 and 4 are both fully here. The only things
outside this directory are data: the raw panel, the saved TabPFN fits, and the
two large inference panels named above.

| Document | Covers |
|---|---|
| `thesis_edits/README.md` | the LaTeX drop-ins and which result backs which claim |
| `results/univariate_summary.txt` | every number in §4.2 |
| `results/tabpfn_audit.txt` | every number in §4.3.1 |
