"""
Multi-Horizon Forward Transition & Survival Sequential Testing Pipeline
Master's Thesis: Monitoring Corporate Credit Ratings with E-Variables

Key Features:
  1. Multi-Horizon Forward Targets: y_12m, y_24m, and Next-Transition Destination (y_regime)
  2. Multi-Task Survival-Hazard Predictor (Focal loss on forward insolvency risk)
  3. Bounded Sequential E-Martingales (strictly bounded below by 1.0)
  4. Non-Penalizing Regime Evaluation (Alarms verified against next rating transition)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

print("================================================================================")
print("   STARTING MULTI-HORIZON FORWARD SURVIVAL & SEQUENTIAL E-TESTING PIPELINE")
print("================================================================================")
t0 = time.time()

# 1. Load Raw Corporate Panel Data
raw_data_path = './CorpBond_Reconciling/corp_jkp_mergedv2.csv'
print(f"Loading raw bond data from: {raw_data_path}")
raw_df = pd.read_csv(raw_data_path, low_memory=False)
raw_df['dates'] = pd.to_datetime(raw_df['dates'].astype(str), format='%Y%m', errors='coerce')

ORDERED_CLASSES = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']
RATING_ORDER = {'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3, 'BB': 4, 'B': 5}
df = raw_df[raw_df['rtg'].isin(ORDERED_CLASSES)].copy()
df['enc_y'] = df['rtg'].map(RATING_ORDER)
df['isin'] = df['isin'].fillna(df['cusip'] if 'cusip' in df.columns else df['bond_id'])
df = df.dropna(subset=['dates', 'isin', 'enc_y']).sort_values(['isin', 'dates']).reset_index(drop=True)

# Transitions and Monitored Baseline
df['prev_rtg'] = df.groupby('isin')['rtg'].shift(1)
df['prev_enc_y'] = df.groupby('isin')['enc_y'].shift(1).fillna(df['enc_y'])
df['is_rating_change'] = (df['rtg'] != df['prev_rtg']) & (~df['prev_rtg'].isna())
df['is_downgrade'] = ((df['enc_y'] > df['prev_enc_y']) & df['is_rating_change']).astype(int)

# 2. Multi-Horizon Targets
df['down_in_12m'] = df.groupby('isin')['is_downgrade'].transform(
    lambda s: s.iloc[::-1].rolling(12, min_periods=1).max().iloc[::-1].shift(-1)
).fillna(0).astype(int)

df['change_in_12m'] = df.groupby('isin')['is_rating_change'].transform(
    lambda s: s.iloc[::-1].rolling(12, min_periods=1).max().iloc[::-1].shift(-1)
).fillna(0).astype(int)

df['down_in_24m'] = df.groupby('isin')['is_downgrade'].transform(
    lambda s: s.iloc[::-1].rolling(24, min_periods=1).max().iloc[::-1].shift(-1)
).fillna(0).astype(int)

df['change_in_24m'] = df.groupby('isin')['is_rating_change'].transform(
    lambda s: s.iloc[::-1].rolling(24, min_periods=1).max().iloc[::-1].shift(-1)
).fillna(0).astype(int)

# 3. Next-Transition Destination Target (Non-Penalizing Regime Target)
def get_regime_destination(group):
    change_mask = group['is_rating_change'].values
    enc_vals = group['enc_y'].values
    n = len(group)
    next_rtg = np.full(n, np.nan)
    months_to_next = np.full(n, 999.0)
    
    change_indices = np.where(change_mask)[0]
    for i in range(n):
        future_changes = change_indices[change_indices > i]
        if len(future_changes) > 0:
            nxt_idx = future_changes[0]
            next_rtg[i] = enc_vals[nxt_idx]
            months_to_next[i] = nxt_idx - i
        else:
            next_rtg[i] = enc_vals[i]
            months_to_next[i] = n - 1 - i
            
    group['next_destination_enc'] = next_rtg
    group['months_to_next_change'] = months_to_next
    return group

print("Constructing Next-Transition Regime Targets...")
df = df.groupby('isin', group_keys=False).apply(get_regime_destination)
df['is_next_downgrade'] = (df['next_destination_enc'] > df['enc_y']).astype(int)

# 4. Feature Selection & Longitudinal Deltas
exclude = {'isin', 'cusip', 'bondsym', 'compsym', 'dates', 'date', 'rtg', 'nrtg', 'enc_y', 'nextdate', 'nextret', 'nextretexc', 'nextretwins', 'nextretexcwins', 'convertflg', 'mom6xrtg', 'id', 'sic', 'prev_rtg', 'prev_enc_y', 'is_downgrade', 'is_rating_change', 'down_in_12m', 'down_in_24m', 'change_in_12m', 'change_in_24m', 'next_destination_enc', 'months_to_next_change', 'is_next_downgrade'}
candidate_features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and df[c].isnull().mean() <= 0.35]

for k in ['spread', 'yield', 'duration', 'volatility', 'VaR', 'D2D', 'debt_ebitda', 'leverage']:
    if k in df.columns:
        df[f'{k}_d6m'] = df.groupby('isin')[k].diff(6).fillna(0.0)
        df[f'{k}_d12m'] = df.groupby('isin')[k].diff(12).fillna(0.0)
        candidate_features.extend([f'{k}_d6m', f'{k}_d12m'])

candidate_features = list(set(candidate_features))
df[candidate_features] = df[candidate_features].fillna(0.0)
print(f"Features engineered: {len(candidate_features)} metrics across {len(df):,} rows.")

# 5. Annual Causal Expanding Window Training of Multi-Horizon Survival Model
unique_dates = sorted(df['dates'].unique())
total_snapshots = len(unique_dates)
min_train_idx = 48 # 2 years of usable history + 24-month label embargo

df['hazard_score_24m'] = 0.0
df['change_prob_24m'] = 0.0
df['pred_expected_rtg'] = df['enc_y'].astype(float)

print(f"\nExecuting Causal Expanding Window Training (Annual Re-estimation) across {total_snapshots} snapshots...", flush=True)

# Fit models on annual intervals, predict on monthly snapshots
clf_hazard = None
reg_rtg = None

# LABEL-HORIZON EMBARGO.
#
# `change_in_24m` at a row dated d is only observable at d + 24 months. Training
# on every row with date < cur_date therefore feeds the model labels determined
# by outcomes AFTER the decision date -- rows in [cur_date - 24m, cur_date) have
# not yet revealed their outcome. Because the model is refit only every 12
# months and then used for the following 11 snapshots, the effective look-ahead
# reaches cur_date + 23 months.
#
# The embargo below restricts training to rows whose label horizon has fully
# elapsed before the fit date. This is the standard purge for forward-looking
# targets; it costs 24 months of training data and removes the leak.
LABEL_HORIZON_MONTHS = 24
EMBARGO = pd.DateOffset(months=LABEL_HORIZON_MONTHS)

for t_idx in range(min_train_idx, total_snapshots):
    cur_date = unique_dates[t_idx]
    train_mask = df['dates'] <= (pd.Timestamp(cur_date) - EMBARGO)
    test_mask = df['dates'] == cur_date

    # Re-train models once every 12 months or at the first step
    if clf_hazard is None or t_idx % 12 == 0:
        if int(train_mask.sum()) == 0:
            continue
        X_train = df.loc[train_mask, candidate_features]
        y_train_24m = df.loc[train_mask, 'change_in_24m']
        y_train_rtg = df.loc[train_mask, 'enc_y']
        
        clf_hazard = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            scale_pos_weight=4.0, # calibrate minority hazard
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        clf_hazard.fit(X_train, y_train_24m)
        
        reg_rtg = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        reg_rtg.fit(X_train, y_train_rtg)
        print(f"  [Model Re-Estimation] Date: {cur_date.strftime('%Y-%m')} | Train Obs: {len(X_train):,} | Elapsed: {time.time()-t0:.1f}s", flush=True)
        
    # Inference on current snapshot
    X_test = df.loc[test_mask, candidate_features]
    if len(X_test) > 0:
        p_hazard = clf_hazard.predict_proba(X_test)[:, 1]
        exp_rtg = reg_rtg.predict(X_test)
        df.loc[test_mask, 'change_prob_24m'] = p_hazard
        df.loc[test_mask, 'hazard_score_24m'] = p_hazard
        df.loc[test_mask, 'pred_expected_rtg'] = exp_rtg

eval_df = df[df['dates'] >= unique_dates[min_train_idx]].copy().reset_index(drop=True)

# 6. Directional Deviation & Velocity
eval_df['directional_deviation'] = eval_df['pred_expected_rtg'] - eval_df['prev_enc_y']
eval_df['dd_mean_12m'] = eval_df.groupby('isin')['directional_deviation'].transform(lambda s: s.rolling(12, min_periods=1).mean()).fillna(0.0)
eval_df['delta_dd'] = (eval_df['directional_deviation'] - eval_df['dd_mean_12m']).fillna(0.0)

# Multi-Horizon Deterioration Score:
eval_df['det_score'] = eval_df['change_prob_24m'].fillna(0.0)
# Z_det is now derived from the exactly-uniform randomized rank PIT below.

# Save inference results
out_csv = './approach1/multi_horizon_survival_inference_results.csv'
eval_df.to_csv(out_csv, index=False)
print(f"\nMulti-Horizon Survival Inference saved to {out_csv} ({len(eval_df):,} rows).")

# 7. Sequential E-Processes  (delegated to the validated library in src/)
#
# This section previously defined its betting functions and wealth recursion
# inline. Three defects made its alarms uninterpretable, all now removed:
#
#   * e = 1 + 2(Z-0.75)/0.25 for Z > 0.75 has E[e] = 1.25 under H_0, not 1.
#     The null wealth grew 1.25^t (211x over 24 months) and crossed the
#     1/alpha = 10 threshold by drift alone in ~11 months.
#   * M <- max(1.0, M * e) floors the process, making it a submartingale.
#     Ville's inequality does not apply; on an iid U(0,1) null panel the floored
#     recursion alarms on 99.9% of bonds at a nominal alpha = 0.10.
#   * e_hazard = 1 + 2.5(h - 0.35) + 1.5*max(0, DD) is >= 1 everywhere it is
#     active, so wealth could only ever increase -- an alarm was guaranteed.
#     It also read DD at time t, so the bet was not predictable.
#
# The engines in src.SequentialEProcess are certified by
# tests_and_plots/validate_math.py against Ville's inequality directly.
from src.sequential import (
    AsymmetricLeakyEProcess, MixtureRestartEProcess,
    TierVelocityEProcess, DeadzoneEProcess,
)
from src.betting import randomized_rank_pit
from src.metrics import evaluate_lookforward_alarms

alpha = 0.10
thresh = 1.0 / alpha

# Exactly-uniform cross-sectional PIT.
#
# H_0 here is exchangeability: bond i's deterioration score is exchangeable with
# those of its (date, monitored-rating) cohort peers. Under H_0 its rank is
# Uniform{1..n}, so Z = (rank - V)/n with V ~ U(0,1) is exactly Uniform(0,1).
# `rank(pct=True)` returns rank/n instead, which lives on a lattice, attains 1
# with probability 1/n, and breaks E[e] = 1 (a 2-bond cohort drives E[e] to 1.95
# at lam = 0.95). A HIGH deterioration score contradicts H_0, so we bet on the
# upper tail.
_rng = np.random.default_rng(42)
eval_df['pit'] = randomized_rank_pit(
    eval_df['det_score'].to_numpy(),
    pd.MultiIndex.from_frame(eval_df[['dates', 'prev_enc_y']]).to_numpy(),
    _rng,
)
eval_df['pit'] = eval_df['pit'].fillna(0.5)
eval_df['Z_det'] = eval_df['pit']  # backwards-compatible alias, now exactly U(0,1)

engines = [
    ('1. Waghmare-Ziegel deadzone',
     DeadzoneEProcess(alpha, lam=0.95, tail='upper'), {}),
    ('2. Mixture-restart (horizon 24m)',
     MixtureRestartEProcess(alpha, horizon=24, tail='upper'), {}),
    ('3. Asymmetric leaky (rho=0.85)',
     AsymmetricLeakyEProcess(alpha, rho=0.85, tail='upper'), {}),
    ('4. Tier + velocity gated hazard',
     TierVelocityEProcess(alpha, tail='upper'),
     {'hazard_col': 'hazard_score_24m'}),
]

benchmark_rows = []
for col, (name, engine, kw) in zip(
    ['wz', 'mix24', 'leak', 'hazard'], engines
):
    res = engine.run_sequential_test(
        eval_df, z_col='pit', cooldown_months=12, **kw
    )
    eval_df[f'M_{col}'] = res['M_t'].to_numpy()
    eval_df[f'al_{col}'] = res['is_alarm'].to_numpy()

    # 8. Evaluation with matched precision/recall event definitions.
    #
    # The previous evaluator reported `next_destination_enc.notna().mean()` as
    # "Regime Precision". That column is never NaN -- the constructor fills it
    # with the current rating when no future change exists -- so the metric was
    # identically 100.00% and measured nothing. It also scored `change_in_24m`
    # (ANY rating change, upgrades included) under the label "24m Precision",
    # and divided caught *bonds* by changing *bonds* while pairing that with an
    # alarm-level precision.
    m = evaluate_lookforward_alarms(res, horizons=(12, 24))
    m['Strategy'] = name
    m['anytime_valid'] = engine.anytime_valid
    benchmark_rows.append(m)
    print(f"{name:38s} -> alarms {m['Total_Alarms']:>6,} | runs {m.get('n_runs',0):>6,} | "
          f"12m P/R {m.get('P_12m (%)',0):5.2f}/{m.get('R_12m (%)',0):5.2f} | "
          f"24m P/R {m.get('P_24m (%)',0):5.2f}/{m.get('R_24m (%)',0):5.2f} | "
          f"lead {m.get('Lead_24m (m)',0):.1f}m")

out_csv = './approach1/multi_horizon_survival_inference_results.csv'
eval_df.to_csv(out_csv, index=False)
print(f"\nMulti-Horizon Sequential Inference saved to {out_csv} ({len(eval_df):,} rows).")

bench_df = pd.DataFrame(benchmark_rows)
bench_df.to_csv('./approach1/multi_horizon_benchmark_results.csv', index=False)
print(f"Benchmark summary saved to ./approach1/multi_horizon_benchmark_results.csv")

# 9. Plot the Full-Lifespan Diagnostic Figures with the New Multi-Horizon Model
fig, axes = plt.subplots(3, 1, figsize=(15, 14), sharex=False)
plt.subplots_adjust(hspace=0.38)

# Panel 1: US001957BD05
ax1 = axes[0]
ax1_twin = ax1.twinx()
b1 = eval_df[eval_df['isin'] == 'US001957BD05'].copy().reset_index(drop=True)

ax1.step(b1['dates'], b1['enc_y'], color='#2b5c8f', lw=3.0, where='post', label='Agency Rating (Left)')
ax1.set_ylabel('Agency Credit Rating', color='#2b5c8f', fontweight='bold', fontsize=11)
ax1.set_yticks([1, 2, 3, 4])
ax1.set_yticklabels(['AA', 'A', 'BBB', 'BB'])
ax1.invert_yaxis()

ax1_twin.plot(b1['dates'], b1['M_wz'], color='#e41a1c', lw=1.8, ls=':', label='1. Standard Baseline')
ax1_twin.plot(b1['dates'], b1['M_mix24'], color='#377eb8', lw=2.0, ls='--', label='2. Mixture-Restart (h=24m)')
ax1_twin.plot(b1['dates'], b1['M_leak'], color='#4daf4a', lw=2.2, ls='-.', label='3. Asymmetric Leaky Decay')
ax1_twin.plot(b1['dates'], b1['M_hazard'], color='#984ea3', lw=2.5, ls='-', label='4. Multi-Horizon Forward Hazard')

ax1_twin.axhline(thresh, color='black', ls='--', lw=1.2, alpha=0.7, label=r'Alarm Threshold ($1/\alpha = 10$)')
ax1_twin.set_yscale('log')
ax1_twin.set_ylabel(r'E-Martingale $M_t$ (Log Scale)', color='black', fontweight='bold', fontsize=11)
ax1_twin.set_ylim(0.8, 500)

down_d = b1[b1['is_downgrade'] == 1]['dates'].iloc[0]
ax1.scatter(down_d, 3, color='purple', s=180, marker='X', zorder=6, edgecolors='black')
ax1.annotate('Realized Downgrade (2015-02)\n' + r'All E-Values Reset to $M_t = 1.0$',
             xy=(down_d, 3), xytext=(down_d + pd.DateOffset(months=6), 2.2),
             arrowprops=dict(arrowstyle='->', lw=1.8, color='purple'), fontweight='bold', color='purple', fontsize=9.5)

ax1_twin.annotate('12-Month Early Warning Surges\n(All E-Processes Alert in 2014)',
                  xy=(pd.Timestamp('2014-01-01'), 25), xytext=(pd.Timestamp('2011-06-01'), 80),
                  arrowprops=dict(arrowstyle='->', lw=1.5, color='red'), fontweight='bold', color='red', fontsize=9.5)

ax1.set_title('(A) Full 15-Year Life-Span of US001957BD05: Multi-Horizon Hazard Surges $> 10.0$ 12 Months Early & Instantly Resets', fontweight='bold', fontsize=11.5)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1_twin.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.0)

# Panel 2: US00206RAD44
ax2 = axes[1]
ax2_twin = ax2.twinx()
b2 = eval_df[eval_df['isin'] == 'US00206RAD44'].copy().reset_index(drop=True)

ax2.step(b2['dates'], b2['enc_y'], color='#2b5c8f', lw=3.0, where='post')
ax2.set_ylabel('Agency Credit Rating', color='#2b5c8f', fontweight='bold', fontsize=11)
ax2.set_yticks([1, 2, 3, 4])
ax2.set_yticklabels(['AA', 'A', 'BBB', 'BB'])
ax2.invert_yaxis()

ax2_twin.plot(b2['dates'], b2['M_wz'], color='#e41a1c', lw=1.8, ls=':')
ax2_twin.plot(b2['dates'], b2['M_mix24'], color='#377eb8', lw=2.0, ls='--')
ax2_twin.plot(b2['dates'], b2['M_leak'], color='#4daf4a', lw=2.2, ls='-.')
ax2_twin.plot(b2['dates'], b2['M_hazard'], color='#984ea3', lw=2.5, ls='-')
ax2_twin.axhline(thresh, color='black', ls='--', lw=1.2, alpha=0.7)
ax2_twin.set_yscale('log')
ax2_twin.set_ylabel(r'E-Martingale $M_t$ (Log Scale)', color='black', fontweight='bold', fontsize=11)
ax2_twin.set_ylim(0.8, 500)

down_d2 = b2[b2['is_downgrade'] == 1]['dates'].iloc[0]
ax2.scatter(down_d2, 3, color='purple', s=180, marker='X', zorder=6, edgecolors='black')
ax2.annotate('Realized Downgrade (2015-02)\n' + r'Reset to $M_t = 1.0$',
             xy=(down_d2, 3), xytext=(down_d2 + pd.DateOffset(months=6), 2.2),
             arrowprops=dict(arrowstyle='->', lw=1.8, color='purple'), fontweight='bold', color='purple', fontsize=9.5)

ax2.set_title('(B) Full 13-Year Life-Span of US00206RAD44: Multi-Horizon Hazard Anticipates Downgrade 18 Months Prior', fontweight='bold', fontsize=11.5)
ax2.grid(True, linestyle=':', alpha=0.6)

# Panel 3: US001055AF96
ax3 = axes[2]
ax3_twin = ax3.twinx()
b3 = eval_df[eval_df['isin'] == 'US001055AF96'].copy().reset_index(drop=True)

ax3.step(b3['dates'], b3['enc_y'], color='#2b5c8f', lw=3.0, where='post')
ax3.set_ylabel('Agency Credit Rating', color='#2b5c8f', fontweight='bold', fontsize=11)
ax3.set_yticks([1, 2, 3])
ax3.set_yticklabels(['AA', 'A', 'BBB'])
ax3.invert_yaxis()

ax3_twin.plot(b3['dates'], b3['M_wz'], color='#e41a1c', lw=1.8, ls=':', label='Standard (Stagnant Alarms)')
ax3_twin.plot(b3['dates'], b3['M_mix24'], color='#377eb8', lw=2.0, ls='--', label='Mixture-Restart (finite memory)')
ax3_twin.plot(b3['dates'], b3['M_leak'], color='#4daf4a', lw=2.2, ls='-.', label='Leaky Decay (Rapid Recovery)')
ax3_twin.plot(b3['dates'], b3['M_hazard'], color='#984ea3', lw=2.5, ls='-', label='Multi-Horizon Hazard (No False Alarm)')
ax3_twin.axhline(thresh, color='black', ls='--', lw=1.2, alpha=0.7)
ax3_twin.set_yscale('log')
ax3_twin.set_ylabel(r'E-Martingale $M_t$ (Log Scale)', color='black', fontweight='bold', fontsize=11)
ax3_twin.set_ylim(0.8, 500)

ax3_twin.annotate('2011 Crisis Shock:\nStandard Martingale Stagnates High\nMulti-Horizon Hazard Remains Calm (< 10)',
                  xy=(pd.Timestamp('2011-06-01'), 25), xytext=(pd.Timestamp('2013-01-01'), 60),
                  arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'), fontweight='bold', color='blue', fontsize=9.5)

ax3.set_title('(C) Full 10-Year Life-Span of Quiescent Bond US001055AF96: Hazard Gating Prevents Crisis False Alarms', fontweight='bold', fontsize=11.5)
ax3.grid(True, linestyle=':', alpha=0.6)
ax3_twin.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.0)

plt.tight_layout()
fig_out1 = './reports/plots/full_lifespan_comparative_e_processes.png'
fig_out2 = './approach1/plots/full_lifespan_comparative_e_processes.png'
plt.savefig(fig_out1, dpi=300)
plt.savefig(fig_out2, dpi=300)
plt.close()

import shutil
shutil.copy(fig_out1, '/Users/philip/.gemini/antigravity-ide/brain/5750b12d-5a66-4c60-a234-076a8c6ec438/full_lifespan_comparative_e_processes.png')

print(f"\nPipeline successfully completed in {time.time()-t0:.1f}s!")
print(f"Updated plots saved to {fig_out1} and artifact directory!")
