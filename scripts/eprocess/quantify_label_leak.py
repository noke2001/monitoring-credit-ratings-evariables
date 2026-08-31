"""
Quantifies the label-horizon leak in the multi-horizon hazard model.

    python3 tests_and_plots/quantify_label_leak.py

The hazard target `change_in_24m` at a row dated d is only observable at
d + 24 months. The original training loop used `train_mask = dates < cur_date`,
which includes rows whose 24-month outcome had not yet happened at the decision
date. This script re-runs the walk-forward twice -- once with that mask, once
with a 24-month embargo -- and scores both on the same out-of-sample snapshots.

Approximation notice: this uses sklearn's HistGradientBoostingClassifier and a
32-feature subset, not the full LightGBM configuration, so the absolute AUCs are
not a replication of the thesis pipeline. The leaked-vs-embargoed *gap* measured
on identical data, features and folds is the quantity of interest.
"""
import time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

CSV = "results/multi_horizon_survival_inference_results.csv"
FEATS = ['spread','yield','duration','volatility','VaR','D2D','debt_ebitda','market_lev',
         'ebitda_debt','mom6','age','ret_6_1','ni_me','at_be','gp_at','turn_vol','be_me',
         'coupon','moddurtn','skew','retexc','spr_to_d2d','vixbeta','oper_lvg','sales_at',
         'at_me','eq_dur','rvol_21d','ebitda_sale','sales','assets','totaldebt']
KEEP = ['dates','isin','enc_y','change_in_24m','down_in_24m'] + FEATS

t0=time.time()
df = pd.read_csv(CSV, usecols=lambda c: c in KEEP, low_memory=False)
df['dates']=pd.to_datetime(df['dates'])
feats=[f for f in FEATS if f in df.columns]
df[feats]=df[feats].astype('float32').fillna(0.0)
df=df.sort_values(['dates']).reset_index(drop=True)
print(f"loaded {len(df):,} rows x {len(feats)} features in {time.time()-t0:.0f}s")
print(f"span {df['dates'].min():%Y-%m} .. {df['dates'].max():%Y-%m}")

EMB = pd.DateOffset(months=24)
fit_dates = pd.date_range('2008-01-31', '2017-01-31', freq='12ME')

def run(embargo: bool):
    aucs, ns = [], []
    for fd in fit_dates:
        tr = df['dates'] <= (fd - EMB) if embargo else df['dates'] < fd
        te = (df['dates'] >= fd) & (df['dates'] < fd + pd.DateOffset(months=12))
        if tr.sum() < 5000 or te.sum() < 500: continue
        y = df.loc[tr,'change_in_24m'].to_numpy()
        if len(np.unique(y)) < 2: continue
        m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05,
                                           max_leaf_nodes=31, random_state=42)
        m.fit(df.loc[tr,feats].to_numpy(), y)
        p = m.predict_proba(df.loc[te,feats].to_numpy())[:,1]
        yt = df.loc[te,'down_in_24m'].to_numpy()
        if len(np.unique(yt)) < 2: continue
        aucs.append(roc_auc_score(yt,p)); ns.append(int(te.sum()))
    return np.array(aucs), np.array(ns)

print(f"\n{'fit date':<12}{'leaked AUC':>12}{'embargoed AUC':>15}{'gap':>9}")
a_leak,_ = run(False); a_emb,n = run(True)
for fd,x,y in zip(fit_dates[:len(a_leak)], a_leak, a_emb):
    print(f"{fd:%Y-%m}     {x:>10.4f}{y:>15.4f}{x-y:>9.4f}")
w=n/n.sum()
print(f"\n  leaked    : mean AUC {a_leak.mean():.4f}   obs-weighted {np.sum(a_leak*w):.4f}")
print(f"  embargoed : mean AUC {a_emb.mean():.4f}   obs-weighted {np.sum(a_emb*w):.4f}")
print(f"  LEAK INFLATION: {a_leak.mean()-a_emb.mean():+.4f} AUC")
print(f"  skill above chance: leaked {a_leak.mean()-0.5:.4f} vs embargoed {a_emb.mean()-0.5:.4f}"
      f"  ({(a_leak.mean()-0.5)/max(1e-9,a_emb.mean()-0.5):.2f}x overstated)")
