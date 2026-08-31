# Migrated from approach1_tabpfn_bash_delta_fit_probs.py (Chapter 4, Section 4.3).
# Paths now resolve from the repository root; override with the environment
# variables BOND_CSV (raw panel) and TABPFN_OUT (where per-date fits are
# appended).  Driven one date at a time by scripts/tabpfn/sweep.sh, which feeds
# each run the date printed by the previous one.
import os

# Credentials come from the environment.  Never hard-code them: this file is
# destined for a public repository.
#   export TABPFN_TOKEN=...      (https://tabpfn.com  -> API key)
#   export HF_TOKEN=...          (https://huggingface.co/settings/tokens)
TABPFN_TOKEN = os.environ.get("TABPFN_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN")
if not TABPFN_TOKEN or not HF_TOKEN:
    raise SystemExit(
        "TABPFN_TOKEN and HF_TOKEN must be set in the environment.\n"
        "  export TABPFN_TOKEN=...\n  export HF_TOKEN=...")

#local
import os
os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

# --- PATHS -------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOND_CSV = os.environ.get(
    "BOND_CSV", os.path.join(os.path.dirname(_ROOT),
                             "CorpBond_Reconciling", "corp_jkp_mergedv2.csv"))
OUT_DIR = os.environ.get("TABPFN_OUT", os.path.join(_ROOT, "results", "tabpfnfit"))
os.makedirs(OUT_DIR, exist_ok=True)

# remote
# from tabpfn_client import set_access_token
# set_access_token(TABPFN_TOKEN)

from tabpfn import TabPFNClassifier
import torch
from scipy.stats import kstest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import polars as pl

from datetime import datetime

import tracemalloc
import gc

import sys
import argparse

TARGET = "rtg"                  # choose "nrtg" or "rtg" as the target variable for prediction
HISTORY_WINDOW_MONTHS = 12      # Set the history window in months  # try 12, 24, 36, 48, 1000 as well
MIN_HISTORY_WINDOW_MONTHS = 12  # Set the minimum history window in months (ideally should be equal to HISTORY_WINDOW_MONTHS for this approach)
MIN_BONDS_PER_DELTA_CLASS = 10  # Minimum context rows per rating delta category
if TARGET == "rtg":
    RATING_MAP = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6}
    ALL_TARGET_DELTAS = list(range(-5, 6))
elif TARGET == "nrtg":
    ALL_TARGET_DELTAS = list(range(-15, 16))
else:
    raise ValueError("TARGET must be either 'rtg' or 'nrtg'.")

# Fixed complete spectrum of rating deltas: from -5 (AAA -> B downgrade) to +5 (B -> AAA upgrade)
ALL_TARGET_DELTAS = list(range(-5, 6))

def main():
    try:
        parser = argparse.ArgumentParser(description="Processes a date step with Delta-based TabPFN fitting.")
        parser.add_argument("input_value", type=str, help="Date passed from bash script (YYYY-MM-DD)")
        args = parser.parse_args()
        current_date = np.datetime64(args.input_value)
    except Exception:
        current_date = np.datetime64('2003-12-01T00:00:00.000000000')

    print(f"\n==========================================")
    print(f"Processing date: {current_date}")
    print(f"Window K = {HISTORY_WINDOW_MONTHS} months")
    print(f"==========================================")

    # ---------------------------------------------------------
    # 1. Load and Clean Dataset
    # ---------------------------------------------------------
    raw_path = BOND_CSV
    dataset = pd.read_csv(raw_path, low_memory=False)

    useless_cols = ['convertflg', 'cusip', 'compsym', 'bondsym', 'permno_CORPTBL', 'PERMNO_permno', 'PERMCO_permco', 'id', 'sic']
    dataset = dataset.drop(columns=[c for c in useless_cols if c in dataset.columns])
    dataset = dataset.rename(columns={"isin": "bond_id"})

    dataset["dates"] = pd.to_datetime(dataset["dates"], format="%Y%m")
    dataset.dropna(subset=[TARGET, "dates", "bond_id"], inplace=True)
    if TARGET == "rtg":
        dataset = dataset[dataset[TARGET].isin(RATING_MAP)].copy()
        dataset['rtg_num'] = dataset[TARGET].map(RATING_MAP)
    elif TARGET == "nrtg":
        dataset['rtg_num'] = dataset[TARGET].astype(int)

    # Ensure strict chronological sorting per bond
    dataset = dataset.sort_values(["bond_id", "dates"]).reset_index(drop=True)

    # ---------------------------------------------------------
    # 2. Compute Entry Baseline & Rating Deltas
    # ---------------------------------------------------------
    # Entry rating per bond (first observed rating in dataset)
    dataset['entry_date'] = dataset.groupby('bond_id')['dates'].transform('min')
    dataset['entry_rtg_num'] = dataset.groupby('bond_id')['rtg_num'].transform('first')

    # Rating Delta relative to entry: Positive = Upgrade, Negative = Downgrade, 0 = Unchanged
    # (e.g. Entry A=3 -> Current AA=2 => 3 - 2 = +1 Upgrade)
    dataset['rating_delta'] = dataset['entry_rtg_num'] - dataset['rtg_num']

    # Mark fresh transition events (rating changed from previous month OR new entry)
    dataset['prev_rtg_num'] = dataset.groupby('bond_id')['rtg_num'].shift(1)
    dataset['is_fresh_rating'] = (
        dataset['prev_rtg_num'].isna() | 
        (dataset['rtg_num'] != dataset['prev_rtg_num'])
    )

    # ---------------------------------------------------------
    # 3. Compute Feature Trajectory Deltas from Entry
    # ---------------------------------------------------------
    excluded_cols = [
        'rtg', 'nrtg', 'dates', 'bond_id', 'mom6xrtg', 'nextdate', 
        'nextret', 'nextretexc', 'nextretwins', 'nextretexcwins', 
        'prev_rtg_num', 'is_fresh_rating', 'entry_date', 'entry_rtg_num', 
        'rating_delta', 'rtg_num'
    ]
    numeric_df = dataset.select_dtypes(include=[np.number])
    base_feature_cols = [c for c in numeric_df.columns if c not in excluded_cols]

    # Calculate feature deltas relative to entry month: Delta_X_t = X_t - X_entry
    delta_feature_cols = []
    for col in base_feature_cols:
        entry_val = dataset.groupby('bond_id')[col].transform('first')
        delta_col_name = f"{col}_delta_entry"
        dataset[delta_col_name] = dataset[col] - entry_val
        delta_feature_cols.append(delta_col_name)

    # Combined feature space: Original Financial Features + Initial Rating + Feature Trajectory Deltas
    # Note: 'entry_rtg_num' (the initial rating when bond entered dataset) is explicitly included!
    all_feature_cols = base_feature_cols + ['entry_rtg_num'] + delta_feature_cols

    # ---------------------------------------------------------
    # 4. Construct Context Set X_train (Causal & Non-overlapping)
    # ---------------------------------------------------------
    date = pd.to_datetime(current_date)
    hist_start = date - pd.DateOffset(months=HISTORY_WINDOW_MONTHS)

    # A. Fresh Transitions in Window [t - K, t)
    fresh_window = dataset[
        (dataset["dates"] < date) &
        (dataset["dates"] >= hist_start) &
        (dataset["is_fresh_rating"] == True)
    ].copy()

    # B. Expanded Quiescent Anchor Regime:
    # Bonds with ZERO rating changes in the full window [t - K, t)
    hist_window_all = dataset[
        (dataset["dates"] < date) &
        (dataset["dates"] >= hist_start)
    ]
    transitions_count = hist_window_all.groupby('bond_id')['is_fresh_rating'].sum()
    quiescent_bond_ids = transitions_count[transitions_count == 0].index

    # Take the oldest K/2 monthly snapshots for these fully stable bonds
    half_k = max(1, HISTORY_WINDOW_MONTHS // 2)
    quiescent_full = hist_window_all[hist_window_all['bond_id'].isin(quiescent_bond_ids)].copy()
    
    quiescent_oldest_half = (
        quiescent_full.sort_values(['bond_id', 'dates'])
                      .groupby('bond_id')
                      .head(half_k)
    )

    # Combine baseline context safely without empty DataFrame warnings
    base_frames = [df for df in [fresh_window, quiescent_oldest_half] if not df.empty]
    if base_frames:
        base_historical = pd.concat(base_frames, axis=0).drop_duplicates()
    else:
        base_historical = pd.DataFrame(columns=dataset.columns)

    # C. Minimum Delta Class Enforcement (-5 to +5 full spectrum backfill)
    prior_pool = dataset[
        (dataset["dates"] < date) & 
        (dataset["is_fresh_rating"] == True)
    ].copy()

    additional_rows = []
    for delta_val in ALL_TARGET_DELTAS:
        existing_count = (base_historical['rating_delta'] == delta_val).sum() if not base_historical.empty else 0
        needed = MIN_BONDS_PER_DELTA_CLASS - existing_count
        if needed > 0:
            # First search prior causal history (dates < current_date)
            candidates = prior_pool[prior_pool['rating_delta'] == delta_val]
            if not base_historical.empty:
                candidates = candidates[~candidates.index.isin(base_historical.index)]
            
            # If prior history doesn't have enough rows for rare delta, search full dataset as fallback anchor
            if len(candidates) < needed:
                future_pool = dataset[dataset['rating_delta'] == delta_val]
                if not base_historical.empty:
                    future_pool = future_pool[~future_pool.index.isin(base_historical.index)]
                candidates = pd.concat([candidates, future_pool], axis=0).drop_duplicates()

            if not candidates.empty:
                top_needed = candidates.sort_values("dates", ascending=False).head(needed)
                additional_rows.append(top_needed)

    if additional_rows:
        backfill_df = pd.concat([df for df in additional_rows if not df.empty], axis=0)
        hist_frames = [df for df in [base_historical, backfill_df] if not df.empty]
        historical_data = pd.concat(hist_frames, axis=0).drop_duplicates()
    else:
        historical_data = base_historical

    current_data = dataset[dataset["dates"] == date].copy()

    # ---------------------------------------------------------
    # Check Minimum History Requirement before evaluating TabPFN
    # ---------------------------------------------------------
    min_dataset_date = dataset["dates"].min()
    min_required_date = min_dataset_date + pd.DateOffset(months=MIN_HISTORY_WINDOW_MONTHS)

    if date < min_required_date or historical_data.empty or current_data.empty:
        print(f"Skipping date {date.strftime('%Y-%m-%d')}: Insufficient history "
              f"(requires at least {MIN_HISTORY_WINDOW_MONTHS} months from dataset start {min_dataset_date.strftime('%Y-%m-%d')}).")
        
        # Advance outer bash script runner to next date
        dates_all = np.sort(np.unique(dataset["dates"].values))
        search_idx = np.searchsorted(dates_all, current_date)
        if search_idx + 1 < len(dates_all):
            print(pd.to_datetime(dates_all[search_idx + 1]).strftime('%Y-%m-%d'))
        else:
            print("STOP")
        return

    print(f"Context Rows (Train): {len(historical_data)} | Target Test Rows: {len(current_data)}")
    print("Train Rating Delta Distribution:\n", historical_data['rating_delta'].value_counts().to_dict())
    print("Initial rating 'entry_rtg_num' included in feature space:", 'entry_rtg_num' in all_feature_cols)

    # ---------------------------------------------------------
    # 5. Fit TabPFN and Predict Probability Distributions
    # ---------------------------------------------------------
    X_train = historical_data[all_feature_cols].to_numpy(copy=True)
    y_train = historical_data['rating_delta'].astype(int).to_numpy(copy=True)
    X_test = current_data[all_feature_cols].to_numpy(copy=True)
    y_test = current_data['rating_delta'].astype(int).to_numpy(copy=True)

    if torch.backends.mps.is_available():
        DEVICE = 'mps'
    elif torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        DEVICE = 'cpu'

    clf = TabPFNClassifier(device=DEVICE)

    with torch.no_grad():
        clf.fit(X_train, y_train)
        prediction_probabilities = clf.predict_proba(X_test)
        pred_indices = np.argmax(prediction_probabilities, axis=1)
        predictions = clf.classes_[pred_indices]
        acc = accuracy_score(y_test, predictions)

    print(f"TabPFN Delta-Prediction Instantaneous Accuracy: {acc*100:.2f}%")

    # ---------------------------------------------------------
    # 6. Format Output DataFrame with Predicted Expected Delta & PIT
    # ---------------------------------------------------------
    raw_prob_cols = [f"prob_delta_{cls}" for cls in clf.classes_]
    proba_df_raw = pd.DataFrame(prediction_probabilities, columns=raw_prob_cols, index=current_data.index)

    # Reindex proba_df to guarantee all 11 delta columns (-5 to +5) exist in fixed order for every date
    all_prob_cols = [f"prob_delta_{cls}" for cls in ALL_TARGET_DELTAS]
    proba_df = proba_df_raw.reindex(columns=all_prob_cols, fill_value=0.0)

    # Compute Expected Delta via dot product of predicted probabilities and numeric delta weights
    delta_weights = pd.Series({cls: cls for cls in ALL_TARGET_DELTAS})
    expected_delta = proba_df.rename(columns={f"prob_delta_{cls}": cls for cls in ALL_TARGET_DELTAS})[delta_weights.index].dot(delta_weights)

    # Reconstruct Predicted Rating = Entry Rating - Expected Delta
    current_data['expected_delta'] = expected_delta.to_numpy(copy=True)
    current_data['pred_delta'] = predictions
    current_data['expected_rtg'] = current_data['entry_rtg_num'] - current_data['expected_delta']
    current_data['actual_numeric'] = current_data['rtg_num']
    current_data['directional_deviation'] = current_data['expected_rtg'] - current_data['actual_numeric']

    group_counts = current_data.groupby(TARGET)["directional_deviation"].transform('count')
    current_data["within_class_std_rank"] = (
        # ASCENDING: directional_deviation > 0 means the model rates the bond
        # WORSE than its label (1=AAA..6=B), i.e. a downgrade candidate, and
        # such a bond must land near 1 -- the convention the write-up and the
        # one-sided E-values in Chapter 4 assume.  This was ascending=False.
        current_data.groupby(TARGET)["directional_deviation"].rank(ascending=True) / group_counts
    )

    # Assemble output slice
    out_df = pd.DataFrame({
        "bond_id": current_data["bond_id"],
        "date": date,
        "entry_rtg_num": current_data["entry_rtg_num"],
        "expected_rtg": current_data["expected_rtg"],
        "expected_delta": current_data["expected_delta"],
        "pred_delta": current_data["pred_delta"],
        "actual_rtg": current_data[TARGET],
        "actual_numeric": current_data["actual_numeric"],
        "directional_deviation": current_data["directional_deviation"],
        "within_class_std_rank": current_data["within_class_std_rank"]
    }, index=current_data.index)

    out_df = pd.concat([out_df, proba_df], axis=1)

    # Save results to CSV
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"tabpfn_delta_fit_{TARGET}_{HISTORY_WINDOW_MONTHS}m.csv")
    file_exists = os.path.isfile(file_path)
    out_df.to_csv(file_path, mode='a', header=not file_exists, index=False)

    print(f"Successfully appended {len(out_df)} bond records to {file_path}")

    # Cleanup
    del clf, X_train, y_train, X_test, y_test, prediction_probabilities, proba_df, out_df
    gc.collect()

    # Script control for outer bash runner
    dates_all = np.sort(np.unique(dataset["dates"].values))
    search_idx = np.searchsorted(dates_all, current_date)
    if search_idx + 1 < len(dates_all):
        print(pd.to_datetime(dates_all[search_idx + 1]).strftime('%Y-%m-%d'))
    else:
        print("STOP")


if __name__ == "__main__":
    main()