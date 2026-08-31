# Migrated from approach1_tabpfn_bash_only_trans_probs.py (Chapter 4, Section 4.3).
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

TARGET = "rtg"                 # choose "nrtg" or "rtg" as the target variable for prediction
HISTORY_WINDOW_MONTHS = 12       # Set the history window in months  # try 2, 3, 6 as well
MIN_HISTORY_WINDOW_MONTHS = 1   # Set the minimum history window in months


def main():
    try:
        parser = argparse.ArgumentParser(description="Processes a value and yields the next one.")
        parser.add_argument("input_value", type=str, help="The value passed from the bash script")
        args = parser.parse_args()
        current_date = args.input_value
        current_date = np.datetime64(current_date)
    except Exception:
        current_date = np.datetime64('2003-12-01T00:00:00.000000000')

    # import dataset
    dataset = pd.read_csv(BOND_CSV, low_memory=False)
    df = pl.from_pandas(dataset)

    # remove superflueous columns
    useless_cols = ['convertflg']
    id_cols = ['cusip', 'compsym', 'bondsym', 'permno_CORPTBL', 'PERMNO_permno', 'PERMCO_permco', 'id', 'sic']
    df = df.drop(useless_cols + id_cols)
    df = df.rename({"isin": "bond_id"})

    dataset = df.to_pandas()
    dataset["dates"] = pd.to_datetime(dataset["dates"], format="%Y%m")
    dataset.dropna(inplace=True)
    # Ensure chronological order per bond before shifting!
    dataset["dates"] = pd.to_datetime(dataset["dates"], format="%Y%m")
    dataset = dataset.sort_values(["bond_id", "dates"]).reset_index(drop=True)
    # Identify transition events across the whole dataset first (outside the date loop)
    dataset['prev_rtg'] = dataset.groupby('bond_id')[TARGET].shift(1)
    dataset['is_fresh_rating'] = (
        dataset['prev_rtg'].isna() |  # Entry/New bond
        (dataset[TARGET] != dataset['prev_rtg'])  # Rating transition
    )

    dates = np.sort(np.unique(dataset["dates"].values))
    next_date = dates[np.searchsorted(dates, current_date) + 1] if np.searchsorted(dates, current_date) + 1 < len(dates) else "STOP"

    if TARGET == "rtg":
        rating_map = {"AAA":1, "AA":2, "A":3, "BBB":4, "BB":5, "B":6}
    accuracy_list = []  

    date = current_date
    print(f"\nProcessing date: {date}")

    # Get the baseline transition window
    current_data = dataset[dataset["dates"] == date]
    historical_data_window = dataset[
        (dataset["dates"] < date) &
        (dataset["dates"] >= (pd.to_datetime(date) - pd.DateOffset(months=HISTORY_WINDOW_MONTHS))) &
        (dataset["is_fresh_rating"] == True)
    ]
    # Identify all possible classes across the ENTIRE dataset
    all_possible_classes = np.unique(dataset[TARGET].dropna())

    # Identify missing classes in the current window
    present_classes = np.unique(historical_data_window[TARGET])
    missing_classes = set(all_possible_classes) - set(present_classes)

    # Fall back to retrieve the most recent example for missing classes
    fallback_rows = []
    if missing_classes:
        # Look at all historical transitions prior to current_date
        prior_transitions = dataset[
            (dataset["dates"] < date) & 
            (dataset["is_fresh_rating"] == True)
        ]
        
        for missing_cls in missing_classes:
            # Find the most recent transition for this specific missing class
            cls_history = prior_transitions[prior_transitions[TARGET] == missing_cls]
            if not cls_history.empty:
                # Take the single most recent row
                most_recent_row = cls_history.sort_values("dates").iloc[[-1]]
                fallback_rows.append(most_recent_row)

    # Combine into final historical_data
    if fallback_rows:
        fallback_df = pd.concat(fallback_rows, axis=0)
        historical_data = pd.concat([historical_data_window, fallback_df], axis=0).drop_duplicates()
    else:
        historical_data = historical_data_window

    excluded_cols = ['rtg', 'nrtg', 'dates', 'bond_id', 'mom6xrtg', 'nextdate', 'nextret', 'nextretexc', 'nextretwins', 'nextretexcwins', 'prev_rtg', 'is_fresh_rating']
    numeric_df = historical_data.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c not in excluded_cols]

    X_train = historical_data[feature_cols].to_numpy(copy=True)
    y_train = historical_data[TARGET].to_numpy(copy=True)
    X_test = current_data[feature_cols].to_numpy(copy=True)
    y_test = current_data[TARGET].to_numpy(copy=True)
    
    print(f"Data shape -> Train: {X_train.shape}, Test: {X_test.shape}")

    if torch.backends.mps.is_available():
        DEVICE='mps'
    elif torch.cuda.is_available():
        DEVICE='cuda'
    else:
        DEVICE='cpu'

    clf = TabPFNClassifier(device=DEVICE)

    with torch.no_grad():
        clf.fit(X_train, y_train)
        prediction_probabilities = clf.predict_proba(X_test)
        
        pred_indices = np.argmax(prediction_probabilities, axis=1)
        predictions = clf.classes_[pred_indices]
        
        accuracy = accuracy_score(y_test, predictions)
        accuracy_list.append((date, accuracy))
        
    bond_ids_test = current_data["bond_id"].to_numpy(copy=True)

    # --- CHANGED: Format probability columns and ensure alignment ---
    prob_cols = [f"prob_{cls}" for cls in clf.classes_]
    proba_df = pd.DataFrame(prediction_probabilities, columns=prob_cols, index=current_data.index)
    
    if TARGET == "rtg":
        numeric_weights = pd.Series({cls: rating_map[cls] for cls in clf.classes_ if cls in rating_map})
        # Remap the probability columns back to original class names just for the dot product
        expected_ratings = proba_df.rename(columns={f"prob_{cls}": cls for cls in clf.classes_})[numeric_weights.index].dot(numeric_weights)
    else:
        numeric_weights = pd.Series({cls: cls for cls in clf.classes_})
        expected_ratings = proba_df.rename(columns={f"prob_{cls}": cls for cls in clf.classes_})[numeric_weights.index].dot(numeric_weights)
    
    bond_rankings = pd.DataFrame({
        "bond_id": bond_ids_test,
        "actual_rtg": y_test,
        "pred_rtg": predictions,
        "expected_rtg": expected_ratings.to_numpy(copy=True), 
    }, index=current_data.index)
    
    # --- CHANGED: Append the probability dataframe to bond_rankings ---
    bond_rankings = pd.concat([bond_rankings, proba_df], axis=1)
    
    if TARGET == "rtg":
        bond_rankings["actual_numeric"] = bond_rankings["actual_rtg"].map(rating_map)
    else:
        bond_rankings["actual_numeric"] = bond_rankings["actual_rtg"]
    bond_rankings["directional_deviation"] = bond_rankings["expected_rtg"] - bond_rankings["actual_numeric"]

    group_counts = bond_rankings.groupby("actual_rtg")["directional_deviation"].transform('count')
    bond_rankings["within_class_std_rank"] = (
        # ASCENDING: directional_deviation > 0 means the model rates the bond
        # WORSE than its label (1=AAA..6=B), i.e. a downgrade candidate, and
        # such a bond must land near 1 -- the convention the write-up and the
        # one-sided E-values in Chapter 4 assume.  This was ascending=False.
        bond_rankings.groupby("actual_rtg")["directional_deviation"].rank(ascending=True) / group_counts
    )

    # --- CHANGED: Include probability columns in the final slice ---
    current_tab_df = bond_rankings[[
        "bond_id", 
        "date" if "date" in bond_rankings.columns else "bond_id", # Placeholder to keep order, replaced below
        "expected_rtg", 
        "pred_rtg", 
        "actual_rtg", 
        "directional_deviation", 
        "within_class_std_rank"
    ] + prob_cols].copy()
    
    # 2. Add the date column to the new dataframe safely
    # Drop the placeholder first to avoid duplication
    current_tab_df = current_tab_df.loc[:, ~current_tab_df.columns.duplicated()].copy()
    if "date" in current_tab_df.columns:
        current_tab_df = current_tab_df.drop(columns=["date"])
    current_tab_df.insert(1, "date", date) 
    
    # 3. Handle the CSV file check and append
    file_path = f"{OUT_DIR}/tabpfn_data_{TARGET}_{HISTORY_WINDOW_MONTHS}_onlytrans.csv"
    
    file_exists = os.path.isfile(file_path)
    current_tab_df.to_csv(file_path, mode='a', header=not file_exists, index=False)

    # --- MEMORY MANAGEMENT AND CLEANUP ---
    del clf 

    del X_train, y_train, X_test, y_test
    del prediction_probabilities, proba_df, bond_rankings
    del current_data, historical_data, feature_cols
    del expected_ratings, predictions, numeric_weights, group_counts, bond_ids_test
    del pred_indices
    
    gc.collect()
    gc.collect()
        
    if next_date == "STOP":
        print("STOP")
    else:
        print(pd.to_datetime(next_date).strftime('%Y-%m-%d'))

if __name__ == "__main__":
    main()