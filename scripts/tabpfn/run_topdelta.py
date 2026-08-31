# Migrated from approach1_tabpfn_bash_topdelta_fit.py (Chapter 4, Section 4.3).
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

import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score

# Environment flags for TabPFN execution
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

PRECOMPUTED_PATH = os.path.join(OUT_DIR, "corp_bond_precomputed_deltas.csv")
TOP_FEATURES_PATH = os.path.join(OUT_DIR, "top_features.txt")
OUT_DIR = OUT_DIR

HISTORY_WINDOW_MONTHS = 24       # Context history K
EVAL_FORWARD_HORIZON_MONTHS = 12 # Stop evaluation K months before max dataset date to evaluate true forward horizon
MAX_CONTEXT_ROWS = 10000          # Cap context rows for maximum attention speed & memory efficiency
ALL_TARGET_DELTAS = list(range(-5, 6))

def get_next_date_str(all_dates, current_date_dt):
    future_dates = all_dates[all_dates > current_date_dt]
    if not future_dates.empty:
        return future_dates.iloc[0].strftime('%Y-%m-%d')
    return "STOP"

def cap_context_size(df, max_rows=MAX_CONTEXT_ROWS):
    """Caps context set to max_rows while preserving rating delta class balance."""
    if len(df) <= max_rows:
        return df
    
    classes = df['rating_delta'].unique()
    per_class_limit = max(10, max_rows // len(classes))
    
    sampled_dfs = []
    for cls in classes:
        cls_df = df[df['rating_delta'] == cls]
        fresh_cls = cls_df[cls_df['is_fresh_rating'] == True]
        stable_cls = cls_df[cls_df['is_fresh_rating'] == False]
        
        # Take all fresh transitions first, then sample stable rows
        if len(fresh_cls) >= per_class_limit:
            sampled_dfs.append(fresh_cls.sample(n=per_class_limit, random_state=42))
        else:
            sampled_dfs.append(fresh_cls)
            needed_stable = per_class_limit - len(fresh_cls)
            if len(stable_cls) > 0:
                sampled_dfs.append(stable_cls.sample(n=min(needed_stable, len(stable_cls)), random_state=42))

    res = pd.concat(sampled_dfs, axis=0).drop_duplicates()
    if len(res) > max_rows:
        return res.sample(n=max_rows, random_state=42)
    return res

def main():
    try:
        parser = argparse.ArgumentParser(description="Processes a date step with TabPFN.")
        parser.add_argument("input_value", type=str, help="Date passed from bash script (YYYY-MM-DD)")
        args = parser.parse_args()
        current_date_dt = pd.to_datetime(args.input_value)
    except Exception:
        current_date_dt = pd.to_datetime('2003-12-01')

    dataset = pd.read_csv(PRECOMPUTED_PATH, low_memory=False)
    dataset["dates"] = pd.to_datetime(dataset["dates"])
    all_dates = pd.Series(dataset["dates"].unique()).sort_values().reset_index(drop=True)
    next_date_str = get_next_date_str(all_dates, current_date_dt)

    max_dataset_date = dataset["dates"].max()
    eval_cutoff_date = max_dataset_date - pd.DateOffset(months=EVAL_FORWARD_HORIZON_MONTHS)

    # Stop training/evaluating early so future horizon labels (y_{t+12}) exist for validation
    if current_date_dt > eval_cutoff_date:
        print(f"Reached Evaluation Cutoff Date ({eval_cutoff_date.strftime('%Y-%m-%d')}). "
              f"Stopping early to allow 12-month forward horizon validation.")
        print("STOP")
        return

    # Load top selected features if file exists, else use core numeric defaults
    if os.path.exists(TOP_FEATURES_PATH):
        with open(TOP_FEATURES_PATH, "r") as f:
            feature_cols = [line.strip() for line in f if line.strip() in dataset.columns]
    else:
        feature_cols = ['entry_rtg_num'] + [c for c in dataset.columns if "_delta_" in c][:20]

    hist_start = current_date_dt - pd.DateOffset(months=HISTORY_WINDOW_MONTHS)
    
    # Context set: Data within [t - K, t)
    hist_window = dataset[(dataset["dates"] < current_date_dt) & (dataset["dates"] >= hist_start)].copy()
    
    # Cap context set to ~1,000 rows to ensure fast Transformer self-attention
    historical_data = cap_context_size(hist_window, max_rows=MAX_CONTEXT_ROWS)
    current_data = dataset[dataset["dates"] == current_date_dt].copy()

    if historical_data.empty or current_data.empty:
        print(f"Skipping date {current_date_dt.strftime('%Y-%m-%d')}: Empty context or test data.")
        print(next_date_str)
        return

    NUM_FEATS = len(feature_cols)

    print(f"\nProcessing date: {current_date_dt.strftime('%Y-%m-%d')}")
    print(f"Context Rows (Train): {len(historical_data)} | Target Test Rows: {len(current_data)} | Features: {NUM_FEATS}")

    X_train = historical_data[feature_cols].to_numpy(copy=True)
    y_train = historical_data['rating_delta'].astype(int).to_numpy(copy=True)
    X_test = current_data[feature_cols].to_numpy(copy=True)
    y_test_instant = current_data['rating_delta'].astype(int).to_numpy(copy=True)
    y_test_fw12m = current_data['delta_fw12m'].fillna(0).astype(int).to_numpy(copy=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    clf = TabPFNClassifier(device=DEVICE)

    with torch.no_grad():
        clf.fit(X_train, y_train)
        prediction_probabilities = clf.predict_proba(X_test)
        pred_indices = np.argmax(prediction_probabilities, axis=1)
        predictions = clf.classes_[pred_indices]

    acc_instant = accuracy_score(y_test_instant, predictions)
    acc_horizon_12m = accuracy_score(y_test_fw12m, predictions)

    print(f"Instantaneous Accuracy (t): {acc_instant*100:.2f}%")
    print(f"Forward Horizon Accuracy (t+12m): {acc_horizon_12m*100:.2f}%")

    # Format probability columns across all 11 possible target deltas
    all_prob_cols = [f"prob_delta_{cls}" for cls in ALL_TARGET_DELTAS]
    raw_prob_cols = [f"prob_delta_{cls}" for cls in clf.classes_]
    proba_df_raw = pd.DataFrame(prediction_probabilities, columns=raw_prob_cols, index=current_data.index)
    proba_df = proba_df_raw.reindex(columns=all_prob_cols, fill_value=0.0)

    delta_weights = pd.Series({cls: cls for cls in ALL_TARGET_DELTAS})
    expected_delta = proba_df.rename(columns={f"prob_delta_{cls}": cls for cls in ALL_TARGET_DELTAS})[delta_weights.index].dot(delta_weights)

    current_data['expected_delta'] = expected_delta.to_numpy(copy=True)
    current_data['pred_delta'] = predictions
    current_data['expected_rtg'] = current_data['entry_rtg_num'] - current_data['expected_delta']

    out_df = pd.DataFrame({
        "bond_id": current_data["bond_id"],
        "date": current_date_dt,
        "entry_rtg_num": current_data["entry_rtg_num"],
        "expected_rtg": current_data["expected_rtg"],
        "expected_delta": current_data["expected_delta"],
        "pred_delta": current_data["pred_delta"],
        "actual_rtg": current_data["rtg"],
        "actual_numeric": current_data["rtg_num"],
        "actual_fw12m_numeric": current_data["rtg_num_fw12m"],
        "delta_fw12m": current_data["delta_fw12m"]
    }, index=current_data.index)

    out_df = pd.concat([out_df, proba_df], axis=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    file_path = os.path.join(OUT_DIR, f"tabpfn_horizon_fit_results_{MAX_CONTEXT_ROWS}_{HISTORY_WINDOW_MONTHS}_{NUM_FEATS}.csv")
    file_exists = os.path.isfile(file_path)
    out_df.to_csv(file_path, mode='a', header=not file_exists, index=False)

    print(f"Appended {len(out_df)} bond records to {file_path}")

    del clf, X_train, y_train, X_test, y_test_instant, y_test_fw12m, prediction_probabilities, proba_df, out_df
    gc.collect()

    print(next_date_str)

if __name__ == "__main__":
    main()