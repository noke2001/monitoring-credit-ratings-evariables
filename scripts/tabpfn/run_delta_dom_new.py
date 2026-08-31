# Migrated from approach1_tabpfn_bash_delta_fit_probs_dom_new.py (Chapter 4, Section 4.3).
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
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import polars as pl
import torch
from scipy.stats import kstest
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier

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

# --- PIPELINE CONFIGURATION PARAMETERS ---
TARGET = "rtg"                     # Target rating column ("rtg" or "nrtg")
HISTORY_WINDOW_MONTHS = 24         # Historical context window K in months
MIN_HISTORY_WINDOW_MONTHS = 24     # Minimum required history before starting evaluation
MIN_BONDS_PER_DELTA_CLASS = 5      # Minimum context samples required per rating delta category
DOMINANT_CLASS_RATIO = 0.40        # Target ratio of Class 0 in the training set (e.g., 0.40 = 40%)
FRESH_RATING_RATIO = 0.35          # Maximum allowed proportion of market entries (Blue X) in Class 0

if TARGET == "rtg":
    RATING_MAP = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6}
    ALL_TARGET_DELTAS = list(range(-5, 6))
elif TARGET == "nrtg":
    ALL_TARGET_DELTAS = list(range(-15, 16))
else:
    raise ValueError("TARGET must be either 'rtg' or 'nrtg'.")


def get_next_date_str(all_dates, current_date_dt):
    """Finds the next chronological date string or 'STOP' if evaluation is complete."""
    future_dates = all_dates[all_dates > current_date_dt]
    if not future_dates.empty:
        return future_dates.iloc[0].strftime('%Y-%m-%d')
    return "STOP"


def generate_train_data(dataset, current_date_dt, history_window_months, dom_ratio, fresh_ratio, min_bonds_within_class):
    """
    Modular pipeline for building balanced TabPFN training context for date current_date_dt.
    Implements non-dominant sampling, class 0 ratio capping, fresh-entry ratio constraints,
    and prior historical backfilling failsafes.
    """
    hist_start = current_date_dt - pd.DateOffset(months=history_window_months)

    # Historical data strictly prior to current evaluation date t
    full_history = dataset[dataset["dates"] < current_date_dt].copy()
    # Data within the active historic window [t - K, t)
    window_history = dataset[(dataset["dates"] < current_date_dt) & (dataset["dates"] >= hist_start)].copy()

    if window_history.empty:
        return pd.DataFrame(columns=dataset.columns)

    def check_failsafe(full_hist, non_dom_df):
        """
        Step 8: Checks if any rating delta class that previously appeared in the past
        has fewer than min_bonds_within_class in non_dom_df, and backfills from past history.
        """
        # All classes that ever appeared prior to t
        past_classes = full_hist['rating_delta'].unique()
        non_zero_past_classes = [c for c in past_classes if c != 0]

        additional_backfill_rows = []

        for delta_val in non_zero_past_classes:
            existing_count = (non_dom_df['rating_delta'] == delta_val).sum() if not non_dom_df.empty else 0
            needed = min_bonds_within_class - existing_count

            if needed > 0:
                # Candidates prior to the current historic window (t - K)
                prior_pool = full_hist[
                    (full_hist['dates'] < hist_start) & 
                    (full_hist['rating_delta'] == delta_val)
                ]

                # Filter out rows already selected
                if not non_dom_df.empty:
                    prior_pool = prior_pool[~prior_pool.index.isin(non_dom_df.index)]

                if not prior_pool.empty:
                    # Prefer fresh transitions or recent snapshots
                    fresh_candidates = prior_pool[prior_pool['is_fresh_rating'] == True]
                    if len(fresh_candidates) >= needed:
                        sampled = fresh_candidates.sort_values('dates', ascending=False).head(needed)
                    else:
                        sampled = prior_pool.sort_values('dates', ascending=False).head(needed)
                    
                    additional_backfill_rows.append(sampled)

        if additional_backfill_rows:
            backfill_df = pd.concat(additional_backfill_rows, axis=0)
            return pd.concat([non_dom_df, backfill_df], axis=0).drop_duplicates()
        
        return non_dom_df

    def generate_non_dom_data(df_window, full_hist):
        """
        Steps 1 & 2: Construct non-delta_rtg == 0 training candidates.
        Takes fresh rating transitions (Red X) and oldest-half post-transition stable snapshots.
        """
        non_dom_window = df_window[df_window['rating_delta'] != 0].copy()
        if non_dom_window.empty:
            return pd.DataFrame(columns=df_window.columns)

        selected_rows = []

        # Process each bond's segment within the historic window
        for bond_id, b_df in non_dom_window.groupby('bond_id'):
            b_df = b_df.sort_values('dates')
            
            # Identify transition indices
            fresh_indices = b_df[b_df['is_fresh_rating'] == True].index.tolist()

            if not fresh_indices:
                # Bond was already in this non-zero class prior to window start (Step 2)
                n_months = len(b_df)
                take_k = max(1, (n_months + 1) // 2)  # Oldest half
                selected_rows.append(b_df.iloc[:take_k])
            else:
                # Process each spell starting from a fresh transition
                for idx in range(len(fresh_indices)):
                    start_pos = b_df.index.get_loc(fresh_indices[idx])
                    end_pos = b_df.index.get_loc(fresh_indices[idx + 1]) if idx + 1 < len(fresh_indices) else len(b_df)
                    
                    spell = b_df.iloc[start_pos:end_pos]
                    n_months = len(spell)

                    # Step 1: Always add the fresh transition snapshot (Red X)
                    selected_rows.append(spell.iloc[[0]])

                    # Step 2: If stable post-switch, add oldest half of remaining stable spell
                    if n_months > 1:
                        take_k = max(1, (n_months - 1) // 2)
                        selected_rows.append(spell.iloc[1:1 + take_k])

        base_non_dom = pd.concat(selected_rows, axis=0).drop_duplicates() if selected_rows else pd.DataFrame(columns=df_window.columns)
        
        # Step 8: Apply Fail-Safe backfill for minority classes
        final_non_dom = check_failsafe(full_hist, base_non_dom)
        return final_non_dom

    def generate_dom_data(df_window, non_dom_df):
        """
        Steps 3 to 6: Construct delta_rtg == 0 training dataset.
        Enforces Cap = (DomRatio / (1 - DomRatio)) * NonDomSize, adds fresh rating transitions (Red X),
        and samples remaining candidates while capping market entries (Blue X) at FreshRatio.
        """
        non_dom_size = len(non_dom_df)
        if non_dom_size == 0:
            return pd.DataFrame(columns=df_window.columns)

        # Step 3: Compute target Cap size for Class 0
        cap_size = int(np.round((dom_ratio / (1.0 - dom_ratio)) * non_dom_size))
        cap_size = max(min_bonds_within_class, cap_size)

        dom_window = df_window[df_window['rating_delta'] == 0].copy()
        if dom_window.empty:
            return pd.DataFrame(columns=df_window.columns)

        # Separate candidates into Red X (rating transition into class 0), Blue X (market entry), and Static
        # Red X: Transitioned into rating_delta == 0 from another rating
        red_x_df = dom_window[
            (dom_window['is_fresh_rating'] == True) & 
            (dom_window['prev_rtg_num'].notna())
        ].copy()

        # Blue X: Market entry (first appearance of bond)
        blue_x_df = dom_window[
            (dom_window['is_fresh_rating'] == True) & 
            (dom_window['prev_rtg_num'].isna())
        ].copy()

        # Static / Ongoing stable snapshots (Step 4: oldest half of stable spells)
        static_rows = []
        for bond_id, b_df in dom_window.groupby('bond_id'):
            b_df = b_df.sort_values('dates')
            stable_only = b_df[b_df['is_fresh_rating'] == False]
            if not stable_only.empty:
                n_months = len(stable_only)
                take_k = max(1, (n_months + 1) // 2)
                static_rows.append(stable_only.iloc[:take_k])

        static_df = pd.concat(static_rows, axis=0) if static_rows else pd.DataFrame(columns=df_window.columns)

        # Step 5: Always include freshly transitioned bonds into Class 0 (Red X)
        selected_dom_indices = list(red_x_df.index)

        # Remaining capacity to reach Cap
        remaining_cap = cap_size - len(selected_dom_indices)

        if remaining_cap > 0:
            # Step 6: Max allowed market entries (Blue X) based on FreshRatio
            max_blue_x = int(np.floor(cap_size * fresh_ratio))
            
            # Take allowed Blue X samples
            blue_x_to_take = min(len(blue_x_df), max_blue_x, remaining_cap)
            if blue_x_to_take > 0:
                sampled_blue_x = blue_x_df.sort_values('dates', ascending=False).head(blue_x_to_take)
                selected_dom_indices.extend(sampled_blue_x.index.tolist())
                remaining_cap -= blue_x_to_take

            # Fill remaining quota using static stable candidates
            if remaining_cap > 0 and not static_df.empty:
                avail_static = static_df[~static_df.index.isin(selected_dom_indices)]
                if not avail_static.empty:
                    sampled_static = avail_static.sort_values('dates', ascending=False).head(remaining_cap)
                    selected_dom_indices.extend(sampled_static.index.tolist())

        return dom_window.loc[dom_window.index.isin(selected_dom_indices)].copy()

    # --- EXECUTE PIPELINE STEPS ---
    non_dom_train = generate_non_dom_data(window_history, full_history)
    dom_train = generate_dom_data(window_history, non_dom_train)

    # Step 7: Combine non-delta==0 and delta==0 sets
    combined_train = pd.concat([non_dom_train, dom_train], axis=0).drop_duplicates().reset_index(drop=True)
    return combined_train


def main():
    try:
        parser = argparse.ArgumentParser(description="Processes a date step with Delta-based TabPFN fitting.")
        parser.add_argument("input_value", type=str, help="Date passed from bash script (YYYY-MM-DD)")
        args = parser.parse_args()
        current_date_dt = pd.to_datetime(args.input_value)
    except Exception:
        current_date_dt = pd.to_datetime('2003-12-01')

    print(f"\n==========================================")
    print(f"Processing date: {current_date_dt.strftime('%Y-%m-%d')}")
    print(f"Window K = {HISTORY_WINDOW_MONTHS} months | DomRatio = {DOMINANT_CLASS_RATIO} | FreshRatio = {FRESH_RATING_RATIO}")
    print(f"==========================================")

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

    # Strict chronological sorting
    dataset = dataset.sort_values(["bond_id", "dates"]).reset_index(drop=True)

    all_dates = pd.Series(dataset["dates"].unique()).sort_values().reset_index(drop=True)
    next_date_str = get_next_date_str(all_dates, current_date_dt)

    dataset['entry_date'] = dataset.groupby('bond_id')['dates'].transform('min')
    dataset['entry_rtg_num'] = dataset.groupby('bond_id')['rtg_num'].transform('first')

    # Rating Delta relative to entry: Positive = Upgrade, Negative = Downgrade, 0 = Unchanged
    dataset['rating_delta'] = dataset['entry_rtg_num'] - dataset['rtg_num']

    # Mark active transition events (month-over-month rating change or new entry)
    dataset['prev_rtg_num'] = dataset.groupby('bond_id')['rtg_num'].shift(1)
    dataset['is_fresh_rating'] = (
        dataset['prev_rtg_num'].isna() | 
        (dataset['rtg_num'] != dataset['prev_rtg_num'])
    )

    excluded_cols = [
        'rtg', 'nrtg', 'dates', 'bond_id', 'mom6xrtg', 'nextdate', 
        'nextret', 'nextretexc', 'nextretwins', 'nextretexcwins', 
        'prev_rtg_num', 'is_fresh_rating', 'entry_date', 'entry_rtg_num', 
        'rating_delta', 'rtg_num'
    ]
    numeric_df = dataset.select_dtypes(include=[np.number])
    base_feature_cols = [c for c in numeric_df.columns if c not in excluded_cols]

    new_cols_dict = {}
    delta_entry_cols, delta_1m_cols = [], []

    for col in base_feature_cols:
        entry_val = dataset.groupby('bond_id')[col].transform('first')
        entry_col_name = f"{col}_delta_entry"
        new_cols_dict[entry_col_name] = dataset[col] - entry_val
        delta_entry_cols.append(entry_col_name)

        m1_col_name = f"{col}_delta_1m"
        new_cols_dict[m1_col_name] = dataset.groupby('bond_id')[col].diff(1).fillna(0.0)
        delta_1m_cols.append(m1_col_name)

    delta_df = pd.DataFrame(new_cols_dict, index=dataset.index)
    dataset = pd.concat([dataset, delta_df], axis=1)

    all_feature_cols = base_feature_cols + ['entry_rtg_num'] + delta_entry_cols + delta_1m_cols

    min_dataset_date = dataset["dates"].min()
    min_required_date = min_dataset_date + pd.DateOffset(months=MIN_HISTORY_WINDOW_MONTHS)

    if current_date_dt < min_required_date:
        print(f"Skipping date {current_date_dt.strftime('%Y-%m-%d')}: Insufficient history "
              f"(requires at least {MIN_HISTORY_WINDOW_MONTHS} months from dataset start {min_dataset_date.strftime('%Y-%m-%d')}).")
        print(next_date_str)
        return

    historical_data = generate_train_data(
        dataset=dataset,
        current_date_dt=current_date_dt,
        history_window_months=HISTORY_WINDOW_MONTHS,
        dom_ratio=DOMINANT_CLASS_RATIO,
        fresh_ratio=FRESH_RATING_RATIO,
        min_bonds_within_class=MIN_BONDS_PER_DELTA_CLASS
    )

    current_data = dataset[dataset["dates"] == current_date_dt].copy()

    if historical_data.empty or current_data.empty:
        print(f"Skipping date {current_date_dt.strftime('%Y-%m-%d')}: Missing historical or test data.")
        print(next_date_str)
        return

    print(f"Context Rows (Train): {len(historical_data)} | Target Test Rows: {len(current_data)}")
    print("Train Rating Delta Distribution:\n", historical_data['rating_delta'].value_counts().to_dict())

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

    overall_acc = accuracy_score(y_test, predictions)

    # Calculate Transition Month vs Stable Month Accuracy
    trans_mask = (current_data['prev_rtg_num'].notna()) & (current_data['rtg_num'] != current_data['prev_rtg_num'])
    trans_mask_np = trans_mask.to_numpy()

    n_transitions = trans_mask_np.sum()
    n_stable = len(y_test) - n_transitions

    trans_acc = accuracy_score(y_test[trans_mask_np], predictions[trans_mask_np]) if n_transitions > 0 else np.nan
    stable_acc = accuracy_score(y_test[~trans_mask_np], predictions[~trans_mask_np]) if n_stable > 0 else np.nan

    print("----------------------------------------------------------")
    print(f"TabPFN Overall Instantaneous Accuracy   : {overall_acc*100:.2f}%")
    print(f"TabPFN Stable Months Accuracy (No Δ)    : {stable_acc*100:.2f}% ({n_stable} rows)")
    if not np.isnan(trans_acc):
        print(f"TabPFN TRANSITION MONTHS ACCURACY (Δ)   : {trans_acc*100:.2f}% ({n_transitions} transitions)  <-- CRITICAL")
    else:
        print(f"TabPFN TRANSITION MONTHS ACCURACY (Δ)   : No transitions on this date")
    print("----------------------------------------------------------")

    raw_prob_cols = [f"prob_delta_{cls}" for cls in clf.classes_]
    proba_df_raw = pd.DataFrame(prediction_probabilities, columns=raw_prob_cols, index=current_data.index)

    all_prob_cols = [f"prob_delta_{cls}" for cls in ALL_TARGET_DELTAS]
    proba_df = proba_df_raw.reindex(columns=all_prob_cols, fill_value=0.0)

    delta_weights = pd.Series({cls: cls for cls in ALL_TARGET_DELTAS})
    expected_delta = proba_df.rename(columns={f"prob_delta_{cls}": cls for cls in ALL_TARGET_DELTAS})[delta_weights.index].dot(delta_weights)

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

    out_df = pd.DataFrame({
        "bond_id": current_data["bond_id"],
        "date": current_date_dt,
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

    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"tabpfn_delta_fit_dom_{DOMINANT_CLASS_RATIO*100:.0f}_{TARGET}_{HISTORY_WINDOW_MONTHS}m_new.csv")
    file_exists = os.path.isfile(file_path)
    out_df.to_csv(file_path, mode='a', header=not file_exists, index=False)

    print(f"Successfully appended {len(out_df)} bond records to {file_path}")

    # Save summary report metrics
    report_df = pd.DataFrame([{
        "date": current_date_dt.strftime('%Y-%m-%d'),
        "overall_accuracy": overall_acc,
        "stable_accuracy": stable_acc,
        "transition_accuracy": trans_acc,
        "n_total": len(y_test),
        "n_stable": n_stable,
        "n_transitions": n_transitions,
        "train_context_rows": len(historical_data)
    }])

    report_file_path = os.path.join(out_dir, f"tabpfn_training_report_dom_{DOMINANT_CLASS_RATIO*100:.0f}_{TARGET}_{HISTORY_WINDOW_MONTHS}m_new.csv")
    report_file_exists = os.path.isfile(report_file_path)
    report_df.to_csv(report_file_path, mode='a', header=not report_file_exists, index=False)

    print(f"Successfully appended training metrics report to {report_file_path}")

    del clf, X_train, y_train, X_test, y_test, prediction_probabilities, proba_df, out_df, report_df
    gc.collect()

    print(next_date_str)


if __name__ == "__main__":
    main()