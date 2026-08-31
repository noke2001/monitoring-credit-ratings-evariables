import pandas as pd

def main():
    file_path = './CorpBond_Reconciling/corp_jkp_mergedv2.csv'
    print(f"Loading {file_path}...")
    
    # Load required columns
    df = pd.read_csv(file_path, usecols=['isin', 'dates', 'rtg', 'nrtg'])
    
    # Ensure it's sorted by time for each bond to accurately check "next month"
    df['dates'] = pd.to_datetime(df['dates'])
    df = df.sort_values(by=['isin', 'dates'])
    
    # ---------------------------------------------------------
    # PART 1: Overall Bond-Level Statistics
    # ---------------------------------------------------------
    total_bonds = df['isin'].nunique()
    
    # A bond changes rating if it has more than 1 unique rating in its history
    nunique_rtg = df.groupby('isin')['rtg'].nunique()
    bonds_that_change_rtg = (nunique_rtg > 1).sum()
    prob_change_rtg_overall = bonds_that_change_rtg / total_bonds if total_bonds > 0 else 0
    
    nunique_nrtg = df.groupby('isin')['nrtg'].nunique()
    bonds_that_change_nrtg = (nunique_nrtg > 1).sum()
    prob_change_nrtg_overall = bonds_that_change_nrtg / total_bonds if total_bonds > 0 else 0
    
    # ---------------------------------------------------------
    # PART 2: Month-to-Month Transition Statistics
    # ---------------------------------------------------------
    df['next_rtg'] = df.groupby('isin')['rtg'].shift(-1)
    df['next_nrtg'] = df.groupby('isin')['nrtg'].shift(-1)
    
    valid_transitions_rtg = df['next_rtg'].notna()
    valid_transitions_nrtg = df['next_nrtg'].notna()
    
    total_transitions = valid_transitions_rtg.sum()
    
    changes_rtg = (df['rtg'] != df['next_rtg']) & valid_transitions_rtg & df['rtg'].notna()
    changes_nrtg = (df['nrtg'] != df['next_nrtg']) & valid_transitions_nrtg & df['nrtg'].notna()
    
    count_changes_rtg = changes_rtg.sum()
    count_changes_nrtg = changes_nrtg.sum()
    
    prob_change_rtg_next = count_changes_rtg / total_transitions if total_transitions > 0 else 0
    prob_change_nrtg_next = count_changes_nrtg / total_transitions if total_transitions > 0 else 0

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" RATING CHANGE STATISTICS SUMMARY")
    print("="*50)
    print(f"Total Unique Bonds (ISIN): {total_bonds:,}")
    print(f"Total Valid Month-to-Month Transitions: {total_transitions:,}")
    
    print("\n--- 'rtg' (Categorical Rating) ---")
    print(f"  Bonds that EVER change rating: {bonds_that_change_rtg:,}")
    print(f"  Probability a bond EVER changes: {prob_change_rtg_overall:.2%} ({prob_change_rtg_overall:.4f})")
    print(f"  Month-to-month transitions with a change: {count_changes_rtg:,}")
    print(f"  Probability of changing NEXT MONTH: {prob_change_rtg_next:.2%} ({prob_change_rtg_next:.4f})")
    
    print("\n--- 'nrtg' (Numerical Rating) ---")
    print(f"  Bonds that EVER change rating: {bonds_that_change_nrtg:,}")
    print(f"  Probability a bond EVER changes: {prob_change_nrtg_overall:.2%} ({prob_change_nrtg_overall:.4f})")
    print(f"  Month-to-month transitions with a change: {count_changes_nrtg:,}")
    print(f"  Probability of changing NEXT MONTH: {prob_change_nrtg_next:.2%} ({prob_change_nrtg_next:.4f})")
    print("="*50)

if __name__ == "__main__":
    main()
