"""
Enhanced Causal Credit Rating Prediction Pipeline
Approach 1: Multi-Task Supervised Ordinal VAE + Rolling Gradient Boosted Tabular Engine
Master's Thesis in Financial Machine Learning & Sequential Testing
"""

import os
import sys
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Rating order definitions
RATING_ORDER = {'AAA': 0, 'AA': 1, 'A': 2, 'BBB': 3, 'BB': 4, 'B': 5}
ORDERED_CLASSES = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B']

# Columns to exclude from feature matrix
EXCLUDE_COLUMNS = {
    'isin', 'cusip', 'bondsym', 'compsym', 'dates', 'date', 'rtg', 'nrtg', 
    'enc_y', 'nextdate', 'nextret', 'nextretexc', 'nextretwins', 'nextretexcwins', 
    'convertflg', 'mom6xrtg', 'permno_CORPTBL', 'PERMNO_permno', 'PERMCO_permco', 'id', 'sic'
}


# ==============================================================================
# 1. MODEL ARCHITECTURE: MULTI-TASK SUPERVISED ORDINAL VAE
# ==============================================================================

class MultiTaskOrdinalVAE(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 6, latent_dim: int = 8, hidden_dim: int = 64):
        super(MultiTaskOrdinalVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * torch.clamp(logvar, min=-10.0, max=5.0))
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        logits = self.classifier(mu)
        return recon_x, mu, logvar, logits


def select_clean_features(df: pd.DataFrame, max_missing_ratio: float = 0.30) -> list:
    candidate_cols = []
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in EXCLUDE_COLUMNS or col in EXCLUDE_COLUMNS:
            continue
        if 'next' in col_lower or 'ret' in col_lower or 'id' in col_lower:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            missing_ratio = df[col].isnull().mean()
            if missing_ratio <= max_missing_ratio:
                candidate_cols.append(col)
    return candidate_cols


def run_enhanced_sequential_pipeline(
    raw_data_path: str = './CorpBond_Reconciling/corp_jkp_mergedv2.csv',
    output_csv_path: str = './results/enhanced_sequential_inference_results.csv',
    lookback_window: int = 24
):
    t0 = time.time()
    print("=======================================================")
    print("   STARTING ENHANCED APPROACH 1 CAUSAL INFERENCE PIPELINE")
    print("=======================================================")
    print(f"Loading raw data from: {raw_data_path}")
    raw_data = pd.read_csv(raw_data_path, low_memory=False)

    raw_data['dates'] = pd.to_datetime(raw_data['dates'].astype(str), format='%Y%m', errors='coerce')
    df = raw_data[raw_data['rtg'].isin(ORDERED_CLASSES)].copy()
    df['enc_y'] = df['rtg'].map(RATING_ORDER)
    df['isin'] = df['isin'].fillna(df['cusip'] if 'cusip' in df.columns else df['bond_id'])
    
    feature_cols = select_clean_features(df, max_missing_ratio=0.30)
    df = df.dropna(subset=['dates', 'isin', 'enc_y']).sort_values(['dates', 'isin']).reset_index(drop=True)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    unique_dates = sorted(df['dates'].unique())
    total_snapshots = len(unique_dates)
    print(f"Dataset: {len(df):,} bond-months, {df['isin'].nunique():,} unique ISINs")
    print(f"Features: {len(feature_cols)} financial metrics across {total_snapshots} monthly snapshots")
    print(f"Rolling historical lookback: {lookback_window} months in F_{{t-1}}\n")

    lgb_params = {
        'objective': 'multiclass',
        'num_class': 6,
        'learning_rate': 0.08,
        'num_leaves': 31,
        'min_data_in_leaf': 25,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': 4,
        'random_state': 42
    }

    input_dim = len(feature_cols)
    vae_model = MultiTaskOrdinalVAE(input_dim=input_dim, num_classes=6, latent_dim=8, hidden_dim=64)
    vae_optimizer = optim.AdamW(vae_model.parameters(), lr=1e-3, weight_decay=1e-4)

    all_records = []
    lgb_model = None

    for t_idx in range(lookback_window, total_snapshots):
        cur_date = unique_dates[t_idx]
        train_start_date = unique_dates[t_idx - lookback_window]
        train_end_date = unique_dates[t_idx - 1]

        train_mask = (df['dates'] >= train_start_date) & (df['dates'] <= train_end_date)
        test_mask = df['dates'] == cur_date

        train_data = df[train_mask]
        test_data = df[test_mask].copy().reset_index(drop=True)

        X_train_raw = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data['enc_y'].values.astype(np.int64)
        X_test_raw = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data['enc_y'].values.astype(np.int64)

        # Scale features for PyTorch stability
        scaler = StandardScaler()
        X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train_raw), nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = np.nan_to_num(scaler.transform(X_test_raw), nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Rolling LightGBM Fit (updated every 3 months for high stability & speed)
        if lgb_model is None or t_idx % 3 == 0:
            train_ds = lgb.Dataset(X_train_raw, label=y_train)
            lgb_model = lgb.train(lgb_params, train_ds, num_boost_round=50)

        probs_lgb = lgb_model.predict(X_test_raw)

        # 2. Multi-Task VAE step
        epochs = 8 if t_idx == lookback_window else 2
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train)), batch_size=256, shuffle=True)
        
        vae_model.train()
        for _ in range(epochs):
            for bx, by in train_loader:
                if bx.size(0) <= 2:
                    continue
                vae_optimizer.zero_grad()
                rx, mu, lv, lg = vae_model(bx)
                recon_l = nn.functional.mse_loss(rx, bx, reduction='none').sum(dim=1).mean()
                kl_l = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
                ce_l = nn.functional.cross_entropy(lg, by)
                loss = recon_l + 0.05 * kl_l + 2.0 * ce_l
                loss.backward()
                vae_optimizer.step()

        vae_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            rx_test, mu_test, _, lg_test = vae_model(X_test_tensor)
            recon_errs = torch.mean((rx_test - X_test_tensor)**2, dim=1).numpy()
            probs_vae = torch.softmax(lg_test, dim=1).numpy()
            latent_z = mu_test.numpy()

        # Calibrated ensemble
        final_probs = 0.75 * probs_lgb + 0.25 * probs_vae

        # Compute Directional Deviation and Downgrade Tail Probability
        enc_vec = np.arange(6)
        expected_rtg = np.dot(final_probs, enc_vec)
        directional_deviation = expected_rtg - y_test

        down_tail_prob = np.zeros(len(test_data))
        for i in range(len(test_data)):
            c = y_test[i]
            if c < 5:
                down_tail_prob[i] = np.sum(final_probs[i, c+1:])

        # Randomized PIT
        cum_probs = np.cumsum(final_probs, axis=1)
        f_lower = np.zeros(len(test_data))
        valid_mask = y_test > 0
        f_lower[valid_mask] = cum_probs[np.arange(len(test_data))[valid_mask], y_test[valid_mask] - 1]
        p_realized = final_probs[np.arange(len(test_data)), y_test]
        v_rand = np.random.uniform(0.0, 1.0, size=len(test_data))
        randomized_pit = f_lower + v_rand * p_realized

        test_data['anomaly_score'] = recon_errs
        test_data['directional_deviation'] = directional_deviation
        test_data['downgrade_tail_prob'] = down_tail_prob
        test_data['randomized_pit'] = randomized_pit

        for k in range(min(4, latent_z.shape[1])):
            test_data[f'latent_z_{k}'] = latent_z[:, k]

        for c_idx, c_name in enumerate(ORDERED_CLASSES):
            test_data[f'prob_class_{c_name}'] = final_probs[:, c_idx]

        all_records.append(test_data)

        if (t_idx + 1) % 24 == 0 or t_idx == total_snapshots - 1:
            print(f"  Processed Snapshot: {cur_date.strftime('%Y-%m')} [{t_idx + 1}/{total_snapshots}] | DD mean: {np.mean(directional_deviation):.3f} | Elapsed: {time.time()-t0:.1f}s")

    out_df = pd.concat(all_records, ignore_index=True)
    base_cols = ['isin', 'dates', 'rtg', 'enc_y', 'anomaly_score', 'directional_deviation', 'downgrade_tail_prob', 'randomized_pit']
    latent_cols = [f'latent_z_{k}' for k in range(4)]
    prob_cols = [f'prob_class_{c}' for c in ORDERED_CLASSES]
    final_cols = base_cols + latent_cols + prob_cols

    out_df = out_df[final_cols]
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    out_df.to_csv('./results/sequential_f_t_minus_1_vae_results.csv', index=False)

    print("\n=======================================================")
    print("Pipeline Complete! Causal inference results saved to:")
    print(f"  - {output_csv_path}")
    print("  - ./results/sequential_f_t_minus_1_vae_results.csv")
    print(f"Total Rows: {len(out_df):,} | Unique Bonds: {out_df['isin'].nunique():,}")
    print(f"Total Elapsed Time: {time.time()-t0:.1f}s")
    print("=======================================================\n")
    return out_df


if __name__ == '__main__':
    run_enhanced_sequential_pipeline()
