"""
Unsupervised & Semi-Supervised Neural Architectures for Corporate Credit Risk
Master's Thesis: Monitoring Credit Ratings with E-Variables
Includes:
  - SemiSupervisedFTVAE: Multi-Task Forward-Transition Guided VAE
  - TabularVAE: Static tabular VAE for cross-sectional feature compression
  - LSTMVAE: Temporal windowed LSTM-VAE for longitudinal sequence encoding
"""

import torch
import torch.nn as nn


class SemiSupervisedFTVAE(nn.Module):
    """
    Multi-Task Forward-Transition Guided Variational Autoencoder (FT-VAE).
    
    Jointly optimizes:
      1. Feature Reconstruction Loss: L_recon(x_t)
      2. Latent Regularization: beta * L_KL(z_t)
      3. Forward Transition Cross-Entropy: gamma * L_CE(y_{t+K} | z_t)
      
    This structures the latent space z around forward credit deterioration,
    ensuring reconstruction errors and latent embeddings reflect genuine insolvency risks.
    """
    def __init__(self, input_dim: int, num_classes: int = 6, latent_dim: int = 8, hidden_dim: int = 64):
        super(SemiSupervisedFTVAE, self).__init__()
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

        # Decoder (Reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

        # Auxiliary Transition Classifier Head
        self.transition_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        logits_transition = self.transition_classifier(z)
        return recon_x, mu, logvar, logits_transition


class TabularVAE(nn.Module):
    """
    Static Tabular VAE for Cross-Sectional Snapshots.
    Compresses financial ratios into latent embeddings z and reconstruction anomalies.
    """
    def __init__(self, input_dim: int, latent_dim: int = 4, hidden_dim: int = 32):
        super(TabularVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class LSTMVAE(nn.Module):
    """
    Temporal Windowed LSTM-VAE for Longitudinal Corporate Financial Time-Series.
    """
    def __init__(self, input_dim: int, latent_dim: int = 4, hidden_dim: int = 32, num_layers: int = 1):
        super(LSTMVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # Encoder: LSTM -> Latent
        self.lstm_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: Latent -> LSTM -> Output
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        _, (h_n, _) = self.lstm_encoder(x)
        h_last = h_n[-1]
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        h = self.latent_to_hidden(z).unsqueeze(1)
        h_repeated = h.repeat(1, seq_len, 1)
        lstm_out, _ = self.lstm_decoder(h_repeated)
        recon_x = self.fc_out(lstm_out)
        return recon_x

    def forward(self, x):
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, seq_len)
        return recon_x, mu, logvar