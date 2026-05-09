"""
Hybrid Anomaly Detection Model — AE-LSTM + Prophet + Isolation Forest.

Implements the hybrid architecture from:
  Ahirwar & Nandanwar (2025) "Enhanced Anomaly Detection in Solar Power
  Plants Using Hybrid Machine Learning Techniques", ICoEIT.

Components:
  1. AELSTM: LSTM autoencoder that learns to reconstruct normal residual
     patterns. Anomalies = high reconstruction error.
  2. IsolationForest: Detects outliers in residual feature space.
  3. HybridScorer: Combines both anomaly scores via weighted averaging.

Usage:
    from src.modeling.anomaly_detection.dl.hybrid_model import AELSTM, HybridScorer

    model = AELSTM(input_dim=1, hidden_dim=64, num_layers=2, latent_dim=16)
    scorer = HybridScorer(
        aelstm_model=model,
        isolation_forest=if_model,
        alpha=0.6,  # weight for AE-LSTM score
    )
    scores = scorer.score(X_res_windows, X_if_features)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class AELSTM(nn.Module):
    """LSTM Autoencoder for residual reconstruction.

    Architecture:
      Encoder: LSTM layers → last hidden state = latent representation
      Decoder: LSTM layers → reconstruct input sequence

    Input:  (batch, seq_len, input_dim)
    Output: (batch, seq_len, input_dim) — reconstruction
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        latent_dim: int = 16,
        dropout: float = 0.3,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.device = device or torch.device("cpu")

        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.latent_expand = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size = x.size(0)

        _, (h_n, c_n) = self.encoder(x)
        h_last = h_n[-1]  # (batch, hidden_dim)

        z = self.latent_proj(h_last)  # (batch, latent_dim)
        h_dec = self.latent_expand(z)  # (batch, hidden_dim)

        h_dec = h_dec.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, batch, hidden_dim)
        c_dec = torch.zeros_like(h_dec, device=self.device)

        seq_len = x.size(1)
        decoder_input = torch.zeros(batch_size, seq_len, self.hidden_dim, device=self.device)
        decoder_output, _ = self.decoder(decoder_input, (h_dec, c_dec))
        reconstruction = self.output_layer(decoder_output)

        return reconstruction

    @torch.no_grad()
    def compute_reconstruction_error(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Compute per-sample MSE reconstruction error.

        Args:
            x: (n_samples, seq_len, input_dim)

        Returns:
            (n_samples,) MSE per sample
        """
        self.eval()
        n = x.shape[0]
        errors = np.zeros(n, dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = torch.from_numpy(x[start:end]).float().to(self.device)
            with torch.no_grad():
                recon = self(batch)
                mse = ((batch - recon) ** 2).mean(dim=(1, 2))
                errors[start:end] = mse.cpu().numpy()

        return errors


class HybridScorer:
    """Combines AE-LSTM and Isolation Forest anomaly scores.

    Score = alpha * aelstm_score + (1 - alpha) * if_score

    Where:
      aelstm_score: normalized [0, 1], higher = more anomalous
      if_score:     normalized [0, 1], higher = more anomalous
      alpha:        weight in [0, 1] (default 0.6)
    """

    def __init__(
        self,
        aelstm_model: AELSTM,
        isolation_forest,  # sklearn IsolationForest
        alpha: float = 0.6,
        device: torch.device | None = None,
    ):
        self.aelstm = aelstm_model
        self.if_model = isolation_forest
        self.alpha = alpha
        self.device = device or aelstm_model.device

        self._aelstm_score_stats: dict[str, float] = {}
        self._if_score_stats: dict[str, float] = {}

    def fit_score_stats(
        self,
        X_res_train: np.ndarray,
        X_if_train: np.ndarray,
    ):
        """Fit min/max stats for normalizing scores.

        Should be called on TRAINING (normal-only) data.
        """
        aelstm_scores = self.aelstm.compute_reconstruction_error(
            X_res_train, batch_size=256,
        )
        if_scores = -self.if_model.decision_function(X_if_train)

        self._aelstm_score_stats = {
            "min": float(aelstm_scores.min()),
            "max": float(aelstm_scores.max()),
        }
        self._if_score_stats = {
            "min": float(if_scores.min()),
            "max": float(if_scores.max()),
        }
        logger.info(
            f"Score stats: AE-LSTM [{self._aelstm_score_stats['min']:.4f}, "
            f"{self._aelstm_score_stats['max']:.4f}] | "
            f"IF [{self._if_score_stats['min']:.4f}, "
            f"{self._if_score_stats['max']:.4f}]"
        )

    def score(self, X_res: np.ndarray, X_if: np.ndarray) -> np.ndarray:
        """Compute ensemble anomaly scores.

        Returns:
            (n_samples,) anomaly scores in [0, 1]
        """
        aelstm_raw = self.aelstm.compute_reconstruction_error(X_res, batch_size=256)
        if_raw = -self.if_model.decision_function(X_if)

        # MinMax normalize to [0, 1]
        a_min, a_max = self._aelstm_score_stats["min"], self._aelstm_score_stats["max"]
        a_range = max(a_max - a_min, 1e-10)
        aelstm_norm = np.clip((aelstm_raw - a_min) / a_range, 0, 1)

        i_min, i_max = self._if_score_stats["min"], self._if_score_stats["max"]
        i_range = max(i_max - i_min, 1e-10)
        if_norm = np.clip((if_raw - i_min) / i_range, 0, 1)

        return self.alpha * aelstm_norm + (1 - self.alpha) * if_norm

    def compute_scores_separately(
        self, X_res: np.ndarray, X_if: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute AE-LSTM, IF, and ensemble scores separately."""
        aelstm_raw = self.aelstm.compute_reconstruction_error(X_res, batch_size=256)
        if_raw = -self.if_model.decision_function(X_if)

        a_min, a_max = self._aelstm_score_stats["min"], self._aelstm_score_stats["max"]
        a_range = max(a_max - a_min, 1e-10)
        aelstm_norm = np.clip((aelstm_raw - a_min) / a_range, 0, 1)

        i_min, i_max = self._if_score_stats["min"], self._if_score_stats["max"]
        i_range = max(i_max - i_min, 1e-10)
        if_norm = np.clip((if_raw - i_min) / i_range, 0, 1)

        ensemble = self.alpha * aelstm_norm + (1 - self.alpha) * if_norm

        return {
            "aelstm": aelstm_norm,
            "isolation_forest": if_norm,
            "ensemble": ensemble,
        }
