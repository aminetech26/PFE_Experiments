"""
Sequential Conditional Variational Autoencoder (SCVAE) for PV anomaly detection.

Implements the architecture from:
  Li et al. (2024) "Sensing anomaly of photovoltaic systems with sequential
  conditional variational autoencoder", Applied Energy 353:122124.

Architecture:
  - Conditional prior:    p(z_t | x_t, h_{t-1})
  - Inference (encoder):  q(z_t | x_t, y_t, h_{t-1})
  - Generative (decoder): p(y_t | z_t, x_t, h_{t-1})
  - Recurrence:           h_t = GRU(cat[φ_x(x_t), φ_y(y_t), φ_z(z_t)], h_{t-1})
  - Prediction pathway:   separate GRU for test-time (only x available)

Target: reconstruct PV power (pdc1, pdc2) conditioned on environmental
measurements (irr, pvt), capturing temporal and conditional dependencies.

Usage:
    model = SCVAE(x_dim=2, label_dim=2, h_dim=512, z_dim=128)
    # Training: call model(X, Y) then use model.kld_loss, model.nll_loss, etc.
    # Test-time reconstruction: model.reconstruct(X, Y) -> (mu, std, score)
    # Test-time prediction (no Y): model.predict(X) -> (mu, std)
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.distributions import Normal


class SCVAE(nn.Module):
    def __init__(
        self,
        x_dim: int,
        label_dim: int,
        h_dim: int = 512,
        z_dim: int = 128,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.label_dim = label_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.device = device or torch.device("cpu")

        # Feature extractors
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.phi_y = nn.Sequential(
            nn.Linear(label_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

        # Encoder (inference): q(z_t | x_t, y_t, h_{t-1})
        self.enc = nn.Sequential(
            nn.Linear(h_dim * 3, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        # Prior: p(z_t | x_t, h_{t-1})
        self.prior = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        # Prediction prior: p_pred(z_t | x_t, h2_{t-1})  (separate GRU state)
        self.predict_z = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.predict_mean = nn.Linear(h_dim, z_dim)
        self.predict_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        # Decoder: p(y_t | z_t, x_t, h_{t-1})
        self.dec = nn.Sequential(
            nn.Linear(h_dim * 3, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.dec_prior = nn.Sequential(
            nn.Linear(h_dim * 3, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.dec_predict = nn.Sequential(
            nn.Linear(h_dim * 3, h_dim), nn.ReLU(),
            nn.Linear(h_dim, h_dim), nn.ReLU(),
        )
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, label_dim),
        )
        self.dec_std = nn.Sequential(nn.Linear(h_dim, label_dim), nn.Softplus())

        # Recurrence modules
        self.rnn = nn.GRUCell(h_dim * 2, h_dim)   # reconstruction path
        self.rnn2 = nn.GRUCell(h_dim * 2, h_dim)  # prediction path

    # ------------------------------------------------------------------
    # Core recurrence step
    # ------------------------------------------------------------------
    def _recurrence_step(self, x_t, y_t, h, h2):
        phi_x_t = self.phi_x(x_t)
        phi_y_t = self.phi_y(y_t)

        # --- Encoder (inference) ---
        enc_in = torch.cat([phi_x_t, phi_y_t, h], dim=1)
        enc_out = self.enc(enc_in)
        enc_mean_t = self.enc_mean(enc_out)
        enc_std_t = self.enc_std(enc_out)

        # --- Prior ---
        prior_in = torch.cat([phi_x_t, h], dim=1)
        prior_out = self.prior(prior_in)
        prior_mean_t = self.prior_mean(prior_out)
        prior_std_t = self.prior_std(prior_out)

        # --- Prediction prior ---
        pred_in = torch.cat([phi_x_t, h2], dim=1)
        pred_out = self.predict_z(pred_in)
        predict_mean_t = self.predict_mean(pred_out)
        predict_std_t = self.predict_std(pred_out)

        # --- Reparameterized samples ---
        z_t = self._reparameterize(enc_mean_t, enc_std_t)
        z_t_prior = self._reparameterize(prior_mean_t, prior_std_t)
        z_t_predict = self._reparameterize(predict_mean_t, predict_std_t)

        phi_z_t = self.phi_z(z_t)
        phi_z_t_prior = self.phi_z(z_t_prior)
        phi_z_t_predict = self.phi_z(z_t_predict)

        # --- Decoder (reconstruction) --- conditions on x_t, z_t, h
        dec_in = torch.cat([phi_x_t, phi_z_t, h], dim=1)
        dec_out = self.dec(dec_in)
        dec_mean_t = self.dec_mean(dec_out)
        dec_std_t = self.dec_std(dec_out)

        # --- Decoder (prior) --- conditions on x_t, z_t_prior, h
        dec_p_in = torch.cat([phi_x_t, phi_z_t_prior, h], dim=1)
        dec_p_out = self.dec_prior(dec_p_in)
        dec_mean_t_prior = self.dec_mean(dec_p_out)
        dec_std_t_prior = self.dec_std(dec_p_out)

        # --- Decoder (predict) --- conditions on x_t, z_t_predict, h2
        dec_pr_in = torch.cat([phi_x_t, phi_z_t_predict, h2], dim=1)
        dec_pr_out = self.dec_predict(dec_pr_in)
        dec_mean_t_predict = self.dec_mean(dec_pr_out)
        dec_std_t_predict = self.dec_std(dec_pr_out)

        # --- Store intermediates ---
        self.Z_mean.append(enc_mean_t)
        self.Z_std.append(enc_std_t)
        self.Xr_mean.append(dec_mean_t)
        self.Xr_std.append(dec_std_t)

        self.pZ_mean.append(prior_mean_t)
        self.pZ_std.append(prior_std_t)
        self.Xr_mean_prior.append(dec_mean_t_prior)
        self.Xr_std_prior.append(dec_std_t_prior)

        self.Z_mean_predict.append(predict_mean_t)
        self.Z_std_predict.append(predict_std_t)
        self.Xr_mean_predict.append(dec_mean_t_predict)
        self.Xr_std_predict.append(dec_std_t_predict)

        self.h_chain.append(h)
        self.h2_chain.append(h2)

        # --- Update hidden states ---
        h_new = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1), h)
        h2_new = self.rnn2(torch.cat([phi_x_t, phi_z_t_predict], dim=1), h2)
        return h_new, h2_new

    # ------------------------------------------------------------------
    # Forward pass (training)
    # ------------------------------------------------------------------
    def forward(self, X, Y):
        # X: (seq_len, batch, x_dim)
        # Y: (seq_len, batch, label_dim)
        self._reset_variables()
        h = torch.zeros(X.shape[1], self.h_dim, device=self.device)
        h2 = torch.zeros(X.shape[1], self.h_dim, device=self.device)

        for t in range(X.shape[0]):
            h, h2 = self._recurrence_step(X[t], Y[t], h, h2)

        self.kld_loss, self.nll_loss, self.smooth_loss, \
            self.kld_loss_predict, self.nll_loss_prior, \
            self.nll_loss_predict, self.smooth_loss_prior = self._calc_loss(Y)

    # ------------------------------------------------------------------
    # Reconstruction (test-time, uses both X and Y)
    # ------------------------------------------------------------------
    def reconstruct(self, X, Y, n_mc: int = 1):
        """
        Reconstruct Y from X and Y using the encoder pathway.

        Returns:
            mu:    (seq_len, batch, label_dim, n_mc) -> averaged over MC
            std:   (seq_len, batch, label_dim, n_mc) -> averaged over MC
            score: (seq_len, batch, label_dim, n_mc) -> NLL score per timestep
        """
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
        seq_len, batch_size = X_np.shape[0], X_np.shape[1]

        mu_chain = np.zeros((seq_len, batch_size, self.label_dim, n_mc))
        std_chain = np.zeros((seq_len, batch_size, self.label_dim, n_mc))
        score_chain = np.zeros((seq_len, batch_size, self.label_dim, n_mc))

        for i in range(n_mc):
            h = torch.zeros(batch_size, self.h_dim, device=self.device)
            for t in range(seq_len):
                x_t = torch.as_tensor(X_np[t], dtype=torch.float32, device=self.device)
                y_t = torch.as_tensor(Y_np[t], dtype=torch.float32, device=self.device)
                y_t_flat = y_t

                phi_x_t = self.phi_x(x_t)
                phi_y_t = self.phi_y(y_t)

                enc_out = self.enc(torch.cat([phi_x_t, phi_y_t, h], dim=1))
                enc_mean_t = self.enc_mean(enc_out)

                phi_z_t = self.phi_z(enc_mean_t)
                dec_out = self.dec(torch.cat([phi_x_t, phi_z_t, h], dim=1))
                dec_mean_t = self.dec_mean(dec_out)
                dec_std_t = self.dec_std(dec_out)

                mu_chain[t, :, :, i] = dec_mean_t.detach().cpu().numpy()
                std_chain[t, :, :, i] = dec_std_t.detach().cpu().numpy()

                nll = -stats.norm.logpdf(
                    y_t_flat.cpu().numpy(),
                    loc=dec_mean_t.detach().cpu().numpy(),
                    scale=dec_std_t.detach().cpu().numpy(),
                )
                score_chain[t, :, :, i] = nll

                h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1), h)

        return mu_chain.mean(axis=3), std_chain.mean(axis=3), score_chain.mean(axis=3)

    # ------------------------------------------------------------------
    # Prediction (test-time, uses only X — no Y available)
    # ------------------------------------------------------------------
    def predict(self, X, n_mc: int = 1):
        """
        Predict Y from X only (using the prediction pathway).

        Returns:
            mu:  (seq_len, batch, label_dim, n_mc) -> averaged
            std: (seq_len, batch, label_dim, n_mc) -> averaged
        """
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        seq_len, batch_size = X_np.shape[0], X_np.shape[1]

        mu_chain = np.zeros((seq_len, batch_size, self.label_dim, n_mc))
        std_chain = np.zeros((seq_len, batch_size, self.label_dim, n_mc))

        for i in range(n_mc):
            h2 = torch.zeros(batch_size, self.h_dim, device=self.device)
            for t in range(seq_len):
                x_t = torch.as_tensor(X_np[t], dtype=torch.float32, device=self.device)
                phi_x_t = self.phi_x(x_t)

                pred_out = self.predict_z(torch.cat([phi_x_t, h2], dim=1))
                pred_mean_t = self.predict_mean(pred_out)

                phi_z_t = self.phi_z(pred_mean_t)
                dec_out = self.dec_predict(torch.cat([phi_x_t, phi_z_t, h2], dim=1))
                dec_mean_t = self.dec_mean(dec_out)
                dec_std_t = self.dec_std(dec_out)

                mu_chain[t, :, :, i] = dec_mean_t.detach().cpu().numpy()
                std_chain[t, :, :, i] = dec_std_t.detach().cpu().numpy()

                h2 = self.rnn2(torch.cat([phi_x_t, phi_z_t], dim=1), h2)

        return mu_chain.mean(axis=3), std_chain.mean(axis=3)

    # ------------------------------------------------------------------
    # Prediction with labels (evaluates NLL score)
    # ------------------------------------------------------------------
    def predict_with_label(self, X, Y, n_mc: int = 1):
        """
        Predict Y from X only (prediction pathway) and compute NLL score
        against the true Y.

        Returns:
            mu, std, score — each shape (seq_len, batch, label_dim)
        """
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
        seq_len, batch_size = X_np.shape[0], X_np.shape[1]

        mu_chain = np.zeros((seq_len, batch_size, self.label_dim, n_mc))
        std_chain = np.zeros((seq_len, batch_size, self.label_dim, n_mc))
        score_chain = np.zeros((seq_len, batch_size, self.label_dim, n_mc))

        for i in range(n_mc):
            h2 = torch.zeros(batch_size, self.h_dim, device=self.device)
            for t in range(seq_len):
                x_t = torch.as_tensor(X_np[t], dtype=torch.float32, device=self.device)
                y_t = torch.as_tensor(Y_np[t], dtype=torch.float32, device=self.device)
                phi_x_t = self.phi_x(x_t)

                pred_out = self.predict_z(torch.cat([phi_x_t, h2], dim=1))
                pred_mean_t = self.predict_mean(pred_out)

                phi_z_t = self.phi_z(pred_mean_t)
                dec_out = self.dec_predict(torch.cat([phi_x_t, phi_z_t, h2], dim=1))
                dec_mean_t = self.dec_mean(dec_out)
                dec_std_t = self.dec_std(dec_out)

                mu_chain[t, :, :, i] = dec_mean_t.detach().cpu().numpy()
                std_chain[t, :, :, i] = dec_std_t.detach().cpu().numpy()

                nll = -stats.norm.logpdf(
                    y_t.cpu().numpy(),
                    loc=dec_mean_t.detach().cpu().numpy(),
                    scale=dec_std_t.detach().cpu().numpy(),
                )
                score_chain[t, :, :, i] = nll

                h2 = self.rnn2(torch.cat([phi_x_t, phi_z_t], dim=1), h2)

        return mu_chain.mean(axis=3), std_chain.mean(axis=3), score_chain.mean(axis=3)

    # ------------------------------------------------------------------
    # Extract latent variables for downstream diagnosis
    # ------------------------------------------------------------------
    def extract_latents(self, X, Y):
        """
        Extract z_post (encoder latent) and z_prior (prior latent) for each timestep.
        Returns z_post, z_prior as numpy arrays of shape (seq_len, batch, z_dim).
        """
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
        seq_len, batch_size = X_np.shape[0], X_np.shape[1]

        z_post_chain = np.zeros((seq_len, batch_size, self.z_dim))
        z_prior_chain = np.zeros((seq_len, batch_size, self.z_dim))
        z_pred_chain = np.zeros((seq_len, batch_size, self.z_dim))

        h = torch.zeros(batch_size, self.h_dim, device=self.device)
        h2 = torch.zeros(batch_size, self.h_dim, device=self.device)

        for t in range(seq_len):
            x_t = torch.as_tensor(X_np[t], dtype=torch.float32, device=self.device)
            y_t = torch.as_tensor(Y_np[t], dtype=torch.float32, device=self.device)
            phi_x_t = self.phi_x(x_t)
            phi_y_t = self.phi_y(y_t)

            enc_out = self.enc(torch.cat([phi_x_t, phi_y_t, h], dim=1))
            z_post = self.enc_mean(enc_out)

            prior_out = self.prior(torch.cat([phi_x_t, h], dim=1))
            z_prior = self.prior_mean(prior_out)

            pred_out = self.predict_z(torch.cat([phi_x_t, h2], dim=1))
            z_pred = self.predict_mean(pred_out)

            z_post_chain[t] = z_post.detach().cpu().numpy()
            z_prior_chain[t] = z_prior.detach().cpu().numpy()
            z_pred_chain[t] = z_pred.detach().cpu().numpy()

            phi_z_t = self.phi_z(z_post)
            phi_z_p = self.phi_z(z_pred)
            h = self.rnn(torch.cat([phi_x_t, phi_z_t], dim=1), h)
            h2 = self.rnn2(torch.cat([phi_x_t, phi_z_p], dim=1), h2)

        return z_post_chain, z_prior_chain, z_pred_chain

    # ------------------------------------------------------------------
    # Internal: reset stored variables
    # ------------------------------------------------------------------
    def _reset_variables(self):
        self.Z_mean, self.Z_std = [], []
        self.Xr_mean, self.Xr_std = [], []
        self.pZ_mean, self.pZ_std = [], []
        self.h_chain, self.h2_chain = [], []

        self.Xr_mean_prior, self.Xr_std_prior = [], []
        self.Xr_mean_predict, self.Xr_std_predict = [], []
        self.Z_mean_predict, self.Z_std_predict = [], []

        self.kld_loss = 0.0
        self.nll_loss = 0.0
        self.smooth_loss = 0.0
        self.kld_loss_predict = 0.0
        self.nll_loss_prior = 0.0
        self.nll_loss_predict = 0.0
        self.smooth_loss_prior = 0.0

    # ------------------------------------------------------------------
    # Internal: compute ELBO and auxiliary losses
    # ------------------------------------------------------------------
    def _calc_loss(self, Y):
        Y_flat = Y.view(Y.shape[0], Y.shape[1], -1)
        T = len(self.h_chain)

        kld_loss = torch.tensor(0.0, device=self.device)
        kld_loss_predict = torch.tensor(0.0, device=self.device)
        nll_loss = torch.tensor(0.0, device=self.device)
        nll_loss_prior = torch.tensor(0.0, device=self.device)
        nll_loss_predict = torch.tensor(0.0, device=self.device)
        smooth_loss = torch.tensor(0.0, device=self.device)
        smooth_loss_prior = torch.tensor(0.0, device=self.device)

        for t in range(T):
            # KL divergence: posterior || prior
            kld_loss = kld_loss + _kld_gauss(
                self.Z_mean[t], self.Z_std[t],
                self.pZ_mean[t], self.pZ_std[t],
            )
            # KL divergence: posterior || predict
            kld_loss_predict = kld_loss_predict + _kld_gauss(
                self.Z_mean[t], self.Z_std[t],
                self.Z_mean_predict[t], self.Z_std_predict[t],
            )
            # Reconstruction NLL
            dist_post = Normal(self.Xr_mean[t], self.Xr_std[t])
            nll_loss = nll_loss - dist_post.log_prob(Y_flat[t]).sum()

            dist_prior = Normal(self.Xr_mean_prior[t], self.Xr_std_prior[t])
            nll_loss_prior = nll_loss_prior - dist_prior.log_prob(Y_flat[t]).sum()

            dist_pred = Normal(self.Xr_mean_predict[t], self.Xr_std_predict[t])
            nll_loss_predict = nll_loss_predict - dist_pred.log_prob(Y_flat[t]).sum()

        for t in range(T - 1):
            smooth_loss = smooth_loss + _kld_gauss(
                self.Xr_mean[t], self.Xr_std[t],
                self.Xr_mean[t + 1], self.Xr_std[t + 1],
            )
            smooth_loss_prior = smooth_loss_prior + _kld_gauss(
                self.Xr_mean_prior[t], self.Xr_std_prior[t],
                self.Xr_mean_prior[t + 1], self.Xr_std_prior[t + 1],
            )

        return kld_loss, nll_loss, smooth_loss, kld_loss_predict, \
            nll_loss_prior, nll_loss_predict, smooth_loss_prior

    # ------------------------------------------------------------------
    # Reparameterization trick
    # ------------------------------------------------------------------
    def _reparameterize(self, mean, std):
        eps = torch.randn_like(std, device=self.device)
        return eps * std + mean


# =========================================================================
# Utility functions
# =========================================================================

def _kld_gauss(mean_1, std_1, mean_2, std_2):
    """KL divergence between two diagonal Gaussians."""
    kld = (2 * torch.log(std_2 + 1e-8) - 2 * torch.log(std_1 + 1e-8)
           + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / (std_2.pow(2) + 1e-8) - 1)
    return 0.5 * torch.sum(kld)


def make_sliding_windows(data: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """Create sliding windows from 2D data.

    Args:
        data: shape (n_samples, n_features)
        window_size: length of each window
        stride: step between windows

    Returns:
        shape (n_windows, window_size, n_features)
    """
    if data.shape[0] < window_size:
        raise ValueError(f"Not enough data ({data.shape[0]}) for window_size={window_size}")
    n_windows = (data.shape[0] - window_size) // stride + 1
    windows = np.zeros((n_windows, window_size, data.shape[1]), dtype=data.dtype)
    for i in range(n_windows):
        start = i * stride
        windows[i] = data[start:start + window_size]
    return windows


def make_sliding_windows_with_labels(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows with corresponding labels.

    Window label = 1 if ANY point in the window has label > 0.
    """
    windows = make_sliding_windows(data, window_size, stride)
    n_windows = windows.shape[0]
    win_labels = np.zeros(n_windows, dtype=np.int32)
    for i in range(n_windows):
        start = i * stride
        win_labels[i] = 1 if np.any(labels[start:start + window_size] > 0) else 0
    return windows, win_labels
