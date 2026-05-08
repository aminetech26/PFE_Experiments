from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualMLP(nn.Module):
    """Two-layer residual block with LayerNorm and Dropout.

    Residual connection projects input to `out_dim` if `in_dim != out_dim`.
    This ensures `z_t` carries information (decoder cannot collapse to `h_prev`
    alone) and gives stable gradients through the inference / prior / decoder paths.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.proj(x)


class DeepLatentStateSpaceModel(nn.Module):
    """Causal deep latent state-space model.

    The GRU maps the input sequence to hidden states `h_t`. We build
    `h_prev` by right-shifting: `h_prev[:, 0] = 0`, `h_prev[:, t] = h_seq[:, t-1]`.
    All three networks (inference, prior, decoder) receive `h_prev`, making
    the prior and KL causally meaningful:

        h_t     = GRU(x_t, h_{t-1})
        z_t     ~ q(z_t | x_t, h_{t-1})       # inference / recognition
        z_t     ~ p(z_t | h_{t-1})             # causal transition prior
        x_hat_t = decoder(z_t, h_{t-1})        # forced to use z_t

    KL(q || p) now measures how much `x_t` surprised the model given its
    previous state — an information-theoretic anomaly signal.

    At eval time (no training noise), `z = q_mu` for deterministic scoring.
    """

    def __init__(
        self,
        n_features: int,
        win_size: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        encoder_dim: int = 128,
        decoder_dim: int = 128,
        n_gru_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )

        # Inference: q(z_t | x_t, h_{t-1}) — sees current observation + causal context
        self.inference_net = nn.Sequential(
            _ResidualMLP(n_features + hidden_dim, encoder_dim, dropout),
            nn.Linear(encoder_dim, 2 * latent_dim),
        )

        # Prior: p(z_t | h_{t-1}) — causal; does NOT see current observation
        self.prior_net = nn.Sequential(
            _ResidualMLP(hidden_dim, encoder_dim, dropout),
            nn.Linear(encoder_dim, 2 * latent_dim),
        )

        # Prediction head: x_pred_t = f(h_{t-1}) — expected normal evolution from state alone
        # h_prev has no current observation info, so this is a genuine one-step prediction.
        self.prediction_head = nn.Sequential(
            _ResidualMLP(hidden_dim, decoder_dim, dropout),
            nn.Linear(decoder_dim, n_features),
        )

        # Residual correction: delta_t = f(z_t, h_{t-1})
        # z_t encodes the observation-specific deviation from the predicted state.
        # Anomalies produce large residuals; normal observations produce small ones.
        self.decoder = nn.Sequential(
            _ResidualMLP(latent_dim + hidden_dim, decoder_dim, dropout),
            nn.Linear(decoder_dim, n_features),
        )

    def _split_params(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split [B, W, 2*Z] → (mu, logvar) with logvar clamped to [-10, 10]."""
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar.clamp(-10.0, 10.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: [B, W, F]
        x_proj = self.drop(self.input_proj(x))     # [B, W, H]
        h_seq, _ = self.gru(x_proj)                # [B, W, H]  — h_seq[:, t] = h_t

        # Build causal context: h_prev[:, t] = h_{t-1}
        # h_prev[:, 0] = 0 (no prior hidden state at the start of the window)
        h_prev = torch.zeros_like(h_seq)
        h_prev[:, 1:] = h_seq[:, :-1]             # [B, W, H]

        # Inference distribution q(z_t | x_t, h_{t-1})
        q_input = torch.cat([x, h_prev], dim=-1)   # [B, W, F+H]
        q_mu, q_logvar = self._split_params(self.inference_net(q_input))  # [B, W, Z]

        # Causal prior p(z_t | h_{t-1}) — does NOT see x_t
        p_mu, p_logvar = self._split_params(self.prior_net(h_prev))       # [B, W, Z]

        # Reparameterize during training; use mean at eval for deterministic scoring
        if self.training:
            z = q_mu + torch.randn_like(q_mu) * torch.exp(0.5 * q_logvar)
        else:
            z = q_mu

        # Prediction: expected observation from state alone (no z_t, no x_t)
        x_pred = self.prediction_head(h_prev)                  # [B, W, F]

        # Residual correction: observation-specific deviation encoded by z_t
        residual = self.decoder(torch.cat([z, h_prev], dim=-1))  # [B, W, F]

        # Final reconstruction: state prediction + latent residual
        x_hat = x_pred + residual                              # [B, W, F]

        return {
            "x_hat": x_hat,
            "x_pred": x_pred,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
            "z": z,
            "h": h_seq,
            "h_prev": h_prev,
        }
