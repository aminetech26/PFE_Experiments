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
        condition_dim: int = 0,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        c = condition_dim  # inputs widen by c when CVAE is on (c=0 → no-op)

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

        # Inference: q(z_t | x_t, h_{t-1}, c_t)
        self.inference_net = nn.Sequential(
            _ResidualMLP(n_features + hidden_dim + c, encoder_dim, dropout),
            nn.Linear(encoder_dim, 2 * latent_dim),
        )

        # Causal prior: p(z_t | h_{t-1}, c_t) — operating-point-aware
        self.prior_net = nn.Sequential(
            _ResidualMLP(hidden_dim + c, encoder_dim, dropout),
            nn.Linear(encoder_dim, 2 * latent_dim),
        )

        # Prediction head: x_pred_t = f(h_{t-1}, c_t)
        self.prediction_head = nn.Sequential(
            _ResidualMLP(hidden_dim + c, decoder_dim, dropout),
            nn.Linear(decoder_dim, n_features),
        )

        # Residual correction: delta_t = f(z_t, h_{t-1}, c_t)
        self.decoder = nn.Sequential(
            _ResidualMLP(latent_dim + hidden_dim + c, decoder_dim, dropout),
            nn.Linear(decoder_dim, n_features),
        )

    def _split_params(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split [B, W, 2*Z] → (mu, logvar) with logvar clamped to [-10, 10]."""
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar.clamp(-10.0, 10.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # x: [B, W, F]    c (optional): [B, W, C]
        x_proj = self.drop(self.input_proj(x))     # [B, W, H]
        h_seq, _ = self.gru(x_proj)                # [B, W, H]  — h_seq[:, t] = h_t

        # Build causal context: h_prev[:, t] = h_{t-1}
        # h_prev[:, 0] = 0 (no prior hidden state at the start of the window)
        h_prev = torch.zeros_like(h_seq)
        h_prev[:, 1:] = h_seq[:, :-1]             # [B, W, H]

        # Validate conditioning expectations
        if self.condition_dim > 0:
            if c is None:
                raise ValueError("condition_dim > 0 but c not provided")
            if c.size(-1) != self.condition_dim:
                raise ValueError(f"c last-dim {c.size(-1)} != condition_dim {self.condition_dim}")
        cond_parts: list[torch.Tensor] = [c] if (self.condition_dim > 0 and c is not None) else []

        # Inference q(z_t | x_t, h_{t-1}, c_t)
        q_input = torch.cat([x, h_prev, *cond_parts], dim=-1)
        q_mu, q_logvar = self._split_params(self.inference_net(q_input))  # [B, W, Z]

        # Causal prior p(z_t | h_{t-1}, c_t) — does NOT see x_t
        p_input = torch.cat([h_prev, *cond_parts], dim=-1)
        p_mu, p_logvar = self._split_params(self.prior_net(p_input))      # [B, W, Z]

        # Reparameterize during training; use mean at eval for deterministic scoring
        if self.training:
            z = q_mu + torch.randn_like(q_mu) * torch.exp(0.5 * q_logvar)
        else:
            z = q_mu

        # Prediction: expected observation from state alone + conditioning context
        pred_input = torch.cat([h_prev, *cond_parts], dim=-1)
        x_pred = self.prediction_head(pred_input)                          # [B, W, F]

        # Residual correction: observation-specific deviation encoded by z_t
        dec_input = torch.cat([z, h_prev, *cond_parts], dim=-1)
        residual = self.decoder(dec_input)                                 # [B, W, F]

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
