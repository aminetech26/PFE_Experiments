from __future__ import annotations

import torch
import torch.nn as nn


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


class FeatureSEBlock(nn.Module):
    """Squeeze-and-Excitation over features, then projection to hidden_dim.

    Input: [B, W, F]; Output: [B, W, out_dim].
    """

    def __init__(
        self, n_features: int, out_dim: int, reduction_ratio: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        reduced_dim = max(1, n_features // max(1, reduction_ratio))
        self.excitation = nn.Sequential(
            nn.Linear(n_features, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, n_features),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(n_features, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.mean(dim=1)
        gates = self.excitation(squeeze).unsqueeze(1)
        gated = x * gates
        out = self.proj(gated)
        out = self.norm(out)
        out = self.act(out)
        return self.drop(out)


def _build_mamba_slow_encoder(d_model: int, n_layers: int = 1) -> nn.Module:
    """Build a Mamba SSM block for the slow-context branch (lazy import for CPU dev)."""
    try:
        from mamba_ssm import Mamba
    except ImportError as exc:
        raise ImportError(
            "mamba_ssm is required for the slow-context branch. "
            "Install with: pip install mamba-ssm (requires CUDA)"
        ) from exc
    blocks = [Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) for _ in range(n_layers)]
    return blocks[0] if n_layers == 1 else nn.Sequential(*blocks)


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
        # DLSSM-FDD additions
        se_reduction_ratio: int = 4,
        slow_hidden_dim: int = 64,
        expected_power_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.expected_power_indices: list[int] = list(expected_power_indices or [])
        c = condition_dim  # inputs widen by c when CVAE is on (c=0 → no-op)

        # Feature SE-gating encoder (replaces heavy feature self-attention)
        self.feature_attn = FeatureSEBlock(
            n_features=n_features,
            out_dim=hidden_dim,
            reduction_ratio=se_reduction_ratio,
            dropout=dropout,
        )
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

        # Slow-context branch: Mamba/GRU SSM on a low-rate preceding window
        # x_slow: [B, T_slow, F] → slow_input_proj → [B, T_slow, slow_hidden_dim]
        # → slow_encoder → slow_proj → [B, hidden_dim] bias added to h_prev
        self.slow_input_proj: nn.Module | None = None
        self.slow_encoder: nn.Module | None = None
        self.slow_proj: nn.Module | None = None
        if slow_hidden_dim > 0:
            self.slow_input_proj = nn.Linear(n_features, slow_hidden_dim)
            self.slow_encoder = _build_mamba_slow_encoder(slow_hidden_dim, n_layers=1)
            self.slow_proj = nn.Linear(slow_hidden_dim, hidden_dim)

        # Voltage-Anomaly Head: c_t → expected vdc envelope under normal operation.
        # Detects DC-side voltage envelope deviation (voltage-collapse fault family).
        # Trained with a quantile-regression objective so it predicts a calibrated lower
        # bound rather than the conditional mean — see expected_voltage_loss in losses.py.
        self.voltage_anomaly_head: nn.Module | None = None
        if self.expected_power_indices and condition_dim > 0:
            self.voltage_anomaly_head = nn.Sequential(
                _ResidualMLP(condition_dim, decoder_dim, dropout),
                nn.Linear(decoder_dim, len(self.expected_power_indices)),
            )

    def _split_params(
        self, params: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split [B, W, 2*Z] → (mu, logvar) with logvar clamped to [-10, 10]."""
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar.clamp(-10.0, 10.0)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor | None = None,
        x_slow: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # x:      [B, W, F]        fast window (standardized features)
        # c:      [B, W, C] opt    CVAE conditioning slice (irr, pvt, ...)
        # x_slow: [B, T_slow, slow_hidden_dim] opt   slow-rate context (already projected)
        x_proj = self.feature_attn(x)              # [B, W, H]  (FeatureSEBlock)
        h_seq, _ = self.gru(x_proj)                # [B, W, H]  — h_seq[:, t] = h_t

        # Build causal context: h_prev[:, t] = h_{t-1}
        # h_prev[:, 0] = 0 (no prior hidden state at the start of the window)
        h_prev = torch.zeros_like(h_seq)
        h_prev[:, 1:] = h_seq[:, :-1]             # [B, W, H]

        # Slow-context branch: inject a slow summary as an additive bias on h_prev
        if (
            x_slow is not None
            and self.slow_input_proj is not None
            and self.slow_encoder is not None
            and self.slow_proj is not None
        ):
            s_in = self.slow_input_proj(x_slow)     # [B, T_slow, slow_hidden_dim]
            s = self.slow_encoder(s_in)              # [B, T_slow, slow_hidden_dim]
            s_summary = s[:, -1, :]                  # [B, slow_hidden_dim] — most recent context
            s_proj = self.slow_proj(s_summary)       # [B, hidden_dim]
            h_prev = h_prev + s_proj.unsqueeze(1)    # broadcast across W

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

        # Voltage-Anomaly Head: c_t → expected vdc envelope under normal operation
        p_expected: torch.Tensor | None = None
        if self.voltage_anomaly_head is not None and c is not None:
            p_expected = self.voltage_anomaly_head(c)          # [B, W, n_voltage]

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
            "p_expected": p_expected,
        }
