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


class FeatureAttentionBlock(nn.Module):
    """Multi-head self-attention over the feature dimension at each timestep.

    Treats each of the F features as a token with embedding dim `feat_embed_dim`,
    runs self-attention over F tokens, then projects to `out_dim`. Lets the encoder
    capture cross-feature interactions (e.g. pdc1 vs pdc2 ratio anomalies) that a
    plain linear projection cannot. Designed for small F (≤ ~30) as used with the
    plus_physics feature profile on A100 GPU.

    Input: [B, W, F]; Output: [B, W, out_dim].
    """

    def __init__(
        self, n_features: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        # feat_embed_dim: embedding size per feature token, divisible by n_heads
        feat_embed_dim = max(n_heads, (out_dim + n_features - 1) // max(n_features, 1))
        feat_embed_dim = feat_embed_dim + (n_heads - feat_embed_dim % n_heads) % n_heads
        self._n_features = n_features
        self._feat_embed_dim = feat_embed_dim
        self.feat_embed = nn.Linear(1, feat_embed_dim)  # each feature scalar → vector
        self.attn = nn.MultiheadAttention(feat_embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(feat_embed_dim)
        self.proj = nn.Linear(n_features * feat_embed_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W, Fv = x.shape
        x_flat = x.reshape(B * W, Fv, 1)                          # [B*W, F, 1]
        tokens = self.feat_embed(x_flat)                           # [B*W, F, feat_embed_dim]
        attended, _ = self.attn(tokens, tokens, tokens)            # [B*W, F, feat_embed_dim]
        tokens = self.norm(tokens + attended)                      # residual + norm
        flat = tokens.reshape(B * W, Fv * self._feat_embed_dim)    # [B*W, F*feat_embed_dim]
        out = self.out_norm(self.act(self.proj(flat)))             # [B*W, out_dim]
        return self.drop(out.reshape(B, W, -1))                    # [B, W, out_dim]


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
        n_attention_heads: int = 4,
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

        # Feature attention encoder (replaces plain input_proj)
        self.feature_attn = FeatureAttentionBlock(n_features, hidden_dim, n_attention_heads, dropout)
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

        # Expected-power head: c_t → P_expected for [pdc1, pdc2]
        # Gives an IEC-61724 Performance Ratio signal for class-2 (degradation) scoring
        self.expected_power_head: nn.Module | None = None
        if self.expected_power_indices and condition_dim > 0:
            self.expected_power_head = nn.Sequential(
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
        x_proj = self.feature_attn(x)              # [B, W, H]  (FeatureAttentionBlock)
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

        # Expected-power head: c_t → P_expected_t (IEC 61724 Performance Ratio signal)
        p_expected: torch.Tensor | None = None
        if self.expected_power_head is not None and c is not None:
            p_expected = self.expected_power_head(c)           # [B, W, n_power]

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
