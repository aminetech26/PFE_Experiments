from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.anomaly_detection.dl.maat.attn import AnomalyAttention, AttentionLayer
from src.modeling.anomaly_detection.dl.maat.embed import DataEmbedding

# mamba_ssm is a CUDA/Linux optional dependency — imported here so the full import
# chain from run.py stays lazy (run.py → ssm_model.py → model.py → mamba_ssm).
from mamba_ssm import Mamba  # noqa: E402

# ARCHITECTURAL DEVIATION FROM UPSTREAM MAAT (Sellam et al. 2025 / ilyesbenaissa/MAAT):
# Upstream places a single Mamba block shared across all encoder layers (after all attention
# layers) and uses a different gating arrangement. This implementation uses one independent
# Mamba block *per encoder layer* running in parallel with the anomaly attention branch,
# fused via a learned sigmoid gate. The association discrepancy mechanism (sparse attention +
# Gaussian prior + KL minimax) is faithful to the paper. The Mamba placement is a deliberate
# adaptation; thesis should note this as a variant rather than a strict replication.


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: AttentionLayer,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)

        # Mamba SSM branch — Lightning handles device placement, no .to("cuda")
        self.mamba_block = Mamba(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Gate: fuse Mamba output (x_skip) and attention output (x_attn)
        # gate = sigmoid(Linear(cat([x_skip, x_attn], dim=-1)))
        self.gate_linear = nn.Linear(2 * d_model, d_model)

        # FFN: Conv1d 1×1 (equivalent to position-wise linear)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self, x: torch.Tensor, attn_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention branch
        new_x, series, prior, sigma = self.attention(x, x, x)

        # Mamba branch (parallel to attention, both see the original x)
        # Mamba expects [B, L, D] — which is exactly our x shape
        x_skip = self.mamba_block(x)  # [B, W, d_model]

        # Gated fusion: gate ⊙ x_skip + (1-gate) ⊙ new_x
        gate = torch.sigmoid(self.gate_linear(torch.cat([x_skip, new_x], dim=-1)))
        fused = gate * x_skip + (1.0 - gate) * new_x

        # Residual + LayerNorm
        x = self.norm1(x + self.dropout(fused))

        # FFN (position-wise, implemented as Conv1d 1×1)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)

        return x, series, prior, sigma


class Encoder(nn.Module):
    def __init__(self, layers: list[EncoderLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, attn_mask=None
    ) -> tuple[torch.Tensor, list, list, list]:
        series_list, prior_list, sigma_list = [], [], []
        for layer in self.layers:
            x, series, prior, sigma = layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
        return x, series_list, prior_list, sigma_list


class MambaAnomalyTransformer(nn.Module):
    def __init__(
        self,
        win_size: int,
        enc_in: int,
        c_out: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        activation: str = "gelu",
        block_size: int = 10,
    ) -> None:
        super().__init__()
        assert win_size % block_size == 0, (
            f"win_size ({win_size}) must be divisible by block_size ({block_size})"
        )
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            win_size=win_size,
                            mask_flag=False,
                            attention_dropout=dropout,
                            block_size=block_size,
                            use_sparse_attention=True,  # explicitly enabled
                        ),
                        d_model=d_model,
                        n_heads=n_heads,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ]
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list, list, list]:
        # x: [B, W, F]
        enc_out = self.enc_embedding(x)  # [B, W, d_model]
        enc_out, series_list, prior_list, sigma_list = self.encoder(enc_out)
        dec_out = self.projection(enc_out)  # [B, W, F]
        return dec_out, series_list, prior_list, sigma_list
