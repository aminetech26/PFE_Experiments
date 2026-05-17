from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from src.modeling.anomaly_detection.dl.maat.attn import AnomalyAttention, AttentionLayer
from src.modeling.anomaly_detection.dl.maat.embed import DataEmbedding


def _build_mamba_block(d_model: int) -> nn.Module:
    # mamba_ssm requires CUDA + Linux; imported at instantiation so the package
    # is importable on CPU dev machines without crashing at module load time.
    try:
        from mamba_ssm import Mamba  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "mamba_ssm is required for MAAT. "
            "Install it in a CUDA/Linux environment: pip install mamba-ssm"
        ) from exc

    return Mamba(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
    )


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: AttentionLayer,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        gate_mode: str = "learned",
    ) -> None:
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.gate_mode = gate_mode

        # Mamba SSM branch — Lightning handles device placement, no .to("cuda")
        self.mamba_block = _build_mamba_block(d_model)

        # Gate: fuse Mamba output (x_skip) and attention output (x_attn)
        # gate = sigmoid(Linear(cat([x_skip, x_attn], dim=-1)))
        # Only instantiated when gate_mode == "learned"
        self.gate_linear: nn.Linear | None = (
            nn.Linear(2 * d_model, d_model) if gate_mode == "learned" else None
        )

        # FFN: Conv1d 1×1 (equivalent to position-wise linear)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = functional.gelu if activation == "gelu" else functional.relu

    def forward(
        self, x: torch.Tensor, attn_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Attention branch
        new_x, series, prior, sigma = self.attention(x, x, x)

        # Mamba branch (parallel to attention, both see the original x)
        # Mamba expects [B, L, D] — which is exactly our x shape
        x_skip = self.mamba_block(x)  # [B, W, d_model]

        # Gated fusion based on gate_mode
        if self.gate_mode == "attention_only":
            gate = torch.zeros_like(x_skip)
            fused = new_x
        elif self.gate_mode == "mamba_only":
            gate = torch.ones_like(x_skip)
            fused = x_skip
        else:
            gate = torch.sigmoid(self.gate_linear(torch.cat([x_skip, new_x], dim=-1)))
            fused = gate * x_skip + (1.0 - gate) * new_x

        # Residual + LayerNorm
        x = self.norm1(x + self.dropout(fused))

        # FFN (position-wise, implemented as Conv1d 1×1)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm2(x + y)

        return x, series, prior, sigma, gate


class Encoder(nn.Module):
    def __init__(self, layers: list[EncoderLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, attn_mask=None
    ) -> tuple[torch.Tensor, list, list, list, list]:
        series_list, prior_list, sigma_list, gate_list = [], [], [], []
        for layer in self.layers:
            x, series, prior, sigma, gate = layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            gate_list.append(gate)
        return x, series_list, prior_list, sigma_list, gate_list


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
        gate_mode: str = "learned",
    ) -> None:
        super().__init__()
        assert win_size % block_size == 0, (
            f"win_size ({win_size}) must be divisible by block_size ({block_size})"
        )
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)

        def _attention_layer() -> AttentionLayer:
            return AttentionLayer(
                AnomalyAttention(
                    win_size=win_size,
                    mask_flag=False,
                    attention_dropout=dropout,
                    block_size=block_size,
                    use_sparse_attention=True,
                ),
                d_model=d_model,
                n_heads=n_heads,
            )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    _attention_layer(),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    gate_mode=gate_mode,
                )
                for _ in range(e_layers)
            ]
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list, list, list, list]:
        # x: [B, W, F]
        enc_out = self.enc_embedding(x)  # [B, W, d_model]
        enc_out, series_list, prior_list, sigma_list, gate_list = self.encoder(enc_out)
        dec_out = self.projection(enc_out)  # [B, W, F]
        return dec_out, series_list, prior_list, sigma_list, gate_list
