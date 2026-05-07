from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyAttention(nn.Module):
    """Dual-path attention: sparse series association + Gaussian prior association.

    Series association S: block-local softmax attention (sparse).
    Prior association P: Gaussian kernel parameterized by learned sigma.
    Both are [B, H, W, W] distributions returned for KL-based anomaly scoring.
    """

    def __init__(
        self,
        win_size: int,
        mask_flag: bool = False,
        scale: float | None = None,
        attention_dropout: float = 0.0,
        block_size: int = 10,
        use_sparse_attention: bool = True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.block_size = block_size
        self.use_sparse_attention = use_sparse_attention

        # Precomputed distance matrix for Gaussian prior: distances[i,j] = |i-j|
        # Stored as buffer so Lightning moves it to the correct device automatically.
        idx = torch.arange(win_size, dtype=torch.float)
        distances = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))  # [W, W]
        self.register_buffer("distances", distances)

    def forward(
        self,
        queries: torch.Tensor,  # [B, L, H, E]
        keys: torch.Tensor,     # [B, S, H, E]
        values: torch.Tensor,   # [B, S, H, D]
        sigma: torch.Tensor,    # [B, L, H]  — learned prior widths (pre-activation)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or (1.0 / math.sqrt(E))

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [B, H, L, S]

        if self.use_sparse_attention:
            # Block-local mask: keep only positions within the same block.
            # Positions i and j are in the same block iff i // block_size == j // block_size.
            # We build the mask on the correct device dynamically.
            idx = torch.arange(L, device=queries.device)
            block_ids = idx // self.block_size  # [L]
            # [L, L] True where same block (keep), False where different block (mask out)
            same_block = block_ids.unsqueeze(0) == block_ids.unsqueeze(1)  # [L, S]
            # Expand to [1, 1, L, S] and fill out-of-block positions with -inf
            scores = scores.masked_fill(
                ~same_block.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Safe softmax: replace any remaining -inf columns (empty blocks) with uniform
        series = torch.softmax(scale * scores, dim=-1)
        # Replace NaN from all-inf rows (shouldn't happen with valid block_size, but guard it)
        series = torch.nan_to_num(series, nan=0.0)
        series = self.dropout(series)  # [B, H, L, S]

        # Gaussian prior: P[b,h,i,j] ∝ exp(-distances[i,j]^2 / (2*sigma_i^2))
        # sigma: [B, L, H] → [B, H, L] → [B, H, L, 1]
        sigma_t = sigma.transpose(1, 2)  # [B, H, L]
        # Clamp sigma to a positive range via sigmoid: range (eps, 5+eps)
        sigma_t = torch.sigmoid(sigma_t) * 5 + 1e-4  # [B, H, L]
        sigma_t = sigma_t.unsqueeze(-1)  # [B, H, L, 1]

        # distances: [W, W] buffer → slice to [L, S] and broadcast
        dists = self.distances[:L, :S]  # [L, S]
        prior = torch.exp(
            -dists.pow(2).unsqueeze(0).unsqueeze(0) / (2.0 * sigma_t.pow(2))
        )  # [B, H, L, S]
        # Normalize prior to sum to 1 over j (last dim)
        prior = prior / (prior.sum(dim=-1, keepdim=True).clamp(min=1e-9))

        V = torch.einsum("bhls,bshd->blhd", series, values)  # [B, L, H, D]

        # Return sigma back as [B, L, H] for logging
        return V.contiguous(), series, prior, sigma_t.squeeze(-1).transpose(1, 2)


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: AnomalyAttention,
        d_model: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.inner_attention = attention
        self.n_heads = n_heads
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.query_projection = nn.Linear(d_model, n_heads * d_k)
        self.key_projection = nn.Linear(d_model, n_heads * d_k)
        self.value_projection = nn.Linear(d_model, n_heads * d_v)
        self.sigma_projection = nn.Linear(d_model, n_heads)  # learned prior widths
        self.out_projection = nn.Linear(n_heads * d_v, d_model)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = queries.shape
        H = self.n_heads

        Q = self.query_projection(queries).view(B, L, H, -1)   # [B, L, H, d_k]
        K = self.key_projection(keys).view(B, L, H, -1)        # [B, L, H, d_k]
        V = self.value_projection(values).view(B, L, H, -1)    # [B, L, H, d_v]
        sigma = self.sigma_projection(queries)                   # [B, L, H]

        out, series, prior, sigma_out = self.inner_attention(Q, K, V, sigma)
        # out: [B, L, H, d_v] → [B, L, H*d_v]
        out = out.view(B, L, -1)
        return self.out_projection(out), series, prior, sigma_out
