from __future__ import annotations

import torch
from torch import nn


class PVGDN(nn.Module):
    """Lightweight graph-temporal predictor for PV anomaly detection.

    Input: x [B, W, F]
    Output: x_hat_next [B, F]
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        temporal_kernel_size: int = 3,
        temporal_layers: int = 2,
        graph_top_k: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.d_model = int(d_model)
        self.graph_top_k = int(graph_top_k)

        pads = temporal_kernel_size // 2
        blocks: list[nn.Module] = []
        in_ch = 1
        for _ in range(max(1, int(temporal_layers))):
            blocks.extend(
                [
                    nn.Conv1d(in_ch, d_model, kernel_size=temporal_kernel_size, padding=pads),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = d_model
        self.temporal_encoder = nn.Sequential(*blocks)

        self.node_emb = nn.Parameter(torch.randn(self.n_features, d_model) * 0.02)
        self.msg_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, 1)

    def _adjacency(self) -> torch.Tensor:
        # Learned similarity graph over feature nodes
        z = nn.functional.normalize(self.node_emb, dim=-1)
        logits = z @ z.T  # [F, F]
        # Use masked_fill (creates new tensor) rather than fill_diagonal_ (in-place on grad-tracked tensor)
        diag_mask = torch.eye(self.n_features, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diag_mask, float("-inf"))

        if self.graph_top_k > 0 and self.graph_top_k < self.n_features:
            topk = torch.topk(logits, k=self.graph_top_k, dim=-1).indices
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, topk, True)
            logits = logits.masked_fill(~mask, float("-inf"))

        return torch.softmax(logits, dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, W, F]
        bsz, _, n_feat = x.shape
        if n_feat != self.n_features:
            raise ValueError(f"Expected n_features={self.n_features}, got {n_feat}")

        # Temporal encode each feature channel independently
        xf = x.transpose(1, 2).contiguous()  # [B, F, W]
        xf = xf.view(bsz * self.n_features, 1, -1)
        ht = self.temporal_encoder(xf)  # [B*F, D, W]
        h_last = ht[:, :, -1].view(bsz, self.n_features, self.d_model)  # [B, F, D]

        # Graph message passing over feature nodes
        a = self._adjacency()  # [F, F]
        h_msg = torch.einsum("ij,bjd->bid", a, h_last)
        h = self.msg_norm(h_msg)
        h = nn.functional.gelu(h)

        # Predict next-step feature values from past temporal context
        x_hat = self.readout(h).squeeze(-1)  # [B, F]
        return x_hat, a
