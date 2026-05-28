"""PC-Flow: Physics-Conditioned Normalizing Flow for PV anomaly detection.

Pure-PyTorch conditional RealNVP (Dinh et al. 2017) with affine coupling layers
conditioned on the exogenous operating point c = (irr, pvt).

Architecture:
    Stack of K conditional affine-coupling blocks. Each block:
      1. Splits x into (x_a, x_b) via a fixed binary mask.
      2. Computes scale/translation from a small MLP: [x_a, c] → (s, t).
      3. Transforms x_b' = x_b * exp(s) + t  (invertible, log|det J| = sum(s)).
      4. Applies a fixed learned permutation so different dimensions mix each block.

Anomaly score:
    score(x | c) = -log p(x | c)
                 = 0.5*||z||² + 0.5*D*log(2π) - sum_k log|det J_k|

No external flow library dependency — pure MLPs + element-wise ops.
ONNX-exportable: standard matmul, exp, sum, concat.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class _AffineCouplingBlock(nn.Module):
    """One conditional affine-coupling block.

    mask: BoolTensor of shape [D]. True indices form x_a (pass-through).
    """

    def __init__(
        self,
        d_in: int,
        n_context: int,
        hidden_dim: int,
        mask: torch.Tensor,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask.bool())

        d_a = int(mask.sum().item())
        d_b = d_in - d_a

        # MLP: [x_a, c] → [d_b * 2] (s and t interleaved)
        self.net = nn.Sequential(
            nn.Linear(d_a + n_context, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_b * 2),
        )
        # Initialize s-head to near-zero so identity transform at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (normalising direction: data → latent).

        Returns:
            (y, log_det): y is transformed x, log_det is per-sample log|det J|.
        """
        x_a = x[:, self.mask]          # [B, d_a]
        x_b = x[:, ~self.mask]         # [B, d_b]

        st = self.net(torch.cat([x_a, c], dim=-1))   # [B, 2*d_b]
        s, t = st.chunk(2, dim=-1)                    # [B, d_b] each
        s = torch.tanh(s) * 2.0                       # stabilise: s ∈ (-2, 2)

        y_b = x_b * s.exp() + t
        log_det = s.sum(dim=-1)         # [B]

        y = x.clone()
        y[:, self.mask] = x_a
        y[:, ~self.mask] = y_b
        return y, log_det

    def inverse(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Inverse pass (latent → data). Used only for sanity-check / generation."""
        y_a = y[:, self.mask]
        y_b = y[:, ~self.mask]

        st = self.net(torch.cat([y_a, c], dim=-1))
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0

        x_b = (y_b - t) * (-s).exp()
        x = y.clone()
        x[:, self.mask] = y_a
        x[:, ~self.mask] = x_b
        return x


class PCFlowModel(nn.Module):
    """Conditional RealNVP normalizing flow for PV anomaly detection.

    Args:
        n_features: Dimensionality of x (non-context features).
        n_context:  Dimensionality of c (e.g. 2 for [irr, pvt]).
        n_coupling_layers: Number of affine-coupling blocks (default 4).
        hidden_dim: Hidden size of each coupling MLP (default 32).
        dropout: Dropout rate inside coupling MLPs (default 0.0).
    """

    def __init__(
        self,
        n_features: int,
        n_context: int,
        n_coupling_layers: int = 4,
        hidden_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_context = n_context
        self._log_2pi = math.log(2 * math.pi)

        # Build alternating binary masks (even indices pass-through in even layers)
        self.blocks = nn.ModuleList()
        for k in range(n_coupling_layers):
            mask = torch.zeros(n_features, dtype=torch.bool)
            if k % 2 == 0:
                mask[::2] = True      # even indices pass-through
            else:
                mask[1::2] = True     # odd indices pass-through
            self.blocks.append(
                _AffineCouplingBlock(n_features, n_context, hidden_dim, mask, dropout)
            )

        # Fixed random permutation between blocks (makes the mask alternation more general)
        perms = []
        for k in range(n_coupling_layers - 1):
            g = torch.Generator()
            g.manual_seed(k + 42)
            perms.append(torch.randperm(n_features, generator=g))
        self.register_buffer("_perms", torch.stack(perms) if perms else torch.empty(0, n_features, dtype=torch.long))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform x → z (normalising direction).

        Returns:
            z:       [B, D] latent; ~ N(0, I) on normal data after training.
            log_det: [B]   sum of log|det J| across all blocks.
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i, block in enumerate(self.blocks):
            z, ld = block(z, c)
            log_det_total = log_det_total + ld
            # Apply permutation between blocks (except after the last)
            if i < len(self.blocks) - 1 and self._perms.numel() > 0:
                z = z[:, self._perms[i]]
        return z, log_det_total

    def log_prob(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Compute per-sample log p(x | c).

        log p(x|c) = -0.5*||z||² - 0.5*D*log(2π) + log|det J|

        Returns:
            [B] log-probabilities (higher = more normal).
        """
        z, log_det = self.forward(x, c)
        D = self.n_features
        log_p_z = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * self._log_2pi
        return log_p_z + log_det

    def anomaly_score(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Per-sample anomaly score = -log p(x | c). Higher = more anomalous."""
        return -self.log_prob(x, c)

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Reconstruct x from latent z (for generation / sanity checks)."""
        x = z
        # Undo permutations in reverse order
        for i in range(len(self.blocks) - 2, -1, -1):
            if self._perms.numel() > 0:
                inv_perm = torch.argsort(self._perms[i])
                x = x[:, inv_perm]
            x = self.blocks[i + 1 if i < len(self.blocks) - 1 else i].inverse(x, c)
        x = self.blocks[0].inverse(x, c)
        return x

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
