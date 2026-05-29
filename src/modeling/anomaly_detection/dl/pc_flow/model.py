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
import torch.nn.functional as F  # noqa: N812

_MIN_DERIVATIVE = 1e-3


def _mask_to_perm(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert a boolean pass-through mask into gather/un-gather permutations.

    Returns (perm, inv_perm, d_a) where `x.index_select(1, perm)` places the d_a
    pass-through dims first and the d_b transformed dims last, and
    `index_select(1, inv_perm)` restores the original order. Using index_select
    (→ ONNX Gather) + slice + concat keeps the graph batch-generalizable; boolean
    mask indexing does NOT export correctly (bakes batch-1 shape constants).
    """
    mask = mask.bool()
    idx_a = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    idx_b = torch.nonzero(~mask, as_tuple=False).squeeze(-1)
    perm = torch.cat([idx_a, idx_b]).long()
    inv_perm = torch.argsort(perm)
    return perm, inv_perm, int(mask.sum().item())


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
        perm, inv_perm, d_a = _mask_to_perm(mask)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)
        self.d_a = d_a
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
        xp = x.index_select(1, self.perm)             # [B, D] reordered: x_a | x_b
        x_a = xp[:, :self.d_a]                         # [B, d_a]
        x_b = xp[:, self.d_a:]                         # [B, d_b]

        st = self.net(torch.cat([x_a, c], dim=-1))   # [B, 2*d_b]
        s, t = st.chunk(2, dim=-1)                    # [B, d_b] each
        s = torch.tanh(s) * 2.0                       # stabilise: s ∈ (-2, 2)

        y_b = x_b * s.exp() + t
        log_det = s.sum(dim=-1)         # [B]

        yp = torch.cat([x_a, y_b], dim=-1)
        y = yp.index_select(1, self.inv_perm)
        return y, log_det

    def inverse(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Inverse pass (latent → data). Used only for sanity-check / generation."""
        yp = y.index_select(1, self.perm)
        y_a = yp[:, :self.d_a]
        y_b = yp[:, self.d_a:]

        st = self.net(torch.cat([y_a, c], dim=-1))
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0

        x_b = (y_b - t) * (-s).exp()
        xp = torch.cat([y_a, x_b], dim=-1)
        return xp.index_select(1, self.inv_perm)


def _rqs_forward(
    x: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    tail_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rational-quadratic spline forward transform (Durkan et al. 2019).

    Args:
        x:           [B, d]      input; identity transform outside ±tail_bound
        widths:      [B, d, K]   positive bin widths, summing to 2*tail_bound
        heights:     [B, d, K]   positive bin heights, summing to 2*tail_bound
        derivatives: [B, d, K+1] positive knot derivatives (incl. boundary)
        tail_bound:  domain half-width

    Returns:
        (y, log_abs_det) both [B, d]
    """
    inside = (x >= -tail_bound) & (x <= tail_bound)
    x_safe = x.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)

    # Cumulative knot positions starting at -tail_bound
    cum_w = F.pad(widths.cumsum(-1), (1, 0), value=0.0) + (-tail_bound)   # [B, d, K+1]
    cum_h = F.pad(heights.cumsum(-1), (1, 0), value=0.0) + (-tail_bound)  # [B, d, K+1]

    # Bin index = number of interior knots <= x  (left-closed bins [p_k, p_{k+1})).
    # Comparison-sum instead of torch.searchsorted: ONNX-exportable (Greater+ReduceSum)
    # and exact for the left-closed convention. Critical for edge deployment.
    interior = cum_w[..., 1:-1]                                   # [B, d, K-1]
    bin_idx = (x_safe.unsqueeze(-1) >= interior).sum(dim=-1)      # [B, d] in [0, K-1]
    bin_idx = bin_idx.clamp(0, widths.shape[-1] - 1)

    idx = bin_idx.unsqueeze(-1)  # [B, d, 1]
    w_k  = widths.gather(-1, idx).squeeze(-1)
    h_k  = heights.gather(-1, idx).squeeze(-1)
    x_k  = cum_w.gather(-1, idx).squeeze(-1)
    y_k  = cum_h.gather(-1, idx).squeeze(-1)
    d_k  = derivatives.gather(-1, idx).squeeze(-1)
    d_k1 = derivatives.gather(-1, (idx + 1).clamp(max=derivatives.shape[-1] - 1)).squeeze(-1)

    s_k  = h_k / w_k.clamp(min=1e-8)
    xi   = ((x_safe - x_k) / w_k.clamp(min=1e-8)).clamp(0.0, 1.0)
    denom = (s_k + (d_k1 + d_k - 2.0 * s_k) * xi * (1.0 - xi)).clamp(min=1e-8)

    y_inside = y_k + h_k * (s_k * xi * xi + d_k * xi * (1.0 - xi)) / denom

    log_dydx = (
        2.0 * s_k.abs().clamp(min=1e-8).log()
        + (d_k1 * xi * xi + 2.0 * s_k * xi * (1.0 - xi) + d_k * (1.0 - xi) * (1.0 - xi)).clamp(min=1e-8).log()
        - 2.0 * denom.log()
    )

    return torch.where(inside, y_inside, x), torch.where(inside, log_dydx, torch.zeros_like(x))


class _SplineCouplingBlock(nn.Module):
    """Conditional rational-quadratic spline coupling block (Durkan et al. 2019).

    Replaces the affine transform x_b' = x_b*exp(s)+t with a per-dimension
    monotonic RQ spline conditioned on [x_a, c].  More expressive per block —
    can represent non-linear density contours that affine coupling cannot.
    """

    def __init__(
        self,
        d_in: int,
        n_context: int,
        hidden_dim: int,
        mask: torch.Tensor,
        dropout: float = 0.0,
        n_bins: int = 8,
        tail_bound: float = 5.0,
    ) -> None:
        super().__init__()
        perm, inv_perm, d_a = _mask_to_perm(mask)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)
        self.d_a = d_a
        self.n_bins = n_bins
        self.tail_bound = tail_bound

        d_b = d_in - d_a
        self.d_b = d_b

        # Network outputs K widths + K heights + (K-1) interior derivatives per dim
        out_features = d_b * (3 * n_bins - 1)
        self.net = nn.Sequential(
            nn.Linear(d_a + n_context, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _spline_params(
        self, x_a: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(torch.cat([x_a, c], dim=-1))              # [B, d_b*(3K-1)]
        raw = raw.reshape(-1, self.d_b, 3 * self.n_bins - 1)     # -1: keep batch dynamic for ONNX

        widths = F.softmax(raw[..., :self.n_bins], dim=-1) * (2.0 * self.tail_bound)
        heights = F.softmax(raw[..., self.n_bins:2 * self.n_bins], dim=-1) * (2.0 * self.tail_bound)
        deriv_interior = F.softplus(raw[..., 2 * self.n_bins:]) + _MIN_DERIVATIVE
        # Boundary derivatives pinned to 1.0 (C1-continuous linear tails). Derived
        # from deriv_interior so the batch dim stays dynamic for ONNX tracing.
        ones = torch.ones_like(deriv_interior[..., :1])
        derivs = torch.cat([ones, deriv_interior, ones], dim=-1)   # [B, d_b, K+1]
        return widths, heights, derivs

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xp = x.index_select(1, self.perm)             # [B, D] reordered: x_a | x_b
        x_a = xp[:, :self.d_a]
        x_b = xp[:, self.d_a:]

        widths, heights, derivs = self._spline_params(x_a, c)
        y_b, log_det_per_dim = _rqs_forward(x_b, widths, heights, derivs, self.tail_bound)

        yp = torch.cat([x_a, y_b], dim=-1)
        y = yp.index_select(1, self.inv_perm)
        return y, log_det_per_dim.sum(dim=-1)  # [B]

    def inverse(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # RQ-spline inversion needs a per-bin quadratic solve; not implemented
        # because scoring only uses the forward (data→latent) direction. Sampling
        # / generation is out of scope. Fail fast rather than crash obscurely.
        raise NotImplementedError(
            "_SplineCouplingBlock.inverse is not implemented (sampling not supported "
            "for spline coupling; only forward NLL scoring is used)."
        )


class PCFlowModel(nn.Module):
    """Conditional normalizing flow for PV anomaly detection.

    Args:
        n_features:        Dimensionality of x (non-context features).
        n_context:         Dimensionality of c (e.g. 2 for [irr, pvt]).
        n_coupling_layers: Number of coupling blocks (default 4).
        hidden_dim:        Hidden size of each coupling MLP (default 32).
        dropout:           Dropout rate inside coupling MLPs (default 0.0).
        coupling_type:     "affine" (RealNVP) or "spline" (RQ-NSF). Default "affine".
        spline_bins:       Number of spline bins when coupling_type="spline" (default 8).
        tail_bound:        Spline domain half-width; identity outside ±tail_bound (default 5.0).
    """

    def __init__(
        self,
        n_features: int,
        n_context: int,
        n_coupling_layers: int = 4,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        coupling_type: str = "affine",
        spline_bins: int = 8,
        tail_bound: float = 5.0,
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
            if coupling_type == "spline":
                self.blocks.append(
                    _SplineCouplingBlock(n_features, n_context, hidden_dim, mask, dropout, spline_bins, tail_bound)
                )
            else:
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
        d = self.n_features
        log_p_z = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * d * self._log_2pi
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
