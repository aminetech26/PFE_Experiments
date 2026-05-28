from __future__ import annotations

import torch


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Per-sample feature-wise MSE. Returns [B]."""
    return ((x_hat - x) ** 2).mean(dim=-1)


def context_invariance_loss(z: torch.Tensor, z_aug: torch.Tensor) -> torch.Tensor:
    """MSE between z computed from x and z computed from x with c shuffled. Returns [B].

    Forces the latent to be invariant to the operating-point values embedded in
    the input.  When this loss is minimized, z encodes only the I-V manifold
    structure that PV faults break — irradiance/temperature changes alone do
    not move z.
    """
    return ((z - z_aug) ** 2).mean(dim=-1)


def shuffle_context_in_x(
    x: torch.Tensor, context_indices: list[int], generator: torch.Generator | None = None
) -> torch.Tensor:
    """Return a copy of x with the context-feature columns shuffled across the batch.

    Uses the empirical distribution of c from the same batch (no synthetic
    sampling), so x_aug stays inside the training data domain.
    """
    x_aug = x.clone()
    if not context_indices:
        return x_aug
    perm = torch.randperm(x.shape[0], device=x.device, generator=generator)
    idx = torch.tensor(context_indices, device=x.device, dtype=torch.long)
    x_aug[:, idx] = x[perm][:, idx]
    return x_aug


def compute_anomaly_scores(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Per-sample anomaly score = feature-averaged MSE. Returns [B]."""
    return reconstruction_loss(x_hat, x)
