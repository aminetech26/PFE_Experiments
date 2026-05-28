from __future__ import annotations

import torch
import torch.nn as nn


class PCAEModel(nn.Module):
    """Point-wise Physics-Conditioned Autoencoder.

    Stateless MLP encoder produces a latent z from a single timestep's sensor
    readings.  The decoder reconstructs the same sensors from (z, c), where
    c = (irradiance, module temperature) is the exogenous operating point.

    No recurrence, no attention, no temporal context inside the model.  The
    c-invariance regularizer (applied during training) forces z to encode only
    the operating-point-invariant part of the signal — the I-V manifold that
    PV faults break.
    """

    def __init__(
        self,
        n_features: int,
        n_context_features: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_context_features = n_context_features
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_context_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_features),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z, c)
        return {"z": z, "x_hat": x_hat}
