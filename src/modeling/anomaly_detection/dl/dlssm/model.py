from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualMLP(nn.Module):
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


class DeepLatentStateSpaceModel(nn.Module):
    """Causal deep latent state-space model with optional physics-conditioned prior (PC-DLSSM).

    When n_context_features > 0, a second prior p_c(z|c_t) is parameterised by
    exogenous context (e.g. irradiance, module temperature).  This prior is used
    only for anomaly scoring; training uses the dynamics prior p_h(z|h_{t-1})
    via the standard ELBO, with an auxiliary loss that fits p_c to the posterior.
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
        se_reduction_ratio: int = 4,
        n_context_features: int = 0,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_context_features = n_context_features

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

        self.inference_net = nn.Sequential(
            _ResidualMLP(n_features + hidden_dim, encoder_dim, dropout),
            nn.Linear(encoder_dim, 2 * latent_dim),
        )

        self.prior_net = nn.Sequential(
            _ResidualMLP(hidden_dim, encoder_dim, dropout),
            nn.Linear(encoder_dim, 2 * latent_dim),
        )

        self.decoder = nn.Sequential(
            _ResidualMLP(latent_dim + hidden_dim, decoder_dim, dropout),
            nn.Linear(decoder_dim, n_features),
        )

        # Context prior: p_c(z | c_t).  Only present when context features are available.
        if n_context_features > 0:
            self.context_prior_net: nn.Module | None = nn.Sequential(
                _ResidualMLP(n_context_features, encoder_dim, dropout),
                nn.Linear(encoder_dim, 2 * latent_dim),
            )
        else:
            self.context_prior_net = None

    def _split_params(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar.clamp(-10.0, 10.0)

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x_proj = self.feature_attn(x)
        h_seq, _ = self.gru(x_proj)

        h_prev = torch.zeros_like(h_seq)
        h_prev[:, 1:] = h_seq[:, :-1]

        q_input = torch.cat([x, h_prev], dim=-1)
        q_mu, q_logvar = self._split_params(self.inference_net(q_input))

        p_mu, p_logvar = self._split_params(self.prior_net(h_prev))

        if self.training:
            z = q_mu + torch.randn_like(q_mu) * torch.exp(0.5 * q_logvar)
        else:
            z = q_mu

        dec_input = torch.cat([z, h_prev], dim=-1)
        x_hat = self.decoder(dec_input)

        result: dict[str, torch.Tensor] = {
            "x_hat": x_hat,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": p_mu,
            "p_logvar": p_logvar,
            "z": z,
            "h": h_seq,
            "h_prev": h_prev,
        }

        if c is not None and self.context_prior_net is not None:
            p_c_mu, p_c_logvar = self._split_params(self.context_prior_net(c))
            result["p_c_mu"] = p_c_mu
            result["p_c_logvar"] = p_c_logvar

        return result
