from __future__ import annotations

import torch


def reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Per-timestep MSE averaged over features. Returns [B, W]."""
    return ((x_hat - x) ** 2).mean(dim=-1)


def kl_divergence(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, summed over latent dims. Returns [B, W]."""
    q_logvar = q_logvar.clamp(-10.0, 10.0)
    p_logvar = p_logvar.clamp(-10.0, 10.0)
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar).clamp(min=1e-8)
    kl = 0.5 * (p_logvar - q_logvar + (q_var + (q_mu - p_mu) ** 2) / p_var - 1.0)
    if free_bits > 0.0:
        kl = kl.clamp(min=free_bits)
    return kl.sum(dim=-1)


def _reduce_score_window(t: torch.Tensor, score_reduction: str) -> torch.Tensor:
    if score_reduction == "mean":
        return t.mean(dim=1)
    if score_reduction == "max":
        return t.max(dim=1).values
    return t[:, t.size(1) // 2]


def _sanitize_score_component(component: torch.Tensor, max_abs: float | None) -> torch.Tensor:
    out = torch.nan_to_num(component, nan=1e6, posinf=1e6, neginf=-1e6)
    if max_abs is not None and max_abs > 0.0:
        out = out.clamp(min=-max_abs, max=max_abs)
    return out


def compute_anomaly_scores(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    lambda_kl_score: float = 0.1,
    score_reduction: str = "center",
    max_abs_base_score_term: float | None = None,
) -> torch.Tensor:
    recon_t = reconstruction_loss(x_hat, x)
    kl_t = kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
    base = _reduce_score_window(recon_t + lambda_kl_score * kl_t, score_reduction)
    return _sanitize_score_component(base, max_abs_base_score_term)
