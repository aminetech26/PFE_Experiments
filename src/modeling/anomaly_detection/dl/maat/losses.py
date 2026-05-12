from __future__ import annotations

import torch


def maat_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) summed over the last dimension.

    Numerically stable: uses log(p + eps) - log(q + eps).
    Input shapes: [..., W, W] → output: [..., W].
    """
    eps = 1e-9
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


def normalize_prior(prior: torch.Tensor) -> torch.Tensor:
    """Normalize prior distribution to sum to 1 over the last dimension."""
    return prior / (prior.sum(dim=-1, keepdim=True).clamp(min=1e-9))


def association_losses(
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-layer KL losses for the minimax objective.

    Returns:
        series_loss: KL(S || P.detach()) + KL(P.detach() || S) — gradients through S only.
        prior_loss:  KL(P || S.detach()) + KL(S.detach() || P) — gradients through P only.
    Both are scalar means over layers, heads, positions, and batch.
    """
    series_loss: torch.Tensor = torch.tensor(0.0, device=series_list[0].device)
    prior_loss: torch.Tensor = torch.tensor(0.0, device=prior_list[0].device)

    for s, p in zip(series_list, prior_list):
        p_norm = normalize_prior(p)

        # series_loss: gradients only through S
        kl_sp = maat_kl_loss(s, p_norm.detach())          # [B, H, W]
        kl_ps = maat_kl_loss(p_norm.detach(), s)          # [B, H, W]
        series_loss = series_loss + (kl_sp + kl_ps).mean()

        # prior_loss: gradients only through P
        kl_ps2 = maat_kl_loss(p_norm, s.detach())         # [B, H, W]
        kl_sp2 = maat_kl_loss(s.detach(), p_norm)         # [B, H, W]
        prior_loss = prior_loss + (kl_ps2 + kl_sp2).mean()

    n = len(series_list)
    return series_loss / n, prior_loss / n


def compute_maat_scores(
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
    recon_error: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute per-window MAAT anomaly scores.

    Args:
        series_list: list of [B, H, W, W] tensors (one per encoder layer)
        prior_list:  list of [B, H, W, W] tensors (one per encoder layer)
        recon_error: [B, W] per-timestep MSE
        temperature: scaling factor for association discrepancy

    Returns:
        scores: [B, W] — higher = more anomalous
    """
    ass_dis = compute_association_discrepancy(series_list, prior_list)
    ass_dis = ass_dis * temperature

    metric = torch.softmax(-ass_dis, dim=-1)  # [B, W]
    return metric * recon_error               # [B, W]


def compute_association_discrepancy(
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
) -> torch.Tensor:
    """Return per-timestep association discrepancy before temperature scaling."""
    kl_per_layer: list[torch.Tensor] = []
    for s, p in zip(series_list, prior_list):
        p_norm = normalize_prior(p)
        kl_sp = maat_kl_loss(s, p_norm)    # [B, H, W]
        kl_ps = maat_kl_loss(p_norm, s)    # [B, H, W]
        kl_per_layer.append((kl_sp + kl_ps).mean(dim=1))  # [B, W]

    return torch.stack(kl_per_layer, dim=0).mean(dim=0)  # [B, W]
