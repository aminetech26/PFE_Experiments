from __future__ import annotations

import torch
import torch.nn.functional as F


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


def inverse_standardize(
    x_scaled: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Invert StandardScaler normalization for [B, W, F] inputs."""
    return x_scaled * scale + mean


def physics_consistency_loss(
    x_hat_scaled: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_idx: dict[str, int],
    huber_delta: float = 0.05,
) -> torch.Tensor:
    """String-symmetry physics loss on reconstructed outputs in physical units.

    Enforces normal-manifold two-string symmetry on the two independent
    physical observables of the Costa DC array:

        vdc1 ~= vdc2,   idc1 ~= idc2

    The product symmetry pdc1 ~= pdc2 is *not* added as a separate term: in this
    pipeline pdc_k = vdc_k * idc_k is computed deterministically, so any
    pdc-side residual is an operating-point-weighted linear combination of the
    V- and I-residuals above and would double-count the same physics.

    Each residual is normalized by the train-normal std of the corresponding
    string-symmetric pair, so loss magnitudes are dimensionless and comparable
    across V and I channels.

    Returns a [B] per-window scalar.
    """
    x_hat_phys = inverse_standardize(x_hat_scaled, scaler_mean, scaler_scale)  # [B, W, F]
    bsz = x_hat_phys.size(0)
    loss = torch.zeros(bsz, device=x_hat_phys.device, dtype=x_hat_phys.dtype)
    n_components = 2

    vdc1 = x_hat_phys[:, :, feature_idx["vdc1"]]
    vdc2 = x_hat_phys[:, :, feature_idx["vdc2"]]
    idc1 = x_hat_phys[:, :, feature_idx["idc1"]]
    idc2 = x_hat_phys[:, :, feature_idx["idc2"]]

    vdc_scale = (
        (scaler_scale[feature_idx["vdc1"]] + scaler_scale[feature_idx["vdc2"]]) / 2.0
    ).abs().clamp(min=1.0)
    idc_scale = (
        (scaler_scale[feature_idx["idc1"]] + scaler_scale[feature_idx["idc2"]]) / 2.0
    ).abs().clamp(min=1.0)

    r_v = (vdc1 - vdc2) / vdc_scale
    r_i = (idc1 - idc2) / idc_scale

    loss = loss + F.huber_loss(
        r_v, torch.zeros_like(r_v), delta=huber_delta, reduction="none"
    ).mean(dim=1)
    loss = loss + F.huber_loss(
        r_i, torch.zeros_like(r_i), delta=huber_delta, reduction="none"
    ).mean(dim=1)

    loss = loss / n_components

    return loss
