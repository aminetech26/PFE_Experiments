from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    import pandas as pd


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
    enable_symmetry: bool = True,
    lambda_symmetry: float = 1.0,
    enable_v_under: bool = False,
    lambda_v_under: float = 0.0,
    vmpp_baseline: torch.Tensor | None = None,
    irr_floor: float = 1.0,
) -> torch.Tensor:
    """Composite PV physics consistency loss on reconstructed outputs.

    Two independent terms, weighted and summed per-window. Each term operates on
    ``x̂`` after inverse-standardization to physical units; both are dimensionless
    (residuals normalized by string-symmetric train-normal std).

    Term 1 — string symmetry (``enable_symmetry``):
        Enforces ``vdc1 ≈ vdc2`` and ``idc1 ≈ idc2`` on the reconstruction.
        Catches asymmetric DC-side mechanisms (single-string short circuit,
        open circuit, asymmetric shading). The product symmetry ``pdc1 ≈ pdc2``
        is *not* a separate term: ``pdc_k = vdc_k · idc_k`` is computed
        deterministically in this pipeline, so adding it would double-count the
        same physics with operating-point weighting.

    Term 2 — gray-box V_mpp under-voltage anchor (``enable_v_under``):
        Enforces ``x̂[vdc_k] ≥ a + b · pvt + c · ln(irr)`` via a one-sided
        ReLU residual. Functional form is the V_oc / V_mpp temperature-and-
        log-irradiance dependence; constants ``(a, b, c)`` are calibrated on
        training-normal data via :func:`fit_vmpp_baseline`. Catches symmetric
        voltage-loss mechanisms (resistive degradation) that string symmetry
        cannot see by construction. Over-voltage is not penalized.

    Args:
        x_hat_scaled:    Reconstruction in standardized space, shape ``[B, W, F]``.
        scaler_mean:     ``StandardScaler.mean_`` as a tensor, shape ``[F]``.
        scaler_scale:    ``StandardScaler.scale_`` as a tensor, shape ``[F]``.
        feature_idx:     Map from feature name to column index in ``x_hat_scaled``.
                         When ``enable_symmetry``, requires
                         ``vdc1, vdc2, idc1, idc2``. When ``enable_v_under``,
                         additionally requires ``irr, pvt``.
        huber_delta:     Huber transition point applied to each normalized residual.
        enable_symmetry: If False, skip the symmetry term entirely.
        lambda_symmetry: Weight for the symmetry term in the returned scalar.
        enable_v_under:  If False, skip the V_mpp anchor term entirely.
        lambda_v_under:  Weight for the anchor term in the returned scalar.
        vmpp_baseline:   Tensor ``[a, b, c]`` from :func:`fit_vmpp_baseline`.
                         Required when ``enable_v_under`` is True.
        irr_floor:       Minimum irradiance for the log; clamps both training
                         (via :func:`fit_vmpp_baseline`) and inference reconstruction.
                         Costa is already filtered to ``irr ≥ 100`` at ingestion;
                         the floor is a numeric safety net for ``log(x̂[irr])``.

    Returns:
        Per-window scalar loss, shape ``[B]``. Already weighted; ready to add
        directly to the optimizer's total loss.
    """
    if not enable_symmetry and not enable_v_under:
        return torch.zeros(
            x_hat_scaled.size(0),
            device=x_hat_scaled.device,
            dtype=x_hat_scaled.dtype,
        )

    x_hat_phys = inverse_standardize(x_hat_scaled, scaler_mean, scaler_scale)  # [B, W, F]
    bsz = x_hat_phys.size(0)
    loss = torch.zeros(bsz, device=x_hat_phys.device, dtype=x_hat_phys.dtype)

    if enable_symmetry:
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

        sym_per_window = 0.5 * (
            F.huber_loss(r_v, torch.zeros_like(r_v), delta=huber_delta, reduction="none").mean(dim=1)
            + F.huber_loss(r_i, torch.zeros_like(r_i), delta=huber_delta, reduction="none").mean(dim=1)
        )
        loss = loss + lambda_symmetry * sym_per_window

    if enable_v_under:
        if vmpp_baseline is None:
            raise ValueError("enable_v_under=True requires vmpp_baseline tensor [a, b, c].")

        a = vmpp_baseline[0]
        b = vmpp_baseline[1]
        c = vmpp_baseline[2]

        irr_phys = x_hat_phys[:, :, feature_idx["irr"]].clamp(min=irr_floor)
        pvt_phys = x_hat_phys[:, :, feature_idx["pvt"]]
        log_irr = torch.log(irr_phys)
        vdc_pred = a + b * pvt_phys + c * log_irr  # [B, W]

        vdc1 = x_hat_phys[:, :, feature_idx["vdc1"]]
        vdc2 = x_hat_phys[:, :, feature_idx["vdc2"]]

        vdc_scale = (
            (scaler_scale[feature_idx["vdc1"]] + scaler_scale[feature_idx["vdc2"]]) / 2.0
        ).abs().clamp(min=1.0)

        # One-sided: penalize only when reconstruction is below the V_mpp baseline.
        r1 = torch.relu(vdc_pred - vdc1) / vdc_scale
        r2 = torch.relu(vdc_pred - vdc2) / vdc_scale

        v_under_per_window = 0.5 * (
            F.huber_loss(r1, torch.zeros_like(r1), delta=huber_delta, reduction="none").mean(dim=1)
            + F.huber_loss(r2, torch.zeros_like(r2), delta=huber_delta, reduction="none").mean(dim=1)
        )
        loss = loss + lambda_v_under * v_under_per_window

    return loss


def fit_vmpp_baseline(
    train_df: "pd.DataFrame",
    irr_floor: float = 1.0,
) -> tuple[tuple[float, float, float], dict[str, float]]:
    """Calibrate gray-box V_mpp(irr, pvt) baseline on training-normal data.

    Functional form (physics-grounded):

        vdc ≈ a + b · pvt + c · ln(max(irr, irr_floor))

    V_oc has linear-in-temperature and logarithmic-in-irradiance dependence
    (Shockley diode equation with temperature dependence of the saturation
    current); V_mpp tracks V_oc closely under MPPT. The form is dictated by
    PV physics; the constants ``(a, b, c)`` are calibrated to data so the
    baseline does not require module datasheet specifics.

    Both strings ``vdc1`` and ``vdc2`` are stacked into a single fit (shared
    coefficients), consistent with the symmetry assumption used in the loss.

    Args:
        train_df: DataFrame in *physical units* containing columns
                  ``vdc1, vdc2, irr, pvt``. Expected to be normal-only
                  (semi-supervised convention).
        irr_floor: Lower bound applied before taking the log. Matches the
                   Costa ingestion floor (100 W/m²) is unnecessary here:
                   1.0 is enough as a numerical safety net.

    Returns:
        Tuple ``((a, b, c), diagnostics)`` where ``diagnostics`` contains
        ``r2``, ``rmse_phys``, ``residual_std_phys``, ``residual_p99_phys``,
        ``mean_vdc_phys``, ``std_vdc_phys``, ``n_samples`` for first-run
        sanity checking.
    """
    required = {"vdc1", "vdc2", "irr", "pvt"}
    missing = required - set(train_df.columns)
    if missing:
        raise KeyError(f"fit_vmpp_baseline: missing columns {sorted(missing)} in train_df.")

    irr = train_df["irr"].to_numpy(dtype=np.float64)
    pvt = train_df["pvt"].to_numpy(dtype=np.float64)
    vdc1 = train_df["vdc1"].to_numpy(dtype=np.float64)
    vdc2 = train_df["vdc2"].to_numpy(dtype=np.float64)

    log_irr = np.log(np.clip(irr, a_min=irr_floor, a_max=None))

    # Stack both strings as separate samples sharing (a, b, c).
    y = np.concatenate([vdc1, vdc2])
    pvt_stack = np.concatenate([pvt, pvt])
    log_irr_stack = np.concatenate([log_irr, log_irr])
    X = np.column_stack([np.ones_like(y), pvt_stack, log_irr_stack])

    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a_c, b_c, c_c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    pred = X @ coeffs
    residuals = y - pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / len(y)))
    residual_std = float(residuals.std(ddof=1))
    residual_p99 = float(np.quantile(np.abs(residuals), 0.99))

    diagnostics: dict[str, float] = {
        "r2": r2,
        "rmse_phys": rmse,
        "residual_std_phys": residual_std,
        "residual_p99_phys": residual_p99,
        "mean_vdc_phys": float(y.mean()),
        "std_vdc_phys": float(y.std(ddof=1)),
        "n_samples": float(len(y) // 2),  # number of timesteps (each contributes vdc1, vdc2)
        "irr_floor": float(irr_floor),
    }
    return (a_c, b_c, c_c), diagnostics
