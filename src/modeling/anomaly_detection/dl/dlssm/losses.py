from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_feature_mse(
    a: torch.Tensor, b: torch.Tensor, recon_mask: torch.Tensor | None
) -> torch.Tensor:
    """Per-timestep MSE averaged over features, optionally masking some dims out.

    recon_mask: [F] tensor of 0.0/1.0 (1.0 = include). When provided, the
    average is taken over kept features so loss magnitude stays comparable.
    """
    se = (a - b) ** 2  # [B, W, F]
    if recon_mask is None:
        return se.mean(dim=-1)
    keep = recon_mask.to(se.dtype)
    n_keep = keep.sum().clamp(min=1.0)
    return (se * keep).sum(dim=-1) / n_keep


def reconstruction_loss(
    x_hat: torch.Tensor, x: torch.Tensor, recon_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-timestep MSE averaged over (kept) features. Returns [B, W]."""
    return _masked_feature_mse(x_hat, x, recon_mask)


def prediction_loss(
    x_pred: torch.Tensor, x: torch.Tensor, recon_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Per-timestep MSE between one-step state prediction and observation. Returns [B, W].

    x_pred comes from prediction_head(h_prev[, c_t]) — no z_t, no current x_t.
    High values indicate the GRU state failed to predict normal PV dynamics.
    """
    return _masked_feature_mse(x_pred, x, recon_mask)


def kl_divergence(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, summed over latent dims. Returns [B, W].

    Closed-form: 0.5 * sum_Z(logvar_p - logvar_q + (var_q + (mu_q - mu_p)^2) / var_p - 1)
    free_bits: minimum KL per latent dim (prevents posterior collapse on easy dims).
    """
    q_logvar = q_logvar.clamp(-10.0, 10.0)
    p_logvar = p_logvar.clamp(-10.0, 10.0)
    q_var = torch.exp(q_logvar)
    p_var = torch.exp(p_logvar).clamp(min=1e-8)
    kl = 0.5 * (
        p_logvar - q_logvar
        + (q_var + (q_mu - p_mu) ** 2) / p_var
        - 1.0
    )  # [B, W, Z]
    if free_bits > 0.0:
        kl = kl.clamp(min=free_bits)
    return kl.sum(dim=-1)  # [B, W]


def inverse_standardize(
    x_scaled: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Invert StandardScaler normalization.

    x_scaled: [B, W, F], mean/scale: [F] (registered buffers on the correct device)
    """
    return x_scaled * scale + mean


def physics_consistency_loss(
    x_hat_scaled: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_idx: dict[str, int],
    enable_string_power: bool = True,
    enable_imbalance: bool = True,
) -> torch.Tensor:
    """Physics consistency loss on reconstructed outputs in physical units.

    Phase 1 — string power (P = V * I):
      residual_i = (pdc_i_hat - vdc_i_hat * idc_i_hat) / pdc_scale
      L_string = Huber(r1).mean(W) + Huber(r2).mean(W)

    Phase 2 — imbalance consistency (requires enable_imbalance=True):
      Computes expected imbalance ratios from raw signals and penalizes
      deviation of the decoded imbalance features from those ratios.

    Returns [B] per-window scalar. Components with missing features are silently
    skipped; if nothing can be computed, returns zeros.

    Physics constraints are only valid in physical (un-scaled) units.
    """
    x_hat_phys = inverse_standardize(x_hat_scaled, scaler_mean, scaler_scale)  # [B, W, F]
    B = x_hat_phys.size(0)
    loss = torch.zeros(B, device=x_hat_phys.device, dtype=x_hat_phys.dtype)
    n_components = 0

    def _get(name: str) -> torch.Tensor | None:
        idx = feature_idx.get(name)
        return x_hat_phys[:, :, idx] if idx is not None else None  # [B, W] or None

    pdc1 = _get("pdc1")
    pdc2 = _get("pdc2")
    vdc1 = _get("vdc1")
    vdc2 = _get("vdc2")
    idc1 = _get("idc1")
    idc2 = _get("idc2")

    # Phase 1: P = V * I for each string
    if enable_string_power and all(v is not None for v in (pdc1, vdc1, idc1, pdc2, vdc2, idc2)):
        mean_pdc1 = scaler_mean[feature_idx["pdc1"]]
        mean_pdc2 = scaler_mean[feature_idx["pdc2"]]
        pdc_scale = (mean_pdc1 + mean_pdc2).abs().clamp(min=1.0)

        r1 = (pdc1 - vdc1 * idc1) / pdc_scale              # [B, W]
        r2 = (pdc2 - vdc2 * idc2) / pdc_scale              # [B, W]
        loss = loss + F.huber_loss(r1, torch.zeros_like(r1), delta=1.0, reduction="none").mean(1)
        loss = loss + F.huber_loss(r2, torch.zeros_like(r2), delta=1.0, reduction="none").mean(1)
        n_components += 2

    # Phase 2: Imbalance consistency
    if enable_imbalance:
        eps = 1e-8

        pow_imb = _get("power_imbalance")
        if all(v is not None for v in (pow_imb, pdc1, pdc2)):
            calc = (pdc1 - pdc2).abs() / (pdc1.abs() + pdc2.abs() + eps)
            loss = loss + F.huber_loss(pow_imb, calc, delta=0.1, reduction="none").mean(1)
            n_components += 1

        cur_imb = _get("current_imbalance")
        if all(v is not None for v in (cur_imb, idc1, idc2)):
            calc = (idc1 - idc2).abs() / (idc1.abs() + idc2.abs() + eps)
            loss = loss + F.huber_loss(cur_imb, calc, delta=0.1, reduction="none").mean(1)
            n_components += 1

        vol_imb = _get("voltage_imbalance")
        if all(v is not None for v in (vol_imb, vdc1, vdc2)):
            calc = (vdc1 - vdc2).abs() / (vdc1.abs() + vdc2.abs() + eps)
            loss = loss + F.huber_loss(vol_imb, calc, delta=0.1, reduction="none").mean(1)
            n_components += 1

    if n_components > 0:
        loss = loss / n_components

    return loss  # [B]


def one_class_loss(q_mu: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Deep SVDD-style compactness: pulls normal latent means toward center c.

    q_mu: [B, W, Z]; c: [Z] (registered buffer, set after warmup).
    Returns [B] window-mean squared distance.
    """
    return ((q_mu - c) ** 2).sum(dim=-1).mean(dim=1)


def consistency_loss(q_mu_x: torch.Tensor, q_mu_aug: torch.Tensor) -> torch.Tensor:
    """Consistency regularization: q_mu(x) ≈ q_mu(x_aug).

    NOT Mean Teacher: there is no EMA teacher network — both forward passes
    use the same (student) weights. The clean branch is detached only to
    stop the augmented branch from pulling q_mu(x) toward itself, mirroring
    the stop-gradient pattern from BYOL/SimSiam-style consistency.
    Returns [B].
    """
    return ((q_mu_x.detach() - q_mu_aug) ** 2).sum(dim=-1).mean(dim=1)


def apply_pv_safe_augmentation(
    x: torch.Tensor,
    noise_std: float = 0.05,
    feature_dropout: float = 0.1,
    protect_idx: list[int] | None = None,
) -> torch.Tensor:
    """PV-safe perturbation on standardized inputs [B, W, F].

    - Gaussian sensor noise (std on the standardized scale).
    - Per-feature dropout: zeroes whole features for the entire window.
      Because inputs are standardized, zero == feature mean, so this
      simulates *mean-imputed* sensor failure rather than a true stuck
      reading or missing-data signal.
    - protect_idx: feature indices to leave untouched. Required for CVAE
      conditioning columns — perturbing them would shift the operating
      regime, breaking the consistency contract q(z|x,c) ≈ q(z|x_aug,c).
    """
    noise = torch.randn_like(x) * noise_std
    if protect_idx:
        noise[:, :, protect_idx] = 0.0
    out = x + noise

    if feature_dropout > 0.0:
        mask = (torch.rand(x.size(0), 1, x.size(2), device=x.device) > feature_dropout).float()
        if protect_idx:
            mask[:, :, protect_idx] = 1.0  # never drop conditioning features
        out = out * mask
    return out


def self_paced_weights(
    recon_w: torch.Tensor,
    phys_w: torch.Tensor,
    lambda_phys: float,
    tau: float,
    w_min: float = 0.2,
) -> torch.Tensor:
    """Per-window self-paced curriculum weights.

    Difficulty d_i = recon_i + lambda_phys * phys_i (both detached).
    Easy windows (low d_i) get high weight early; tau anneals toward 1 to
    progressively include harder windows.

    Returns [B] weights normalized to unit mean, clamped to [w_min, inf].
    """
    d = recon_w.detach() + lambda_phys * phys_w.detach()  # [B]
    w = torch.exp(-d / max(float(tau), 1e-6))
    # Clamp first so w_min is respected before normalization;
    # then renormalize so mean stays near 1 and loss scale is stable.
    w = w.clamp(min=w_min)
    return w / (w.mean() + 1e-8)


def compute_anomaly_scores(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    lambda_kl_score: float = 0.1,
    score_reduction: str = "center",
    lambda_phys_score: float = 0.0,
    scaler_mean: torch.Tensor | None = None,
    scaler_scale: torch.Tensor | None = None,
    feature_idx: dict[str, int] | None = None,
    enable_string_power: bool = True,
    enable_imbalance: bool = True,
    x_pred: torch.Tensor | None = None,
    lambda_pred_score: float = 0.0,
    c: torch.Tensor | None = None,
    lambda_oc_score: float = 0.0,
    recon_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-window anomaly score combining three complementary signals.

    recon_error       — how poorly the model reconstructs the observation
    latent_innovation — KL(q || p): how much z_t deviates from predicted latent dynamics
    pred_error        — how poorly the GRU state alone predicts the observation (optional)
    physics_residual  — physics constraint violation on reconstructed outputs (optional)

    Returns [B] scalar scores. Higher = more anomalous.
    score_reduction: "center" (default) | "mean" | "max"
    Physics and pred terms use window-level aggregation regardless of score_reduction.
    """

    def _reduce(t: torch.Tensor) -> torch.Tensor:
        if score_reduction == "mean":
            return t.mean(dim=1)
        if score_reduction == "max":
            return t.max(dim=1).values
        return t[:, t.size(1) // 2]  # center timestep

    recon_t = reconstruction_loss(x_hat, x, recon_mask=recon_mask)        # [B, W]
    kl_t = kl_divergence(q_mu, q_logvar, p_mu, p_logvar)                 # [B, W]
    base = _reduce(recon_t + lambda_kl_score * kl_t)                     # [B]

    if lambda_pred_score > 0.0 and x_pred is not None:
        base = base + lambda_pred_score * _reduce(prediction_loss(x_pred, x, recon_mask=recon_mask))

    if lambda_oc_score > 0.0 and c is not None:
        # Deep SVDD distance: how far q_mu lies from the normal latent center
        oc_t = ((q_mu - c) ** 2).sum(dim=-1)  # [B, W]
        base = base + lambda_oc_score * _reduce(oc_t)

    if (
        lambda_phys_score > 0.0
        and scaler_mean is not None
        and scaler_scale is not None
        and feature_idx is not None
    ):
        phys_s = physics_consistency_loss(
            x_hat, scaler_mean, scaler_scale, feature_idx,
            enable_string_power=enable_string_power,
            enable_imbalance=enable_imbalance,
        )
        base = base + lambda_phys_score * phys_s  # phys_s is already [B] window-mean

    return base
