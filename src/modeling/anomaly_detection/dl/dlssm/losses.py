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


def expected_voltage_loss(
    p_expected: torch.Tensor,
    x: torch.Tensor,
    expected_power_indices: list[int],
    tau: float = 0.15,
) -> torch.Tensor:
    """Pinball loss training p_expected to predict the τ-th percentile lower bound of normal vdc.

    With τ=0.15, p_expected learns the 15th-percentile of normal vdc given (irr, pvt).
    Normal windows exceed p_expected ~85% of the time; genuinely degraded panels (series
    resistance fault, class 2) produce a sustained voltage drop that falls below this bound,
    making ReLU(p_expected − actual) a strong, low-noise anomaly signal.

    p_expected: [B, W, n_voltage] — head predictions for [vdc1, vdc2] (or [vdc])
    x:          [B, W, F]         — standardized observations
    Returns [B] window-mean pinball loss. Silently returns zeros if indices missing.
    """
    if not expected_power_indices or p_expected is None:
        return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    x_voltage = x[:, :, expected_power_indices]                          # [B, W, n_voltage]
    residual = x_voltage - p_expected                                    # positive = above bound (normal)
    pinball = tau * F.relu(residual) + (1.0 - tau) * F.relu(-residual)  # [B, W, n_voltage]
    return pinball.mean(dim=-1).mean(dim=1)                              # [B]


# Back-compat alias — existing call sites using the old name continue to work
expected_power_loss = expected_voltage_loss


def performance_gap_score(
    p_expected: torch.Tensor,
    x: torch.Tensor,
    expected_power_indices: list[int],
    score_reduction: str = "mean",  # kept for API compat; always uses mean (degradation is sustained)
) -> torch.Tensor:
    """Asymmetric performance-ratio score: ReLU(P_expected − P_observed).

    Penalises under-performance only — the signature of slow degradation (class 2).
    Always uses mean reduction: degradation is a persistent drift across the full window,
    so mean accumulates the signal correctly regardless of the global score_reduction setting.
    p_expected: [B, W, n_power]; x: [B, W, F]. Returns [B].
    """
    if not expected_power_indices or p_expected is None:
        return torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    x_power = x[:, :, expected_power_indices]  # [B, W, n_power]
    gap = F.relu(p_expected - x_power)         # [B, W, n_power] — only under-performance
    gap_t = gap.mean(dim=-1)                   # [B, W]
    return gap_t.mean(dim=1)                   # always mean — sustained drift, not peak event


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


def compute_anomaly_score_components(
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
    p_expected: torch.Tensor | None = None,
    lambda_pr_score: float = 0.0,
    expected_power_indices: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Return raw, pre-sanitization score components for diagnostics and scoring.

    Values are unweighted except ``base``, which already includes ``lambda_kl_score``
    because reconstruction and KL form the core anomaly score.
    """
    components: dict[str, torch.Tensor] = {}
    recon_t = reconstruction_loss(x_hat, x, recon_mask=recon_mask)
    kl_t = kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
    components["base"] = _reduce_score_window(recon_t + lambda_kl_score * kl_t, score_reduction)

    if lambda_pred_score > 0.0 and x_pred is not None:
        pred_t = prediction_loss(x_pred, x, recon_mask=recon_mask)
        components["pred"] = _reduce_score_window(pred_t, score_reduction)

    if lambda_oc_score > 0.0 and c is not None:
        oc_t = ((q_mu - c) ** 2).sum(dim=-1)
        components["oc"] = _reduce_score_window(oc_t, score_reduction)

    if (
        lambda_phys_score > 0.0
        and scaler_mean is not None
        and scaler_scale is not None
        and feature_idx is not None
    ):
        components["phys"] = physics_consistency_loss(
            x_hat,
            scaler_mean,
            scaler_scale,
            feature_idx,
            enable_string_power=enable_string_power,
            enable_imbalance=enable_imbalance,
        )

    if lambda_pr_score > 0.0 and p_expected is not None and expected_power_indices:
        components["pr"] = performance_gap_score(
            p_expected, x, expected_power_indices, score_reduction
        )

    return components


def combine_anomaly_score_components(
    components: dict[str, torch.Tensor],
    lambda_pred_score: float = 0.0,
    lambda_oc_score: float = 0.0,
    lambda_phys_score: float = 0.0,
    lambda_pr_score: float = 0.0,
    max_abs_base_score_term: float | None = None,
    max_abs_pred_score_term: float | None = None,
    max_abs_oc_score_term: float | None = None,
    max_abs_phys_score_term: float | None = None,
    max_abs_pr_score_term: float | None = None,
) -> torch.Tensor:
    """Combine raw components into a finite score for metric computation."""
    base = _sanitize_score_component(components["base"], max_abs_base_score_term)
    if "pred" in components:
        base = base + lambda_pred_score * _sanitize_score_component(
            components["pred"], max_abs_pred_score_term
        )
    if "oc" in components:
        base = base + lambda_oc_score * _sanitize_score_component(
            components["oc"], max_abs_oc_score_term
        )
    if "phys" in components:
        base = base + lambda_phys_score * _sanitize_score_component(
            components["phys"], max_abs_phys_score_term
        )
    if "pr" in components:
        base = base + lambda_pr_score * _sanitize_score_component(
            components["pr"], max_abs_pr_score_term
        )
    return base


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
    p_expected: torch.Tensor | None = None,
    lambda_pr_score: float = 0.0,
    expected_power_indices: list[int] | None = None,
    max_abs_base_score_term: float | None = None,
    max_abs_pred_score_term: float | None = None,
    max_abs_oc_score_term: float | None = None,
    max_abs_phys_score_term: float | None = None,
    max_abs_pr_score_term: float | None = None,
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

    components = compute_anomaly_score_components(
        x=x,
        x_hat=x_hat,
        q_mu=q_mu,
        q_logvar=q_logvar,
        p_mu=p_mu,
        p_logvar=p_logvar,
        lambda_kl_score=lambda_kl_score,
        score_reduction=score_reduction,
        lambda_phys_score=lambda_phys_score,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_idx=feature_idx,
        enable_string_power=enable_string_power,
        enable_imbalance=enable_imbalance,
        x_pred=x_pred,
        lambda_pred_score=lambda_pred_score,
        c=c,
        lambda_oc_score=lambda_oc_score,
        recon_mask=recon_mask,
        p_expected=p_expected,
        lambda_pr_score=lambda_pr_score,
        expected_power_indices=expected_power_indices,
    )
    return combine_anomaly_score_components(
        components,
        lambda_pred_score=lambda_pred_score,
        lambda_oc_score=lambda_oc_score,
        lambda_phys_score=lambda_phys_score,
        lambda_pr_score=lambda_pr_score,
        max_abs_base_score_term=max_abs_base_score_term,
        max_abs_pred_score_term=max_abs_pred_score_term,
        max_abs_oc_score_term=max_abs_oc_score_term,
        max_abs_phys_score_term=max_abs_phys_score_term,
        max_abs_pr_score_term=max_abs_pr_score_term,
    )
