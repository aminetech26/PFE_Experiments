from __future__ import annotations

import torch
import torch.nn.functional as F


def maat_kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) summed over the last dimension."""
    eps = 1e-9
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)


def normalize_prior(prior: torch.Tensor) -> torch.Tensor:
    """Normalize prior distribution to sum to 1 over the last dimension."""
    return prior / (prior.sum(dim=-1, keepdim=True).clamp(min=1e-9))


def association_losses(
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-layer KL losses for the MAAT minimax objective."""
    series_loss: torch.Tensor = torch.tensor(0.0, device=series_list[0].device)
    prior_loss: torch.Tensor = torch.tensor(0.0, device=prior_list[0].device)

    for s, p in zip(series_list, prior_list):
        p_norm = normalize_prior(p)
        kl_sp = maat_kl_loss(s, p_norm.detach())
        kl_ps = maat_kl_loss(p_norm.detach(), s)
        series_loss = series_loss + (kl_sp + kl_ps).mean()

        kl_ps2 = maat_kl_loss(p_norm, s.detach())
        kl_sp2 = maat_kl_loss(s.detach(), p_norm)
        prior_loss = prior_loss + (kl_ps2 + kl_sp2).mean()

    n = len(series_list)
    return series_loss / n, prior_loss / n


def compute_maat_scores(
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
    recon_error: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute per-timestep MAAT product scores."""
    ass_dis = compute_association_discrepancy(series_list, prior_list)
    ass_dis = ass_dis * temperature
    metric = torch.softmax(-ass_dis, dim=-1)
    return metric * recon_error


def compute_association_discrepancy(
    series_list: list[torch.Tensor],
    prior_list: list[torch.Tensor],
) -> torch.Tensor:
    """Return per-timestep association discrepancy before temperature scaling."""
    kl_per_layer: list[torch.Tensor] = []
    for s, p in zip(series_list, prior_list):
        p_norm = normalize_prior(p)
        kl_sp = maat_kl_loss(s, p_norm)
        kl_ps = maat_kl_loss(p_norm, s)
        kl_per_layer.append((kl_sp + kl_ps).mean(dim=1))

    return torch.stack(kl_per_layer, dim=0).mean(dim=0)


def pairwise_margin_loss(
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Rank corrupted windows above clean windows for a family head."""
    return F.softplus(margin - (corrupt_logits - clean_logits)).mean()


def _uniform(
    batch_size: int,
    low: float,
    high: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(batch_size, device=device, dtype=dtype)


def _sample_span_mask(
    batch_size: int,
    window_size: int,
    *,
    device: torch.device,
    min_frac: float,
    max_frac: float,
) -> torch.Tensor:
    lengths = torch.round(
        _uniform(batch_size, min_frac * window_size, max_frac * window_size, device=device, dtype=torch.float32)
    ).to(torch.long).clamp(min=1, max=window_size)
    max_start = (window_size - lengths).clamp(min=0)
    starts = torch.floor(
        torch.rand(batch_size, device=device) * (max_start.to(torch.float32) + 1.0)
    ).to(torch.long)
    steps = torch.arange(window_size, device=device).unsqueeze(0)
    return (steps >= starts.unsqueeze(1)) & (steps < (starts + lengths).unsqueeze(1))


def _apply_shift(
    x: torch.Tensor,
    channel_idx: int,
    per_sample_shift: torch.Tensor,
    sample_mask: torch.Tensor,
    time_mask: torch.Tensor,
) -> None:
    active = sample_mask.unsqueeze(1) & time_mask
    x[:, :, channel_idx] = x[:, :, channel_idx] + active.to(x.dtype) * per_sample_shift.unsqueeze(1)


def build_factorized_corruptions(
    x: torch.Tensor,
    feature_idx: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Generate physics-grounded synthetic fault families from normal windows.

    Corruptions are created in standardized feature space so magnitudes are
    dataset-scaled but still structured:
    - voltage: localized single-string voltage collapse with coupled power/current drop
    - mismatch: persistent inter-string asymmetry across voltage/current/power
    - dynamic: post-break temporal rupture / flatline across electrical channels
    """
    batch_size, window_size, _ = x.shape
    device = x.device
    dtype = x.dtype

    voltage = x.clone()
    mismatch = x.clone()
    dynamic = x.clone()

    v_mask = _sample_span_mask(batch_size, window_size, device=device, min_frac=0.25, max_frac=0.55)
    m_mask = _sample_span_mask(batch_size, window_size, device=device, min_frac=0.50, max_frac=1.00)

    choose_first = torch.rand(batch_size, device=device) < 0.5
    choose_second = ~choose_first
    symmetric_voltage = torch.rand(batch_size, device=device) < 0.5
    asymmetric_voltage = ~symmetric_voltage
    asym_first = asymmetric_voltage & choose_first
    asym_second = asymmetric_voltage & choose_second

    v_amp = _uniform(batch_size, 2.0, 4.0, device=device, dtype=dtype)
    i_amp = _uniform(batch_size, 0.5, 1.5, device=device, dtype=dtype)
    p_amp = _uniform(batch_size, 1.0, 2.5, device=device, dtype=dtype)

    _apply_shift(voltage, feature_idx["vdc1"], -v_amp, asym_first, v_mask)
    _apply_shift(voltage, feature_idx["vdc2"], -v_amp, asym_second, v_mask)
    _apply_shift(voltage, feature_idx["idc1"], -i_amp, asym_first, v_mask)
    _apply_shift(voltage, feature_idx["idc2"], -i_amp, asym_second, v_mask)
    _apply_shift(voltage, feature_idx["pdc1"], -p_amp, asym_first, v_mask)
    _apply_shift(voltage, feature_idx["pdc2"], -p_amp, asym_second, v_mask)

    _apply_shift(voltage, feature_idx["vdc1"], -v_amp, symmetric_voltage, v_mask)
    _apply_shift(voltage, feature_idx["vdc2"], -v_amp, symmetric_voltage, v_mask)
    _apply_shift(voltage, feature_idx["idc1"], -i_amp, symmetric_voltage, v_mask)
    _apply_shift(voltage, feature_idx["idc2"], -i_amp, symmetric_voltage, v_mask)
    _apply_shift(voltage, feature_idx["pdc1"], -p_amp, symmetric_voltage, v_mask)
    _apply_shift(voltage, feature_idx["pdc2"], -p_amp, symmetric_voltage, v_mask)

    mis_i = _uniform(batch_size, 1.5, 3.0, device=device, dtype=dtype)
    mis_p = _uniform(batch_size, 1.5, 3.0, device=device, dtype=dtype)
    mis_v = _uniform(batch_size, 0.5, 1.5, device=device, dtype=dtype)

    partner_coeff = _uniform(batch_size, 0.0, 0.7, device=device, dtype=dtype)

    _apply_shift(mismatch, feature_idx["idc1"], -mis_i, choose_first, m_mask)
    _apply_shift(mismatch, feature_idx["idc2"], partner_coeff * mis_i, choose_first, m_mask)
    _apply_shift(mismatch, feature_idx["pdc1"], -mis_p, choose_first, m_mask)
    _apply_shift(mismatch, feature_idx["pdc2"], partner_coeff * mis_p, choose_first, m_mask)
    _apply_shift(mismatch, feature_idx["vdc1"], -mis_v, choose_first, m_mask)
    _apply_shift(mismatch, feature_idx["vdc2"], partner_coeff * mis_v, choose_first, m_mask)

    _apply_shift(mismatch, feature_idx["idc2"], -mis_i, choose_second, m_mask)
    _apply_shift(mismatch, feature_idx["idc1"], partner_coeff * mis_i, choose_second, m_mask)
    _apply_shift(mismatch, feature_idx["pdc2"], -mis_p, choose_second, m_mask)
    _apply_shift(mismatch, feature_idx["pdc1"], partner_coeff * mis_p, choose_second, m_mask)
    _apply_shift(mismatch, feature_idx["vdc2"], -mis_v, choose_second, m_mask)
    _apply_shift(mismatch, feature_idx["vdc1"], partner_coeff * mis_v, choose_second, m_mask)

    min_break = max(1, window_size // 4)
    max_break = max(min_break + 1, (3 * window_size) // 4)
    break_idx = torch.randint(min_break, max_break, (batch_size,), device=device)
    steps = torch.arange(window_size, device=device).unsqueeze(0)
    post_mask = steps >= break_idx.unsqueeze(1)
    boundary_vals = dynamic[torch.arange(batch_size, device=device), break_idx.clamp(max=window_size - 1)]
    dyn_channels = [
        feature_idx["vdc1"],
        feature_idx["vdc2"],
        feature_idx["idc1"],
        feature_idx["idc2"],
        feature_idx["pdc1"],
        feature_idx["pdc2"],
    ]
    for idx in dyn_channels:
        dynamic[:, :, idx] = torch.where(post_mask, boundary_vals[:, None, idx], dynamic[:, :, idx])

    dyn_drop = _uniform(batch_size, 1.0, 2.0, device=device, dtype=dtype)
    for idx in (feature_idx["idc1"], feature_idx["idc2"], feature_idx["pdc1"], feature_idx["pdc2"]):
        dynamic[:, :, idx] = dynamic[:, :, idx] - post_mask.to(dtype) * dyn_drop.unsqueeze(1)

    return {
        "voltage": voltage,
        "mismatch": mismatch,
        "dynamic": dynamic,
    }
