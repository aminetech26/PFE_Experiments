"""Shared threshold calibration for anomaly detection.

All baseline models (PC-AE, PC-DLSSM, MAAT, OC-SVM, IForest, BOCD) use the same
calibration procedure so cross-model comparisons are apples-to-apples. Strategy
and parameters are configured once under `anomaly_detection.threshold` in
configs/model_config.yaml and applied uniformly.

Strategies:
- "gpd"  : Peaks-Over-Threshold + Generalized Pareto Distribution fit (default).
           Extrapolates the score tail rather than relying on a single empirical
           quantile. More principled when targeting a low FAR.
- "quantile" : Threshold = q-th percentile of normal-class scores.

Input is always a 1D array of normal-class scores (higher = more anomalous).
The choice of which normal scores to use (train-time or val-time) is left to
each trainer because some models fit on normal-only training data and naturally
produce train scores, while others (e.g., BOCD applied to a sequence) more
naturally produce val-normal scores. The calibration *procedure* itself is
identical.
"""
from __future__ import annotations

import numpy as np
from loguru import logger


def calibrate_threshold_quantile(
    normal_scores: np.ndarray, percentile: float = 95.0
) -> tuple[float, dict]:
    """Threshold = q-th percentile of normal-class scores."""
    thr = float(np.percentile(normal_scores, percentile))
    diag = {
        "strategy": "quantile",
        "percentile": percentile,
        "n_normal_scores": int(len(normal_scores)),
        "threshold": thr,
    }
    return thr, diag


def calibrate_threshold_gpd(
    normal_scores: np.ndarray,
    pot_quantile: float = 0.90,
    target_tail_prob: float = 0.05,
    min_exceedances: int = 30,
) -> tuple[float, dict]:
    """Peaks-Over-Threshold + GPD threshold calibration.

    Standard EVT procedure:
      1. u = q_{pot_quantile}(normal_scores)  — start of the tail
      2. Y = (normal_scores[normal_scores > u]) - u — positive exceedances
      3. Fit GPD(ξ, σ) to Y via MLE.
      4. Solve for threshold t with P(score > t) = target_tail_prob:
             t = u + (σ/ξ) * ((Nu/N * 1/p)^ξ - 1)   if ξ ≠ 0
             t = u + σ * log(Nu/N * 1/p)             if ξ → 0

    Falls back to quantile calibration if exceedance count is too small.
    """
    from scipy.stats import genpareto

    n = len(normal_scores)
    u = float(np.quantile(normal_scores, pot_quantile))
    exceedances = normal_scores[normal_scores > u] - u
    n_u = len(exceedances)

    if n_u < min_exceedances:
        logger.warning(
            "GPD threshold: only {} exceedances above u={:.4f} (q{:.2f}). "
            "Falling back to quantile.", n_u, u, pot_quantile,
        )
        return calibrate_threshold_quantile(normal_scores, 100.0 * (1.0 - target_tail_prob))

    xi, _, sigma = genpareto.fit(exceedances, floc=0.0)
    rate = n_u / n
    p = float(target_tail_prob)
    if abs(xi) < 1e-6:
        threshold = u + sigma * np.log(rate / p)
    else:
        threshold = u + (sigma / xi) * (np.power(rate / p, xi) - 1.0)
    threshold = float(threshold)

    diag = {
        "strategy": "gpd",
        "pot_quantile": pot_quantile,
        "pot_threshold_u": u,
        "n_exceedances": n_u,
        "n_normal_scores": n,
        "exceedance_rate": rate,
        "gpd_xi_shape": float(xi),
        "gpd_sigma_scale": float(sigma),
        "target_tail_prob": p,
        "threshold": threshold,
    }
    logger.info(
        "GPD calibration: u={:.4f} (q{:.2f}, {} exceedances) | ξ={:.4f} σ={:.4f} | "
        "target P>{:.4f}={:.4f} → threshold={:.4f}",
        u, pot_quantile, n_u, xi, sigma, p, p, threshold,
    )
    return threshold, diag


def calibrate_threshold(
    normal_scores: np.ndarray,
    strategy: str = "gpd",
    *,
    percentile: float = 95.0,
    pot_quantile: float = 0.90,
    target_tail_prob: float = 0.05,
) -> tuple[float, dict]:
    """Dispatch by strategy. Returns (threshold, diagnostics_dict)."""
    s = str(strategy).lower().strip()
    if s in ("quantile", "q", "qtrain", "q_train"):
        return calibrate_threshold_quantile(normal_scores, percentile)
    if s in ("gpd", "pot_gpd", "evt"):
        return calibrate_threshold_gpd(normal_scores, pot_quantile, target_tail_prob)
    raise ValueError(f"Unknown threshold strategy: {strategy!r}. Use 'quantile' or 'gpd'.")


def load_threshold_config(model_config: dict, per_model_cfg: dict | None = None) -> dict:
    """Resolve the active threshold config.

    Precedence:
      1. Per-model overrides under `<model>.threshold_*` keys (back-compat).
      2. Shared `anomaly_detection.threshold` section.
      3. Hard-coded defaults (gpd, pot=0.90, p=0.05).
    """
    shared = (
        model_config.get("anomaly_detection", {}).get("threshold", {})
        if isinstance(model_config, dict) else {}
    )
    cfg = {
        "strategy": "gpd",
        "percentile": 95.0,
        "pot_quantile": 0.90,
        "target_tail_prob": 0.05,
    }
    cfg.update({k: v for k, v in shared.items() if k in cfg})
    if per_model_cfg:
        # Back-compat: per-model threshold_strategy / threshold_percentile / etc.
        if "threshold_strategy" in per_model_cfg:
            cfg["strategy"] = per_model_cfg["threshold_strategy"]
        if "threshold_percentile" in per_model_cfg:
            cfg["percentile"] = float(per_model_cfg["threshold_percentile"])
        if "pot_quantile" in per_model_cfg:
            cfg["pot_quantile"] = float(per_model_cfg["pot_quantile"])
        if "target_tail_prob" in per_model_cfg:
            cfg["target_tail_prob"] = float(per_model_cfg["target_tail_prob"])
    return cfg


def threshold_policy_str(diag: dict) -> str:
    """Compact policy descriptor for artifact metadata."""
    s = diag.get("strategy")
    if s == "gpd":
        return f"gpd_pot{diag.get('pot_quantile')}_p{diag.get('target_tail_prob')}"
    if s == "quantile":
        return f"q{diag.get('percentile')}_normal_scores"
    return str(s)
