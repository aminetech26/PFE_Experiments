from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def q95_normal_val_threshold(
    *,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    normal_label: float | int = 0,
    quantile: float = 0.95,
) -> float:
    """Return the `quantile`-th percentile of validation scores on normal rows.

    Legacy helper retained for methods that still use normal-quantile thresholding.
    The current Task A benchmark policy uses validation PR-curve F1 calibration.
    Raises ValueError if no normal rows exist in validation.
    """
    mask = np.asarray(val_labels) == normal_label
    if not mask.any():
        raise ValueError("No normal-labeled rows in validation — cannot compute q95 threshold")
    normal_scores = np.asarray(val_scores, dtype=float)[mask]
    finite_scores = normal_scores[np.isfinite(normal_scores)]
    if len(finite_scores) == 0:
        raise ValueError("All normal validation scores are non-finite — cannot compute q95 threshold")
    return float(np.quantile(finite_scores, float(quantile)))


def _optimal_f1_threshold(
    scores: np.ndarray,
    binary_labels: np.ndarray,
) -> tuple[float, float]:
    """Return (threshold, best_f1) maximising F1 on the PR curve."""
    prec, rec, thresholds = precision_recall_curve(binary_labels, scores)
    denom = prec[:-1] + rec[:-1]
    f1_vals = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / np.maximum(denom, 1e-10), 0.0)
    best_idx = int(np.argmax(f1_vals))
    return float(thresholds[best_idx]), float(f1_vals[best_idx])


def build_score_calibration_payload(
    *,
    threshold: float,
    threshold_policy: str = "normal_validation_quantile",
    threshold_quantile: float | None = 0.95,
    score_direction: str = "higher_is_more_anomalous",
    score_stats: dict | None = None,
    candidate_per_true_class_thresholds: dict | None = None,
) -> dict:
    """Build score_calibration.json payload.

    Policy fields are explicit and caller-controlled via `threshold_policy` and
    `threshold_quantile` to support both legacy quantile and current PR-F1
    threshold calibration contracts.
    """
    payload = {
        "score_direction": score_direction,
        "threshold_policy": threshold_policy,
        "threshold": float(threshold),
        "test_not_used_for_calibration": True,
        "score_stats": score_stats or {},
    }
    if threshold_quantile is not None:
        payload["threshold_quantile"] = float(threshold_quantile)
    if candidate_per_true_class_thresholds is not None:
        payload["candidate_per_true_class_thresholds"] = candidate_per_true_class_thresholds
    return payload


def compute_macro_per_class_pr_auc(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    normal_label: float | int = 0,
) -> dict[str, float | dict[str, float | None] | None]:
    """Compute class-vs-normal PR-AUC aggregates across non-normal classes.

    Returns macro/worst PR-AUC and a per-class map for ``{normal_label, cls}``
    one-vs-normal subsets.
    """
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores, dtype=float)

    per_class: dict[str, float | None] = {}
    values: list[float] = []
    unique_labels = sorted(np.unique(labels_arr).tolist())
    for cls in unique_labels:
        if float(cls) == float(normal_label):
            continue

        class_mask = (labels_arr == cls) | (labels_arr == normal_label)
        y_true = (labels_arr[class_mask] == cls).astype(int)
        if y_true.sum() == 0 or (y_true == 0).sum() == 0:
            continue

        cls_scores = scores_arr[class_mask]
        cls_key = str(int(cls) if float(cls).is_integer() else cls)
        try:
            pr_auc = float(average_precision_score(y_true, cls_scores))
            values.append(pr_auc)
            per_class[cls_key] = pr_auc
        except Exception:
            per_class[cls_key] = None

    macro = float(np.mean(values)) if values else None
    worst = float(np.min(values)) if values else None
    return {
        "macro_per_class_pr_auc": macro,
        "worst_class_pr_auc": worst,
        "per_class_pr_auc_vs_normal": per_class,
    }


def compute_episode_level_pr_auc(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    original_labels: np.ndarray,
    group_ids: np.ndarray,
    agg: str = "p95",
    normal_label: float | int = 0,
) -> dict[str, float | dict[str, float | None] | None]:
    """Aggregate per-sample scores to per-episode then compute PR-AUC.

    For each unique group_id, the episode score is computed via ``agg`` over
    the constituent sample scores; the episode label is the majority original
    label of those samples (ties broken toward the non-normal class).  Returns
    binary PR-AUC, macro per-class PR-AUC, worst-class PR-AUC, and a per-class
    breakdown — all at the episode level.
    """
    scores_arr = np.asarray(scores, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    orig_arr = np.asarray(original_labels)
    groups = np.asarray(group_ids, dtype=object)

    if groups.size == 0 or len(np.unique(groups)) <= 1:
        return {
            "episode_binary_pr_auc": None,
            "episode_macro_per_class_pr_auc": None,
            "episode_worst_class_pr_auc": None,
            "episode_per_class_pr_auc_vs_normal": {},
            "n_episodes": 0,
        }

    if agg == "p95":
        agg_fn = lambda s: float(np.percentile(s, 95))
    elif agg == "max":
        agg_fn = lambda s: float(np.max(s))
    elif agg == "mean":
        agg_fn = lambda s: float(np.mean(s))
    elif agg == "trimmed_mean":
        def agg_fn(s):
            lo, hi = np.percentile(s, [10, 90])
            return float(np.mean(s[(s >= lo) & (s <= hi)]))
    else:
        raise ValueError(f"Unsupported agg: {agg}")

    unique_groups, inv = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)
    ep_scores = np.empty(n_groups, dtype=float)
    ep_orig_labels = np.empty(n_groups, dtype=float)
    ep_binary_labels = np.empty(n_groups, dtype=int)

    for gi in range(n_groups):
        mask = inv == gi
        ep_scores[gi] = agg_fn(scores_arr[mask])
        # Episode label = majority of original labels; ties favor non-normal
        sub = orig_arr[mask]
        vals, counts = np.unique(sub, return_counts=True)
        # If there is any non-normal value, prefer the most common non-normal
        non_normal_mask = vals != normal_label
        if non_normal_mask.any():
            nn_vals = vals[non_normal_mask]
            nn_counts = counts[non_normal_mask]
            ep_orig_labels[gi] = float(nn_vals[np.argmax(nn_counts)])
        else:
            ep_orig_labels[gi] = float(normal_label)
        ep_binary_labels[gi] = int(ep_orig_labels[gi] != normal_label)

    binary_pr_auc: float | None = None
    if ep_binary_labels.sum() not in (0, n_groups):
        binary_pr_auc = float(average_precision_score(ep_binary_labels, ep_scores))

    per_class_summary = compute_macro_per_class_pr_auc(
        labels=ep_orig_labels, scores=ep_scores, normal_label=normal_label
    )

    return {
        "episode_binary_pr_auc": binary_pr_auc,
        "episode_macro_per_class_pr_auc": per_class_summary.get("macro_per_class_pr_auc"),
        "episode_worst_class_pr_auc": per_class_summary.get("worst_class_pr_auc"),
        "episode_per_class_pr_auc_vs_normal": per_class_summary.get(
            "per_class_pr_auc_vs_normal", {}
        ),
        "n_episodes": n_groups,
        "agg": agg,
    }


def compute_anomaly_per_class_metrics(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    p2_threshold: float | None = None,
    cusum_k: float | None = None,
    mu_normal: float | None = None,
    val_labels: np.ndarray | None = None,
    val_scores: np.ndarray | None = None,
    normal_label: float | int = 0,
) -> dict[str, dict[str, float | int | None]]:
    """Compute per-class metrics aligned with the three deployed alarm tiers.

    For each fault class, evaluates on the OvN subset {normal_label, cls}.

    Tiers (all optional — only computed when the relevant threshold is provided):
      P3 Advisory  → threshold          (sensitive_threshold / GPD)
      P2 High      → p2_threshold       (conformal threshold)
      P1 Critical  → CUSUM stateful; proxy = fraction of fault samples with
                     score > mu_normal + cusum_k (samples that contribute to accumulation)

    The legacy per-class-optimal threshold group is dropped — it does not
    correspond to any deployed threshold.
    """
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores, dtype=float)
    p3_thr = float(threshold)

    has_val = val_labels is not None and val_scores is not None
    val_labels_arr = np.asarray(val_labels) if has_val else None
    val_scores_arr = np.asarray(val_scores, dtype=float) if has_val else None

    cusum_contrib_thr: float | None = None
    if cusum_k is not None and mu_normal is not None:
        cusum_contrib_thr = float(mu_normal) + float(cusum_k)

    out: dict[str, dict[str, float | int | None]] = {}
    unique_labels = sorted(np.unique(labels_arr).tolist())
    for cls in unique_labels:
        if float(cls) == float(normal_label):
            continue

        class_mask = (labels_arr == cls) | (labels_arr == normal_label)
        if not np.any(class_mask):
            continue

        y_true = (labels_arr[class_mask] == cls).astype(int)
        cls_scores = scores_arr[class_mask]
        fault_scores = cls_scores[y_true == 1]

        support = int(y_true.sum())
        if support == 0:
            continue
        support_normal = int((labels_arr[class_mask] == normal_label).sum())

        try:
            pr_auc = float(average_precision_score(y_true, cls_scores))
        except Exception:
            pr_auc = None

        # P3: precision/recall/F1 at the deployed sensitive threshold
        preds_p3 = (cls_scores >= p3_thr).astype(int)
        p3_precision = float(precision_score(y_true, preds_p3, zero_division=0))
        p3_recall = float(recall_score(y_true, preds_p3, zero_division=0))
        p3_f1 = float(f1_score(y_true, preds_p3, zero_division=0))

        # P2: metrics at conformal threshold (if provided)
        p2_precision: float | None = None
        p2_recall: float | None = None
        p2_f1: float | None = None
        if p2_threshold is not None:
            preds_p2 = (cls_scores >= float(p2_threshold)).astype(int)
            p2_precision = float(precision_score(y_true, preds_p2, zero_division=0))
            p2_recall = float(recall_score(y_true, preds_p2, zero_division=0))
            p2_f1 = float(f1_score(y_true, preds_p2, zero_division=0))

        # P1/CUSUM proxy: fraction of fault samples that contribute to accumulation
        # (score > mu_normal + cusum_k). These samples push the CUSUM statistic up.
        cusum_contrib_rate: float | None = None
        if cusum_contrib_thr is not None and len(fault_scores) > 0:
            cusum_contrib_rate = float((fault_scores > cusum_contrib_thr).mean())

        # Validation support (for context only)
        val_support_fault: int | None = None
        val_support_normal: int | None = None
        if has_val:
            val_class_mask = (val_labels_arr == cls) | (val_labels_arr == normal_label)
            y_val_k = (val_labels_arr[val_class_mask] == cls).astype(int)
            val_support_fault = int(y_val_k.sum())
            val_support_normal = int((val_labels_arr[val_class_mask] == normal_label).sum())

        out[str(int(cls) if float(cls).is_integer() else cls)] = {
            "support_fault": support,
            "support_normal": support_normal,
            "val_support_fault": val_support_fault,
            "val_support_normal": val_support_normal,
            # Threshold-free ranking quality
            "pr_auc_vs_normal": pr_auc,
            # P3 Advisory (sensitive_threshold / GPD) — deployed
            "p3_precision": p3_precision,
            "p3_recall": p3_recall,
            "p3_f1": p3_f1,
            # P2 High (conformal threshold) — deployed
            "p2_precision": p2_precision,
            "p2_recall": p2_recall,
            "p2_f1": p2_f1,
            # P1 Critical CUSUM proxy — fraction of fault samples that contribute to accumulation
            "p1_cusum_contrib_rate": cusum_contrib_rate,
        }
    return out


def build_candidate_per_true_class_thresholds(
    per_class_metrics: dict[str, dict[str, float | int | None]],
    *,
    normal_label: float | int = 0,
) -> dict:
    """Build diagnostic candidate thresholds payload for score calibration artifacts."""
    thresholds: dict[str, dict[str, float | int | None]] = {}
    for cls_str, m in per_class_metrics.items():
        threshold = m.get("candidate_per_true_class_threshold", m.get("per_class_threshold"))
        if threshold is None:
            continue
        thresholds[cls_str] = {
            "threshold": threshold,
            "val_f1": m.get("val_f1_at_per_class_threshold"),
            "val_support_fault": m.get("val_support_fault"),
            "val_support_normal": m.get("val_support_normal"),
            "test_precision": m.get(
                "precision_at_candidate_per_true_class_threshold",
                m.get("precision_at_per_class_threshold"),
            ),
            "test_recall": m.get(
                "recall_at_candidate_per_true_class_threshold",
                m.get("recall_at_per_class_threshold"),
            ),
            "test_f1": m.get(
                "f1_at_candidate_per_true_class_threshold",
                m.get("f1_at_per_class_threshold"),
            ),
            "test_pr_auc": m.get("pr_auc_vs_normal"),
        }

    return {
        "threshold_policy": "validation_true_class_pr_curve_f1",
        "deployment_ready": False,
        "intended_use": "diagnostic_or_candidate_class_conditioned_gate",
        "normal_label": normal_label,
        "thresholds": thresholds,
    }


def build_run_manifest(
    *,
    task: str,
    model: str,
    model_family: str,
    dataset: str,
    split_path: str,
    feature_profile: str,
    feature_run_dir: str,
    seed: int,
    run_type: str,
    extras: dict | None = None,
) -> dict:
    payload = {
        "task": task,
        "model": model,
        "model_family": model_family,
        "dataset": dataset,
        "split_path": split_path,
        "feature_profile": feature_profile,
        "feature_run_dir": feature_run_dir,
        "seed": int(seed),
        "run_type": run_type,
        "artifact_contract_version": 1,
    }
    if extras:
        payload.update(extras)
    return payload


def build_deployment_manifest(
    *,
    task: str,
    model: str,
    model_family: str,
    model_artifact: str,
    feature_names: list[str],
    label_column: str,
    threshold: float | None = None,
    scaler_artifact: str | None = None,
    classes: list[str] | None = None,
    window_size: int | None = None,
    score_direction: str | None = None,
    extras: dict | None = None,
) -> dict:
    payload = {
        "task": task,
        "model": model,
        "model_family": model_family,
        "model_artifact": model_artifact,
        "feature_names": feature_names,
        "label_column": label_column,
    }
    if threshold is not None:
        payload["threshold"] = float(threshold)
    if scaler_artifact is not None:
        payload["scaler_artifact"] = scaler_artifact
    if classes is not None:
        payload["classes"] = classes
    if window_size is not None:
        payload["window_size"] = int(window_size)
    if score_direction is not None:
        payload["score_direction"] = score_direction
    if extras:
        payload.update(extras)
    return payload
