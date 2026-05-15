from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


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
    """Return the `quantile`-th percentile of val scores on normal-labeled rows.

    Standard operating-point rule for all finalized Task A models.
    Raises ValueError if no normal rows exist in val (misconfigured split).
    """
    mask = np.asarray(val_labels) == normal_label
    if not mask.any():
        raise ValueError("No normal-labeled rows in validation — cannot compute q95 threshold")
    normal_scores = np.asarray(val_scores, dtype=float)[mask]
    finite_scores = normal_scores[np.isfinite(normal_scores)]
    if len(finite_scores) == 0:
        raise ValueError("All normal validation scores are non-finite — cannot compute q95 threshold")
    return float(np.quantile(finite_scores, float(quantile)))


def build_score_calibration_payload(
    *,
    threshold: float,
    threshold_quantile: float = 0.95,
    score_direction: str = "higher_is_more_anomalous",
    score_stats: dict | None = None,
) -> dict:
    """Standard score_calibration.json contents for finalized Task A methods."""
    return {
        "score_direction": score_direction,
        "threshold_policy": "normal_validation_quantile",
        "threshold_quantile": float(threshold_quantile),
        "threshold": float(threshold),
        "test_not_used_for_calibration": True,
        "score_stats": score_stats or {},
    }


def compute_anomaly_per_class_metrics(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    normal_label: float | int = 0,
) -> dict[str, dict[str, float | int | None]]:
    """Compute one-vs-normal metrics for each non-normal class."""
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores, dtype=float)
    preds = (scores_arr >= float(threshold)).astype(int)

    out: dict[str, dict[str, float | int | None]] = {}
    unique_labels = sorted(np.unique(labels_arr).tolist())
    for cls in unique_labels:
        if float(cls) == float(normal_label):
            continue
        y_true = (labels_arr == cls).astype(int)
        support = int(y_true.sum())
        if support == 0:
            continue
        try:
            pr_auc = float(average_precision_score(y_true, scores_arr))
        except Exception:
            pr_auc = None
        out[str(int(cls) if float(cls).is_integer() else cls)] = {
            "support": support,
            "pr_auc_one_vs_rest": pr_auc,
            "precision_at_threshold": float(precision_score(y_true, preds, zero_division=0)),
            "recall_at_threshold": float(recall_score(y_true, preds, zero_division=0)),
            "f1_at_threshold": float(f1_score(y_true, preds, zero_division=0)),
        }
    return out


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
