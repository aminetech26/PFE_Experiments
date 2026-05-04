from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

from src.modeling.common.system_resources import compute_thread_budget, detect_cpu_resources
from src.utils.paths import get_experiments_root

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


def default_comparison_records_path() -> Path:
    return get_experiments_root() / "metrics" / "classification_comparison_records.jsonl"


def default_artifacts_dir(model_name: str, run_name: str) -> Path:
    return get_experiments_root() / "classification" / model_name / run_name


def build_study_name_prefix(base_prefix: str, model_name: str) -> str:
    normalized_base = str(base_prefix).strip() or "classification_ml"
    normalized_model = str(model_name).strip()
    suffix = f"_{normalized_model}"
    if normalized_base.endswith(suffix):
        return normalized_base
    return f"{normalized_base}{suffix}"


def load_model_config() -> dict:
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_classification_ml_config(config: dict) -> tuple[dict, dict, dict]:
    classification_cfg = config.get("classification")
    if not isinstance(classification_cfg, dict):
        raise KeyError("Missing 'classification' section in model_config.yaml")

    ml_cfg = classification_cfg.get("ml")
    if not isinstance(ml_cfg, dict):
        raise KeyError("Missing 'classification.ml' section in model_config.yaml")

    active_model = ml_cfg.get("active_model")
    if not isinstance(active_model, str) or not active_model:
        raise KeyError("Missing non-empty 'classification.ml.active_model' in model_config.yaml")

    model_spaces = ml_cfg.get("models")
    if not isinstance(model_spaces, dict):
        raise KeyError("Missing 'classification.ml.models' section in model_config.yaml")

    hpo_cfg = ml_cfg.get("hpo")
    if not isinstance(hpo_cfg, dict):
        raise KeyError("Missing 'classification.ml.hpo' section in model_config.yaml")

    return {"active_model": active_model, "model_spaces": model_spaces}, classification_cfg, hpo_cfg


def resolve_runtime_seed(runtime_config: dict, cli_seed: int | None) -> int:
    if cli_seed is not None:
        return int(cli_seed)
    exp = runtime_config.get("experiment", {})
    seeds = exp.get("seeds", [42])
    return int(seeds[0] if isinstance(seeds, list) else seeds)


def prepare_xy(
    df: pd.DataFrame, features: list[str], label_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing features in dataframe: {missing_features}")
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataframe")

    x_data = pd.DataFrame(df.loc[:, features])
    y_data = pd.Series(df.loc[:, label_column])
    return x_data, y_data


def compute_pr_auc_multiclass(
    y_true_encoded: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> float:
    if len(classes) <= 1:
        return float("nan")
    y_true_bin = label_binarize(y_true_encoded, classes=np.arange(len(classes)))
    return float(average_precision_score(y_true_bin, y_proba, average="weighted"))


def compute_pr_auc_by_class(
    y_true_encoded: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    y_true_bin = label_binarize(y_true_encoded, classes=np.arange(len(classes)))
    pr_auc_by_class: dict[str, float] = {}
    for idx, cls in enumerate(classes):
        pr_auc_by_class[str(cls)] = float(
            average_precision_score(y_true_bin[:, idx], y_proba[:, idx])
        )
    return pr_auc_by_class


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    summary_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "pr_auc_weighted": compute_pr_auc_multiclass(y_true, y_proba, classes),
    }
    pr_auc_by_class = compute_pr_auc_by_class(y_true, y_proba, classes)
    report = cast(
        dict[str, Any],
        classification_report(
            y_true,
            y_pred,
            target_names=[str(x) for x in classes],
            output_dict=True,
        ),
    )
    return summary_metrics, pr_auc_by_class, report


def resolve_artifact_paths(
    *,
    model_name: str,
    run_name: str,
    artifacts_dir: str | None,
    metrics_path: str | None,
    leakage_report_path: str | None,
    model_path: str | None,
) -> dict[str, Path]:
    root = Path(artifacts_dir) if artifacts_dir else default_artifacts_dir(model_name, run_name)
    root.mkdir(parents=True, exist_ok=True)

    paths = {
        "artifacts_dir": root,
        "metrics_path": Path(metrics_path) if metrics_path else root / "metrics.json",
        "leakage_report_path": Path(leakage_report_path)
        if leakage_report_path
        else root / "leakage_report.json",
        "model_path": Path(model_path) if model_path else root / "model.joblib",
        "confusion_matrix_path": root / "confusion_matrix.png",
        "confusion_matrix_normalized_path": root / "confusion_matrix_normalized.png",
        "per_class_pr_curve_path": root / "per_class_pr_curve.png",
        "prediction_distribution_path": root / "prediction_distribution.png",
        "feature_importance_path": root / "feature_importance_top20.png",
    }
    return paths


def _save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_labels: list[str],
    path: Path,
    *,
    title: str,
    value_fmt: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_labels)), labels=class_labels)
    ax.set_yticks(np.arange(len(class_labels)), labels=class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = float(np.nanmax(matrix)) / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = format(value, value_fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_classification_plots(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    model_name: str,
    paths: dict[str, Path],
    feature_names: list[str],
    model,
) -> None:
    class_labels = [str(cls) for cls in classes]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    _save_confusion_matrix_plot(
        cm,
        class_labels,
        paths["confusion_matrix_path"],
        title=f"Confusion Matrix - {model_name}",
        value_fmt="d",
    )

    cm_normalized = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        normalize="true",
    )
    _save_confusion_matrix_plot(
        cm_normalized,
        class_labels,
        paths["confusion_matrix_normalized_path"],
        title=f"Normalized Confusion Matrix - {model_name}",
        value_fmt=".2f",
    )

    y_true_bin = label_binarize(y_true, classes=np.arange(len(classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], y_proba[:, idx])
        pr_auc = average_precision_score(y_true_bin[:, idx], y_proba[:, idx])
        ax.plot(recall, precision, label=f"Class {cls} (AP={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Per-Class PR Curves - {model_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["per_class_pr_curve_path"], dpi=140)
    plt.close(fig)

    true_counts = np.bincount(y_true, minlength=len(classes))
    pred_counts = np.bincount(y_pred, minlength=len(classes))
    x = np.arange(len(classes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, true_counts, width, label="True")
    ax.bar(x + width / 2, pred_counts, width, label="Predicted")
    ax.set_xticks(x, class_labels)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction Distribution - {model_name}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["prediction_distribution_path"], dpi=140)
    plt.close(fig)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        if importances.size == len(feature_names) and importances.size > 0:
            top_k = min(20, len(feature_names))
            top_idx = np.argsort(importances)[-top_k:]
            top_features = [feature_names[i] for i in top_idx]
            top_importances = importances[top_idx]
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(top_features, top_importances)
            ax.set_xlabel("Importance")
            ax.set_title(f"Top {top_k} Feature Importances - {model_name}")
            ax.grid(True, axis="x", alpha=0.3)
            fig.tight_layout()
            fig.savefig(paths["feature_importance_path"], dpi=140)
            plt.close(fig)


def write_json_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def resolve_threading(config: dict, args: argparse.Namespace) -> dict:
    threading_cfg = config.get("training", {}).get("threading", {})
    cpu = detect_cpu_resources()

    prefer_physical = bool(threading_cfg.get("prefer_physical_cores", True))
    reserve_cores = int(threading_cfg.get("reserve_cores", 1))
    max_threads = threading_cfg.get("max_threads", None)
    optuna_parallel_trials = int(threading_cfg.get("optuna_parallel_trials", 1))

    if args.threads is not None:
        max_threads = int(args.threads)
    if args.optuna_jobs is not None:
        optuna_parallel_trials = int(args.optuna_jobs)

    thread_budget = compute_thread_budget(
        cpu,
        prefer_physical_cores=prefer_physical,
        reserve_cores=reserve_cores,
        max_threads=int(max_threads) if max_threads is not None else None,
    )

    optuna_parallel_trials = max(1, min(optuna_parallel_trials, thread_budget))
    threads_per_trial = max(1, thread_budget // optuna_parallel_trials)

    return {
        "cpu_logical_cores": cpu.logical_cores,
        "cpu_physical_cores": cpu.physical_cores,
        "thread_budget": thread_budget,
        "optuna_parallel_trials": optuna_parallel_trials,
        "threads_per_trial": threads_per_trial,
        "prefer_physical_cores": prefer_physical,
        "reserve_cores": reserve_cores,
    }
