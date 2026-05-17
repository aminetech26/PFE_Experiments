from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import yaml
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

from src.modeling.common.artifact_contract import (
    build_deployment_manifest,
    build_run_manifest,
)
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
    if len(classes) == 2:
        # sklearn.label_binarize returns shape (n, 1) for binary classes;
        # expand to OvR two-column target to align with predict_proba shape (n, 2).
        y_true_bin = np.zeros((len(y_true_encoded), 2), dtype=int)
        y_true_bin[:, 1] = (y_true_encoded == 1).astype(int)
        y_true_bin[:, 0] = 1 - y_true_bin[:, 1]
    else:
        y_true_bin = label_binarize(y_true_encoded, classes=np.arange(len(classes)))
    return float(average_precision_score(y_true_bin, y_proba, average="weighted"))


def compute_pr_auc_by_class(
    y_true_encoded: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    if len(classes) == 2:
        y_true_bin = np.zeros((len(y_true_encoded), 2), dtype=int)
        y_true_bin[:, 1] = (y_true_encoded == 1).astype(int)
        y_true_bin[:, 0] = 1 - y_true_bin[:, 1]
    else:
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
    labels = np.arange(len(classes))
    summary_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "pr_auc_weighted": compute_pr_auc_multiclass(y_true, y_proba, classes),
    }
    pr_auc_by_class = compute_pr_auc_by_class(y_true, y_proba, classes)
    report = cast(
        dict[str, Any],
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[str(x) for x in classes],
            output_dict=True,
            zero_division=0,
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
    resolved_model_path = Path(model_path) if model_path else root / "model.joblib"
    if not resolved_model_path.suffix:
        resolved_model_path = resolved_model_path.with_suffix(".joblib")

    paths = {
        "artifacts_dir": root,
        "metrics_path": Path(metrics_path) if metrics_path else root / "metrics.json",
        "leakage_report_path": Path(leakage_report_path)
        if leakage_report_path
        else root / "leakage_report.json",
        "model_path": resolved_model_path,
        "confusion_matrix_path": root / "confusion_matrix.png",
        "confusion_matrix_normalized_path": root / "confusion_matrix_normalized.png",
        "per_class_pr_curve_path": root / "per_class_pr_curve.png",
        "prediction_distribution_path": root / "prediction_distribution.png",
        "feature_importance_path": root / "feature_importance_top20.png",
        "global_metrics_path": root / "global_metrics.json",
        "per_class_metrics_path": root / "per_class_metrics.json",
        "run_manifest_path": root / "run_manifest.json",
        "deployment_manifest_path": root / "deployment_manifest.json",
        "classification_report_path": root / "classification_report.json",
        "features_manifest_path": root / "features_manifest.json",
        "probability_calibration_path": root / "probability_calibration.json",
        "reliability_diagram_path": root / "reliability_diagram.png",
        "confidence_histogram_path": root / "confidence_histogram.png",
    }
    return paths


def write_classification_contract_artifacts(
    *,
    paths: dict[str, Path],
    args,
    model_name: str,
    feature_profile: str,
    resolved_run_dir: Path,
    seed: int,
    run_name: str,
    features: list[str],
    label_column: str,
    classes: np.ndarray,
    summary_metrics: dict[str, float],
    pr_auc_by_class: dict[str, float],
    classification_report_payload: dict[str, Any],
    features_manifest_payload: dict[str, Any],
    calibration_payload: dict[str, Any] | None = None,
    calibration_impact: dict[str, Any] | None = None,
) -> None:
    global_metrics: dict[str, Any] = {
        "accuracy": summary_metrics.get("accuracy"),
        "balanced_accuracy": summary_metrics.get("balanced_accuracy"),
        "f1_weighted": summary_metrics.get("f1_weighted"),
        "f1_macro": summary_metrics.get("f1_macro"),
        "precision_weighted": summary_metrics.get("precision_weighted"),
        "precision_macro": summary_metrics.get("precision_macro"),
        "recall_weighted": summary_metrics.get("recall_weighted"),
        "recall_macro": summary_metrics.get("recall_macro"),
        "pr_auc_weighted": summary_metrics.get("pr_auc_weighted"),
    }
    if calibration_payload is not None:
        cal_metrics = calibration_payload.get("metrics", {})
        global_metrics.update(
            {
                "log_loss": cal_metrics.get("log_loss"),
                "brier_score_multiclass": cal_metrics.get("brier_score_multiclass"),
                "expected_calibration_error": cal_metrics.get("expected_calibration_error"),
                "maximum_calibration_error": cal_metrics.get("maximum_calibration_error"),
                "mean_confidence": cal_metrics.get("mean_confidence"),
                "accuracy_confidence_gap": cal_metrics.get("accuracy_confidence_gap"),
            }
        )
    if calibration_impact is not None:
        global_metrics["calibration_impact"] = calibration_impact

    per_class: dict[str, dict[str, Any]] = {}
    for cls in classes:
        key = str(cls)
        report_entry = classification_report_payload.get(key, {})
        per_class[key] = {
            "support": report_entry.get("support"),
            "precision": report_entry.get("precision"),
            "recall": report_entry.get("recall"),
            "f1_score": report_entry.get("f1-score"),
            "pr_auc": pr_auc_by_class.get(key),
        }

    run_manifest = build_run_manifest(
        task=str(args.task),
        model=model_name,
        model_family="classification_ml",
        dataset=str(args.dataset),
        split_path=str(args.split_path),
        feature_profile=feature_profile,
        feature_run_dir=str(resolved_run_dir),
        seed=int(seed),
        run_type=str(args.run_type),
        extras={"run_name": run_name},
    )
    deployment_extras: dict[str, Any] = {"label_encoder_embedded": True}
    if calibration_payload is not None:
        deployment_extras["confidence_is_calibrated"] = True
        deployment_extras["probability_calibration_artifact"] = paths[
            "probability_calibration_path"
        ].name
        deployment_extras["calibration_method"] = calibration_payload.get("calibration_method")
    else:
        deployment_extras["confidence_is_calibrated"] = False

    deployment_manifest = build_deployment_manifest(
        task=str(args.task),
        model=model_name,
        model_family="classification_ml",
        model_artifact=paths["model_path"].name,
        scaler_artifact=None,
        feature_names=features,
        label_column=label_column,
        classes=[str(c) for c in classes],
        extras=deployment_extras,
    )

    write_json_artifact(paths["global_metrics_path"], global_metrics)
    write_json_artifact(paths["per_class_metrics_path"], per_class)
    write_json_artifact(paths["run_manifest_path"], run_manifest)
    write_json_artifact(paths["deployment_manifest_path"], deployment_manifest)
    write_json_artifact(paths["classification_report_path"], classification_report_payload)
    write_json_artifact(paths["features_manifest_path"], features_manifest_payload)
    if calibration_payload is not None:
        write_json_artifact(paths["probability_calibration_path"], calibration_payload)


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


def resolve_calibration_config(config: dict) -> dict:
    classification_cfg = config.get("classification", {})
    cal_cfg = classification_cfg.get("calibration", {})
    return {
        "enabled": bool(cal_cfg.get("enabled", False)),
        "method": str(cal_cfg.get("method", "sigmoid")),
    }


class _PrefitEstimator(ClassifierMixin, BaseEstimator):
    """Wraps a fitted model so CalibratedClassifierCV never refits it.

    sklearn >= 1.4 removed cv='prefit'. CalibratedClassifierCV with an iterable cv
    clones the estimator before calling .fit(). __sklearn_clone__ returns self so the
    original fitted model is preserved through cloning. .fit() is a no-op; predict_proba
    delegates to the wrapped model, giving the calibrator valid probabilities on val data.
    """

    def __init__(self, fitted_model=None):
        self.fitted_model = fitted_model

    def __sklearn_clone__(self):
        return self

    def fit(self, X, y, **kwargs):
        return self

    def predict_proba(self, X):
        return self.fitted_model.predict_proba(X)

    @property
    def classes_(self):
        return self.fitted_model.classes_


def fit_probability_calibrator(base_model, x_val: pd.DataFrame, y_val: np.ndarray, method: str):
    # sklearn >= 1.4 removed cv='prefit'. We wrap the fitted model in _PrefitEstimator
    # so clone()+fit() is a no-op, then pass all val data as a single calibration fold.
    n = len(x_val)
    prefit_cv = [(np.array([], dtype=np.intp), np.arange(n, dtype=np.intp))]
    wrapper = _PrefitEstimator(fitted_model=base_model)
    calibrated = CalibratedClassifierCV(estimator=wrapper, method=method, cv=prefit_cv)
    calibrated.fit(x_val, y_val)
    return calibrated


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    ll = float(log_loss(y_true, y_proba, labels=np.arange(len(classes))))

    # Multiclass Brier: mean over OvR Brier scores
    y_true_bin = label_binarize(y_true, classes=np.arange(len(classes)))
    brier = float(np.mean([brier_score_loss(y_true_bin[:, i], y_proba[:, i]) for i in range(len(classes))]))

    # ECE / MCE over max-class confidence (classwise calibration proxy)
    confidence = np.max(y_proba, axis=1)
    correctness = (np.argmax(y_proba, axis=1) == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Last bin is closed on the right to capture confidence == 1.0 (common with ExtraTrees)
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidence[mask].mean()
        gap = abs(bin_acc - bin_conf)
        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)

    mean_conf = float(confidence.mean())
    acc = float(correctness.mean())

    return {
        "log_loss": ll,
        "brier_score_multiclass": brier,
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "mean_confidence": mean_conf,
        "accuracy_confidence_gap": float(abs(acc - mean_conf)),
    }


def build_probability_calibration_payload(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    method: str,
    n_calibration_samples: int,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    cal_metrics = metrics if metrics is not None else compute_calibration_metrics(y_true, y_proba, classes)
    return {
        "confidence_is_calibrated": True,
        "calibration_method": method,
        "fit_split": "validation",
        "test_not_used_for_calibration": True,
        "probability_output": "calibrated_predict_proba",
        "n_calibration_samples": n_calibration_samples,
        "n_classes": int(len(classes)),
        "classes": [str(c) for c in classes],
        "metrics_split": "test",
        "metrics": cal_metrics,
    }


def build_calibration_impact_payload(
    *,
    y_true: np.ndarray,
    classes: np.ndarray,
    uncalibrated_pred_proba: np.ndarray,
    calibrated_pred_proba: np.ndarray,
) -> dict[str, Any]:
    uncalibrated_pred = np.argmax(uncalibrated_pred_proba, axis=1)
    calibrated_pred = np.argmax(calibrated_pred_proba, axis=1)

    uncalibrated_summary_metrics, _, _ = compute_classification_metrics(
        y_true,
        uncalibrated_pred,
        uncalibrated_pred_proba,
        classes,
    )
    calibrated_summary_metrics, _, _ = compute_classification_metrics(
        y_true,
        calibrated_pred,
        calibrated_pred_proba,
        classes,
    )

    uncalibrated_cal_metrics = compute_calibration_metrics(y_true, uncalibrated_pred_proba, classes)
    calibrated_cal_metrics = compute_calibration_metrics(y_true, calibrated_pred_proba, classes)

    return {
        "uncalibrated": {
            "f1_macro": uncalibrated_summary_metrics["f1_macro"],
            "f1_weighted": uncalibrated_summary_metrics["f1_weighted"],
            "log_loss": uncalibrated_cal_metrics["log_loss"],
            "expected_calibration_error": uncalibrated_cal_metrics["expected_calibration_error"],
        },
        "calibrated": {
            "f1_macro": calibrated_summary_metrics["f1_macro"],
            "f1_weighted": calibrated_summary_metrics["f1_weighted"],
            "log_loss": calibrated_cal_metrics["log_loss"],
            "expected_calibration_error": calibrated_cal_metrics["expected_calibration_error"],
        },
        "delta": {
            "f1_macro": calibrated_summary_metrics["f1_macro"]
            - uncalibrated_summary_metrics["f1_macro"],
            "f1_weighted": calibrated_summary_metrics["f1_weighted"]
            - uncalibrated_summary_metrics["f1_weighted"],
            "log_loss": calibrated_cal_metrics["log_loss"] - uncalibrated_cal_metrics["log_loss"],
            "expected_calibration_error": calibrated_cal_metrics["expected_calibration_error"]
            - uncalibrated_cal_metrics["expected_calibration_error"],
        },
        "delta_interpretation": {
            "f1_macro": "higher_is_better",
            "f1_weighted": "higher_is_better",
            "log_loss": "lower_is_better",
            "expected_calibration_error": "lower_is_better",
        },
        "calibrated_calibration_metrics": calibrated_cal_metrics,
    }


def save_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    path: Path,
) -> None:
    y_true_bin = label_binarize(y_true, classes=np.arange(len(classes)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for idx, cls in enumerate(classes):
        prob_true, prob_pred = calibration_curve(
            y_true_bin[:, idx], y_proba[:, idx], n_bins=10, strategy="uniform"
        )
        ax.plot(prob_pred, prob_true, marker="o", label=f"Class {cls}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram (per-class OvR)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_confidence_histogram(
    y_proba: np.ndarray,
    path: Path,
    *,
    model_name: str = "",
) -> None:
    confidence = np.max(y_proba, axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(confidence, bins=20, edgecolor="black")
    ax.set_xlabel("Max-class confidence")
    ax.set_ylabel("Count")
    title = f"Confidence Histogram{' - ' + model_name if model_name else ''}"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


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
