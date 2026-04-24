from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import stumpy
import yaml
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.mlflow_setup import init_tracking
from src.modeling.common.feature_loader import load_features_for_task

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "experiments" / "metrics" / "anomaly_matrix_profile_results.json"
)
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures" / "anomaly" / "matrix_profile"


def _load_model_config() -> dict:
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_matrix_profile_cfg(config: dict) -> dict:
    anomaly_cfg = config.get("anomaly_detection")
    if not isinstance(anomaly_cfg, dict):
        raise KeyError("Missing 'anomaly_detection' section in model_config.yaml")

    ml_cfg = anomaly_cfg.get("ml")
    if not isinstance(ml_cfg, dict):
        raise KeyError("Missing 'anomaly_detection.ml' section in model_config.yaml")

    model_cfg = ml_cfg.get("models", {}).get("matrix_profile")
    if not isinstance(model_cfg, dict):
        raise KeyError("Missing 'anomaly_detection.ml.models.matrix_profile' in model_config.yaml")

    return model_cfg


def _infer_signal_column(df: pd.DataFrame, feature_cols: list[str]) -> str:
    priority = ["pdc", "pdc1", "idc1", "Eg", "vdc1"]
    for col in priority:
        if col in df.columns and col in feature_cols and pd.api.types.is_numeric_dtype(df[col]):
            return col

    for col in feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return col

    raise ValueError("No numeric signal column found for Matrix Profile baseline")


def _binary_labels(df: pd.DataFrame, label_col: str) -> np.ndarray:
    return (pd.to_numeric(df[label_col], errors="coerce").fillna(0.0).to_numpy() > 0).astype(int)


def _matrix_profile_scores(series: np.ndarray, window_size: int) -> np.ndarray:
    mp = stumpy.stump(series, m=window_size)
    subseq_scores = np.asarray(mp[:, 0], dtype=float)

    pad = np.full(window_size - 1, np.nan, dtype=float)
    scores = np.concatenate([pad, subseq_scores])

    finite_scores = scores[np.isfinite(scores)]
    fill_value = float(np.nanmedian(finite_scores)) if finite_scores.size else 0.0
    scores = np.where(np.isfinite(scores), scores, fill_value)
    return scores


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    quantiles = np.linspace(0.80, 0.999, 120)
    candidates = np.quantile(scores, quantiles)

    best_thr = float(candidates[0])
    best_f1 = -1.0
    for thr in candidates:
        pred = (scores >= thr).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr, best_f1


def _compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    pr_auc = (
        float("nan")
        if np.unique(y_true).size < 2
        else float(average_precision_score(y_true, scores))
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "pr_auc": pr_auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(np.mean(y_pred)),
        "confusion_matrix": cm,
    }


def _plot_pr_curve(y_true: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation PR Curve - Matrix Profile")
    plt.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_score_hist(
    y_true: np.ndarray, scores: np.ndarray, threshold: float, out_path: Path
) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(scores[y_true == 0], bins=50, alpha=0.6, label="normal", density=True)
    plt.hist(scores[y_true == 1], bins=50, alpha=0.6, label="fault", density=True)
    plt.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.3f}")
    plt.xlabel("Matrix Profile Score")
    plt.ylabel("Density")
    plt.title("Validation Score Distribution")
    plt.legend()
    plt.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_timeline(
    scores: np.ndarray, y_true: np.ndarray, threshold: float, out_path: Path
) -> None:
    n = min(len(scores), 5000)
    x = np.arange(n)
    plt.figure(figsize=(10, 4))
    plt.plot(x, scores[:n], lw=1.0, label="score")
    fault_idx = x[y_true[:n] == 1]
    if fault_idx.size > 0:
        plt.scatter(fault_idx, scores[:n][y_true[:n] == 1], s=8, alpha=0.6, label="fault points")
    plt.axhline(threshold, color="red", linestyle="--", label="threshold")
    plt.xlabel("Sample index")
    plt.ylabel("Score")
    plt.title("Validation Scores Timeline (first 5000 samples)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_matrix_profile(config: dict | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run Task A Matrix Profile anomaly baseline")
    parser.add_argument(
        "--task", default="anomaly_semisup", choices=["anomaly_semisup", "anomaly_supervised"]
    )
    parser.add_argument("--dataset", default="costa")
    parser.add_argument("--split-path", default="path_b", choices=["path_a", "path_b"])
    parser.add_argument("--profile", default="baseline_raw")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--signal-col", default=None, help="Optional override signal column")
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    args = parser.parse_args()

    runtime_config = config or _load_model_config()
    model_cfg = _resolve_matrix_profile_cfg(runtime_config)

    configured_windows = model_cfg.get("window_size", [60])
    if isinstance(configured_windows, list) and configured_windows:
        default_window = int(configured_windows[0])
    else:
        default_window = int(configured_windows)
    window_size = int(args.window_size or default_window)

    train_df, val_df, test_df, manifest, resolved_run_dir = load_features_for_task(
        task=args.task,
        profile=args.profile,
        run_dir=args.run_dir,
        run_id=args.run_id,
        dataset=args.dataset,
        split_path=args.split_path,
    )

    feature_cols = list(manifest.get("final_features", []))
    label_col = str(manifest.get("label_column", "label"))
    if label_col not in train_df.columns:
        raise KeyError(f"Label column '{label_col}' missing in loaded split data")

    signal_col = args.signal_col or _infer_signal_column(train_df, feature_cols)
    if signal_col not in train_df.columns:
        raise KeyError(f"Signal column '{signal_col}' not found in train split")

    train_signal = (
        pd.to_numeric(train_df[signal_col], errors="coerce").ffill().fillna(0.0).to_numpy()
    )
    val_signal = pd.to_numeric(val_df[signal_col], errors="coerce").ffill().fillna(0.0).to_numpy()
    test_signal = pd.to_numeric(test_df[signal_col], errors="coerce").ffill().fillna(0.0).to_numpy()

    if (
        len(train_signal) <= window_size
        or len(val_signal) <= window_size
        or len(test_signal) <= window_size
    ):
        raise ValueError(
            f"window_size={window_size} is too large for one split length: "
            f"train={len(train_signal)} val={len(val_signal)} test={len(test_signal)}"
        )

    logger.info(
        "Matrix Profile baseline | dataset={} split_path={} task={} profile={} signal={} window={} run_dir={}",
        args.dataset,
        args.split_path,
        args.task,
        args.profile,
        signal_col,
        window_size,
        resolved_run_dir,
    )

    train_scores = _matrix_profile_scores(train_signal, window_size)
    val_scores = _matrix_profile_scores(val_signal, window_size)
    test_scores = _matrix_profile_scores(test_signal, window_size)

    y_train = _binary_labels(train_df, label_col)
    y_val = _binary_labels(val_df, label_col)
    y_test = _binary_labels(test_df, label_col)

    threshold, best_val_f1 = _best_threshold(y_val, val_scores)
    train_metrics = _compute_metrics(y_train, train_scores, threshold)
    val_metrics = _compute_metrics(y_val, val_scores, threshold)
    test_metrics = _compute_metrics(y_test, test_scores, threshold)

    figures_dir = Path(args.figures_dir)
    pr_curve_path = figures_dir / "val_pr_curve.png"
    hist_path = figures_dir / "val_score_hist.png"
    timeline_path = figures_dir / "val_scores_timeline.png"
    _plot_pr_curve(y_val, val_scores, pr_curve_path)
    _plot_score_hist(y_val, val_scores, threshold, hist_path)
    _plot_timeline(val_scores, y_val, threshold, timeline_path)

    init_tracking("anomaly")
    run_name = f"anomaly_matrix_profile_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    metrics_payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "task": args.task,
        "dataset": args.dataset,
        "split_path": args.split_path,
        "model": "matrix_profile",
        "feature_profile": str(manifest.get("profile") or args.profile),
        "feature_run_id": resolved_run_dir.name,
        "feature_run_dir": str(resolved_run_dir),
        "signal_column": signal_col,
        "window_size": window_size,
        "threshold": threshold,
        "best_val_f1": best_val_f1,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "figures": {
            "val_pr_curve": str(pr_curve_path),
            "val_score_hist": str(hist_path),
            "val_scores_timeline": str(timeline_path),
        },
    }

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "task": args.task,
                "dataset": args.dataset,
                "split_path": args.split_path,
                "model_family": "anomaly_detection_ml",
                "model_name": "matrix_profile",
            }
        )

        mlflow.log_params(
            {
                "task": args.task,
                "dataset": args.dataset,
                "split_path": args.split_path,
                "feature_profile": str(manifest.get("profile") or args.profile),
                "feature_run_id": resolved_run_dir.name,
                "signal_column": signal_col,
                "window_size": window_size,
                "threshold": threshold,
            }
        )

        mlflow.log_metrics(
            {
                "train_pr_auc": train_metrics["pr_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "val_f1": val_metrics["f1"],
                "test_f1": test_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "test_precision": test_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "test_recall": test_metrics["recall"],
            }
        )

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(pr_curve_path))
        mlflow.log_artifact(str(hist_path))
        mlflow.log_artifact(str(timeline_path))

    logger.success(
        "Matrix Profile done | test_pr_auc={:.4f} test_f1={:.4f} threshold={:.4f} metrics={}",
        test_metrics["pr_auc"],
        test_metrics["f1"],
        threshold,
        metrics_path,
    )


if __name__ == "__main__":
    run_matrix_profile()
