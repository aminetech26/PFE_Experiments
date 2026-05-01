from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.mlflow_setup import init_tracking
from src.modeling.common.feature_loader import load_features_for_task
from src.modeling.common.hyperparameter_optimizer import (
    midpoint_params_from_space,
    run_optuna,
    suggest_params_from_space,
)
from src.utils.paths import get_experiments_root

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _default_comparison_records_path() -> Path:
    return get_experiments_root() / "metrics" / "anomaly_comparison_records.jsonl"


optuna.logging.set_verbosity(optuna.logging.WARNING)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-Class SVM anomaly detection baseline")
    p.add_argument("--task", default="anomaly_semisup")
    p.add_argument("--dataset", default="costa")
    p.add_argument("--split-path", default="path_a")
    p.add_argument("--profile", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--kernel", choices=["rbf", "poly"], required=True)
    p.add_argument("--no-optuna", action="store_true")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--metrics-path", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--comparison-records-path", default=str(_default_comparison_records_path()))
    p.add_argument("--run-type", default="baseline", help="baseline | ablation | final")
    return p.parse_args()


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _prepare_xy(
    df: pd.DataFrame, features: list[str], label_col: str
) -> tuple[np.ndarray, np.ndarray]:
    x = df[features].to_numpy(dtype=np.float64)
    y = df[label_col].to_numpy()
    return x, y


def _anomaly_score(model: OneClassSVM, x: np.ndarray) -> np.ndarray:
    """Higher score = more anomalous (negated signed distance)."""
    return -model.decision_function(x)


def _pick_sampling_group_column(df: pd.DataFrame, split_path: str) -> str | None:
    if split_path == "path_b":
        preferred = ["operating_day_id", "episode_id", "segment_id"]
    else:
        preferred = ["episode_id", "segment_id", "operating_day_id"]
    for col in preferred:
        if col in df.columns and bool(df[col].notna().any()):
            return col
    return None


def _subsample_train_indices(
    train_df: pd.DataFrame,
    n_target: int,
    seed: int,
    split_path: str,
) -> tuple[np.ndarray, dict]:
    n_rows = len(train_df)
    if n_target >= n_rows:
        return np.arange(n_rows, dtype=np.int64), {
            "strategy": "all_rows",
            "group_col": None,
            "n_groups": 0,
            "n_selected": int(n_rows),
        }

    group_col = _pick_sampling_group_column(train_df, split_path)
    rng = np.random.default_rng(seed)
    if group_col is None:
        idx = rng.choice(n_rows, size=n_target, replace=False)
        return np.sort(idx.astype(np.int64)), {
            "strategy": "row_random",
            "group_col": None,
            "n_groups": 0,
            "n_selected": int(n_target),
        }

    groups = train_df[group_col]
    valid_mask = groups.notna()
    valid_df = train_df.loc[valid_mask].copy()
    valid_df["__row_idx"] = np.flatnonzero(valid_mask.to_numpy())
    by_group = valid_df.groupby(group_col, sort=False)["__row_idx"].apply(list)
    group_keys = list(by_group.index)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    if n_groups == 0:
        idx = rng.choice(n_rows, size=n_target, replace=False)
        return np.sort(idx.astype(np.int64)), {
            "strategy": "row_random",
            "group_col": None,
            "n_groups": 0,
            "n_selected": int(n_target),
        }

    base_quota = n_target // n_groups
    remainder = n_target % n_groups
    selected: list[int] = []

    for rank, g in enumerate(group_keys):
        rows = by_group[g]
        if not rows:
            continue
        quota = base_quota + (1 if rank < remainder else 0)
        if quota <= 0:
            continue
        if quota >= len(rows):
            selected.extend(rows)
            continue

        rows_sorted = np.array(sorted(rows), dtype=np.int64)
        pos = np.linspace(0, len(rows_sorted) - 1, num=quota, dtype=int)
        selected.extend(rows_sorted[pos].tolist())

    selected_arr = np.array(sorted(set(selected)), dtype=np.int64)
    if len(selected_arr) < n_target:
        remaining = np.setdiff1d(np.arange(n_rows, dtype=np.int64), selected_arr)
        top_up = rng.choice(remaining, size=n_target - len(selected_arr), replace=False)
        selected_arr = np.sort(np.concatenate([selected_arr, top_up.astype(np.int64)]))
    elif len(selected_arr) > n_target:
        selected_arr = np.sort(
            rng.choice(selected_arr, size=n_target, replace=False).astype(np.int64)
        )

    return selected_arr, {
        "strategy": "group_quota_temporal_spacing",
        "group_col": group_col,
        "n_groups": int(n_groups),
        "n_selected": int(len(selected_arr)),
    }


def _calibrate_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
    """Return (threshold, best_f1, precision, recall) by maximising F1 on the PR curve."""
    prec, rec, thresholds = precision_recall_curve(labels, scores)
    # prec/rec have len = n_thresholds + 1; zip with thresholds (len = n_thresholds)
    f1_vals = np.where(
        (prec[:-1] + rec[:-1]) > 0,
        2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]),
        0.0,
    )
    best_idx = int(np.argmax(f1_vals))
    return (
        float(thresholds[best_idx]),
        float(f1_vals[best_idx]),
        float(prec[best_idx]),
        float(rec[best_idx]),
    )


def _save_pr_curve(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for scores, labels, split in [
        (val_scores, val_labels, "Val"),
        (test_scores, test_labels, "Test"),
    ]:
        prec, rec, _ = precision_recall_curve(labels, scores)
        auc = average_precision_score(labels, scores)
        ax.plot(rec, prec, label=f"{split} (PR-AUC={auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve — One-Class SVM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_score_histogram(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(train_scores, bins=80, alpha=0.5, label="Train (normal)", density=True)
    ax.hist(val_scores[val_labels == 0], bins=60, alpha=0.5, label="Val — normal", density=True)
    if val_labels.sum() > 0:
        ax.hist(val_scores[val_labels == 1], bins=60, alpha=0.5, label="Val — fault", density=True)
    ax.axvline(
        threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.3f}"
    )
    ax.set_xlabel("Anomaly score (−decision_function)")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution — One-Class SVM")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_score_timeline(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = np.where(test_labels == 1, "red", "steelblue")
    ax.scatter(np.arange(len(test_scores)), test_scores, c=colors, s=2, alpha=0.5, rasterized=True)
    ax.axhline(
        threshold, color="orange", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.3f}"
    )
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Anomaly score")
    ax.set_title("Score Timeline (Test) — One-Class SVM  |  blue=normal  red=fault")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run_one_class_svm(config: dict | None = None) -> None:
    args = _parse_args()
    if config is None:
        config = _load_config()

    kernel = args.kernel
    ocsvm_cfg = config["anomaly_detection"]["ml"]["models"]["one_class_svm"]
    hpo_cfg = config["anomaly_detection"]["ml"]["hpo"]
    kernel_space: dict = ocsvm_cfg["kernels"][kernel]

    # Fixed SVM params (not tuned)
    fixed_params: dict = {
        "cache_size": int(ocsvm_cfg.get("cache_size_mb", 1000)),
        "shrinking": bool(ocsvm_cfg.get("shrinking", True)),
        "tol": float(ocsvm_cfg.get("tol", 1e-3)),
        "max_iter": -1,
    }
    if kernel == "poly":
        fixed_params["degree"] = int(kernel_space.get("degree", 3))

    # HPO search space = everything except fixed scalars like 'degree'
    search_space = {k: v for k, v in kernel_space.items() if k != "degree"}

    _cfg_cap = ocsvm_cfg.get("max_train_samples")  # null in YAML → None → no cap
    max_train_samples: int | None = args.max_train_samples or (int(_cfg_cap) if _cfg_cap else None)
    n_trials: int = args.n_trials or int(hpo_cfg.get("n_trials_phase1", 20))
    seed: int = args.seed

    # ── Load features ────────────────────────────────────────────────────────
    logger.info(
        f"Loading features | task={args.task} dataset={args.dataset} "
        f"split_path={args.split_path} profile={args.profile}"
    )
    train_df, val_df, test_df, manifest, resolved_run_dir = load_features_for_task(
        task=args.task,
        profile=args.profile,
        run_dir=args.run_dir,
        run_id=args.run_id,
        dataset=args.dataset,
        split_path=args.split_path,
    )
    features: list[str] = manifest.get("final_features", [])
    label_col: str = str(manifest.get("label_column", "label"))

    x_train, y_train = _prepare_xy(train_df, features, label_col)
    x_val, y_val = _prepare_xy(val_df, features, label_col)
    x_test, y_test = _prepare_xy(test_df, features, label_col)

    # Binary labels for evaluation (0=normal, 1=any fault)
    y_val_bin = (y_val != 0).astype(int)
    y_test_bin = (y_test != 0).astype(int)

    non_normal_in_train = int((y_train != 0).sum())
    if non_normal_in_train:
        logger.warning(
            f"Train contains {non_normal_in_train} non-normal rows — expected all-normal for semisup."
        )

    logger.info(
        f"Rows — train: {len(x_train):,}  val: {len(x_val):,} (faults: {y_val_bin.sum():,})  "
        f"test: {len(x_test):,} (faults: {y_test_bin.sum():,})"
    )
    logger.info(
        f"Features: {len(features)} | Subsample cap: {max_train_samples or 'none (all rows)'}"
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # Training subsample for SVM fit
    sampling_meta: dict = {
        "strategy": "all_rows",
        "group_col": None,
        "n_groups": 0,
        "n_selected": int(len(x_train_scaled)),
    }
    if max_train_samples and len(x_train_scaled) > max_train_samples:
        idx, sampling_meta = _subsample_train_indices(
            train_df=train_df,
            n_target=max_train_samples,
            seed=seed,
            split_path=args.split_path,
        )
        x_fit = x_train_scaled[idx]
    else:
        x_fit = x_train_scaled
    logger.info(
        "SVM fit on {:,} samples (from {:,} train rows) | sampling={} group_col={} groups={}",
        len(x_fit),
        len(x_train_scaled),
        sampling_meta.get("strategy"),
        sampling_meta.get("group_col"),
        sampling_meta.get("n_groups"),
    )

    # ── HPO ───────────────────────────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        trial_params = suggest_params_from_space(trial, search_space)
        model = OneClassSVM(kernel=kernel, **fixed_params, **trial_params)
        model.fit(x_fit)
        scores = _anomaly_score(model, x_val_scaled)
        return float(average_precision_score(y_val_bin, scores))

    if args.no_optuna:
        best_params = midpoint_params_from_space(search_space)
        study = None
        logger.info(f"HPO skipped — using midpoint params: {best_params}")
    else:
        logger.info(f"Running Optuna HPO: {n_trials} trials | kernel={kernel}")
        best_params, study = run_optuna(
            objective,
            search_space=search_space,
            n_trials=n_trials,
            direction="maximize",
            seed=seed,
            sampler_name=str(hpo_cfg.get("sampler", "tpe")),
            pruner_name=str(hpo_cfg.get("pruner", "none")),
            study_name=(f"{hpo_cfg.get('study_name_prefix', 'anomaly_ml')}_ocsvm_{kernel}"),
        )
        logger.info(f"Best params: {best_params} | Best val PR-AUC: {study.best_value:.4f}")

    # ── Final model ───────────────────────────────────────────────────────────
    logger.info("Fitting final model…")
    t0 = time.perf_counter()
    final_model = OneClassSVM(kernel=kernel, **fixed_params, **best_params)
    final_model.fit(x_fit)
    fit_time = time.perf_counter() - t0
    n_sv = int(final_model.support_vectors_.shape[0])
    logger.info(f"Fit done in {fit_time:.1f}s | support vectors: {n_sv:,}")

    # ── Scores ────────────────────────────────────────────────────────────────
    train_scores = _anomaly_score(final_model, x_train_scaled)
    val_scores = _anomaly_score(final_model, x_val_scaled)
    test_scores = _anomaly_score(final_model, x_test_scaled)

    # ── Threshold calibration ─────────────────────────────────────────────────
    threshold, val_f1, val_prec, val_rec = _calibrate_threshold(val_scores, y_val_bin)
    logger.info(
        f"Val — threshold={threshold:.4f} F1={val_f1:.4f} Prec={val_prec:.4f} Rec={val_rec:.4f}"
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    val_pr_auc = float(average_precision_score(y_val_bin, val_scores))
    val_roc_auc = float(roc_auc_score(y_val_bin, val_scores))
    test_pr_auc = float(average_precision_score(y_test_bin, test_scores))
    test_roc_auc = float(roc_auc_score(y_test_bin, test_scores))

    test_preds = (test_scores >= threshold).astype(int)
    val_preds = (val_scores >= threshold).astype(int)
    val_acc = float(accuracy_score(y_val_bin, val_preds))
    test_f1 = float(f1_score(y_test_bin, test_preds, zero_division=0))
    test_acc = float(accuracy_score(y_test_bin, test_preds))
    test_prec_val = float(precision_score(y_test_bin, test_preds, zero_division=0))
    test_rec_val = float(recall_score(y_test_bin, test_preds, zero_division=0))

    metrics: dict = {
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "val_f1_at_threshold": val_f1,
        "val_accuracy_at_threshold": val_acc,
        "val_precision_at_threshold": val_prec,
        "val_recall_at_threshold": val_rec,
        "threshold": threshold,
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_f1_at_threshold": test_f1,
        "test_accuracy_at_threshold": test_acc,
        "test_precision_at_threshold": test_prec_val,
        "test_recall_at_threshold": test_rec_val,
        "n_train_total": int(len(x_train)),
        "n_train_used_for_fit": int(len(x_fit)),
        "sampling_groups": int(sampling_meta.get("n_groups", 0)),
        "n_support_vectors": n_sv,
        "fit_time_s": round(fit_time, 2),
    }
    logger.info(
        f"Test — PR-AUC={test_pr_auc:.4f}  ROC-AUC={test_roc_auc:.4f}  ACC@thr={test_acc:.4f}  "
        f"F1@thr={test_f1:.4f}  Prec={test_prec_val:.4f}  Rec={test_rec_val:.4f}"
    )

    # ── Save artifacts ────────────────────────────────────────────────────────
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        artifacts_dir = get_experiments_root() / "anomaly" / "one_class_svm" / f"{kernel}_{ts}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics_path) if args.metrics_path else artifacts_dir / "metrics.json"
    model_path = Path(args.model_path) if args.model_path else artifacts_dir / "model.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"
    pr_curve_path = artifacts_dir / "pr_curve.png"
    histogram_path = artifacts_dir / "score_histogram.png"
    timeline_path = artifacts_dir / "score_timeline.png"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)
    _save_pr_curve(val_scores, y_val_bin, test_scores, y_test_bin, pr_curve_path)
    _save_score_histogram(train_scores, val_scores, y_val_bin, threshold, histogram_path)
    _save_score_timeline(test_scores, y_test_bin, threshold, timeline_path)
    logger.info(f"Artifacts saved → {artifacts_dir}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    try:
        init_tracking("anomaly")
        run_name = f"anomaly_one_class_svm_{kernel}_{ts}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "task": args.task,
                    "dataset": args.dataset,
                    "split_path": args.split_path,
                    "profile": str(args.profile),
                    "model": "one_class_svm",
                    "kernel": kernel,
                }
            )
            mlflow.log_params(
                {
                    **best_params,
                    "kernel": kernel,
                    "max_train_samples_cap": max_train_samples,
                    "n_train_used_for_fit": len(x_fit),
                    "sampling_strategy": str(sampling_meta.get("strategy")),
                    "sampling_group_col": str(sampling_meta.get("group_col")),
                    "sampling_n_groups": int(sampling_meta.get("n_groups", 0)),
                    "n_features": len(features),
                    "scaling": ocsvm_cfg.get("scaling", "standard"),
                    "no_optuna": args.no_optuna,
                    "n_optuna_trials": n_trials if not args.no_optuna else 0,
                    "seed": seed,
                }
            )
            mlflow.log_metrics(metrics)
            for p in (
                metrics_path,
                model_path,
                scaler_path,
                pr_curve_path,
                histogram_path,
                timeline_path,
            ):
                if p.exists():
                    mlflow.log_artifact(str(p))
            mlflow.log_dict(manifest, "features_manifest.json")

            comparison_record = {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "run_name": run_name,
                "task": args.task,
                "dataset": args.dataset,
                "split_path": args.split_path,
                "model": "one_class_svm",
                "model_family": "anomaly_ml",
                "kernel": kernel,
                "run_type": args.run_type,
                "seed": seed,
                "feature_profile": str(args.profile),
                "feature_run_dir": str(resolved_run_dir),
                "optuna_enabled": not args.no_optuna,
                "optuna_n_trials_requested": n_trials if not args.no_optuna else 0,
                "n_features": len(features),
                "max_train_samples_cap": max_train_samples,
                "n_train_used_for_fit": int(len(x_fit)),
                "sampling_strategy": str(sampling_meta.get("strategy")),
                "sampling_group_col": str(sampling_meta.get("group_col")),
                "sampling_n_groups": int(sampling_meta.get("n_groups", 0)),
                "n_support_vectors": n_sv,
                "fit_time_s": round(fit_time, 2),
                "threshold": threshold,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "val_f1_at_threshold": val_f1,
                "val_accuracy_at_threshold": val_acc,
                "test_pr_auc": test_pr_auc,
                "test_roc_auc": test_roc_auc,
                "test_accuracy_at_threshold": test_acc,
                "test_f1_at_threshold": test_f1,
                "test_precision_at_threshold": test_prec_val,
                "test_recall_at_threshold": test_rec_val,
                "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
            }
            records_path = Path(args.comparison_records_path)
            records_path.parent.mkdir(parents=True, exist_ok=True)
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(comparison_record, default=str) + "\n")
            mlflow.log_artifact(str(records_path))

        logger.info(f"MLflow run logged: {run_name}")
    except Exception as exc:
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")


if __name__ == "__main__":
    run_one_class_svm()
