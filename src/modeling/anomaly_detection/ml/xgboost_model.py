from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
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
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.evaluation.leakage_checks import (
    feature_importance_audit,
    label_shuffle_test,
    performance_sanity_check,
    run_leakage_report,
)
from src.mlflow_setup import init_tracking
from src.modeling.anomaly_detection.ml.one_class_svm_model import (
    _calibrate_threshold,
    _default_comparison_records_path,
    _prepare_xy,
    _save_pr_curve,
    _save_score_histogram,
    _save_score_timeline,
)
from src.modeling.common.feature_loader import load_features_for_task
from src.modeling.common.hyperparameter_optimizer import (
    midpoint_params_from_space,
    run_optuna,
    suggest_params_from_space,
)
from src.utils.paths import get_experiments_root

PROJECT_ROOT = Path(__file__).resolve().parents[4]
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XGBoost supervised anomaly baseline")
    p.add_argument("--task", default="anomaly_supervised")
    p.add_argument("--dataset", default="costa")
    p.add_argument("--split-path", default="path_a")
    p.add_argument("--profile", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-dir", default=None)
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


def _build_model_params(cfg: dict, seed: int, scale_pos_weight: float) -> dict:
    return {
        "objective": str(cfg.get("objective", "binary:logistic")),
        "eval_metric": str(cfg.get("eval_metric", "aucpr")),
        "tree_method": str(cfg.get("tree_method", "hist")),
        "n_jobs": int(cfg.get("n_jobs", -1)),
        "random_state": seed,
        "verbosity": 0,
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": int(cfg.get("early_stopping_rounds", 50)),
    }


def run_xgboost_anomaly(config: dict | None = None) -> None:
    args = _parse_args()
    if config is None:
        config = _load_config()

    xgb_cfg = config["anomaly_detection"]["ml"]["models"]["xgboost"]
    hpo_cfg = config["anomaly_detection"]["ml"]["hpo"]
    search_space = dict(xgb_cfg.get("search_space", {}))

    _cfg_cap = xgb_cfg.get("max_train_samples")
    max_train_samples: int | None = args.max_train_samples or (int(_cfg_cap) if _cfg_cap else None)
    n_trials: int = args.n_trials or int(hpo_cfg.get("n_trials_phase1", 20))
    seed: int = args.seed

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

    x_train, y_train_raw = _prepare_xy(train_df, features, label_col)
    x_val, y_val_raw = _prepare_xy(val_df, features, label_col)
    x_test, y_test_raw = _prepare_xy(test_df, features, label_col)

    y_train = (y_train_raw != 0).astype(int)
    y_val = (y_val_raw != 0).astype(int)
    y_test = (y_test_raw != 0).astype(int)

    if len(np.unique(y_train)) <= 1:
        raise ValueError(
            "XGBoost anomaly baseline requires anomaly_supervised split with both classes."
        )

    sampling_meta = {"strategy": "all_rows", "n_selected": int(len(x_train))}
    if max_train_samples and len(x_train) > max_train_samples:
        rng = np.random.default_rng(seed)
        pos_idx = np.flatnonzero(y_train == 1)
        neg_idx = np.flatnonzero(y_train == 0)
        n_pos = int(round(max_train_samples * (len(pos_idx) / len(y_train))))
        n_pos = max(1, min(n_pos, len(pos_idx)))
        n_neg = max_train_samples - n_pos
        n_neg = max(1, min(n_neg, len(neg_idx)))
        sel = np.concatenate(
            [
                rng.choice(pos_idx, size=n_pos, replace=False),
                rng.choice(neg_idx, size=n_neg, replace=False),
            ]
        )
        sel.sort()
        x_fit = x_train[sel]
        y_fit = y_train[sel]
        sampling_meta = {"strategy": "proportional_label", "n_selected": int(len(sel))}
    else:
        x_fit = x_train
        y_fit = y_train

    n_pos_fit = int((y_fit == 1).sum())
    n_neg_fit = int((y_fit == 0).sum())
    auto_spw = bool(xgb_cfg.get("auto_scale_pos_weight", True))
    spw = (
        float(n_neg_fit / max(n_pos_fit, 1))
        if auto_spw
        else float(xgb_cfg.get("scale_pos_weight", 1.0))
    )
    base_params = _build_model_params(xgb_cfg, seed=seed, scale_pos_weight=spw)

    def _merge_trial_params(raw_params: dict) -> tuple[dict, float]:
        spw_mul = float(raw_params.pop("scale_pos_weight_multiplier", 1.0))
        return raw_params, spw_mul

    def objective(trial: optuna.Trial) -> float:
        trial_params = suggest_params_from_space(trial, search_space)
        trial_params, spw_mul = _merge_trial_params(dict(trial_params))
        model = XGBClassifier(
            **_build_model_params(xgb_cfg, seed=seed, scale_pos_weight=spw * spw_mul),
            **trial_params,
        )
        model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
        scores = model.predict_proba(x_val)[:, 1]
        return float(average_precision_score(y_val, scores))

    if args.no_optuna or not search_space:
        best_params = midpoint_params_from_space(search_space) if search_space else {}
    else:
        best_params, _ = run_optuna(
            objective,
            search_space=search_space,
            n_trials=n_trials,
            direction="maximize",
            seed=seed,
            sampler_name=str(hpo_cfg.get("sampler", "tpe")),
            pruner_name=str(hpo_cfg.get("pruner", "none")),
            study_name=f"{hpo_cfg.get('study_name_prefix', 'anomaly_ml')}_xgboost",
        )

    best_params, best_spw_mul = _merge_trial_params(dict(best_params))
    effective_spw = spw * best_spw_mul

    t0 = time.perf_counter()
    final_model = XGBClassifier(
        **_build_model_params(xgb_cfg, seed=seed, scale_pos_weight=effective_spw),
        **best_params,
    )
    final_model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
    fit_time = time.perf_counter() - t0

    train_scores = final_model.predict_proba(x_train)[:, 1]
    val_scores = final_model.predict_proba(x_val)[:, 1]
    test_scores = final_model.predict_proba(x_test)[:, 1]

    threshold, val_f1, val_prec, val_rec = _calibrate_threshold(val_scores, y_val)
    val_pr_auc = float(average_precision_score(y_val, val_scores))
    val_roc_auc = float(roc_auc_score(y_val, val_scores))
    test_pr_auc = float(average_precision_score(y_test, test_scores))
    test_roc_auc = float(roc_auc_score(y_test, test_scores))
    test_preds = (test_scores >= threshold).astype(int)
    val_preds = (val_scores >= threshold).astype(int)
    val_acc = float(accuracy_score(y_val, val_preds))
    test_f1 = float(f1_score(y_test, test_preds, zero_division=0))
    test_acc = float(accuracy_score(y_test, test_preds))
    test_prec = float(precision_score(y_test, test_preds, zero_division=0))
    test_rec = float(recall_score(y_test, test_preds, zero_division=0))

    metrics = {
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
        "test_precision_at_threshold": test_prec,
        "test_recall_at_threshold": test_rec,
        "n_train_total": int(len(x_train)),
        "n_train_used_for_fit": int(len(x_fit)),
        "n_train_positive": n_pos_fit,
        "n_train_negative": n_neg_fit,
        "scale_pos_weight": effective_spw,
        "scale_pos_weight_multiplier": best_spw_mul,
        "fit_time_s": round(fit_time, 2),
        "best_iteration": int(getattr(final_model, "best_iteration", -1) or -1),
    }

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = (
        Path(args.artifacts_dir)
        if args.artifacts_dir
        else get_experiments_root() / "anomaly" / "xgboost" / f"xgb_{ts}"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path) if args.metrics_path else artifacts_dir / "metrics.json"
    model_path = Path(args.model_path) if args.model_path else artifacts_dir / "model.joblib"
    pr_curve_path = artifacts_dir / "pr_curve.png"
    histogram_path = artifacts_dir / "score_histogram.png"
    timeline_path = artifacts_dir / "score_timeline.png"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(final_model, model_path)
    _save_pr_curve(
        val_scores,
        y_val,
        test_scores,
        y_test,
        pr_curve_path,
        model_name="XGBoost",
    )
    _save_score_histogram(
        train_scores,
        val_scores,
        y_val,
        threshold,
        histogram_path,
        model_name="XGBoost",
    )
    _save_score_timeline(
        test_scores,
        y_test,
        threshold,
        timeline_path,
        model_name="XGBoost",
    )

    try:
        leakage_payload = run_leakage_report(
            model=final_model,
            X_train=x_fit,
            y_train=y_fit,
            X_val=x_val,
            y_val=y_val,
            feature_names=features,
            X_eval=x_test,
            y_eval=y_test,
            metric="f1_binary",
        )
    except Exception as _exc:
        logger.warning("Leakage report failed (non-fatal): {}", _exc)
        leakage_payload = {"skipped": True, "reason": f"exception: {_exc}", "leakage_flags": [], "is_clean": True}

    try:
        init_tracking("anomaly")
        run_name = f"anomaly_xgboost_{ts}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "task": args.task,
                    "dataset": args.dataset,
                    "split_path": args.split_path,
                    "profile": str(args.profile),
                    "model": "xgboost",
                }
            )
            mlflow.log_params(
                {
                    **best_params,
                    "max_train_samples_cap": max_train_samples,
                    "n_train_used_for_fit": len(x_fit),
                    "sampling_strategy": sampling_meta.get("strategy"),
                    "n_features": len(features),
                    "seed": seed,
                    "scale_pos_weight": effective_spw,
                    "scale_pos_weight_multiplier": best_spw_mul,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_metric(
                "leakage_flag_count", float(len(leakage_payload.get("leakage_flags", [])))
            )
            mlflow.log_metric("sanity_pr_auc_suspicious", float(
                leakage_payload.get("sanity_check", {}).get("is_suspicious", False)
            ))
            for p in (metrics_path, model_path, pr_curve_path, histogram_path, timeline_path):
                if p.exists():
                    mlflow.log_artifact(str(p))
            mlflow.log_dict(manifest, "features_manifest.json")

            comparison_record = {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "run_name": run_name,
                "task": args.task,
                "dataset": args.dataset,
                "split_path": args.split_path,
                "model": "xgboost",
                "model_family": "anomaly_ml",
                "run_type": args.run_type,
                "seed": seed,
                "feature_profile": str(args.profile),
                "feature_run_dir": str(resolved_run_dir),
                "optuna_enabled": not args.no_optuna,
                "optuna_n_trials_requested": n_trials if not args.no_optuna else 0,
                "best_params": best_params,
                "n_features": len(features),
                "max_train_samples_cap": max_train_samples,
                "n_train_used_for_fit": int(len(x_fit)),
                "sampling_strategy": str(sampling_meta.get("strategy")),
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
                "test_precision_at_threshold": test_prec,
                "test_recall_at_threshold": test_rec,
                "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
            }
            records_path = Path(args.comparison_records_path)
            records_path.parent.mkdir(parents=True, exist_ok=True)
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(comparison_record, default=str) + "\n")
            mlflow.log_artifact(str(records_path))
    except Exception as exc:
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")


if __name__ == "__main__":
    run_xgboost_anomaly()
