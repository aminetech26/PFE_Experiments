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
import yaml
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.evaluation.leakage_checks import performance_sanity_check
from src.mlflow_setup import init_tracking
from src.modeling.anomaly_detection.ml.one_class_svm_model import (
    _calibrate_threshold,
    _default_comparison_records_path,
    _pick_sampling_group_column,
    _prepare_xy,
    _save_pr_curve,
    _set_ocsvm_threshold_config as _set_shared_threshold_config,
    _save_score_histogram,
    _save_score_timeline,
    _subsample_train_indices,
)
from src.modeling.common.artifact_contract import (
    build_candidate_per_true_class_thresholds,
    build_deployment_manifest,
    build_run_manifest,
    build_score_calibration_payload,
    compute_macro_per_class_pr_auc,
    compute_anomaly_per_class_metrics,
    write_json,
)
from src.modeling.common.episode_metrics import episode_macro_f1_binary
from src.modeling.common.feature_loader import load_features_for_task
from src.modeling.common.fold_override import add_fold_override_args, apply_fold_overrides
from src.modeling.common.hyperparameter_optimizer import (
    midpoint_params_from_space,
    run_optuna,
    suggest_params_from_space,
)
from src.utils.paths import get_experiments_root

PROJECT_ROOT = Path(__file__).resolve().parents[4]
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolation Forest anomaly detection baseline")
    p.add_argument("--task", default="anomaly_semisup")
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
    p.add_argument("--best-params", default=None,
                   help="JSON string of best params to apply directly (skips HPO; equivalent to --no-optuna with explicit params)")
    add_fold_override_args(p)
    return p.parse_args()


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _anomaly_score(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    """Higher score = more anomalous."""
    return -model.score_samples(x)


def run_isolation_forest(config: dict | None = None) -> None:
    args = _parse_args()
    if config is None:
        config = _load_config()

    if_cfg = config["anomaly_detection"]["ml"]["models"]["isolation_forest"]
    # Shared threshold calibration (GPD by default)
    from src.modeling.common.threshold_calibration import (
        load_threshold_config as _load_thr_cfg,
        threshold_policy_str as _thr_policy_str,
    )
    _thr_cfg = _load_thr_cfg(config, if_cfg)
    _set_shared_threshold_config(_thr_cfg)
    _iforest_threshold_policy = _thr_policy_str(
        {"strategy": _thr_cfg["strategy"], "percentile": _thr_cfg["percentile"],
         "pot_quantile": _thr_cfg["pot_quantile"], "target_tail_prob": _thr_cfg["target_tail_prob"]}
    )
    hpo_cfg = config["anomaly_detection"]["ml"]["hpo"]

    fixed_params: dict = {
        "random_state": args.seed,
        "n_jobs": int(if_cfg.get("n_jobs", -1)),
        "bootstrap": bool(if_cfg.get("bootstrap", False)),
        "warm_start": bool(if_cfg.get("warm_start", False)),
        "contamination": if_cfg.get("contamination", "auto"),
    }

    search_space = dict(if_cfg.get("search_space", {}))
    _cfg_cap = if_cfg.get("max_train_samples")
    max_train_samples: int | None = args.max_train_samples or (int(_cfg_cap) if _cfg_cap else None)
    n_trials: int = args.n_trials or int(hpo_cfg.get("n_trials_phase1", 20))
    seed: int = args.seed

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
    val_df, test_df = apply_fold_overrides(args, val_df, test_df)
    features: list[str] = manifest.get("final_features", [])
    label_col: str = str(manifest.get("label_column", "label"))

    x_train, y_train = _prepare_xy(train_df, features, label_col)
    x_val, y_val = _prepare_xy(val_df, features, label_col)
    x_test, y_test = _prepare_xy(test_df, features, label_col)

    y_val_bin = (y_val != 0).astype(int)
    y_test_bin = (y_test != 0).astype(int)

    _ep_group_col = _pick_sampling_group_column(val_df, args.split_path)
    val_group_ids = val_df[_ep_group_col].to_numpy() if _ep_group_col and _ep_group_col in val_df.columns else None
    test_group_ids = test_df[_ep_group_col].to_numpy() if _ep_group_col and _ep_group_col in test_df.columns else None

    non_normal_in_train = int((y_train != 0).sum())
    if non_normal_in_train:
        logger.warning(
            "Train contains {} non-normal rows — expected all-normal for semisup.",
            non_normal_in_train,
        )

    logger.info(
        f"Rows — train: {len(x_train):,}  val: {len(x_val):,} (faults: {y_val_bin.sum():,})  "
        f"test: {len(x_test):,} (faults: {y_test_bin.sum():,})"
    )
    logger.info(
        f"Features: {len(features)} | Subsample cap: {max_train_samples or 'none (all rows)'}"
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

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
        "IF fit on {:,} samples (from {:,} train rows) | sampling={} group_col={} groups={}",
        len(x_fit),
        len(x_train_scaled),
        sampling_meta.get("strategy"),
        sampling_meta.get("group_col"),
        sampling_meta.get("n_groups"),
    )

    def objective(trial: optuna.Trial) -> float:
        trial_params = suggest_params_from_space(trial, search_space)
        model = IsolationForest(**fixed_params, **trial_params)
        model.fit(x_fit)
        scores = _anomaly_score(model, x_val_scaled)
        macro = compute_macro_per_class_pr_auc(labels=y_val, scores=scores)
        val_macro = macro.get("macro_per_class_pr_auc")
        return float(val_macro) if isinstance(val_macro, (int, float, np.integer, np.floating)) else 0.0

    if args.best_params:
        best_params = json.loads(args.best_params)
        logger.info("Applying best_params from CLI (skipping HPO): {}", best_params)
        study = None
    elif args.no_optuna or not search_space:
        best_params = midpoint_params_from_space(search_space) if search_space else {}
        logger.info("HPO skipped — using midpoint params: {}", best_params)
        study = None
    else:
        logger.info("Running Optuna HPO: {} trials", n_trials)
        best_params, study = run_optuna(
            objective,
            search_space=search_space,
            n_trials=n_trials,
            direction="maximize",
            seed=seed,
            sampler_name=str(hpo_cfg.get("sampler", "tpe")),
            pruner_name=str(hpo_cfg.get("pruner", "none")),
            study_name=f"{hpo_cfg.get('study_name_prefix', 'anomaly_ml')}_iforest",
        )
        logger.info(
            "Best params: {} | Best val objective (macro per-class PR-AUC): {:.4f}",
            best_params,
            study.best_value,
        )

    logger.info("Fitting final model...")
    t0 = time.perf_counter()
    final_model = IsolationForest(**fixed_params, **best_params)
    final_model.fit(x_fit)
    fit_time = time.perf_counter() - t0

    train_scores = _anomaly_score(final_model, x_train_scaled)
    val_scores = _anomaly_score(final_model, x_val_scaled)
    test_scores = _anomaly_score(final_model, x_test_scaled)

    threshold, val_f1, val_prec, val_rec = _calibrate_threshold(val_scores, y_val_bin)
    logger.info(
        "Val — threshold={:.4f} F1={:.4f} Prec={:.4f} Rec={:.4f}",
        threshold,
        val_f1,
        val_prec,
        val_rec,
    )

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

    val_episode_macro_f1 = episode_macro_f1_binary(y_val_bin, val_preds, val_group_ids)
    test_episode_macro_f1 = episode_macro_f1_binary(y_test_bin, test_preds, test_group_ids)
    val_macro_stats = compute_macro_per_class_pr_auc(labels=y_val, scores=val_scores)
    test_macro_stats = compute_macro_per_class_pr_auc(labels=y_test, scores=test_scores)
    val_macro_per_class_pr_auc = val_macro_stats.get("macro_per_class_pr_auc")
    val_worst_class_pr_auc = val_macro_stats.get("worst_class_pr_auc")
    test_macro_per_class_pr_auc = test_macro_stats.get("macro_per_class_pr_auc")
    test_worst_class_pr_auc = test_macro_stats.get("worst_class_pr_auc")
    val_selection_score = (
        float(val_macro_per_class_pr_auc)
        if isinstance(val_macro_per_class_pr_auc, (int, float, np.integer, np.floating))
        else 0.0
    )

    metrics: dict = {
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "val_f1_at_threshold": val_f1,
        "val_accuracy_at_threshold": val_acc,
        "val_precision_at_threshold": val_prec,
        "val_recall_at_threshold": val_rec,
        "val_episode_macro_f1": val_episode_macro_f1,
        "val_macro_per_class_pr_auc": val_macro_per_class_pr_auc,
        "val_worst_class_pr_auc": val_worst_class_pr_auc,
        "val_selection_score": val_selection_score,
        "threshold": threshold,
        "threshold_policy": _iforest_threshold_policy,
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_f1_at_threshold": test_f1,
        "test_accuracy_at_threshold": test_acc,
        "test_precision_at_threshold": test_prec_val,
        "test_recall_at_threshold": test_rec_val,
        "test_episode_macro_f1": test_episode_macro_f1,
        "test_macro_per_class_pr_auc": test_macro_per_class_pr_auc,
        "test_worst_class_pr_auc": test_worst_class_pr_auc,
        "test_class1_pr_auc_vs_normal": test_macro_stats.get("per_class_pr_auc_vs_normal", {}).get("1"),
        "test_class2_pr_auc_vs_normal": test_macro_stats.get("per_class_pr_auc_vs_normal", {}).get("2"),
        "test_class3_pr_auc_vs_normal": test_macro_stats.get("per_class_pr_auc_vs_normal", {}).get("3"),
        "test_class4_pr_auc_vs_normal": test_macro_stats.get("per_class_pr_auc_vs_normal", {}).get("4"),
        "n_train_total": int(len(x_train)),
        "n_train_used_for_fit": int(len(x_fit)),
        "sampling_groups": int(sampling_meta.get("n_groups", 0)),
        "fit_time_s": round(fit_time, 2),
    }

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        artifacts_dir = get_experiments_root() / "anomaly" / "isolation_forest" / f"iforest_{ts}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path) if args.model_path else artifacts_dir / "model.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"
    global_metrics_path = artifacts_dir / "global_metrics.json"
    per_class_metrics_path = artifacts_dir / "per_class_metrics.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    deployment_manifest_path = artifacts_dir / "deployment_manifest.json"
    features_manifest_path = artifacts_dir / "features_manifest.json"
    pr_curve_path = artifacts_dir / "pr_curve.png"
    histogram_path = artifacts_dir / "score_histogram.png"
    timeline_path = artifacts_dir / "score_timeline.png"
    best_params_path = artifacts_dir / "hpo_best_params.json"
    best_params_path.write_text(json.dumps(best_params, indent=2, default=str), encoding="utf-8")

    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)
    per_class_metrics = compute_anomaly_per_class_metrics(
        labels=y_test,
        scores=test_scores,
        threshold=threshold,
        val_labels=y_val,
        val_scores=val_scores,
    )
    candidate_per_true_class_thresholds = build_candidate_per_true_class_thresholds(
        per_class_metrics,
        normal_label=0,
    )
    run_name = f"anomaly_isolation_forest_{ts}"
    run_manifest = build_run_manifest(
        task=args.task,
        model="isolation_forest",
        model_family="anomaly_ml",
        dataset=args.dataset,
        split_path=args.split_path,
        feature_profile=str(args.profile),
        feature_run_dir=str(resolved_run_dir),
        seed=seed,
        run_type=args.run_type,
        extras={"run_name": run_name},
    )
    score_calibration_path = artifacts_dir / "score_calibration.json"
    deployment_manifest = build_deployment_manifest(
        task=args.task,
        model="isolation_forest",
        model_family="anomaly_ml",
        model_artifact=model_path.name,
        scaler_artifact=scaler_path.name,
        feature_names=features,
        label_column=label_col,
        threshold=threshold,
        score_direction="higher_is_more_anomalous",
        classes=[str(c) for c in sorted(np.unique(y_test).tolist())],
        extras={
            "threshold_policy": _iforest_threshold_policy,
            "score_calibration_artifact": score_calibration_path.name,
        },
    )
    write_json(
        score_calibration_path,
        build_score_calibration_payload(
            threshold=threshold,
            threshold_policy=_iforest_threshold_policy,
            threshold_quantile=None,
            candidate_per_true_class_thresholds=candidate_per_true_class_thresholds,
        ),
    )
    write_json(global_metrics_path, metrics)
    write_json(per_class_metrics_path, per_class_metrics)
    write_json(run_manifest_path, run_manifest)
    write_json(deployment_manifest_path, deployment_manifest)
    write_json(features_manifest_path, manifest)
    _save_pr_curve(
        val_scores,
        y_val_bin,
        test_scores,
        y_test_bin,
        pr_curve_path,
        model_name="Isolation Forest",
    )
    _save_score_histogram(
        train_scores,
        val_scores,
        y_val_bin,
        threshold,
        histogram_path,
        model_name="Isolation Forest",
    )
    _save_score_timeline(
        test_scores,
        y_test_bin,
        threshold,
        timeline_path,
        model_name="Isolation Forest",
    )

    try:
        sanity_check = performance_sanity_check("test_pr_auc", test_pr_auc)
    except Exception as _exc:
        logger.warning("Sanity check failed (non-fatal): {}", _exc)
        sanity_check = {"is_suspicious": False, "skipped": True}

    try:
        init_tracking("anomaly")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags(
                {
                    "task": args.task,
                    "dataset": args.dataset,
                    "split_path": args.split_path,
                    "profile": str(args.profile),
                    "model": "isolation_forest",
                }
            )
            mlflow.log_params(
                {
                    **best_params,
                    "max_train_samples_cap": max_train_samples,
                    "n_train_used_for_fit": len(x_fit),
                    "sampling_strategy": str(sampling_meta.get("strategy")),
                    "sampling_group_col": str(sampling_meta.get("group_col")),
                    "sampling_n_groups": int(sampling_meta.get("n_groups", 0)),
                    "n_features": len(features),
                    "scaling": "standard",
                    "no_optuna": args.no_optuna,
                    "n_optuna_trials": n_trials if not args.no_optuna else 0,
                    "seed": seed,
                }
            )
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                }
            )
            mlflow.log_metric("sanity_pr_auc_suspicious", float(sanity_check["is_suspicious"]))
            for cls_str, m in per_class_metrics.items():
                if m.get("per_class_threshold") is not None:
                    mlflow.log_metric(f"test_f1_class{cls_str}_per_class", m["f1_at_per_class_threshold"])
                    mlflow.log_metric(f"test_f1_class{cls_str}_global", m["f1_at_threshold_vs_normal"])
                    mlflow.log_metric(f"per_class_threshold_class{cls_str}", m["per_class_threshold"])
                    mlflow.log_metric(
                        f"candidate_per_true_class_threshold_class{cls_str}",
                        m["candidate_per_true_class_threshold"],
                    )
                if m.get("pr_auc_vs_normal") is not None:
                    mlflow.log_metric(f"test_pr_auc_class{cls_str}_vs_normal", m["pr_auc_vs_normal"])
            for p in (
                global_metrics_path,
                per_class_metrics_path,
                run_manifest_path,
                deployment_manifest_path,
                features_manifest_path,
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
                "model": "isolation_forest",
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
                "sampling_group_col": str(sampling_meta.get("group_col")),
                "sampling_n_groups": int(sampling_meta.get("n_groups", 0)),
                "fit_time_s": round(fit_time, 2),
                "threshold": threshold,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "val_f1_at_threshold": val_f1,
                "val_macro_per_class_pr_auc": val_macro_per_class_pr_auc,
                "val_worst_class_pr_auc": val_worst_class_pr_auc,
                "val_accuracy_at_threshold": val_acc,
                "test_pr_auc": test_pr_auc,
                "test_roc_auc": test_roc_auc,
                "test_macro_per_class_pr_auc": test_macro_per_class_pr_auc,
                "test_worst_class_pr_auc": test_worst_class_pr_auc,
                "test_class1_pr_auc_vs_normal": test_macro_stats.get("per_class_pr_auc_vs_normal", {}).get("1"),
                "test_class2_pr_auc_vs_normal": test_macro_stats.get("per_class_pr_auc_vs_normal", {}).get("2"),
                "test_class3_pr_auc_vs_normal": test_macro_stats.get("per_class_pr_auc_vs_normal", {}).get("3"),
                "test_class4_pr_auc_vs_normal": test_macro_stats.get("per_class_pr_auc_vs_normal", {}).get("4"),
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
    except Exception as exc:
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")


if __name__ == "__main__":
    run_isolation_forest()
