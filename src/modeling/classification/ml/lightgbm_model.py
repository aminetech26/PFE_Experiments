from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import LabelEncoder, label_binarize

from src.data.splitting import PerClassSegmentTimeSeriesCV
from src.evaluation.leakage_checks import exact_duplicate_overlap_check, run_leakage_report
from src.mlflow_setup import init_tracking
from src.modeling.classification.ml.common import (
    build_probability_calibration_payload,
    build_study_name_prefix,
    compute_classification_metrics,
    default_comparison_records_path,
    fit_probability_calibrator,
    resolve_artifact_paths,
    resolve_calibration_config,
    save_classification_plots,
    save_confidence_histogram,
    save_reliability_diagram,
    write_classification_contract_artifacts,
    write_json_artifact,
)
from src.modeling.common.feature_loader import load_features_for_task
from src.modeling.common.hyperparameter_optimizer import (
    midpoint_params_from_space,
    run_optuna,
    suggest_params_from_space,
)
from src.modeling.common.system_resources import compute_thread_budget, detect_cpu_resources
from src.utils.paths import get_experiments_root

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


def _load_model_config() -> dict:
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_classification_ml_config(config: dict) -> tuple[dict, dict, dict]:
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


def _resolve_runtime_seed(runtime_config: dict, cli_seed: int | None) -> int:
    if cli_seed is not None:
        return int(cli_seed)
    exp = runtime_config.get("experiment", {})
    seeds = exp.get("seeds", [42])
    return int(seeds[0] if isinstance(seeds, list) else seeds)


def _prepare_xy(
    df: pd.DataFrame, features: list[str], label_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing features in dataframe: {missing_features}")
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataframe")

    x_data = df[features]
    y_data = df[label_column]
    return x_data, y_data


def _compute_pr_auc_multiclass(
    y_true_encoded: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> float:
    if len(classes) <= 1:
        return float("nan")
    y_true_bin = label_binarize(y_true_encoded, classes=np.arange(len(classes)))
    return float(average_precision_score(y_true_bin, y_proba, average="weighted"))


def _train_lightgbm(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict,
    eval_sets: list | None = None,
) -> tuple[lgb.LGBMClassifier, dict]:
    evals_result: dict = {}
    model = lgb.LGBMClassifier(**params)
    if eval_sets:
        model.fit(
            x_train,
            y_train,
            eval_set=eval_sets,
            callbacks=[lgb.record_evaluation(evals_result)],
        )
    else:
        model.fit(x_train, y_train)
    return model, evals_result


def _build_lightgbm_params(base: dict, seed: int, n_classes: int) -> dict:
    params = dict(base)
    params.update(
        {
            "objective": "multiclass",
            "num_class": int(n_classes),
            "random_state": int(seed),
            "class_weight": "balanced",
            "verbosity": -1,
        }
    )
    return params


def _resolve_threading(config: dict, args: argparse.Namespace) -> dict:
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


def run_lightgbm(config: dict | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Train Task B classification model (LightGBM + Optuna + MLflow)"
    )
    parser.add_argument(
        "--task", default="classification", choices=["classification"], help="Task name"
    )
    parser.add_argument("--dataset", default="costa", help="Dataset namespace under features root")
    parser.add_argument(
        "--split-path",
        default="path_a",
        choices=["path_a", "path_b"],
        help="Feature split path namespace",
    )
    parser.add_argument(
        "--profile", default="plus_physics", help="Feature profile used by featurize stage"
    )
    parser.add_argument("--run-dir", default=None, help="Optional explicit feature run directory")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional exact feature run id under data/processed/features/<task>/runs/",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Where to write metrics json",
    )
    parser.add_argument(
        "--leakage-report-path",
        default=None,
        help="Where to write leakage report json",
    )
    parser.add_argument(
        "--comparison-records-path",
        default=str(default_comparison_records_path()),
        help="Where to append comparison record jsonl",
    )
    parser.add_argument("--model-path", default=None, help="Where to persist trained model")
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional directory for metrics, model, and evaluation plots",
    )
    parser.add_argument(
        "--no-optuna", action="store_true", help="Disable Optuna and use midpoint baseline params"
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Override Optuna trial count")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment seed")
    parser.add_argument(
        "--run-type",
        default="baseline",
        help="Experiment run type label for comparison records (baseline|ablation|final)",
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="Override max model training threads"
    )
    parser.add_argument(
        "--optuna-jobs",
        type=int,
        default=None,
        help="Override parallel Optuna trials",
    )
    parser.add_argument(
        "--show-thread-plan", action="store_true", help="Print computed threading plan and exit"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logs")
    parser.add_argument(
        "--skip-leakage-checks", action="store_true", help="Skip leakage validation suite"
    )
    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    runtime_config = config or _load_model_config()
    runtime_cfg, _, hpo_cfg = _resolve_classification_ml_config(runtime_config)
    calibration_cfg = resolve_calibration_config(runtime_config)
    active_model = str(runtime_cfg.get("active_model", "lightgbm"))
    if active_model != "lightgbm":
        raise NotImplementedError(
            f"Model '{active_model}' is configured but this entrypoint only runs LightGBM."
        )

    model_spaces = runtime_cfg.get("model_spaces", {})
    lgb_space = model_spaces.get("lightgbm")
    if not lgb_space:
        raise KeyError(
            "Missing classification.ml.models.lightgbm search space in model_config.yaml"
        )

    n_trials = args.n_trials or int(hpo_cfg.get("n_trials_phase1", 50))
    hpo_direction = str(hpo_cfg.get("direction", "maximize"))
    hpo_seed = _resolve_runtime_seed(runtime_config, args.seed)
    hpo_timeout = hpo_cfg.get("timeout_seconds", None)
    hpo_sampler = str(hpo_cfg.get("sampler", "tpe"))
    hpo_pruner = hpo_cfg.get("pruner", "none")
    hpo_storage = hpo_cfg.get("storage_url", None)
    hpo_study_prefix = build_study_name_prefix(
        str(hpo_cfg.get("study_name_prefix", "classification_ml")),
        "lightgbm",
    )
    hpo_validation_mode = str(hpo_cfg.get("validation_mode", "holdout")).lower()
    threading_plan = _resolve_threading(runtime_config, args)

    if args.show_thread_plan:
        print(json.dumps(threading_plan, indent=2))
        return

    logger.info(
        "Thread plan | logical={} physical={} budget={} optuna_jobs={} threads_per_trial={}",
        threading_plan["cpu_logical_cores"],
        threading_plan["cpu_physical_cores"],
        threading_plan["thread_budget"],
        threading_plan["optuna_parallel_trials"],
        threading_plan["threads_per_trial"],
    )

    train_df, val_df, test_df, manifest, resolved_run_dir = load_features_for_task(
        task=args.task,
        profile=args.profile,
        run_dir=args.run_dir,
        run_id=args.run_id,
        dataset=args.dataset,
        split_path=args.split_path,
    )

    features = manifest.get("final_features", [])
    label_column = str(manifest.get("label_column", "label"))
    effective_profile = str(manifest.get("profile") or args.profile)
    requested_profile = args.profile
    effective_run_id = resolved_run_dir.name
    run_selection_mode = (
        "run_id" if args.run_id else "run_dir" if args.run_dir else "profile_latest"
    )
    if not features:
        raise ValueError("features_manifest.json does not contain final_features")

    logger.info(
        "Using feature run | dir={} requested_profile={} effective_profile={} run_id={} n_features={}",
        resolved_run_dir,
        requested_profile,
        effective_profile,
        args.run_id,
        len(features),
    )

    x_train, y_train_raw = _prepare_xy(train_df, features, label_column)
    x_val, y_val_raw = _prepare_xy(val_df, features, label_column)
    x_test, y_test_raw = _prepare_xy(test_df, features, label_column)

    duplicate_precheck = exact_duplicate_overlap_check(train_df, val_df, features)
    if not duplicate_precheck.get("is_clean", False):
        logger.warning(
            "Pre-model exact duplicate overlap detected (train/val): {} samples ({:.4f}%)",
            duplicate_precheck.get("overlapping_samples", 0),
            duplicate_precheck.get("overlap_pct", 0.0),
        )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_val = encoder.transform(y_val_raw)
    y_test = encoder.transform(y_test_raw)

    logger.info(
        "Data loaded | train={} val={} test={} classes={}",
        x_train.shape,
        x_val.shape,
        x_test.shape,
        list(encoder.classes_),
    )

    x_cv = pd.concat([x_train, x_val], axis=0, ignore_index=True)
    y_cv = np.concatenate([y_train, y_val])
    n_cv_folds = int(runtime_config.get("experiment", {}).get("n_cv_folds", 3))
    use_segment_cv = hpo_validation_mode == "segment_temporal_cv"
    if use_segment_cv and "segment_id" in train_df.columns:
        segments_cv = pd.concat(
            [train_df["segment_id"], val_df["segment_id"]], ignore_index=True
        ).values
        logger.info(
            "Segment-aware CV enabled | n_splits={} combined_samples={}",
            n_cv_folds,
            len(x_cv),
        )
    elif use_segment_cv:
        segments_cv = None
        logger.warning(
            "HPO validation_mode=segment_temporal_cv but segment_id is missing; falling back to holdout"
        )
    else:
        segments_cv = None

    init_tracking("classification")
    run_name = f"classification_lightgbm_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        logger.info("MLflow run started | run_name={}", run_name)
        mlflow.set_tags(
            {
                "task": args.task,
                "model_family": "classification_ml",
                "model_name": "lightgbm",
                "feature_profile": effective_profile,
                "feature_profile_requested": requested_profile,
                "feature_run_id": effective_run_id,
                "feature_run_selection_mode": run_selection_mode,
                "feature_run_dir": str(resolved_run_dir),
            }
        )

        mlflow.log_params(
            {
                "task": args.task,
                "dataset": args.dataset,
                "split_path": args.split_path,
                "seed": hpo_seed,
                "run_type": args.run_type,
                "feature_profile": effective_profile,
                "feature_profile_requested": requested_profile,
                "feature_run_id": effective_run_id,
                "feature_run_selection_mode": run_selection_mode,
                "feature_count": len(features),
                "label_column": label_column,
                "class_weighting_mode": "native_auto_balanced",
                "optuna_enabled": not args.no_optuna,
                "optuna_n_trials_requested": int(n_trials),
                "optuna_n_trials_executed": 0 if args.no_optuna else int(n_trials),
                "optuna_sampler": hpo_sampler,
                "optuna_pruner": str(hpo_pruner),
                "optuna_storage_enabled": bool(hpo_storage),
                "cpu_logical_cores": int(threading_plan["cpu_logical_cores"]),
                "cpu_physical_cores": int(threading_plan["cpu_physical_cores"])
                if threading_plan["cpu_physical_cores"]
                else -1,
                "thread_budget": int(threading_plan["thread_budget"]),
                "optuna_parallel_trials": int(threading_plan["optuna_parallel_trials"]),
                "threads_per_trial": int(threading_plan["threads_per_trial"]),
                "cv_strategy": "segment_aware_per_class"
                if segments_cv is not None
                else "fixed_holdout",
                "cv_n_splits": n_cv_folds if segments_cv is not None else 1,
                "duplicate_precheck_overlaps": int(
                    duplicate_precheck.get("overlapping_samples", 0)
                ),
                "duplicate_precheck_is_clean": bool(duplicate_precheck.get("is_clean", True)),
            }
        )

        study = None
        if args.no_optuna:
            best_params = _build_lightgbm_params(
                midpoint_params_from_space(lgb_space),
                seed=hpo_seed,
                n_classes=len(encoder.classes_),
            )
            best_params["n_jobs"] = int(threading_plan["thread_budget"])
            best_params["num_threads"] = int(threading_plan["thread_budget"])
            logger.info("Optuna disabled | using midpoint params")
        else:

            def objective(trial):
                trial_params = suggest_params_from_space(trial, lgb_space)
                model_params = _build_lightgbm_params(
                    trial_params,
                    seed=hpo_seed,
                    n_classes=len(encoder.classes_),
                )
                model_params["n_jobs"] = int(threading_plan["threads_per_trial"])
                model_params["num_threads"] = int(threading_plan["threads_per_trial"])

                if segments_cv is not None:
                    cv = PerClassSegmentTimeSeriesCV(n_splits=n_cv_folds, min_train_segments=1)
                    fold_scores = []
                    for fold_train_idx, fold_val_idx in cv.split(x_cv, y_cv, groups=segments_cv):
                        model = lgb.LGBMClassifier(**model_params)
                        model.fit(x_cv.iloc[fold_train_idx], y_cv[fold_train_idx])
                        fold_pred = model.predict(x_cv.iloc[fold_val_idx])
                        fold_scores.append(
                            float(
                                f1_score(
                                    y_cv[fold_val_idx],
                                    fold_pred,
                                    average="macro",
                                    zero_division=0,
                                )
                            )
                        )
                    return float(np.mean(fold_scores)) if fold_scores else 0.0

                model = lgb.LGBMClassifier(**model_params)
                model.fit(x_train, y_train)
                val_pred = model.predict(x_val)
                return float(f1_score(y_val, val_pred, average="macro", zero_division=0))

            def on_trial_complete(study_obj, trial_obj):
                logger.info(
                    "Optuna trial complete | number={} value={:.6f} best={:.6f}",
                    trial_obj.number,
                    float(trial_obj.value) if trial_obj.value is not None else float("nan"),
                    float(study_obj.best_value),
                )

            best_base_params, study = run_optuna(
                objective,
                search_space=lgb_space,
                n_trials=n_trials,
                direction=hpo_direction,
                seed=hpo_seed,
                n_jobs=int(threading_plan["optuna_parallel_trials"]),
                timeout_seconds=int(hpo_timeout) if hpo_timeout is not None else None,
                sampler_name=hpo_sampler,
                pruner_name=str(hpo_pruner) if hpo_pruner is not None else None,
                storage_url=str(hpo_storage) if hpo_storage else None,
                study_name=f"{hpo_study_prefix}_{args.dataset}_{args.split_path}_{effective_profile}",
                load_if_exists=True,
                on_trial_complete=on_trial_complete,
            )

            best_params = _build_lightgbm_params(
                best_base_params,
                seed=hpo_seed,
                n_classes=len(encoder.classes_),
            )
            best_params["n_jobs"] = int(threading_plan["thread_budget"])
            best_params["num_threads"] = int(threading_plan["thread_budget"])
            mlflow.log_metric("optuna_best_val_f1_macro", float(study.best_value))
            mlflow.log_params(
                {
                    f"best_{k}": v
                    for k, v in best_params.items()
                    if isinstance(v, (str, int, float, bool))
                }
            )
            logger.info("Optuna complete | best_val_f1_weighted={:.6f}", float(study.best_value))

            trials_artifact = (
                get_experiments_root() / "metrics" / "classification_optuna_trials.csv"
            )
            trials_artifact.parent.mkdir(parents=True, exist_ok=True)
            study.trials_dataframe().to_csv(trials_artifact, index=False)
            mlflow.log_artifact(str(trials_artifact))

        if calibration_cfg["enabled"]:
            logger.info("Training final model on train only (calibration enabled) | params={}", best_params)
            final_model, evals_result = _train_lightgbm(
                x_train,
                y_train,
                params=best_params,
                eval_sets=[(x_train, y_train)],
            )
            deployed_model = fit_probability_calibrator(
                final_model, x_val, y_val, calibration_cfg["method"]
            )
            logger.info("Probability calibration fitted | method={}", calibration_cfg["method"])
        else:
            x_train_final = pd.concat([x_train, x_val], axis=0, ignore_index=True)
            y_train_final = np.concatenate([y_train, y_val])
            logger.info("Training final model on train+val | params={}", best_params)
            final_model, evals_result = _train_lightgbm(
                x_train_final,
                y_train_final,
                params=best_params,
                eval_sets=[(x_train_final, y_train_final)],
            )
            deployed_model = final_model

        test_pred_proba = deployed_model.predict_proba(x_test)
        test_pred = np.argmax(test_pred_proba, axis=1)

        summary_metrics, pr_auc_by_class, report = compute_classification_metrics(
            y_test,
            test_pred,
            test_pred_proba,
            encoder.classes_,
        )
        accuracy = summary_metrics["accuracy"]
        f1_weighted = summary_metrics["f1_weighted"]
        f1_macro = summary_metrics["f1_macro"]
        pr_auc_weighted = summary_metrics["pr_auc_weighted"]

        metrics_payload = {
            "task": args.task,
            "dataset": args.dataset,
            "split_path": args.split_path,
            "model": "lightgbm",
            "run_name": run_name,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "feature_profile": effective_profile,
            "feature_profile_requested": requested_profile,
            "feature_run_id": effective_run_id,
            "feature_run_selection_mode": run_selection_mode,
            "feature_run_dir": str(resolved_run_dir),
            "feature_count": len(features),
            "n_classes": int(len(encoder.classes_)),
            "classes": [str(c) for c in encoder.classes_],
            "class_weighting_mode": "native_auto_balanced",
            "metrics": summary_metrics,
            "pr_auc_by_class": pr_auc_by_class,
            "best_params": best_params,
            "classification_report": report,
        }

        leakage_payload = {
            "skipped": bool(args.skip_leakage_checks),
            "reason": "--skip-leakage-checks" if args.skip_leakage_checks else None,
        }

        if not args.skip_leakage_checks:
            try:
                leakage_payload = run_leakage_report(
                    model=final_model,  # use base model (pre-calibration) for leakage diagnostics
                    X_train=x_train,
                    y_train=y_train,
                    X_val=x_val,
                    y_val=y_val,
                    feature_names=features,
                    X_eval=x_test,
                    y_eval=y_test,
                )
            except Exception as _exc:
                logger.warning("Leakage report failed (non-fatal): {}", _exc)
                leakage_payload = {
                    "skipped": True,
                    "reason": f"exception: {_exc}",
                    "leakage_flags": [],
                    "is_clean": True,
                }

        calibration_payload = None
        if calibration_cfg["enabled"]:
            calibration_payload = build_probability_calibration_payload(
                y_true=y_test,
                y_proba=test_pred_proba,
                classes=encoder.classes_,
                method=calibration_cfg["method"],
                n_calibration_samples=len(y_val),
            )

        artifact_paths = resolve_artifact_paths(
            model_name="lightgbm",
            run_name=run_name,
            artifacts_dir=args.artifacts_dir,
            metrics_path=args.metrics_path,
            leakage_report_path=args.leakage_report_path,
            model_path=args.model_path,
        )
        metrics_path = artifact_paths["metrics_path"]
        leakage_report_path = artifact_paths["leakage_report_path"]
        model_path = artifact_paths["model_path"]

        write_json_artifact(metrics_path, metrics_payload)
        write_json_artifact(leakage_report_path, leakage_payload)
        write_classification_contract_artifacts(
            paths=artifact_paths,
            args=args,
            model_name="lightgbm",
            feature_profile=effective_profile,
            resolved_run_dir=resolved_run_dir,
            seed=hpo_seed,
            run_name=run_name,
            features=features,
            label_column=label_column,
            classes=encoder.classes_,
            summary_metrics=summary_metrics,
            pr_auc_by_class=pr_auc_by_class,
            classification_report_payload=report,
            features_manifest_payload=manifest,
            calibration_payload=calibration_payload,
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": deployed_model,
                "base_model": final_model,
                "label_encoder": encoder,
                "features": features,
                "label_column": label_column,
                "manifest": manifest,
                "calibration": calibration_payload,
            },
            model_path,
        )
        save_classification_plots(
            y_true=y_test,
            y_pred=test_pred,
            y_proba=test_pred_proba,
            classes=encoder.classes_,
            model_name="LightGBM",
            paths=artifact_paths,
            feature_names=features,
            model=final_model,
        )
        if calibration_cfg["enabled"] and calibration_payload is not None:
            save_reliability_diagram(
                y_true=y_test,
                y_proba=test_pred_proba,
                classes=encoder.classes_,
                path=artifact_paths["reliability_diagram_path"],
            )
            save_confidence_histogram(
                y_proba=test_pred_proba,
                path=artifact_paths["confidence_histogram_path"],
                model_name="LightGBM",
            )

        mlflow.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_balanced_accuracy": summary_metrics["balanced_accuracy"],
                "test_f1_weighted": f1_weighted,
                "test_f1_macro": f1_macro,
                "test_precision_weighted": summary_metrics["precision_weighted"],
                "test_precision_macro": summary_metrics["precision_macro"],
                "test_recall_weighted": summary_metrics["recall_weighted"],
                "test_recall_macro": summary_metrics["recall_macro"],
            }
        )
        if not np.isnan(pr_auc_weighted):
            mlflow.log_metric("test_pr_auc_weighted", pr_auc_weighted)
        for class_name, class_pr_auc in pr_auc_by_class.items():
            mlflow.log_metric(f"test_pr_auc_class_{class_name}", class_pr_auc)
        if calibration_payload is not None:
            cal_metrics = calibration_payload.get("metrics", {})
            mlflow.log_metrics(
                {f"cal_{k}": v for k, v in cal_metrics.items() if isinstance(v, float)}
            )

        if evals_result:
            split_names = list(evals_result.keys())
            first_split_metrics = evals_result.get(split_names[0], {})
            curve_metric = next(iter(first_split_metrics.keys()), None)
            if curve_metric:
                split_labels = ["train", "test"][: len(split_names)]
                all_curves: dict[str, list] = {
                    label: evals_result[split_name].get(curve_metric, [])
                    for label, split_name in zip(split_labels, split_names)
                }
                n_steps = max(len(v) for v in all_curves.values()) if all_curves else 0
                log_interval = max(1, n_steps // 100)
                for step in range(0, n_steps, log_interval):
                    for label, vals in all_curves.items():
                        if step < len(vals):
                            mlflow.log_metric(
                                f"final_{label}_{curve_metric}", vals[step], step=step
                            )
                curves_artifact_path = (
                    get_experiments_root() / "metrics" / "classification_training_curves.json"
                )
                curves_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                curves_artifact_path.write_text(
                    json.dumps(evals_result, default=str), encoding="utf-8"
                )
                mlflow.log_artifact(str(curves_artifact_path))

        for artifact_key, artifact_path in artifact_paths.items():
            if artifact_key == "artifacts_dir":
                continue
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
        mlflow.log_dict(manifest, "features_manifest.json")
        mlflow.log_dict(threading_plan, "threading_plan.json")
        mlflow.log_dict(duplicate_precheck, "duplicate_precheck.json")
        mlflow.log_metric(
            "leakage_flag_count", float(len(leakage_payload.get("leakage_flags", [])))
        )
        mlflow.log_metric(
            "duplicate_precheck_overlaps", float(duplicate_precheck.get("overlapping_samples", 0))
        )
        mlflow.log_param("leakage_checks_enabled", not bool(args.skip_leakage_checks))

        comparison_record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "run_name": run_name,
            "task": args.task,
            "dataset": args.dataset,
            "split_path": args.split_path,
            "model": "lightgbm",
            "model_family": "classification_ml",
            "run_type": args.run_type,
            "seed": hpo_seed,
            "feature_profile": effective_profile,
            "feature_profile_requested": requested_profile,
            "feature_run_id": effective_run_id,
            "feature_run_selection_mode": run_selection_mode,
            "feature_run_dir": str(resolved_run_dir),
            "class_weighting_mode": "native_auto_balanced",
            "optuna_enabled": not args.no_optuna,
            "optuna_n_trials_requested": int(n_trials),
            "optuna_n_trials_executed": 0 if args.no_optuna else int(n_trials),
            "test_accuracy": accuracy,
            "test_balanced_accuracy": summary_metrics["balanced_accuracy"],
            "test_f1_weighted": f1_weighted,
            "test_f1_macro": f1_macro,
            "test_precision_weighted": summary_metrics["precision_weighted"],
            "test_precision_macro": summary_metrics["precision_macro"],
            "test_recall_weighted": summary_metrics["recall_weighted"],
            "test_recall_macro": summary_metrics["recall_macro"],
            "test_pr_auc_weighted": pr_auc_weighted,
            "leakage_flag_count": len(leakage_payload.get("leakage_flags", [])),
            "leakage_is_clean": bool(leakage_payload.get("is_clean", False)),
            "duplicate_precheck_overlaps": int(duplicate_precheck.get("overlapping_samples", 0)),
            "duplicate_precheck_is_clean": bool(duplicate_precheck.get("is_clean", True)),
            "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
        }
        records_path = Path(args.comparison_records_path)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(comparison_record, default=str) + "\n")
        mlflow.log_artifact(str(records_path))

        logger.success(
            "Training complete | accuracy={:.4f} f1_weighted={:.4f} pr_auc_weighted={:.4f} leakage_flags={} metrics={} model={}",
            accuracy,
            f1_weighted,
            pr_auc_weighted,
            len(leakage_payload.get("leakage_flags", [])),
            metrics_path,
            model_path,
        )
