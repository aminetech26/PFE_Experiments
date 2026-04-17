from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, label_binarize

from src.data.splitting import PerClassSegmentTimeSeriesCV
from src.evaluation.leakage_checks import run_leakage_report
from src.mlflow_setup import init_tracking
from src.training.feature_loader import load_features_for_task
from src.training.hyperparameter_optimizer import midpoint_params_from_space, run_optuna, suggest_params_from_space
from src.training.system_resources import compute_thread_budget, detect_cpu_resources


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "experiments" / "metrics" / "classification_results.json"
DEFAULT_LEAKAGE_REPORT_PATH = PROJECT_ROOT / "experiments" / "metrics" / "classification_leakage_report.json"
DEFAULT_COMPARISON_RECORDS_PATH = PROJECT_ROOT / "experiments" / "metrics" / "classification_comparison_records.jsonl"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments" / "checkpoints" / "classification" / "lightgbm_model.pkl"


def _load_model_config() -> dict:
    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _prepare_xy(df, features: list[str], label_column: str):
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        raise KeyError(f"Missing features in dataframe: {missing_features}")
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataframe")

    X = df[features]
    y = df[label_column]
    return X, y


def _compute_pr_auc_multiclass(y_true_encoded: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> float:
    if len(classes) <= 1:
        return float("nan")
    y_true_bin = label_binarize(y_true_encoded, classes=np.arange(len(classes)))
    return float(average_precision_score(y_true_bin, y_proba, average="weighted"))


def _train_lightgbm(
    X_train,
    y_train,
    params: dict,
    eval_sets: list | None = None,
) -> tuple:
    """Train a LightGBM classifier and optionally capture per-tree eval curves.

    Args:
        eval_sets: list of (X, y) tuples passed to eval_set. First entry is typically
                   (X_train, y_train), second is (X_test, y_test). If None, no curves.

    Returns:
        (model, evals_result) where evals_result is a dict of {split_name: {metric: [values]}}
        or an empty dict if eval_sets is None.
    """
    evals_result: dict = {}
    model = lgb.LGBMClassifier(**params)
    if eval_sets:
        model.fit(
            X_train,
            y_train,
            eval_set=eval_sets,
            callbacks=[lgb.record_evaluation(evals_result)],
        )
    else:
        model.fit(X_train, y_train)
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


def _resolve_threading(config: dict, args) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task B classification model (LightGBM + Optuna + MLflow)")
    parser.add_argument("--task", default="classification", choices=["classification"], help="Task name")
    parser.add_argument("--profile", default="plus_tsfresh_minimal", help="Feature profile used by featurize stage")
    parser.add_argument("--run-dir", default=None, help="Optional explicit feature run directory")
    parser.add_argument("--run-id", default=None, help="Optional exact feature run id under data/processed/features/<task>/runs/")
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH), help="Where to write metrics json")
    parser.add_argument("--leakage-report-path", default=str(DEFAULT_LEAKAGE_REPORT_PATH), help="Where to write leakage report json")
    parser.add_argument("--comparison-records-path", default=str(DEFAULT_COMPARISON_RECORDS_PATH), help="Where to append comparison record jsonl")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Where to persist trained model")
    parser.add_argument("--no-optuna", action="store_true", help="Disable Optuna and use midpoint baseline params")
    parser.add_argument("--n-trials", type=int, default=None, help="Override Optuna trial count")
    parser.add_argument("--threads", type=int, default=None, help="Override max model training threads")
    parser.add_argument("--optuna-jobs", type=int, default=None, help="Override parallel Optuna trials")
    parser.add_argument("--show-thread-plan", action="store_true", help="Print computed threading plan and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logs")
    parser.add_argument("--skip-leakage-checks", action="store_true", help="Skip leakage validation suite")
    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    config = _load_model_config()
    cls_cfg = config["classification"]
    active_model = cls_cfg.get("active_model", "lightgbm")
    if active_model != "lightgbm":
        raise NotImplementedError(f"Model '{active_model}' is configured but not yet implemented in trainer")

    lgb_space = cls_cfg["r1_models"][active_model]
    hpo_cfg = cls_cfg.get("hpo", {})
    n_trials = args.n_trials or int(hpo_cfg.get("n_trials_phase1", 50))
    hpo_direction = hpo_cfg.get("direction", "maximize")
    hpo_seed = int(config.get("experiment", {}).get("seed", 42))
    hpo_timeout = hpo_cfg.get("timeout_seconds", None)
    threading_plan = _resolve_threading(config, args)

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
    )

    features = manifest.get("final_features", [])
    label_column = manifest.get("label_column", "label")
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

    X_train, y_train_raw = _prepare_xy(train_df, features, label_column)
    X_val, y_val_raw = _prepare_xy(val_df, features, label_column)
    X_test, y_test_raw = _prepare_xy(test_df, features, label_column)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_val = encoder.transform(y_val_raw)
    y_test = encoder.transform(y_test_raw)

    logger.info(
        "Data loaded | train={} val={} test={} classes={}",
        X_train.shape,
        X_val.shape,
        X_test.shape,
        list(encoder.classes_),
    )

    # Prepare combined train+val arrays for segment-aware CV during HPO
    X_cv = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_cv = np.concatenate([y_train, y_val])
    n_cv_folds = int(config.get("experiment", {}).get("n_cv_folds", 3))
    if "segment_id" in train_df.columns:
        segments_cv = pd.concat(
            [train_df["segment_id"], val_df["segment_id"]], ignore_index=True
        ).values
        logger.info(
            "Segment-aware CV enabled | n_splits={} combined_samples={}",
            n_cv_folds,
            len(X_cv),
        )
    else:
        segments_cv = None
        logger.warning("segment_id column not found — falling back to fixed holdout for HPO")

    init_tracking("classification")
    run_name = f"classification_lightgbm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        logger.info("MLflow run started | run_name={}", run_name)
        mlflow.set_tags(
            {
                "task": args.task,
                "model_family": "lightgbm",
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
                "feature_profile": effective_profile,
                "feature_profile_requested": requested_profile,
                "feature_run_id": effective_run_id,
                "feature_run_selection_mode": run_selection_mode,
                "feature_count": len(features),
                "label_column": label_column,
                "optuna_enabled": not args.no_optuna,
                "optuna_n_trials_requested": int(n_trials),
                "optuna_n_trials_executed": 0 if args.no_optuna else int(n_trials),
                "cpu_logical_cores": int(threading_plan["cpu_logical_cores"]),
                "cpu_physical_cores": int(threading_plan["cpu_physical_cores"]) if threading_plan["cpu_physical_cores"] else -1,
                "thread_budget": int(threading_plan["thread_budget"]),
                "optuna_parallel_trials": int(threading_plan["optuna_parallel_trials"]),
                "threads_per_trial": int(threading_plan["threads_per_trial"]),
                "cv_strategy": "segment_aware_per_class" if segments_cv is not None else "fixed_holdout",
                "cv_n_splits": n_cv_folds if segments_cv is not None else 1,
            }
        )

        best_params: dict
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
                model_params = _build_lightgbm_params(trial_params, seed=hpo_seed, n_classes=len(encoder.classes_))
                model_params["n_jobs"] = int(threading_plan["threads_per_trial"])
                model_params["num_threads"] = int(threading_plan["threads_per_trial"])

                if segments_cv is not None:
                    # Segment-aware temporal CV: respects segment boundaries and class balance
                    cv = PerClassSegmentTimeSeriesCV(n_splits=n_cv_folds, min_train_segments=1)
                    fold_scores = []
                    for fold_train_idx, fold_val_idx in cv.split(X_cv, y_cv, groups=segments_cv):
                        m = lgb.LGBMClassifier(**model_params)
                        m.fit(X_cv.iloc[fold_train_idx], y_cv[fold_train_idx])
                        fold_pred = m.predict(X_cv.iloc[fold_val_idx])
                        fold_scores.append(
                            float(f1_score(y_cv[fold_val_idx], fold_pred, average="weighted", zero_division=0))
                        )
                    return float(np.mean(fold_scores)) if fold_scores else 0.0
                else:
                    # Fallback: fixed holdout (no segment_id available)
                    model = lgb.LGBMClassifier(**model_params)
                    model.fit(X_train, y_train)
                    val_pred = model.predict(X_val)
                    return float(f1_score(y_val, val_pred, average="weighted"))

            def on_trial_complete(study, trial):
                logger.info(
                    "Optuna trial complete | number={} value={:.6f} best={:.6f}",
                    trial.number,
                    float(trial.value) if trial.value is not None else float("nan"),
                    float(study.best_value),
                )

            best_base_params, study = run_optuna(
                objective,
                search_space=lgb_space,
                n_trials=n_trials,
                direction=hpo_direction,
                seed=hpo_seed,
                n_jobs=int(threading_plan["optuna_parallel_trials"]),
                timeout_seconds=int(hpo_timeout) if hpo_timeout is not None else None,
                on_trial_complete=on_trial_complete,
            )
            best_params = _build_lightgbm_params(
                best_base_params,
                seed=hpo_seed,
                n_classes=len(encoder.classes_),
            )
            best_params["n_jobs"] = int(threading_plan["thread_budget"])
            best_params["num_threads"] = int(threading_plan["thread_budget"])
            mlflow.log_metric("optuna_best_val_f1_weighted", float(study.best_value))
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items() if isinstance(v, (str, int, float, bool))})
            logger.info("Optuna complete | best_val_f1_weighted={:.6f}", float(study.best_value))

            trials_artifact = PROJECT_ROOT / "experiments" / "metrics" / "classification_optuna_trials.csv"
            trials_artifact.parent.mkdir(parents=True, exist_ok=True)
            study.trials_dataframe().to_csv(trials_artifact, index=False)
            mlflow.log_artifact(str(trials_artifact))

        X_train_final = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_train_final = np.concatenate([y_train, y_val])

        logger.info("Training final model | params={}", best_params)
        final_model, evals_result = _train_lightgbm(
            X_train_final,
            y_train_final,
            params=best_params,
            eval_sets=[(X_train_final, y_train_final), (X_test, y_test)],
        )

        test_pred = final_model.predict(X_test)
        test_pred_proba = final_model.predict_proba(X_test)

        accuracy = float(accuracy_score(y_test, test_pred))
        f1_weighted = float(f1_score(y_test, test_pred, average="weighted"))
        f1_macro = float(f1_score(y_test, test_pred, average="macro"))
        pr_auc_weighted = _compute_pr_auc_multiclass(y_test, test_pred_proba, encoder.classes_)

        report = classification_report(
            y_test,
            test_pred,
            target_names=[str(x) for x in encoder.classes_],
            output_dict=True,
            zero_division=0,
        )

        metrics_payload = {
            "task": args.task,
            "model": "lightgbm",
            "run_name": run_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "feature_profile": effective_profile,
            "feature_profile_requested": requested_profile,
            "feature_run_id": effective_run_id,
            "feature_run_selection_mode": run_selection_mode,
            "feature_run_dir": str(resolved_run_dir),
            "feature_count": len(features),
            "n_classes": int(len(encoder.classes_)),
            "classes": [str(c) for c in encoder.classes_],
            "metrics": {
                "accuracy": accuracy,
                "f1_weighted": f1_weighted,
                "f1_macro": f1_macro,
                "pr_auc_weighted": pr_auc_weighted,
            },
            "best_params": best_params,
            "classification_report": report,
        }

        leakage_payload = {
            "skipped": bool(args.skip_leakage_checks),
            "reason": "--skip-leakage-checks" if args.skip_leakage_checks else None,
        }

        if not args.skip_leakage_checks:
            leakage_payload = run_leakage_report(
                model=final_model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_names=features,
                df_train=train_df,
                df_val=val_df,
            )

        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        leakage_report_path = Path(args.leakage_report_path)
        leakage_report_path.parent.mkdir(parents=True, exist_ok=True)
        leakage_report_path.write_text(json.dumps(leakage_payload, indent=2, default=str), encoding="utf-8")

        model_path = Path(args.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": final_model,
                "label_encoder": encoder,
                "features": features,
                "label_column": label_column,
                "manifest": manifest,
            },
            model_path,
        )

        mlflow.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_f1_weighted": f1_weighted,
                "test_f1_macro": f1_macro,
            }
        )
        if not np.isnan(pr_auc_weighted):
            mlflow.log_metric("test_pr_auc_weighted", pr_auc_weighted)

        # Log per-tree training curves as step metrics and JSON artifact
        if evals_result:
            split_names = list(evals_result.keys())  # e.g. ["valid_0", "valid_1"]
            first_split_metrics = evals_result.get(split_names[0], {})
            curve_metric = next(iter(first_split_metrics.keys()), None)  # e.g. "multi_logloss"
            if curve_metric:
                split_labels = ["train", "test"][: len(split_names)]
                all_curves: dict[str, list] = {
                    label: evals_result[sn].get(curve_metric, [])
                    for label, sn in zip(split_labels, split_names)
                }
                n_steps = max(len(v) for v in all_curves.values()) if all_curves else 0
                log_interval = max(1, n_steps // 100)  # at most 100 logged points per metric
                for step in range(0, n_steps, log_interval):
                    for label, vals in all_curves.items():
                        if step < len(vals):
                            mlflow.log_metric(f"final_{label}_{curve_metric}", vals[step], step=step)
                curves_artifact_path = PROJECT_ROOT / "experiments" / "metrics" / "classification_training_curves.json"
                curves_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                curves_artifact_path.write_text(json.dumps(evals_result, default=str), encoding="utf-8")
                mlflow.log_artifact(str(curves_artifact_path))
                logger.info(
                    "Training curves logged | metric={} n_trees={} splits={}",
                    curve_metric,
                    n_steps,
                    split_labels,
                )

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(leakage_report_path))
        mlflow.log_artifact(str(model_path))
        mlflow.log_dict(manifest, "features_manifest.json")
        mlflow.log_dict(threading_plan, "threading_plan.json")
        mlflow.log_metric("leakage_flag_count", float(len(leakage_payload.get("leakage_flags", []))))
        mlflow.log_param("leakage_checks_enabled", not bool(args.skip_leakage_checks))

        comparison_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": run_name,
            "task": args.task,
            "model": "lightgbm",
            "feature_profile": effective_profile,
            "feature_profile_requested": requested_profile,
            "feature_run_id": effective_run_id,
            "feature_run_selection_mode": run_selection_mode,
            "feature_run_dir": str(resolved_run_dir),
            "optuna_enabled": not args.no_optuna,
            "optuna_n_trials_requested": int(n_trials),
            "optuna_n_trials_executed": 0 if args.no_optuna else int(n_trials),
            "test_accuracy": accuracy,
            "test_f1_weighted": f1_weighted,
            "test_f1_macro": f1_macro,
            "test_pr_auc_weighted": pr_auc_weighted,
            "leakage_flag_count": len(leakage_payload.get("leakage_flags", [])),
            "leakage_is_clean": bool(leakage_payload.get("is_clean", False)),
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


if __name__ == "__main__":
    main()
