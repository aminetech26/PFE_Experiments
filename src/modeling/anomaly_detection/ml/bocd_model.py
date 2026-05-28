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
from scipy.special import gammaln
from scipy.special import logsumexp as scipy_logsumexp
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
    _prepare_xy,
    _save_pr_curve,
    _save_score_histogram,
    _set_ocsvm_threshold_config as _set_shared_threshold_config,
    _save_score_timeline,
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
from src.modeling.common.hyperparameter_optimizer import run_optuna, suggest_params_from_space
from src.utils.paths import get_experiments_root

PROJECT_ROOT = Path(__file__).resolve().parents[4]
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Hazard probability for a single step: h = 1 / hazard_lambda
# Under the constant hazard model, the prior on run length r is geometric:
# P(changepoint at step t) = h * prod_{s<t}(1-h)


class BOCDDetector:
    """Joint multivariate BOCD with Normal-Inverse-Gamma conjugate per feature.

    Features are assumed independent — the joint predictive is the product of
    univariate Student-t predictives. Run-length distribution is truncated at
    `max_run_length` to keep memory O(max_rl * F) instead of O(n * F).
    """

    def __init__(
        self,
        hazard_lambda: float,
        max_run_length: int = 200,
        kappa0: float = 1.0,
        alpha0: float = 2.0,
    ) -> None:
        self.hazard_lambda = float(hazard_lambda)
        self.max_run_length = int(max_run_length)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)

        # Set by fit()
        self.mu0: np.ndarray | None = None
        self.beta0: np.ndarray | None = None
        self.n_features: int = 0

    def fit(self, X_scaled: np.ndarray) -> None:
        self.n_features = X_scaled.shape[1]
        self.mu0 = X_scaled.mean(axis=0)
        self.beta0 = self.alpha0 * np.var(X_scaled, axis=0, ddof=1).clip(min=1e-8)

    def score_sequence(self, X: np.ndarray) -> np.ndarray:
        """Run BOCD and return per-step negative log mixture predictive.

        Score at step t = -log sum_r P(run_length=r) * p(x_t | r).
        Higher = more surprising observation = more likely anomalous.
        """
        assert self.mu0 is not None, "Call fit() before score_sequence()"
        n, F = X.shape
        max_rl = self.max_run_length
        h = 1.0 / self.hazard_lambda

        # Run-length distribution R[r] = P(run length == r at current step).
        # The first step is always scored from the global train prior (R[0]=1),
        # not from local history. This is expected BOCD behavior, but can create
        # a small spike at group reset boundaries in the score timeline.
        R = np.zeros(max_rl + 1)
        R[0] = 1.0

        # NIG sufficient stats per run-length hypothesis
        mu = np.tile(self.mu0, (max_rl + 1, 1))    # [max_rl+1, F]
        kappa = np.full(max_rl + 1, self.kappa0)    # [max_rl+1]
        alpha = np.full(max_rl + 1, self.alpha0)    # [max_rl+1]
        beta = np.tile(self.beta0, (max_rl + 1, 1)) # [max_rl+1, F]

        scores = np.zeros(n)

        for t in range(n):
            x = X[t]  # [F]

            # Predictive: Student-t with df=2*alpha, loc=mu, scale^2=beta*(kappa+1)/(alpha*kappa)
            denom = (alpha[:, None] * kappa[:, None]).clip(min=1e-30)
            scale2 = (beta * (kappa[:, None] + 1.0) / denom).clip(min=1e-16)  # [max_rl+1, F]

            # Vectorized Student-t log-pdf via gammaln (avoids per-call scipy overhead)
            # half_df = alpha  (since df/2 = alpha)
            half_df = alpha[:, None]  # [max_rl+1, 1] for broadcasting
            log_norm = (
                gammaln(half_df + 0.5)
                - gammaln(half_df)
                - 0.5 * np.log(2.0 * half_df * np.pi * scale2)
            )  # [max_rl+1, F]
            z2 = (x[None, :] - mu) ** 2 / (2.0 * half_df * scale2)  # [max_rl+1, F]
            log_pred_joint = (log_norm - (half_df + 0.5) * np.log1p(z2)).sum(axis=1)  # [max_rl+1]

            # --- Anomaly score: -log mixture predictive BEFORE R update ---
            # = -logsumexp(log R[r] + log p(x | r))
            # R[r]=0 entries contribute -inf to logsumexp and are ignored automatically.
            with np.errstate(divide="ignore"):
                log_R = np.log(R)
            scores[t] = float(-scipy_logsumexp(log_R + log_pred_joint))

            # --- BOCD R update ---
            # Use shifted pred_prob for numerics; normalization cancels the shift.
            pred_prob = np.exp(log_pred_joint - log_pred_joint.max())
            growth_prob = R * pred_prob * (1.0 - h)  # [max_rl+1]
            cp_prob = float((R * pred_prob * h).sum())

            new_R = np.zeros(max_rl + 1)
            new_R[1:] = growth_prob[:max_rl]
            new_R[0] = cp_prob

            Z = new_R.sum()
            if Z > 0:
                new_R /= Z

            # --- NIG posterior update (closed-form) ---
            kappa_new = kappa + 1.0
            mu_new = (kappa[:, None] * mu + x[None, :]) / kappa_new[:, None]
            alpha_new = alpha + 0.5
            beta_new = beta + 0.5 * kappa[:, None] * (x[None, :] - mu) ** 2 / kappa_new[:, None]

            # Shift: index r+1 gets posterior from r; index 0 resets to prior
            new_mu = np.tile(self.mu0, (max_rl + 1, 1))
            new_kappa = np.full(max_rl + 1, self.kappa0)
            new_alpha = np.full(max_rl + 1, self.alpha0)
            new_beta = np.tile(self.beta0, (max_rl + 1, 1))

            new_mu[1:] = mu_new[:max_rl]
            new_kappa[1:] = kappa_new[:max_rl]
            new_alpha[1:] = alpha_new[:max_rl]
            new_beta[1:] = beta_new[:max_rl]

            R = new_R
            mu = new_mu
            kappa = new_kappa
            alpha = new_alpha
            beta = new_beta

        return scores

    def score(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        group_col: str | None,
    ) -> np.ndarray:
        """Group-aware scoring: reset BOCD state at episode/segment boundaries."""
        scores_out = np.zeros(len(df))

        if group_col and group_col in df.columns:
            for _gid, grp in df.groupby(group_col, sort=False):
                idx = grp.index
                X_grp = grp[feature_cols].to_numpy(dtype=np.float64)
                grp_scores = self.score_sequence(X_grp)
                scores_out[df.index.get_indexer(idx)] = grp_scores
        else:
            X = df[feature_cols].to_numpy(dtype=np.float64)
            scores_out = self.score_sequence(X)

        return scores_out


def _resolve_group_col(df: pd.DataFrame, split_path: str | None = None) -> str | None:
    # Path B is a continuous daily stream — prefer day-level reset to avoid fragmenting
    # the time series. Path A is episode-based, so episode_id is the natural boundary.
    if split_path and "path_b" in str(split_path).lower():
        priority = ("operating_day_id", "segment_id", "episode_id")
    else:
        priority = ("episode_id", "segment_id", "operating_day_id")
    for col in priority:
        if col in df.columns:
            return col
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BOCD anomaly detection baseline")
    p.add_argument("--task", default="anomaly_semisup")
    p.add_argument("--dataset", default="costa")
    p.add_argument("--split-path", default="path_a")
    p.add_argument("--profile", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--hpo", action="store_true")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--comparison-records-path", default=str(_default_comparison_records_path()))
    p.add_argument("--run-type", default="baseline", help="baseline | ablation | final")
    p.add_argument("--best-params", default=None,
                   help="JSON string of best params to apply directly (skips HPO)")
    add_fold_override_args(p)
    return p.parse_args()


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_bocd(config: dict | None = None) -> None:
    args = _parse_args()
    if config is None:
        config = _load_config()

    bocd_cfg = config["anomaly_detection"]["ml"]["models"]["bocd"]
    # Shared threshold calibration (GPD by default)
    from src.modeling.common.threshold_calibration import (
        load_threshold_config as _load_thr_cfg,
        threshold_policy_str as _thr_policy_str,
    )
    _thr_cfg = _load_thr_cfg(config, bocd_cfg)
    _set_shared_threshold_config(_thr_cfg)
    _bocd_threshold_policy = _thr_policy_str(
        {"strategy": _thr_cfg["strategy"], "percentile": _thr_cfg["percentile"],
         "pot_quantile": _thr_cfg["pot_quantile"], "target_tail_prob": _thr_cfg["target_tail_prob"]}
    )
    hpo_cfg = config["anomaly_detection"]["ml"]["hpo"]

    max_run_length: int = int(bocd_cfg.get("max_run_length", 500))
    kappa0: float = float(bocd_cfg.get("kappa0", 1.0))
    alpha0: float = float(bocd_cfg.get("alpha0", 2.0))
    search_space: dict = dict(bocd_cfg.get("search_space", {}))
    n_trials: int = args.n_trials or int(hpo_cfg.get("n_trials_phase1", 20))
    seed: int = args.seed

    logger.info(
        "Loading features | task={} dataset={} split_path={} profile={}",
        args.task, args.dataset, args.split_path, args.profile,
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

    non_normal_in_train = int((y_train != 0).sum())
    if non_normal_in_train:
        logger.warning(
            "Train contains {} non-normal rows — expected all-normal for semisup.",
            non_normal_in_train,
        )

    logger.info(
        "Rows — train: {:,}  val: {:,} (faults: {:,})  test: {:,} (faults: {:,})",
        len(x_train), len(x_val), int(y_val_bin.sum()), len(x_test), int(y_test_bin.sum()),
    )
    logger.info("Features: {} | max_run_length: {}", len(features), max_run_length)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    group_col = _resolve_group_col(val_df, split_path=args.split_path)
    logger.info("Group column for sequential scoring: {}", group_col)
    val_group_ids = val_df[group_col].to_numpy() if group_col and group_col in val_df.columns else None
    test_group_ids = test_df[group_col].to_numpy() if group_col and group_col in test_df.columns else None

    def _make_detector(hazard_lambda: float) -> BOCDDetector:
        det = BOCDDetector(
            hazard_lambda=hazard_lambda,
            max_run_length=max_run_length,
            kappa0=kappa0,
            alpha0=alpha0,
        )
        det.fit(x_train_scaled)
        return det

    def _objective_clean(trial: optuna.Trial) -> float:
        trial_params = suggest_params_from_space(trial, search_space)
        det = _make_detector(float(trial_params["hazard_lambda"]))
        val_df_scaled = val_df.copy()
        val_df_scaled[features] = scaler.transform(x_val)
        scores = det.score(val_df_scaled, features, label_col, group_col)
        macro = compute_macro_per_class_pr_auc(labels=y_val, scores=scores)
        val_macro = macro.get("macro_per_class_pr_auc")
        return float(val_macro) if isinstance(val_macro, (int, float, np.integer, np.floating)) else 0.0

    if args.best_params:
        best_params = json.loads(args.best_params)
        logger.info("Applying best_params from CLI (skipping HPO): {}", best_params)
        study = None
    elif args.hpo and search_space:
        logger.info("Running Optuna HPO: {} trials", n_trials)
        best_params, study = run_optuna(
            _objective_clean,
            search_space=search_space,
            n_trials=n_trials,
            direction="maximize",
            seed=seed,
            sampler_name=str(hpo_cfg.get("sampler", "tpe")),
            pruner_name=str(hpo_cfg.get("pruner", "none")),
            study_name=f"{hpo_cfg.get('study_name_prefix', 'anomaly_ml')}_bocd",
            storage_url=hpo_cfg.get("storage_url") or None,
            load_if_exists=True,
        )
        logger.info(
            "Best params: {} | Best val objective (macro per-class PR-AUC): {:.4f}",
            best_params,
            study.best_value,
        )
    else:
        # Default hazard_lambda: geometric midpoint of log-scale search range
        if search_space and "hazard_lambda" in search_space:
            spec = search_space["hazard_lambda"]
            best_params = {"hazard_lambda": float(np.sqrt(float(spec["low"]) * float(spec["high"])))}
        else:
            best_params = {"hazard_lambda": float(bocd_cfg.get("hazard_lambda", 300.0))}
        logger.info("HPO skipped — using default params: {}", best_params)
        study = None

    logger.info("Fitting final detector with params: {}", best_params)
    t0 = time.perf_counter()
    final_detector = _make_detector(float(best_params["hazard_lambda"]))
    fit_time = time.perf_counter() - t0

    # Score all splits (detector uses pre-fit scaler via df copy)
    train_df_scaled = train_df.copy()
    train_df_scaled[features] = x_train_scaled
    val_df_scaled = val_df.copy()
    val_df_scaled[features] = scaler.transform(x_val)
    test_df_scaled = test_df.copy()
    test_df_scaled[features] = scaler.transform(x_test)

    t1 = time.perf_counter()
    train_scores = final_detector.score(train_df_scaled, features, label_col, group_col)
    val_scores = final_detector.score(val_df_scaled, features, label_col, group_col)
    test_scores = final_detector.score(test_df_scaled, features, label_col, group_col)
    score_time = time.perf_counter() - t1

    threshold, val_f1, val_prec, val_rec = _calibrate_threshold(val_scores, y_val_bin)
    logger.info(
        "Val — threshold={:.4f} F1={:.4f} Prec={:.4f} Rec={:.4f}",
        threshold, val_f1, val_prec, val_rec,
    )

    val_pr_auc = float(average_precision_score(y_val_bin, val_scores))
    val_roc_auc = float(roc_auc_score(y_val_bin, val_scores))
    test_pr_auc = float(average_precision_score(y_test_bin, test_scores))
    test_roc_auc = float(roc_auc_score(y_test_bin, test_scores))

    val_preds = (val_scores >= threshold).astype(int)
    test_preds = (test_scores >= threshold).astype(int)
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
        "threshold_policy": _bocd_threshold_policy,
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
        "n_train": int(len(x_train)),
        "fit_time_s": round(fit_time, 2),
        "score_time_s": round(score_time, 2),
    }

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        artifacts_dir = get_experiments_root() / "anomaly" / "bocd" / f"bocd_{ts}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    detector_path = artifacts_dir / "detector.joblib"
    scaler_path = artifacts_dir / "scaler.joblib"
    global_metrics_path = artifacts_dir / "global_metrics.json"
    per_class_metrics_path = artifacts_dir / "per_class_metrics.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    deployment_manifest_path = artifacts_dir / "deployment_manifest.json"
    features_manifest_path = artifacts_dir / "features_manifest.json"
    best_params_path = artifacts_dir / "best_params.json"
    pr_curve_path = artifacts_dir / "pr_curve.png"
    histogram_path = artifacts_dir / "score_histogram.png"
    timeline_path = artifacts_dir / "score_timeline.png"

    best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    joblib.dump(final_detector, detector_path)
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
    run_name = f"anomaly_bocd_{ts}"
    run_manifest = build_run_manifest(
        task=args.task,
        model="bocd",
        model_family="anomaly_ml",
        dataset=args.dataset,
        split_path=args.split_path,
        feature_profile=str(args.profile),
        feature_run_dir=str(resolved_run_dir),
        seed=seed,
        run_type=args.run_type,
        extras={"run_name": run_name, "group_col": group_col},
    )
    score_calibration_path = artifacts_dir / "score_calibration.json"
    deployment_manifest = build_deployment_manifest(
        task=args.task,
        model="bocd",
        model_family="anomaly_ml",
        model_artifact=detector_path.name,
        scaler_artifact=scaler_path.name,
        feature_names=features,
        label_column=label_col,
        threshold=threshold,
        score_direction="higher_is_more_anomalous",
        classes=[str(c) for c in sorted(np.unique(y_test).tolist())],
        extras={
            "group_column": str(group_col),
            "threshold_policy": _bocd_threshold_policy,
            "score_calibration_artifact": score_calibration_path.name,
        },
    )
    write_json(
        score_calibration_path,
        build_score_calibration_payload(
            threshold=threshold,
            threshold_policy=_bocd_threshold_policy,
            threshold_quantile=None,
            candidate_per_true_class_thresholds=candidate_per_true_class_thresholds,
        ),
    )
    write_json(global_metrics_path, metrics)
    write_json(per_class_metrics_path, per_class_metrics)
    write_json(run_manifest_path, run_manifest)
    write_json(deployment_manifest_path, deployment_manifest)
    write_json(features_manifest_path, manifest)
    _save_pr_curve(val_scores, y_val_bin, test_scores, y_test_bin, pr_curve_path, model_name="BOCD")
    _save_score_histogram(train_scores, val_scores, y_val_bin, threshold, histogram_path, model_name="BOCD")
    _save_score_timeline(test_scores, y_test_bin, threshold, timeline_path, model_name="BOCD")

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
                    "model": "bocd",
                }
            )
            mlflow.log_params(
                {
                    **best_params,
                    "max_run_length": max_run_length,
                    "kappa0": kappa0,
                    "alpha0": alpha0,
                    "group_col": str(group_col),
                    "n_features": len(features),
                    "scaling": "standard",
                    "hpo_enabled": args.hpo,
                    "n_optuna_trials": n_trials if args.hpo else 0,
                    "seed": seed,
                    "score_time_s": round(score_time, 2),
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
                detector_path,
                scaler_path,
                best_params_path,
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
                "model": "bocd",
                "model_family": "anomaly_ml",
                "run_type": args.run_type,
                "seed": seed,
                "feature_profile": str(args.profile),
                "feature_run_dir": str(resolved_run_dir),
                "hpo_enabled": args.hpo,
                "optuna_n_trials_requested": n_trials if args.hpo else 0,
                "best_params": best_params,
                "n_features": len(features),
                "n_train": int(len(x_train)),
                "fit_time_s": round(fit_time, 2),
                "score_time_s": round(score_time, 2),
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
        logger.warning("MLflow logging failed (non-fatal): {}", exc)


if __name__ == "__main__":
    run_bocd()
