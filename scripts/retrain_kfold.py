"""Full anomaly-detection pipeline: HPO once → K-fold retrain → aggregate.

Generalized orchestrator across all baseline models. Methodology (Option #1):
HPO once on the temporal-stratified split, then retrain the winning config
across K episode-stratified folds. Train (normal-only) is held fixed; only
val/test rotate. Same per-class mean ± std reporting for every model.

Usage:
    python -m scripts.retrain_kfold --model pc_ae    --n-folds 5
    python -m scripts.retrain_kfold --model pc_dlssm --n-folds 5
    python -m scripts.retrain_kfold --model maat     --n-folds 5
    python -m scripts.retrain_kfold --model oc_svm   --n-folds 5
    python -m scripts.retrain_kfold --model iforest  --n-folds 5
    python -m scripts.retrain_kfold --model bocd     --n-folds 5

Flags:
    --no-hpo            Skip HPO stage entirely (use config defaults).
    --n-trials N        Override HPO trial count.
    --best-params PATH  Reuse a prior HPO winning config (skips HPO).
    --smoke             1 epoch + 2 HPO trials for end-to-end validation.

Stages:
    [0] HPO            — single training run on original split, writes best_params
    [1] K-fold split   — pools val+test episodes, partitions per class into K folds
    [2] Per-fold retrain — K subprocess calls with --best-params + fold overrides
    [3] Aggregate      — mean ± std + median/IQR per metric across folds
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.modeling.common.feature_loader import load_features_for_task
from src.modeling.common.kfold_episode_split import make_episode_stratified_folds


def _subprocess_env() -> dict[str, str]:
    """Force UTF-8 stdio in spawned trainers (fixes Windows MLflow logging errors)."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ─────────────────────────────────────────────────────────────────────────────
# Model registry — per-model trainer module, HPO trigger convention, extras
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    "pc_ae": {
        "module": "src.modeling.anomaly_detection.dl.pc_ae.trainer",
        "best_params_file": "hpo_best_params.json",
        "hpo_on_args": ["--hpo"], "hpo_off_args": [],
        "default_args": [], "subdir": "pc_ae", "supports_smoke": True,
    },
    "pc_dlssm": {
        "module": "src.modeling.anomaly_detection.dl.dlssm.trainer",
        "best_params_file": "hpo_best_params.json",
        "hpo_on_args": ["--hpo"], "hpo_off_args": [],
        "default_args": [], "subdir": "dlssm", "supports_smoke": True,
    },
    "maat": {
        "module": "src.modeling.anomaly_detection.dl.maat.trainer",
        "best_params_file": "best_params.json",
        "hpo_on_args": ["--hpo"], "hpo_off_args": [],
        "default_args": [], "subdir": "maat", "supports_smoke": True,
    },
    "oc_svm": {
        "module": "src.modeling.anomaly_detection.ml.one_class_svm_model",
        "best_params_file": "hpo_best_params.json",
        "hpo_on_args": [], "hpo_off_args": ["--no-optuna"],
        "default_args": ["--kernel", "rbf"], "subdir": "one_class_svm", "supports_smoke": False,
    },
    "iforest": {
        "module": "src.modeling.anomaly_detection.ml.isolation_forest_model",
        "best_params_file": "hpo_best_params.json",
        "hpo_on_args": [], "hpo_off_args": ["--no-optuna"],
        "default_args": [], "subdir": "isolation_forest", "supports_smoke": False,
    },
    "bocd": {
        "module": "src.modeling.anomaly_detection.ml.bocd_model",
        "best_params_file": "best_params.json",
        "hpo_on_args": ["--hpo"], "hpo_off_args": [],
        "default_args": [], "subdir": "bocd", "supports_smoke": False,
    },
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anomaly detection: HPO + K-fold retrain orchestrator")
    p.add_argument("--model", choices=sorted(MODEL_REGISTRY.keys()), default="pc_ae",
                   help="Which baseline to orchestrate")
    p.add_argument("--no-hpo", action="store_true",
                   help="Skip the HPO stage entirely (use default config for K-fold retrain)")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Override HPO trial count")
    p.add_argument("--best-params", default=None,
                   help="Path to existing best-params JSON — skips HPO and uses this directly")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", default="anomaly_semisup")
    p.add_argument("--dataset", default="costa")
    p.add_argument("--split-path", default="path_a")
    p.add_argument("--profile", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-dir", default=None,
                   help="Where to write fold parquets + summary "
                        "(default: experiments/anomaly/<subdir>/full_<ts>)")
    p.add_argument("--label-col", default="label")
    p.add_argument("--group-col", default="segment_id",
                   help="Episode grouping column. Featurized parquets carry segment_id "
                        "(1:1 with episode_id in this dataset).")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: HPO 2 trials × 1 epoch, K folds × 1 epoch")
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    subdir = MODEL_REGISTRY[args.model]["subdir"]
    return PROJECT_ROOT / "experiments" / "anomaly" / subdir / f"full_{ts}"


def _model_cmd_common(args, model_spec: dict) -> list[str]:
    """Args shared by every subprocess invocation of this model's trainer."""
    cmd = [
        sys.executable, "-m", model_spec["module"],
        "--task", args.task,
        "--dataset", args.dataset,
        "--split-path", args.split_path,
        "--seed", str(args.seed),
    ]
    cmd += list(model_spec.get("default_args", []))
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.run_dir:
        cmd += ["--run-dir", args.run_dir]
    if args.run_id:
        cmd += ["--run-id", args.run_id]
    return cmd


def _run_hpo_stage(args, model_spec: dict, hpo_artifacts_dir: Path) -> Path:
    """Run HPO once. Returns path to the best-params JSON the trainer wrote."""
    hpo_artifacts_dir.mkdir(parents=True, exist_ok=True)
    cmd = _model_cmd_common(args, model_spec)
    cmd += ["--artifacts-dir", str(hpo_artifacts_dir), "--run-type", "hpo_stage"]
    cmd += list(model_spec.get("hpo_on_args", []))

    effective_n_trials = args.n_trials
    if args.smoke and effective_n_trials is None:
        effective_n_trials = 2
    if effective_n_trials is not None:
        cmd += ["--n-trials", str(effective_n_trials)]
    if args.smoke and model_spec.get("supports_smoke", False):
        cmd += ["--smoke"]

    logger.info("STAGE 0: HPO — {} ({} trials){}",
                args.model, effective_n_trials or "config-default",
                " [smoke]" if args.smoke else "")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, env=_subprocess_env())
    if proc.returncode != 0:
        raise RuntimeError(f"HPO stage failed with code {proc.returncode}")

    best_params_path = hpo_artifacts_dir / model_spec["best_params_file"]
    if not best_params_path.exists():
        raise FileNotFoundError(
            f"HPO stage completed but produced no {model_spec['best_params_file']} at "
            f"{best_params_path}. Check that the HPO run actually completed."
        )
    logger.info("STAGE 0 done. Best params at: {}", best_params_path)
    return best_params_path


def _load_best_params(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"best-params file not found: {p}")
    return p.read_text(encoding="utf-8")


def _run_fold(args, model_spec: dict, fold_idx: int,
              val_parquet: Path, test_parquet: Path,
              fold_artifacts_dir: Path, best_params_json: str | None) -> dict:
    """Spawn the model trainer for one fold and parse its global_metrics.json."""
    cmd = _model_cmd_common(args, model_spec)
    cmd += [
        "--val-parquet-override", str(val_parquet),
        "--test-parquet-override", str(test_parquet),
        "--fold-id", str(fold_idx),
        "--artifacts-dir", str(fold_artifacts_dir),
        "--run-type", "kfold",
    ]
    if best_params_json:
        cmd += ["--best-params", best_params_json]
        # For ML models (HPO default-on), also pass --no-optuna; harmless for DL.
        cmd += list(model_spec.get("hpo_off_args", []))
    if args.smoke and model_spec.get("supports_smoke", False):
        cmd += ["--smoke"]

    logger.info("Fold {}: launching {} trainer", fold_idx, args.model)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False, env=_subprocess_env())
    if proc.returncode != 0:
        raise RuntimeError(f"Fold {fold_idx} trainer exited with code {proc.returncode}")

    metrics_path = fold_artifacts_dir / "global_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Fold {fold_idx}: global_metrics.json not produced")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _aggregate(per_fold_metrics: list[dict]) -> dict:
    """mean / std / median / IQR of every numeric metric across folds."""
    if not per_fold_metrics:
        return {}
    numeric_keys: set[str] = set()
    for m in per_fold_metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                numeric_keys.add(k)
    summary: dict[str, dict] = {}
    for k in sorted(numeric_keys):
        vals = [m.get(k) for m in per_fold_metrics if isinstance(m.get(k), (int, float))]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        summary[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "n_folds": len(arr),
            "values": vals,
        }
    return summary


def _print_headline(summary: dict, model: str) -> None:
    keys = [
        ("test_class1_pr_auc_vs_normal", "cls1 (window)"),
        ("test_class2_pr_auc_vs_normal", "cls2 (window)"),
        ("test_class3_pr_auc_vs_normal", "cls3 (window)"),
        ("test_class4_pr_auc_vs_normal", "cls4 (window)"),
        ("test_episode_class1_pr_auc",   "cls1 (episode)"),
        ("test_episode_class2_pr_auc",   "cls2 (episode)"),
        ("test_episode_class3_pr_auc",   "cls3 (episode)"),
        ("test_episode_class4_pr_auc",   "cls4 (episode)"),
        ("test_pr_auc",                  "binary (window)"),
        ("test_episode_binary_pr_auc",   "binary (episode)"),
        ("test_macro_per_class_pr_auc",  "macro per-class (window)"),
        ("test_episode_macro_per_class_pr_auc", "macro per-class (episode)"),
    ]
    print("\n" + "=" * 78)
    print(f"MODEL: {model}")
    print("-" * 78)
    print(f"{'metric':<42}  {'mean':>8}  {'std':>8}  {'n':>3}")
    print("-" * 78)
    for k, label in keys:
        if k in summary:
            s = summary[k]
            print(f"{label:<42}  {s['mean']:>8.4f}  {s['std']:>8.4f}  {s['n_folds']:>3d}")
    print("=" * 78)


def main() -> None:
    args = _parse_args()
    model_spec = MODEL_REGISTRY[args.model]

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = output_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Model: {} | Output dir: {}", args.model, output_dir)

    # STAGE 0 — HPO
    resolved_best_params_path: Path | None = None
    if args.best_params:
        resolved_best_params_path = Path(args.best_params)
        if not resolved_best_params_path.exists():
            raise FileNotFoundError(f"--best-params not found: {resolved_best_params_path}")
        logger.info("STAGE 0: SKIPPED (using --best-params from {})", resolved_best_params_path)
    elif args.no_hpo:
        logger.info("STAGE 0: SKIPPED (--no-hpo)")
    else:
        hpo_artifacts_dir = output_dir / "hpo_stage"
        resolved_best_params_path = _run_hpo_stage(args, model_spec, hpo_artifacts_dir)

    # STAGE 1
    logger.info("STAGE 1: K-fold split (n_folds={})", args.n_folds)
    logger.info("Loading featurized data (task={} dataset={})", args.task, args.dataset)
    _, val_df, test_df, manifest, resolved_run_dir = load_features_for_task(
        task=args.task, profile=args.profile, run_dir=args.run_dir,
        run_id=args.run_id, dataset=args.dataset, split_path=args.split_path,
    )
    logger.info("Featurized val rows={}, test rows={}, features={}",
                len(val_df), len(test_df), len(manifest.get("final_features", [])))

    folds = make_episode_stratified_folds(
        val_df=val_df, test_df=test_df,
        n_folds=args.n_folds, seed=args.seed,
        label_col=args.label_col, group_col=args.group_col,
    )

    # STAGE 2
    logger.info("STAGE 2: Per-fold retrain")
    best_params_json = _load_best_params(resolved_best_params_path)
    per_fold_metrics: list[dict] = []
    fold_records: list[dict] = []
    for fold_idx, (val_fold_df, test_fold_df, fa) in enumerate(folds):
        fold_parquet_val = fold_dir / f"fold_{fold_idx}_val.parquet"
        fold_parquet_test = fold_dir / f"fold_{fold_idx}_test.parquet"
        val_fold_df.to_parquet(fold_parquet_val, index=False)
        test_fold_df.to_parquet(fold_parquet_test, index=False)
        fold_artifacts_dir = output_dir / f"fold_{fold_idx}"
        fold_artifacts_dir.mkdir(parents=True, exist_ok=True)

        try:
            metrics = _run_fold(args, model_spec, fold_idx, fold_parquet_val,
                                fold_parquet_test, fold_artifacts_dir, best_params_json)
            per_fold_metrics.append(metrics)
            fold_records.append({
                "fold_idx": fold_idx,
                "n_val_episodes": len(fa.val_episode_ids),
                "n_test_episodes": len(fa.test_episode_ids),
                "test_pr_auc": metrics.get("test_pr_auc"),
                "test_cls2_pr_auc": metrics.get("test_class2_pr_auc_vs_normal"),
                "test_episode_cls2_pr_auc": metrics.get("test_episode_class2_pr_auc"),
                "test_macro_pc_pr_auc": metrics.get("test_macro_per_class_pr_auc"),
            })
            logger.info("Fold {} done: test_pr_auc={:.4f}  cls2={:.4f}",
                        fold_idx,
                        metrics.get("test_pr_auc") or 0.0,
                        metrics.get("test_class2_pr_auc_vs_normal") or 0.0)
        except Exception as exc:
            logger.error("Fold {} FAILED: {}", fold_idx, exc)
            fold_records.append({"fold_idx": fold_idx, "error": str(exc)})

    if not per_fold_metrics:
        logger.error("No folds completed successfully — aborting summary.")
        return

    # STAGE 3
    logger.info("STAGE 3: Aggregate")
    summary = _aggregate(per_fold_metrics)
    summary_path = output_dir / "kfold_summary.json"
    summary_path.write_text(json.dumps({
        "model": args.model,
        "n_folds": args.n_folds,
        "n_folds_completed": len(per_fold_metrics),
        "seed": args.seed,
        "hpo_stage_ran": resolved_best_params_path is not None and not args.best_params and not args.no_hpo,
        "best_params_source": str(resolved_best_params_path) if resolved_best_params_path else None,
        "feature_run_dir": str(resolved_run_dir),
        "summary": summary,
        "fold_records": fold_records,
    }, indent=2, default=float), encoding="utf-8")
    logger.info("Summary saved: {}", summary_path)

    pd.DataFrame(fold_records).to_csv(output_dir / "fold_records.csv", index=False)
    _print_headline(summary, args.model)

    # ── DagsHub MLflow aggregate run ─────────────────────────────────────────
    try:
        import mlflow
        from src.mlflow_setup import init_tracking

        init_tracking("anomaly")
        run_name = f"kfold_summary_{args.model}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "task": args.task, "dataset": args.dataset, "split_path": args.split_path,
                "model": args.model, "run_type": "kfold_summary",
                "n_folds": str(args.n_folds), "n_folds_completed": str(len(per_fold_metrics)),
                "methodology": "hpo_once_then_kfold_retrain",
            })
            mlflow.log_params({
                "model": args.model, "n_folds": args.n_folds, "seed": args.seed,
                "task": args.task, "dataset": args.dataset,
                "hpo_stage_ran": bool(resolved_best_params_path is not None
                                      and not args.best_params and not args.no_hpo),
                "best_params_source": str(resolved_best_params_path) if resolved_best_params_path else None,
            })
            # Flatten mean/std per metric to MLflow scalar metrics
            for metric_name, s in summary.items():
                mlflow.log_metric(f"{metric_name}__mean", float(s["mean"]))
                mlflow.log_metric(f"{metric_name}__std", float(s["std"]))
                mlflow.log_metric(f"{metric_name}__median", float(s["median"]))
            # Log summary JSON + per-fold CSV as artifacts
            for p in (summary_path, output_dir / "fold_records.csv"):
                if p.exists():
                    mlflow.log_artifact(str(p))
            # If HPO ran, attach the best-params file too
            if resolved_best_params_path and resolved_best_params_path.exists():
                mlflow.log_artifact(str(resolved_best_params_path))
        logger.info("DagsHub MLflow aggregate run logged: {}", run_name)
    except Exception as exc:
        logger.warning("Aggregate MLflow logging failed (non-fatal): {}", exc)


if __name__ == "__main__":
    main()
