"""Full PC-AE pipeline: HPO once → K-fold retrain → aggregate.

Single-command orchestration of Option #1 (HPO-then-K-fold-retrain).

Default usage (one command, HPO + 5-fold retrain):
    python -m scripts.retrain_kfold_pc_ae --n-folds 5 --seed 42

Skip HPO (use defaults from config):
    python -m scripts.retrain_kfold_pc_ae --no-hpo --n-folds 5

Reuse an earlier HPO winning config:
    python -m scripts.retrain_kfold_pc_ae \
        --best-params experiments/anomaly/pc_ae/<ts>/hpo_best_params.json \
        --n-folds 5

Stages:
    [0] HPO — runs PC-AE trainer with --hpo on the original temporal-stratified
        split (single train/val pair). Writes hpo_best_params.json into the
        HPO sub-directory. Skipped if --no-hpo or --best-params is provided.
    [1] K-fold split — pools val+test episodes, partitions per class into K
        stratified folds (test = 1/K, val = (K-1)/K). Train is held fixed.
    [2] Per-fold retrain — spawns the PC-AE trainer K times in subprocess with
        --best-params (from stage 0 or --best-params), --val-parquet-override,
        --test-parquet-override, --fold-id.
    [3] Aggregate — mean ± std (and median/IQR) of every numeric metric across
        folds. Headline per-class table printed to stdout.

Artifacts (under experiments/anomaly/pc_ae/full_<ts>/):
    hpo_stage/                 — full HPO trainer artifacts incl. hpo_best_params.json
    folds/fold_{k}_*.parquet   — per-fold val/test slices used by each fold
    fold_{k}/                  — per-fold trainer artifacts (full set)
    kfold_summary.json         — aggregate metrics
    fold_records.csv           — one row per fold, headline columns
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.modeling.common.feature_loader import load_features_for_task
from src.modeling.common.kfold_episode_split import make_episode_stratified_folds


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full PC-AE pipeline: HPO + K-fold retrain")
    # HPO stage controls
    p.add_argument("--no-hpo", action="store_true",
                   help="Skip the HPO stage entirely (use default config for K-fold retrain)")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Override HPO trial count (default: from model_config.yaml)")
    p.add_argument("--best-params", default=None,
                   help="Path to an existing hpo_best_params.json — skips HPO stage and uses this directly")
    # K-fold controls
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", default="anomaly_semisup")
    p.add_argument("--dataset", default="costa")
    p.add_argument("--split-path", default="path_a")
    p.add_argument("--profile", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-dir", default=None,
                   help="Where to write fold parquets + summary (default: experiments/anomaly/pc_ae/full_<ts>)")
    p.add_argument("--label-col", default="label")
    p.add_argument("--group-col", default="segment_id",
                   help="Episode grouping column. Featurized parquets carry segment_id (1:1 with episode_id in this dataset).")
    p.add_argument("--smoke", action="store_true",
                   help="Pass --smoke to HPO + every fold (1 epoch each). HPO will still run with reduced trial count.")
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "experiments" / "anomaly" / "pc_ae" / f"full_{ts}"


def _run_hpo_stage(args, hpo_artifacts_dir: Path) -> Path:
    """Run the PC-AE trainer with --hpo and return path to the produced hpo_best_params.json."""
    hpo_artifacts_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.modeling.anomaly_detection.dl.pc_ae.trainer",
        "--task", args.task,
        "--dataset", args.dataset,
        "--split-path", args.split_path,
        "--seed", str(args.seed),
        "--artifacts-dir", str(hpo_artifacts_dir),
        "--run-type", "hpo_stage",
        "--hpo",
    ]
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.run_dir:
        cmd += ["--run-dir", args.run_dir]
    if args.run_id:
        cmd += ["--run-id", args.run_id]
    # Smoke override: 2 trials max, 1 epoch each
    effective_n_trials = args.n_trials
    if args.smoke and effective_n_trials is None:
        effective_n_trials = 2
    if effective_n_trials is not None:
        cmd += ["--n-trials", str(effective_n_trials)]
    if args.smoke:
        cmd += ["--smoke"]

    logger.info("STAGE 0: HPO — launching trainer ({} trials){}",
                effective_n_trials or "config-default", " [smoke]" if args.smoke else "")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"HPO stage failed with code {proc.returncode}")

    best_params_path = hpo_artifacts_dir / "hpo_best_params.json"
    if not best_params_path.exists():
        raise FileNotFoundError(
            f"HPO stage completed but produced no hpo_best_params.json at {best_params_path}. "
            "Check that the HPO run actually completed at least one trial."
        )
    logger.info("STAGE 0 done. Best params at: {}", best_params_path)
    return best_params_path


def _load_best_params(path: str | None) -> str | None:
    """Return the best-params JSON string, or None."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--best-params file not found: {p}")
    return p.read_text(encoding="utf-8")


def _run_fold(args, fold_idx: int, val_parquet: Path, test_parquet: Path,
              fold_artifacts_dir: Path, best_params_json: str | None) -> dict:
    """Spawn the PC-AE trainer for one fold and parse its global_metrics.json."""
    cmd = [
        sys.executable, "-m", "src.modeling.anomaly_detection.dl.pc_ae.trainer",
        "--task", args.task,
        "--dataset", args.dataset,
        "--split-path", args.split_path,
        "--seed", str(args.seed),
        "--val-parquet-override", str(val_parquet),
        "--test-parquet-override", str(test_parquet),
        "--fold-id", str(fold_idx),
        "--artifacts-dir", str(fold_artifacts_dir),
        "--run-type", "kfold",
    ]
    if args.profile:
        cmd += ["--profile", args.profile]
    if args.run_dir:
        cmd += ["--run-dir", args.run_dir]
    if args.run_id:
        cmd += ["--run-id", args.run_id]
    if best_params_json:
        cmd += ["--best-params", best_params_json]
    if args.smoke:
        cmd += ["--smoke"]

    logger.info("Fold {}: launching trainer", fold_idx)
    logger.debug("CMD: {}", " ".join(cmd[:6]) + " ...")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Fold {fold_idx} trainer exited with code {proc.returncode}")

    metrics_path = fold_artifacts_dir / "global_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Fold {fold_idx}: global_metrics.json not produced")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _aggregate(per_fold_metrics: list[dict]) -> dict:
    """mean ± std (and median, IQR) of every numeric metric across folds."""
    if not per_fold_metrics:
        return {}
    # union of numeric keys
    numeric_keys = set()
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


def _print_headline(summary: dict) -> None:
    """Headline per-class table: PR-AUC mean ± std across folds."""
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
    print("\n" + "=" * 72)
    print(f"{'metric':<42}  {'mean':>8}  {'std':>8}  {'n':>3}")
    print("-" * 72)
    for k, label in keys:
        if k in summary:
            s = summary[k]
            print(f"{label:<42}  {s['mean']:>8.4f}  {s['std']:>8.4f}  {s['n_folds']:>3d}")
    print("=" * 72)


def main() -> None:
    args = _parse_args()

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = output_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Output dir: {}", output_dir)

    # STAGE 0 — HPO (optional)
    resolved_best_params_path: Path | None = None
    if args.best_params:
        resolved_best_params_path = Path(args.best_params)
        if not resolved_best_params_path.exists():
            raise FileNotFoundError(f"--best-params not found: {resolved_best_params_path}")
        logger.info("STAGE 0: SKIPPED (using --best-params from {})", resolved_best_params_path)
    elif args.no_hpo:
        logger.info("STAGE 0: SKIPPED (--no-hpo — folds will use default config from model_config.yaml)")
    else:
        hpo_artifacts_dir = output_dir / "hpo_stage"
        resolved_best_params_path = _run_hpo_stage(args, hpo_artifacts_dir)

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

    # STAGE 2 — Per-fold retrain
    logger.info("STAGE 2: Per-fold retrain")
    best_params_json = _load_best_params(str(resolved_best_params_path) if resolved_best_params_path else None)
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
            metrics = _run_fold(args, fold_idx, fold_parquet_val, fold_parquet_test,
                                fold_artifacts_dir, best_params_json)
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

    # STAGE 3 — Aggregate
    logger.info("STAGE 3: Aggregate")
    summary = _aggregate(per_fold_metrics)
    summary_path = output_dir / "kfold_summary.json"
    summary_path.write_text(json.dumps({
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

    # CSV per-fold for spreadsheet/thesis-table use
    pd.DataFrame(fold_records).to_csv(output_dir / "fold_records.csv", index=False)

    _print_headline(summary)


if __name__ == "__main__":
    main()
