"""Task B classification: HPO once -> grouped K-fold retrain -> aggregate.

Variance-estimation counterpart of ``scripts/retrain_kfold.py`` (Task A), built
for *consistency* with the anomaly-detection CV: same "HPO once on the original
holdout, then K-fold retrain with fixed params" methodology and the same
mean +/- std / median / IQR summary shape.

KEY DIFFERENCE vs Task A (and why):
    Task A is semi-supervised -> the normal-only TRAIN is held FIXED and only
    val/test rotate. Task B is SUPERVISED -> fault episodes must appear in the
    training folds, so the train set *must* rotate. This is therefore a proper
    ``StratifiedGroupKFold`` over the pooled data, grouped by episode
    (``segment_id``, which is 1:1 with ``episode_id`` in these splits) so no
    fault event is ever split across train and test of a fold. Plain
    ``StratifiedKFold`` would leak episodes and is intentionally NOT used.

CAVEATS to state in the thesis:
    - K-fold relaxes strict temporal ordering. Acceptable for fault-TYPE
      classification (a fault signature is being labelled, not forecast), but it
      is a deviation from the temporal-holdout used elsewhere -- report it.
    - Minority fault classes have few episodes (Costa: ~15-23); 5 folds give
      ~3-5 test episodes per minority class per fold, so per-class numbers carry
      wide error bars. Per-fold per-class episode support is reported alongside
      the metrics so this is visible, not hidden.

Usage:
    python -m scripts.retrain_kfold_classification --dataset costa --n-folds 5
    python -m scripts.retrain_kfold_classification --no-hpo          # midpoint params
    python -m scripts.retrain_kfold_classification --best-params params.json

Stages:
    [0] HPO        -- one Optuna run on the ORIGINAL train/val holdout (f1_macro),
                      mirroring the production trainer; or --no-hpo / --best-params.
    [1] Pool       -- concat train+val+test; X / y / groups(segment_id).
    [2] K-fold     -- StratifiedGroupKFold(segment_id); per fold refit LightGBM
                      (fixed params) + scoring via compute_classification_metrics.
    [3] Aggregate  -- mean / std / median / IQR per metric; per-class + support.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.modeling.classification.ml.common import compute_classification_metrics
from src.modeling.common.feature_loader import load_features_for_task
from src.modeling.common.hyperparameter_optimizer import (
    midpoint_params_from_space,
    run_optuna,
    suggest_params_from_space,
)
from src.utils.paths import get_experiments_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"


def _load_model_config() -> dict:
    import yaml

    with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_lgbm_params(base: dict, seed: int, n_classes: int) -> dict:
    """Mirror lightgbm_model._build_lightgbm_params so CV uses identical model setup."""
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task B classification: HPO + grouped K-fold CV")
    p.add_argument("--dataset", default="costa")
    p.add_argument("--task", default="classification", choices=["classification"])
    p.add_argument("--split-path", default="path_a", choices=["path_a", "path_b"])
    p.add_argument("--profile", default="plus_physics")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--n-folds", type=int, default=None, help="default: experiment.n_cv_folds or 5")
    p.add_argument("--seed", type=int, default=None, help="default: experiment.seeds[0] or 42")
    p.add_argument("--group-col", default="segment_id",
                   help="Episode grouping column (1:1 with episode_id in these splits).")
    p.add_argument("--no-hpo", action="store_true", help="Skip HPO; use midpoint params.")
    p.add_argument("--best-params", default=None,
                   help="Path to a JSON of base LightGBM params — skips HPO.")
    p.add_argument("--n-trials", type=int, default=None, help="Override HPO trial count.")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def _resolve_seed(config: dict, cli: int | None) -> int:
    if cli is not None:
        return int(cli)
    seeds = config.get("experiment", {}).get("seeds", [42])
    return int(seeds[0] if isinstance(seeds, list) else seeds)


def _aggregate(per_fold_metrics: list[dict]) -> dict:
    """mean / std / median / IQR of every numeric metric across folds (Task A format)."""
    numeric_keys: set[str] = set()
    for m in per_fold_metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None:
                numeric_keys.add(k)
    summary: dict[str, dict] = {}
    for k in sorted(numeric_keys):
        vals = [float(m[k]) for m in per_fold_metrics
                if isinstance(m.get(k), (int, float)) and m.get(k) is not None]
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


def _resolve_best_params(
    args, config: dict, lgb_space: dict, hpo_cfg: dict, seed: int,
    x_train, y_train, x_val, y_val, n_classes: int,
) -> tuple[dict, bool]:
    """Stage 0: load / midpoint / HPO-once on the ORIGINAL holdout. Returns (base_params, hpo_ran)."""
    if args.best_params:
        base = json.loads(Path(args.best_params).read_text(encoding="utf-8"))
        logger.info("STAGE 0: skipped — using --best-params from {}", args.best_params)
        return base, False
    if args.no_hpo:
        logger.info("STAGE 0: skipped (--no-hpo) — midpoint params")
        return midpoint_params_from_space(lgb_space), False

    n_trials = args.n_trials or int(hpo_cfg.get("n_trials_phase1", 50))
    logger.info("STAGE 0: HPO once on original train/val holdout ({} trials, f1_macro)", n_trials)

    def objective(trial):
        params = _build_lgbm_params(suggest_params_from_space(trial, lgb_space), seed, n_classes)
        model = lgb.LGBMClassifier(**params)
        model.fit(x_train, y_train)
        return float(f1_score(y_val, model.predict(x_val), average="macro", zero_division=0))

    best_base, study = run_optuna(
        objective, search_space=lgb_space, n_trials=n_trials,
        direction=str(hpo_cfg.get("direction", "maximize")), seed=seed,
        n_jobs=1, timeout_seconds=hpo_cfg.get("timeout_seconds"),
        sampler_name=str(hpo_cfg.get("sampler", "tpe")),
        pruner_name=str(hpo_cfg.get("pruner", "none")),
        storage_url=None, study_name=f"clf_kfold_hpo_{args.dataset}_{args.profile}",
        load_if_exists=False,
    )
    logger.info("STAGE 0 done — best holdout f1_macro={:.4f}", float(study.best_value))
    return best_base, True


def main() -> None:
    args = _parse_args()
    config = _load_model_config()
    seed = _resolve_seed(config, args.seed)
    n_folds = args.n_folds or int(config.get("experiment", {}).get("n_cv_folds", 5))

    ml_cfg = config["classification"]["ml"]
    lgb_space = ml_cfg["models"]["lightgbm"]
    hpo_cfg = ml_cfg["hpo"]

    # ── STAGE 1: load + pool ──────────────────────────────────────────────────
    train_df, val_df, test_df, manifest, resolved_run_dir = load_features_for_task(
        task=args.task, profile=args.profile, run_dir=args.run_dir,
        run_id=args.run_id, dataset=args.dataset, split_path=args.split_path,
    )
    features = manifest.get("final_features", [])
    label_col = str(manifest.get("label_column", "label"))
    if not features:
        raise ValueError("features_manifest.json has no final_features")
    if args.group_col not in train_df.columns:
        raise KeyError(
            f"group column '{args.group_col}' not in feature parquet "
            f"(have: {[c for c in train_df.columns if 'segment' in c or 'episode' in c]})"
        )

    pool = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    encoder = LabelEncoder().fit(pool[label_col])
    classes = encoder.classes_
    n_classes = len(classes)
    x_pool = pool[features]
    y_pool = encoder.transform(pool[label_col])
    groups = pool[args.group_col].to_numpy()

    # per-class episode support over the whole pool (for the report)
    ep_per_class = (
        pool.assign(_y=pool[label_col]).groupby("_y")[args.group_col].nunique().to_dict()
    )
    logger.info(
        "Pooled {} rows | classes={} | episodes/class={} | total episodes={}",
        len(pool), [str(c) for c in classes],
        {str(k): int(v) for k, v in ep_per_class.items()}, int(pool[args.group_col].nunique()),
    )
    min_eps = min(ep_per_class.values())
    if min_eps < n_folds:
        logger.warning(
            "Smallest class has {} episodes < n_folds={} — reduce folds or some folds "
            "will miss that class.", min_eps, n_folds,
        )

    # ── STAGE 0: HPO once on the ORIGINAL holdout (not the pool) ───────────────
    x_tr, y_tr = train_df[features], encoder.transform(train_df[label_col])
    x_va, y_va = val_df[features], encoder.transform(val_df[label_col])
    base_params, hpo_ran = _resolve_best_params(
        args, config, lgb_space, hpo_cfg, seed, x_tr, y_tr, x_va, y_va, n_classes,
    )
    params = _build_lgbm_params(base_params, seed, n_classes)
    logger.info("Fixed params for CV: {}", {k: params[k] for k in list(params)[:8]})

    # ── STAGE 2: StratifiedGroupKFold retrain ─────────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    per_fold_metrics: list[dict] = []
    fold_records: list[dict] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(sgkf.split(x_pool, y_pool, groups)):
        model = lgb.LGBMClassifier(**params)
        model.fit(x_pool.iloc[tr_idx], y_pool[tr_idx])
        proba = model.predict_proba(x_pool.iloc[te_idx])
        pred = np.argmax(proba, axis=1)
        y_te = y_pool[te_idx]

        summary_metrics, pr_auc_by_class, report = compute_classification_metrics(
            y_te, pred, proba, classes,
        )
        flat = dict(summary_metrics)
        for cls, v in pr_auc_by_class.items():
            flat[f"pr_auc_class_{cls}"] = v
        for cls in classes:
            flat[f"f1_class_{cls}"] = float(report.get(str(cls), {}).get("f1-score", float("nan")))
        per_fold_metrics.append(flat)

        te_groups = pool.iloc[te_idx]
        test_eps_per_class = (
            te_groups.assign(_y=te_groups[label_col]).groupby("_y")[args.group_col].nunique().to_dict()
        )
        fold_records.append({
            "fold_idx": fold_idx,
            "n_train_rows": int(len(tr_idx)),
            "n_test_rows": int(len(te_idx)),
            "n_test_episodes": int(te_groups[args.group_col].nunique()),
            "test_episodes_per_class": {str(k): int(v) for k, v in test_eps_per_class.items()},
            "f1_macro": flat["f1_macro"],
            "f1_weighted": flat["f1_weighted"],
            "balanced_accuracy": flat["balanced_accuracy"],
        })
        logger.info(
            "Fold {} | test_rows={:,} test_eps={} | f1_macro={:.4f} f1_weighted={:.4f}",
            fold_idx, len(te_idx), int(te_groups[args.group_col].nunique()),
            flat["f1_macro"], flat["f1_weighted"],
        )

    # ── STAGE 3: aggregate + write ────────────────────────────────────────────
    summary = _aggregate(per_fold_metrics)
    out_dir = Path(args.output_dir) if args.output_dir else (
        get_experiments_root() / "classification" / "lightgbm"
        / f"kfold_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "kfold_summary_classification.json"
    summary_path.write_text(json.dumps({
        "model": "lightgbm",
        "task": args.task,
        "dataset": args.dataset,
        "split_path": args.split_path,
        "cv_strategy": f"StratifiedGroupKFold({args.group_col})",
        "n_folds": n_folds,
        "n_folds_completed": len(per_fold_metrics),
        "seed": seed,
        "hpo_stage_ran": hpo_ran,
        "selection_metric": "f1_macro (holdout HPO)",
        "classes": [str(c) for c in classes],
        "episodes_per_class": {str(k): int(v) for k, v in ep_per_class.items()},
        "feature_run_dir": str(resolved_run_dir),
        "best_params": base_params,
        "summary": summary,
        "fold_records": fold_records,
    }, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(fold_records).to_csv(out_dir / "fold_records.csv", index=False)

    # headline
    print("\n" + "=" * 70)
    print(f"Task B classification — {n_folds}-fold StratifiedGroupKFold({args.group_col})")
    print("-" * 70)
    print(f"{'metric':<28}{'mean':>10}{'std':>10}{'n':>5}")
    print("-" * 70)
    headline = ["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy", "pr_auc_weighted"]
    headline += [f"f1_class_{c}" for c in classes]
    for k in headline:
        if k in summary:
            s = summary[k]
            print(f"{k:<28}{s['mean']:>10.4f}{s['std']:>10.4f}{s['n_folds']:>5d}")
    print("=" * 70)
    logger.success("Summary saved: {}", summary_path)

    # ── MLflow aggregate run ──────────────────────────────────────────────────
    try:
        import mlflow

        from src.mlflow_setup import init_tracking
        init_tracking("classification")
        run_name = f"kfold_summary_lightgbm_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "task": args.task, "dataset": args.dataset, "split_path": args.split_path,
                "model": "lightgbm", "run_type": "kfold_summary",
                "cv_strategy": f"StratifiedGroupKFold({args.group_col})",
                "methodology": "hpo_once_then_grouped_kfold",
            })
            mlflow.log_params({
                "n_folds": n_folds, "seed": seed, "hpo_stage_ran": hpo_ran,
                "n_folds_completed": len(per_fold_metrics),
            })
            for metric_name, s in summary.items():
                mlflow.log_metric(f"{metric_name}__mean", float(s["mean"]))
                mlflow.log_metric(f"{metric_name}__std", float(s["std"]))
            for p in (summary_path, out_dir / "fold_records.csv"):
                if p.exists():
                    mlflow.log_artifact(str(p))
        logger.info("MLflow aggregate run logged: {}", run_name)
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): {}", exc)


if __name__ == "__main__":
    main()
