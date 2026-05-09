"""
Evaluation script for the Hybrid Anomaly Detection model.

Evaluates: AE-LSTM standalone, Isolation Forest standalone, and Ensemble.

Paper: Ahirwar & Nandanwar (2025) ICoEIT.

Usage:
    uv run python -m src.evaluation.evaluate_hybrid
    uv run python -m src.evaluation.evaluate_hybrid --model-path path/to/hybrid_best.pt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.modeling.anomaly_detection.dl.hybrid_model import AELSTM, HybridScorer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments" / "checkpoints" / "hybrid" / "hybrid_best.pt"
DEFAULT_NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_hybrid" / "hybrid_sequences.npz"
DEFAULT_META_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_hybrid" / "hybrid_metadata.json"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "experiments" / "metrics"

FAULT_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}

EVALUABLE_CLASSES = [1, 2, 3, 4]


def _resolve_device(d: str | None) -> torch.device:
    if d is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(d)


def compute_thresholds(train_scores: np.ndarray) -> dict[str, float]:
    thresholds = {}
    for pct in [90, 95, 99]:
        thresholds[f"percentile_{pct}"] = float(np.percentile(train_scores, pct))
    mu, sigma = train_scores.mean(), train_scores.std()
    for k in [2, 3, 4]:
        thresholds[f"mean_plus_{k}std"] = float(mu + k * sigma)
    thresholds["max"] = float(train_scores.max())
    return thresholds


def bootstrap_ci(
    scores: np.ndarray, labels: np.ndarray,
    n_bootstrap: int = 1000, ci_level: float = 0.95,
    metric: str = "pr_auc", seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(scores), size=len(scores), replace=True)
        s, l = scores[idx], labels[idx]
        if metric == "pr_auc":
            v = average_precision_score(l, s) if np.any(l > 0) else 0.0
        elif metric == "roc_auc":
            v = roc_auc_score(l, s) if len(np.unique(l)) > 1 else 0.0
        else:
            continue
        vals.append(v)
    vals = np.array(vals)
    alpha = (1.0 - ci_level) / 2.0
    return {
        "metric": metric,
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "ci_lower": float(np.quantile(vals, alpha)),
        "ci_upper": float(np.quantile(vals, 1 - alpha)),
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid model")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--npz-path", type=str, default=str(DEFAULT_NPZ_PATH))
    parser.add_argument("--meta-path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--metrics-dir", type=str, default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("HYBRID MODEL EVALUATION")
    logger.info("=" * 60)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    aelstm = AELSTM(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        latent_dim=ckpt["latent_dim"],
        dropout=ckpt["dropout"],
        device=device,
    ).to(device)
    aelstm.load_state_dict(ckpt["aelstm_state_dict"])
    aelstm.eval()
    logger.info(f"Loaded AE-LSTM: hidden={ckpt['hidden_dim']}, latent={ckpt['latent_dim']}")

    # ── Load data ────────────────────────────────────────────────────
    data = dict(np.load(args.npz_path))
    X_train_res = data["train_X_res"]
    X_train_if = data["train_X_if"]
    X_test_res = data["test_X_res"]
    X_test_if = data["test_X_if"]
    y_test_bin = data["test_y_bin"]
    y_test_multi = data["test_y_multi"]

    # ── Fit IF + scorer on training data ─────────────────────────────
    logger.info("Fitting Isolation Forest on training data...")
    if_model = IsolationForest(
        n_estimators=100, contamination=0.05,
        random_state=args.seed, n_jobs=-1,
    )
    if_model.fit(X_train_if)

    alpha = ckpt.get("ensemble_alpha", 0.6)
    scorer = HybridScorer(aelstm, if_model, alpha=alpha, device=device)
    scorer.fit_score_stats(X_train_res, X_train_if)

    # ── Compute scores ───────────────────────────────────────────────
    logger.info("Computing anomaly scores...")
    all_scores = scorer.compute_scores_separately(X_test_res, X_test_if)

    # ── Evaluate each component ──────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 50)
    logger.info(f"  {'Model':<20} | {'PR-AUC':>8} | {'ROC-AUC':>8}")
    logger.info(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}")

    comp_results = {}
    for name, scores in all_scores.items():
        pr = average_precision_score(y_test_bin, scores) if np.any(y_test_bin > 0) else 0.0
        roc = roc_auc_score(y_test_bin, scores) if len(np.unique(y_test_bin)) > 1 else 0.0
        comp_results[name] = {"pr_auc": round(float(pr), 4), "roc_auc": round(float(roc), 4)}
        logger.info(f"  {name:<20} | {pr:>8.4f} | {roc:>8.4f}")

    # ── Threshold-based evaluation (ensemble) ────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("THRESHOLD EVALUATION (Ensemble)")
    logger.info("=" * 50)

    train_scores = scorer.score(X_train_res, X_train_if)
    test_scores = all_scores["ensemble"]
    thresholds = compute_thresholds(train_scores)

    if np.any(y_test_bin > 0):
        precision, recall, thresh_vals = precision_recall_curve(y_test_bin, test_scores)
        f1_s = 2 * precision * recall / (precision + recall + 1e-10)
        thresholds["f1_optimal"] = float(thresh_vals[np.argmax(f1_s)])

    for method, thresh in thresholds.items():
        if np.any(y_test_bin > 0):
            preds = (test_scores > thresh).astype(int)
            logger.info(f"\n  --- {method} = {thresh:.4f} ---")
            logger.info(f"  {'Class':<14} | {'Fault Type':<15} | {'Total':>7} | {'Detect':>7} | {'Recall':>7}")
            logger.info(f"  {'-'*14}-+-{'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

            for cls in EVALUABLE_CLASSES:
                mask = (y_test_multi == cls)
                ct = int(mask.sum())
                if ct == 0:
                    continue
                cd = int(preds[mask].sum())
                cr = cd / ct
                logger.info(
                    f"  {cls:<14} | {FAULT_NAMES[cls]:<15} | {ct:>7} | {cd:>7} | {cr:>7.4f}"
                )

            fmask = (y_test_bin == 1)
            ft = int(fmask.sum())
            fd = int(preds[fmask].sum())
            fr = fd / ft if ft > 0 else 0.0
            logger.info(f"  {'-'*14}-+-{'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
            logger.info(f"  {'ALL FAULTS':<14} | {'':<15} | {ft:>7} | {fd:>7} | {fr:>7.4f}")

    # ── Bootstrap CIs ────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("BOOTSTRAP CONFIDENCE INTERVALS")
    logger.info("=" * 50)

    bootstrap_results = {}
    for name, scores in all_scores.items():
        pr_ci = bootstrap_ci(scores, y_test_bin, args.bootstrap, metric="pr_auc", seed=args.seed)
        roc_ci = bootstrap_ci(scores, y_test_bin, args.bootstrap, metric="roc_auc", seed=args.seed)
        bootstrap_results[name] = {"pr_auc": pr_ci, "roc_auc": roc_ci}

        if np.any(y_test_bin > 0):
            logger.info(
                f"  {name:<18} PR-AUC: {pr_ci['mean']:.4f} ± {pr_ci['std']:.4f} "
                f"[{pr_ci['ci_lower']:.4f}, {pr_ci['ci_upper']:.4f}]"
            )

    # ── Save report ──────────────────────────────────────────────────
    report = {
        "model": "Hybrid (AE-LSTM + Prophet + IsolationForest)",
        "dataset": "costa",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "component_performance": comp_results,
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "bootstrap": bootstrap_results,
    }

    report_path = Path(args.metrics_dir) / "hybrid_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.success(f"\n  Evaluation report → {report_path}")

    logger.success("=" * 60)
    logger.success("EVALUATION COMPLETE")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
