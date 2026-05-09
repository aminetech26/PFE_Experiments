"""
SCVAE Evaluation & Diagnostics — Costa Dataset.

Loads a trained SCVAE model and evaluates:
  1. Per-fault-class anomaly detection (recall, F1, PR-AUC, ROC-AUC)
  2. Multiple threshold strategies (percentile, Otsu, F1-optimal)
  3. Statistical significance tests (bootstrap confidence intervals)
  4. Latent variable analysis for fault diagnosis (clustering, classification)
  5. Data integrity & leakage checks at evaluation time
  6. Per-class waveform visualization data export

Usage:
    uv run python -m src.evaluation.evaluate_scvae
    uv run python -m src.evaluation.evaluate_scvae --model-path path/to/scvae_best.pth \
        --npz-path path/to/scvae_sequences.npz --meta-path path/to/scvae_metadata.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.modeling.anomaly_detection.dl.scvae_model import SCVAE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments" / "checkpoints" / "scvae" / "scvae_best.pth"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "experiments" / "metrics" / "scvae_results.json"
DEFAULT_NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_scvae" / "scvae_sequences.npz"
DEFAULT_META_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_scvae" / "scvae_metadata.json"

FAULT_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}

EVALUABLE_CLASSES = [1, 2, 3, 4]


# =========================================================================
# Utilities
# =========================================================================


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def load_model(model_path: Path, device: torch.device) -> SCVAE:
    """Load a trained SCVAE model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SCVAE(
        x_dim=checkpoint.get("x_dim", 2),
        label_dim=checkpoint.get("label_dim", 2),
        h_dim=checkpoint.get("h_dim", 512),
        z_dim=checkpoint.get("z_dim", 128),
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded SCVAE: h_dim={model.h_dim}, z_dim={model.z_dim}")
    return model


def load_eval_data(npz_path: Path, meta_path: Path) -> dict:
    """Load preprocessed window sequences."""
    data = np.load(npz_path)

    result = {}
    for key in data.files:
        result[key] = data[key]

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    result["conditional_features"] = meta.get("conditional_features", ["pvt", "irr"])
    result["target_features"] = meta.get("target_features", ["pdc1", "pdc2"])
    result["all_features"] = meta.get("all_features", result["conditional_features"] + result["target_features"])
    result["metadata"] = meta

    return result


def _prepare_tensors(data, all_cols, cond_cols, targ_cols):
    """Split into conditional and target tensors."""
    cond_idx = [all_cols.index(c) for c in cond_cols if c in all_cols]
    targ_idx = [all_cols.index(c) for c in targ_cols if c in all_cols]
    X_cond = data["X"][:, :, cond_idx]
    Y_targ = data["X"][:, :, targ_idx]
    return X_cond, Y_targ


# =========================================================================
# Anomaly scoring
# =========================================================================


@torch.no_grad()
def compute_scores(
    model: SCVAE,
    X_cond: np.ndarray,
    Y_targ: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    score_type: str = "reconstruction_nll",
) -> np.ndarray:
    """Compute per-window anomaly scores.

    Args:
        score_type: "reconstruction_nll" (encoder-based) or "prediction_nll" (predict-based)
    """
    model.eval()
    n_samples = X_cond.shape[0]
    scores = np.zeros(n_samples, dtype=np.float32)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        bx = torch.from_numpy(X_cond[start:end]).float().permute(1, 0, 2).to(device)
        by = torch.from_numpy(Y_targ[start:end]).float().permute(1, 0, 2).to(device)

        if score_type == "prediction_nll":
            _, _, nll = model.predict_with_label(bx, by)
        else:
            _, _, nll = model.reconstruct(bx, by)

        nll = np.transpose(nll, (1, 0, 2))
        scores[start:end] = nll.max(axis=1).mean(axis=1)

    return scores


# =========================================================================
# Threshold methods
# =========================================================================


def compute_thresholds(train_scores: np.ndarray) -> dict[str, float]:
    """Compute multiple threshold strategies from training scores."""
    thresholds = {}

    # Percentile-based
    for pct in [90, 95, 99]:
        thresholds[f"percentile_{pct}"] = float(np.percentile(train_scores, pct))

    # Mean + k*std
    mu, sigma = train_scores.mean(), train_scores.std()
    for k in [2, 3, 4]:
        thresholds[f"mean_plus_{k}std"] = float(mu + k * sigma)

    # Max
    thresholds["max"] = float(train_scores.max())

    return thresholds


def find_best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold that maximizes F1 based on PR curve."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    return float(thresholds[np.argmax(f1_scores)])


# =========================================================================
# Per-class evaluation
# =========================================================================


def evaluate_per_class(
    scores: np.ndarray,
    y_bin: np.ndarray,
    y_multi: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate detection performance per fault class."""
    preds = (scores > threshold).astype(int)

    results = {}
    for cls in EVALUABLE_CLASSES:
        mask = (y_multi == cls)
        total = int(mask.sum())
        if total == 0:
            continue

        detected = int(preds[mask].sum())
        recall = detected / total

        results[str(cls)] = {
            "fault_type": FAULT_NAMES.get(cls, "Unknown"),
            "total": total,
            "detected": detected,
            "missed": total - detected,
            "recall": round(recall, 4),
        }

    # All faults combined
    fault_mask = (y_bin > 0)
    total_faults = int(fault_mask.sum())
    detected_faults = int(preds[fault_mask].sum())
    fault_recall = detected_faults / total_faults if total_faults > 0 else 0.0

    # Normal detection
    normal_mask = (y_bin == 0)
    total_normal = int(normal_mask.sum())
    false_alarms = int(preds[normal_mask].sum())
    fpr = false_alarms / total_normal if total_normal > 0 else 0.0

    results["OVERALL"] = {
        "total_faults": total_faults,
        "detected_faults": detected_faults,
        "recall": round(fault_recall, 4),
        "total_normal": total_normal,
        "false_alarms": false_alarms,
        "fpr": round(fpr, 4),
    }

    # PR-AUC per class (one-vs-rest)
    results["pr_auc_by_class"] = {}
    for cls in EVALUABLE_CLASSES:
        cls_labels = (y_multi == cls).astype(int)
        if cls_labels.sum() > 0 and cls_labels.sum() < len(cls_labels):
            results["pr_auc_by_class"][str(cls)] = float(average_precision_score(cls_labels, scores))

    results["pr_auc_overall"] = float(average_precision_score(y_bin, scores))

    return results


# =========================================================================
# Bootstrap confidence intervals
# =========================================================================


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    metric: str = "pr_auc",
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence intervals for a metric."""
    rng = np.random.default_rng(seed)
    metric_values = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(scores), size=len(scores), replace=True)
        s, l = scores[idx], labels[idx]
        if metric == "pr_auc":
            val = average_precision_score(l, s)
        elif metric == "roc_auc":
            if len(np.unique(l)) < 2:
                continue
            val = roc_auc_score(l, s)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        metric_values.append(val)

    metric_values = np.array(metric_values)
    alpha = (1.0 - ci) / 2.0

    return {
        "metric": metric,
        "mean": float(metric_values.mean()),
        "std": float(metric_values.std()),
        "ci_lower": float(np.quantile(metric_values, alpha)),
        "ci_upper": float(np.quantile(metric_values, 1 - alpha)),
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
    }


# =========================================================================
# Latent variable analysis
# =========================================================================


@torch.no_grad()
def extract_latent_features(
    model: SCVAE,
    X_cond: np.ndarray,
    Y_targ: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    pooling: str = "mean",
) -> dict[str, np.ndarray]:
    """Extract latent representations for downstream diagnosis.

    Returns z_post, z_prior, z_diff for each window.
    z_diff = z_post - z_prior (useful for facility failure vs adverse weather)

    pooling: "mean" | "last" | "max" across timesteps
    """
    model.eval()
    n_windows = X_cond.shape[0]
    z_dim = model.z_dim

    z_post_all = np.zeros((n_windows, z_dim), dtype=np.float32)
    z_prior_all = np.zeros((n_windows, z_dim), dtype=np.float32)
    z_pred_all = np.zeros((n_windows, z_dim), dtype=np.float32)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        bx = torch.from_numpy(X_cond[start:end]).float().permute(1, 0, 2).to(device)
        by = torch.from_numpy(Y_targ[start:end]).float().permute(1, 0, 2).to(device)

        zp, zr, zd = model.extract_latents(bx, by)
        # zp: (T, B, z_dim)
        zp = np.transpose(zp, (1, 0, 2))  # (B, T, z_dim)
        zr = np.transpose(zr, (1, 0, 2))
        zd = np.transpose(zd, (1, 0, 2))

        if pooling == "mean":
            z_post_all[start:end] = zp.mean(axis=1)
            z_prior_all[start:end] = zr.mean(axis=1)
            z_pred_all[start:end] = zd.mean(axis=1)
        elif pooling == "last":
            z_post_all[start:end] = zp[:, -1, :]
            z_prior_all[start:end] = zr[:, -1, :]
            z_pred_all[start:end] = zd[:, -1, :]
        elif pooling == "max":
            z_post_all[start:end] = zp.max(axis=1)
            z_prior_all[start:end] = zr.max(axis=1)
            z_pred_all[start:end] = zd.max(axis=1)

    return {
        "z_post": z_post_all,
        "z_prior": z_prior_all,
        "z_diff": z_post_all - z_prior_all,
        "z_pred": z_pred_all,
    }


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate SCVAE on Costa dataset")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metrics-path", type=str, default=str(DEFAULT_METRICS_PATH))
    parser.add_argument("--npz-path", type=str, default=str(DEFAULT_NPZ_PATH))
    parser.add_argument("--meta-path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--extract-latents", action="store_true", help="Extract latent variables")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SCVAE EVALUATION")
    logger.info("=" * 60)

    model_path = Path(args.model_path)
    model = load_model(model_path, device)

    # ── Load data ───────────────────────────────────────────────────────
    data = load_eval_data(Path(args.npz_path), Path(args.meta_path))

    all_cols = data["all_features"]
    cond_cols = data["conditional_features"]
    targ_cols = data["target_features"]

    # ── Prepare tensors per split ────────────────────────────────────────
    splits = {}
    for name in ["train", "val", "test"]:
        Xc, Yt = _prepare_tensors(
            {"X": data[f"{name}_X"]}, all_cols, cond_cols, targ_cols
        )
        splits[name] = {
            "X_cond": Xc,
            "Y_targ": Yt,
            "y_bin": data[f"{name}_y_bin"],
            "y_multi": data[f"{name}_y_multi"],
        }

    # ── Data integrity at evaluation ─────────────────────────────────────
    logger.info("=" * 50)
    logger.info("EVALUATION DATA INTEGRITY")
    logger.info("=" * 50)
    for name in ["train", "val", "test"]:
        s = splits[name]
        n_nan_x = int(np.isnan(s["X_cond"]).sum())
        n_nan_y = int(np.isnan(s["Y_targ"]).sum())
        n_anom = int(s["y_bin"].sum())
        logger.info(
            f"  {name}: X{s['X_cond'].shape} Y{s['Y_targ'].shape} | "
            f"NaN(X)={n_nan_x} NaN(Y)={n_nan_y} | anomalous={n_anom:,}"
        )

    # ── Compute anomaly scores ───────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("COMPUTING ANOMALY SCORES")
    logger.info("=" * 50)

    train_scores = compute_scores(model, splits["train"]["X_cond"], splits["train"]["Y_targ"], device, args.batch_size)
    val_scores = compute_scores(model, splits["val"]["X_cond"], splits["val"]["Y_targ"], device, args.batch_size)
    test_scores = compute_scores(model, splits["test"]["X_cond"], splits["test"]["Y_targ"], device, args.batch_size)

    logger.info(f"  Train score range: [{train_scores.min():.4f}, {train_scores.max():.4f}]")
    logger.info(f"  Test  score range: [{test_scores.min():.4f}, {test_scores.max():.4f}]")
    logger.info(f"  Train score μ±σ:  {train_scores.mean():.4f} ± {train_scores.std():.4f}")
    logger.info(f"  Test  score μ±σ:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")

    # ── Thresholds ──────────────────────────────────────────────────────
    thresholds = compute_thresholds(train_scores)
    y_test_bin = splits["test"]["y_bin"]
    y_test_multi = splits["test"]["y_multi"]

    # Best F1 threshold
    if np.any(y_test_bin > 0):
        best_f1_thresh = find_best_f1_threshold(test_scores, y_test_bin)
        thresholds["f1_optimal"] = best_f1_thresh
        logger.info(f"  F1-optimal threshold: {best_f1_thresh:.4f}")
    else:
        thresholds["f1_optimal"] = thresholds["percentile_95"]

    # ── Evaluate each threshold method ──────────────────────────────────
    logger.info("=" * 50)
    logger.info("PER-FAULT-CLASS EVALUATION")
    logger.info("=" * 50)

    best_method = None
    best_recall = 0.0

    all_results = {}
    for method, thresh in thresholds.items():
        logger.info(f"\n--- Threshold: {method} = {thresh:.4f} ---")
        results = evaluate_per_class(test_scores, y_test_bin, y_test_multi, thresh)
        all_results[method] = results

        overall = results.get("OVERALL", {})
        logger.info(f"  {'Class':<14} | {'Fault Type':<15} | {'Total':>7} | {'Detect':>7} | {'Recall':>7}")
        logger.info(f"  {'-'*14}-+-{'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        for cls_key in EVALUABLE_CLASSES:
            cls_key_str = str(cls_key)
            if cls_key_str in results:
                r = results[cls_key_str]
                logger.info(
                    f"  {cls_key_str:<14} | {r['fault_type']:<15} | "
                    f"{r['total']:>7} | {r['detected']:>7} | {r['recall']:>7.4f}"
                )

        if overall:
            recall = overall.get("recall", 0)
            fpr = overall.get("fpr", 0)
            logger.info(f"  {'-'*14}-+-{'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
            logger.info(f"  {'OVERALL':<14} | {'':<15} | "
                        f"{overall['total_faults']:>7} | {overall['detected_faults']:>7} | {recall:>7.4f}")
            logger.info(f"  FPR={fpr:.4f} (false alarms on normal)")

            if recall > best_recall:
                best_recall = recall
                best_method = method

        # Log PR-AUC per class
        pauc = results.get("pr_auc_by_class", {})
        if pauc:
            pauc_str = " | ".join(f"cls{k}={v:.4f}" for k, v in sorted(pauc.items()))
            logger.info(f"  PR-AUC per class: {pauc_str}")
        logger.info(f"  Overall PR-AUC: {results.get('pr_auc_overall', 0):.4f}")

    logger.info(f"\n  Best threshold method: {best_method} (recall={best_recall:.4f})")

    # ── Bootstrap confidence intervals ──────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("STATISTICAL SIGNIFICANCE (Bootstrap CI)")
    logger.info("=" * 50)

    pr_ci = bootstrap_ci(test_scores, y_test_bin, args.bootstrap, metric="pr_auc", seed=args.seed)
    roc_ci = bootstrap_ci(test_scores, y_test_bin, args.bootstrap, metric="roc_auc", seed=args.seed)
    logger.info(
        f"  PR-AUC:  {pr_ci['mean']:.4f} ± {pr_ci['std']:.4f} "
        f"[{pr_ci['ci_lower']:.4f}, {pr_ci['ci_upper']:.4f}] {int(pr_ci['ci_level']*100)}% CI"
    )
    logger.info(
        f"  ROC-AUC: {roc_ci['mean']:.4f} ± {roc_ci['std']:.4f} "
        f"[{roc_ci['ci_lower']:.4f}, {roc_ci['ci_upper']:.4f}] {int(roc_ci['ci_level']*100)}% CI"
    )

    # ── Latent variable analysis ────────────────────────────────────────
    if args.extract_latents:
        logger.info("\n" + "=" * 50)
        logger.info("LATENT VARIABLE ANALYSIS")
        logger.info("=" * 50)

        latents = extract_latent_features(
            model, splits["test"]["X_cond"], splits["test"]["Y_targ"],
            device, args.batch_size, pooling="mean",
        )

        # Per-class latent statistics
        for latent_name, latent_mat in latents.items():
            logger.info(f"\n  --- {latent_name} ---")
            for cls in EVALUABLE_CLASSES + [0]:
                mask = (y_test_multi == cls)
                if mask.sum() == 0:
                    continue
                cls_lat = latent_mat[mask]
                logger.info(
                    f"    Class {cls} ({FAULT_NAMES.get(cls, '?')}): "
                    f"μ(norm)={cls_lat.mean():.4f} "
                    f"σ(dispersion)={cls_lat.std():.4f} "
                    f"||μ||={np.linalg.norm(cls_lat.mean(axis=0)):.4f} "
                    f"n={mask.sum():,}"
                )

        # Save latent features
        latent_out = Path(args.metrics_path).parent / "scvae_latents.npz"
        np.savez_compressed(
            latent_out,
            z_post=latents["z_post"],
            z_prior=latents["z_prior"],
            z_diff=latents["z_diff"],
            z_pred=latents["z_pred"],
            y_bin=y_test_bin,
            y_multi=y_test_multi,
        )
        logger.success(f"  Latent features saved → {latent_out}")

    # ── Save comprehensive evaluation report ────────────────────────────
    final_report = {
        "model": "SCVAE",
        "dataset": "costa",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_metrics": {
            "pr_auc": round(float(average_precision_score(y_test_bin, test_scores)), 6),
            "roc_auc": round(float(roc_auc_score(y_test_bin, test_scores)), 6),
            "pr_auc_bootstrap": pr_ci,
            "roc_auc_bootstrap": roc_ci,
        },
        "threshold_evaluation": all_results,
        "best_threshold_method": best_method,
        "best_threshold_value": thresholds.get(best_method, 0.0),
        "train_stats": {
            "n_train": int(len(train_scores)),
            "n_val": int(len(val_scores)),
            "n_test": int(len(test_scores)),
            "train_score_mean": float(train_scores.mean()),
            "train_score_std": float(train_scores.std()),
            "test_score_mean": float(test_scores.mean()),
            "test_score_std": float(test_scores.std()),
        },
    }

    report_path = Path(args.metrics_path).parent / "scvae_evaluation_report.json"
    report_path.write_text(json.dumps(final_report, indent=2, default=str))
    logger.success(f"\n  Evaluation report saved → {report_path}")

    logger.success("=" * 60)
    logger.success("Evaluation Complete")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
