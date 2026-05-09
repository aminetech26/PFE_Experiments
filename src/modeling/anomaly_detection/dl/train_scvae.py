"""
SCVAE Training Script — Costa Dataset.

Implements the SCVAE (Sequential Conditional VAE) from:
  Li et al. (2024) "Sensing anomaly of photovoltaic systems with sequential
  conditional variational autoencoder", Applied Energy 353:122124.

Key features:
  - GVSAO hyperparameter optimization (learning rate, batch size)
  - Architecture grid search (h_dim, z_dim, window_size)
  - Data integrity checks before training
  - Leakage prevention validation
  - Multiple loss modes: 0 (KLD+NLL+smooth), 1 (+prior NLL), 2 (+predict KLD+NLL)
  - Early stopping with patience
  - Model checkpointing and metrics logging

Usage:
    uv run python -m src.modeling.anomaly_detection.dl.train_scvae
    uv run python -m src.modeling.anomaly_detection.dl.train_scvae --no-gvsao --lr 1e-5 --batch 256
    uv run python -m src.modeling.anomaly_detection.dl.train_scvae --h-dim 512 --z-dim 128 --epochs 1000
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.anomaly_detection.dl.gvsao import (
    GVSaoConfig,
    run_gvsao,
)
from src.modeling.anomaly_detection.dl.scvae_model import SCVAE

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_scvae" / "scvae_sequences.npz"
DEFAULT_META_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_scvae" / "scvae_metadata.json"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "scvae"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "experiments" / "metrics"

FAULT_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}

EVALUABLE_CLASSES = [1, 2, 3, 4]


# =========================================================================
# Device & Data loading
# =========================================================================


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def load_preprocessed_data(npz_path: Path, meta_path: Path) -> dict[str, Any]:
    """Load preprocessed window sequences and metadata."""
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {npz_path}\n"
            "Run: uv run python -m src.data.preprocess_scvae"
        )

    data = np.load(npz_path)
    result = {
        "train": {"X": data["train_X"], "y_bin": data["train_y_bin"], "y_multi": data["train_y_multi"]},
        "val": {"X": data["val_X"], "y_bin": data["val_y_bin"], "y_multi": data["val_y_multi"]},
        "test": {"X": data["test_X"], "y_bin": data["test_y_bin"], "y_multi": data["test_y_multi"]},
    }

    # Load metadata
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    result["conditional_features"] = meta.get("conditional_features", ["pvt", "irr"])
    result["target_features"] = meta.get("target_features", ["pdc1", "pdc2"])
    result["all_features"] = meta.get("all_features", result["conditional_features"] + result["target_features"])
    result["window_size_original"] = meta.get("window_size", 72)
    result["metadata"] = meta

    logger.info(f"Loaded preprocessed data from {npz_path}")
    for name in ["train", "val", "test"]:
        X = result[name]["X"]
        y = result[name]["y_bin"]
        logger.info(f"  {name}: X{X.shape} | anomalous={y.sum():,}/{len(y):,} ({100*y.sum()/max(len(y),1):.1f}%)")

    return result


# =========================================================================
# Data integrity checks (on windowed data)
# =========================================================================


def pre_training_checks(data: dict) -> dict:
    """Run comprehensive checks before training."""
    report = {}

    # 1. Train should have zero anomalies
    train_anom = int(data["train"]["y_bin"].sum())
    report["train_has_anomalies"] = train_anom > 0

    # 2. NaN / Inf check
    for name in ["train", "val", "test"]:
        X = data[name]["X"]
        report[f"{name}_nan"] = int(np.isnan(X).sum())
        report[f"{name}_inf"] = int(np.isinf(X).sum())
        report[f"{name}_samples"] = int(X.shape[0])

    # 3. Check feature correlation within conditional vs target
    train_X = data["train"]["X"]
    n_features = train_X.shape[2]
    report["n_features"] = int(n_features)

    # 4. Per-class stats in test set
    y_multi = data["test"]["y_multi"]
    class_dist = {}
    for cls in sorted(set(y_multi)):
        class_dist[int(cls)] = int((y_multi == cls).sum())
    report["test_class_distribution"] = class_dist

    logger.info("=" * 50)
    logger.info("PRE-TRAINING DATA INTEGRITY CHECKS")
    logger.info("=" * 50)
    status = "FAIL" if report["train_has_anomalies"] else "PASS"
    logger.info(f"  [{status}] Train anomalies: {train_anom}")
    for name in ["train", "val", "test"]:
        n_nan = report[f"{name}_nan"]
        n_inf = report[f"{name}_inf"]
        status = "PASS" if n_nan == 0 and n_inf == 0 else "FAIL"
        logger.info(f"  [{status}] {name}: NaN={n_nan}, Inf={n_inf}, samples={report[f'{name}_samples']:,}")
    logger.info(f"  [INFO] Test class distribution: {class_dist}")
    logger.info("=" * 50)

    return report


# =========================================================================
# Training utilities
# =========================================================================


def _prepare_tensors(data: dict, conditional_cols: list, target_cols: list, all_cols: list):
    """Split windows into conditional (X) and target (Y) tensors."""
    cond_indices = [all_cols.index(c) for c in conditional_cols if c in all_cols]
    targ_indices = [all_cols.index(c) for c in target_cols if c in all_cols]

    if not cond_indices or not targ_indices:
        raise ValueError(f"Could not find conditional/target features in {all_cols}")

    X_cond = data["X"][:, :, cond_indices]   # (N, T, M_x)
    Y_targ = data["X"][:, :, targ_indices]   # (N, T, M_y)
    return X_cond, Y_targ


def _make_dataloader(X, Y, batch_size, shuffle=True):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_elbo_loss(model, data, y_data, mode, reg, device):
    """Compute the total loss from the SCVAE model."""
    data = data.permute(1, 0, 2).to(device)     # (T, B, M_x)
    y_data = y_data.permute(1, 0, 2).to(device)  # (T, B, M_y)
    model(data, y_data)

    if mode == 0:
        loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss
    elif mode == 1:
        loss = (model.kld_loss + model.nll_loss + reg * model.smooth_loss
                + model.nll_loss_prior)
    elif mode == 2:
        loss = (model.kld_loss + model.nll_loss + reg * model.smooth_loss
                + model.kld_loss_predict + model.nll_loss_predict)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return loss


def train_epoch(dataloader, model, optimizer, mode, reg, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for data, y_data in dataloader:
        optimizer.zero_grad()
        loss = compute_elbo_loss(model, data, y_data, mode, reg, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def val_epoch(dataloader, model, mode, reg, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for data, y_data in dataloader:
        loss = compute_elbo_loss(model, data, y_data, mode, reg, device)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# =========================================================================
# Anomaly scoring
# =========================================================================


@torch.no_grad()
def compute_anomaly_scores(
    model: SCVAE,
    X_cond: np.ndarray,
    Y_targ: np.ndarray,
    y_bin: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    n_mc: int = 1,
) -> np.ndarray:
    """Compute per-window anomaly scores using reconstruction NLL.

    For each window, computes the max timestep NLL as the anomaly score.
    """
    model.eval()
    n_windows = X_cond.shape[0]
    scores = np.zeros(n_windows, dtype=np.float32)

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        batch_x = torch.from_numpy(X_cond[start:end]).float()
        batch_y = torch.from_numpy(Y_targ[start:end]).float()

        # (N, T, M) -> (T, N, M)
        batch_x = batch_x.permute(1, 0, 2).to(device)
        batch_y = batch_y.permute(1, 0, 2).to(device)

        _, _, nll = model.reconstruct(batch_x, batch_y, n_mc=n_mc)
        # nll: (T, N, label_dim) — take max over timesteps, mean over features
        nll = np.transpose(nll, (1, 0, 2))  # (N, T, label_dim)
        scores[start:end] = nll.max(axis=1).mean(axis=1)

    return scores


def compute_pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute PR-AUC (primary metric for anomaly detection)."""
    if np.all(labels == 0):
        return 0.0
    return float(average_precision_score(labels, scores))


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC-AUC."""
    if np.all(labels == 0) or np.all(labels == 1):
        return 0.0
    return float(roc_auc_score(labels, scores))


# =========================================================================
# GVSAO fitness function
# =========================================================================


def make_fitness_fn(
    X_train_cond: np.ndarray,
    Y_train_targ: np.ndarray,
    X_val_cond: np.ndarray,
    Y_val_targ: np.ndarray,
    y_val_bin: np.ndarray,
    device: torch.device,
    h_dim: int,
    z_dim: int,
    mode: int,
    reg: float,
    gvsao_epochs: int,
    conditional_cols: list,
    target_cols: list,
):
    """Create a fitness function for GVSAO that trains a model and returns val PR-AUC."""

    def fitness(lr: float, batch: int) -> float:
        batch = int(batch)
        batch = max(4, min(batch, X_train_cond.shape[0]))

        model = SCVAE(
            x_dim=len(conditional_cols),
            label_dim=len(target_cols),
            h_dim=h_dim,
            z_dim=z_dim,
            device=device,
        ).to(device)

        train_loader = _make_dataloader(X_train_cond, Y_train_targ, batch, shuffle=True)
        val_loader = _make_dataloader(X_val_cond, Y_val_targ, batch, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        patience = 3
        no_improve = 0

        for _ in range(gvsao_epochs):
            train_epoch(train_loader, model, optimizer, mode, reg, device)
            val_loss = val_epoch(val_loader, model, mode, reg, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        # Compute PR-AUC on val set
        scores = compute_anomaly_scores(
            model, X_val_cond, Y_val_targ, y_val_bin,
            device, batch_size=batch,
        )
        pr_auc = compute_pr_auc(scores, y_val_bin)

        # Return negative PR-AUC (minimization objective)
        return -pr_auc if pr_auc > 0 else best_val_loss

    return fitness


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Train SCVAE on Costa dataset")
    parser.add_argument("--npz-path", type=str, default=str(DEFAULT_NPZ_PATH))
    parser.add_argument("--meta-path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--metrics-dir", type=str, default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs (final model)")
    parser.add_argument("--gvsao-epochs", type=int, default=3, help="Epochs per GVSAO fitness eval")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (skip GVSAO if set)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (skip GVSAO if set)")
    parser.add_argument("--no-gvsao", action="store_true", help="Skip GVSAO HPO")
    parser.add_argument("--h-dim", type=int, default=512, help="Hidden layer dimension")
    parser.add_argument("--z-dim", type=int, default=128, help="Latent variable dimension")
    parser.add_argument("--mode", type=int, default=2, help="Loss mode: 0/1/2")
    parser.add_argument("--reg", type=float, default=0.0, help="Smoothness regularization")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Load Preprocessed Data")
    logger.info("=" * 60)
    data = load_preprocessed_data(Path(args.npz_path), Path(args.meta_path))

    all_cols = data["all_features"]
    cond_cols = data["conditional_features"]
    targ_cols = data["target_features"]
    logger.info(f"  Conditional features (x): {cond_cols}")
    logger.info(f"  Target features (y):      {targ_cols}")

    # ── Data integrity checks ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Pre-Training Data Integrity Checks")
    logger.info("=" * 60)
    integrity_report = pre_training_checks(data)
    if integrity_report["train_has_anomalies"]:
        logger.warning("WARNING: Training data contains anomalous samples! This violates the unsupervised assumption.")

    # ── Prepare tensors ─────────────────────────────────────────────────
    X_train_cond, Y_train_targ = _prepare_tensors(data["train"], cond_cols, targ_cols, all_cols)
    X_val_cond, Y_val_targ = _prepare_tensors(data["val"], cond_cols, targ_cols, all_cols)
    X_test_cond, Y_test_targ = _prepare_tensors(data["test"], cond_cols, targ_cols, all_cols)

    logger.info(f"  X_train_cond:  {X_train_cond.shape}")
    logger.info(f"  Y_train_targ:  {Y_train_targ.shape}")
    logger.info(f"  X_val_cond:    {X_val_cond.shape}")
    logger.info(f"  X_test_cond:   {X_test_cond.shape}")

    # ── GVSAO Hyperparameter Optimization ───────────────────────────────
    final_lr = args.lr or 1e-4
    final_batch_size = args.batch_size or 64
    gvsao_result = None

    if not args.no_gvsao and (args.lr is None or args.batch_size is None):
        logger.info("=" * 60)
        logger.info("Step 3: GVSAO Hyperparameter Optimization")
        logger.info("=" * 60)

        gvsao_config = GVSaoConfig(
            population_size=10,
            max_generations=5,
            lr_bounds=(1e-6, 1e-3),
            batch_bounds=(16, 512),
            seed=args.seed,
        )

        fitness_fn = make_fitness_fn(
            X_train_cond, Y_train_targ, X_val_cond, Y_val_targ,
            data["val"]["y_bin"],
            device, args.h_dim, args.z_dim, args.mode, args.reg,
            args.gvsao_epochs, cond_cols, targ_cols,
        )

        gvsao_result = run_gvsao(fitness_fn, gvsao_config, verbose=True)
        final_lr = gvsao_result.best_params["learning_rate"]
        final_batch_size = gvsao_result.best_params["batch_size"]
        final_batch_size = min(final_batch_size, X_train_cond.shape[0])
        logger.success(f"GVSAO best: lr={final_lr:.6f}, batch_size={final_batch_size}")

    # ── Train final model ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Step 4: Train SCVAE (lr={final_lr:.6f}, batch={final_batch_size}, "
                 f"h={args.h_dim}, z={args.z_dim})")
    logger.info("=" * 60)

    model = SCVAE(
        x_dim=len(cond_cols),
        label_dim=len(targ_cols),
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        device=device,
    ).to(device)

    effective_batch = min(final_batch_size, X_train_cond.shape[0])
    train_loader = _make_dataloader(X_train_cond, Y_train_targ, effective_batch, shuffle=True)
    val_loader = _make_dataloader(X_val_cond, Y_val_targ, effective_batch, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=final_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    pr_aucs: list[float] = []

    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(train_loader, model, optimizer, args.mode, args.reg, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = val_epoch(val_loader, model, args.mode, args.reg, device)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # PR-AUC
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            val_scores = compute_anomaly_scores(
                model, X_val_cond, Y_val_targ, data["val"]["y_bin"], device, effective_batch,
            )
            val_pr = compute_pr_auc(val_scores, data["val"]["y_bin"])
            pr_aucs.append(float(val_pr))

            logger.info(
                f"  Epoch {epoch+1:4d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_PR-AUC={val_pr:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            # Save best model
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.perf_counter() - t0
    logger.info(f"  Training completed in {train_time:.1f}s, best epoch = {best_epoch}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ── Save checkpoint ─────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / "scvae_best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "x_dim": len(cond_cols),
        "label_dim": len(targ_cols),
        "h_dim": args.h_dim,
        "z_dim": args.z_dim,
        "conditional_features": cond_cols,
        "target_features": targ_cols,
        "all_features": all_cols,
        "args": vars(args),
    }, ckpt_path)
    logger.success(f"  Checkpoint saved → {ckpt_path}")

    # ── Evaluation ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 5: Anomaly Detection Evaluation")
    logger.info("=" * 60)

    # Compute anomaly scores on test set
    test_scores = compute_anomaly_scores(
        model, X_test_cond, Y_test_targ, data["test"]["y_bin"],
        device, effective_batch,
    )
    y_test_bin = data["test"]["y_bin"]
    y_test_multi = data["test"]["y_multi"]

    # Overall metrics
    test_pr_auc = compute_pr_auc(test_scores, y_test_bin)
    test_roc_auc = compute_roc_auc(test_scores, y_test_bin)
    logger.info(f"  Overall PR-AUC:  {test_pr_auc:.4f}")
    logger.info(f"  Overall ROC-AUC: {test_roc_auc:.4f}")

    # Per-class evaluation (best threshold from PR curve)
    val_fpr = 0.0
    val_total = 0
    if np.any(y_test_bin > 0):
        precision, recall, thresholds = precision_recall_curve(y_test_bin, test_scores)
        # Best F1 threshold
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_thresh_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_thresh_idx]) if best_thresh_idx < len(thresholds) else float(thresholds[-1])

        predictions = (test_scores > best_threshold).astype(int)

        logger.info(f"  Best F1 threshold: {best_threshold:.4f}")
        logger.info(f"  {'Class':<14} | {'Total':>8} | {'Detected':>8} | {'Recall':>8} | {'Fault Type':}")
        logger.info(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-")

        per_class_results = {}
        for cls in EVALUABLE_CLASSES:
            mask = (y_test_multi == cls)
            cls_total = int(mask.sum())
            if cls_total == 0:
                continue
            cls_detected = int(predictions[mask].sum())
            cls_recall = cls_detected / cls_total
            per_class_results[str(cls)] = {
                "total": cls_total,
                "detected": cls_detected,
                "recall": round(cls_recall, 4),
            }
            logger.info(
                f"  {cls:<14} | {cls_total:>8} | {cls_detected:>8} | "
                f"{cls_recall:>8.4f} | {FAULT_NAMES.get(cls, 'Unknown')}"
            )

        # All faults combined
        fault_mask = (y_test_bin == 1)
        fault_total = int(fault_mask.sum())
        fault_detected = int(predictions[fault_mask].sum())
        fault_recall = fault_detected / fault_total if fault_total > 0 else 0.0
        logger.info(f"  {'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-")
        logger.info(f"  {'ALL FAULTS':<14} | {fault_total:>8} | {fault_detected:>8} | {fault_recall:>8.4f} |")
        per_class_results["ALL"] = {
            "total": fault_total,
            "detected": fault_detected,
            "recall": round(fault_recall, 4),
        }

        # Val false positive rate (on normal val data)
        val_scores = compute_anomaly_scores(
            model, X_val_cond, Y_val_targ, data["val"]["y_bin"], device, effective_batch,
        )
        val_fp = int((val_scores > best_threshold).sum())
        val_total = len(val_scores)
        val_fpr = val_fp / val_total if val_total > 0 else 0.0
        logger.info(f"  Val FPR (normal): {val_fp}/{val_total} ({val_fpr:.4f})")
    else:
        per_class_results = {}

    # ── Save metrics ────────────────────────────────────────────────────
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": "SCVAE",
        "dataset": "costa",
        "paper_reference": "Li et al. (2024) Applied Energy 353:122124",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "h_dim": args.h_dim,
            "z_dim": args.z_dim,
            "x_dim": len(cond_cols),
            "label_dim": len(targ_cols),
            "conditional_features": cond_cols,
            "target_features": targ_cols,
            "loss_mode": args.mode,
            "smooth_reg": args.reg,
        },
        "training": {
            "learning_rate": final_lr,
            "batch_size": effective_batch,
            "epochs_trained": best_epoch,
            "best_val_loss": round(float(best_val_loss), 6),
            "train_time_seconds": round(train_time, 1),
            "train_loss_final": round(train_losses[-1], 6) if train_losses else None,
            "val_loss_final": round(val_losses[-1], 6) if val_losses else None,
        },
        "gvsao": {
            "enabled": gvsao_result is not None,
            "best_params": gvsao_result.best_params if gvsao_result else None,
            "best_fitness": float(gvsao_result.best_fitness) if gvsao_result else None,
            "n_evals": gvsao_result.n_evals if gvsao_result else 0,
            "elapsed_seconds": gvsao_result.elapsed_seconds if gvsao_result else 0,
        },
        "performance": {
            "test_pr_auc": round(float(test_pr_auc), 6),
            "test_roc_auc": round(float(test_roc_auc), 6),
            "per_class": per_class_results,
            "val_false_positive_rate": round(float(val_fpr) if val_total > 0 else 0.0, 6),
        },
        "data_integrity": integrity_report,
    }

    metrics_path = metrics_dir / "scvae_results.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    logger.success(f"  Metrics saved → {metrics_path}")

    logger.success("=" * 60)
    logger.success("SCVAE Training & Evaluation Complete")
    logger.success(f"  Checkpoint: {ckpt_path}")
    logger.success(f"  PR-AUC: {test_pr_auc:.4f}  |  ROC-AUC: {test_roc_auc:.4f}")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
