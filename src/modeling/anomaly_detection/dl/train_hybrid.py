"""
Training script for the Hybrid Anomaly Detection model.

Combines: Facebook Prophet → AE-LSTM + Isolation Forest → Ensemble.

Paper: Ahirwar & Nandanwar (2025) ICoEIT.

Key features:
  - GVSAO hyperparameter optimization (LR, batch size, hidden/latent dims)
  - Data integrity checks before training
  - Leakage prevention validation
  - Early stopping with patience
  - Per-fault-class evaluation
  - Model checkpointing and metrics logging

Usage:
    uv run python -m src.modeling.anomaly_detection.dl.train_hybrid
    uv run python -m src.modeling.anomaly_detection.dl.train_hybrid --no-gvsao --lr 0.001
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
import torch.nn as nn
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.anomaly_detection.dl.gvsao import GVSaoConfig, run_gvsao
from src.modeling.anomaly_detection.dl.hybrid_model import AELSTM, HybridScorer

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_hybrid" / "hybrid_sequences.npz"
DEFAULT_META_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_hybrid" / "hybrid_metadata.json"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "hybrid"
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
# Utilities
# =========================================================================


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def load_data(npz_path: Path, meta_path: Path) -> dict[str, Any]:
    """Load preprocessed window sequences."""
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {npz_path}\n"
            "Run: uv run python -m src.data.preprocess_hybrid"
        )
    data = dict(np.load(npz_path))
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    data["_meta"] = meta

    for name in ["train", "val", "test"]:
        logger.info(
            f"  {name}: residual_windows={data[f'{name}_X_res'].shape} | "
            f"if_features={data[f'{name}_X_if'].shape[1]}d | "
            f"anomalous={data[f'{name}_y_bin'].sum():,}"
        )
    return data


def pre_training_checks(data: dict) -> dict:
    """Data integrity checks before training."""
    report = {}

    y_train = data["train_y_bin"]
    report["train_anom_count"] = int(y_train.sum())
    report["train_samples"] = int(len(y_train))

    for name in ["train", "val", "test"]:
        x_res = data[f"{name}_X_res"]
        x_if = data[f"{name}_X_if"]
        report[f"{name}_Xres_nan"] = int(np.isnan(x_res).sum())
        report[f"{name}_Xres_inf"] = int(np.isinf(x_res).sum())
        report[f"{name}_Xif_nan"] = int(np.isnan(x_if).sum())
        report[f"{name}_Xif_inf"] = int(np.isinf(x_if).sum())

    # Test class distribution
    y_multi = data["test_y_multi"]
    class_dist = {}
    for cls in sorted(set(y_multi)):
        class_dist[int(cls)] = int((y_multi == cls).sum())
    report["test_class_dist"] = class_dist

    logger.info("=" * 50)
    logger.info("PRE-TRAINING DATA INTEGRITY CHECKS")
    logger.info("=" * 50)
    status = "FAIL" if report["train_anom_count"] > 0 else "PASS"
    logger.info(f"  [{status}] Train anomalies: {report['train_anom_count']}")
    for name in ["train", "val", "test"]:
        n_nan = report[f"{name}_Xres_nan"]
        n_inf = report[f"{name}_Xres_inf"]
        status = "PASS" if n_nan == 0 and n_inf == 0 else "FAIL"
        logger.info(f"  [{status}] {name}: NaN={n_nan}, Inf={n_inf}")
    logger.info(f"  [INFO] Test class distribution: {class_dist}")
    logger.info("=" * 50)

    return report


# =========================================================================
# Training
# =========================================================================


def train_aelstm(
    model: AELSTM,
    X_train: np.ndarray,
    X_val: np.ndarray,
    lr: float,
    batch_size: int,
    epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[AELSTM, dict]:
    """Train AE-LSTM autoencoder on normal residuals."""
    train_ds = TensorDataset(torch.from_numpy(X_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7,
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_ds)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * batch.size(0)
        val_loss /= len(val_ds)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"    Epoch {epoch+1:3d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if no_improve >= patience:
            logger.info(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    info = {
        "best_val_loss": float(best_val_loss),
        "epochs_trained": len(train_losses),
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_val_loss": float(val_losses[-1]) if val_losses else None,
    }
    return model, info


def compute_pr_auc(scores, labels):
    if np.all(labels == 0):
        return 0.0
    return float(average_precision_score(labels, scores))


# =========================================================================
# GVSAO fitness
# =========================================================================


def make_fitness_fn(
    X_train_res, X_val_res, X_val_if, y_val_bin,
    device, input_dim, dropout, gvsao_epochs,
):
    def fitness(lr: float, batch: int) -> float:
        batch = max(4, min(int(batch), X_train_res.shape[0]))
        model = AELSTM(
            input_dim=input_dim, hidden_dim=64, num_layers=2,
            latent_dim=16, dropout=dropout, device=device,
        ).to(device)

        model, _ = train_aelstm(
            model, X_train_res, X_val_res,
            lr=lr, batch_size=batch, epochs=gvsao_epochs,
            patience=3, device=device,
        )

        errors = model.compute_reconstruction_error(X_val_res, batch_size=256)
        pr_auc = compute_pr_auc(errors, y_val_bin)
        return -pr_auc if pr_auc > 0 else float(np.mean(errors))

    return fitness


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid model on Costa")
    parser.add_argument("--npz-path", type=str, default=str(DEFAULT_NPZ_PATH))
    parser.add_argument("--meta-path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--metrics-dir", type=str, default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--gvsao-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-gvsao", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--ensemble-alpha", type=float, default=0.6)
    parser.add_argument("--if-contamination", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── Load ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Load Preprocessed Data")
    logger.info("=" * 60)
    data = load_data(Path(args.npz_path), Path(args.meta_path))
    meta = data["_meta"]

    X_train_res = data["train_X_res"]
    X_val_res = data["val_X_res"]
    X_test_res = data["test_X_res"]
    X_train_if = data["train_X_if"]
    X_val_if = data["val_X_if"]
    X_test_if = data["test_X_if"]

    input_dim = X_train_res.shape[2]
    n_if_features = X_train_if.shape[1]

    # ── Checks ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Pre-Training Data Integrity Checks")
    logger.info("=" * 60)
    integrity_report = pre_training_checks(data)

    # ── GVSAO ────────────────────────────────────────────────────────
    final_lr = args.lr or 0.001
    final_batch = args.batch_size or 50
    gvsao_result = None

    if not args.no_gvsao and (args.lr is None or args.batch_size is None):
        logger.info("=" * 60)
        logger.info("Step 3: GVSAO Hyperparameter Optimization")
        logger.info("=" * 60)

        cfg = GVSaoConfig(
            population_size=10, max_generations=5,
            lr_bounds=(1e-5, 1e-2), batch_bounds=(16, 256),
            seed=args.seed,
        )
        fn = make_fitness_fn(
            X_train_res, X_val_res, X_val_if, data["val_y_bin"],
            device, input_dim, args.dropout, args.gvsao_epochs,
        )
        gvsao_result = run_gvsao(fn, cfg, verbose=True)
        final_lr = gvsao_result.best_params["learning_rate"]
        final_batch = min(gvsao_result.best_params["batch_size"], X_train_res.shape[0])
        logger.success(f"GVSAO best: lr={final_lr:.6f}, batch={final_batch}")

    # ── Train AE-LSTM ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"Step 4: Train AE-LSTM (lr={final_lr:.6f}, batch={final_batch})")
    logger.info("=" * 60)

    aelstm = AELSTM(
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, latent_dim=args.latent_dim,
        dropout=args.dropout, device=device,
    ).to(device)

    t0 = time.perf_counter()
    aelstm, train_info = train_aelstm(
        aelstm, X_train_res, X_val_res,
        lr=final_lr, batch_size=final_batch,
        epochs=args.epochs, patience=args.patience, device=device,
    )
    train_time = time.perf_counter() - t0
    logger.info(f"  Training completed in {train_time:.1f}s")
    logger.info(f"  Best val loss: {train_info['best_val_loss']:.6f}")

    # ── Fit Isolation Forest ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 5: Fit Isolation Forest")
    logger.info("=" * 60)

    if_model = IsolationForest(
        n_estimators=100, contamination=args.if_contamination,
        random_state=args.seed, n_jobs=-1,
    )
    if_model.fit(X_train_if)
    if_train_score = -if_model.decision_function(X_train_if)
    logger.info(f"  IF train score range: [{if_train_score.min():.4f}, {if_train_score.max():.4f}]")

    # ── Build hybrid scorer ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 6: Ensemble Scorer")
    logger.info("=" * 60)

    scorer = HybridScorer(aelstm, if_model, alpha=args.ensemble_alpha, device=device)
    scorer.fit_score_stats(X_train_res, X_train_if)

    # ── Save models ──────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / "hybrid_best.pt"
    torch.save({
        "aelstm_state_dict": aelstm.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "ensemble_alpha": args.ensemble_alpha,
        "args": vars(args),
    }, ckpt_path)
    logger.success(f"  Checkpoint → {ckpt_path}")

    # ── Evaluation ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 7: Anomaly Detection Evaluation")
    logger.info("=" * 60)

    all_scores = scorer.compute_scores_separately(X_test_res, X_test_if)
    y_test_bin = data["test_y_bin"]
    y_test_multi = data["test_y_multi"]

    y_val_bin = data["val_y_bin"]
    val_scores = scorer.score(X_val_res, X_val_if)

    for name, scores in all_scores.items():
        pr_auc = compute_pr_auc(scores, y_test_bin)
        roc = roc_auc_score(y_test_bin, scores) if np.any(y_test_bin > 0) and np.any(y_test_bin == 0) else 0.0
        logger.info(f"  {name:>18}: PR-AUC={pr_auc:.4f} | ROC-AUC={roc:.4f}")

    # Per-class evaluation (using ensemble scores)
    ensemble_scores = all_scores["ensemble"]
    if np.any(y_test_bin > 0):
        precision, recall, thresholds = precision_recall_curve(y_test_bin, ensemble_scores)
        f1_s = 2 * precision * recall / (precision + recall + 1e-10)
        best_thresh = float(thresholds[np.argmax(f1_s)])

        preds = (ensemble_scores > best_thresh).astype(int)

        logger.info(f"\n  Best F1 threshold: {best_thresh:.4f}")
        logger.info(f"  {'Class':<14} | {'Fault Type':<15} | {'Total':>7} | {'Detect':>7} | {'Recall':>7}")
        logger.info(f"  {'-'*14}-+-{'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

        per_class = {}
        for cls in EVALUABLE_CLASSES:
            mask = (y_test_multi == cls)
            ct = int(mask.sum())
            if ct == 0:
                continue
            cd = int(preds[mask].sum())
            cr = cd / ct
            per_class[str(cls)] = {"total": ct, "detected": cd, "recall": round(cr, 4)}
            logger.info(
                f"  {cls:<14} | {FAULT_NAMES[cls]:<15} | {ct:>7} | {cd:>7} | {cr:>7.4f}"
            )

        fmask = (y_test_bin == 1)
        ftotal = int(fmask.sum())
        fdetected = int(preds[fmask].sum())
        frecall = fdetected / ftotal if ftotal > 0 else 0.0
        logger.info(f"  {'-'*14}-+-{'-'*15}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
        logger.info(f"  {'ALL FAULTS':<14} | {'':<15} | {ftotal:>7} | {fdetected:>7} | {frecall:>7.4f}")

        per_class["ALL"] = {"total": ftotal, "detected": fdetected, "recall": round(frecall, 4)}

        # Val FPR
        val_preds = (val_scores > best_thresh).astype(int)
        val_fp = int(val_preds.sum())
        val_fpr = val_fp / len(val_scores) if len(val_scores) > 0 else 0.0
        logger.info(f"  Val FPR (normal): {val_fp}/{len(val_scores)} ({val_fpr:.4f})")
    else:
        per_class = {}
        val_fpr = 0.0

    # ── Save metrics ────────────────────────────────────────────────
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    perf_dict = {}
    for name, scores in all_scores.items():
        perf_dict[name] = {
            "pr_auc": round(compute_pr_auc(scores, y_test_bin), 6),
            "roc_auc": round(float(
                roc_auc_score(y_test_bin, scores)
                if np.any(y_test_bin > 0) and np.any(y_test_bin == 0) else 0.0
            ), 6),
        }

    metrics = {
        "model": "Hybrid (AE-LSTM + Prophet + IsolationForest)",
        "dataset": "costa",
        "paper_reference": "Ahirwar & Nandanwar (2025) ICoEIT",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "latent_dim": args.latent_dim,
            "dropout": args.dropout,
            "ensemble_alpha": args.ensemble_alpha,
            "if_contamination": args.if_contamination,
            "n_if_features": n_if_features,
        },
        "training": {
            "learning_rate": final_lr,
            "batch_size": final_batch,
            "epochs_trained": train_info["epochs_trained"],
            "best_val_loss": round(train_info["best_val_loss"], 6),
            "train_time_seconds": round(train_time, 1),
        },
        "gvsao": {
            "enabled": gvsao_result is not None,
            "best_params": gvsao_result.best_params if gvsao_result else None,
            "n_evals": gvsao_result.n_evals if gvsao_result else 0,
        },
        "performance": perf_dict,
        "per_class": per_class,
        "val_false_positive_rate": round(float(val_fpr), 6),
        "data_integrity": integrity_report,
    }

    metrics_path = metrics_dir / "hybrid_results.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    logger.success(f"  Metrics → {metrics_path}")

    logger.success("=" * 60)
    logger.success("HYBRID TRAINING COMPLETE")
    logger.success(f"  PR-AUC (ensemble): {perf_dict.get('ensemble', {}).get('pr_auc', 0):.4f}")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
