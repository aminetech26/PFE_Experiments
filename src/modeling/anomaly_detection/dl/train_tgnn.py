"""
Training script for the Temporal GNN (T-GNN) on the Costa PV Fault Dataset.

Implements the full pipeline from the paper:
  1. Load and normalize Costa data
  2. Build graph structure (adjacency matrix)
  3. Create sliding-window datasets
  4. Train T-GNN (GCN + GRU + FC) with MSE loss, Adam optimizer
  5. Evaluate with MAE on test set
  6. Anomaly detection via Z-score of residuals + IQR thresholding

Usage:
    python -m src.modeling.anomaly_detection.dl.train_tgnn
    python -m src.modeling.anomaly_detection.dl.train_tgnn --epochs 200 --lr 0.005 --seq-len 20
    python -m src.modeling.anomaly_detection.dl.train_tgnn --graph-type causal --gcn-dim 64

Reference:
    Mukherjee et al., "Temporal Graph Neural Networks for Early Anomaly Detection
    and Performance Prediction via PV System Monitoring Data", EUPVSEC 2025.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from src.modeling.anomaly_detection.dl.tgnn import (
    COSTA_INPUT_NODES,
    COSTA_TARGET_COL,
    CostaGraphDataset,
    TemporalGNN,
    build_adjacency_matrix,
    build_causal_adjacency,
    load_costa_for_tgnn,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "tgnn"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "experiments" / "metrics"


# ============================================================================
# ANOMALY DETECTION (per paper: Z-score + IQR thresholding)
# ============================================================================


def detect_anomalies_zscore_iqr(
    residuals: np.ndarray,
) -> dict:
    """
    Detect anomalies using Z-score of residuals with IQR-based thresholds.

    Per the paper:
      - Compute residual e = |actual - predicted|
      - Z-score: z = (e - \u03bc_e) / \u03c3_e
      - IQR method: lower = Q1 - 1.5*IQR, upper = Q3 + 1.5*IQR
      - Points outside [lower, upper] are anomalies.

    Returns dict with anomaly indices, thresholds, and statistics.
    """
    abs_errors = np.abs(residuals)

    # Z-score
    mu_e = abs_errors.mean()
    sigma_e = abs_errors.std()
    if sigma_e < 1e-12:
        sigma_e = 1.0
    z_scores = (abs_errors - mu_e) / sigma_e

    # IQR thresholds
    q1 = np.percentile(z_scores, 25)
    q3 = np.percentile(z_scores, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    anomaly_mask = (z_scores < lower_bound) | (z_scores > upper_bound)
    anomaly_indices = np.where(anomaly_mask)[0]

    return {
        "n_anomalies": int(anomaly_mask.sum()),
        "anomaly_fraction": float(anomaly_mask.mean()),
        "anomaly_indices": anomaly_indices.tolist(),
        "z_score_mean": float(z_scores.mean()),
        "z_score_std": float(z_scores.std()),
        "iqr_lower": float(lower_bound),
        "iqr_upper": float(upper_bound),
        "residual_mean": float(mu_e),
        "residual_std": float(sigma_e),
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================


def train_one_epoch(
    model: TemporalGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    adj_norm: torch.Tensor,
    device: torch.device,
) -> float:
    """Train for one epoch, return average MSE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch, adj_norm).squeeze(-1)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: TemporalGNN,
    loader: DataLoader,
    adj_norm: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a DataLoader.

    Returns:
        mae: Mean Absolute Error (on normalized scale)
        predictions: array of predicted values
        actuals: array of actual values
    """
    model.eval()
    all_preds = []
    all_actuals = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_pred = model(x_batch, adj_norm).squeeze(-1)

        all_preds.append(y_pred.cpu().numpy())
        all_actuals.append(y_batch.numpy())

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    mae = float(np.mean(np.abs(predictions - actuals)))

    return mae, predictions, actuals


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train T-GNN on Costa PV dataset (Mukherjee et al. architecture)"
    )

    # Data args
    parser.add_argument(
        "--parquet-path", type=str, default=None,
        help="Path to Costa ingested parquet. Default: auto-detect."
    )
    parser.add_argument(
        "--seq-len", type=int, default=10,
        help="Temporal window length (number of time steps per sample)."
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Stride for sliding window."
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2,
        help="Fraction of data for testing (paper uses 80/20)."
    )
    parser.add_argument(
        "--irr-threshold", type=float, default=50.0,
        help="Minimum irradiance for daytime filtering (W/m\u00b2)."
    )

    # Model architecture args
    parser.add_argument(
        "--gcn-dim", type=int, default=32,
        help="GCN hidden dimension."
    )
    parser.add_argument(
        "--gru-dim", type=int, default=64,
        help="GRU hidden dimension."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout rate."
    )
    parser.add_argument(
        "--graph-type", type=str, default="full",
        choices=["full", "causal"],
        help="Graph topology: 'full' (fully connected) or 'causal' (domain-knowledge edges)."
    )

    # Training args
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (paper: 100).")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (paper: 0.01).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (0=disabled).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Output args
    parser.add_argument(
        "--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory to save model checkpoints."
    )
    parser.add_argument(
        "--metrics-dir", type=str, default=str(DEFAULT_METRICS_DIR),
        help="Directory to save metrics JSON."
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")

    args = parser.parse_args()

    # ---- Seed ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}", device)

    # ---- Load data ----
    logger.info("Loading Costa dataset for T-GNN ...")
    data, targets, metadata = load_costa_for_tgnn(
        parquet_path=args.parquet_path,
        daytime_irr_threshold=args.irr_threshold,
    )
    n_samples = metadata["n_samples"]
    input_nodes = metadata["input_nodes"]
    num_nodes = len(input_nodes)
    logger.info(
        "Data loaded | samples={:,} nodes={} features={}",
        n_samples, num_nodes, input_nodes,
    )

    # ---- Train/Test split (80/20 random, per paper) ----
    n_test = int(n_samples * args.test_split)
    n_train = n_samples - n_test

    indices = np.arange(n_samples)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)

    train_idx = np.sort(indices[:n_train])  # sort to maintain temporal order within sets
    test_idx = np.sort(indices[n_train:])

    train_data, train_targets = data[train_idx], targets[train_idx]
    test_data, test_targets = data[test_idx], targets[test_idx]

    logger.info("Split | train={:,} test={:,}", len(train_data), len(test_data))

    # ---- Build datasets ----
    train_dataset = CostaGraphDataset(train_data, train_targets, seq_len=args.seq_len, stride=args.stride)
    test_dataset = CostaGraphDataset(test_data, test_targets, seq_len=args.seq_len, stride=args.stride)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(
        "Datasets | train_windows={:,} test_windows={:,} seq_len={} stride={}",
        len(train_dataset), len(test_dataset), args.seq_len, args.stride,
    )

    # ---- Build adjacency matrix ----
    if args.graph_type == "causal":
        adj_norm = build_causal_adjacency(input_nodes).to(device)
        logger.info("Graph type: causal (domain-knowledge directed edges)")
    else:
        adj_norm = build_adjacency_matrix(num_nodes).to(device)
        logger.info("Graph type: fully connected")

    # ---- Build model ----
    model = TemporalGNN(
        num_nodes=num_nodes,
        node_feature_dim=1,
        gcn_hidden_dim=args.gcn_dim,
        gru_hidden_dim=args.gru_dim,
        output_dim=1,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model | total_params={:,} trainable={:,}", total_params, trainable_params)
    logger.info("Architecture:\n{}", model)

    # ---- Optimizer and loss (per paper: MSE + Adam, lr=0.01) ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    # ---- Training loop ----
    logger.info("=" * 60)
    logger.info("Training T-GNN | epochs={} lr={} batch_size={}", args.epochs, args.lr, args.batch_size)
    logger.info("=" * 60)

    best_test_mae = float("inf")
    best_epoch = 0
    patience_counter = 0
    training_history: list[dict] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, adj_norm, device)

        # Evaluate
        test_mae, test_preds, test_actuals = evaluate(model, test_loader, adj_norm, device)
        train_mae, _, _ = evaluate(model, train_loader, adj_norm, device)

        epoch_time = time.time() - epoch_start

        training_history.append({
            "epoch": epoch,
            "train_loss_mse": float(train_loss),
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "epoch_time_s": float(epoch_time),
        })

        # Logging
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            logger.info(
                "Epoch {:>3d}/{} | train_loss={:.6f} train_mae={:.4f} test_mae={:.4f} ({:.1f}s)",
                epoch, args.epochs, train_loss, train_mae, test_mae, epoch_time,
            )

        # Best model tracking
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = checkpoint_dir / "tgnn_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_mae": test_mae,
                "train_loss": train_loss,
                "args": vars(args),
                "metadata": {k: v for k, v in metadata.items() if k != "labels"},
            }, best_model_path)
        else:
            patience_counter += 1

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch {} | best_mae={:.4f} at epoch {}",
                epoch, best_test_mae, best_epoch,
            )
            break

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(
        "Training complete | best_test_mae={:.4f} at epoch {} | total_time={:.1f}s",
        best_test_mae, best_epoch, total_time,
    )

    # ---- Load best model for final evaluation ----
    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # ---- Final evaluation on test set ----
    final_mae, final_preds, final_actuals = evaluate(model, test_loader, adj_norm, device)
    residuals = final_actuals - final_preds

    # Compute MSE on test
    final_mse = float(np.mean(residuals ** 2))

    # Denormalize for reporting
    target_min = metadata["target_min"]
    target_range = metadata["target_range"]
    preds_original = final_preds * target_range + target_min
    actuals_original = final_actuals * target_range + target_min
    mae_original = float(np.mean(np.abs(actuals_original - preds_original)))

    logger.info("=" * 60)
    logger.info("FINAL TEST RESULTS")
    logger.info("  MAE (normalized) : {:.4f}", final_mae)
    logger.info("  MSE (normalized) : {:.6f}", final_mse)
    logger.info("  MAE (original W) : {:.2f} W", mae_original)
    logger.info("=" * 60)

    # ---- Anomaly Detection (Z-score + IQR, per paper) ----
    logger.info("Running anomaly detection on test residuals ...")
    anomaly_results = detect_anomalies_zscore_iqr(residuals)
    logger.info(
        "Anomalies detected: {} / {} ({:.2f}%)",
        anomaly_results["n_anomalies"],
        len(residuals),
        anomaly_results["anomaly_fraction"] * 100,
    )
    logger.info(
        "  IQR bounds: [{:.4f}, {:.4f}]",
        anomaly_results["iqr_lower"],
        anomaly_results["iqr_upper"],
    )

    # ---- Save metrics ----
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "tgnn_results.json"

    metrics_payload = {
        "model": "TemporalGNN",
        "paper": "Mukherjee et al., EUPVSEC 2025 (arXiv:2512.03114v1)",
        "dataset": "Costa PV Fault Dataset",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "num_nodes": num_nodes,
            "input_nodes": input_nodes,
            "target": COSTA_TARGET_COL,
            "gcn_hidden_dim": args.gcn_dim,
            "gru_hidden_dim": args.gru_dim,
            "dropout": args.dropout,
            "graph_type": args.graph_type,
            "seq_len": args.seq_len,
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "training": {
            "epochs_run": len(training_history),
            "epochs_max": args.epochs,
            "best_epoch": best_epoch,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "loss": "MSE",
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "total_time_s": total_time,
        },
        "results": {
            "test_mae_normalized": final_mae,
            "test_mse_normalized": final_mse,
            "test_mae_watts": mae_original,
        },
        "anomaly_detection": {
            k: v for k, v in anomaly_results.items() if k != "anomaly_indices"
        },
        "training_history": training_history,
    }

    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.success("Metrics saved \u2192 {}", metrics_path)
    logger.success("Best model saved \u2192 {}", best_model_path)


if __name__ == "__main__":
    main()
