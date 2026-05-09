#!/usr/bin/env python3
"""
GTBAD Training & Evaluation — Costa dataset.

Implements the GVSAO-Transformer-BiLSTM anomaly detection approach from:
  Zhu, Ma, Xu, Xu, Du (2026) "GTBAD: GVSAO-Transformer-BiLSTM-based
  time-series anomaly detection for photovoltaic power generation"
  Applied Intelligence 56:140.

Adaptations for Costa (16-day, 1 Hz dataset):
  - Resampled to 1-minute intervals (mean aggregation)
  - Window length = 1 (point-wise reconstruction, no historical windows)
  - No multi-period positional encoding (dataset too short for seasonal patterns)
  - No Savitzky-Golay smoothing (1-min resampling already denoises)
  - No correlation-based feature screening (9 raw sensor features kept)
  - Trained exclusively on healthy data (label == 0)
  - Evaluated per fault class (1=ShortCircuit, 2=Degradation, 3=OpenCircuit, 4=Shadowing)
  - GVSAO offline hyperparameter optimisation (learning rate, batch size)

No data leakage: MinMax scaler fitted on training data only; threshold computed
from training reconstruction errors; strict temporal train/val separation.

Usage:
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad --no-gvsao
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad --epochs 100 --lr 0.001
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.anomaly_detection.dl.gtbad_model import GTBADModel, reconstruction_error
from src.modeling.anomaly_detection.dl.gvsao import GVSaoConfig, GVSaoResult, run_gvsao

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "ingestion" / "costa" / "costa_merged.parquet"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "gtbad"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "experiments" / "metrics"

SENSOR_COLS = ["vdc1", "vdc2", "idc1", "idc2", "pdc1", "pdc2", "pdc", "irr", "pvt"]

FAULT_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}

EVALUABLE_CLASSES = [1, 2, 3, 4]

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def load_and_resample(parquet_path: str | Path, resample_minutes: int = 1) -> pd.DataFrame:
    """Load ingested Costa data and resample to 1-minute intervals.

    Sensor columns are mean-aggregated. Label is max-aggregated (any fault
    within the minute window marks the whole minute as faulty).

    Returns DataFrame with timestamp index.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Ingested parquet not found: {parquet_path}\nRun: uv run python -m src.data.ingestion --dataset costa")

    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    logger.info(f"  Loaded: {len(df):,} rows, {df.shape[1]} columns")

    # Resample to 1-minute
    resample_rule = f"{resample_minutes}min"
    sensor_cols_present = [c for c in SENSOR_COLS if c in df.columns]
    sensor_df = df[sensor_cols_present].resample(resample_rule).mean()
    label_df = df["label"].resample(resample_rule).max()
    df_resampled = pd.concat([sensor_df, label_df], axis=1)
    df_resampled = df_resampled.dropna(subset=sensor_cols_present)
    df_resampled["label"] = df_resampled["label"].fillna(0).astype(int)

    logger.info(f"  After {resample_minutes}-min resample: {len(df_resampled):,} rows")
    logger.info("  Label distribution after resample:")
    for lbl, cnt in df_resampled["label"].value_counts().sort_index().items():
        logger.info(f"    {FAULT_NAMES.get(int(lbl), '?')} ({int(lbl)}): {cnt:,}")

    return df_resampled


def split_healthy_faulty(
    df: pd.DataFrame,
    healthy_train_frac: float = 0.80,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split data into train (healthy), val (healthy), and test (faulty per class).

    TRAIN: first healthy_train_frac of healthy data (label == 0)
    VAL:   remaining healthy data
    TEST:  all faulty data, grouped by fault class

    Temporal order is preserved within each split.
    """
    healthy = df[df["label"] == 0].copy()
    faulty = df[df["label"] > 0].copy()

    n_healthy = len(healthy)
    split_idx = int(n_healthy * healthy_train_frac)

    train_df = healthy.iloc[:split_idx]
    val_df = healthy.iloc[split_idx:]

    # Group faulty data by class for per-class evaluation
    test_by_class: dict[str, pd.DataFrame] = {}
    for cls in EVALUABLE_CLASSES:
        cls_data = faulty[faulty["label"] == cls]
        if len(cls_data) > 0:
            test_by_class[f"fault_class_{cls}"] = cls_data

    logger.info(f"  Train (healthy): {len(train_df):,}")
    logger.info(f"  Val   (healthy): {len(val_df):,}")
    for name, cls_df in test_by_class.items():
        logger.info(f"  Test  ({name}): {len(cls_df):,}")

    return {
        "train": train_df,
        "val": val_df,
        **test_by_class,
    }


class MinMaxScaler:
    """MinMax scaler that stores fit params for later transform."""

    def fit(self, data: np.ndarray) -> MinMaxScaler:
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ < 1e-10] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min_) / self.range_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)


def prepare_tensors(
    splits: dict[str, pd.DataFrame],
    scaler: MinMaxScaler | None = None,
    window_length: int = 1,
) -> dict[str, Any]:
    """MinMax-normalize features and create windowed tensors.

    Scaler is fitted on TRAIN data only (leakage prevention).
    Window length = 1 for point-wise reconstruction.
    """
    sensor_cols_present = [c for c in SENSOR_COLS if c in splits["train"].columns]
    n_features = len(sensor_cols_present)

    # Fit scaler on TRAIN data only
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(splits["train"][sensor_cols_present].values)

    result: dict[str, Any] = {"scaler": scaler, "n_features": n_features, "feature_names": sensor_cols_present}

    for split_name, df in splits.items():
        X = scaler.transform(df[sensor_cols_present].values).astype(np.float32)
        labels = df["label"].values.astype(np.int32)

        if window_length == 1:
            X_windows = X[:, np.newaxis, :]  # (n_samples, 1, n_features)
            l_windows = labels
        else:
            n = len(X)
            if n < window_length:
                raise ValueError(f"Not enough data ({n}) for window length {window_length} in '{split_name}'")
            X_windows = np.stack([X[i:i + window_length] for i in range(n - window_length + 1)])
            l_windows = labels[window_length - 1:]
            l_windows = np.max([labels[i:i + window_length] for i in range(n - window_length + 1)], axis=1)

        result[split_name] = {
            "X": torch.from_numpy(X_windows),
            "labels": l_windows,
            "n_samples": len(X_windows),
        }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────────────────────────────────────


def _make_dataloader(
    X: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(X, X)  # autoencoder: input == target
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, _ in dataloader:
        X_batch = X_batch.to(device)
        optimizer.zero_grad()
        recon = model(X_batch)
        loss = criterion(recon, X_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_reconstruction(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute per-sample MSE reconstruction error."""
    model.eval()
    all_errors: list[np.ndarray] = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end].to(device)
        recon = model(X_batch)
        err = reconstruction_error(X_batch, recon).cpu().numpy()
        all_errors.append(err)
    return np.concatenate(all_errors)


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    device: torch.device,
    lr: float,
    batch_size: int,
    epochs: int,
    patience: int = 15,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """Train GTBAD model on healthy data only with early stopping."""
    train_loader = _make_dataloader(X_train, batch_size, shuffle=True)
    val_loader = _make_dataloader(X_val, batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                recon = model(X_batch)
                val_loss += criterion(recon, X_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            logger.info(f"    Epoch {epoch+1:3d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if epochs_without_improvement >= patience:
            if verbose:
                logger.info(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    training_info = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val_loss),
        "epochs_trained": len(train_losses),
    }
    return model, training_info


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detection & Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def compute_threshold(errors: np.ndarray, percentile: float = 95.0) -> float:
    """Compute anomaly threshold as p-th percentile of training errors."""
    return float(np.percentile(errors, percentile))


def evaluate_anomaly_detection(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute precision, recall, F1 for anomaly detection."""
    preds = (errors > threshold).astype(int)
    true = (labels > 0).astype(int)

    tp = int(np.sum((preds == 1) & (true == 1)))
    fp = int(np.sum((preds == 1) & (true == 0)))
    fn = int(np.sum((preds == 0) & (true == 1)))
    tn = int(np.sum((preds == 0) & (true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def build_model(n_features: int, d_model: int, nhead: int, num_encoder_layers: int, lstm_hidden: int, dropout: float) -> GTBADModel:
    return GTBADModel(
        input_dim=n_features,
        output_dim=n_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        lstm_hidden=lstm_hidden,
        dropout=dropout,
    )


def main():
    parser = argparse.ArgumentParser(description="Train GTBAD anomaly detection on Costa dataset")
    parser.add_argument("--parquet-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda:0")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (final model)")
    parser.add_argument("--gvsao-epochs", type=int, default=5, help="Epochs per GVSAO fitness eval")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (if set, skips GVSAO)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (if set, skips GVSAO)")
    parser.add_argument("--no-gvsao", action="store_true", help="Skip GVSAO, use defaults or --lr/--batch-size")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--lstm-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--window-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Seed ──────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Window length: {args.window_length}")

    # ── Load & Resample ───────────────────────────────────────────────────
    logger.info("=== Step 1: Load & Resample Costa Data ===")
    df = load_and_resample(args.parquet_path)

    logger.info("=== Step 2: Split (healthy train/val, faulty test) ===")
    splits = split_healthy_faulty(df, healthy_train_frac=args.train_frac, seed=args.seed)

    logger.info("=== Step 3: Normalize & Create Tensors ===")
    tensors = prepare_tensors(splits, window_length=args.window_length)
    n_features = tensors["n_features"]
    feature_names = tensors["feature_names"]

    X_train = tensors["train"]["X"]
    X_val = tensors["val"]["X"]
    logger.info(f"  n_features: {n_features} ({feature_names})")
    logger.info(f"  X_train: {tuple(X_train.shape)} | X_val: {tuple(X_val.shape)}")

    # ── GVSAO Hyperparameter Optimisation ─────────────────────────────────
    final_lr = args.lr or 0.001
    final_batch_size = args.batch_size or 32
    gvsao_result = None

    if not args.no_gvsao and (args.lr is None or args.batch_size is None):
        logger.info("=== Step 4: GVSAO Hyperparameter Tuning ===")

        gvsao_config = GVSaoConfig(
            population_size=10,
            max_generations=5,
            lr_bounds=(1e-5, 1e-1),
            batch_bounds=(16, 128),
            seed=args.seed,
        )

        def fitness_fn(lr: float, batch: int) -> float:
            """Train lightweight model, return validation loss."""
            model = build_model(
                n_features, args.d_model, args.nhead,
                args.num_encoder_layers, args.lstm_hidden, args.dropout,
            ).to(device)
            effective_batch = min(batch, X_train.shape[0])
            _, info = train_model(
                model, X_train, X_val, device,
                lr=lr, batch_size=effective_batch,
                epochs=args.gvsao_epochs, patience=3, verbose=False,
            )
            return info["best_val_loss"]

        gvsao_result = run_gvsao(fitness_fn, gvsao_config, verbose=True)
        final_lr = gvsao_result.best_params["learning_rate"]
        final_batch_size = gvsao_result.best_params["batch_size"]
        final_batch_size = min(final_batch_size, X_train.shape[0])
        logger.success(f"GVSAO best: lr={final_lr:.6f}, batch={final_batch_size}")

    # ── Final Model Training ──────────────────────────────────────────────
    logger.info(f"=== Step 5: Train GTBAD (lr={final_lr:.6f}, batch={final_batch_size}) ===")
    model = build_model(
        n_features, args.d_model, args.nhead,
        args.num_encoder_layers, args.lstm_hidden, args.dropout,
    ).to(device)

    t0 = time.perf_counter()
    model, training_info = train_model(
        model, X_train, X_val, device,
        lr=final_lr, batch_size=final_batch_size,
        epochs=args.epochs, patience=args.patience, verbose=True,
    )
    train_time = time.perf_counter() - t0
    logger.info(f"  Training completed in {train_time:.1f}s")

    # ── Checkpoint ────────────────────────────────────────────────────────
    checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "gtbad_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "feature_names": feature_names,
        "scaler_min": tensors["scaler"].min_.tolist(),
        "scaler_max": tensors["scaler"].max_.tolist(),
        "args": vars(args),
    }, ckpt_path)
    logger.success(f"  Saved checkpoint → {ckpt_path}")

    # ── Anomaly Detection Evaluation ──────────────────────────────────────
    logger.info("=== Step 6: Anomaly Detection Evaluation ===")

    # Compute threshold from training reconstruction errors
    train_errors = evaluate_reconstruction(model, X_train, device, final_batch_size)
    threshold = compute_threshold(train_errors, args.threshold_percentile)
    logger.info(f"  Anomaly threshold ({args.threshold_percentile}th pctl): {threshold:.6f}")

    # Evaluate on validation healthy data (should have low anomaly rate)
    val_errors = evaluate_reconstruction(model, X_val, device, final_batch_size)
    val_fp = int(np.sum(val_errors > threshold))
    val_result = evaluate_anomaly_detection(val_errors, tensors["val"]["labels"], threshold)
    logger.info(f"  Val (healthy): FP={val_fp}/{len(val_errors)} ({100*val_fp/len(val_errors):.2f}%)")

    # Evaluate on each fault class
    class_results: dict[str, dict] = {}
    for cls in EVALUABLE_CLASSES:
        key = f"fault_class_{cls}"
        if key not in tensors:
            logger.warning(f"  No data for fault class {cls}")
            continue

        X_fault = tensors[key]["X"]
        labels_fault = tensors[key]["labels"]
        fault_errors = evaluate_reconstruction(model, X_fault, device, final_batch_size)
        result = evaluate_anomaly_detection(fault_errors, labels_fault, threshold)
        class_results[key] = result
        logger.info(
            f"  {FAULT_NAMES[cls]:<14} "
            f"Precision={result['precision']:.4f} "
            f"Recall={result['recall']:.4f} "
            f"F1={result['f1_score']:.4f} "
            f"TP={result['TP']} FP={result['FP']} FN={result['FN']}"
        )

    # ── Overall (all fault classes combined) ───────────────────────────────
    all_fault_errors = []
    all_fault_labels = []
    for cls in EVALUABLE_CLASSES:
        key = f"fault_class_{cls}"
        if key in tensors:
            X_f = tensors[key]["X"]
            errs_f = evaluate_reconstruction(model, X_f, device, final_batch_size)
            all_fault_errors.append(errs_f)
            all_fault_labels.append(tensors[key]["labels"])
    if all_fault_errors:
        combined_errors = np.concatenate(all_fault_errors)
        combined_labels = np.concatenate(all_fault_labels)
        combined_val = np.concatenate([val_errors, combined_errors])
        combined_label = np.concatenate([np.zeros(len(val_errors)), combined_labels])
        overall_result = evaluate_anomaly_detection(combined_val, combined_label, threshold)
        logger.info(
            f"  {'OVERALL':<14} "
            f"Precision={overall_result['precision']:.4f} "
            f"Recall={overall_result['recall']:.4f} "
            f"F1={overall_result['f1_score']:.4f} "
            f"TP={overall_result['TP']} FP={overall_result['FP']} FN={overall_result['FN']}"
        )

    # ── Save Results ───────────────────────────────────────────────────────
    metrics_dir = Path(DEFAULT_METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "model": "GTBAD (GVSAO-Transformer-BiLSTM)",
        "dataset": "Costa PV Fault Dataset",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "window_length": args.window_length,
        "input_features": feature_names,
        "model_config": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_encoder_layers": args.num_encoder_layers,
            "lstm_hidden": args.lstm_hidden,
            "dropout": args.dropout,
        },
        "training": {
            "learning_rate": final_lr,
            "batch_size": final_batch_size,
            "epochs": training_info["epochs_trained"],
            "best_val_loss": training_info["best_val_loss"],
            "train_time_seconds": round(train_time, 1),
        },
        "gvsao": {
            "enabled": gvsao_result is not None,
            "best_params": gvsao_result.best_params if gvsao_result else None,
            "best_fitness": gvsao_result.best_fitness if gvsao_result else None,
            "history": gvsao_result.history if gvsao_result else None,
            "n_evals": gvsao_result.n_evals if gvsao_result else 0,
        },
        "anomaly_detection": {
            "threshold_percentile": args.threshold_percentile,
            "threshold_value": threshold,
            "val_healthy_fp_rate": float(val_fp / len(val_errors)) if len(val_errors) > 0 else 0.0,
            "per_class": {k: v for k, v in class_results.items()},
            "overall": overall_result if all_fault_errors else {},
        },
    }

    results_path = metrics_dir / "gtbad_results.json"
    results_path.write_text(json.dumps(results_payload, indent=2, default=str), encoding="utf-8")
    logger.success(f"  Results saved → {results_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.success("=" * 60)
    logger.success("GTBAD Training & Evaluation Complete")
    if all_fault_errors:
        logger.success(f"  Overall F1: {overall_result['f1_score']:.4f}")
        logger.success(f"  Overall Precision: {overall_result['precision']:.4f}")
        logger.success(f"  Overall Recall: {overall_result['recall']:.4f}")
    logger.success(f"  Checkpoint: {ckpt_path}")
    logger.success(f"  Metrics: {results_path}")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
