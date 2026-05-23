#!/usr/bin/env python3
"""
GTBAD v3 — Mixed-validation variant with macro-F1 GVSAO fitness.

Key changes from train_gtbad.py (original first experiment):
  - Validation set contains a **mix** of healthy and faulty data (temporal split
    with autocorrelation-prevention gaps), not healthy-only.
  - GVSAO fitness function is **macro-F1 per class (including healthy/class 0)**
    instead of validation MSE.  GVSAO minimises — returning -macro_F1 so that
    higher F1 → lower (better) fitness.
  - EVALUABLE_CLASSES extended to [0, 1, 2, 3, 4] so that per-class and
    macro-F1 metrics include the healthy class.
  - Test set is also mixed (remainder of the temporal split).
  - Uses gvsao_v3.py (fitness-agnostic logging with automatic F1 display).

Everything else stays close to the original experiment:
  - Point-wise reconstruction (window_length = 1)
  - StandardScaler fitted on train only
  - Single-threshold anomaly detection (percentile of training errors)
  - 9 raw sensor features (vdc1, vdc2, idc1, idc2, pdc1, pdc2, pdc, irr, pvt)
  - Original GVSAO 2-parameter search space (learning_rate, batch_size)

Usage:
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_v3
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_v3 --no-gvsao
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_v3 --epochs 100 --lr 0.001
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.anomaly_detection.dl.gtbad_model import GTBADModel, reconstruction_error
from src.modeling.anomaly_detection.dl.gvsao_v3 import GVSaoV3Config, GVSaoV3Result, run_gvsao_v3

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "ingestion" / "costa" / "costa_merged.parquet"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "gtbad_v3"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "experiments" / "metrics"

SENSOR_COLS = ["vdc1", "vdc2", "idc1", "idc2", "pdc1", "pdc2", "pdc", "irr", "pvt"]

FAULT_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}

EVALUABLE_CLASSES = [0, 1, 2, 3, 4]


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Splitting
# ─────────────────────────────────────────────────────────────────────────────


def load_data(parquet_path: str | Path) -> pd.DataFrame:
    """Load ingested Costa data at native 1 Hz sampling.

    Returns DataFrame with timestamp index, sorted chronologically.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Ingested parquet not found: {parquet_path}\n"
            "Run: uv run python -m src.data.ingestion --dataset costa"
        )

    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    logger.info(f"  Loaded: {len(df):,} rows, {df.shape[1]} columns")
    logger.info("  Label distribution:")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        logger.info(f"    {FAULT_NAMES.get(int(lbl), '?')} ({int(lbl)}): {cnt:,}")

    return df


def split_temporal_mixed(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    gap_samples: int = 300,
) -> dict[str, pd.DataFrame]:
    """Temporal split with autocorrelation-prevention gaps and mixed validation.

    Splits the chronologically-sorted DataFrame into three periods:
      - Train:  healthy rows only from the first ``train_frac`` of the timeline
      - Val:    **all rows** (healthy + faults) from the middle ``val_frac``
      - Test:   **all rows** (healthy + faults) from the remaining portion

    ``gap_samples`` rows are dropped at each split boundary to prevent
    autocorrelation leakage between train/val and val/test.
    """
    n = len(df)
    split1 = int(n * train_frac)
    split2 = int(n * (train_frac + val_frac))

    train_df = df.iloc[:split1]
    train_df = train_df[train_df["label"] == 0].copy()

    val_start = min(split1 + gap_samples, n)
    val_end = split2
    test_start = min(split2 + gap_samples, n)

    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:].copy()

    logger.info(f"  Split: total={n:,} | train_frac={train_frac} | val_frac={val_frac} | gap={gap_samples}")
    logger.info(f"  Train (healthy-only): {len(train_df):,}")
    logger.info(f"  Val   (healthy+faults): {len(val_df):,}")
    logger.info(f"    Val labels: {dict(val_df['label'].value_counts().sort_index())}")
    logger.info(f"  Test  (healthy+faults): {len(test_df):,}")
    logger.info(f"    Test labels: {dict(test_df['label'].value_counts().sort_index())}")

    return {"train": train_df, "val": val_df, "test": test_df}


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation & Tensorisation
# ─────────────────────────────────────────────────────────────────────────────


def prepare_tensors(
    splits: dict[str, pd.DataFrame],
    scaler: StandardScaler | None = None,
    window_length: int = 1,
) -> dict[str, Any]:
    """StandardScaler-normalize features and create windowed tensors.

    Scaler is fitted on TRAIN data only (leakage prevention).
    Window length = 1 for point-wise reconstruction.
    """
    sensor_cols_present = [c for c in SENSOR_COLS if c in splits["train"].columns]
    n_features = len(sensor_cols_present)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(splits["train"][sensor_cols_present].values)

    result: dict[str, Any] = {
        "scaler": scaler,
        "n_features": n_features,
        "feature_names": sensor_cols_present,
    }

    for split_name, df in splits.items():
        X = scaler.transform(df[sensor_cols_present].values).astype(np.float32)
        labels = df["label"].values.astype(np.int32)

        if window_length == 1:
            X_windows = X[:, np.newaxis, :]  # (n_samples, 1, n_features)
            l_windows = labels
        else:
            n = len(X)
            if n < window_length:
                raise ValueError(
                    f"Not enough data ({n}) for window length {window_length} in '{split_name}'"
                )
            X_windows = np.stack([X[i : i + window_length] for i in range(n - window_length + 1)])
            l_windows = labels[window_length - 1 :]
            l_windows = np.max(
                [labels[i : i + window_length] for i in range(n - window_length + 1)], axis=1
            )

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
    """Compute per-sample MSE reconstruction error (scalar per sample)."""
    model.eval()
    all_errors: list[np.ndarray] = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end].to(device)
        recon = model(X_batch)
        err = reconstruction_error(X_batch, recon).cpu().numpy()
        all_errors.append(err)
    return np.concatenate(all_errors) if all_errors else np.array([], dtype=np.float32)


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
    """Train GTBAD model with early stopping on validation MSE."""
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
            logger.info(
                f"    Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

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


def evaluate_binary(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Binary anomaly detection metrics: TP, FP, FN, TN, precision, recall, F1."""
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


def compute_per_class_f1(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    classes: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Compute per-class F1 for each class in ``classes``.

    For each class k, the subset of samples with label == k is evaluated using
    the binary anomaly detection convention  (true positive = anomaly detected).
    The macro-F1 is the unweighted mean of the per-class F1 scores (including
    class 0 / healthy, whose F1 is 0 when there are no false alarms).
    """
    if classes is None:
        classes = EVALUABLE_CLASSES

    preds = (errors > threshold).astype(int)
    true = (labels > 0).astype(int)

    per_class: dict[int, dict[str, float]] = {}
    class_f1s: list[float] = []

    for cls in classes:
        mask = labels == cls
        n_cls = int(mask.sum())
        if n_cls == 0:
            continue

        cls_preds = preds[mask]
        cls_true = true[mask]

        tp_c = int(np.sum((cls_preds == 1) & (cls_true == 1)))
        fp_c = int(np.sum((cls_preds == 1) & (cls_true == 0)))
        fn_c = int(np.sum((cls_preds == 0) & (cls_true == 1)))
        tn_c = int(np.sum((cls_preds == 0) & (cls_true == 0)))

        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0

        per_class[cls] = {
            "class_name": FAULT_NAMES.get(cls, f"Class_{cls}"),
            "n_samples": n_cls,
            "TP": tp_c,
            "FP": fp_c,
            "FN": fn_c,
            "TN": tn_c,
            "precision": round(p_c, 6),
            "recall": round(r_c, 6),
            "f1_score": round(f1_c, 6),
        }
        class_f1s.append(f1_c)

    macro_f1 = float(np.mean(class_f1s)) if class_f1s else 0.0
    return {"per_class": per_class, "macro_f1": round(macro_f1, 6)}


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────────────────────


def build_model(
    n_features: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    lstm_hidden: int,
    dropout: float,
) -> GTBADModel:
    return GTBADModel(
        input_dim=n_features,
        output_dim=n_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        lstm_hidden=lstm_hidden,
        dropout=dropout,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train GTBAD v3: mixed validation + macro-F1 GVSAO")
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
    parser.add_argument("--train-frac", type=float, default=0.80, help="Fraction of timeline for train (healthy-only)")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Fraction of timeline for val (mixed)")
    parser.add_argument("--gap-samples", type=int, default=300, help="Rows dropped at split boundaries")
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--fitness-threshold-pct", type=float, default=95.0,
                        help="Percentile for threshold during GVSAO fitness eval")
    parser.add_argument("--window-length", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Seed ──────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Device: {device} | Seed: {args.seed} | Window length: {args.window_length}")

    # ── Load ──────────────────────────────────────────────────────────────
    logger.info("=== Step 1: Load Costa Data (native 1 Hz) ===")
    df = load_data(args.parquet_path)

    # ── Temporal split with mixed val ─────────────────────────────────────
    logger.info("=== Step 2: Temporal Split (train=healthy-only, val=mixed, test=mixed) ===")
    splits = split_temporal_mixed(
        df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        gap_samples=args.gap_samples,
    )

    # ── Normalise & Tensorise ──────────────────────────────────────────────
    logger.info("=== Step 3: Normalize & Create Tensors ===")
    tensors = prepare_tensors(splits, window_length=args.window_length)
    n_features = tensors["n_features"]
    feature_names = tensors["feature_names"]

    X_train = tensors["train"]["X"]
    X_val = tensors["val"]["X"]
    val_labels = tensors["val"]["labels"]
    logger.info(f"  n_features: {n_features} ({feature_names})")
    logger.info(f"  X_train: {tuple(X_train.shape)} | X_val: {tuple(X_val.shape)}")

    # ── GVSAO Hyperparameter Optimisation (macro-F1 fitness) ───────────────
    final_lr = args.lr or 0.001
    final_batch_size = args.batch_size or 32
    gvsao_result = None

    if not args.no_gvsao and (args.lr is None or args.batch_size is None):
        logger.info("=== Step 4: GVSAO v3 Hyperparameter Tuning (macro-F1 fitness) ===")

        gvsao_config = GVSaoV3Config(
            population_size=10,
            max_generations=5,
            lr_bounds=(1e-5, 1e-1),
            batch_bounds=(16, 128),
            seed=args.seed,
        )

        def fitness_fn(lr: float, batch: int) -> float:
            """Train lightweight model, return -macro_F1 (GVSAO minimises)."""
            model = build_model(
                n_features, args.d_model, args.nhead,
                args.num_encoder_layers, args.lstm_hidden, args.dropout,
            ).to(device)
            effective_batch = min(batch, X_train.shape[0])

            train_model(
                model, X_train, X_val, device,
                lr=lr, batch_size=effective_batch,
                epochs=args.gvsao_epochs, patience=3, verbose=False,
            )

            # Compute reconstruction errors
            train_errs = evaluate_reconstruction(model, X_train, device, effective_batch)
            val_errs = evaluate_reconstruction(model, X_val, device, effective_batch)

            # Threshold from training errors
            threshold = np.percentile(train_errs, args.fitness_threshold_pct)

            # Per-class F1 on validation (mixed) — includes healthy / class 0
            per_class_info = compute_per_class_f1(
                val_errs, val_labels, threshold, classes=EVALUABLE_CLASSES,
            )
            macro_f1 = per_class_info["macro_f1"]

            return -macro_f1  # GVSAO minimises — higher F1 → lower fitness

        gvsao_result = run_gvsao_v3(fitness_fn, gvsao_config, verbose=True)
        final_lr = gvsao_result.best_params["learning_rate"]
        final_batch_size = gvsao_result.best_params["batch_size"]
        final_batch_size = min(final_batch_size, X_train.shape[0])
        logger.success(
            f"GVSAO v3 best: lr={final_lr:.6f}, batch={final_batch_size} "
            f"(macro_F1={-gvsao_result.best_fitness:.4f})"
        )

    # ── Final Model Training ──────────────────────────────────────────────
    logger.info(f"=== Step 5: Train GTBAD v3 (lr={final_lr:.6f}, batch={final_batch_size}) ===")
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
    ckpt_path = checkpoint_dir / "gtbad_v3_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "feature_names": feature_names,
        "scaler_mean": tensors["scaler"].mean_.tolist(),
        "scaler_scale": tensors["scaler"].scale_.tolist(),
        "args": vars(args),
    }, ckpt_path)
    logger.success(f"  Saved checkpoint → {ckpt_path}")

    # ── Anomaly Detection Evaluation ──────────────────────────────────────
    logger.info("=== Step 6: Anomaly Detection Evaluation ===")

    # Compute threshold from training reconstruction errors
    train_errors = evaluate_reconstruction(model, X_train, device, final_batch_size)
    threshold = compute_threshold(train_errors, args.threshold_percentile)
    logger.info(f"  Anomaly threshold ({args.threshold_percentile}th pctl): {threshold:.6f}")

    # ── Validation (mixed) ──────────────────────────────────────────────
    X_val_t = tensors["val"]["X"]
    val_labels_arr = tensors["val"]["labels"]
    val_errors = evaluate_reconstruction(model, X_val_t, device, final_batch_size)
    val_binary = evaluate_binary(val_errors, val_labels_arr, threshold)
    val_per_class = compute_per_class_f1(val_errors, val_labels_arr, threshold, classes=EVALUABLE_CLASSES)

    logger.info(f"\n  ── Validation (mixed) ──")
    logger.info(
        f"    Overall binary:  P={val_binary['precision']:.4f}  "
        f"R={val_binary['recall']:.4f}  F1={val_binary['f1_score']:.4f}  "
        f"TP={val_binary['TP']} FP={val_binary['FP']} FN={val_binary['FN']} TN={val_binary['TN']}"
    )
    logger.info(f"    Macro-F1 (0-4):  {val_per_class['macro_f1']:.4f}")
    for cls in EVALUABLE_CLASSES:
        info = val_per_class["per_class"].get(cls)
        if info is None:
            continue
        logger.info(
            f"    Class {info['class_name']:<14}  "
            f"F1={info['f1_score']:.4f}  P={info['precision']:.4f}  R={info['recall']:.4f}  "
            f"n={info['n_samples']:,}  TP={info['TP']} FP={info['FP']} FN={info['FN']}"
        )

    # ── Test (mixed) ────────────────────────────────────────────────────
    X_test_t = tensors["test"]["X"]
    test_labels_arr = tensors["test"]["labels"]
    test_errors = evaluate_reconstruction(model, X_test_t, device, final_batch_size)
    test_binary = evaluate_binary(test_errors, test_labels_arr, threshold)
    test_per_class = compute_per_class_f1(test_errors, test_labels_arr, threshold, classes=EVALUABLE_CLASSES)

    logger.info(f"\n  ── Test (mixed) ──")
    logger.info(
        f"    Overall binary:  P={test_binary['precision']:.4f}  "
        f"R={test_binary['recall']:.4f}  F1={test_binary['f1_score']:.4f}  "
        f"TP={test_binary['TP']} FP={test_binary['FP']} FN={test_binary['FN']} TN={test_binary['TN']}"
    )
    logger.info(f"    Macro-F1 (0-4):  {test_per_class['macro_f1']:.4f}")
    for cls in EVALUABLE_CLASSES:
        info = test_per_class["per_class"].get(cls)
        if info is None:
            continue
        logger.info(
            f"    Class {info['class_name']:<14}  "
            f"F1={info['f1_score']:.4f}  P={info['precision']:.4f}  R={info['recall']:.4f}  "
            f"n={info['n_samples']:,}  TP={info['TP']} FP={info['FP']} FN={info['FN']}"
        )

    # ── Save Results ───────────────────────────────────────────────────────
    metrics_dir = Path(DEFAULT_METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    def _per_class_serializable(pc):
        return {str(k): v for k, v in pc.items()}

    results_payload = {
        "model": "GTBAD v3 (GVSAO-Transformer-BiLSTM, mixed-val, macro-F1 fitness)",
        "dataset": "Costa PV Fault Dataset",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "window_length": args.window_length,
        "input_features": feature_names,
        "split": {
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "gap_samples": args.gap_samples,
            "train_only_healthy": True,
            "val_mixed": True,
            "test_mixed": True,
        },
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
            "fitness_metric": "macro_F1" if gvsao_result is not None else None,
            "fitness_threshold_pct": args.fitness_threshold_pct if gvsao_result else None,
            "best_params": gvsao_result.best_params if gvsao_result else None,
            "best_fitness": gvsao_result.best_fitness if gvsao_result else None,
            "history": gvsao_result.history if gvsao_result else None,
            "n_evals": gvsao_result.n_evals if gvsao_result else 0,
        },
        "anomaly_detection": {
            "threshold_percentile": args.threshold_percentile,
            "threshold_value": threshold,
            "val": {
                "binary": val_binary,
                "macro_f1": val_per_class["macro_f1"],
                "per_class": _per_class_serializable(val_per_class["per_class"]),
            },
            "test": {
                "binary": test_binary,
                "macro_f1": test_per_class["macro_f1"],
                "per_class": _per_class_serializable(test_per_class["per_class"]),
            },
        },
    }

    results_path = metrics_dir / "gtbad_v3_results.json"
    results_path.write_text(json.dumps(results_payload, indent=2, default=str), encoding="utf-8")
    logger.success(f"  Results saved → {results_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.success("=" * 60)
    logger.success("GTBAD v3 Training & Evaluation Complete")
    logger.success(f"  Val  macro-F1: {val_per_class['macro_f1']:.4f}  |  binary F1: {val_binary['f1_score']:.4f}")
    logger.success(f"  Test macro-F1: {test_per_class['macro_f1']:.4f}  |  binary F1: {test_binary['f1_score']:.4f}")
    logger.success(f"  Checkpoint: {ckpt_path}")
    logger.success(f"  Metrics: {results_path}")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
