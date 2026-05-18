#!/usr/bin/env python3
"""
GTBAD v2 — Extended experiment: per-feature/group thresholds, GVSAO 7D HPO.

Key additions over train_gtbad.py:
  - Optimises window_size, d_model/nhead, num_encoder_layers, lstm_hidden via GVSAO
  - Label-pure sliding windows (no mixed-label windows)
  - Per-feature threshold computation at multiple percentiles
  - Group-based (PDC/IDC/VDC AND/OR) and all-feature decision logic
  - Supports original 9-sensor and plus_physics feature profiles
  - Full MLflow → DagsHub integration
  - Primary metric: F1 (PR-AUC as secondary)

All config driven by data_config.yaml (profile) and model_config.yaml.

Usage:
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_v2
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_v2 --no-gvsao
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_v2 --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.data.features import add_physics_features
from src.modeling.anomaly_detection.dl.gtbad_model import GTBADModel
from src.modeling.anomaly_detection.dl.gvsao_v2 import (
    GVSaoV2Config,
    GVSaoV2Result,
    ParamDef,
    decode_individual,
    run_gvsao_v2,
)
from src.mlflow_setup import init_tracking

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "ingestion" / "costa" / "costa_merged.parquet"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "gtbad_v2"
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


# ── Config Loading ───────────────────────────────────────────────────────────


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_profile_config(profile_name: str) -> dict:
    data_cfg = load_yaml(DATA_CONFIG_PATH)
    profiles = data_cfg.get("feature_engineering", {}).get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"Profile '{profile_name}' not found in data_config.yaml profiles")
    return profiles[profile_name]


def load_model_config() -> dict:
    return load_yaml(MODEL_CONFIG_PATH)


# ── Device ───────────────────────────────────────────────────────────────────


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


# ── Data Loading & Splitting ─────────────────────────────────────────────────


def load_data(parquet_path: str | Path) -> pd.DataFrame:
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
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        logger.info(f"    {FAULT_NAMES.get(int(lbl), '?')} ({int(lbl)}): {cnt:,}")
    return df


def split_healthy_faulty(
    df: pd.DataFrame,
    healthy_train_frac: float = 0.80,
) -> dict[str, pd.DataFrame]:
    healthy = df[df["label"] == 0].copy()
    faulty = df[df["label"] > 0].copy()
    n_healthy = len(healthy)
    split_idx = int(n_healthy * healthy_train_frac)
    train_df = healthy.iloc[:split_idx]
    val_df = healthy.iloc[split_idx:]
    test_by_class: dict[str, pd.DataFrame] = {}
    for cls in EVALUABLE_CLASSES:
        cls_data = faulty[faulty["label"] == cls]
        if len(cls_data) > 0:
            test_by_class[f"fault_class_{cls}"] = cls_data
    logger.info(f"  Train (healthy): {len(train_df):,}")
    logger.info(f"  Val   (healthy): {len(val_df):,}")
    for name, cls_df in test_by_class.items():
        logger.info(f"  Test  ({name}): {len(cls_df):,}")
    return {"train": train_df, "val": val_df, **test_by_class}


# ── Scaling ──────────────────────────────────────────────────────────────────


class MinMaxScaler:
    def fit(self, data: np.ndarray) -> MinMaxScaler:
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ < 1e-10] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min_) / self.range_


# ── Label-Pure Windowing ─────────────────────────────────────────────────────


def create_label_pure_windows(
    X: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows keeping only those where ALL labels are identical.

    Returns (X_windows, labels_windows) or (empty, empty) if no pure windows.
    """
    n = len(X)
    if n < window_size:
        return np.empty((0, window_size, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=labels.dtype)

    windows_x: list[np.ndarray] = []
    windows_y: list = []

    for i in range(0, n - window_size + 1, stride):
        window_labels = labels[i : i + window_size]
        if np.all(window_labels == window_labels[0]):
            windows_x.append(X[i : i + window_size])
            windows_y.append(window_labels[0])

    if not windows_x:
        return np.empty((0, window_size, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=labels.dtype)

    return np.stack(windows_x), np.array(windows_y)


# ── Model Building & Training ────────────────────────────────────────────────


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


def _make_dataloader(X: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(X, X)
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
def evaluate_reconstruction_scalar(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Scalar reconstruction error per sample (sum over features and time)."""
    model.eval()
    all_errors: list[np.ndarray] = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end].to(device)
        recon = model(X_batch)
        err = torch.sum((X_batch - recon) ** 2, dim=(1, 2)).cpu().numpy()
        all_errors.append(err)
    return np.concatenate(all_errors) if all_errors else np.array([])


@torch.no_grad()
def evaluate_reconstruction_per_feature(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Per-feature reconstruction error averaged over time: (n_samples, n_features)."""
    if X.shape[0] == 0:
        return np.empty((0, X.shape[2]))
    model.eval()
    all_errors: list[np.ndarray] = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end].to(device)
        recon = model(X_batch)
        err = torch.mean((X_batch - recon) ** 2, dim=1).cpu().numpy()  # (B, D)
        all_errors.append(err)
    return np.concatenate(all_errors) if all_errors else np.empty((0, X.shape[2]))


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


# ── Thresholds & Anomaly Scoring ─────────────────────────────────────────────


def compute_per_feature_thresholds(
    train_errors: np.ndarray,
    percentiles: list[float],
) -> dict[str, np.ndarray]:
    """Compute per-feature thresholds at given percentiles from training errors.

    train_errors: (n_train, n_features) — per-feature reconstruction errors.
    Returns: {str(pct): np.array of shape (n_features,)} for each percentile.
    """
    return {str(pct): np.percentile(train_errors, pct, axis=0) for pct in percentiles}


def compute_normalized_errors(
    errors: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Normalize per-feature errors by their thresholds."""
    thresh = thresholds.copy()
    thresh[thresh < 1e-12] = 1e-12
    return errors / thresh


def anomaly_score(normalized_errors: np.ndarray) -> np.ndarray:
    """Max normalized error across features → scalar anomaly score per sample."""
    return np.max(normalized_errors, axis=1)


def group_anomaly_score(
    errors: np.ndarray,
    thresholds: np.ndarray,
    feature_names: list[str],
    feature_groups: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Compute per-group anomaly scores (mean normalized error per group).

    Returns: {group_name: scores (n_samples,)} for each group that has features present.
    """
    n_samples = errors.shape[0]
    norm = compute_normalized_errors(errors, thresholds)
    group_scores: dict[str, np.ndarray] = {}
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    for group_name, group_features in feature_groups.items():
        indices = [name_to_idx[f] for f in group_features if f in name_to_idx]
        if not indices:
            group_scores[group_name] = np.zeros(n_samples)
            continue
        group_scores[group_name] = np.mean(norm[:, indices], axis=1)

    return group_scores


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_binary(
    preds: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Compute TP/FP/FN/TN, precision, recall, F1."""
    true = (labels > 0).astype(int)
    preds_bin = preds.astype(int)

    tp = int(np.sum((preds_bin == 1) & (true == 1)))
    fp = int(np.sum((preds_bin == 1) & (true == 0)))
    fn = int(np.sum((preds_bin == 0) & (true == 1)))
    tn = int(np.sum((preds_bin == 0) & (true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
    }


def evaluate_pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """PR-AUC from continuous anomaly scores."""
    from sklearn.metrics import average_precision_score
    true = (labels > 0).astype(int)
    if len(np.unique(true)) < 2:
        return 0.0
    return float(average_precision_score(true, scores))


# ── Decision Logic ───────────────────────────────────────────────────────────


def decide_group(
    group_scores: dict[str, np.ndarray],
    group_logic: str,
) -> np.ndarray:
    """Group-based binary decision.

    A group is flagged if its mean normalized error > 1.0.
    group_logic: 'and' → all groups must flag; 'or' → any group flags.
    """
    flags = {name: scores > 1.0 for name, scores in group_scores.items()}
    if not flags:
        return np.zeros(0, dtype=bool)

    stack = np.stack(list(flags.values()), axis=0)  # (n_groups, n_samples)
    if group_logic == "and":
        return np.all(stack, axis=0)
    return np.any(stack, axis=0)


def decide_all_features(
    errors: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """All-feature binary decision: ALL features must exceed their thresholds."""
    norm = compute_normalized_errors(errors, thresholds)
    return np.all(norm > 1.0, axis=1)


# ── Run Pipeline for One Feature Mode ────────────────────────────────────────


def run_experiment(
    feature_mode: str,
    device: torch.device,
    profile: dict,
    model_cfg: dict,
    args: argparse.Namespace,
) -> dict:
    """Run the full pipeline for one feature_mode (original or plus_physics).

    Returns results dict used for saving and MLflow.
    """
    mode_label = f"GTBAD_v2_{feature_mode}"
    logger.info(f"\n{'='*60}\n  {mode_label}\n{'='*60}")

    # ── Load and split ───────────────────────────────────────────────────
    df = load_data(args.parquet_path)
    train_frac = float(profile.get("train_frac", model_cfg.get("train_frac", 0.80)))
    splits = split_healthy_faulty(df, healthy_train_frac=train_frac)

    # ── Feature engineering ──────────────────────────────────────────────
    if feature_mode == "plus_physics":
        logger.info("  Applying plus_physics features...")
        physics_flags = {k: v for k, v in profile.items() if k in {
            "enable_dP_dt", "enable_dV_dt", "enable_dI_dt",
            "enable_power_imbalance", "enable_current_imbalance",
            "enable_voltage_imbalance", "enable_temp_power_correction",
            "temp_ref_c", "gamma_pmax_pct_per_c", "temp_power_eps", "irr_norm_floor",
        }}
        for split_name in splits:
            if "segment_id" not in splits[split_name].columns:
                splits[split_name]["segment_id"] = 0
            if "timestamp" not in splits[split_name].columns:
                splits[split_name]["timestamp"] = splits[split_name].index
            splits[split_name] = add_physics_features(
                splits[split_name],
                segment_col="segment_id",
                time_col="timestamp",
                flags=physics_flags,
            )
        sensor_cols_present = [
            c for c in splits["train"].columns
            if c not in ("label", "segment_id", "timestamp", "Fault", "episode_id",
                         "operating_day_id", "continuity_segment_id")
            and pd.api.types.is_numeric_dtype(splits["train"][c])
        ]
    else:
        sensor_cols_present = [c for c in SENSOR_COLS if c in splits["train"].columns]

    n_features = len(sensor_cols_present)
    logger.info(f"  Feature mode: {feature_mode} | n_features: {n_features}")
    logger.info(f"  Features: {sensor_cols_present}")

    # ── Extract raw arrays ───────────────────────────────────────────────
    raw_arrays: dict[str, dict[str, np.ndarray]] = {}
    for split_name, split_df in splits.items():
        X_raw = split_df[sensor_cols_present].values.astype(np.float32)
        labels_raw = split_df["label"].values.astype(np.int32)
        raw_arrays[split_name] = {"X": X_raw, "labels": labels_raw}

    # ── Scale ────────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    scaler.fit(raw_arrays["train"]["X"])
    scaled: dict[str, dict[str, np.ndarray]] = {}
    for split_name, arr in raw_arrays.items():
        scaled[split_name] = {
            "X": scaler.transform(arr["X"]),
            "labels": arr["labels"],
        }

    # ── GVSAO config ─────────────────────────────────────────────────────
    gvsao_cfg = model_cfg.get("gvsao", {})
    candidates = profile.get("gvsao_candidates", {})
    window_sizes = candidates.get("window_size", [3, 5, 10, 20, 30, 60, 90])
    dmodel_nhead_pairs = candidates.get("d_model_nhead_pairs", [[64, 2]])
    num_enc_layers_list = candidates.get("num_encoder_layers", [1, 2, 3, 4])
    lstm_hidden_list = candidates.get("lstm_hidden", [16, 32, 64, 128])

    param_defs = [
        ParamDef(name="lr", kind="continuous_log", bounds=tuple(gvsao_cfg.get("lr_bounds", [1e-5, 1e-1]))),
        ParamDef(name="batch_size", kind="discrete", candidates=list(range(
            int(gvsao_cfg.get("batch_bounds", [16, 128])[0]),
            int(gvsao_cfg.get("batch_bounds", [16, 128])[1]) + 1,
        ))),
        ParamDef(name="window_size", kind="discrete", candidates=window_sizes),
        ParamDef(name="d_model_nhead", kind="discrete", candidates=dmodel_nhead_pairs),
        ParamDef(name="num_encoder_layers", kind="discrete", candidates=num_enc_layers_list),
        ParamDef(name="lstm_hidden", kind="discrete", candidates=lstm_hidden_list),
    ]

    gvsao_v2_config = GVSaoV2Config(
        param_defs=param_defs,
        population_size=int(gvsao_cfg.get("population_size", 20)),
        max_generations=int(gvsao_cfg.get("max_generations", 10)),
        seed=args.seed,
    )

    dropout = float(profile.get("dropout", model_cfg.get("dropout", 0.1)))
    gvsao_epochs = int(gvsao_cfg.get("gvsao_epochs", 5))

    # ── Window cache for GVSAO efficiency ────────────────────────────────
    window_cache: dict[int, dict] = {}

    def get_windowed(window_size: int) -> dict:
        w = int(window_size)
        if w not in window_cache:
            w_data: dict[str, dict] = {}
            for split_name, arr in scaled.items():
                X_w, labels_w = create_label_pure_windows(
                    arr["X"], arr["labels"], w, stride=1
                )
                w_data[split_name] = {
                    "X": torch.from_numpy(X_w),
                    "labels": labels_w,
                    "n_windows": len(X_w),
                }
            window_cache[w] = w_data
        return window_cache[w]

    # ── GVSAO fitness ────────────────────────────────────────────────────
    def fitness_fn(params: dict[str, Any]) -> float:
        model = build_model(
            n_features=n_features,
            d_model=params["d_model_nhead"][0],
            nhead=params["d_model_nhead"][1],
            num_encoder_layers=params["num_encoder_layers"],
            lstm_hidden=params["lstm_hidden"],
            dropout=dropout,
        ).to(device)

        w_data = get_windowed(params["window_size"])
        X_train = w_data["train"]["X"]
        X_val = w_data["val"]["X"]
        effective_batch = min(params["batch_size"], X_train.shape[0])

        _, info = train_model(
            model, X_train, X_val, device,
            lr=params["lr"], batch_size=effective_batch,
            epochs=gvsao_epochs, patience=3, verbose=False,
        )
        return info["best_val_loss"]

    # ── GVSAO optimisation ───────────────────────────────────────────────
    final_params: dict[str, Any] = {}
    gvsao_result = None

    if not args.no_gvsao:
        logger.info(f"  GVSAO: pop={gvsao_v2_config.population_size}, gen={gvsao_v2_config.max_generations}")
        gvsao_result = run_gvsao_v2(fitness_fn, gvsao_v2_config, verbose=True)
        final_params = gvsao_result.best_params
        logger.success(f"  GVSAO best: {final_params}")
    else:
        final_params = {
            "lr": args.lr or 0.001,
            "batch_size": args.batch_size or 32,
            "window_size": args.window_size or 10,
            "d_model_nhead": [args.d_model or 64, args.nhead or 2],
            "num_encoder_layers": args.num_encoder_layers or 3,
            "lstm_hidden": args.lstm_hidden or 32,
        }

    # ── Final training ───────────────────────────────────────────────────
    final_window_size = int(final_params["window_size"])
    w_data_final = get_windowed(final_window_size)
    X_train_final = w_data_final["train"]["X"]
    X_val_final = w_data_final["val"]["X"]
    final_batch = min(int(final_params["batch_size"]), X_train_final.shape[0])

    logger.info(f"  Final training: W={final_window_size}, lr={final_params['lr']:.6f}, batch={final_batch}")

    model = build_model(
        n_features=n_features,
        d_model=final_params["d_model_nhead"][0],
        nhead=final_params["d_model_nhead"][1],
        num_encoder_layers=final_params["num_encoder_layers"],
        lstm_hidden=final_params["lstm_hidden"],
        dropout=dropout,
    ).to(device)

    t0 = time.perf_counter()
    epochs = int(profile.get("epochs", model_cfg.get("epochs", 50)))
    patience = int(profile.get("patience", model_cfg.get("patience", 15)))
    model, training_info = train_model(
        model, X_train_final, X_val_final, device,
        lr=float(final_params["lr"]), batch_size=final_batch,
        epochs=epochs, patience=patience, verbose=True,
    )
    train_time = time.perf_counter() - t0
    logger.info(f"  Training completed in {train_time:.1f}s")

    # ── Checkpoint ───────────────────────────────────────────────────────
    checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"gtbad_v2_{feature_mode}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "feature_names": sensor_cols_present,
        "window_size": final_window_size,
        "scaler_min": scaler.min_.tolist(),
        "scaler_max": scaler.max_.tolist(),
        "final_params": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in final_params.items()},
        "seed": args.seed,
    }, ckpt_path)
    logger.success(f"  Checkpoint → {ckpt_path}")

    # ── Threshold computation ────────────────────────────────────────────
    percentiles = profile.get("threshold_percentiles", [90, 95, 99])
    train_per_feature = evaluate_reconstruction_per_feature(model, X_train_final, device, final_batch)
    thresholds_dict = compute_per_feature_thresholds(train_per_feature, percentiles)
    logger.info(f"  Per-feature thresholds computed at percentiles: {percentiles}")

    # ── Evaluation ───────────────────────────────────────────────────────
    decision_logic = profile.get("decision_logic", "group")
    group_logic = profile.get("group_logic", "and")
    feature_groups = profile.get("feature_groups", {})

    all_results: dict[str, dict] = {}

    for pct_key, pct_thresholds in thresholds_dict.items():
        pct_label = f"pct_{pct_key}"

        # --- Val healthy ---
        val_errors = evaluate_reconstruction_per_feature(model, X_val_final, device, final_batch)
        val_scores = anomaly_score(compute_normalized_errors(val_errors, pct_thresholds))

        if decision_logic == "group":
            val_group_scores = group_anomaly_score(val_errors, pct_thresholds, sensor_cols_present, feature_groups)
            val_preds = decide_group(val_group_scores, group_logic)
        else:
            val_preds = decide_all_features(val_errors, pct_thresholds)

        val_result = evaluate_binary(val_preds, w_data_final["val"]["labels"])
        val_fp = int(np.sum(val_preds))

        # --- Per-class evaluation ---
        class_results: dict[str, dict] = {}
        for cls in EVALUABLE_CLASSES:
            key = f"fault_class_{cls}"
            if key not in w_data_final:
                continue
            cls_errors = evaluate_reconstruction_per_feature(
                model, w_data_final[key]["X"], device, final_batch
            )
            if cls_errors.shape[0] == 0:
                continue

            cls_scores = anomaly_score(compute_normalized_errors(cls_errors, pct_thresholds))

            if decision_logic == "group":
                cls_group_scores = group_anomaly_score(cls_errors, pct_thresholds, sensor_cols_present, feature_groups)
                cls_preds = decide_group(cls_group_scores, group_logic)
            else:
                cls_preds = decide_all_features(cls_errors, pct_thresholds)

            cls_res = evaluate_binary(cls_preds, w_data_final[key]["labels"])
            cls_res["pr_auc"] = round(evaluate_pr_auc(cls_scores, w_data_final[key]["labels"]), 6)
            class_results[FAULT_NAMES[cls]] = cls_res

        # --- Overall (val healthy + all fault) ---
        all_fault_errors_list = []
        all_fault_labels_list = []
        for cls in EVALUABLE_CLASSES:
            key = f"fault_class_{cls}"
            if key not in w_data_final:
                continue
            all_fault_errors_list.append(
                evaluate_reconstruction_per_feature(model, w_data_final[key]["X"], device, final_batch)
            )
            all_fault_labels_list.append(w_data_final[key]["labels"])

        if all_fault_errors_list:
            combined_errors = np.concatenate(all_fault_errors_list)
            combined_labels = np.concatenate(all_fault_labels_list)
            comb_norm = np.concatenate([val_errors, combined_errors])
            comb_labels_all = np.concatenate([
                np.zeros(len(val_errors), dtype=np.int32), combined_labels
            ])

            comb_scores = anomaly_score(compute_normalized_errors(comb_norm, pct_thresholds))

            if decision_logic == "group":
                comb_group = group_anomaly_score(comb_norm, pct_thresholds, sensor_cols_present, feature_groups)
                comb_preds = decide_group(comb_group, group_logic)
            else:
                comb_preds = decide_all_features(comb_norm, pct_thresholds)

            overall = evaluate_binary(comb_preds, comb_labels_all)
            overall["pr_auc"] = round(evaluate_pr_auc(comb_scores, comb_labels_all), 6)
        else:
            overall = {}

        all_results[pct_label] = {
            "threshold_percentile": float(pct_key),
            "thresholds": pct_thresholds.tolist(),
            "val_healthy_fp_rate": float(val_fp / len(val_errors)) if len(val_errors) > 0 else 0.0,
            "val": val_result,
            "per_class": class_results,
            "overall": overall,
        }

        f1_str = f"{overall.get('f1_score', 0):.4f}" if overall else "N/A"
        logger.info(
            f"  [{pct_label}] Overall F1={f1_str} "
            f"Precision={overall.get('precision', 0):.4f} "
            f"Recall={overall.get('recall', 0):.4f} "
            f"PR-AUC={overall.get('pr_auc', 0):.4f}"
            if overall else f"  [{pct_label}] No overall result"
        )

    # ── Assemble results payload ─────────────────────────────────────────
    results_payload = {
        "model": f"GTBAD v2 — {feature_mode}",
        "dataset": "Costa PV Fault Dataset",
        "feature_mode": feature_mode,
        "decision_logic": decision_logic,
        "group_logic": group_logic if decision_logic == "group" else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "window_size": final_window_size,
        "input_features": sensor_cols_present,
        "n_features": n_features,
        "model_config": {
            "d_model": final_params["d_model_nhead"][0],
            "nhead": final_params["d_model_nhead"][1],
            "num_encoder_layers": final_params["num_encoder_layers"],
            "lstm_hidden": final_params["lstm_hidden"],
            "dropout": dropout,
        },
        "training": {
            "learning_rate": float(final_params["lr"]),
            "batch_size": final_batch,
            "epochs": training_info["epochs_trained"],
            "best_val_loss": training_info["best_val_loss"],
            "train_time_seconds": round(train_time, 1),
            "seed": args.seed,
        },
        "gvsao": {
            "enabled": gvsao_result is not None,
            "best_params": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in final_params.items()} if gvsao_result else None,
            "best_fitness": gvsao_result.best_fitness if gvsao_result else None,
            "history": gvsao_result.history if gvsao_result else None,
            "n_evals": gvsao_result.n_evals if gvsao_result else 0,
        },
        "anomaly_detection": all_results,
    }

    # ── Save results ─────────────────────────────────────────────────────
    metrics_dir = Path(DEFAULT_METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_path = metrics_dir / f"gtbad_v2_results_{feature_mode}.json"
    results_path.write_text(json.dumps(results_payload, indent=2, default=str), encoding="utf-8")
    logger.success(f"  Results → {results_path}")

    # ── Save thresholds ──────────────────────────────────────────────────
    thresholds_payload = {
        "feature_mode": feature_mode,
        "feature_names": sensor_cols_present,
        "thresholds": {k: v.tolist() for k, v in thresholds_dict.items()},
    }
    thresholds_path = metrics_dir / f"gtbad_v2_thresholds_{feature_mode}.json"
    thresholds_path.write_text(json.dumps(thresholds_payload, indent=2), encoding="utf-8")
    logger.success(f"  Thresholds → {thresholds_path}")

    return {
        "results_payload": results_payload,
        "results_path": results_path,
        "ckpt_path": ckpt_path,
        "thresholds_path": thresholds_path,
        "sensor_cols_present": sensor_cols_present,
        "n_features": n_features,
        "final_params": final_params,
        "final_batch": final_batch,
        "training_info": training_info,
        "train_time": train_time,
        "gvsao_result": gvsao_result,
        "final_window_size": final_window_size,
        "all_results": all_results,
        "dropout": dropout,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GTBAD v2 with extended HPO and thresholding")
    parser.add_argument("--parquet-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gvsao", action="store_true", help="Skip GVSAO HPO")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    # Fallback values when --no-gvsao is used
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num-encoder-layers", type=int, default=None)
    parser.add_argument("--lstm-hidden", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Device: {device} | Seed: {args.seed}")

    # ── Load configs ─────────────────────────────────────────────────────
    model_cfg_raw = load_model_config()
    model_cfg = model_cfg_raw.get("anomaly_detection", {}).get("dl", {}).get("models", {}).get("gtbad_v2", {})
    profile_name = model_cfg.get("profile", "gtbad_experiment")
    profile = load_profile_config(profile_name)
    logger.info(f"Profile: {profile_name} | decision_logic={profile.get('decision_logic')}")

    # ── Init MLflow ──────────────────────────────────────────────────────
    run_name = f"gtbad_v2_seed{args.seed}"
    if not args.no_mlflow:
        try:
            init_tracking("anomaly")
            mlflow.start_run(run_name=run_name)
            mlflow.set_tags({
                "model": "GTBAD_v2",
                "dataset": "costa",
                "profile": profile_name,
                "seed": str(args.seed),
            })
            logger.info("MLflow tracking active")
        except Exception as exc:
            logger.warning(f"MLflow init failed (non-fatal): {exc}")

    # ── Run for both feature modes ───────────────────────────────────────
    all_mode_results: list[dict] = []

    for mode in ["original", "plus_physics"]:
        try:
            mode_result = run_experiment(mode, device, profile, model_cfg, args)
            all_mode_results.append(mode_result)
        except Exception as exc:
            logger.error(f"Experiment '{mode}' failed: {exc}")
            if not args.no_mlflow and mlflow.active_run():
                mlflow.log_param(f"{mode}_error", str(exc))

    # ── MLflow logging ───────────────────────────────────────────────────
    if not args.no_mlflow and mlflow.active_run():
        try:
            all_params: dict[str, Any] = {}
            all_metrics: dict[str, float] = {}

            for mr in all_mode_results:
                mode = mr["results_payload"]["feature_mode"]
                all_params[f"{mode}_n_features"] = mr["n_features"]
                all_params[f"{mode}_window_size"] = mr["final_window_size"]
                all_params[f"{mode}_d_model"] = mr["final_params"]["d_model_nhead"][0]
                all_params[f"{mode}_nhead"] = mr["final_params"]["d_model_nhead"][1]
                all_params[f"{mode}_num_encoder_layers"] = mr["final_params"]["num_encoder_layers"]
                all_params[f"{mode}_lstm_hidden"] = mr["final_params"]["lstm_hidden"]
                all_params[f"{mode}_lr"] = float(mr["final_params"]["lr"])
                all_params[f"{mode}_batch_size"] = mr["final_batch"]
                all_params[f"{mode}_epochs_trained"] = mr["training_info"]["epochs_trained"]
                all_params[f"{mode}_best_val_loss"] = mr["training_info"]["best_val_loss"]
                all_params[f"{mode}_dropout"] = mr["dropout"]
                all_params[f"{mode}_gvsao_enabled"] = mr["gvsao_result"] is not None

                if mr["gvsao_result"]:
                    all_params[f"{mode}_gvsao_n_evals"] = mr["gvsao_result"].n_evals
                    all_params[f"{mode}_gvsao_best_fitness"] = mr["gvsao_result"].best_fitness

                all_metrics[f"{mode}_train_time_seconds"] = round(mr["train_time"], 1)

                for pct_label, pct_res in mr["all_results"].items():
                    overall = pct_res.get("overall", {})
                    if overall:
                        all_metrics[f"{mode}_{pct_label}_overall_f1"] = overall.get("f1_score", 0.0)
                        all_metrics[f"{mode}_{pct_label}_overall_precision"] = overall.get("precision", 0.0)
                        all_metrics[f"{mode}_{pct_label}_overall_recall"] = overall.get("recall", 0.0)
                        all_metrics[f"{mode}_{pct_label}_overall_pr_auc"] = overall.get("pr_auc", 0.0)

                    for cls_name, cls_res in pct_res.get("per_class", {}).items():
                        all_metrics[f"{mode}_{pct_label}_{cls_name}_f1"] = cls_res.get("f1_score", 0.0)
                        all_metrics[f"{mode}_{pct_label}_{cls_name}_precision"] = cls_res.get("precision", 0.0)
                        all_metrics[f"{mode}_{pct_label}_{cls_name}_recall"] = cls_res.get("recall", 0.0)

                all_params[f"{mode}_decision_logic"] = profile.get("decision_logic", "group")
                if profile.get("decision_logic") == "group":
                    all_params[f"{mode}_group_logic"] = profile.get("group_logic", "and")

            mlflow.log_params(all_params)
            mlflow.log_metrics(all_metrics)

            for mr in all_mode_results:
                if mr["ckpt_path"].exists():
                    mlflow.log_artifact(str(mr["ckpt_path"]))
                if mr["results_path"].exists():
                    mlflow.log_artifact(str(mr["results_path"]))
                if mr["thresholds_path"].exists():
                    mlflow.log_artifact(str(mr["thresholds_path"]))

            run_id = mlflow.active_run().info.run_id
            logger.success(f"MLflow run logged: {run_name} [{run_id}]")
            mlflow.end_run()
        except Exception as exc:
            logger.warning(f"MLflow logging failed (non-fatal): {exc}")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.success("=" * 60)
    logger.success("GTBAD v2 Experiment Complete")
    for mr in all_mode_results:
        mode = mr["results_payload"]["feature_mode"]
        logger.success(f"  [{mode}] Checkpoint: {mr['ckpt_path']}")
        logger.success(f"  [{mode}] Results: {mr['results_path']}")
        logger.success(f"  [{mode}] Thresholds: {mr['thresholds_path']}")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
