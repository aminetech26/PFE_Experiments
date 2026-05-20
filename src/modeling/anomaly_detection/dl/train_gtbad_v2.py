#!/usr/bin/env python3
"""
GTBAD v2 — Extended experiment: per-feature/group thresholds, GVSAO 7D HPO.

Key additions over train_gtbad.py:
  - Optimises threshold_percentile, d_model/nhead, num_encoder_layers, lstm_hidden via GVSAO
  - Point-wise reconstruction (window_size=1)
  - Per-feature threshold computation at multiple percentiles
  - Group-based (PDC/IDC/VDC AND/OR) and all-feature decision logic
  - Supports original 9-sensor and plus_physics feature profiles
  - Full MLflow → DagsHub integration
  - Primary metric: F1 (PR-AUC as secondary)

All config driven by data_config.yaml (profile) and model_config.yaml.

Usage:
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_v2
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_v2 --mini
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_v2 --feature-mode original
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_v2 --no-gvsao --feature-mode plus_physics
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

EVALUABLE_CLASSES = [0, 1, 2, 3, 4]


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


def split_temporal_mixed(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    gap_samples: int = 300,
) -> dict[str, pd.DataFrame]:
    """Temporal split with autocorrelation-prevention gaps.

    Splits the chronologically-sorted DataFrame into three periods:
      - Train:  healthy rows only from the first `train_frac` of the timeline
      - Val:    all rows (healthy + faults) from the middle `val_frac`
      - Test:   all rows (healthy + faults) from the last portion

    `gap_samples` rows are dropped at each split boundary to prevent
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
    """Normalize per-feature errors by their thresholds.

    Clamps thresholds to a meaningful floor to prevent tiny thresholds from
    causing all samples to be flagged anomalous (e.g. when model is undertrained).
    """
    global_median = float(np.median(np.abs(errors))) if errors.size > 0 else 1e-6
    floor = max(float(np.percentile(thresholds, 10)) if len(thresholds) > 1 else float(thresholds[0]),
                global_median * 0.01, 1e-8)
    thresh = thresholds.copy()
    thresh[thresh < floor] = floor
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
    train_frac = float(profile.get("train_frac", 0.60))
    val_frac = float(profile.get("val_frac", 0.20))
    gap_samples = int(profile.get("split_gap_samples", 300))
    splits = split_temporal_mixed(df, train_frac=train_frac, val_frac=val_frac, gap_samples=gap_samples)

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

    # ── Architecture defaults (from gtbad_model.py) ──────────────────────
    ARCH_D_MODEL = 64
    ARCH_NHEAD = 2
    ARCH_NUM_ENCODER_LAYERS = 3
    ARCH_LSTM_HIDDEN = 32
    FITNESS_THRESHOLD_PCT = 95

    # ── GVSAO config (lr + batch_size only) ───────────────────────────────
    gvsao_cfg = model_cfg.get("gvsao", {})
    param_defs = [
        ParamDef(name="lr", kind="continuous_log", bounds=tuple(gvsao_cfg.get("lr_bounds", [1e-5, 1e-1]))),
        ParamDef(name="batch_size", kind="discrete", candidates=list(range(
            int(gvsao_cfg.get("batch_bounds", [16, 128])[0]),
            int(gvsao_cfg.get("batch_bounds", [16, 128])[1]) + 1,
        ))),
    ]

    pop_size = int(gvsao_cfg.get("population_size", 10))
    max_gens = int(gvsao_cfg.get("max_generations", 5))
    gvsao_epochs = int(gvsao_cfg.get("gvsao_epochs", 5))
    final_epochs = int(profile.get("epochs", model_cfg.get("epochs", 50)))
    final_patience = int(profile.get("patience", model_cfg.get("patience", 15)))

    if args.mini:
        pop_size = 4
        max_gens = 3
        gvsao_epochs = 5
        final_epochs = 5
        final_patience = 3
        logger.info("  MINI RUN: reduced budget (pop={}, gen={}, gvsao_ep={}, final_ep={})",
                    pop_size, max_gens, gvsao_epochs, final_epochs)

    gvsao_v2_config = GVSaoV2Config(
        param_defs=param_defs,
        population_size=pop_size,
        max_generations=max_gens,
        seed=args.seed,
    )

    dropout = float(profile.get("dropout", model_cfg.get("dropout", 0.1)))
    WINDOW_SIZE = 1  # point-wise reconstruction

    # ── Prepare point-wise tensors ───────────────────────────────────────
    tensors: dict[str, dict] = {}
    for split_name, arr in scaled.items():
        X_w = arr["X"][:, np.newaxis, :]  # (n, 1, D)
        tensors[split_name] = {
            "X": torch.from_numpy(X_w),
            "labels": arr["labels"],
        }

    # ── GVSAO fitness (macro F1 over all evaluable classes) ──────────────
    def fitness_fn(params: dict[str, Any]) -> float:
        model = build_model(
            n_features=n_features,
            d_model=ARCH_D_MODEL,
            nhead=ARCH_NHEAD,
            num_encoder_layers=ARCH_NUM_ENCODER_LAYERS,
            lstm_hidden=ARCH_LSTM_HIDDEN,
            dropout=dropout,
        ).to(device)

        X_tr = tensors["train"]["X"]
        X_v = tensors["val"]["X"]
        effective_batch = min(params["batch_size"], X_tr.shape[0])

        train_model(
            model, X_tr, X_v, device,
            lr=params["lr"], batch_size=effective_batch,
            epochs=gvsao_epochs, patience=3, verbose=False,
        )

        val_labels = tensors["val"]["labels"]
        true_bin = (val_labels > 0).astype(int)

        if threshold_mode == "single":
            train_scalar = evaluate_reconstruction_scalar(model, X_tr, device, effective_batch)
            val_scalar = evaluate_reconstruction_scalar(model, X_v, device, effective_batch)
            t = np.percentile(train_scalar, FITNESS_THRESHOLD_PCT)
            val_preds = (val_scalar > t).astype(int)
        else:
            train_errs = evaluate_reconstruction_per_feature(model, X_tr, device, effective_batch)
            val_errs = evaluate_reconstruction_per_feature(model, X_v, device, effective_batch)
            thresholds = np.percentile(train_errs, FITNESS_THRESHOLD_PCT, axis=0)
            norm_val = compute_normalized_errors(val_errs, thresholds)
            val_preds = (anomaly_score(norm_val) > 1.0).astype(int)

        # Per-class F1, averaged equally across all evaluable classes
        per_class_f1: list[float] = []
        for cls in EVALUABLE_CLASSES:
            mask = val_labels == cls
            if int(mask.sum()) == 0:
                continue
            cls_preds = val_preds[mask]
            cls_true = true_bin[mask]
            tp_c = int(np.sum((cls_preds == 1) & (cls_true == 1)))
            fp_c = int(np.sum((cls_preds == 1) & (cls_true == 0)))
            fn_c = int(np.sum((cls_preds == 0) & (cls_true == 1)))
            p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
            per_class_f1.append(f1_c)

        macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0

        return -macro_f1  # GVSAO minimises → return negative macro F1

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
        }

    # ── Final training ───────────────────────────────────────────────────
    X_train_final = tensors["train"]["X"]
    X_val_final = tensors["val"]["X"]
    final_batch = min(int(final_params["batch_size"]), X_train_final.shape[0])

    logger.info(f"  Final training: W={WINDOW_SIZE}, lr={final_params['lr']:.6f}, batch={final_batch}")

    model = build_model(
        n_features=n_features,
        d_model=ARCH_D_MODEL,
        nhead=ARCH_NHEAD,
        num_encoder_layers=ARCH_NUM_ENCODER_LAYERS,
        lstm_hidden=ARCH_LSTM_HIDDEN,
        dropout=dropout,
    ).to(device)

    t0 = time.perf_counter()
    model, training_info = train_model(
        model, X_train_final, X_val_final, device,
        lr=float(final_params["lr"]), batch_size=final_batch,
        epochs=final_epochs, patience=final_patience, verbose=True,
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
        "window_size": WINDOW_SIZE,
        "scaler_min": scaler.min_.tolist(),
        "scaler_max": scaler.max_.tolist(),
        "final_params": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in final_params.items()},
        "seed": args.seed,
    }, ckpt_path)
    logger.success(f"  Checkpoint → {ckpt_path}")

    # ── Threshold computation ────────────────────────────────────────────
    percentiles = profile.get("threshold_percentiles", [90, 95, 99])
    threshold_mode = args.threshold_mode

    if threshold_mode == "single":
        # Single scalar threshold (original GTBAD approach)
        train_scalar = evaluate_reconstruction_scalar(model, X_train_final, device, final_batch)
        train_scores = train_scalar  # raw MSE per sample
        thresholds_dict = {str(pct): np.percentile(train_scalar, pct) for pct in percentiles}
        logger.info(f"  Single-scalar thresholds computed at percentiles: {percentiles}")
    else:
        train_per_feature = evaluate_reconstruction_per_feature(model, X_train_final, device, final_batch)
        thresholds_dict = compute_per_feature_thresholds(train_per_feature, percentiles)
        logger.info(f"  Per-feature thresholds computed at percentiles: {percentiles}")

    # ── Evaluation ───────────────────────────────────────────────────────
    decision_logic = profile.get("decision_logic", "group")
    group_logic = profile.get("group_logic", "and")
    feature_groups = profile.get("feature_groups", {})

    all_results: dict[str, dict] = {}
    val_tensor = tensors["val"]["X"]
    val_labels_arr = tensors["val"]["labels"]
    test_tensor = tensors["test"]["X"]
    test_labels_arr = tensors["test"]["labels"]

    for pct_key, threshold_val in thresholds_dict.items():
        pct_label = f"pct_{pct_key}"

        if threshold_mode == "single":
            # Single scalar threshold
            val_scalar = evaluate_reconstruction_scalar(model, val_tensor, device, final_batch)
            test_scalar = evaluate_reconstruction_scalar(model, test_tensor, device, final_batch)
            val_preds = (val_scalar > threshold_val).astype(int)
            test_preds = (test_scalar > threshold_val).astype(int)
            val_scores = val_scalar
            test_scores = test_scalar

            def _class_results(preds, scores, labels):
                overall = evaluate_binary(preds, labels)
                overall["pr_auc"] = round(evaluate_pr_auc(scores, labels), 6)
                per_class = {}
                for cls in EVALUABLE_CLASSES:
                    mask = labels == cls
                    if int(mask.sum()) == 0:
                        continue
                    cls_res = evaluate_binary(preds[mask], labels[mask])
                    cls_res["pr_auc"] = round(evaluate_pr_auc(scores[mask], labels[mask]), 6)
                    per_class[FAULT_NAMES[cls]] = cls_res
                return overall, per_class

            val_overall, val_per_class = _class_results(val_preds, val_scores, val_labels_arr)
            test_overall, test_per_class = _class_results(test_preds, test_scores, test_labels_arr)

            all_results[pct_label] = {
                "threshold_percentile": float(pct_key),
                "threshold": float(threshold_val),
                "val": {"overall": val_overall, "per_class": val_per_class},
                "test": {"overall": test_overall, "per_class": test_per_class},
            }
        else:
            pct_thresholds = threshold_val  # per-feature: np.ndarray
            val_errors = evaluate_reconstruction_per_feature(model, val_tensor, device, final_batch)
            test_errors = evaluate_reconstruction_per_feature(model, test_tensor, device, final_batch)

            def _eval_split(errors, labels, thresholds):
                scores = anomaly_score(compute_normalized_errors(errors, thresholds))
                if decision_logic == "group":
                    grp = group_anomaly_score(errors, thresholds, sensor_cols_present, feature_groups)
                    preds = decide_group(grp, group_logic)
                else:
                    preds = decide_all_features(errors, thresholds)
                overall = evaluate_binary(preds, labels)
                overall["pr_auc"] = round(evaluate_pr_auc(scores, labels), 6)
                per_class = {}
                for cls in EVALUABLE_CLASSES:
                    mask = labels == cls
                    if int(mask.sum()) == 0:
                        continue
                    cls_res = evaluate_binary(preds[mask], labels[mask])
                    cls_res["pr_auc"] = round(evaluate_pr_auc(scores[mask], labels[mask]), 6)
                    per_class[FAULT_NAMES[cls]] = cls_res
                return overall, per_class

            val_overall, val_per_class = _eval_split(val_errors, val_labels_arr, pct_thresholds)
            test_overall, test_per_class = _eval_split(test_errors, test_labels_arr, pct_thresholds)

            all_results[pct_label] = {
                "threshold_percentile": float(pct_key),
                "thresholds": pct_thresholds.tolist(),
                "val": {"overall": val_overall, "per_class": val_per_class},
                "test": {"overall": test_overall, "per_class": test_per_class},
            }

        logger.info(
            f"  [{pct_label}] Val  F1={val_overall['f1_score']:.4f} "
            f"P={val_overall['precision']:.4f} R={val_overall['recall']:.4f} "
            f"PR-AUC={val_overall.get('pr_auc', 0):.4f}"
        )
        logger.info(
            f"  [{pct_label}] Test F1={test_overall['f1_score']:.4f} "
            f"P={test_overall['precision']:.4f} R={test_overall['recall']:.4f} "
            f"PR-AUC={test_overall.get('pr_auc', 0):.4f}"
        )

    # ── Assemble results payload ─────────────────────────────────────────
    results_payload = {
        "model": f"GTBAD v2 — {feature_mode}",
        "dataset": "Costa PV Fault Dataset",
        "feature_mode": feature_mode,
        "threshold_mode": threshold_mode,
        "decision_logic": decision_logic if threshold_mode != "single" else None,
        "group_logic": group_logic if (threshold_mode != "single" and decision_logic == "group") else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "window_size": WINDOW_SIZE,
        "input_features": sensor_cols_present,
        "n_features": n_features,
        "split": {
            "train_frac": train_frac,
            "val_frac": val_frac,
            "gap_samples": gap_samples,
        },
        "model_config": {
            "d_model": ARCH_D_MODEL,
            "nhead": ARCH_NHEAD,
            "num_encoder_layers": ARCH_NUM_ENCODER_LAYERS,
            "lstm_hidden": ARCH_LSTM_HIDDEN,
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
        "final_window_size": WINDOW_SIZE,
        "all_results": all_results,
        "dropout": dropout,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GTBAD v2 with extended HPO and thresholding")
    parser.add_argument("--parquet-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-mode", type=str, default="all",
                        choices=["original", "plus_physics", "all"],
                        help="Which feature set to train (default: both)")
    parser.add_argument("--threshold-mode", type=str, default="per_feature",
                        choices=["per_feature", "single"],
                        help="Threshold mode: per_feature (default) or single scalar")
    parser.add_argument("--mini", action="store_true",
                        help="Mini run: reduced GVSAO budget and epochs for quick validation")
    parser.add_argument("--no-gvsao", action="store_true", help="Skip GVSAO HPO")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    # Fallback values when --no-gvsao is used
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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
                "threshold_mode": args.threshold_mode,
                "seed": str(args.seed),
            })
            logger.info("MLflow tracking active")
        except Exception as exc:
            logger.warning(f"MLflow init failed (non-fatal): {exc}")

    # ── Run selected feature modes ────────────────────────────────────────
    all_mode_results: list[dict] = []
    modes_to_run = ["original", "plus_physics"] if args.feature_mode == "all" else [args.feature_mode]

    for mode in modes_to_run:
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
                all_params[f"{mode}_d_model"] = 64
                all_params[f"{mode}_nhead"] = 2
                all_params[f"{mode}_num_encoder_layers"] = 3
                all_params[f"{mode}_lstm_hidden"] = 32
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
                    for split_name in ["val", "test"]:
                        split_data = pct_res.get(split_name, {})
                        overall = split_data.get("overall", {})
                        if overall:
                            all_metrics[f"{mode}_{pct_label}_{split_name}_f1"] = overall.get("f1_score", 0.0)
                            all_metrics[f"{mode}_{pct_label}_{split_name}_precision"] = overall.get("precision", 0.0)
                            all_metrics[f"{mode}_{pct_label}_{split_name}_recall"] = overall.get("recall", 0.0)
                            all_metrics[f"{mode}_{pct_label}_{split_name}_pr_auc"] = overall.get("pr_auc", 0.0)
                            all_metrics[f"{mode}_{pct_label}_{split_name}_TP"] = float(overall.get("TP", 0))
                            all_metrics[f"{mode}_{pct_label}_{split_name}_FP"] = float(overall.get("FP", 0))
                            all_metrics[f"{mode}_{pct_label}_{split_name}_FN"] = float(overall.get("FN", 0))
                            all_metrics[f"{mode}_{pct_label}_{split_name}_TN"] = float(overall.get("TN", 0))

                        for cls_name, cls_res in split_data.get("per_class", {}).items():
                            all_metrics[f"{mode}_{pct_label}_{split_name}_{cls_name}_f1"] = cls_res.get("f1_score", 0.0)
                            all_metrics[f"{mode}_{pct_label}_{split_name}_{cls_name}_precision"] = cls_res.get("precision", 0.0)
                            all_metrics[f"{mode}_{pct_label}_{split_name}_{cls_name}_recall"] = cls_res.get("recall", 0.0)

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
