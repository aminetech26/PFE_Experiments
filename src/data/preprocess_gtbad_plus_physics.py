#!/usr/bin/env python3
"""
GTBAD Plus-Physics Preprocessing — Costa dataset.

Applies the plus_physics feature-engineering profile (derivatives, imbalances,
temperature-corrected power) to the raw Costa ingestion output, then prepares
GTBAD-ready tensors with strict data-leakage prevention.

Outputs:
  - data/processed/preprocessed/costa_gtbad_pp/
    ├── gtbad_pp_data.npz        (X_train, X_val, labels_train, labels_val, X_test, labels_test)
    └── gtbad_pp_metadata.json   (scaler params, feature names, split info)

Usage:
    uv run python -m src.data.preprocess_gtbad_plus_physics
    uv run python -m src.data.preprocess_gtbad_plus_physics --train-frac 0.85
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.data.features import (
    add_physics_features,
    apply_hygiene_pruning,
    apply_mrmr_selection,
    infer_label_column,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"
DEFAULT_PARQUET_PATH = PROJECT_ROOT / "data" / "interim" / "ingestion" / "costa" / "costa_merged.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_gtbad_pp"

COSTA_FEATURE_COLS = ["vdc1", "vdc2", "idc1", "idc2", "pdc1", "pdc2", "pdc", "irr", "pvt"]

PLUS_PHYSICS_FLAGS = {
    "enable_delta_temp": False,
    "enable_dP_dt": True,
    "enable_dV_dt": True,
    "enable_dI_dt": True,
    "enable_Vg_normalized": False,
    "enable_power_imbalance": True,
    "enable_current_imbalance": True,
    "enable_voltage_imbalance": True,
    "enable_string_share": False,
    "enable_temp_power_correction": True,
    "temp_ref_c": 25.0,
    "gamma_pmax_pct_per_c": -0.40,
    "temp_power_eps": 1e-8,
    "irr_norm_floor": 1.0,
    "enable_hygiene_pruning": True,
    "enable_mrmr_selection": True,
    "mrmr_k": 64,
}

EVALUABLE_CLASSES = [1, 2, 3, 4]

FAULT_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}


def load_config() -> dict:
    with open(DATA_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Ingested parquet not found: {parquet_path}\n"
            "Run: uv run python -m src.data.ingestion --dataset costa"
        )
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    logger.info(f"Loaded: {len(df):,} rows, {df.shape[1]} columns")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        logger.info(f"  {FAULT_NAMES.get(int(lbl), '?')} ({int(lbl)}): {cnt:,}")
    return df


def split_gtbad(
    df: pd.DataFrame,
    healthy_train_frac: float = 0.80,
) -> dict[str, pd.DataFrame]:
    healthy = df[df["label"] == 0].copy()
    faulty = df[df["label"] > 0].copy()
    split_idx = int(len(healthy) * healthy_train_frac)
    train_df = healthy.iloc[:split_idx].copy()
    val_df = healthy.iloc[split_idx:].copy()
    result = {"train": train_df, "val": val_df, "test": faulty}
    logger.info(f"  Train (healthy): {len(train_df):,}")
    logger.info(f"  Val   (healthy): {len(val_df):,}")
    logger.info(f"  Test  (faulty):   {len(faulty):,}")
    for cls in EVALUABLE_CLASSES:
        count = int((faulty["label"] == cls).sum())
        if count > 0:
            logger.info(f"    └─ {FAULT_NAMES[cls]}: {count:,}")
    return result


class MinMaxScaler:
    def fit(self, data: np.ndarray) -> "MinMaxScaler":
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        self.range_ = self.max_ - self.min_
        self.range_[self.range_ < 1e-10] = 1.0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.min_) / self.range_

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess GTBAD plus_physics data")
    parser.add_argument("--parquet-path", type=str, default=str(DEFAULT_PARQUET_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-frac", type=float, default=0.80)
    args = parser.parse_args()

    parquet_path = Path(args.parquet_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Step 1: Load raw Costa data ===")
    df = load_raw_data(parquet_path)

    logger.info("=== Step 2: Apply plus_physics feature engineering ===")
    physics_flags = {k: v for k, v in PLUS_PHYSICS_FLAGS.items() if k.startswith("enable_") or k in ("temp_ref_c", "gamma_pmax_pct_per_c", "temp_power_eps", "irr_norm_floor")}
    df = add_physics_features(df, segment_col="segment_id", time_col="timestamp", flags=physics_flags)
    all_feature_cols = [c for c in df.columns if c not in ("label", "segment_id", "timestamp", "Fault")]

    labelled_feature_cols = [c for c in all_feature_cols if df[c].dtype in (np.float32, np.float64, np.int32, np.int64)]
    logger.info(f"  Features after physics engineering: {len(labelled_feature_cols)}")

    logger.info("=== Step 3: GTBAD split (healthy train/val, faulty test) ===")
    splits = split_gtbad(df, healthy_train_frac=args.train_frac)

    hygiene_dropped: list[dict] = []
    mrmr_dropped: list[dict] = []

    logger.info("=== Step 4: Hygiene pruning (train-only) ===")
    if PLUS_PHYSICS_FLAGS.get("enable_hygiene_pruning"):
        present_cols = [c for c in labelled_feature_cols if c in splits["train"].columns]
        (splits["train"], splits["val"], splits["test"],
         selected_features, hygiene_dropped) = apply_hygiene_pruning(
            splits["train"], splits["val"], splits["test"],
            present_cols, near_constant_std=1e-10,
        )
        for drop in hygiene_dropped:
            logger.info(f"  Dropped: {drop['dropped']} ({drop['reason']})")
        logger.info(f"  Features after hygiene: {len(selected_features)}")
    else:
        selected_features = [c for c in labelled_feature_cols if c in splits["train"].columns]

    logger.info("=== Step 5: MRMR selection (train-only) ===")
    if PLUS_PHYSICS_FLAGS.get("enable_mrmr_selection") and selected_features:
        label_col = infer_label_column(splits["train"])
        mrmr_k = int(PLUS_PHYSICS_FLAGS.get("mrmr_k", 64))
        if mrmr_k < len(selected_features):
            (splits["train"], splits["val"], splits["test"],
             selected_features, mrmr_dropped) = apply_mrmr_selection(
                splits["train"], splits["val"], splits["test"],
                selected_features, label_col=label_col, k=mrmr_k,
            )
            logger.info(f"  Features after MRMR: {len(selected_features)}")
            for drop in mrmr_dropped:
                logger.info(f"  Dropped: {drop.get('dropped', drop)}")
        else:
            logger.info(f"  MRMR skipped: k={mrmr_k} >= n_features={len(selected_features)}")

    logger.info(f"  Final feature count: {len(selected_features)}")
    logger.info(f"  Features: {selected_features}")

    logger.info("=== Step 6: MinMax scale & create tensors ===")
    scaler = MinMaxScaler()
    X_train_np = scaler.fit_transform(splits["train"][selected_features].values.astype(np.float32))
    X_val_np = scaler.transform(splits["val"][selected_features].values.astype(np.float32))
    X_test_np = scaler.transform(splits["test"][selected_features].values.astype(np.float32))

    labels_train = splits["train"]["label"].values.astype(np.int32)
    labels_val = splits["val"]["label"].values.astype(np.int32)
    labels_test = splits["test"]["label"].values.astype(np.int32)

    X_train = X_train_np[:, np.newaxis, :]
    X_val = X_val_np[:, np.newaxis, :]
    X_test = X_test_np[:, np.newaxis, :]

    logger.info(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    logger.info("=== Step 7: Save preprocessed data ===")
    npz_path = output_dir / "gtbad_pp_data.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        labels_train=labels_train,
        labels_val=labels_val,
        labels_test=labels_test,
    )
    logger.info(f"  Saved: {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.1f} MB)")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_parquet": str(parquet_path),
        "profile": "plus_physics",
        "feature_flags": {k: v for k, v in PLUS_PHYSICS_FLAGS.items()},
        "feature_names": selected_features,
        "n_features": len(selected_features),
        "scaler_min": scaler.min_.tolist(),
        "scaler_max": scaler.max_.tolist(),
        "scaler_range": scaler.range_.tolist(),
        "train_frac": args.train_frac,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "fault_classes_present": [int(c) for c in EVALUABLE_CLASSES if int((labels_test == c).sum()) > 0],
        "hygiene_dropped": [d["dropped"] for d in hygiene_dropped],
        "mrmr_dropped": [d if isinstance(d, str) else d.get("dropped", str(d)) for d in mrmr_dropped],
    }

    meta_path = output_dir / "gtbad_pp_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.success(f"  Saved: {meta_path}")
    logger.success("Preprocessing complete.")


if __name__ == "__main__":
    main()
