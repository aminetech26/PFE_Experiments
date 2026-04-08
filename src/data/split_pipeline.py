"""
Split Pipeline — Generates all task-specific splits.

Outputs to data/interim/splits/:
  - anomaly_semisup/    (Task A semi-supervised)
  - anomaly_supervised/ (Task A supervised)  
  - classification/     (Task B, evaluable classes only)

Usage:
    python -m src.data.split_pipeline
    
    # Or from project root:
    cd PFE_Experiments && python -m src.data.split_pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import yaml
from loguru import logger

from src.data.splitting import (
    segment_stratified_split,
    hybrid_semisup_split,
    filter_to_evaluable_classes,
    prediction_episode_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"


def load_config() -> dict:
    """Load split configuration from data_config.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_segment_data(config: dict) -> pd.DataFrame:
    """
    Load merged data and apply segmentation.
    
    Returns DataFrame with 'timestamp', 'label', 'segment_id' columns.
    """
    paths = config["paths"]
    split_cfg = config["splits"]
    label_col = "Fault"
    
    interim_dir = PROJECT_ROOT / paths["interim_dir"]
    input_path = interim_dir / "reunion_dt2_merged.parquet"
    
    logger.info(f"Loading data from: {input_path}")
    df = pl.read_parquet(input_path).to_pandas()
    
    # Standardize column names
    df["timestamp"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if label_col in df.columns:
        df["label"] = df[label_col]
    elif "label" not in df.columns:
        raise KeyError(f"Missing required label column '{label_col}' and no fallback 'label' column found.")
    df = df.dropna(subset=["timestamp", "label"]).sort_values("timestamp").reset_index(drop=True)
    
    logger.info(f"Loaded {len(df):,} rows")

    # Contiguous segmentation
    segmentation_gap_seconds = int(split_cfg.get("segmentation_gap_seconds", 300))
    dt_s = df["timestamp"].diff().dt.total_seconds().fillna(0)
    df["segment_id"] = (dt_s > segmentation_gap_seconds).cumsum().astype(int)
    
    n_segments = df["segment_id"].nunique()
    logger.info(f"Created {n_segments} segments (gap threshold: {segmentation_gap_seconds}s)")
    
    return df


def save_split(artifacts, output_dir: Path, split_name: str) -> None:
    """Save split artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    artifacts.train.to_parquet(output_dir / "train.parquet", index=False)
    artifacts.val.to_parquet(output_dir / "val.parquet", index=False)
    artifacts.test.to_parquet(output_dir / "test.parquet", index=False)
    
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(artifacts.manifest, f, indent=2, default=str)
    
    logger.success(
        f"{split_name}: train={len(artifacts.train):,}, "
        f"val={len(artifacts.val):,}, test={len(artifacts.test):,}"
    )


def run_anomaly_semisup_split(df: pd.DataFrame, config: dict, output_base: Path) -> None:
    """Generate hybrid split for semi-supervised anomaly detection."""
    logger.info("=" * 50)
    logger.info("[1/3] Anomaly Detection — Semi-Supervised (Temporal Hybrid)")
    logger.info("Train: Normal-only (temporal) | Val/Test: Normal + faults (temporal)")
    
    split_cfg = config.get("splits", {})
    
    artifacts = hybrid_semisup_split(
        df=df,
        segment_col="segment_id",
        label_col="label",
        time_col="timestamp",
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        embargo_seconds=split_cfg.get("embargo_seconds", 1260),
    )
    
    save_split(artifacts, output_base / "anomaly_semisup", "anomaly_semisup")
    
    # Verify train is normal-only
    train_labels = artifacts.train["label"].unique()
    assert all(l == 0.0 for l in train_labels), "Train should be normal-only!"
    logger.info(f"Verified: Train contains only normal data ({len(artifacts.train):,} rows)")


def run_anomaly_supervised_split(df: pd.DataFrame, config: dict, output_base: Path) -> None:
    """Generate temporal-stratified split for supervised anomaly detection."""
    logger.info("=" * 50)
    logger.info("[2/3] Anomaly Detection — Supervised (Temporal-Stratified)")
    logger.info("All sets: Temporal order preserved within each class")
    
    split_cfg = config.get("splits", {})
    
    artifacts = segment_stratified_split(
        df=df,
        segment_col="segment_id",
        label_col="label",
        time_col="timestamp",
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        embargo_seconds=split_cfg.get("embargo_seconds", 1260),
    )
    
    save_split(artifacts, output_base / "anomaly_supervised", "anomaly_supervised")


def run_classification_split(df: pd.DataFrame, config: dict, output_base: Path) -> None:
    """Generate segment-stratified split for fault classification (evaluable classes only)."""
    logger.info("=" * 50)
    logger.info("[3/3] Fault Classification (Temporal-Stratified, Evaluable Classes)")
    logger.info("Classes: 3.1, 3.2, 4.0 only (no normal, no 1.0/2.x)")
    
    split_cfg = config.get("splits", {})
    evaluable_classes = [3.1, 3.2, 4.0]
    
    # First, do temporal-stratified split on full data
    artifacts = segment_stratified_split(
        df=df,
        segment_col="segment_id",
        label_col="label",
        time_col="timestamp",
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        embargo_seconds=split_cfg.get("embargo_seconds", 1260),
    )
    
    # Filter to evaluable classes only
    train_filtered = filter_to_evaluable_classes(artifacts.train, evaluable_classes)
    val_filtered = filter_to_evaluable_classes(artifacts.val, evaluable_classes)
    test_filtered = filter_to_evaluable_classes(artifacts.test, evaluable_classes)
    
    output_dir = output_base / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_filtered.to_parquet(output_dir / "train.parquet", index=False)
    val_filtered.to_parquet(output_dir / "val.parquet", index=False)
    test_filtered.to_parquet(output_dir / "test.parquet", index=False)
    
    # Update manifest for filtered data
    manifest = {
        "split_type": "temporal_stratified_classification",
        "evaluable_classes": evaluable_classes,
        "train_rows": len(train_filtered),
        "val_rows": len(val_filtered),
        "test_rows": len(test_filtered),
        "train_class_counts": train_filtered["label"].value_counts().sort_index().to_dict(),
        "val_class_counts": val_filtered["label"].value_counts().sort_index().to_dict(),
        "test_class_counts": test_filtered["label"].value_counts().sort_index().to_dict(),
        "note": "Temporal order preserved. For use after Task A flags anomalies.",
    }
    
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    
    logger.success(
        f"classification: train={len(train_filtered):,}, "
        f"val={len(val_filtered):,}, test={len(test_filtered):,}"
    )
    
    # Report class distribution
    logger.info("Class distribution in test set:")
    for cls in evaluable_classes:
        count = (test_filtered["label"] == cls).sum()
        logger.info(f"  {cls}: {count:,} samples")


def run_prediction_split(df: pd.DataFrame, config: dict, output_base: Path) -> None:
    """Generate episode-based split for fault prediction (Task C)."""
    logger.info("=" * 50)
    logger.info("[4/4] Fault Prediction — Episode-Based (Task C)")
    logger.info("Episodes split temporally. Pre-fault zones go with their episode.")
    
    split_cfg = config.get("splits", {})
    prediction_cfg = config.get("prediction", {})
    
    # Get forecast horizon from config or use default (30 min)
    forecast_horizon = prediction_cfg.get("forecast_horizon_seconds", 1800)
    
    artifacts = prediction_episode_split(
        df=df,
        segment_col="segment_id",
        label_col="label",
        time_col="timestamp",
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        forecast_horizon_seconds=forecast_horizon,
    )
    
    save_split(artifacts, output_base / "prediction", "prediction")
    
    # Report episode distribution
    manifest = artifacts.manifest
    logger.info(f"Total fault episodes: {manifest['n_fault_episodes']}")
    logger.info(f"Train episodes: {manifest['train_fault_episodes']}, Test episodes: {manifest['test_fault_episodes']}")
    logger.info(f"Train pre-fault samples: {manifest['train_prefault_samples']:,}")
    logger.info(f"Test pre-fault samples: {manifest['test_prefault_samples']:,}")
    
    # Show per-class breakdown
    logger.info("Episodes by class:")
    for cls, info in manifest["episodes_by_class"].items():
        status = "✓ evaluable" if info["evaluable"] else "✗ train-only"
        logger.info(f"  {cls}: {info['n_episodes']} episodes ({status})")


def main() -> None:
    """Run all split pipelines."""
    logger.info("=" * 60)
    logger.info("SPLIT PIPELINE — Task-Specific Splits")
    logger.info("=" * 60)
    
    config = load_config()
    df = load_and_segment_data(config)
    
    output_base = PROJECT_ROOT / "data" / "interim" / "splits"
    logger.info(f"Output directory: {output_base}")
    
    # Generate all splits
    run_anomaly_semisup_split(df, config, output_base)
    run_anomaly_supervised_split(df, config, output_base)
    run_classification_split(df, config, output_base)
    run_prediction_split(df, config, output_base)
    
    logger.info("=" * 60)
    logger.success("All splits generated successfully!")
    logger.info("=" * 60)
    
    # Summary
    logger.info("Generated splits:")
    logger.info("  • anomaly_semisup/    — Task A semi-supervised (train=normal-only)")
    logger.info("  • anomaly_supervised/ — Task A supervised (temporal-stratified)")
    logger.info("  • classification/     — Task B (evaluable classes only)")
    logger.info("  • prediction/         — Task C (episode-based, pre-fault zones)")


if __name__ == "__main__":
    main()
