"""
Split Pipeline — Generates all task-specific splits.

Outputs to data/interim/splits/<dataset>/:
  - anomaly_semisup/    (Task A semi-supervised)
  - anomaly_supervised/ (Task A supervised)
  - classification/     (Task B, evaluable classes only)

Usage:
    python -m src.data.split_pipeline                    # reunion (default)
    python -m src.data.split_pipeline --dataset costa
    python -m src.data.split_pipeline --dataset mendeley
"""

from __future__ import annotations

import argparse
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
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"


def load_config() -> dict:
    """Load split configuration from data_config.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_dataset(config: dict) -> str:
    return str(config.get("active_dataset", "la_reunion"))


def load_and_segment_data(config: dict, dataset: str = "reunion") -> pd.DataFrame:
    """
    Load merged data for the specified dataset and apply segmentation.

    Reads the dataset-specific merged parquet from data/interim/ and applies
    contiguous segmentation (gap > segmentation_gap_seconds → new segment).

    Returns DataFrame with standardised columns: timestamp | label | segment_id | <sensors>
    """
    paths = config["paths"]
    split_cfg = config["splits"]

    # Resolve dataset config from registry
    dataset_cfg = paths.get("datasets", {}).get(dataset)
    if dataset_cfg is None:
        raise KeyError(
            f"Dataset '{dataset}' not found in data_config.yaml paths.datasets. "
            f"Available: {list(paths.get('datasets', {}).keys())}"
        )

    interim_dir = PROJECT_ROOT / paths["interim_dir"]
    input_path = interim_dir / dataset_cfg["merged_output"]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Merged parquet not found: {input_path}\n"
            f"Run: uv run python -m src.data.ingestion --dataset {dataset}"
        )

    logger.info(f"Loading dataset={dataset} from: {input_path}")
    df = pl.read_parquet(input_path).to_pandas()

    # Standardize timestamp column
    ts_col = dataset_cfg.get("timestamp_col", "time")
    if ts_col not in df.columns:
        raise KeyError(
            f"Timestamp column '{ts_col}' not found. Check paths.datasets.{dataset}.timestamp_col"
        )
    df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    # Standardize label column
    label_col = dataset_cfg.get("label_col", "label")
    if label_col in df.columns:
        df["label"] = df[label_col]
    elif "label" not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found. "
            f"Check paths.datasets.{dataset}.label_col in data_config.yaml"
        )

    df = df.dropna(subset=["timestamp", "label"]).sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(df):,} rows | dataset={dataset}")

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


def main() -> None:
    """Run all split pipelines for a given dataset."""
    config = load_config()
    default_dataset = get_active_dataset(config)
    parser = argparse.ArgumentParser(description="Generate task-specific splits for a PV dataset")
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help=f"Which dataset to split (must match a key in data_config.yaml paths.datasets). Default: {default_dataset}",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SPLIT PIPELINE — dataset={}", args.dataset)
    logger.info("=" * 60)

    df = load_and_segment_data(config, dataset=args.dataset)

    # Dataset-namespaced output directory so different datasets don't clobber each other
    output_base = PROJECT_ROOT / "data" / "interim" / "splits" / args.dataset
    logger.info(f"Output directory: {output_base}")

    # Task A splits (FDD — Fault Detection)
    run_anomaly_semisup_split(df, config, output_base)
    run_anomaly_supervised_split(df, config, output_base)

    # Task B split (FDD — Fault Diagnosis)
    run_classification_split(df, config, output_base)

    logger.info("=" * 60)
    logger.success("All splits generated | dataset={}", args.dataset)
    logger.info("=" * 60)

    logger.info("Generated splits:")
    logger.info("  • anomaly_semisup/    — Task A semi-supervised (train=normal-only)")
    logger.info("  • anomaly_supervised/ — Task A supervised (temporal-stratified)")
    logger.info("  • classification/     — Task B (evaluable classes only)")


if __name__ == "__main__":
    main()
