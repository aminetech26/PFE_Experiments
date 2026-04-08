"""
Preprocessing Pipeline — Runs preprocessing on all split datasets.

Reads from data/interim/splits/{task}/ and outputs to data/processed/preprocessed/{task}/

Usage:
    python -m src.data.preprocess_pipeline
    
    # Or from project root:
    cd PFE_Experiments && python -m src.data.preprocess_pipeline
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from src.data.features import DEFAULT_BASE_FEATURE_COLUMNS
from src.data.preprocessing import preprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"


def load_config() -> dict:
    """Load configuration from data_config.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_cols() -> list[str]:
    """Get default base feature columns used by preprocessing."""
    return list(DEFAULT_BASE_FEATURE_COLUMNS)


def preprocess_split(
    input_dir: Path,
    output_dir: Path,
    split_name: str,
    feature_cols: list[str],
    preprocess_config: dict,
    label_col: str = "Fault",
) -> dict:
    """
    Preprocess a single split (train/val/test).

    Args:
        input_dir: Directory containing train.parquet, val.parquet, test.parquet
        output_dir: Output directory for preprocessed files
        split_name: Name of the split for logging
        feature_cols: Feature columns to preprocess
        preprocess_config: Preprocessing configuration dict
        label_col: Label column name

    Returns:
        Combined statistics dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    total_input_rows = 0
    total_output_rows = 0

    # Track IQR bounds from training set for consistent clipping
    train_bounds = None

    for subset in ["train", "val", "test"]:
        input_path = input_dir / f"{subset}.parquet"
        if not input_path.exists():
            logger.warning(f"  {subset}.parquet not found, skipping")
            continue

        logger.info(f"  Processing {subset}...")
        df = pd.read_parquet(input_path)

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Check for required columns
        missing_cols = set(feature_cols) - set(df.columns)
        if missing_cols:
            logger.warning(f"  Missing feature columns: {missing_cols}")
            available_features = [c for c in feature_cols if c in df.columns]
        else:
            available_features = feature_cols

        # Check for segment_id
        if "segment_id" not in df.columns:
            logger.warning("  No segment_id column, creating single segment")
            df["segment_id"] = 0

        # Check for label column
        if label_col not in df.columns:
            # Try 'label' as fallback
            if "label" in df.columns:
                df[label_col] = df["label"]
            else:
                logger.error(f"  No label column '{label_col}' found!")
                continue

        total_input_rows += len(df)

        # Run preprocessing
        df_processed, stats = preprocess(
            df=df,
            feature_cols=available_features,
            config=preprocess_config,
            timestamp_col="timestamp",
            segment_col="segment_id",
            label_col=label_col,
        )

        total_output_rows += len(df_processed)

        # Save
        output_path = output_dir / f"{subset}.parquet"
        df_processed.to_parquet(output_path, index=False)
        logger.info(f"    Saved: {output_path.name} ({len(df_processed):,} rows)")

        all_stats[subset] = stats

        # Store train bounds for reference
        if subset == "train":
            train_bounds = stats.get("outliers", {}).get("bounds", {})

    return {
        "split_name": split_name,
        "total_input_rows": total_input_rows,
        "total_output_rows": total_output_rows,
        "rows_lost": total_input_rows - total_output_rows,
        "train_bounds": train_bounds,
        "subset_stats": all_stats,
    }


def create_manifest(
    split_stats: dict,
    config: dict,
    output_path: Path,
) -> None:
    """Create preprocessing manifest JSON."""
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_used": config.get("preprocessing", {}),
        "splits": split_stats,
        "features_created": [],
    }

    # Extract features created
    stat_config = config.get("preprocessing", {}).get("stationarity", {})
    if stat_config.get("irradiance_normalize"):
        features = stat_config["irradiance_normalize"].get("features", [])
        suffix = stat_config["irradiance_normalize"].get("suffix", "_norm")
        manifest["features_created"].extend([f"{f}{suffix}" for f in features])

    detrend_cfg = stat_config.get("polynomial_detrend", stat_config.get("linear_detrend"))
    if detrend_cfg:
        features = detrend_cfg.get("features", [])
        suffix = detrend_cfg.get("suffix", "_detrend")
        manifest["features_created"].extend([f"{f}{suffix}" for f in features])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(f"Manifest saved: {output_path}")


def main() -> None:
    """Run preprocessing pipeline on all splits."""
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    config = load_config()
    preprocess_config = config.get("preprocessing", {})
    feature_cols = get_feature_cols()
    label_col = "Fault"

    logger.info(f"Feature columns: {feature_cols}")
    logger.info(f"Label column: {label_col}")

    input_base = PROJECT_ROOT / "data" / "interim" / "splits"
    output_base = PROJECT_ROOT / "data" / "processed" / "preprocessed"

    if not input_base.exists():
        logger.error(f"Input directory not found: {input_base}")
        logger.error("Run split_pipeline.py first!")
        return

    # Define splits to process
    splits = [
        "anomaly_semisup",
        "anomaly_supervised",
        "classification",
        "prediction",
    ]

    all_split_stats = {}

    for split_name in splits:
        input_dir = input_base / split_name
        output_dir = output_base / split_name

        if not input_dir.exists():
            logger.warning(f"Split directory not found: {input_dir}, skipping")
            continue

        logger.info("=" * 50)
        logger.info(f"Processing: {split_name}")

        stats = preprocess_split(
            input_dir=input_dir,
            output_dir=output_dir,
            split_name=split_name,
            feature_cols=feature_cols,
            preprocess_config=preprocess_config,
            label_col=label_col,
        )

        all_split_stats[split_name] = stats

        logger.success(
            f"{split_name}: {stats['total_input_rows']:,} → {stats['total_output_rows']:,} rows "
            f"({stats['rows_lost']:,} lost)"
        )

    # Save global manifest
    manifest_path = output_base / "preprocess_manifest.json"
    create_manifest(all_split_stats, config, manifest_path)

    logger.info("=" * 60)
    logger.success("Preprocessing pipeline complete!")
    logger.info("=" * 60)

    # Summary
    total_in = sum(s["total_input_rows"] for s in all_split_stats.values())
    total_out = sum(s["total_output_rows"] for s in all_split_stats.values())
    logger.info(f"Total rows processed: {total_in:,} → {total_out:,} ({total_in - total_out:,} lost)")


if __name__ == "__main__":
    main()
