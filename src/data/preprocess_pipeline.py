"""
Preprocessing Pipeline — Runs preprocessing on all split datasets.

Reads from either:
- data/interim/splits/<dataset>/{task}/                (Path A)
- data/interim/splits/<dataset>/path_b/{task}/         (Path B)

and outputs to either:
- data/processed/preprocessed/<dataset>/{task}/        (Path A)
- data/processed/preprocessed/<dataset>/path_b/{task}/ (Path B)

Usage:
    python -m src.data.preprocess_pipeline                      # default: la_reunion
    python -m src.data.preprocess_pipeline --dataset costa
    python -m src.data.preprocess_pipeline --dataset mendeley
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
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


def get_active_dataset(config: dict) -> str:
    return str(config.get("active_dataset", "la_reunion"))


def get_feature_cols(config: dict, dataset: str) -> list[str]:
    """Resolve dataset-aware base feature columns used by preprocessing."""
    dataset_cfg = config.get("paths", {}).get("datasets", {}).get(dataset, {})
    preprocess_sensor_cols = dataset_cfg.get("preprocessing", {}).get("sensor_columns")
    if preprocess_sensor_cols:
        return list(preprocess_sensor_cols)

    sensor_cols = dataset_cfg.get("feature_engineering", {}).get("sensor_columns")
    if sensor_cols:
        return list(sensor_cols)
    return list(DEFAULT_BASE_FEATURE_COLUMNS)


def get_label_col(config: dict, dataset: str) -> str:
    """Resolve dataset-aware label column name."""
    dataset_cfg = config.get("paths", {}).get("datasets", {}).get(dataset, {})
    return str(dataset_cfg.get("label_col", "label"))


def _restore_costa_power_channels(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Restore Costa power-channel invariants from cleaned primary sensors."""
    if dataset != "costa":
        return df
    required = {"vdc1", "vdc2", "idc1", "idc2"}
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    out["pdc1"] = out["vdc1"] * out["idc1"]
    out["pdc2"] = out["vdc2"] * out["idc2"]
    out["pdc"] = out["pdc1"] + out["pdc2"]
    return out


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_preprocess_config(config: dict, dataset: str) -> dict:
    base_cfg = config.get("preprocessing", {})
    dataset_cfg = config.get("paths", {}).get("datasets", {}).get(dataset, {})
    dataset_override = dataset_cfg.get("preprocessing", {})
    return deep_merge(base_cfg, dataset_override)


# Preprocessing config is resolved by merge order only (global -> dataset -> path override).


def resolve_effective_preprocess_config(config: dict, dataset: str, split_path: str) -> dict:
    """Resolve dataset and split-path aware preprocessing config.

    This intentionally avoids any legacy key remapping.
    """
    effective = resolve_preprocess_config(config, dataset)
    dataset_cfg = config.get("paths", {}).get("datasets", {}).get(dataset, {})
    path_override = (
        dataset_cfg.get("splits", {})
        .get("comparison_paths", {})
        .get(split_path, {})
        .get("preprocessing", {})
    )
    if path_override:
        effective = deep_merge(effective, path_override)
    return effective


def resolve_io_roots(project_root: Path, dataset: str, split_path: str) -> tuple[Path, Path]:
    input_root = project_root / "data" / "interim" / "splits" / dataset
    output_root = project_root / "data" / "processed" / "preprocessed" / dataset
    if split_path == "path_b":
        input_root = input_root / "path_b"
        output_root = output_root / "path_b"
    return input_root, output_root


def _sanity_check_subset(
    df: pd.DataFrame,
    subset: str,
    split_name: str,
    feature_cols: list[str],
    label_col: str,
) -> None:
    """Per-subset sanity checks. Raises ValueError on hard failures, warns on soft ones."""
    tag = f"[sanity:{split_name}/{subset}]"
    errors: list[str] = []

    # 1. Label integrity
    if bool(df[label_col].isna().any()):
        errors.append(f"{tag} label column contains NaN after preprocessing")
    if len(df) == 0:
        errors.append(f"{tag} output is empty — all rows dropped")

    # 2. No NaN / inf in feature columns
    feat_present = [c for c in feature_cols if c in df.columns]
    nan_counts = df[feat_present].isna().sum()
    nan_feats = nan_counts[nan_counts > 0]
    if not nan_feats.empty:
        errors.append(f"{tag} NaN in features after preprocessing: {nan_feats.to_dict()}")

    inf_mask = df[feat_present].isin([float("inf"), float("-inf")])
    inf_feats = inf_mask.sum()
    inf_feats = inf_feats[inf_feats > 0]
    if not inf_feats.empty:
        errors.append(f"{tag} ±inf in features: {inf_feats.to_dict()}")

    if errors:
        raise ValueError("\n".join(errors))


def _sanity_check_cross_split(
    subsets: dict[str, pd.DataFrame],
    split_name: str,
    split_path: str = "path_a",
) -> None:
    """
    Cross-split checks grounded in the actual split strategy.

    Path A (episode_id atomic units, temporal-stratified):
      - No group (segment_id, episode_id) may appear in more than one partition.
        A fragmented group means a fault event was split across train and val/test — direct leakage.
      - Row-level timestamp ordering is NOT checked: temporal-stratified assignment
        intentionally interleaves group timestamps across partitions.

    Path B (operating_day_id atomic units, chronological blocks):
      - Same group integrity check.
      - Additionally: all train operating_day_id values must be strictly less than
        all val values, which must be strictly less than all test values.
    """
    tag = f"[sanity:{split_name}/cross]"
    errors: list[str] = []

    train = subsets.get("train")
    val = subsets.get("val")
    test = subsets.get("test")

    # Group integrity: no relevant grouping unit fragmented across partitions
    available_columns: set[str] = set()
    for frame in (train, val, test):
        if frame is not None:
            available_columns.update(frame.columns.tolist())

    if split_path == "path_b":
        candidate_group_cols = ("operating_day_id", "segment_id", "episode_id")
    else:
        candidate_group_cols = ("segment_id", "episode_id")

    group_cols = [c for c in candidate_group_cols if c in available_columns]
    for gcol in group_cols:
        train_groups = set(train[gcol].dropna().unique()) if train is not None else set()
        val_groups = set(val[gcol].dropna().unique()) if val is not None else set()
        test_groups = set(test[gcol].dropna().unique()) if test is not None else set()

        tv_overlap = train_groups & val_groups
        tt_overlap = train_groups & test_groups
        vt_overlap = val_groups & test_groups

        if tv_overlap:
            errors.append(
                f"{tag} {gcol} fragmentation: {len(tv_overlap)} group(s) span train+val "
                f"(e.g. {sorted(tv_overlap)[:3]}) — atomic unit integrity violated"
            )
        if tt_overlap:
            errors.append(
                f"{tag} {gcol} fragmentation: {len(tt_overlap)} group(s) span train+test "
                f"(e.g. {sorted(tt_overlap)[:3]}) — atomic unit integrity violated"
            )
        if vt_overlap:
            errors.append(
                f"{tag} {gcol} fragmentation: {len(vt_overlap)} group(s) span val+test "
                f"(e.g. {sorted(vt_overlap)[:3]}) — atomic unit integrity violated"
            )

    # Path B only: chronological day ordering between partitions
    if split_path == "path_b" and "operating_day_id" in (
        train.columns if train is not None else []
    ):
        train_days = (
            set(train["operating_day_id"].dropna().unique()) if train is not None else set()
        )
        val_days = set(val["operating_day_id"].dropna().unique()) if val is not None else set()
        test_days = set(test["operating_day_id"].dropna().unique()) if test is not None else set()

        if train_days and val_days and max(train_days) >= min(val_days):
            errors.append(
                f"{tag} Path B day ordering violated: max train day ({max(train_days)}) "
                f">= min val day ({min(val_days)})"
            )
        if val_days and test_days and max(val_days) >= min(test_days):
            errors.append(
                f"{tag} Path B day ordering violated: max val day ({max(val_days)}) "
                f">= min test day ({min(test_days)})"
            )

    if errors:
        raise ValueError("\n".join(errors))

    logger.success(f"{tag} cross-split sanity passed")


def preprocess_split(
    input_dir: Path,
    output_dir: Path,
    split_name: str,
    dataset: str,
    feature_cols: list[str],
    preprocess_config: dict,
    label_col: str = "label",
    split_path: str = "path_a",
) -> dict:
    """
    Preprocess a single split (train/val/test).

    Args:
        input_dir: Directory containing train.parquet, val.parquet, test.parquet
        output_dir: Output directory for preprocessed files
        split_name: Name of the split for logging
        dataset: Dataset name
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
    processed_frames: dict[str, pd.DataFrame] = {}

    train_bounds = None
    processing_order = ["train", "val", "test"]

    for subset in processing_order:
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

        # Resolve label column for this subset without fabricating schema
        if label_col in df.columns:
            active_label_col = label_col
        elif "label" in df.columns:
            logger.warning(
                "  Requested label column '{}' missing, using fallback 'label'",
                label_col,
            )
            active_label_col = "label"
        else:
            logger.error(f"  No label column '{label_col}' or fallback 'label' found!")
            continue

        total_input_rows += len(df)

        # Run preprocessing
        df_processed, stats = preprocess(
            df=df,
            feature_cols=available_features,
            config=preprocess_config,
            timestamp_col="timestamp",
            segment_col="segment_id",
            label_col=active_label_col,
            outlier_reference_bounds=train_bounds if subset != "train" else None,
        )

        df_processed = _restore_costa_power_channels(df_processed, dataset)

        total_output_rows += len(df_processed)

        # Sanity checks before saving
        _sanity_check_subset(
            df=df_processed,
            subset=subset,
            split_name=split_name,
            feature_cols=available_features,
            label_col=active_label_col,
        )

        # Save
        output_path = output_dir / f"{subset}.parquet"
        df_processed.to_parquet(output_path, index=False)
        logger.info(f"    Saved: {output_path.name} ({len(df_processed):,} rows)")
        logger.success(f"    Sanity checks passed: {subset}")

        processed_frames[subset] = df_processed
        all_stats[subset] = stats

        # Store train-fit parameters for val/test reference (leakage-safe)
        if subset == "train":
            train_bounds = {
                col: tuple(bounds)
                for col, bounds in stats.get("outliers", {}).get("bounds", {}).items()
            }

    # Cross-split checks after all subsets processed
    _sanity_check_cross_split(processed_frames, split_name, split_path=split_path)

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
        "created_at": datetime.now(UTC).isoformat(),
        "config_used": (config.get("preprocessing") or {}),
        "split_path": config.get("split_path", "path_a"),
        "splits": split_stats,
        "features_created": [],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(f"Manifest saved: {output_path}")


def main() -> None:
    """Run preprocessing pipeline on all splits."""
    config = load_config()
    default_dataset = get_active_dataset(config)
    parser = argparse.ArgumentParser(description="Preprocess split data for a given dataset")
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help=f"Dataset to preprocess (must match data_config.yaml paths.datasets key). Default: {default_dataset}",
    )
    parser.add_argument(
        "--split-path",
        default="path_a",
        choices=["path_a", "path_b"],
        help="Which split path to preprocess. Default: path_a",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE — dataset={}", args.dataset)
    logger.info("=" * 60)

    preprocess_config = resolve_effective_preprocess_config(config, args.dataset, args.split_path)
    feature_cols = get_feature_cols(config, args.dataset)
    label_col = get_label_col(config, args.dataset)

    logger.info(f"Feature columns: {feature_cols}")
    logger.info(f"Label column: {label_col}")

    input_base, output_base = resolve_io_roots(PROJECT_ROOT, args.dataset, args.split_path)

    if not input_base.exists():
        logger.error(f"Input directory not found: {input_base}")
        logger.error("Run split_pipeline.py first!")
        return

    # Define splits to process
    splits = [
        "anomaly_semisup",
        "anomaly_supervised",
        "classification",
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
            dataset=args.dataset,
            feature_cols=feature_cols,
            preprocess_config=preprocess_config,
            label_col=label_col,
            split_path=args.split_path,
        )

        all_split_stats[split_name] = stats

        logger.success(
            f"{split_name}: {stats['total_input_rows']:,} → {stats['total_output_rows']:,} rows "
            f"({stats['rows_lost']:,} lost)"
        )

    # Save global manifest
    manifest_path = output_base / "preprocess_manifest.json"
    create_manifest(
        all_split_stats,
        {"preprocessing": preprocess_config, "split_path": args.split_path},
        manifest_path,
    )

    logger.info("=" * 60)
    logger.success("Preprocessing pipeline complete!")
    logger.info("=" * 60)

    # Summary
    total_in = sum(s["total_input_rows"] for s in all_split_stats.values())
    total_out = sum(s["total_output_rows"] for s in all_split_stats.values())
    logger.info(
        f"Total rows processed: {total_in:,} → {total_out:,} ({total_in - total_out:,} lost)"
    )


if __name__ == "__main__":
    main()
