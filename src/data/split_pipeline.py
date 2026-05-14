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
    blocked_semisup_split,
    blocked_temporal_split,
    filter_to_evaluable_classes,
    hybrid_semisup_split,
    segment_stratified_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"


def load_config() -> dict:
    """Load split configuration from data_config.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_dataset(config: dict) -> str:
    return str(config.get("active_dataset", "la_reunion"))


def attach_group_ids(df: pd.DataFrame, gap_seconds: int) -> pd.DataFrame:
    """Attach continuity-aware and episode-aware group identifiers."""
    df = df.copy()
    dt_s = df["timestamp"].diff().dt.total_seconds().fillna(0)
    label_change = df["label"].ne(df["label"].shift()).fillna(False)
    operating_day = df["timestamp"].dt.strftime("%Y-%m-%d")

    df["continuity_segment_id"] = (dt_s > gap_seconds).cumsum().astype(int)
    df["episode_id"] = ((dt_s > gap_seconds) | label_change).cumsum().astype(int)
    df["operating_day"] = operating_day
    df["operating_day_id"] = pd.factorize(operating_day, sort=True)[0].astype(int)
    return df


def resolve_atomic_group_col(config: dict, dataset: str, df: pd.DataFrame) -> str:
    """Resolve the canonical grouped split unit for the dataset."""
    dataset_cfg = config.get("paths", {}).get("datasets", {}).get(dataset, {})
    split_cfg = dataset_cfg.get("splits", {})
    requested = str(split_cfg.get("atomic_group_col", "segment_id"))
    if requested not in df.columns:
        logger.warning(
            "Requested atomic_group_col '{}' missing for dataset={}, falling back to continuity_segment_id",
            requested,
            dataset,
        )
        return "continuity_segment_id"
    return requested


def load_and_segment_data(config: dict, dataset: str = "reunion") -> pd.DataFrame:
    """
    Load merged data for the specified dataset and apply segmentation.

    Reads the dataset-specific merged parquet from data/interim/ and applies
    contiguous segmentation (gap > segmentation_gap_seconds → new segment).

    Returns DataFrame with standardised columns: timestamp | label | segment_id | <sensors>
    """
    paths = config["paths"]
    split_cfg = effective_split_cfg(config, dataset)

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

    # Attach continuity-aware and episode-aware group IDs
    segmentation_gap_seconds = int(split_cfg.get("segmentation_gap_seconds", 300))
    df = attach_group_ids(df, segmentation_gap_seconds)

    atomic_group_col = resolve_atomic_group_col(config, dataset, df)
    df["segment_id"] = df[atomic_group_col].astype(int)

    n_continuity = df["continuity_segment_id"].nunique()
    n_episodes = df["episode_id"].nunique()
    n_atomic = df["segment_id"].nunique()
    logger.info(
        "Created grouped units | continuity_segments={} | episodes={} | atomic_group_col={} ({})",
        n_continuity,
        n_episodes,
        atomic_group_col,
        n_atomic,
    )

    return df


def save_split(
    artifacts, output_dir: Path, split_name: str, extra_manifest: dict | None = None
) -> None:
    """Save split artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts.train.to_parquet(output_dir / "train.parquet", index=False)
    artifacts.val.to_parquet(output_dir / "val.parquet", index=False)
    artifacts.test.to_parquet(output_dir / "test.parquet", index=False)

    manifest = dict(artifacts.manifest)
    if extra_manifest:
        manifest.update(extra_manifest)

    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.success(
        f"{split_name}: train={len(artifacts.train):,}, "
        f"val={len(artifacts.val):,}, test={len(artifacts.test):,}"
    )


def dataset_split_cfg(config: dict, dataset: str) -> dict:
    return config.get("paths", {}).get("datasets", {}).get(dataset, {}).get("splits", {})


def _merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def effective_split_cfg(config: dict, dataset: str) -> dict:
    """Resolve split settings with dataset overrides on top of global defaults."""
    global_cfg = dict(config.get("splits", {}))
    ds_cfg = dict(dataset_split_cfg(config, dataset))
    ds_cfg.pop("comparison_paths", None)
    return _merge_dict(global_cfg, ds_cfg)


def available_group_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in ("segment_id", "episode_id", "continuity_segment_id", "operating_day_id")
        if c in df.columns
    ]


def build_multilabel_support_report(
    full_df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    group_col: str,
    label_col: str = "label",
) -> dict:
    """Report row support plus grouped-unit presence for mixed-label day units."""
    unit_labels = full_df.groupby(group_col, observed=True)[label_col].apply(
        lambda x: set(pd.Series(x).dropna().tolist())
    )
    report: dict[str, dict] = {}
    for label in sorted(full_df[label_col].dropna().unique().tolist()):
        total_rows = int((full_df[label_col] == label).sum())
        total_units = int(sum(label in labels for labels in unit_labels))
        per_split = {}
        for split_name, frame in split_frames.items():
            split_unit_labels = frame.groupby(group_col, observed=True)[label_col].apply(
                lambda x: set(pd.Series(x).dropna().tolist())
            )
            split_rows = int((frame[label_col] == label).sum())
            split_units = int(sum(label in labels for labels in split_unit_labels))
            per_split[split_name] = {
                "rows": split_rows,
                "row_ratio_within_class": split_rows / total_rows if total_rows else None,
                "units": split_units,
                "unit_ratio_within_class": split_units / total_units if total_units else None,
            }
        report[str(int(label) if float(label).is_integer() else label)] = {
            "total_rows": total_rows,
            "total_units": total_units,
            "splits": per_split,
        }
    return report


def run_anomaly_semisup_split(
    df: pd.DataFrame, config: dict, output_base: Path, dataset: str
) -> None:
    """Generate hybrid split for semi-supervised anomaly detection."""
    logger.info("=" * 50)
    logger.info("[1/3] Anomaly Detection — Semi-Supervised (Temporal Hybrid)")
    logger.info("Train: Normal-only (temporal) | Val/Test: Normal + faults (temporal)")

    split_cfg = effective_split_cfg(config, dataset)

    artifacts = hybrid_semisup_split(
        df=df,
        segment_col="segment_id",
        label_col="label",
        time_col="timestamp",
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        embargo_seconds=split_cfg.get("embargo_seconds", 1260),
    )

    atomic_group_col = (
        config["paths"]["datasets"]
        .get(dataset, {})
        .get("splits", {})
        .get("atomic_group_col", "segment_id")
    )
    save_split(
        artifacts,
        output_base / "anomaly_semisup",
        "anomaly_semisup",
        extra_manifest={
            "split_path": "path_a",
            "split_protocol": "episode_based",
            "group_column": atomic_group_col,
            "available_group_columns": available_group_columns(df),
        },
    )

    # Verify train is normal-only
    train_labels = artifacts.train["label"].unique()
    assert all(lbl == 0.0 for lbl in train_labels), "Train should be normal-only!"
    logger.info(f"Verified: Train contains only normal data ({len(artifacts.train):,} rows)")


def run_anomaly_supervised_split(
    df: pd.DataFrame, config: dict, output_base: Path, dataset: str
) -> None:
    """Generate temporal-stratified split for supervised anomaly detection."""
    logger.info("=" * 50)
    logger.info("[2/3] Anomaly Detection — Supervised (Temporal-Stratified)")
    logger.info("All sets: Temporal order preserved within each class")

    split_cfg = effective_split_cfg(config, dataset)

    artifacts = segment_stratified_split(
        df=df,
        segment_col="segment_id",
        label_col="label",
        time_col="timestamp",
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        embargo_seconds=split_cfg.get("embargo_seconds", 1260),
    )

    atomic_group_col = (
        config["paths"]["datasets"]
        .get(dataset, {})
        .get("splits", {})
        .get("atomic_group_col", "segment_id")
    )
    save_split(
        artifacts,
        output_base / "anomaly_supervised",
        "anomaly_supervised",
        extra_manifest={
            "split_path": "path_a",
            "split_protocol": "episode_based",
            "group_column": atomic_group_col,
            "available_group_columns": available_group_columns(df),
        },
    )


def run_classification_split(
    df: pd.DataFrame, config: dict, output_base: Path, dataset: str
) -> None:
    """Generate segment-stratified split for fault classification (evaluable classes only)."""
    logger.info("=" * 50)
    logger.info("[3/3] Fault Classification (Temporal-Stratified, Evaluable Classes)")

    split_cfg = effective_split_cfg(config, dataset)
    dataset_split_cfg = (
        config.get("paths", {}).get("datasets", {}).get(dataset, {}).get("splits", {})
    )
    evaluable_classes = dataset_split_cfg.get("evaluable_classes", [])
    logger.info("Classes: {}", evaluable_classes)

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
    # support_report keys are str(float) because _build_split_support_report uses str(label)
    # and labels are floats from the source data. Normalise lookups accordingly.
    support_report = artifacts.manifest.get("support_report", {})
    filtered_support_report = {}
    for cls in evaluable_classes:
        key = str(float(cls))
        entry = support_report.get(key, {})
        if not entry:
            logger.warning("support_report missing key '{}' for class {}", key, cls)
        filtered_support_report[str(cls)] = entry

    class_temporal_info = artifacts.manifest.get("class_temporal_info", {})
    filtered_temporal_info = {}
    for cls in evaluable_classes:
        key = str(float(cls))
        entry = class_temporal_info.get(key, class_temporal_info.get(str(cls), {}))
        filtered_temporal_info[str(cls)] = entry

    manifest = {
        "split_type": "temporal_stratified_classification",
        "split_path": "path_a",
        "split_protocol": "episode_based",
        "group_column": dataset_split_cfg.get("atomic_group_col", "segment_id"),
        "available_group_columns": available_group_columns(df),
        "evaluable_classes": evaluable_classes,
        "train_rows": len(train_filtered),
        "val_rows": len(val_filtered),
        "test_rows": len(test_filtered),
        "train_class_counts": train_filtered["label"].value_counts().sort_index().to_dict(),
        "val_class_counts": val_filtered["label"].value_counts().sort_index().to_dict(),
        "test_class_counts": test_filtered["label"].value_counts().sort_index().to_dict(),
        "class_temporal_info": filtered_temporal_info,
        "support_report": filtered_support_report,
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


def run_path_b_splits(df: pd.DataFrame, config: dict, output_base: Path, dataset: str) -> None:
    """Generate Costa Path B day-based chronological splits with day-level purge."""
    split_cfg = effective_split_cfg(config, dataset)
    ds_split_cfg = dataset_split_cfg(config, dataset)
    path_b_cfg = ds_split_cfg.get("comparison_paths", {}).get("path_b", {})
    if dataset != "costa" or not path_b_cfg.get("enabled", False):
        return

    logger.info("=" * 60)
    logger.info("Generating Costa Path B comparison splits")
    logger.info("Day-based chronological blocks + forward purge")

    path_b_base = output_base / "path_b"
    group_col = str(path_b_cfg.get("atomic_group_col", "operating_day_id"))
    purge_days = int(path_b_cfg.get("purge_days", 1))
    boundary_purge_fraction = float(path_b_cfg.get("forward_boundary_purge_fraction", 0.0))

    semisup_artifacts = blocked_semisup_split(
        df=df,
        segment_col=group_col,
        label_col="label",
        time_col="timestamp",
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        purge_units=purge_days,
        boundary_purge_fraction=boundary_purge_fraction,
        unit_name="operating_days",
    )
    save_split(
        semisup_artifacts,
        path_b_base / "anomaly_semisup",
        "path_b/anomaly_semisup",
        extra_manifest={
            "split_path": "path_b",
            "split_protocol": "day_based_purged",
            "group_column": group_col,
            "available_group_columns": available_group_columns(df),
            "forward_boundary_purge_fraction": boundary_purge_fraction,
        },
    )

    supervised_artifacts = blocked_temporal_split(
        df=df,
        segment_col=group_col,
        label_col="label",
        time_col="timestamp",
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        purge_units=purge_days,
        boundary_purge_fraction=boundary_purge_fraction,
        unit_name="operating_days",
    )
    save_split(
        supervised_artifacts,
        path_b_base / "anomaly_supervised",
        "path_b/anomaly_supervised",
        extra_manifest={
            "split_path": "path_b",
            "split_protocol": "day_based_purged",
            "group_column": group_col,
            "available_group_columns": available_group_columns(df),
            "forward_boundary_purge_fraction": boundary_purge_fraction,
        },
    )

    evaluable_classes = ds_split_cfg.get("evaluable_classes", [])
    train_filtered = filter_to_evaluable_classes(supervised_artifacts.train, evaluable_classes)
    val_filtered = filter_to_evaluable_classes(supervised_artifacts.val, evaluable_classes)
    test_filtered = filter_to_evaluable_classes(supervised_artifacts.test, evaluable_classes)
    cls_output_dir = path_b_base / "classification"
    cls_output_dir.mkdir(parents=True, exist_ok=True)
    train_filtered.to_parquet(cls_output_dir / "train.parquet", index=False)
    val_filtered.to_parquet(cls_output_dir / "val.parquet", index=False)
    test_filtered.to_parquet(cls_output_dir / "test.parquet", index=False)

    filtered_support_report = build_multilabel_support_report(
        full_df=filter_to_evaluable_classes(df, evaluable_classes),
        split_frames={"train": train_filtered, "val": val_filtered, "test": test_filtered},
        group_col=group_col,
        label_col="label",
    )
    manifest = {
        "split_type": "blocked_temporal_purged_classification",
        "split_path": "path_b",
        "split_protocol": "day_based_purged",
        "group_column": group_col,
        "available_group_columns": available_group_columns(df),
        "purge_days": purge_days,
        "forward_boundary_purge_fraction": boundary_purge_fraction,
        "evaluable_classes": evaluable_classes,
        "train_rows": len(train_filtered),
        "val_rows": len(val_filtered),
        "test_rows": len(test_filtered),
        "train_class_counts": train_filtered["label"].value_counts().sort_index().to_dict(),
        "val_class_counts": val_filtered["label"].value_counts().sort_index().to_dict(),
        "test_class_counts": test_filtered["label"].value_counts().sort_index().to_dict(),
        "support_report": filtered_support_report,
        "global_temporal_diagnostics": supervised_artifacts.manifest.get(
            "global_temporal_diagnostics", {}
        ),
        "note": (
            "Costa Path B comparison split. Chronological day blocks with one-day forward purge; "
            "fault-only classification derived after day-level assignment."
        ),
    }
    with open(cls_output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.success(
        "path_b/classification: train={}, val={}, test={}",
        len(train_filtered),
        len(val_filtered),
        len(test_filtered),
    )


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
    run_anomaly_semisup_split(df, config, output_base, args.dataset)
    run_anomaly_supervised_split(df, config, output_base, args.dataset)

    # Task B split (FDD — Fault Diagnosis)
    run_classification_split(df, config, output_base, args.dataset)
    run_path_b_splits(df, config, output_base, args.dataset)

    logger.info("=" * 60)
    logger.success("All splits generated | dataset={}", args.dataset)
    logger.info("=" * 60)

    logger.info("Generated splits:")
    logger.info("  • anomaly_semisup/    — Path A Task A semi-supervised (episode-based)")
    logger.info("  • anomaly_supervised/ — Path A Task A supervised (episode-based)")
    logger.info("  • classification/     — Path A Task B (episode-based, evaluable classes only)")
    if args.dataset == "costa":
        logger.info("  • path_b/*            — Path B Costa comparison (day-based + purge)")


if __name__ == "__main__":
    main()
