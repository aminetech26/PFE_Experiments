"""
EDA Pipeline — Per-dataset exploratory analysis and findings export.

Runs statistical analysis on the ingested parquet for a given dataset and writes
structured findings to data/interim/eda/<dataset>/. These findings drive all
downstream feature selection, preprocessing column choices, and split validation.

Outputs
-------
  eda_feature_findings.json     — consolidated (read by featurize_pipeline)
  eda_mannwhitney_findings.json — Mann-Whitney U per feature
  eda_spearman_findings.json    — Spearman correlation pairs
  eda_vif_findings.json         — VIF collinearity
  eda_mutual_info_findings.json — mutual information scores
  eda_dataset_report.json       — gap analysis, class distribution, segment stats,
                                   dataset-specific diagnostics, split recommendations

Usage
-----
    uv run python -m src.data.eda_pipeline                      # default: la_reunion
    uv run python -m src.data.eda_pipeline --dataset costa
    uv run python -m src.data.eda_pipeline --dataset mendeley
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import polars as pl
import yaml
from loguru import logger

from src.data.eda_findings import export_eda_feature_findings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data_config.yaml"


def _build_artifact_manifest(
    dataset: str, output_dir: Path, report_path: Path, file_paths: dict[str, str]
) -> dict:
    return {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "output_dir": str(output_dir),
        "artifacts": {
            "dataset_report": str(report_path),
            **file_paths,
        },
    }


def _to_json_safe(value):
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# =============================================================================
# CONFIG HELPERS
# =============================================================================


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_dataset(config: dict) -> str:
    return str(config.get("active_dataset", "la_reunion"))


def get_dataset_cfg(config: dict, dataset: str) -> dict:
    ds_cfg = config["paths"]["datasets"].get(dataset)
    if ds_cfg is None:
        available = list(config["paths"]["datasets"].keys())
        raise KeyError(f"Dataset '{dataset}' not in data_config.yaml. Available: {available}")
    return ds_cfg


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================


def load_dataset(config: dict, dataset: str) -> pd.DataFrame:
    """Load the ingested parquet and standardise label + timestamp columns."""
    ds_cfg = get_dataset_cfg(config, dataset)
    interim_dir = PROJECT_ROOT / config["paths"]["interim_dir"]
    parquet_path = interim_dir / ds_cfg["merged_output"]

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Ingested parquet not found: {parquet_path}\n"
            f"Run: uv run python -m src.data.ingestion --dataset {dataset}"
        )

    logger.info("Loading {} from {}", dataset, parquet_path)
    df = pl.read_parquet(parquet_path).to_pandas()
    logger.info("  Loaded {:,} rows × {} cols", len(df), len(df.columns))

    # Standardise timestamp
    ts_col = ds_cfg.get("timestamp_col", "timestamp")
    if ts_col in df.columns and ts_col != "timestamp":
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    elif ts_col == "timestamp":
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Standardise label → always "label" as float for uniform downstream handling
    label_col_raw = ds_cfg.get("label_col", "label")
    if label_col_raw in df.columns and label_col_raw != "label":
        df["label"] = df[label_col_raw].astype(float)
    elif "label" in df.columns:
        df["label"] = df["label"].astype(float)

    df = df.dropna(subset=["timestamp", "label"]).sort_values("timestamp").reset_index(drop=True)
    return df


def prepare_eda_frame(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, str]:
    """
    Apply dataset-specific filtering to get a clean frame for statistical tests.

    Returns (eda_df, description_string).

    Costa
      Current ingestion already trims Costa to meaningful operating irradiance
      (`irr >= 100 W/m²`). No extra day/night filtering is required in EDA.

    Mendeley
      Each experiment has a pre-fault half (label=0) and a fault half (label=fault_class).
      For meaningful normal-vs-fault stats we keep:
        normal: fault_class==0 (F0 experiments only — genuinely fault-free)
        fault:  phase=="fault"  (active fault half of F1-F7 experiments only)
      Pre-fault rows of F1-F7 are dropped to avoid contaminating the normal class
      with data that was recorded while the system was about to fault.

    La Réunion
      No special filtering needed. All rows with valid label are used.
    """
    desc_parts = []

    if dataset == "costa":
        desc_parts.append("irradiance_trimmed_at_ingestion=True")

    elif dataset == "mendeley":
        if "phase" in df.columns and "fault_class" in df.columns:
            n_before = len(df)
            is_normal = pd.Series(df["fault_class"] == 0, index=df.index)  # F0 experiments
            is_fault = pd.Series(
                df["phase"] == "fault", index=df.index
            )  # active fault half of F1-F7
            filtered_df = df.loc[is_normal | is_fault].reset_index(drop=True)
            df = filtered_df
            logger.info(
                "Mendeley phase filter: {:,} → {:,} rows (F0 normal + fault-phase only)",
                n_before,
                len(filtered_df),
            )
            desc_parts.append("phase_filter=F0_normal+fault_active")
        else:
            desc_parts.append("phase_filter=none (phase column missing)")

    desc = f"dataset={dataset}" + (f", {', '.join(desc_parts)}" if desc_parts else "")
    logger.info("EDA frame: {:,} rows — {}", len(df), desc)
    return df, desc


# =============================================================================
# GAP ANALYSIS
# =============================================================================


def analyse_gaps(df: pd.DataFrame, configured_gap_s: int) -> dict:
    """
    Compute time-gap distribution between consecutive rows and evaluate whether
    the configured segmentation_gap_seconds threshold is appropriate.

    Also computes label-transition segmentation (segment boundary whenever label
    changes OR gap > threshold) — important for datasets with continuous synthetic
    timestamps (Costa).
    """
    dt_s = df["timestamp"].diff().dt.total_seconds().fillna(0).values

    # Gap distribution
    non_zero = dt_s[dt_s > 0]
    if len(non_zero) == 0:
        return {"error": "no non-zero gaps found"}

    percentiles = {
        "p50": float(np.percentile(non_zero, 50)),
        "p90": float(np.percentile(non_zero, 90)),
        "p95": float(np.percentile(non_zero, 95)),
        "p99": float(np.percentile(non_zero, 99)),
        "max": float(non_zero.max()),
        "mean": float(non_zero.mean()),
    }
    nominal_interval_s = float(np.percentile(non_zero, 50))  # typical step

    # Gaps above various thresholds
    thresholds = [30, 60, 120, 300, 600, 1800, 3600]
    gaps_above = {f">{t}s": int((dt_s > t).sum()) for t in thresholds}

    # Gap-based segment count (current strategy)
    gap_seg_ids = (dt_s > configured_gap_s).cumsum()
    n_gap_segments = int(gap_seg_ids.max()) + 1

    # Label-transition + gap segmentation
    label_change = df["label"].ne(df["label"].shift()).fillna(False).values
    combined_mask = (dt_s > configured_gap_s) | label_change
    trans_seg_ids = combined_mask.cumsum()
    n_trans_segments = int(trans_seg_ids.max()) + 1

    # Warn if gap-based gives very few segments (synthetic continuous timestamps)
    gap_appropriate = n_gap_segments >= 3
    recommendation = "gap_based" if n_gap_segments >= 3 else "label_transition"

    return {
        "configured_gap_seconds": int(configured_gap_s),
        "nominal_interval_seconds": round(nominal_interval_s, 4),
        "gap_percentiles": {k: round(v, 4) for k, v in percentiles.items()},
        "gaps_above_threshold": gaps_above,
        "n_segments_gap_based": int(n_gap_segments),
        "n_segments_label_transition": int(n_trans_segments),
        "gap_based_appropriate": bool(gap_appropriate),
        "recommended_segmentation_strategy": recommendation,
        "note": (
            "gap_based produces ≥3 segments — threshold is appropriate."
            if gap_appropriate
            else "WARNING: gap_based yields <3 segments (likely synthetic/continuous timestamps). "
            "Use label_transition strategy instead."
        ),
    }


# =============================================================================
# CLASS AND SEGMENT DISTRIBUTION
# =============================================================================


def analyse_class_distribution(
    df: pd.DataFrame,
    gap_seconds: int,
    min_segments_for_eval: int = 3,
) -> dict:
    """
    Count rows and segments per label class.
    Segments computed using label-transition + gap strategy (most general).
    Recommends evaluable_classes and train_only_classes based on segment counts.
    """
    dt_s = df["timestamp"].diff().dt.total_seconds().fillna(0).values
    label_change = df["label"].ne(df["label"].shift()).fillna(False).values
    seg_ids = ((dt_s > gap_seconds) | label_change).cumsum()
    df = df.copy()
    df["_seg_id"] = seg_ids

    total = len(df)
    class_stats = []
    for lbl in sorted(df["label"].unique()):
        mask = df["label"] == lbl
        rows = int(mask.sum())
        segs = int(df.loc[mask, "_seg_id"].nunique())
        class_stats.append(
            {
                "label": float(lbl),
                "n_rows": rows,
                "pct_rows": round(100 * rows / total, 2),
                "n_segments": segs,
                "evaluable": segs >= min_segments_for_eval and lbl != 0.0,
            }
        )

    evaluable = [r["label"] for r in class_stats if r["evaluable"]]
    train_only = [r["label"] for r in class_stats if not r["evaluable"] and r["label"] != 0.0]
    n_total_seg = int(df["_seg_id"].nunique())

    return {
        "total_rows": total,
        "total_segments": n_total_seg,
        "min_segments_for_eval": min_segments_for_eval,
        "segmentation_strategy": "label_transition_plus_gap",
        "per_class": class_stats,
        "recommended_evaluable_classes": evaluable,
        "recommended_train_only_classes": train_only,
    }


# =============================================================================
# DATASET-SPECIFIC DIAGNOSTICS
# =============================================================================


def diagnose_costa(df: pd.DataFrame) -> dict:
    """
    Costa-specific diagnostics:
    - Validate that ingest-time irradiance trimming is in effect
    - Summarize the retained operating regime
    """
    return {
        "ingestion_trimmed": True,
        "total_rows": int(len(df)),
        "irr_min": float(df["irr"].min()) if "irr" in df.columns and len(df) else None,
        "irr_max": float(df["irr"].max()) if "irr" in df.columns and len(df) else None,
        "recommendation": (
            "Costa ingestion already removed low-irradiance rows; no additional "
            "day/night filtering is needed for EDA or downstream modeling."
        ),
    }


def diagnose_mendeley(df: pd.DataFrame) -> dict:
    """
    Mendeley-specific diagnostics:
    - Phase distribution per fault class
    - Pre-fault vs fault class separation quality
    """
    if "phase" not in df.columns or "fault_class" not in df.columns:
        return {"error": "phase or fault_class column not found"}

    total = len(df)
    phase_counts = df["phase"].value_counts().to_dict()

    # Per fault_class: how many experiments (mode × fault_class)
    exp_counts = []
    if "mode" in df.columns:
        for keys, grp in df.groupby(["fault_class", "mode"]):
            fc, mode = cast(tuple[float | int, object], keys)
            exp_counts.append(
                {
                    "fault_class": int(fc),
                    "mode": str(mode),
                    "rows": len(grp),
                    "phases": grp["phase"].value_counts().to_dict(),
                }
            )

    return {
        "total_rows": total,
        "phase_distribution": {k: int(v) for k, v in phase_counts.items()},
        "experiments": exp_counts,
        "usable_for_eda": {
            "normal_rows": int((df["fault_class"] == 0).sum()),
            "fault_rows": int((df["phase"] == "fault").sum()),
            "prefault_rows_excluded": int((df["phase"] == "pre_fault").sum()),
        },
        "recommendation": (
            "Use fault_class==0 for normal class and phase=='fault' for fault class. "
            "Pre-fault rows (~50% of each F1-F7 experiment) look normal but belong to "
            "fault experiments — exclude them from both normal and fault class to avoid "
            "contamination. These rows can be used as additional normal training data "
            "only if explicitly needed."
        ),
    }


# =============================================================================
# SPLIT RECOMMENDATIONS
# =============================================================================


def build_split_recommendations(
    gap_analysis: dict,
    class_dist: dict,
    config: dict,
    dataset: str,
) -> dict:
    """Generate recommended data_config.yaml overrides for the splits section."""
    ds_splits = config["paths"]["datasets"][dataset].get("splits", {})
    global_splits = config.get("splits", {})

    current_strategy = ds_splits.get(
        "segmentation_strategy",
        gap_analysis.get("recommended_segmentation_strategy", "gap_based"),
    )
    current_gap = ds_splits.get(
        "segmentation_gap_seconds",
        global_splits.get("segmentation_gap_seconds", 300),
    )
    recommended_strategy = gap_analysis.get("recommended_segmentation_strategy", "gap_based")

    changes_needed = []
    if recommended_strategy != current_strategy:
        changes_needed.append(
            f"segmentation_strategy: '{current_strategy}' → '{recommended_strategy}'"
        )

    current_eval = ds_splits.get("evaluable_classes", [])
    rec_eval = class_dist.get("recommended_evaluable_classes", [])
    if sorted(current_eval) != sorted(rec_eval):
        changes_needed.append(f"evaluable_classes: {current_eval} → {rec_eval}")

    return {
        "current_segmentation_strategy": current_strategy,
        "recommended_segmentation_strategy": recommended_strategy,
        "current_segmentation_gap_seconds": current_gap,
        "current_evaluable_classes": current_eval,
        "recommended_evaluable_classes": rec_eval,
        "recommended_train_only_classes": class_dist.get("recommended_train_only_classes", []),
        "config_changes_needed": changes_needed,
        "action_required": len(changes_needed) > 0,
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    config = load_config()
    default_dataset = get_active_dataset(config)
    parser = argparse.ArgumentParser(description="Run per-dataset EDA and export findings.")
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help=f"Dataset to analyse (must match data_config.yaml paths.datasets key). Default: {default_dataset}",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EDA PIPELINE — dataset={}", args.dataset)
    logger.info("=" * 60)

    ds_cfg = get_dataset_cfg(config, args.dataset)
    ds_splits = ds_cfg.get("splits", {})
    gap_s = ds_splits.get(
        "segmentation_gap_seconds",
        config.get("splits", {}).get("segmentation_gap_seconds", 300),
    )
    ds_fe = ds_cfg.get("feature_engineering", {})
    sensor_cols = ds_fe.get("sensor_columns", [])
    global_sel = config.get("feature_engineering", {}).get("selection", {})
    corr_thr = float(global_sel.get("corr_threshold", 0.95))
    vif_thr = float(global_sel.get("vif_threshold", 10.0))

    # Output directory
    output_dir = PROJECT_ROOT / "data" / "interim" / "eda" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}", output_dir)

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    df_raw = load_dataset(config, args.dataset)

    # -------------------------------------------------------------------------
    # 2. Gap analysis (on raw continuous data before any filtering)
    # -------------------------------------------------------------------------
    logger.info("[1/5] Gap analysis ...")
    gap_analysis = analyse_gaps(df_raw, gap_s)
    logger.info(
        "  nominal interval: {}s | gap-based segments: {} | label-transition segments: {}",
        gap_analysis["nominal_interval_seconds"],
        gap_analysis["n_segments_gap_based"],
        gap_analysis["n_segments_label_transition"],
    )
    if not gap_analysis["gap_based_appropriate"]:
        logger.warning(
            "  {} → recommend switching to label_transition strategy", gap_analysis["note"]
        )

    # -------------------------------------------------------------------------
    # 3. Class / segment distribution
    # -------------------------------------------------------------------------
    logger.info("[2/5] Class distribution ...")
    class_dist = analyse_class_distribution(df_raw, gap_s)
    for row in class_dist["per_class"]:
        logger.info(
            "  label={} | {:>8,} rows ({:5.1f}%) | {:3d} segments | evaluable={}",
            row["label"],
            row["n_rows"],
            row["pct_rows"],
            row["n_segments"],
            row["evaluable"],
        )

    # -------------------------------------------------------------------------
    # 4. Dataset-specific diagnostics
    # -------------------------------------------------------------------------
    logger.info("[3/5] Dataset-specific diagnostics ...")
    dataset_specific: dict = {}
    if args.dataset == "costa":
        dataset_specific = diagnose_costa(df_raw)
        logger.info(
            "  Costa ingestion-trimmed: {} | irr_min={}",
            dataset_specific.get("ingestion_trimmed"),
            dataset_specific.get("irr_min"),
        )
    elif args.dataset == "mendeley":
        dataset_specific = diagnose_mendeley(df_raw)
        logger.info("  Mendeley usable rows: {}", dataset_specific.get("usable_for_eda"))

    # -------------------------------------------------------------------------
    # 5. Statistical tests — on filtered/prepared EDA frame
    # -------------------------------------------------------------------------
    logger.info("[4/5] Statistical tests (Mann-Whitney, Spearman, VIF, MI) ...")
    df_eda, eda_desc = prepare_eda_frame(df_raw, args.dataset)

    # Only keep sensor columns that actually exist in this dataset's parquet
    available_sensors = [c for c in sensor_cols if c in df_eda.columns]
    missing = [c for c in sensor_cols if c not in df_eda.columns]
    if missing:
        logger.warning("  Sensor columns not found in parquet (skipped): {}", missing)
    if not available_sensors:
        raise ValueError(
            f"None of the configured sensor_columns {sensor_cols} found in the parquet. "
            f"Check paths.datasets.{args.dataset}.feature_engineering.sensor_columns in data_config.yaml."
        )
    logger.info("  Running statistics on {} sensor columns", len(available_sensors))

    consolidated, file_paths = export_eda_feature_findings(
        pdf_complete=df_eda,
        feature_cols=available_sensors,
        label_col="label",
        dataset=args.dataset,
        output_dir=output_dir,
        corr_threshold=corr_thr,
        vif_threshold=vif_thr,
    )
    consolidated["eda_frame_description"] = eda_desc

    # -------------------------------------------------------------------------
    # 6. Split recommendations
    # -------------------------------------------------------------------------
    logger.info("[5/5] Building split recommendations ...")
    split_recs = build_split_recommendations(gap_analysis, class_dist, config, args.dataset)
    if split_recs["action_required"]:
        logger.warning("  Config changes recommended:")
        for change in split_recs["config_changes_needed"]:
            logger.warning("    → {}", change)
    else:
        logger.info("  Current config is consistent with observed data.")

    # -------------------------------------------------------------------------
    # 7. Write dataset report
    # -------------------------------------------------------------------------
    report = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "eda_frame_description": eda_desc,
        "gap_analysis": gap_analysis,
        "class_distribution": class_dist,
        "dataset_specific": dataset_specific,
        "split_recommendations": split_recs,
        "sensor_columns_used": available_sensors,
        "sensor_columns_missing": missing,
    }

    report_path = output_dir / "eda_dataset_report.json"
    report_path.write_text(
        json.dumps(_to_json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.success("Dataset report → {}", report_path)

    artifact_manifest = _build_artifact_manifest(args.dataset, output_dir, report_path, file_paths)
    artifact_manifest_path = output_dir / "eda_artifact_manifest.json"
    artifact_manifest_path.write_text(
        json.dumps(_to_json_safe(artifact_manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.success("Artifact manifest → {}", artifact_manifest_path)

    # Re-write consolidated findings with patched dataset field
    consolidated_path = output_dir / "eda_feature_findings.json"
    consolidated_path.write_text(
        json.dumps(_to_json_safe(consolidated), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    logger.success("Consolidated findings → {}", consolidated_path)

    # Summary
    logger.info("=" * 60)
    logger.info("EDA complete | dataset={}", args.dataset)
    logger.info("  Segments (gap-based):        {:>4}", gap_analysis["n_segments_gap_based"])
    logger.info("  Segments (label-transition): {:>4}", gap_analysis["n_segments_label_transition"])
    logger.info("  Evaluable fault classes:     {}", split_recs["recommended_evaluable_classes"])
    logger.info(
        "  Top binary MI features:      {}",
        consolidated.get("mutual_information", {}).get("top_features_binary", [])[:5],
    )
    if split_recs["action_required"]:
        logger.warning("  ACTION REQUIRED — update data_config.yaml:")
        for c in split_recs["config_changes_needed"]:
            logger.warning("    {}", c)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
