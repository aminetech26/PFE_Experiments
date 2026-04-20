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
    uv run python -m src.data.eda_pipeline --dataset costa --daytime-only
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
from loguru import logger

from src.data.eda_findings import export_eda_feature_findings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "configs" / "data_config.yaml"


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def prepare_eda_frame(df: pd.DataFrame, dataset: str, daytime_only: bool) -> tuple[pd.DataFrame, str]:
    """
    Apply dataset-specific filtering to get a clean frame for statistical tests.

    Returns (eda_df, description_string).

    Costa
      Faults 1-3 are daytime-only artificially induced blocks. Night rows are all
      label=0 (normal) and would trivially inflate normal-vs-fault separation via
      the irradiance channel. daytime_only=True filters to is_daytime=True first.

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
        if daytime_only and "is_daytime" in df.columns:
            n_before = len(df)
            df = df[df["is_daytime"]].reset_index(drop=True)
            logger.info("Costa daytime filter: {:,} → {:,} rows", n_before, len(df))
            desc_parts.append("daytime_only=True")
        else:
            desc_parts.append("daytime_only=False (all rows including night)")

    elif dataset == "mendeley":
        if "phase" in df.columns and "fault_class" in df.columns:
            n_before = len(df)
            is_normal  = df["fault_class"] == 0           # F0 experiments
            is_fault   = df["phase"] == "fault"            # active fault half of F1-F7
            df = df[is_normal | is_fault].reset_index(drop=True)
            logger.info("Mendeley phase filter: {:,} → {:,} rows (F0 normal + fault-phase only)", n_before, len(df))
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
    gap_seg_ids   = (dt_s > configured_gap_s).cumsum()
    n_gap_segments = int(gap_seg_ids.max()) + 1

    # Label-transition + gap segmentation
    label_change  = df["label"].ne(df["label"].shift()).fillna(False).values
    combined_mask = (dt_s > configured_gap_s) | label_change
    trans_seg_ids = combined_mask.cumsum()
    n_trans_segments = int(trans_seg_ids.max()) + 1

    # Warn if gap-based gives very few segments (synthetic continuous timestamps)
    gap_appropriate = n_gap_segments >= 3
    recommendation = (
        "gap_based"
        if n_gap_segments >= 3
        else "label_transition"
    )

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
            else
            "WARNING: gap_based yields <3 segments (likely synthetic/continuous timestamps). "
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
        class_stats.append({
            "label": float(lbl),
            "n_rows": rows,
            "pct_rows": round(100 * rows / total, 2),
            "n_segments": segs,
            "evaluable": segs >= min_segments_for_eval and lbl != 0.0,
        })

    evaluable   = [r["label"] for r in class_stats if r["evaluable"]]
    train_only  = [r["label"] for r in class_stats if not r["evaluable"] and r["label"] != 0.0]
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
    - Daytime vs nighttime label distribution
    - Night data = trivially all-normal → quantify how much this inflates separation
    """
    if "is_daytime" not in df.columns:
        return {"error": "is_daytime column not found"}

    total = len(df)
    day   = df["is_daytime"]
    night = ~df["is_daytime"]

    def label_dist(mask: pd.Series) -> dict:
        sub = df[mask]
        counts = sub["label"].value_counts().sort_index()
        return {str(int(k)): int(v) for k, v in counts.items()}

    night_fault_rows  = int((df[night]["label"] != 0).sum())
    day_normal_rows   = int((df[day]["label"] == 0).sum())
    day_fault_rows    = int((df[day]["label"] != 0).sum())

    return {
        "total_rows": total,
        "daytime_rows": int(day.sum()),
        "nighttime_rows": int(night.sum()),
        "night_label_distribution": label_dist(night),
        "day_label_distribution":   label_dist(day),
        "night_fault_rows": night_fault_rows,
        "night_is_all_normal": night_fault_rows == 0,
        "daytime_normal_rows": day_normal_rows,
        "daytime_fault_rows":  day_fault_rows,
        "recommendation": (
            "Filter to is_daytime=True before Task A/B — night data is trivially all-normal "
            "and would inflate normal-vs-fault separation via irradiance. "
            f"Daytime rows: {int(day.sum()):,} ({100*day.mean():.1f}% of dataset)."
            if night_fault_rows == 0
            else "Night contains fault rows — daytime filter not strictly required."
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
        for (fc, mode), grp in df.groupby(["fault_class", "mode"]):
            exp_counts.append({
                "fault_class": int(fc),
                "mode": str(mode),
                "rows": len(grp),
                "phases": grp["phase"].value_counts().to_dict(),
            })

    return {
        "total_rows": total,
        "phase_distribution": {k: int(v) for k, v in phase_counts.items()},
        "experiments": exp_counts,
        "usable_for_eda": {
            "normal_rows":  int((df["fault_class"] == 0).sum()),
            "fault_rows":   int((df["phase"] == "fault").sum()),
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
    current_gap    = ds_splits.get(
        "segmentation_gap_seconds",
        global_splits.get("segmentation_gap_seconds", 300),
    )
    recommended_strategy = gap_analysis.get("recommended_segmentation_strategy", "gap_based")

    changes_needed = []
    if recommended_strategy != current_strategy:
        changes_needed.append(
            f"segmentation_strategy: '{current_strategy}' → '{recommended_strategy}'"
        )

    current_eval  = ds_splits.get("evaluable_classes", [])
    rec_eval      = class_dist.get("recommended_evaluable_classes", [])
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
    parser = argparse.ArgumentParser(description="Run per-dataset EDA and export findings.")
    parser.add_argument(
        "--dataset",
        default="la_reunion",
        help="Dataset to analyse (must match data_config.yaml paths.datasets key). Default: la_reunion",
    )
    parser.add_argument(
        "--daytime-only",
        action="store_true",
        default=False,
        help="Costa only: filter to is_daytime=True before computing statistics.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EDA PIPELINE — dataset={}", args.dataset)
    logger.info("=" * 60)

    config     = load_config()
    ds_cfg     = get_dataset_cfg(config, args.dataset)
    ds_splits  = ds_cfg.get("splits", {})
    gap_s      = ds_splits.get(
        "segmentation_gap_seconds",
        config.get("splits", {}).get("segmentation_gap_seconds", 300),
    )
    ds_fe      = ds_cfg.get("feature_engineering", {})
    sensor_cols = ds_fe.get("sensor_columns", [])
    global_sel  = config.get("feature_engineering", {}).get("selection", {})
    corr_thr    = float(global_sel.get("corr_threshold", 0.95))
    vif_thr     = float(global_sel.get("vif_threshold", 10.0))

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
        logger.warning("  {} → recommend switching to label_transition strategy",
                       gap_analysis["note"])

    # -------------------------------------------------------------------------
    # 3. Class / segment distribution
    # -------------------------------------------------------------------------
    logger.info("[2/5] Class distribution ...")
    class_dist = analyse_class_distribution(df_raw, gap_s)
    for row in class_dist["per_class"]:
        logger.info(
            "  label={} | {:>8,} rows ({:5.1f}%) | {:3d} segments | evaluable={}",
            row["label"], row["n_rows"], row["pct_rows"],
            row["n_segments"], row["evaluable"],
        )

    # -------------------------------------------------------------------------
    # 4. Dataset-specific diagnostics
    # -------------------------------------------------------------------------
    logger.info("[3/5] Dataset-specific diagnostics ...")
    dataset_specific: dict = {}
    if args.dataset == "costa":
        dataset_specific = diagnose_costa(df_raw)
        logger.info("  Costa night-is-all-normal: {}", dataset_specific.get("night_is_all_normal"))
    elif args.dataset == "mendeley":
        dataset_specific = diagnose_mendeley(df_raw)
        logger.info("  Mendeley usable rows: {}", dataset_specific.get("usable_for_eda"))

    # -------------------------------------------------------------------------
    # 5. Statistical tests — on filtered/prepared EDA frame
    # -------------------------------------------------------------------------
    logger.info("[4/5] Statistical tests (Mann-Whitney, Spearman, VIF, MI) ...")
    df_eda, eda_desc = prepare_eda_frame(df_raw, args.dataset, args.daytime_only)

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
        output_dir=output_dir,
        corr_threshold=corr_thr,
        vif_threshold=vif_thr,
    )
    # Patch dataset field (export_eda_feature_findings hardcodes "reunion_dt2")
    consolidated["dataset"] = args.dataset
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
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.success("Dataset report → {}", report_path)

    # Re-write consolidated findings with patched dataset field
    consolidated_path = output_dir / "eda_feature_findings.json"
    consolidated_path.write_text(
        json.dumps(consolidated, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    logger.success("Consolidated findings → {}", consolidated_path)

    # Summary
    logger.info("=" * 60)
    logger.info("EDA complete | dataset={}", args.dataset)
    logger.info("  Segments (gap-based):        {:>4}", gap_analysis["n_segments_gap_based"])
    logger.info("  Segments (label-transition): {:>4}", gap_analysis["n_segments_label_transition"])
    logger.info("  Evaluable fault classes:     {}", split_recs["recommended_evaluable_classes"])
    logger.info("  Top binary MI features:      {}",
                consolidated.get("mutual_information", {}).get("top_features_binary", [])[:5])
    if split_recs["action_required"]:
        logger.warning("  ACTION REQUIRED — update data_config.yaml:")
        for c in split_recs["config_changes_needed"]:
            logger.warning("    {}", c)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
