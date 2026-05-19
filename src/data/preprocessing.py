"""
Preprocessing module for PV Fault Detection.

This module implements data preprocessing steps:
1. Missing value handling (tiered strategy)
2. Outlier treatment (IQR-based row dropping)
3. Optional feature transforms (handled outside core preprocessing pipeline)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES FOR TRACKING
# =============================================================================


@dataclass
class MissingValueStats:
    """Statistics from missing value handling."""

    input_rows: int
    output_rows: int
    rows_dropped_fault_overlap: int
    rows_dropped_long_gap: int
    episodes_ffilled: int
    episodes_interpolated: int

    @property
    def total_dropped(self) -> int:
        return self.rows_dropped_fault_overlap + self.rows_dropped_long_gap


@dataclass
class OutlierStats:
    """Statistics from outlier row dropping."""

    dropped_counts: dict[str, int]  # feature -> count of outlier rows flagged by feature
    bounds: dict[str, tuple[float, float]]  # feature -> (lower, upper)
    rows_dropped_total: int


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================


def identify_null_episodes(
    df: pd.DataFrame,
    feature_cols: list[str],
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Identify contiguous blocks of null values.

    Returns DataFrame with columns:
    - start: timestamp of first null
    - end: timestamp of last null
    - n_rows: number of null rows
    - duration_seconds: duration of the null episode
    """
    # Create combined null mask across all features
    null_mask = df[feature_cols].isnull().any(axis=1)

    if not null_mask.any():
        return pd.DataFrame(columns=["start", "end", "n_rows", "duration_seconds", "indices"])

    # Find episode boundaries using diff
    episode_starts = null_mask & ~null_mask.shift(1, fill_value=False)
    episode_ends = null_mask & ~null_mask.shift(-1, fill_value=False)

    start_indices = df.index[episode_starts].tolist()
    end_indices = df.index[episode_ends].tolist()

    episodes = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        episode_mask = (df.index >= start_idx) & (df.index <= end_idx)
        episode_df = df.loc[episode_mask]

        episodes.append(
            {
                "start": episode_df[timestamp_col].iloc[0],
                "end": episode_df[timestamp_col].iloc[-1],
                "n_rows": len(episode_df),
                "duration_seconds": (
                    episode_df[timestamp_col].iloc[-1] - episode_df[timestamp_col].iloc[0]
                ).total_seconds(),
                "indices": episode_df.index.tolist(),
            }
        )

    return pd.DataFrame(episodes)


def handle_missing_values(
    df: pd.DataFrame,
    feature_cols: list[str],
    ffill_max_gap_seconds: float = 60.0,
    interp_max_gap_seconds: float = 300.0,
    timestamp_col: str = "timestamp",
    segment_col: str = "segment_id",
    label_col: str = "Fault",
) -> tuple[pd.DataFrame, MissingValueStats]:
    """
    Handle missing values with tiered strategy.

    Strategy:
    - < ffill_max_gap_seconds: Forward-fill (normal data only)
    - ffill_max_gap_seconds to interp_max_gap_seconds: Linear interpolation (normal data only)
    - > interp_max_gap_seconds: DROP rows
    - Any gap on fault data: DROP rows

    Args:
        df: Input DataFrame (will be copied)
        feature_cols: Columns to check for nulls
        ffill_max_gap_seconds: Max gap duration for forward-fill
        interp_max_gap_seconds: Max gap duration for interpolation
        timestamp_col: Name of timestamp column
        segment_col: Name of segment ID column
        label_col: Name of label column (0 = normal)

    Returns:
        Tuple of (processed DataFrame, statistics)
    """
    df = df.copy()
    input_rows = len(df)

    rows_dropped_fault = 0
    rows_dropped_long = 0
    episodes_ffilled = 0
    episodes_interp = 0

    # Process each segment independently
    segments = df[segment_col].unique()

    indices_to_drop = []

    for seg_id in segments:
        seg_mask = df[segment_col] == seg_id
        seg_df = df.loc[seg_mask]

        # Identify null episodes in this segment
        episodes = identify_null_episodes(seg_df, feature_cols, timestamp_col)

        if episodes.empty:
            continue

        for _, episode in episodes.iterrows():
            indices = episode["indices"]
            duration = episode["duration_seconds"]

            # Check if any row in episode has fault label
            has_fault = (df.loc[indices, label_col] != 0).any()

            if has_fault:
                # Rule: DROP any nulls on fault data
                indices_to_drop.extend(indices)
                rows_dropped_fault += len(indices)
                logger.debug(f"Segment {seg_id}: Dropping {len(indices)} null rows (fault overlap)")

            elif duration > interp_max_gap_seconds:
                # Rule: DROP gaps > interp_max_gap_seconds
                indices_to_drop.extend(indices)
                rows_dropped_long += len(indices)
                logger.debug(
                    f"Segment {seg_id}: Dropping {len(indices)} null rows "
                    f"(gap {duration:.0f}s > {interp_max_gap_seconds}s)"
                )

            elif duration <= ffill_max_gap_seconds:
                # Forward-fill short gaps within segment context
                # Use segment-level ffill (not just the indices) to get previous values
                for col in feature_cols:
                    # Fill within segment, then take only the filled values for our indices
                    segment_filled = df.loc[seg_mask, col].ffill()
                    # If ffill didn't work (first row of segment is NaN), try bfill
                    if segment_filled.loc[indices].isna().any():
                        segment_filled = segment_filled.bfill()
                    df.loc[indices, col] = segment_filled.loc[indices]
                episodes_ffilled += 1

            else:
                # Linear interpolation for medium gaps
                for col in feature_cols:
                    df.loc[seg_mask, col] = df.loc[seg_mask, col].interpolate(
                        method="linear", limit_area="inside"
                    )
                episodes_interp += 1

    # Drop collected indices
    if indices_to_drop:
        df = df.drop(index=indices_to_drop)

    stats = MissingValueStats(
        input_rows=input_rows,
        output_rows=len(df),
        rows_dropped_fault_overlap=rows_dropped_fault,
        rows_dropped_long_gap=rows_dropped_long,
        episodes_ffilled=episodes_ffilled,
        episodes_interpolated=episodes_interp,
    )

    logger.info(
        f"Missing value handling: {input_rows} → {len(df)} rows "
        f"(dropped {stats.total_dropped}: {rows_dropped_fault} fault overlap, "
        f"{rows_dropped_long} long gaps)"
    )

    return df, stats


# =============================================================================
# OUTLIER TREATMENT
# =============================================================================


def compute_iqr_bounds(
    series: pd.Series,
    multiplier: float = 3.0,
) -> tuple[float, float]:
    """Compute IQR-based bounds for outlier detection."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return lower, upper


def drop_outliers_iqr(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "Fault",
    iqr_multiplier: float = 3.0,
    scope: Literal["normal_only", "all"] = "normal_only",
    reference_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, OutlierStats]:
    """
    IQR-based outlier handling by dropping outlier rows.

    For each feature, outliers are detected with IQR bounds (Q1 - k×IQR, Q3 + k×IQR).
    Any row flagged as an outlier in at least one feature is dropped.

    Args:
        df: Input DataFrame (will be copied)
        feature_cols: Columns to process
        label_col: Label column (0 = normal)
        iqr_multiplier: IQR multiplier for detection (3.0 = far outliers)
        scope: only "normal_only" is supported for dropping
        reference_bounds: Train-fit IQR bounds for val/test

    Returns:
        Tuple of (processed DataFrame, statistics)
    """
    df = df.copy()

    if scope != "normal_only":
        raise ValueError("IQR outlier dropping supports only scope='normal_only'")

    normal_mask = df[label_col] == 0
    dropped_counts: dict[str, int] = {}
    bounds: dict[str, tuple[float, float]] = {}
    rows_to_drop = pd.Series(False, index=df.index)

    for col in feature_cols:
        # Compute on normal data only
        col_data = df.loc[normal_mask, col].dropna()
        if col_data.empty and not (reference_bounds and col in reference_bounds):
            continue

        # Fit IQR bounds (train) or reuse train-fit bounds (val/test)
        if reference_bounds and col in reference_bounds:
            iqr_lower, iqr_upper = reference_bounds[col]
        elif col_data.empty:
            iqr_lower, iqr_upper = -float("inf"), float("inf")
        else:
            iqr_lower, iqr_upper = compute_iqr_bounds(col_data, iqr_multiplier)

        bounds[col] = (float(iqr_lower), float(iqr_upper))

        process_mask = normal_mask

        # Identify outliers using IQR bounds
        outlier_mask = process_mask & ((df[col] < iqr_lower) | (df[col] > iqr_upper))
        dropped_counts[col] = int(outlier_mask.sum())
        rows_to_drop |= outlier_mask

        if dropped_counts[col] > 0:
            logger.debug(
                f"  {col}: detected {dropped_counts[col]} normal-row outliers "
                f"(IQR bounds [{iqr_lower:.2f}, {iqr_upper:.2f}])"
            )

    rows_dropped_total = int(rows_to_drop.sum())
    if rows_dropped_total > 0:
        df = df.loc[~rows_to_drop].copy()

    logger.info(
        f"Outlier dropping (normal-only): dropped {rows_dropped_total} rows "
        f"across {len(feature_cols)} features"
    )

    stats = OutlierStats(
        dropped_counts=dropped_counts,
        bounds=bounds,
        rows_dropped_total=rows_dropped_total,
    )
    return df, stats


# =============================================================================
# UNIFIED PREPROCESSING FUNCTION
# =============================================================================


def preprocess(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
    timestamp_col: str = "timestamp",
    segment_col: str = "segment_id",
    label_col: str = "Fault",
    outlier_reference_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run full preprocessing pipeline.

    Args:
        df: Input DataFrame
        feature_cols: Feature columns to process
        config: Preprocessing config dict (from data_config.yaml)
        timestamp_col: Timestamp column name
        segment_col: Segment ID column name
        label_col: Label column name
        outlier_reference_bounds: Train-fit outlier bounds for val/test

    Returns:
        Tuple of (preprocessed DataFrame, statistics dict)
        Statistics include outlier bounds for the caller to store and forward to val/test.
    """
    logger.info(f"Starting preprocessing pipeline on {len(df)} rows...")

    # 1. Missing value handling
    mv_config = config.get("missing_values", {})
    if mv_config.get("enabled", True):
        df, mv_stats = handle_missing_values(
            df,
            feature_cols=feature_cols,
            ffill_max_gap_seconds=mv_config.get("ffill_max_gap_seconds", 60),
            interp_max_gap_seconds=mv_config.get("interp_max_gap_seconds", 300),
            timestamp_col=timestamp_col,
            segment_col=segment_col,
            label_col=label_col,
        )
    else:
        mv_stats = MissingValueStats(
            input_rows=len(df),
            output_rows=len(df),
            rows_dropped_fault_overlap=0,
            rows_dropped_long_gap=0,
            episodes_ffilled=0,
            episodes_interpolated=0,
        )
        logger.info("Missing value handling disabled by config; skipping imputation stage")

    # 2. Outlier treatment
    outlier_config = config.get("outliers", {})
    method = outlier_config.get("method", "iqr_drop")
    if method != "iqr_drop":
        raise ValueError(
            f"Unsupported outlier method '{method}'. Supported method: 'iqr_drop'."
        )

    df, outlier_stats = drop_outliers_iqr(
        df,
        feature_cols=feature_cols,
        label_col=label_col,
        iqr_multiplier=outlier_config.get("iqr_multiplier", 3.0),
        scope=outlier_config.get("scope", "normal_only"),
        reference_bounds=outlier_reference_bounds,
    )

    stats = {
        "missing_values": {
            "input_rows": mv_stats.input_rows,
            "output_rows": mv_stats.output_rows,
            "rows_dropped_fault_overlap": mv_stats.rows_dropped_fault_overlap,
            "rows_dropped_long_gap": mv_stats.rows_dropped_long_gap,
            "episodes_ffilled": mv_stats.episodes_ffilled,
            "episodes_interpolated": mv_stats.episodes_interpolated,
        },
        "outliers": {
            "dropped_counts": outlier_stats.dropped_counts,
            "rows_dropped_total": outlier_stats.rows_dropped_total,
            "bounds": {k: list(v) for k, v in outlier_stats.bounds.items()},
        },
    }

    logger.info(f"Preprocessing complete: {len(df)} rows output")
    return df, stats
