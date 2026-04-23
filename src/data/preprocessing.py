"""
Preprocessing module for PV Fault Detection.

This module implements data preprocessing steps:
1. Missing value handling (tiered strategy)
2. Outlier treatment (IQR-based clipping)
3. Physics normalization:
   - Irradiance normalization: power/current ÷ irr ≈ efficiency/yield coefficients
   - Irradiance-conditioned residualization: temp − (β·irr + α), OLS on normal train
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
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
    """Statistics from outlier clipping."""

    clipped_counts: dict[str, int]  # feature -> count of clipped values
    bounds: dict[str, tuple[float, float]]  # feature -> (lower, upper)


@dataclass
class PhysicsNormStats:
    """Statistics from physics normalization transforms."""

    features_normalized: list[str]
    features_residualized: list[str]
    irr_residual_params: dict[str, tuple[float, float]]  # feat -> (beta, alpha)


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


def winsorize_outliers(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "Fault",
    iqr_multiplier: float = 3.0,
    lower_percentile: float = 0.05,
    upper_percentile: float = 0.95,
    scope: Literal["normal_only", "all"] = "normal_only",
    reference_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, OutlierStats]:
    """
    Two-step outlier handling: IQR detection + percentile winsorizing.

    Step 1: Detect outliers using IQR bounds (Q1 - k×IQR, Q3 + k×IQR)
    Step 2: Replace detected outliers with percentile values (5th, 95th)

    This approach:
    - Uses conservative IQR bounds to flag extreme values
    - Replaces them with less extreme percentile values
    - Preserves more data distribution than hard clipping

    Args:
        df: Input DataFrame (will be copied)
        feature_cols: Columns to process
        label_col: Label column (0 = normal)
        iqr_multiplier: IQR multiplier for detection (3.0 = far outliers)
        lower_percentile: Lower replacement value (e.g., 0.05 = 5th percentile)
        upper_percentile: Upper replacement value (e.g., 0.95 = 95th percentile)
        scope: "normal_only" processes only normal data, "all" processes everything

    Returns:
        Tuple of (processed DataFrame, statistics)
    """
    df = df.copy()

    normal_mask = df[label_col] == 0
    clipped_counts: dict[str, int] = {}
    bounds: dict[str, tuple[float, float]] = {}

    for col in feature_cols:
        # Compute on normal data only
        col_data = df.loc[normal_mask, col].dropna()
        if col_data.empty and not (reference_bounds and col in reference_bounds):
            continue

        # Step 1: Compute IQR bounds for outlier DETECTION
        if col_data.empty:
            iqr_lower, iqr_upper = -np.inf, np.inf
        else:
            iqr_lower, iqr_upper = compute_iqr_bounds(col_data, iqr_multiplier)

        # Step 2: Compute percentile values for REPLACEMENT or reuse train-fit values
        if reference_bounds and col in reference_bounds:
            percentile_lower, percentile_upper = reference_bounds[col]
        else:
            percentile_lower = col_data.quantile(lower_percentile)
            percentile_upper = col_data.quantile(upper_percentile)

        bounds[col] = (float(percentile_lower), float(percentile_upper))

        # Apply based on scope
        if scope == "normal_only":
            process_mask = normal_mask
        else:
            process_mask = pd.Series(True, index=df.index)

        # Identify outliers using IQR bounds
        outlier_mask = process_mask & ((df[col] < iqr_lower) | (df[col] > iqr_upper))
        clipped_counts[col] = outlier_mask.sum()

        # Replace outliers with percentile values
        df.loc[process_mask & (df[col] < iqr_lower), col] = percentile_lower
        df.loc[process_mask & (df[col] > iqr_upper), col] = percentile_upper

        if clipped_counts[col] > 0:
            logger.debug(
                f"  {col}: detected {clipped_counts[col]} outliers (IQR bounds [{iqr_lower:.2f}, {iqr_upper:.2f}]), "
                f"replaced with percentiles [{percentile_lower:.2f}, {percentile_upper:.2f}]"
            )

    total_clipped = sum(clipped_counts.values())
    logger.info(
        f"Outlier winsorizing: {total_clipped} values winsorized across {len(feature_cols)} features"
    )

    stats = OutlierStats(clipped_counts=clipped_counts, bounds=bounds)
    return df, stats


# =============================================================================
# PHYSICS NORMALIZATION
# =============================================================================


def apply_irradiance_normalization(
    df: pd.DataFrame,
    features: list[str],
    denominator: str = "GTI",
    suffix: str = "_norm",
    min_denominator: float = 5.0,
) -> pd.DataFrame:
    """
    Normalize power/current features by irradiance: feature ÷ irr ≈ efficiency/yield.

    Removes the irradiance-driven diurnal swing from power and current channels,
    making the normal manifold approximately time-invariant for anomaly detectors.
    Floors the denominator at min_denominator to avoid division by near-zero at dawn/dusk.
    """
    df = df.copy()

    low_irr_count = (df[denominator] < min_denominator).sum()
    if low_irr_count > 0:
        logger.info(
            f"Found {low_irr_count} rows with {denominator} < {min_denominator} W/m²; "
            f"flooring denominator for safe division."
        )

    safe_irr = df[denominator].clip(lower=min_denominator)
    for feat in features:
        df[f"{feat}{suffix}"] = df[feat] / safe_irr
        logger.debug(f"  {feat}{suffix} = {feat} / max({denominator}, {min_denominator})")

    logger.info(f"Irradiance normalization: created {len(features)} features ({suffix})")
    return df


def apply_irr_conditioned_residualization(
    df: pd.DataFrame,
    features: list[str],
    irr_col: str,
    label_col: str = "Fault",
    suffix: str = "_irr_residual",
    reference_params: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Irradiance-conditioned residualization for temperature features.

    Fits feature = β·irr + α via OLS on normal-class rows, then subtracts the
    irradiance-predicted baseline: residual = feature − (β·irr + α).

    The residual is stationary where raw temperature was not: it captures thermal
    deviation from what the irradiance level would predict, which is near-zero
    for healthy operation and anomalous during fault-induced thermal events.

    Args:
        df: Input DataFrame (will be copied)
        features: Temperature columns to residualize (e.g., ["pvt"])
        irr_col: Irradiance column used as the conditioning variable
        label_col: Label column; OLS is fit on label==0 rows only
        suffix: Suffix for new residual columns
        reference_params: Pre-fitted (beta, alpha) per feature for val/test splits.
                          If None, fits from df (train mode).

    Returns:
        (DataFrame with residual columns, fitted_params dict feat->(beta, alpha))
    """
    df = df.copy()
    fitted_params: dict[str, tuple[float, float]] = {}

    normal_mask = df[label_col] == 0
    irr_all = df[irr_col].values

    for feat in features:
        if reference_params and feat in reference_params:
            beta, alpha = reference_params[feat]
        else:
            irr_train = irr_all[normal_mask]
            y_train = df.loc[normal_mask, feat].values

            irr_mean = irr_train.mean()
            y_mean = y_train.mean()
            denom = float(np.dot(irr_train - irr_mean, irr_train - irr_mean))
            if denom == 0:
                beta, alpha = 0.0, float(y_mean)
            else:
                beta = float(np.dot(irr_train - irr_mean, y_train - y_mean) / denom)
                alpha = float(y_mean - beta * irr_mean)

        fitted_params[feat] = (beta, alpha)
        df[f"{feat}{suffix}"] = df[feat] - (beta * irr_all + alpha)
        logger.debug(f"  {feat}{suffix} = {feat} - ({beta:.4f}·{irr_col} + {alpha:.4f})")

    logger.info(
        f"Irradiance-conditioned residualization: created {len(features)} features ({suffix})"
    )
    return df, fitted_params


def apply_physics_normalization(
    df: pd.DataFrame,
    irradiance_config: dict | None = None,
    irr_residualize_config: dict | None = None,
    label_col: str = "Fault",
    reference_irr_residual_params: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, PhysicsNormStats]:
    """
    Apply all physics normalization transforms.

    Args:
        df: Input DataFrame
        irradiance_config: Dict with keys: features, denominator, suffix
        irr_residualize_config: Dict with keys: features, irr_col, suffix
        label_col: Label column (used for OLS fit scope)
        reference_irr_residual_params: Pre-fitted OLS params for val/test splits

    Returns:
        (processed DataFrame, PhysicsNormStats)
    """
    features_normalized: list[str] = []
    features_residualized: list[str] = []
    irr_residual_params: dict[str, tuple[float, float]] = {}

    if irradiance_config:
        features = irradiance_config.get("features", [])
        denominator = irradiance_config.get("denominator", "GTI")
        suffix = irradiance_config.get("suffix", "_norm")
        df = apply_irradiance_normalization(df, features, denominator, suffix)
        features_normalized = [f"{f}{suffix}" for f in features]

    if irr_residualize_config:
        features = irr_residualize_config.get("features", [])
        irr_col = irr_residualize_config.get("irr_col", "irr")
        suffix = irr_residualize_config.get("suffix", "_irr_residual")
        df, irr_residual_params = apply_irr_conditioned_residualization(
            df,
            features=features,
            irr_col=irr_col,
            label_col=label_col,
            suffix=suffix,
            reference_params=reference_irr_residual_params,
        )
        features_residualized = [f"{f}{suffix}" for f in features]

    return df, PhysicsNormStats(
        features_normalized=features_normalized,
        features_residualized=features_residualized,
        irr_residual_params=irr_residual_params,
    )


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
    irr_residual_reference_params: dict[str, tuple[float, float]] | None = None,
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
        irr_residual_reference_params: Train-fit OLS params for val/test (feat->(beta,alpha))

    Returns:
        Tuple of (preprocessed DataFrame, statistics dict)
        Statistics include irr_residual_params for the caller to store and forward to val/test.
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
    df, outlier_stats = winsorize_outliers(
        df,
        feature_cols=feature_cols,
        label_col=label_col,
        iqr_multiplier=outlier_config.get("iqr_multiplier", 3.0),
        lower_percentile=outlier_config.get("lower_percentile", 0.05),
        upper_percentile=outlier_config.get("upper_percentile", 0.95),
        scope=outlier_config.get("scope", "normal_only"),
        reference_bounds=outlier_reference_bounds,
    )

    # 3. Physics normalization
    stat_config = config.get("physics_normalization") or {}
    df, stat_stats = apply_physics_normalization(
        df,
        irradiance_config=stat_config.get("irradiance_normalize"),
        irr_residualize_config=stat_config.get("irr_residualize"),
        label_col=label_col,
        reference_irr_residual_params=irr_residual_reference_params,
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
            "clipped_counts": outlier_stats.clipped_counts,
            "bounds": {k: list(v) for k, v in outlier_stats.bounds.items()},
        },
        "physics_normalization": {
            "features_normalized": stat_stats.features_normalized,
            "features_residualized": stat_stats.features_residualized,
            "irr_residual_params": {k: list(v) for k, v in stat_stats.irr_residual_params.items()},
        },
    }

    logger.info(f"Preprocessing complete: {len(df)} rows output")
    return df, stats
