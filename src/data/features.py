"""
Feature engineering utilities for PV fault detection.

This module provides:
1. Flag-driven physics/temporal feature creation
2. Segment-safe windowing
3. Correlation/VIF pruning
4. Optional tsfresh extraction for ML baselines
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
from loguru import logger
from scipy.fft import rfft

# ============================================================================
# BASIC HELPERS
# ============================================================================

DEFAULT_BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    "Ia",
    "Ig",
    "Eg",
    "Fg",
    "Pg",
    "Va",
    "Vg",
    "GTI",
    "DTI",
    "TA",
    "TPV",
)


def infer_label_column(df: pd.DataFrame, preferred: tuple[str, ...] = ("label", "Fault")) -> str:
    """Infer the label column from preferred candidates."""
    for col in preferred:
        if col in df.columns:
            return col
    raise ValueError(f"Could not infer label column from candidates: {preferred}")


def infer_base_feature_columns(
    df: pd.DataFrame,
    preferred: Iterable[str] = DEFAULT_BASE_FEATURE_COLUMNS,
) -> list[str]:
    """Infer base engineering columns from known canonical signals."""
    return [col for col in preferred if col in df.columns]


def add_time_cyclic_features(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    hour_sin_col: str = "hour_sin",
    hour_cos_col: str = "hour_cos",
) -> pd.DataFrame:
    """Add cyclic time-of-day features."""
    if time_col not in df.columns:
        return df

    out = df.copy()
    dt = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    hour_float = dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0
    angle = 2.0 * np.pi * hour_float / 24.0
    out[hour_sin_col] = np.sin(angle)
    out[hour_cos_col] = np.cos(angle)
    return out


def _safe_groupby_diff_per_second(
    df: pd.DataFrame,
    value_col: str,
    time_col: str,
    segment_col: str,
) -> pd.Series:
    """Compute segment-safe derivative d(value)/dt in seconds."""
    if value_col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype=np.float64)
    if time_col not in df.columns or segment_col not in df.columns:
        return df[value_col].diff().fillna(0.0)

    dt_seconds = (
        df.groupby(segment_col)[time_col]
        .diff()
        .dt.total_seconds()
        .replace(0, np.nan)
    )

    median_dt = dt_seconds[dt_seconds > 0].median()
    if pd.isna(median_dt) or median_dt <= 0:
        median_dt = 1.0

    dt_seconds = dt_seconds.fillna(median_dt)
    dv = df.groupby(segment_col)[value_col].diff().fillna(0.0)
    return dv / dt_seconds


def add_physics_features(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    time_col: str = "timestamp",
    flags: dict | None = None,
    gti_min: float = 5.0,
    vg_window: int = 100,
) -> pd.DataFrame:
    """
    Add physics-informed features controlled by flags.

    Supported flags:
      - enable_performance_ratio
      - enable_delta_temp
      - enable_dP_dt
      - enable_dV_dt
      - enable_dI_dt
      - enable_Vg_normalized
    """
    out = df.copy()
    flags = flags or {}

    # Keep deterministic temporal behavior for segment operations.
    if segment_col in out.columns and time_col in out.columns:
        original_index = out.index
        out = out.sort_values([segment_col, time_col])
    else:
        original_index = None

    if flags.get("enable_performance_ratio", True) and {"Pg", "GTI"}.issubset(out.columns):
        gti_safe = out["GTI"].clip(lower=gti_min)
        out["performance_ratio"] = out["Pg"] / (gti_safe / 1000.0)

    if flags.get("enable_delta_temp", True) and {"TPV", "TA"}.issubset(out.columns):
        out["delta_temp"] = out["TPV"] - out["TA"]

    if flags.get("enable_dP_dt", True) and "Pg" in out.columns:
        out["dP_dt"] = _safe_groupby_diff_per_second(out, "Pg", time_col, segment_col)

    if flags.get("enable_dV_dt", True) and "Vg" in out.columns:
        out["dV_dt"] = _safe_groupby_diff_per_second(out, "Vg", time_col, segment_col)

    if flags.get("enable_dI_dt", True) and "Ig" in out.columns:
        out["dI_dt"] = _safe_groupby_diff_per_second(out, "Ig", time_col, segment_col)

    if flags.get("enable_Vg_normalized", True) and "Vg" in out.columns:
        if segment_col in out.columns:
            rolling_med = out.groupby(segment_col)["Vg"].transform(
                lambda s: s.rolling(window=vg_window, min_periods=1).median()
            )
        else:
            rolling_med = out["Vg"].rolling(window=vg_window, min_periods=1).median()
        out["Vg_normalized"] = out["Vg"] / rolling_med.replace(0, np.nan)
        out["Vg_normalized"] = out["Vg_normalized"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if original_index is not None:
        out = out.loc[original_index]

    return out


# ============================================================================
# SIGNAL FEATURES
# ============================================================================


def wavelet_denoise_series(series: np.ndarray, wavelet: str = "db4", level: int = 4) -> np.ndarray:
    """Denoise a 1D series using wavelet thresholding."""
    import pywt

    coeffs = pywt.wavedec(series, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(series))) * np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs_thresh = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)[: len(series)]


def add_wavelet_feature(
    df: pd.DataFrame,
    source_col: str = "Pg",
    segment_col: str = "segment_id",
    target_col: str = "Pg_wavelet",
) -> pd.DataFrame:
    """Add a denoised wavelet feature from source_col."""
    if source_col not in df.columns:
        return df

    out = df.copy()

    if segment_col not in out.columns:
        out[target_col] = wavelet_denoise_series(out[source_col].to_numpy(dtype=np.float64))
        return out

    out[target_col] = 0.0
    for seg_id, seg_df in out.groupby(segment_col):
        den = wavelet_denoise_series(seg_df[source_col].to_numpy(dtype=np.float64))
        out.loc[seg_df.index, target_col] = den
    return out


# ============================================================================
# WINDOWING + STATS
# ============================================================================


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int,
    step: int,
    feature_cols: list[str],
    label_col: str | None = None,
    segment_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Convert DataFrame into overlapping windows.

    If segment_col is provided, windows are created independently per segment,
    so windows never cross segment/day boundaries.
    """
    x_list: list[np.ndarray] = []
    y_list: list[float] = []

    if segment_col and segment_col in df.columns:
        iter_frames: Iterable[pd.DataFrame] = (g for _, g in df.groupby(segment_col, sort=False))
    else:
        iter_frames = [df]

    for frame in iter_frames:
        values = frame[feature_cols].to_numpy(dtype=np.float32)
        labels = frame[label_col].to_numpy() if label_col else None

        for start in range(0, len(frame) - window_size + 1, step):
            end = start + window_size
            x_list.append(values[start:end])
            if labels is not None:
                window_labels = labels[start:end]
                y_list.append(pd.Series(window_labels).mode().iloc[0])

    x = np.stack(x_list) if x_list else np.empty((0, window_size, len(feature_cols)))
    y = np.array(y_list) if y_list else None
    return x, y


def extract_window_statistics(x: np.ndarray) -> np.ndarray:
    """
    Collapse windows to statistical features.
    Input: (n_windows, window_size, n_features)
    Output: (n_windows, n_features * 8)
    """
    from scipy.stats import kurtosis, skew

    n, w, f = x.shape
    features = []
    for i in range(f):
        ch = x[:, :, i]
        features.append(ch.mean(axis=1))
        features.append(ch.std(axis=1))
        features.append(ch.min(axis=1))
        features.append(ch.max(axis=1))
        features.append(skew(ch, axis=1))
        features.append(kurtosis(ch, axis=1))
        features.append(np.sqrt((ch**2).mean(axis=1)))
        zcr = ((np.diff(np.sign(ch), axis=1) != 0).sum(axis=1)) / max(w - 1, 1)
        features.append(zcr)

    return np.column_stack(features).astype(np.float32)


def extract_fft_features(x: np.ndarray, n_top_freqs: int = 5) -> np.ndarray:
    """
    Extract top-N spectral power features from each window and channel.
    Input: (n_windows, window_size, n_features)
    Output: (n_windows, n_features * n_top_freqs)
    """
    _, _, f = x.shape
    results = []
    for i in range(f):
        ch = x[:, :, i]
        fft_mag = np.abs(rfft(ch, axis=1))[:, 1:]
        idx = np.argsort(fft_mag, axis=1)[:, -n_top_freqs:]
        top_mags = np.take_along_axis(fft_mag, idx, axis=1)
        results.append(top_mags)
    return np.concatenate(results, axis=1).astype(np.float32)


# ============================================================================
# CORRELATION / COLLINEARITY PRUNING
# ============================================================================


def _numeric_existing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def apply_correlation_pruning(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.95,
    anchor_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[dict]]:
    """Drop highly correlated features using Spearman correlation on train split only."""
    anchor_set = set(anchor_cols or [])
    cols = _numeric_existing_columns(train_df, feature_cols)

    if len(cols) < 2:
        return train_df, val_df, test_df, cols, []

    corr = train_df[cols].corr(method="spearman").abs()
    dropped: set[str] = set()
    dropped_log: list[dict] = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in dropped or b in dropped:
                continue
            rho = corr.loc[a, b]
            if pd.isna(rho) or rho < threshold:
                continue

            if a in anchor_set and b in anchor_set:
                continue
            if a in anchor_set:
                drop_col = b
                reason = "drop_non_anchor_against_anchor"
            elif b in anchor_set:
                drop_col = a
                reason = "drop_non_anchor_against_anchor"
            else:
                mean_a = corr[a].drop(a).mean()
                mean_b = corr[b].drop(b).mean()
                drop_col = a if mean_a > mean_b else b
                reason = "drop_higher_mean_abs_corr"

            dropped.add(drop_col)
            dropped_log.append({"dropped": drop_col, "paired_with": b if drop_col == a else a, "rho": float(rho), "reason": reason})

    selected_cols = [c for c in cols if c not in dropped]
    keep_extra = [c for c in train_df.columns if c not in feature_cols]
    keep_cols = keep_extra + selected_cols

    logger.info(f"Correlation pruning: kept {len(selected_cols)}/{len(cols)} features (threshold={threshold})")
    return train_df[keep_cols].copy(), val_df[keep_cols].copy(), test_df[keep_cols].copy(), selected_cols, dropped_log


def apply_vif_pruning(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 10.0,
    anchor_cols: Iterable[str] | None = None,
    max_rows: int = 50000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[dict]]:
    """Iteratively prune collinear features based on VIF computed on train split only."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    anchor_set = set(anchor_cols or [])
    selected = _numeric_existing_columns(train_df, feature_cols)
    dropped_log: list[dict] = []

    if len(selected) < 2:
        return train_df, val_df, test_df, selected, dropped_log

    work = train_df[selected].dropna().copy()
    if len(work) > max_rows:
        work = work.sample(max_rows, random_state=42)

    while len(selected) >= 2:
        x = work[selected]
        # Drop constant columns immediately.
        constant_cols = [c for c in selected if np.isclose(x[c].std(), 0.0, atol=1e-12)]
        if constant_cols:
            for c in constant_cols:
                selected.remove(c)
                dropped_log.append({"dropped": c, "vif": np.inf, "reason": "constant_column"})
            continue

        vifs: dict[str, float] = {}
        for i, col in enumerate(selected):
            try:
                vif_val = float(variance_inflation_factor(x.values, i))
            except Exception:
                vif_val = float("inf")
            if not np.isfinite(vif_val):
                vif_val = float("inf")
            vifs[col] = vif_val

        max_col = max(vifs, key=vifs.get)
        max_vif = vifs[max_col]
        if max_vif <= threshold:
            break

        if max_col in anchor_set:
            non_anchor = [c for c in selected if c not in anchor_set]
            if not non_anchor:
                break
            max_col = max(non_anchor, key=lambda c: vifs.get(c, -np.inf))
            max_vif = vifs[max_col]
            if max_vif <= threshold:
                break

        selected.remove(max_col)
        dropped_log.append({"dropped": max_col, "vif": float(max_vif), "reason": "vif_pruning"})

    keep_extra = [c for c in train_df.columns if c not in feature_cols]
    keep_cols = keep_extra + selected
    logger.info(f"VIF pruning: kept {len(selected)}/{len(feature_cols)} features (threshold={threshold})")
    return train_df[keep_cols].copy(), val_df[keep_cols].copy(), test_df[keep_cols].copy(), selected, dropped_log


# ============================================================================
# TSFRESH (OPTIONAL, ML BASELINES)
# ============================================================================


def _prepare_tsfresh_long(
    df: pd.DataFrame,
    feature_cols: list[str],
    segment_col: str,
    time_col: str,
    max_rows_per_segment: int,
) -> pd.DataFrame:
    """Prepare long-format dataframe for tsfresh extraction."""
    available = _numeric_existing_columns(df, feature_cols)
    if not available:
        return pd.DataFrame()

    sampled = (
        df.sort_values([segment_col, time_col])
        .groupby(segment_col, group_keys=False)
        .head(max_rows_per_segment)
    )
    long_df = sampled[[segment_col, time_col] + available].melt(
        id_vars=[segment_col, time_col],
        value_vars=available,
        var_name="kind",
        value_name="value",
    )
    return long_df.dropna(subset=["value"])


def extract_tsfresh_segment_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    mode: str = "minimal",
    top_k: int = 20,
    segment_col: str = "segment_id",
    time_col: str = "timestamp",
    label_col: str = "label",
    n_segments_sample: int = 60,
    max_rows_per_segment: int = 800,
    n_jobs: int = -1,
    label_strategy: str = "any_fault",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict]:
    """
    Extract tsfresh segment-level features and merge them back per row by segment_id.

    Returns transformed train/val/test, list of added tsfresh columns, and metadata.
    """
    mode = (mode or "off").lower()
    if mode == "off":
        return train_df, val_df, test_df, [], {"mode": "off", "selected": 0}

    if segment_col not in train_df.columns or time_col not in train_df.columns:
        logger.warning("tsfresh skipped: missing segment or timestamp column")
        return train_df, val_df, test_df, [], {"mode": mode, "selected": 0, "skipped": "missing_columns"}

    from tsfresh import extract_features
    from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
    from tsfresh.feature_selection import select_features
    from tsfresh.utilities.dataframe_functions import impute

    fc_params = MinimalFCParameters() if mode == "minimal" else ComprehensiveFCParameters()
    effective_jobs = n_jobs
    if effective_jobs is None or effective_jobs <= 0:
        effective_jobs = max(1, os.cpu_count() or 1)

    sampled_segments = train_df[segment_col].drop_duplicates().head(n_segments_sample)
    train_sample = train_df[train_df[segment_col].isin(sampled_segments)].copy()
    long_train = _prepare_tsfresh_long(train_sample, feature_cols, segment_col, time_col, max_rows_per_segment)
    if long_train.empty:
        logger.warning("tsfresh skipped: no valid train long-format rows")
        return train_df, val_df, test_df, [], {"mode": mode, "selected": 0, "skipped": "empty_long_train"}

    train_feat = extract_features(
        long_train,
        column_id=segment_col,
        column_sort=time_col,
        column_kind="kind",
        column_value="value",
        default_fc_parameters=fc_params,
        disable_progressbar=True,
        n_jobs=effective_jobs,
        impute_function=impute,
    )

    if label_strategy == "any_fault":
        segment_y = train_sample.groupby(segment_col)[label_col].apply(lambda s: int((s != 0).any()))
    elif label_strategy == "majority_label":
        segment_y = train_sample.groupby(segment_col)[label_col].apply(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1]
        )
    elif label_strategy == "fault_fraction":
        segment_y = train_sample.groupby(segment_col)[label_col].apply(lambda s: float((s != 0).mean()))
    else:
        raise ValueError(f"Unsupported tsfresh label_strategy: {label_strategy}")

    segment_y = segment_y.reindex(train_feat.index).fillna(0)

    if segment_y.nunique() > 1:
        selected = select_features(train_feat, segment_y)
    else:
        selected = train_feat

    if selected.empty:
        selected = train_feat

    chosen = list(selected.columns[:top_k])
    if not chosen:
        return train_df, val_df, test_df, [], {"mode": mode, "selected": 0}

    chosen_prefixed = [f"tsfresh__{c}" for c in chosen]

    def _transform_subset(df: pd.DataFrame) -> pd.DataFrame:
        long_df = _prepare_tsfresh_long(df, feature_cols, segment_col, time_col, max_rows_per_segment)
        if long_df.empty:
            out = df.copy()
            for col in chosen_prefixed:
                out[col] = 0.0
            return out

        feat = extract_features(
            long_df,
            column_id=segment_col,
            column_sort=time_col,
            column_kind="kind",
            column_value="value",
            default_fc_parameters=fc_params,
            disable_progressbar=True,
            n_jobs=effective_jobs,
            impute_function=impute,
        )
        feat = feat.reindex(columns=chosen, fill_value=0.0)
        feat.columns = chosen_prefixed
        out = df.merge(feat, left_on=segment_col, right_index=True, how="left")
        out[chosen_prefixed] = out[chosen_prefixed].fillna(0.0)
        return out

    train_out = _transform_subset(train_df)
    val_out = _transform_subset(val_df)
    test_out = _transform_subset(test_df)

    meta = {
        "mode": mode,
        "selected": len(chosen_prefixed),
        "columns": chosen_prefixed,
        "label_strategy": label_strategy,
    }
    logger.info(f"tsfresh {mode}: selected {len(chosen_prefixed)} features")
    return train_out, val_out, test_out, chosen_prefixed, meta

