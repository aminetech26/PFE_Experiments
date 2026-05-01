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
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from loguru import logger

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


_PHYSICS_SUFFIXES = ("_norm", "_irr_residual")


def infer_base_feature_columns(
    df: pd.DataFrame,
    preferred: Iterable[str] = DEFAULT_BASE_FEATURE_COLUMNS,
) -> list[str]:
    """Infer base feature columns: known sensor signals + physics-derived columns.

    Physics-derived columns (_norm, _irr_residual) are created at preprocessing
    and are automatically included here so no separate detection flag is needed.
    """
    sensor_cols = [col for col in preferred if col in df.columns]
    physics_cols = [c for c in df.columns if any(c.endswith(s) for s in _PHYSICS_SUFFIXES)]
    return sorted(set(sensor_cols + physics_cols))


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


def _parse_int_list(values: Iterable[int] | str | None, default: list[int]) -> list[int]:
    if values is None:
        return default
    if isinstance(values, str):
        parsed = [v.strip() for v in values.split(",") if v.strip()]
        out = [int(v) for v in parsed]
    else:
        out = [int(v) for v in values]
    out = sorted({v for v in out if v > 0})
    return out or default


def _parse_str_list(values: Iterable[str] | str | None, default: list[str]) -> list[str]:
    if values is None:
        return default
    if isinstance(values, str):
        out = [v.strip() for v in values.split(",") if v.strip()]
    else:
        out = [str(v).strip() for v in values if str(v).strip()]
    return out or default


def add_rolling_statistics_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    segment_col: str = "segment_id",
    time_col: str = "timestamp",
    windows: Iterable[int] | str | None = None,
    stats: Iterable[str] | str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add causal segment-safe rolling statistics as row-aligned features."""
    out = df.copy()
    windows_parsed = _parse_int_list(windows, default=[5, 10, 30])
    stats_parsed = _parse_str_list(stats, default=["mean", "std", "min", "max"])
    supported = {"mean", "std", "min", "max", "median", "range", "skew", "kurtosis", "rms", "zcr"}
    stats_parsed = [s for s in stats_parsed if s in supported]
    if not stats_parsed:
        stats_parsed = ["mean", "std", "min", "max"]

    usable_cols = [
        c for c in feature_cols if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]
    if not usable_cols:
        return out, []

    if segment_col in out.columns and time_col in out.columns:
        original_index = out.index
        out = out.sort_values([segment_col, time_col])
    else:
        original_index = None

    added: list[str] = []
    generated_cols: dict[str, pd.Series] = {}
    grouped = out.groupby(segment_col, sort=False) if segment_col in out.columns else None
    for col in usable_cols:
        base_series = out[col]
        for w in windows_parsed:
            rolling_obj = (
                grouped[col].rolling(window=w, min_periods=1)
                if grouped is not None
                else base_series.rolling(window=w, min_periods=1)
            )

            stats_data: dict[str, pd.Series] = {}
            if "mean" in stats_parsed:
                stats_data["mean"] = rolling_obj.mean()
            if "std" in stats_parsed:
                stats_data["std"] = rolling_obj.std(ddof=0).fillna(0.0)
            if "min" in stats_parsed:
                stats_data["min"] = rolling_obj.min()
            if "max" in stats_parsed:
                stats_data["max"] = rolling_obj.max()
            if "median" in stats_parsed:
                stats_data["median"] = rolling_obj.median()
            if "range" in stats_parsed:
                max_s = stats_data["max"] if "max" in stats_data else rolling_obj.max()
                min_s = stats_data["min"] if "min" in stats_data else rolling_obj.min()
                stats_data["range"] = max_s - min_s
            if "skew" in stats_parsed:
                stats_data["skew"] = rolling_obj.skew().fillna(0.0)
            if "kurtosis" in stats_parsed:
                stats_data["kurtosis"] = rolling_obj.kurt().fillna(0.0)
            if "rms" in stats_parsed:
                rms_vals = rolling_obj.apply(lambda x: float(np.sqrt(np.mean(x**2))), raw=True)
                stats_data["rms"] = rms_vals.fillna(0.0)
            if "zcr" in stats_parsed:
                zcr_obj = (
                    grouped[col].rolling(window=w, min_periods=2)
                    if grouped is not None
                    else base_series.rolling(window=w, min_periods=2)
                )
                zcr_vals = zcr_obj.apply(
                    lambda x: float(np.sum(np.diff(np.sign(x)) != 0)) / max(len(x) - 1, 1),
                    raw=True,
                )
                stats_data["zcr"] = zcr_vals.fillna(0.0)

            for stat_name, values in stats_data.items():
                col_name = f"{col}_roll{w}_{stat_name}"
                if grouped is not None:
                    series = values.reset_index(level=0, drop=True)
                else:
                    series = values
                generated_cols[col_name] = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                added.append(col_name)

    if generated_cols:
        out = pd.concat([out, pd.DataFrame(generated_cols, index=out.index)], axis=1)

    if original_index is not None:
        out = out.loc[original_index]

    return out, added


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

    dt_seconds = df.groupby(segment_col)[time_col].diff().dt.total_seconds().replace(0, np.nan)

    median_dt = dt_seconds[dt_seconds > 0].median()
    if pd.isna(median_dt) or median_dt <= 0:
        median_dt = 1.0

    dt_seconds = dt_seconds.fillna(median_dt)
    dv = df.groupby(segment_col)[value_col].diff().fillna(0.0)
    return dv / dt_seconds


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-8) -> pd.Series:
    denom = denominator.astype(float).abs() + eps
    ratio = numerator.astype(float) / denom
    return ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)


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
      - enable_delta_temp
      - enable_dP_dt
      - enable_dV_dt
      - enable_dI_dt
      - enable_Vg_normalized
      - enable_power_imbalance
      - enable_current_imbalance
      - enable_voltage_imbalance
      - enable_string_share
      - enable_temp_power_correction
    """
    out = df.copy()
    flags = flags or {}

    # Keep deterministic temporal behavior for segment operations.
    if segment_col in out.columns and time_col in out.columns:
        original_index = out.index
        out = out.sort_values([segment_col, time_col])
    else:
        original_index = None

    if flags.get("enable_delta_temp", True) and {"TPV", "TA"}.issubset(out.columns):
        out["delta_temp"] = out["TPV"] - out["TA"]

    if flags.get("enable_dP_dt", True):
        if "Pg" in out.columns:
            out["dP_dt"] = _safe_groupby_diff_per_second(out, "Pg", time_col, segment_col)
        elif "pdc" in out.columns:
            out["dP_dt"] = _safe_groupby_diff_per_second(out, "pdc", time_col, segment_col)
        elif {"pdc1", "pdc2"}.issubset(out.columns):
            out["_pdc_total_tmp"] = out["pdc1"] + out["pdc2"]
            out["dP_dt"] = _safe_groupby_diff_per_second(
                out, "_pdc_total_tmp", time_col, segment_col
            )
            out = out.drop(columns=["_pdc_total_tmp"])

    if flags.get("enable_dV_dt", True):
        if "Vg" in out.columns:
            out["dV_dt"] = _safe_groupby_diff_per_second(out, "Vg", time_col, segment_col)
        elif {"vdc1", "vdc2"}.issubset(out.columns):
            out["_vdc_mean_tmp"] = 0.5 * (out["vdc1"] + out["vdc2"])
            out["dV_dt"] = _safe_groupby_diff_per_second(
                out, "_vdc_mean_tmp", time_col, segment_col
            )
            out = out.drop(columns=["_vdc_mean_tmp"])

    if flags.get("enable_dI_dt", True):
        if "Ig" in out.columns:
            out["dI_dt"] = _safe_groupby_diff_per_second(out, "Ig", time_col, segment_col)
        elif {"idc1", "idc2"}.issubset(out.columns):
            out["_idc_total_tmp"] = out["idc1"] + out["idc2"]
            out["dI_dt"] = _safe_groupby_diff_per_second(
                out, "_idc_total_tmp", time_col, segment_col
            )
            out = out.drop(columns=["_idc_total_tmp"])

    if flags.get("enable_Vg_normalized", True) and "Vg" in out.columns:
        if segment_col in out.columns:
            rolling_med = out.groupby(segment_col)["Vg"].transform(
                lambda s: s.rolling(window=vg_window, min_periods=1).median()
            )
        else:
            rolling_med = out["Vg"].rolling(window=vg_window, min_periods=1).median()
        out["Vg_normalized"] = out["Vg"] / rolling_med.replace(0, np.nan)
        out["Vg_normalized"] = out["Vg_normalized"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if flags.get("enable_power_imbalance", False) and {"pdc1", "pdc2"}.issubset(out.columns):
        total_power = out["pdc1"] + out["pdc2"]
        out["power_imbalance"] = _safe_ratio((out["pdc1"] - out["pdc2"]).abs(), total_power)

    if flags.get("enable_current_imbalance", False) and {"idc1", "idc2"}.issubset(out.columns):
        total_current = out["idc1"] + out["idc2"]
        out["current_imbalance"] = _safe_ratio((out["idc1"] - out["idc2"]).abs(), total_current)

    if flags.get("enable_voltage_imbalance", False) and {"vdc1", "vdc2"}.issubset(out.columns):
        total_voltage = out["vdc1"].abs() + out["vdc2"].abs()
        out["voltage_imbalance"] = _safe_ratio((out["vdc1"] - out["vdc2"]).abs(), total_voltage)

    if flags.get("enable_string_share", False) and {"pdc1", "pdc2", "idc1", "idc2"}.issubset(
        out.columns
    ):
        out["string1_power_share"] = _safe_ratio(out["pdc1"], out["pdc1"] + out["pdc2"])
        out["string1_current_share"] = _safe_ratio(out["idc1"], out["idc1"] + out["idc2"])

    if flags.get("enable_temp_power_correction", False) and {"pvt", "pdc", "irr"}.issubset(
        out.columns
    ):
        temp_ref_c = float(flags.get("temp_ref_c", 25.0))
        gamma_pct_per_c = float(flags.get("gamma_pmax_pct_per_c", -0.40))
        gamma_frac_per_c = gamma_pct_per_c / 100.0
        eps = float(flags.get("temp_power_eps", 1e-8))
        irr_floor = float(flags.get("irr_norm_floor", 1.0))

        out["temp_loss_pmax"] = gamma_frac_per_c * (out["pvt"] - temp_ref_c)
        correction_factor = (1.0 + out["temp_loss_pmax"]).replace(0, np.nan)
        out["pdc_temp_corrected"] = (out["pdc"] / correction_factor).replace(
            [np.inf, -np.inf], np.nan
        )
        out["pdc_temp_corrected"] = out["pdc_temp_corrected"].fillna(out["pdc"])
        irr_denom = out["irr"].clip(lower=irr_floor)
        out["pdc_temp_corrected_norm_irr"] = _safe_ratio(
            out["pdc_temp_corrected"], irr_denom, eps=eps
        )

    if original_index is not None:
        out = out.loc[original_index]

    return out


# ============================================================================
# SIGNAL FEATURES
# ============================================================================


def _estimate_wavelet_threshold(series: np.ndarray, wavelet: str = "db4", level: int = 4) -> float:
    """Estimate a universal wavelet threshold for one signal series."""
    import pywt

    coeffs = pywt.wavedec(series, wavelet, level=level)
    if len(coeffs) < 2 or len(series) <= 1:
        return 0.0
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    return float(np.sqrt(2 * np.log(len(series))) * sigma)


def wavelet_denoise_series(
    series: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
    threshold: float | None = None,
) -> np.ndarray:
    """Denoise a 1D series using wavelet soft-thresholding."""
    import pywt

    coeffs = pywt.wavedec(series, wavelet, level=level)
    if threshold is None:
        threshold = _estimate_wavelet_threshold(series, wavelet=wavelet, level=level)

    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs_thresh, wavelet)[: len(series)]


def add_wavelet_feature(
    df: pd.DataFrame,
    source_col: str = "Pg",
    segment_col: str = "segment_id",
    target_col: str = "Pg_wavelet",
    wavelet: str = "db4",
    level: int = 4,
    threshold_strategy: str = "per_segment",
) -> pd.DataFrame:
    """Add a denoised wavelet feature from source_col.

    threshold_strategy:
      - "per_segment": estimate threshold independently for each segment.
      - "global": estimate one threshold on full source column and reuse it.
    """
    if source_col not in df.columns:
        return df

    out = df.copy()
    strategy = str(threshold_strategy).lower()
    if strategy not in {"per_segment", "global"}:
        raise ValueError(f"Unsupported wavelet threshold strategy: {threshold_strategy}")

    if segment_col not in out.columns:
        values = out[source_col].to_numpy(dtype=np.float64)
        threshold = _estimate_wavelet_threshold(values, wavelet=wavelet, level=level)
        out[target_col] = wavelet_denoise_series(
            values, wavelet=wavelet, level=level, threshold=threshold
        )
        return out

    out[target_col] = 0.0
    global_threshold = None
    if strategy == "global":
        full_values = out[source_col].to_numpy(dtype=np.float64)
        global_threshold = _estimate_wavelet_threshold(full_values, wavelet=wavelet, level=level)

    for _, seg_df in out.groupby(segment_col, sort=False):
        values = seg_df[source_col].to_numpy(dtype=np.float64)
        threshold = (
            global_threshold
            if strategy == "global"
            else _estimate_wavelet_threshold(values, wavelet=wavelet, level=level)
        )
        den = wavelet_denoise_series(values, wavelet=wavelet, level=level, threshold=threshold)
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
    Output: (n_windows, n_features * 6)
    """
    n, w, f = x.shape
    features = []
    for i in range(f):
        ch = x[:, :, i]
        features.append(ch.mean(axis=1))
        features.append(ch.std(axis=1))
        features.append(ch.min(axis=1))
        features.append(ch.max(axis=1))
        features.append(np.sqrt((ch**2).mean(axis=1)))
        zcr = ((np.diff(np.sign(ch), axis=1) != 0).sum(axis=1)) / max(w - 1, 1)
        features.append(zcr)

    return np.column_stack(features).astype(np.float32)


def add_multiscale_window_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_sizes: list[int],
    step: int,
    label_col: str | None = None,
    segment_col: str = "segment_id",
    fs: float = 1.0,
    enable_wpd: bool = False,
    wpd_level: int = 3,
    wpd_wavelet: str = "db4",
    enable_ceemdan: bool = False,
    ceemdan_n_imfs: int = 3,
    ceemdan_n_ensemble: int = 20,
    spectral_min_fs: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Collapse a row-aligned DataFrame into fixed-size windows with multi-scale statistics
    and optional spectral features (WPD, CEEMDAN — Path B only).

    The largest value in window_sizes is the primary window (determines row count and stride).
    Each smaller scale takes the last `w` samples of the primary window — nested sub-windows.
    Spectral features are always computed on the primary window only.

    Spectral methods requiring a minimum sampling rate are skipped when fs < spectral_min_fs.
    Returns a new DataFrame with one row per window and the list of feature column names.
    """
    window_sizes_sorted = sorted(set(window_sizes), reverse=True)
    primary = window_sizes_sorted[0]

    usable_cols = [
        c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not usable_cols:
        return pd.DataFrame(), []

    _spectral_ok = fs >= spectral_min_fs
    if not _spectral_ok and any([enable_wpd, enable_ceemdan]):
        logger.warning(
            f"Sampling rate {fs} Hz < spectral_min_fs {spectral_min_fs} Hz — "
            "skipping all spectral features for this dataset."
        )
        enable_wpd = enable_ceemdan = False

    groups: list[tuple] = (
        list(df.groupby(segment_col, sort=False)) if segment_col in df.columns else [(None, df)]
    )

    scale_buffers: dict[int, list[np.ndarray]] = {w: [] for w in window_sizes_sorted}
    primary_buffer: list[np.ndarray] = []
    label_buffer: list = []

    for _, seg in groups:
        values = seg[usable_cols].to_numpy(dtype=np.float32)
        labels = seg[label_col].to_numpy() if label_col and label_col in seg.columns else None
        n = len(values)

        if n < primary:
            continue

        starts = range(0, n - primary + 1, step)
        primary_windows = np.stack([values[s : s + primary] for s in starts])

        for w in window_sizes_sorted:
            scale_buffers[w].append(primary_windows[:, primary - w :, :])

        if any([enable_wpd, enable_ceemdan]):
            primary_buffer.append(primary_windows)

        if labels is not None:
            for s in starts:
                label_buffer.append(pd.Series(labels[s : s + primary]).mode().iloc[0])

    total_windows = sum(len(b) for b in scale_buffers[primary])
    if total_windows == 0:
        logger.warning("No windows produced — check window_size vs segment lengths.")
        return pd.DataFrame(), []

    stat_names = ["mean", "std", "min", "max", "rms", "zcr"]
    parts: list[np.ndarray] = []
    col_names: list[str] = []

    for w in window_sizes_sorted:
        x_w = np.concatenate(scale_buffers[w], axis=0)
        stats_w = extract_window_statistics(x_w)
        parts.append(stats_w)
        for feat in usable_cols:
            for stat in stat_names:
                col_names.append(f"{feat}_w{w}_{stat}")

    if primary_buffer:
        x_primary = np.concatenate(primary_buffer, axis=0)

        if enable_wpd:
            wpd_feats = extract_wpd_features(x_primary, wavelet=wpd_wavelet, level=wpd_level)
            parts.append(wpd_feats)
            n_bands = 2**wpd_level
            for feat in usable_cols:
                for b in range(n_bands):
                    col_names.append(f"{feat}_wpd_band{b}")

        if enable_ceemdan:
            logger.info(
                f"CEEMDAN: {total_windows} windows × {len(usable_cols)} features "
                f"× {ceemdan_n_ensemble} ensembles — this may be slow."
            )
            ceemdan_feats = extract_ceemdan_features(
                x_primary, n_imfs=ceemdan_n_imfs, n_ensemble=ceemdan_n_ensemble
            )
            parts.append(ceemdan_feats)
            for feat in usable_cols:
                for k in range(ceemdan_n_imfs):
                    col_names.append(f"{feat}_ceemdan_imf{k}")

    result_arr = np.hstack(parts).astype(np.float32)
    result_df = pd.DataFrame(result_arr, columns=col_names)

    if label_col and label_buffer:
        result_df[label_col] = label_buffer

    return result_df, col_names


def extract_wpd_features(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> np.ndarray:
    """
    Wavelet Packet Decomposition subband energies per window and channel.

    Input:  (n_windows, window_size, n_features)
    Output: (n_windows, n_features * 2**level)

    Each subband's feature is its energy fraction: sum(coeff²) / sum(signal²).
    Requires PyWavelets (pywt).
    """
    try:
        import pywt
    except ImportError:
        logger.error(
            "PyWavelets not installed — skipping WPD features. Install with: pip install PyWavelets"
        )
        n_wins, _, n_feats = x.shape
        return np.zeros((n_wins, n_feats * (2**level)), dtype=np.float32)

    n_wins, _, n_feats = x.shape
    n_bands = 2**level
    results = []
    for i in range(n_feats):
        ch = x[:, :, i]
        band_energies = np.zeros((n_wins, n_bands), dtype=np.float32)
        for w in range(n_wins):
            signal = ch[w].astype(np.float64)
            total_energy = float(np.sum(signal**2)) + 1e-10
            wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode="symmetric", maxlevel=level)
            nodes = [node.path for node in wp.get_level(level, "freq")]
            for b, path in enumerate(nodes):
                coeffs = wp[path].data
                band_energies[w, b] = float(np.sum(coeffs**2)) / total_energy
        results.append(band_energies)
    return np.hstack(results).astype(np.float32)


def extract_ceemdan_features(
    x: np.ndarray,
    n_imfs: int = 3,
    n_ensemble: int = 20,
) -> np.ndarray:
    """
    CEEMDAN IMF energy ratios per window and channel.

    Input:  (n_windows, window_size, n_features)
    Output: (n_windows, n_features * n_imfs)

    For each window and channel, CEEMDAN decomposes the signal into IMFs.
    Feature = energy ratio of each IMF (IMF energy / total signal energy).
    If fewer than n_imfs IMFs are produced, remaining slots are zero-padded.
    Requires emd-signal package (already in project deps).
    """
    try:
        from PyEMD import CEEMDAN
    except ImportError as exc:
        raise ImportError(
            "PyEMD is required for CEEMDAN features but is not available. "
            "Install dependency 'EMD-signal' / 'PyEMD' before running CEEMDAN profiles."
        ) from exc

    n_wins, _, n_feats = x.shape
    results = []
    ceemdan = CEEMDAN(trials=n_ensemble)
    min_energy = 1e-12
    min_std = 1e-8
    skipped_flat = 0
    skipped_nonfinite = 0
    failed_decomp = 0

    for i in range(n_feats):
        ch = x[:, :, i]
        imf_energies = np.zeros((n_wins, n_imfs), dtype=np.float32)
        for w in range(n_wins):
            signal = np.nan_to_num(ch[w].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            if not np.all(np.isfinite(signal)):
                skipped_nonfinite += 1
                continue

            total_energy = float(np.sum(signal**2))
            if total_energy <= min_energy or float(np.std(signal)) <= min_std:
                skipped_flat += 1
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    imfs = np.asarray(ceemdan.ceemdan(signal, max_imf=n_imfs), dtype=np.float64)

                if imfs.ndim != 2 or imfs.shape[0] == 0:
                    continue

                imfs = np.nan_to_num(imfs, nan=0.0, posinf=0.0, neginf=0.0)
                usable_imfs = imfs[:-1] if imfs.shape[0] > 1 else imfs
                for k in range(min(n_imfs, usable_imfs.shape[0])):
                    imf_energies[w, k] = float(np.sum(usable_imfs[k] ** 2)) / total_energy
            except Exception as e:
                failed_decomp += 1
                logger.debug(f"CEEMDAN failed for window {w}, feature {i}: {e}")
        results.append(imf_energies)

    total_pairs = n_wins * n_feats
    if skipped_flat or skipped_nonfinite or failed_decomp:
        logger.info(
            "CEEMDAN guards: skipped_flat={} skipped_nonfinite={} failed_decomp={} total_pairs={}",
            skipped_flat,
            skipped_nonfinite,
            failed_decomp,
            total_pairs,
        )

    return np.hstack(results).astype(np.float32)


# ============================================================================
# CORRELATION / COLLINEARITY PRUNING
# ============================================================================


def _numeric_existing_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def apply_hygiene_pruning(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    near_constant_std: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[dict]]:
    """Light pre-selection hygiene: constants, exact duplicates, deterministic aliases."""
    cols = _numeric_existing_columns(train_df, feature_cols)
    if not cols:
        return train_df, val_df, test_df, cols, []

    dropped_log: list[dict] = []
    keep: list[str] = []

    x = train_df[cols].replace([np.inf, -np.inf], np.nan)

    # 1) Near-constant columns.
    near_constant = [
        c
        for c in cols
        if np.isclose(float(x[c].std(skipna=True) or 0.0), 0.0, atol=near_constant_std)
    ]
    for c in near_constant:
        dropped_log.append({"dropped": c, "reason": "near_constant"})

    candidate = [c for c in cols if c not in set(near_constant)]
    if not candidate:
        keep_extra = [c for c in train_df.columns if c not in feature_cols]
        keep_cols = keep_extra
        return (
            train_df[keep_cols].copy(),
            val_df[keep_cols].copy(),
            test_df[keep_cols].copy(),
            [],
            dropped_log,
        )

    # 2) Exact duplicate columns by hash fingerprint.
    seen_hash: dict[int, str] = {}
    for c in candidate:
        sig = pd.util.hash_pandas_object(x[c].fillna(-999999.123456), index=False).sum()
        if sig in seen_hash:
            dropped_log.append(
                {
                    "dropped": c,
                    "paired_with": seen_hash[sig],
                    "reason": "exact_duplicate_column",
                }
            )
            continue
        seen_hash[sig] = c
        keep.append(c)

    # 3) Deterministic Costa alias: pdc == pdc1+pdc2 (if exact).
    keep_set = set(keep)
    if {"pdc", "pdc1", "pdc2"}.issubset(keep_set):
        lhs = train_df["pdc"].to_numpy(dtype=np.float64)
        rhs = (train_df["pdc1"] + train_df["pdc2"]).to_numpy(dtype=np.float64)
        if np.allclose(lhs, rhs, equal_nan=True, atol=1e-9, rtol=1e-9):
            keep = [c for c in keep if c != "pdc"]
            dropped_log.append(
                {
                    "dropped": "pdc",
                    "paired_with": "pdc1+pdc2",
                    "reason": "deterministic_alias",
                }
            )

    keep_extra = [c for c in train_df.columns if c not in feature_cols]
    keep_cols = keep_extra + keep
    logger.info(
        f"Hygiene pruning: kept {len(keep)}/{len(cols)} features (dropped={len(dropped_log)})"
    )
    return (
        train_df[keep_cols].copy(),
        val_df[keep_cols].copy(),
        test_df[keep_cols].copy(),
        keep,
        dropped_log,
    )


def apply_mrmr_selection(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[dict]]:
    """Greedy mRMR (MID criterion): maximize relevance - redundancy."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    cols = _numeric_existing_columns(train_df, feature_cols)
    if not cols:
        return train_df, val_df, test_df, cols, []
    if label_col not in train_df.columns:
        logger.warning("mRMR skipped: missing label column")
        return train_df, val_df, test_df, cols, []

    if k <= 0 or k >= len(cols):
        return train_df, val_df, test_df, cols, []

    x_df = train_df[cols].replace([np.inf, -np.inf], np.nan)
    x_df = x_df.fillna(x_df.median(numeric_only=True))
    x = x_df.to_numpy(dtype=np.float64)

    y_raw = train_df[label_col]
    y_codes, _ = pd.factorize(y_raw, sort=True)
    if len(np.unique(y_codes)) <= 1:
        logger.warning("mRMR skipped: single-class train target")
        return train_df, val_df, test_df, cols, []

    relevance_arr = mutual_info_classif(x, y_codes, random_state=42)
    relevance = {c: float(relevance_arr[i]) for i, c in enumerate(cols)}

    selected: list[str] = [max(cols, key=lambda c: relevance.get(c, -np.inf))]
    remaining = [c for c in cols if c not in selected]
    redundancy_cache: dict[tuple[str, str], float] = {}

    def _feature_vector(col: str) -> np.ndarray:
        return x_df[col].to_numpy(dtype=np.float64).reshape(-1, 1)

    def _pair_redundancy(a: str, b: str) -> float:
        key = tuple(sorted((a, b)))
        if key in redundancy_cache:
            return redundancy_cache[key]
        xb = x_df[b].to_numpy(dtype=np.float64)
        val = float(mutual_info_regression(_feature_vector(a), xb, random_state=42)[0])
        if not np.isfinite(val):
            val = 0.0
        redundancy_cache[key] = val
        return val

    while remaining and len(selected) < k:
        best_col = None
        best_score = -np.inf
        for c in remaining:
            red = np.mean([_pair_redundancy(c, s) for s in selected]) if selected else 0.0
            score = relevance.get(c, 0.0) - red
            if score > best_score:
                best_score = score
                best_col = c
        if best_col is None:
            break
        selected.append(best_col)
        remaining.remove(best_col)

    dropped_log = [
        {"dropped": c, "reason": "mrmr_not_selected", "relevance": relevance.get(c, 0.0)}
        for c in cols
        if c not in selected
    ]

    keep_extra = [c for c in train_df.columns if c not in feature_cols]
    keep_cols = keep_extra + selected
    logger.info(f"mRMR: kept {len(selected)}/{len(cols)} features (k={k})")
    return (
        train_df[keep_cols].copy(),
        val_df[keep_cols].copy(),
        test_df[keep_cols].copy(),
        selected,
        dropped_log,
    )


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
            dropped_log.append(
                {
                    "dropped": drop_col,
                    "paired_with": b if drop_col == a else a,
                    "rho": float(rho),
                    "reason": reason,
                }
            )

    selected_cols = [c for c in cols if c not in dropped]
    keep_extra = [c for c in train_df.columns if c not in feature_cols]
    keep_cols = keep_extra + selected_cols

    logger.info(
        f"Correlation pruning: kept {len(selected_cols)}/{len(cols)} features (threshold={threshold})"
    )
    return (
        train_df[keep_cols].copy(),
        val_df[keep_cols].copy(),
        test_df[keep_cols].copy(),
        selected_cols,
        dropped_log,
    )


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
    logger.info(
        f"VIF pruning: kept {len(selected)}/{len(feature_cols)} features (threshold={threshold})"
    )
    return (
        train_df[keep_cols].copy(),
        val_df[keep_cols].copy(),
        test_df[keep_cols].copy(),
        selected,
        dropped_log,
    )


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
        return (
            train_df,
            val_df,
            test_df,
            [],
            {"mode": mode, "selected": 0, "skipped": "missing_columns"},
        )

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
    long_train = _prepare_tsfresh_long(
        train_sample, feature_cols, segment_col, time_col, max_rows_per_segment
    )
    if long_train.empty:
        logger.warning("tsfresh skipped: no valid train long-format rows")
        return (
            train_df,
            val_df,
            test_df,
            [],
            {"mode": mode, "selected": 0, "skipped": "empty_long_train"},
        )

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
        segment_y = train_sample.groupby(segment_col)[label_col].apply(
            lambda s: int((s != 0).any())
        )
    elif label_strategy == "majority_label":
        segment_y = train_sample.groupby(segment_col)[label_col].apply(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1]
        )
    elif label_strategy == "fault_fraction":
        segment_y = train_sample.groupby(segment_col)[label_col].apply(
            lambda s: float((s != 0).mean())
        )
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
        long_df = _prepare_tsfresh_long(
            df, feature_cols, segment_col, time_col, max_rows_per_segment
        )
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
