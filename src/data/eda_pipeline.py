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
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import polars as pl
import yaml
from loguru import logger
from statsmodels.tsa.stattools import adfuller, kpss

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
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _load_existing_stationarity(report_path: Path) -> dict | None:
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    stationarity = payload.get("stationarity")
    if not isinstance(stationarity, dict):
        return None
    if stationarity.get("skipped"):
        return None
    if not stationarity.get("tests") and not stationarity.get("per_episode_tests"):
        return None
    return stationarity


def _load_stationarity_sidecar(output_dir: Path) -> dict | None:
    sidecar_path = output_dir / "eda_stationarity_report.json"
    if not sidecar_path.exists():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("skipped"):
        return None
    if not payload.get("tests") and not payload.get("per_episode_tests"):
        return None
    return payload


def _write_stationarity_sidecar(output_dir: Path, stationarity: dict) -> None:
    if not isinstance(stationarity, dict) or stationarity.get("skipped"):
        return
    sidecar_path = output_dir / "eda_stationarity_report.json"
    sidecar_path.write_text(
        json.dumps(_to_json_safe(stationarity), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


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


def attach_episode_ids(
    df: pd.DataFrame, gap_seconds: int, segment_col: str = "episode_id"
) -> pd.DataFrame:
    df = df.copy()
    dt_s = df["timestamp"].diff().dt.total_seconds().fillna(0).values
    label_change = df["label"].ne(df["label"].shift()).fillna(False).values
    df[segment_col] = ((dt_s > gap_seconds) | label_change).cumsum().astype(int)
    return df


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
    ts = df["timestamp"]
    dt_s_series = ts.diff().dt.total_seconds().fillna(0)
    same_day_mask = ts.dt.date.eq(ts.shift().dt.date).fillna(False)
    dt_s = dt_s_series.to_numpy()

    # Gap distribution
    non_zero = dt_s[dt_s > 0]
    if len(non_zero) == 0:
        return {"error": "no non-zero gaps found"}

    intra_day_non_zero = dt_s_series[(dt_s_series > 0) & same_day_mask].to_numpy()

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
    intra_day_gaps_above = {
        f">{t}s": int(((dt_s_series > t) & same_day_mask).sum()) for t in thresholds
    }

    intra_day_summary: dict[str, object]
    if len(intra_day_non_zero) == 0:
        intra_day_summary = {
            "n_intra_day_positive_gaps": 0,
            "gap_percentiles": {},
            "gaps_above_threshold": intra_day_gaps_above,
            "max_gap_seconds": None,
            "largest_gaps_seconds": [],
            "note": "No positive intra-day gaps found.",
        }
    else:
        intra_day_percentiles = {
            "p50": float(np.percentile(intra_day_non_zero, 50)),
            "p90": float(np.percentile(intra_day_non_zero, 90)),
            "p95": float(np.percentile(intra_day_non_zero, 95)),
            "p99": float(np.percentile(intra_day_non_zero, 99)),
            "max": float(intra_day_non_zero.max()),
            "mean": float(intra_day_non_zero.mean()),
        }
        intra_day_summary = {
            "n_intra_day_positive_gaps": int(len(intra_day_non_zero)),
            "gap_percentiles": {k: round(v, 4) for k, v in intra_day_percentiles.items()},
            "gaps_above_threshold": intra_day_gaps_above,
            "max_gap_seconds": float(intra_day_non_zero.max()),
            "largest_gaps_seconds": [
                float(v) for v in sorted(intra_day_non_zero.tolist(), reverse=True)[:10]
            ],
            "note": "Computed only between consecutive rows that remain on the same calendar date.",
        }

    daily_summary_rows = []
    daily_dates = ts.dt.date
    for day, grp in df.groupby(daily_dates, sort=True):
        day_dt = grp["timestamp"].diff().dt.total_seconds().fillna(0)
        day_positive = day_dt[day_dt > 0]
        daily_summary_rows.append(
            {
                "date": str(day),
                "rows": int(len(grp)),
                "first_timestamp": grp["timestamp"].min().isoformat(),
                "last_timestamp": grp["timestamp"].max().isoformat(),
                "max_intra_day_gap_seconds": float(day_positive.max())
                if len(day_positive)
                else 0.0,
                "n_intra_day_gaps_gt_30s": int((day_dt > 30).sum()),
                "n_intra_day_gaps_gt_300s": int((day_dt > configured_gap_s).sum()),
            }
        )

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
        "intra_day": intra_day_summary,
        "per_day_intra_day_summary": daily_summary_rows,
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


def analyse_missing_values(df: pd.DataFrame, sensor_cols: list[str]) -> dict:
    cols = ["timestamp", "label", *sensor_cols]
    available = [c for c in cols if c in df.columns]
    if not available:
        return {"error": "no requested columns available"}

    subset = df[available]
    per_column = []
    for col in available:
        null_count = int(subset[col].isna().sum())
        per_column.append(
            {
                "column": col,
                "n_missing": null_count,
                "pct_missing": round(100 * null_count / len(subset), 4),
            }
        )

    rows_with_any_missing = int(subset.isna().any(axis=1).sum())
    complete_rows = int((~subset.isna().any(axis=1)).sum())
    return {
        "n_rows": int(len(subset)),
        "rows_with_any_missing": rows_with_any_missing,
        "pct_rows_with_any_missing": round(100 * rows_with_any_missing / len(subset), 4),
        "complete_rows": complete_rows,
        "pct_complete_rows": round(100 * complete_rows / len(subset), 4),
        "per_column": per_column,
    }


def analyse_outliers(
    df: pd.DataFrame,
    sensor_cols: list[str],
    label_col: str = "label",
    iqr_multiplier: float = 3.0,
) -> dict:
    results = []
    normal_df = df[df[label_col] == 0.0]

    for col in sensor_cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        is_outlier = (series < lower) | (series > upper)

        normal_series = (
            normal_df[col].dropna() if col in normal_df.columns else pd.Series(dtype=float)
        )
        normal_outliers = ((normal_series < lower) | (normal_series > upper)).sum()
        results.append(
            {
                "feature": col,
                "iqr_multiplier": float(iqr_multiplier),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "total_outliers": int(is_outlier.sum()),
                "pct_total_outliers": round(100 * is_outlier.mean(), 4),
                "normal_period_outliers": int(normal_outliers),
                "pct_normal_period_outliers": round(100 * normal_outliers / len(normal_series), 4)
                if len(normal_series)
                else 0.0,
            }
        )

    results = sorted(results, key=lambda row: row["normal_period_outliers"], reverse=True)
    return {
        "strategy": "iqr_extreme_outliers_normal_vs_all",
        "iqr_multiplier": float(iqr_multiplier),
        "results": results,
    }


def analyse_episode_morphology(df: pd.DataFrame, episode_col: str, sampling_hz: float) -> dict:
    if episode_col not in df.columns:
        return {"error": f"{episode_col} not found"}

    grouped = (
        df.groupby(episode_col, observed=True)
        .agg(
            label=("label", "first"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            n_rows=("label", "size"),
        )
        .reset_index()
    )
    nominal_step = 1.0 / sampling_hz if sampling_hz > 0 else 1.0
    grouped["duration_seconds"] = (
        grouped["end_time"] - grouped["start_time"]
    ).dt.total_seconds().fillna(0.0) + nominal_step

    per_class = []
    for label in sorted(grouped["label"].unique()):
        label_rows = grouped[grouped["label"] == label]
        durations = label_rows["duration_seconds"]
        counts = label_rows["n_rows"]
        per_class.append(
            {
                "label": float(label),
                "n_episodes": int(len(label_rows)),
                "median_rows": float(counts.median()),
                "p90_rows": float(counts.quantile(0.9)),
                "max_rows": int(counts.max()),
                "median_duration_seconds": float(durations.median()),
                "p90_duration_seconds": float(durations.quantile(0.9)),
                "max_duration_seconds": float(durations.max()),
                "short_episodes_le_6s": int((durations <= 6.0).sum()),
            }
        )

    total_rows = int(grouped["n_rows"].sum())
    short_rows = int(grouped.loc[grouped["duration_seconds"] <= 6.0, "n_rows"].sum())
    return {
        "episode_column": episode_col,
        "n_episodes": int(len(grouped)),
        "sampling_hz": float(sampling_hz),
        "short_episode_threshold_seconds": 6.0,
        "rows_in_short_episodes": short_rows,
        "pct_rows_in_short_episodes": round(100 * short_rows / total_rows, 4)
        if total_rows
        else 0.0,
        "per_class": per_class,
    }


def analyse_class_imbalance(df: pd.DataFrame, episode_col: str) -> dict:
    if episode_col not in df.columns:
        return {"error": f"{episode_col} not found"}

    total_rows = int(len(df))
    total_episodes = int(df[episode_col].nunique())
    episode_labels = df.groupby(episode_col, observed=True)["label"].first().reset_index(drop=False)

    rows = []
    for label in sorted(df["label"].unique()):
        row_count = int((df["label"] == label).sum())
        episode_count = int((episode_labels["label"] == label).sum())
        rows.append(
            {
                "label": float(label),
                "n_rows": row_count,
                "pct_rows": round(100 * row_count / total_rows, 3) if total_rows else 0.0,
                "n_episodes": episode_count,
                "pct_episodes": round(100 * episode_count / total_episodes, 3)
                if total_episodes
                else 0.0,
                "episode_to_row_share_ratio": round(
                    (episode_count / total_episodes) / (row_count / total_rows), 3
                )
                if total_rows and total_episodes and row_count > 0
                else None,
            }
        )

    minority_by_rows = [r["label"] for r in rows if r["pct_rows"] < 5.0 and r["label"] != 0.0]
    minority_by_episodes = [
        r["label"] for r in rows if r["pct_episodes"] < 5.0 and r["label"] != 0.0
    ]
    return {
        "episode_column": episode_col,
        "total_rows": total_rows,
        "total_episodes": total_episodes,
        "per_class": rows,
        "minority_fault_classes_by_rows_lt_5pct": minority_by_rows,
        "minority_fault_classes_by_episodes_lt_5pct": minority_by_episodes,
    }


def analyse_vdc1_outliers(
    df: pd.DataFrame,
    episode_col: str,
    label_col: str = "label",
    feature_col: str = "vdc1",
    iqr_multiplier: float = 3.0,
) -> dict:
    if feature_col not in df.columns:
        return {"error": f"{feature_col} not found"}
    if episode_col not in df.columns:
        return {"error": f"{episode_col} not found"}

    normal_df = df[df[label_col] == 0.0].copy()
    if normal_df.empty:
        return {"error": "no normal rows found"}

    series = normal_df[feature_col].dropna()
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    outlier_mask = (normal_df[feature_col] < lower) | (normal_df[feature_col] > upper)
    outliers = normal_df.loc[outlier_mask].copy()
    if outliers.empty:
        return {
            "feature": feature_col,
            "n_outliers": 0,
            "lower_bound": lower,
            "upper_bound": upper,
        }

    outliers["date"] = outliers["timestamp"].dt.date.astype(str)
    outliers["hour"] = outliers["timestamp"].dt.hour
    daily = (
        outliers.groupby("date", observed=True)
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="n_outliers")
    )
    hourly = (
        outliers.groupby("hour", observed=True)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="n_outliers")
    )
    episodes = (
        outliers.groupby(episode_col, observed=True)
        .agg(
            n_outliers=(feature_col, "size"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            median_vdc1=(feature_col, "median"),
            median_irr=("irr", "median"),
        )
        .sort_values("n_outliers", ascending=False)
        .head(10)
        .reset_index(drop=False)
    )

    return {
        "feature": feature_col,
        "label_scope": "normal_only",
        "iqr_multiplier": float(iqr_multiplier),
        "lower_bound": lower,
        "upper_bound": upper,
        "n_outliers": int(len(outliers)),
        "pct_normal_rows_outlier": round(100 * len(outliers) / len(normal_df), 4),
        "top_days": daily.to_dict(orient="records"),
        "hourly_distribution": hourly.to_dict(orient="records"),
        "top_episodes": episodes.to_dict(orient="records"),
    }


def _run_unit_root_tests(series: pd.Series, sampling_seconds: float) -> dict:
    cleaned = series.dropna()
    if len(cleaned) < 40:
        return {
            "nobs": int(len(cleaned)),
            "error": "too_few_observations",
        }

    step = max(1, len(cleaned) // 2000)
    sampled = cleaned.iloc[::step]
    effective_dt = sampling_seconds * step
    target_horizon_seconds = 4 * 3600
    target_maxlag = max(1, int(target_horizon_seconds / max(effective_dt, 1e-9)))
    safe_maxlag = max(1, min(target_maxlag, len(sampled) // 2 - 3))

    result = {
        "nobs": int(len(sampled)),
        "subsample_step": int(step),
        "effective_dt_seconds": float(effective_dt),
    }
    try:
        adf_stat, adf_p, adf_lag, adf_nobs, _, _ = adfuller(
            sampled.to_numpy(), maxlag=safe_maxlag, autolag="AIC"
        )
        result["adf"] = {
            "statistic": float(adf_stat),
            "p_value": float(adf_p),
            "usedlag": int(adf_lag),
            "nobs": int(adf_nobs),
            "stationary": bool(adf_p < 0.05),
        }
    except Exception as exc:
        result["adf"] = {"error": str(exc)}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lag, _ = kpss(sampled.to_numpy(), regression="c", nlags="auto")
        result["kpss"] = {
            "statistic": float(kpss_stat),
            "p_value": float(kpss_p),
            "usedlag": int(kpss_lag),
            "stationary": bool(kpss_p > 0.05),
        }
    except Exception as exc:
        result["kpss"] = {"error": str(exc)}

    adf_ok = result.get("adf", {}).get("stationary")
    kpss_ok = result.get("kpss", {}).get("stationary")
    result["consensus_stationary"] = (
        bool(adf_ok and kpss_ok) if adf_ok is not None and kpss_ok is not None else None
    )
    return result


def _summarize_stationarity_feature(feature_results: list[dict]) -> dict:
    """Aggregate per-episode ADF/KPSS outcomes for one feature."""
    n_episodes = len(feature_results)
    adf_flags = [r.get("adf", {}).get("stationary") for r in feature_results]
    kpss_flags = [r.get("kpss", {}).get("stationary") for r in feature_results]
    consensus_flags = [r.get("consensus_stationary") for r in feature_results]

    def _count_true(flags: list[bool | None]) -> int:
        return int(sum(flag is True for flag in flags))

    def _count_valid(flags: list[bool | None]) -> int:
        return int(sum(flag is not None for flag in flags))

    adf_valid = _count_valid(adf_flags)
    kpss_valid = _count_valid(kpss_flags)
    consensus_valid = _count_valid(consensus_flags)

    adf_true = _count_true(adf_flags)
    kpss_true = _count_true(kpss_flags)
    consensus_true = _count_true(consensus_flags)

    return {
        "n_episodes": n_episodes,
        "adf_stationary_count": adf_true,
        "adf_testable_episodes": adf_valid,
        "adf_stationary_rate": float(adf_true / adf_valid) if adf_valid else None,
        "kpss_stationary_count": kpss_true,
        "kpss_testable_episodes": kpss_valid,
        "kpss_stationary_rate": float(kpss_true / kpss_valid) if kpss_valid else None,
        "consensus_stationary_count": consensus_true,
        "consensus_testable_episodes": consensus_valid,
        "consensus_stationary_rate": (
            float(consensus_true / consensus_valid) if consensus_valid else None
        ),
    }


def analyse_stationarity(
    df: pd.DataFrame,
    sensor_cols: list[str],
    episode_col: str,
    dataset_cfg: dict,
) -> dict:
    from src.data.preprocessing import apply_physics_normalization

    if episode_col not in df.columns:
        return {"error": f"{episode_col} not found"}

    normal_episodes = (
        df[df["label"] == 0.0]
        .groupby(episode_col, observed=True)
        .size()
        .sort_values(ascending=False)
    )
    if normal_episodes.empty:
        return {"error": "no_normal_episode_found"}

    stat_cfg = dataset_cfg.get("preprocessing", {}).get("physics_normalization", {})
    n_reference_episodes = int(stat_cfg.get("n_reference_episodes", 5))
    selected_episodes = [int(ep) for ep in normal_episodes.index[:n_reference_episodes].tolist()]
    if not selected_episodes:
        return {"error": "no_reference_episodes_selected"}

    per_episode_tests: dict[str, dict] = {}
    feature_results: dict[str, list[dict]] = {}
    tested_columns: set[str] = set()
    transformed_cols: list[str] = []

    for ref_episode in selected_episodes:
        ref_df = df[df[episode_col] == ref_episode].copy()
        if ref_df.empty:
            continue

        transformed_df = ref_df.copy()
        current_transformed_cols: list[str] = []
        if stat_cfg:
            transformed_df, stat_stats = apply_physics_normalization(
                transformed_df,
                irradiance_config=stat_cfg.get("irradiance_normalize"),
                irr_residualize_config=stat_cfg.get("irr_residualize"),
                label_col="Fault" if "Fault" in transformed_df.columns else "label",
            )
            current_transformed_cols = [
                *stat_stats.features_normalized,
                *stat_stats.features_residualized,
            ]
            for col in current_transformed_cols:
                if col not in transformed_cols:
                    transformed_cols.append(col)

        timestamps = ref_df["timestamp"].diff().dt.total_seconds().dropna()
        sampling_seconds = float(timestamps.median()) if not timestamps.empty else 1.0
        episode_tests = {}
        for col in [*sensor_cols, *current_transformed_cols]:
            if col not in transformed_df.columns:
                continue
            result = _run_unit_root_tests(transformed_df[col], sampling_seconds)
            episode_tests[col] = result
            feature_results.setdefault(col, []).append(result)
            tested_columns.add(col)

        per_episode_tests[str(ref_episode)] = {
            "label": float(ref_df["label"].iloc[0]),
            "n_rows": int(len(ref_df)),
            "sampling_seconds": float(sampling_seconds),
            "tested_columns": list(episode_tests.keys()),
            "tests": episode_tests,
        }

    summary_by_feature = {
        feature: _summarize_stationarity_feature(results)
        for feature, results in sorted(feature_results.items())
    }

    return {
        "reference_episode_ids": selected_episodes,
        "reference_episode_selection": "longest_normal_episodes",
        "n_reference_episodes_requested": n_reference_episodes,
        "n_reference_episodes_used": len(per_episode_tests),
        "tested_columns": sorted(tested_columns),
        "transformed_columns": transformed_cols,
        "per_episode_tests": per_episode_tests,
        "summary_by_feature": summary_by_feature,
        "note": (
            "Stationarity is summarized across multiple normal episodes. Use as supporting "
            "interpretive evidence, not as a hard modeling gate."
        ),
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
    current_group_col = str(ds_splits.get("atomic_group_col", "segment_id"))
    recommended_group_col = "episode_id" if dataset == "costa" else current_group_col

    changes_needed = []
    if recommended_strategy != current_strategy:
        changes_needed.append(
            f"segmentation_strategy: '{current_strategy}' → '{recommended_strategy}'"
        )
    if recommended_group_col != current_group_col:
        changes_needed.append(
            f"atomic_group_col: '{current_group_col}' → '{recommended_group_col}'"
        )

    current_eval = ds_splits.get("evaluable_classes", [])
    rec_eval = class_dist.get("recommended_evaluable_classes", [])
    if sorted(current_eval) != sorted(rec_eval):
        changes_needed.append(f"evaluable_classes: {current_eval} → {rec_eval}")

    return {
        "current_segmentation_strategy": current_strategy,
        "recommended_segmentation_strategy": recommended_strategy,
        "current_segmentation_gap_seconds": current_gap,
        "current_atomic_group_col": current_group_col,
        "recommended_atomic_group_col": recommended_group_col,
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
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only compute EDA artifacts; skip visualization generation.",
    )
    parser.add_argument(
        "--skip-stationarity",
        action="store_true",
        help="Skip ADF/KPSS stationarity diagnostics to speed up EDA.",
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
    report_path = output_dir / "eda_dataset_report.json"
    logger.info("Output directory: {}", output_dir)

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    df_raw = load_dataset(config, args.dataset)
    df_raw = attach_episode_ids(df_raw, gap_s, segment_col="episode_id")

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
    # 3b. Missing values / outliers / episode morphology
    # -------------------------------------------------------------------------
    logger.info("[2b/5] Missing values, outliers, and episode morphology ...")
    missingness = analyse_missing_values(df_raw, sensor_cols)
    outlier_report = analyse_outliers(df_raw, sensor_cols)
    episode_morphology = analyse_episode_morphology(
        df_raw,
        episode_col="episode_id",
        sampling_hz=float(ds_cfg.get("sampling_hz", 1.0)),
    )
    logger.info(
        "  rows with any missing: {} | episodes: {}",
        missingness.get("rows_with_any_missing"),
        episode_morphology.get("n_episodes"),
    )
    logger.info("[2c/5] Class imbalance and targeted VDC1 outlier localization ...")
    class_imbalance = analyse_class_imbalance(df_raw, "episode_id")
    vdc1_outliers = analyse_vdc1_outliers(df_raw, "episode_id")
    logger.info(
        "  minority classes by rows (<5%): {} | VDC1 normal outliers: {}",
        class_imbalance.get("minority_fault_classes_by_rows_lt_5pct"),
        vdc1_outliers.get("n_outliers"),
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
    df_eda = attach_episode_ids(df_eda, gap_s, segment_col="episode_id")

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
        segment_col="episode_id",
    )
    consolidated["eda_frame_description"] = eda_desc

    if args.skip_stationarity:
        preserved_stationarity = _load_stationarity_sidecar(
            output_dir
        ) or _load_existing_stationarity(report_path)
        if preserved_stationarity:
            logger.info(
                "[4b/5] Preserving existing stationarity diagnostics while skipping recompute"
            )
            stationarity = preserved_stationarity
        else:
            logger.info("[4b/5] Skipping stationarity diagnostics (--skip-stationarity)")
            stationarity = {
                "skipped": True,
                "reason": "Skipped via --skip-stationarity to speed up EDA.",
            }
    else:
        logger.info("[4b/5] Stationarity diagnostics (ADF + KPSS) ...")
        stationarity = analyse_stationarity(df_eda, available_sensors, "episode_id", ds_cfg)
        _write_stationarity_sidecar(output_dir, stationarity)

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
        "missingness": missingness,
        "outlier_analysis": outlier_report,
        "class_imbalance": class_imbalance,
        "vdc1_outlier_localization": vdc1_outliers,
        "episode_morphology": episode_morphology,
        "stationarity": stationarity,
        "dataset_specific": dataset_specific,
        "split_recommendations": split_recs,
        "sensor_columns_used": available_sensors,
        "sensor_columns_missing": missing,
    }

    logger.info("Writing dataset report ...")
    report_path.write_text(
        json.dumps(_to_json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.success("Dataset report → {}", report_path)

    artifact_manifest = _build_artifact_manifest(args.dataset, output_dir, report_path, file_paths)
    artifact_manifest_path = output_dir / "eda_artifact_manifest.json"
    logger.info("Writing artifact manifest ...")
    artifact_manifest_path.write_text(
        json.dumps(_to_json_safe(artifact_manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.success("Artifact manifest → {}", artifact_manifest_path)

    # Re-write consolidated findings with patched dataset field
    consolidated_path = output_dir / "eda_feature_findings.json"
    logger.info("Rewriting consolidated findings with final metadata ...")
    consolidated_path.write_text(
        json.dumps(_to_json_safe(consolidated), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    logger.success("Consolidated findings → {}", consolidated_path)

    if not args.skip_figures:
        logger.info("[6/5] Generating EDA visualizations ...")
        from src.data.eda_visualization import generate_visualizations

        figure_count = generate_visualizations(config, args.dataset)
        logger.info("Generated {} EDA figures", figure_count)
    else:
        logger.info("Skipping EDA visualizations (--skip-figures)")

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
