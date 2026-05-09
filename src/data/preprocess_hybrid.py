"""
Preprocessing pipeline for the Hybrid Anomaly Detection model.

Implements the data preparation from:
  Ahirwar & Nandanwar (2025) "Enhanced Anomaly Detection in Solar Power
  Plants Using Hybrid Machine Learning Techniques", ICoEIT.

Pipeline:
  1. Load ingested Costa dataset (Parquet, native 1-second resolution)
  2. Temporal split: train (normal-only, first 70%) / val (normal-only, next 15%)
     / test (mixed normal + faults, remaining time)
  3. Facebook Prophet: fit on train-normal pdc with irr/pvt as regressors,
     predict for all splits, compute residuals
  4. Create sliding residual windows for AE-LSTM (window_size × n_residual_features)
  5. Create residual feature vectors for Isolation Forest (window-level statistics)
  6. Data integrity checks: NaN, Inf, label distributions, temporal ordering
  7. Leakage checks: temporal contiguity, duplicate samples
  8. Save preprocessed sequences + metadata

NOTE: Prophet fitting on 500K+ data points at 1-second resolution may be slow.
Consider using --window-size 300 --stride 150 for ~5-minute equivalent windows.

Usage:
    uv run python -m src.data.preprocess_hybrid
    uv run python -m src.data.preprocess_hybrid --input path/to/costa_merged.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "ingestion" / "costa" / "costa_merged.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_hybrid"

TARGET_COL = "pdc"
REGRESSOR_COLS = ["irr", "pvt"]
RESIDUAL_COLS = [TARGET_COL]
ALL_SENSOR_COLS = ["vdc1", "vdc2", "idc1", "idc2", "pdc1", "pdc2", "pdc", "irr", "pvt"]

FAULT_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}


def load_costa_data(parquet_path: Path) -> pd.DataFrame:
    """Load ingested Costa dataset and set timestamp index."""
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Ingested data not found: {parquet_path}\n"
            "Run: uv run python -m src.data.ingestion --dataset costa"
        )
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded Costa: {len(df):,} rows, columns={list(df.columns)}")
    return df


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split respecting chronological order.

    Train  = first train_frac of normal data (label == 0)
    Val    = next val_frac of normal data
    Test   = remaining normal + ALL faulty data (from any time period)
    """
    normal = df[df["label"] == 0].copy()
    faulty = df[df["label"] > 0].copy()

    n_normal = len(normal)
    train_end = int(n_normal * train_frac)
    val_end = int(n_normal * (train_frac + val_frac))

    train_df = normal.iloc[:train_end].copy()
    val_df = normal.iloc[train_end:val_end].copy()
    test_normal = normal.iloc[val_end:].copy()

    test_df = pd.concat([test_normal, faulty], ignore_index=True)
    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logger.info(
        f"Temporal split: train={len(train_df):,} (normal only) | "
        f"val={len(val_df):,} (normal only) | "
        f"test={len(test_df):,} (normal={len(test_normal):,} + faults={len(faulty):,})"
    )
    logger.info(
        f"  Train time range: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}"
    )
    if len(val_df) > 0:
        logger.info(
            f"  Val   time range: {val_df['timestamp'].min()} → {val_df['timestamp'].max()}"
        )
    logger.info(
        f"  Test  time range: {test_df['timestamp'].min()} → {test_df['timestamp'].max()}"
    )

    return train_df, val_df, test_df


def fit_prophet_residuals(
    train_df: pd.DataFrame,
    predict_dfs: list[pd.DataFrame],
    target_col: str = TARGET_COL,
    regressor_cols: list[str] | None = None,
) -> tuple[list[pd.DataFrame], dict]:
    """Fit Facebook Prophet on train data, predict and compute residuals.

    Prophet is fit ONLY on training data (leakage prevention).
    Returns each df with added '_prediction' and '_residual' columns.

    Falls back to linear regression if Prophet cannot be imported or fails.
    """
    try:
        import os as _os
        _os.environ.setdefault("MPLBACKEND", "Agg")
        from prophet import Prophet
    except Exception as e:
        logger.warning(f"prophet unavailable ({e}). Falling back to linear regression residuals.")
        return _fallback_linear_residuals(train_df, predict_dfs, target_col, regressor_cols)

    prop_data = pd.DataFrame({
        "ds": pd.to_datetime(train_df["timestamp"]),
        "y": train_df[target_col].values.astype(float),
    })

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )

    if regressor_cols:
        for col in regressor_cols:
            if col in train_df.columns:
                prop_data[col] = train_df[col].values.astype(float)
                model.add_regressor(col)

    logger.info("Fitting Prophet on training data...")
    model.fit(prop_data)

    results = []
    for df in predict_dfs:
        future = pd.DataFrame({"ds": pd.to_datetime(df["timestamp"])})
        if regressor_cols:
            for col in regressor_cols:
                if col in df.columns:
                    future[col] = df[col].values.astype(float)

        forecast = model.predict(future)
        df = df.copy()
        df["_prediction"] = forecast["yhat"].values.astype(np.float32)
        df["_residual"] = (df[target_col].values - df["_prediction"].values).astype(np.float32)
        results.append(df)

    prophet_info = {
        "target_col": target_col,
        "regressor_cols": regressor_cols or [],
        "train_rmse": float(np.sqrt(np.mean(
            (prop_data["y"].values - model.predict(prop_data)["yhat"].values) ** 2
        ))),
    }
    logger.info(f"  Prophet train RMSE: {prophet_info['train_rmse']:.4f}")
    return results, prophet_info


def _fallback_linear_residuals(
    train_df: pd.DataFrame,
    predict_dfs: list[pd.DataFrame],
    target_col: str,
    regressor_cols: list[str] | None,
) -> tuple[list[pd.DataFrame], dict]:
    """Fallback: linear regression for residuals if Prophet not available."""
    from sklearn.linear_model import Ridge

    X_train = train_df[regressor_cols or []].values.astype(float)
    y_train = train_df[target_col].values.astype(float)

    if regressor_cols and len(regressor_cols) > 0:
        model = Ridge(alpha=1.0).fit(X_train, y_train)

        results = []
        for df in predict_dfs:
            df = df.copy()
            X = df[regressor_cols].values.astype(float)
            df["_prediction"] = model.predict(X).astype(np.float32)
            df["_residual"] = (df[target_col].values - df["_prediction"].values).astype(np.float32)
            results.append(df)

        train_pred = model.predict(X_train)
        rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
    else:
        mu = float(y_train.mean())
        results = []
        for df in predict_dfs:
            df = df.copy()
            df["_prediction"] = mu
            df["_residual"] = (df[target_col].values - mu).astype(np.float32)
            results.append(df)
        rmse = float(np.sqrt(np.mean((y_train - mu) ** 2)))

    logger.info(f"  Linear regression train RMSE: {rmse:.4f}")
    return results, {
        "target_col": target_col,
        "regressor_cols": regressor_cols or [],
        "train_rmse": rmse,
        "method": "linear_regression",
    }


def normalize_residuals(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    residual_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Z-score normalize residuals: fit stats on train, apply to all splits."""
    stats = {}
    for col in residual_cols:
        if col not in train_df.columns:
            continue
        mu = float(train_df[col].mean())
        sigma = float(train_df[col].std())
        if sigma < 1e-8:
            sigma = 1.0
        stats[col] = {"mean": mu, "std": sigma}
        train_df[col + "_norm"] = (train_df[col] - mu) / sigma
        val_df[col + "_norm"] = (val_df[col] - mu) / sigma
        test_df[col + "_norm"] = (test_df[col] - mu) / sigma
    return train_df, val_df, test_df, stats


def create_residual_windows(
    df: pd.DataFrame,
    window_size: int,
    residual_cols: list[str],
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows of normalized residuals with labels.

    Returns:
        X:      (n_windows, window_size, n_residual_features)
        y_bin:  (n_windows,) binary anomaly label
        y_multi:(n_windows,) multi-class fault label
    """
    norm_cols = [f"{c}_norm" for c in residual_cols if f"{c}_norm" in df.columns]
    if not norm_cols:
        raise ValueError("No normalized residual columns found")

    data = df[norm_cols].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int32)

    n = len(data)
    n_windows = max(0, (n - window_size) // stride + 1)

    X = np.zeros((n_windows, window_size, len(norm_cols)), dtype=np.float32)
    y_bin = np.zeros(n_windows, dtype=np.int32)
    y_multi = np.zeros(n_windows, dtype=np.int32)

    for i in range(n_windows):
        start = i * stride
        X[i] = data[start:start + window_size]
        win_labels = labels[start:start + window_size]
        y_bin[i] = 1 if np.any(win_labels > 0) else 0
        faults = win_labels[win_labels > 0]
        y_multi[i] = int(np.bincount(faults).argmax()) if len(faults) > 0 else 0

    return X, y_bin, y_multi


def create_isolation_forest_features(
    df: pd.DataFrame,
    window_size: int,
    residual_cols: list[str],
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create window-level feature vectors for Isolation Forest.

    For each window of residuals, compute:
      - mean, std, min, max, range of each normalized residual
      - additional: target column mean, std over the window
    """
    norm_cols = [f"{c}_norm" for c in residual_cols if f"{c}_norm" in df.columns]
    if not norm_cols:
        raise ValueError("No normalized residual columns found")

    data = df[norm_cols].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int32)

    n = len(data)
    n_windows = max(0, (n - window_size) // stride + 1)
    n_feats = len(norm_cols)

    features_list = []
    y_bin = np.zeros(n_windows, dtype=np.int32)
    y_multi = np.zeros(n_windows, dtype=np.int32)

    for i in range(n_windows):
        start = i * stride
        win = data[start:start + window_size]

        feats = []
        for j in range(n_feats):
            col_win = win[:, j]
            feats.extend([
                np.mean(col_win),
                np.std(col_win) if len(col_win) > 1 else 0.0,
                np.min(col_win),
                np.max(col_win),
                np.ptp(col_win),
            ])

        features_list.append(feats)

        win_labels = labels[start:start + window_size]
        y_bin[i] = 1 if np.any(win_labels > 0) else 0
        faults = win_labels[win_labels > 0]
        y_multi[i] = int(np.bincount(faults).argmax()) if len(faults) > 0 else 0

    return np.array(features_list, dtype=np.float32), y_bin, y_multi


def data_integrity_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Comprehensive data integrity checks."""
    report = {}

    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for col in ["_residual", "_residual_norm"]:
            if col in sdf.columns:
                vals = sdf[col].values
                report[f"{name}_{col}_nan"] = int(np.isnan(vals).sum())
                report[f"{name}_{col}_inf"] = int(np.isinf(vals).sum())
        report[f"{name}_samples"] = int(len(sdf))

    # Train must be all normal
    report["train_anomaly_count"] = int((train_df["label"] != 0).sum())

    # Label distributions
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = {int(k): int(v) for k, v in sdf["label"].value_counts().items()}
        report[f"{name}_label_dist"] = dist

    # Temporal checks
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "timestamp" in sdf.columns and len(sdf) > 0:
            report[f"{name}_time_start"] = str(sdf["timestamp"].min())
            report[f"{name}_time_end"] = str(sdf["timestamp"].max())

    return report


def leakage_checks(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Temporal leakage and duplicate detection."""
    checks = {}
    checks["temporal_leak"] = False
    checks["duplicate_leak"] = False

    if "timestamp" in train_df.columns and len(val_df) > 0:
        train_max = train_df["timestamp"].max()
        val_min = val_df["timestamp"].min()
        checks["train_max_ts"] = str(train_max)
        checks["val_min_ts"] = str(val_min)
        checks["temporal_leak"] = bool(train_max > val_min)

    # Check for exact row duplicates between train and val (small sample)
    check_cols = [c for c in ALL_SENSOR_COLS if c in train_df.columns]
    if check_cols:
        train_sample = set(
            tuple(float(x) for x in row)
            for row in train_df[check_cols].iloc[:10000].values
        )
        val_sample = set(
            tuple(float(x) for x in row)
            for row in val_df[check_cols].iloc[:10000].values
        )
        overlap = train_sample & val_sample
        checks["duplicate_count"] = int(len(overlap))
        checks["duplicate_leak"] = bool(
            len(overlap) > 0 and len(val_sample) > 0
            and len(overlap) / max(len(val_sample), 1) > 0.01
        )

    return checks


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Preprocess Costa data for Hybrid model")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window-size", type=int, default=300, help="Residual window size (datapoints at native resolution)")
    parser.add_argument("--stride", type=int, default=150, help="Stride between windows")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("HYBRID MODEL PREPROCESSING — Costa Dataset (native resolution)")
    logger.info("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────────
    logger.info("Step 1: Load Costa data (1-second resolution)")
    df = load_costa_data(Path(args.input))
    logger.info(f"  {len(df):,} rows at native resolution")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        logger.info(f"    class {int(lbl)} ({FAULT_NAMES.get(int(lbl), '?')}): {cnt:,}")

    # ── Step 2: Temporal split ────────────────────────────────────────
    logger.info("\nStep 2: Temporal split")
    train_df, val_df, test_df = temporal_split(df, args.train_frac, args.val_frac)

    # ── Step 3: Prophet fit & residuals ────────────────────────────────
    logger.info("\nStep 3: Prophet fit & residual computation")
    predict_dfs_in = [train_df, val_df, test_df]
    result_dfs, prophet_info = fit_prophet_residuals(
        train_df, predict_dfs_in,
        target_col=TARGET_COL,
        regressor_cols=REGRESSOR_COLS,
    )
    train_df, val_df, test_df = result_dfs

    # ── Step 4: Normalize residuals ────────────────────────────────────
    logger.info("\nStep 4: Normalize residuals (fit on train only)")
    residual_cols = ["_residual"]
    train_df, val_df, test_df, norm_stats = normalize_residuals(
        train_df, val_df, test_df, residual_cols,
    )

    # ── Step 5: Create windowed data ──────────────────────────────────
    stride = args.stride
    logger.info(f"\nStep 5: Create residual windows (size={args.window_size}, stride={stride})")

    window_data = {}
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X_res, y_bin, y_multi = create_residual_windows(
            sdf, args.window_size, residual_cols, stride=stride,
        )
        X_if, _, _ = create_isolation_forest_features(
            sdf, args.window_size, residual_cols, stride=stride,
        )

        window_data[f"{name}_X_res"] = X_res
        window_data[f"{name}_X_if"] = X_if
        window_data[f"{name}_y_bin"] = y_bin
        window_data[f"{name}_y_multi"] = y_multi

        n_anom = int(y_bin.sum())
        logger.info(
            f"  {name}: {X_res.shape[0]:,} residual windows | "
            f"{n_anom} anomalous ({100*n_anom/max(X_res.shape[0],1):.1f}%) | "
            f"IF features={X_if.shape[1]}"
        )

    # ── Step 6: Data integrity report ──────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("DATA INTEGRITY REPORT")
    logger.info("=" * 50)

    integrity = data_integrity_report(train_df, val_df, test_df)
    logger.info(f"  Train: {integrity['train_samples']:,} rows, all_normal={integrity['train_anomaly_count']==0}")
    for name in ["train", "val", "test"]:
        dist = integrity.get(f"{name}_label_dist", {})
        dist_str = " | ".join(f"class {k}: {v}" for k, v in sorted(dist.items()))
        logger.info(f"  {name.capitalize()}: {integrity.get(f'{name}_samples',0):,} rows | {dist_str}")
    if integrity["train_anomaly_count"] > 0:
        logger.warning(f"WARNING: {integrity['train_anomaly_count']} anomalous rows in train!")

    # ── Step 7: Leakage checks ────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("LEAKAGE CHECKS")
    logger.info("=" * 50)

    leakage = leakage_checks(train_df, val_df, test_df)
    logger.info(f"  Temporal leak: {leakage['temporal_leak']}")
    logger.info(f"  Duplicate leak: {leakage.get('duplicate_leak', False)}")
    if leakage["temporal_leak"]:
        logger.warning("WARNING: Temporal leakage detected! Train after val.")
    if leakage.get("duplicate_leak"):
        logger.warning("WARNING: Duplicate samples across splits!")

    # ── Step 8: Save ──────────────────────────────────────────────────
    logger.info("\nStep 8: Save preprocessed data")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "hybrid_sequences.npz"
    np.savez_compressed(npz_path, **window_data)
    logger.success(f"  Saved sequences → {npz_path}")

    metadata = {
        "dataset": "costa",
        "model": "hybrid_aelstm_prophet_if",
        "paper_reference": "Ahirwar & Nandanwar (2025) ICoEIT",
        "window_size": args.window_size,
        "stride": stride,
        "target_col": TARGET_COL,
        "regressor_cols": REGRESSOR_COLS,
        "residual_cols": residual_cols,
        "all_sensor_cols": ALL_SENSOR_COLS,
        "prophet_info": prophet_info,
        "norm_stats": norm_stats,
        "data_integrity": {k: v for k, v in integrity.items()},
        "leakage_checks": leakage,
        "n_if_features": window_data.get("train_X_if", np.array([])).shape[1],
    }

    meta_path = output_dir / "hybrid_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.success(f"  Saved metadata → {meta_path}")

    logger.success("=" * 60)
    logger.success("HYBRID PREPROCESSING COMPLETE")
    logger.success(f"  Output: {output_dir}")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
