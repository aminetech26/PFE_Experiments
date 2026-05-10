"""
Preprocessing pipeline for SCVAE model following Li et al. (2024).

Pipeline:
  1. Load ingested Costa dataset (Parquet)
  2. MAD filtering: select "normal" samples with lowest mean absolute
     difference between scaled power yield and site irradiance
  3. Data augmentation: slice-exchange & scaling to enlarge normal set
  4. Z-score standardization on normal-only training data
  5. Create sliding window sequences (window_size × n_features)
  6. Split: train (normal-only) / val (normal-only) / test (mixed normal + faults)
  7. Save preprocessed Parquet per split

Usage:
    uv run python -m src.data.preprocess_scvae
    uv run python -m src.data.preprocess_scvae --input path/to/costa_merged.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "ingestion" / "costa" / "costa_merged.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_scvae"

# Conditional features (environmental measurements)
CONDITIONAL_FEATURES = ["pvt", "irr"]
# Target features (PV power output to reconstruct)
TARGET_FEATURES = ["pdc1", "pdc2"]
# All features
ALL_FEATURES = CONDITIONAL_FEATURES + TARGET_FEATURES
# Peak power for normalization
PEAK_POWER_W = 2500.0


def load_costa_data(parquet_path: Path) -> pd.DataFrame:
    """Load ingested Costa dataset."""
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


def compute_mad_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Mean Absolute Difference between scaled power and irradiance.

    MAD = mean(|pdc / peak_power - irr / max(irr)|) over each window.
    Lower MAD → more likely to be normal operation.

    Uses pdc1 + pdc2 if 'pdc' column is absent; falls back to pdc1.
    """
    if "pdc" in df.columns:
        pdc = df["pdc"].values
    elif "pdc1" in df.columns and "pdc2" in df.columns:
        pdc = df["pdc1"].values + df["pdc2"].values
    elif "pdc1" in df.columns:
        pdc = df["pdc1"].values
    else:
        raise KeyError("No power column (pdc, pdc1, pdc2) found in DataFrame")

    irr = df["irr"].values
    irr_max = max(irr.max(), 1.0)
    pdc_max = max(pdc.max(), 1.0)

    pdc_scaled = pdc / pdc_max
    irr_scaled = irr / irr_max
    mad = np.abs(pdc_scaled - irr_scaled)

    df = df.copy()
    df["_mad_score"] = mad
    return df


def augment_data(
    df: pd.DataFrame,
    target_rows: int,
    slice_size: int = 72,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Data augmentation via feature perturbation (adapted from Li et al. 2024).

    Since the Costa dataset uses per-point features (not daily patches), the
    original slice-exchange method is adapted here: random feature-level scaling
    with multiplicative noise, applied to randomly selected rows.

    The paper's original method exchanges time slices between daily samples;
    this adaptation applies equivalent controlled perturbation per feature.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_current = len(df)
    if n_current < 2 or target_rows <= n_current:
        logger.info(f"No augmentation needed (have {n_current:,} rows)")
        return df

    n_to_generate = target_rows - n_current
    logger.info(f"Augmenting from {n_current:,} → {target_rows:,} ({n_to_generate:,} new)")

    cols = [c for c in ALL_FEATURES if c in df.columns]
    data = df[cols].values.astype(np.float32)
    n_features = data.shape[1]

    augmented = []
    for _ in range(n_to_generate):
        i, j = rng.integers(0, n_current, size=2)

        new_sample = data[i].copy()
        # Randomly select which features to exchange/perturb
        k = rng.integers(1, n_features + 1)
        feat_idx = rng.choice(n_features, size=k, replace=False)

        # Mix features between two random samples and apply scaling noise
        alpha = rng.uniform(0.3, 0.7)
        new_sample[feat_idx] = alpha * data[i][feat_idx] + (1 - alpha) * data[j][feat_idx]

        # Add small multiplicative noise
        noise = rng.normal(1.0, 0.05, size=n_features)
        new_sample = new_sample * noise

        augmented.append(new_sample)

    aug_data = np.vstack([data, np.array(augmented)])
    df_aug = pd.DataFrame(aug_data, columns=cols)
    df_aug = df_aug.iloc[:target_rows]
    logger.info(f"Augmentation complete: {len(df_aug):,} samples")
    return df_aug


def standardize(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Z-score standardize: fit on train, apply to all splits."""
    stats = {}
    for col in columns:
        mu = float(train_df[col].mean())
        sigma = float(train_df[col].std())
        if sigma < 1e-8:
            sigma = 1.0
        stats[col] = {"mean": mu, "std": sigma}

        train_df[col] = (train_df[col] - mu) / sigma
        if col in val_df.columns:
            val_df[col] = (val_df[col] - mu) / sigma
        if col in test_df.columns:
            test_df[col] = (test_df[col] - mu) / sigma

    return train_df, val_df, test_df, stats


def split_data(
    normal_df: pd.DataFrame,
    faulty_df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split normal data chronologically (no temporal leakage).

    Assumes normal_df is already sorted by timestamp (earliest first).
    First train_frac goes to train, next val_frac to val, remainder to test.
    Faulty data always goes to test only.
    """
    n_normal = len(normal_df)
    n_train = int(n_normal * train_frac)
    n_val = int(n_normal * val_frac)

    train_df = normal_df.iloc[:n_train].copy()
    val_df = normal_df.iloc[n_train:n_train + n_val].copy()

    test_normal = normal_df.iloc[n_train + n_val:].copy()
    test_df = pd.concat([test_normal, faulty_df], ignore_index=True)

    logger.info(
        f"Split: train={len(train_df):,} | val={len(val_df):,} | "
        f"test={len(test_df):,} (normal={len(test_normal):,} + faults={len(faulty_df):,})"
    )
    return train_df, val_df, test_df


def create_window_sequences(
    df: pd.DataFrame,
    window_size: int,
    columns: list[str],
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows with labels.

    Returns:
        X:      (n_windows, window_size, n_features)
        y_bin:  (n_windows,) binary anomaly label (0=normal, 1=fault)
        y_multi:(n_windows,) multi-class fault label
    """
    data = df[columns].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int32)

    n_samples = len(data)
    n_windows = max(0, (n_samples - window_size) // stride + 1)

    X = np.zeros((n_windows, window_size, len(columns)), dtype=np.float32)
    y_bin = np.zeros(n_windows, dtype=np.int32)
    y_multi = np.zeros(n_windows, dtype=np.int32)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        X[i] = data[start:end]
        win_labels = labels[start:end]
        y_bin[i] = 1 if np.any(win_labels > 0) else 0
        # Dominant fault class in window
        faults_only = win_labels[win_labels > 0]
        y_multi[i] = int(np.bincount(faults_only).argmax()) if len(faults_only) > 0 else 0

    return X, y_bin, y_multi


def data_integrity_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Run comprehensive data integrity checks and return report."""
    report = {}

    # 1. Check for NaN/Inf
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        cols_check = [c for c in ALL_FEATURES if c in df.columns]
        n_nan = df[cols_check].isna().sum().sum()
        n_inf = np.isinf(df[cols_check].to_numpy()).sum()
        report[f"{name}_nan_count"] = int(n_nan)
        report[f"{name}_inf_count"] = int(n_inf)

    # 2. Check train only contains label 0
    train_anomalies = (train_df["label"] != 0).sum()
    report["train_anomaly_count"] = int(train_anomalies)

    # 3. Label distribution
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = {}
        for lbl, cnt in df["label"].value_counts().items():
            dist[int(lbl)] = int(cnt)
        report[f"{name}_label_dist"] = dist

    # 4. Timestamp continuity check
    if "timestamp" in train_df.columns:
        time_diffs = train_df["timestamp"].diff().dropna()
        report["train_median_time_step_s"] = float(time_diffs.dt.total_seconds().median() or 0)
        report["train_max_gap_s"] = float(time_diffs.dt.total_seconds().max() or 0)

    # 5. Statistics
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for col in [c for c in ALL_FEATURES if c in df.columns]:
            report[f"{name}_{col}_mean"] = float(df[col].mean())
            report[f"{name}_{col}_std"] = float(df[col].std())

    return report


def leakage_checks_for_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Check for data leakage across splits."""
    checks = {}

    # Temporal leakage: train timestamps should be before val/test
    if "timestamp" in train_df.columns:
        train_max = train_df["timestamp"].max()
        val_min = val_df["timestamp"].min() if len(val_df) > 0 else train_max
        test_min = test_df["timestamp"].min() if len(test_df) > 0 else train_max
        checks["train_max_ts"] = str(train_max)
        checks["val_min_ts"] = str(val_min)
        checks["test_min_ts"] = str(test_min)
        checks["temporal_leak"] = bool(train_max > val_min)

    # Exact duplicate check on feature values
    cols = [c for c in ALL_FEATURES if c in train_df.columns]
    train_set = set(tuple(map(float, row)) for row in train_df[cols].iloc[:50000].to_numpy())
    val_set = set(tuple(map(float, row)) for row in val_df[cols].iloc[:50000].to_numpy())

    overlap = train_set & val_set
    checks["duplicate_samples"] = int(len(overlap))
    checks["duplicate_leak"] = bool(len(overlap) > 0 and len(val_set) > 0
                                    and len(overlap) / max(len(val_set), 1) > 0.01)

    return checks


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Preprocess Costa data for SCVAE")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window-size", type=int, default=72)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--mad-retention", type=float, default=80.0)
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Step 1: Load data and separate normal/faulty
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SCVAE PREPROCESSING — Costa Dataset")
    logger.info("=" * 60)

    df = load_costa_data(Path(args.input))

    normal_df = df[df["label"] == 0].copy().reset_index(drop=True)
    faulty_df = df[df["label"] > 0].copy().reset_index(drop=True)
    logger.info(
        f"Separated: {len(normal_df):,} normal | {len(faulty_df):,} faulty "
        f"({len(faulty_df)/len(df)*100:.1f}%)"
    )

    # ------------------------------------------------------------------
    # Step 2: MAD-based cleaning of normal data
    # ------------------------------------------------------------------
    normal_df = compute_mad_score(normal_df)
    normal_df = normal_df.sort_values("_mad_score").reset_index(drop=True)

    n_keep = max(1, int(len(normal_df) * args.mad_retention / 100))
    normal_clean = normal_df.iloc[:n_keep].copy()
    # Restore chronological order after MAD-based selection
    normal_clean = normal_clean.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        f"MAD filtering: retained {len(normal_clean):,} / {len(normal_df):,} "
        f"({args.mad_retention:.0f}%) lowest-MAD normal samples"
    )

    # ------------------------------------------------------------------
    # Step 3: Data augmentation (optional)
    # ------------------------------------------------------------------
    if args.augment:
        target = max(len(normal_clean), len(normal_df))  # restore to original size
        normal_clean = augment_data(
            normal_clean,
            target_rows=target,
            slice_size=min(args.window_size, len(normal_clean) - 1),
            rng=rng,
        )

    # ------------------------------------------------------------------
    # Step 4: Split data
    # ------------------------------------------------------------------
    train_df, val_df, test_df = split_data(
        normal_clean, faulty_df,
        train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Step 5: Z-score standardization (fit on train only)
    # ------------------------------------------------------------------
    std_cols = [c for c in ALL_FEATURES if c in train_df.columns]
    train_df, val_df, test_df, std_stats = standardize(
        train_df, val_df, test_df, std_cols,
    )

    # ------------------------------------------------------------------
    # Step 6: Data integrity report
    # ------------------------------------------------------------------
    integrity = data_integrity_report(train_df, val_df, test_df)
    logger.info("=" * 40)
    logger.info("DATA INTEGRITY REPORT")
    logger.info("=" * 40)
    logger.info(f"  Train: {len(train_df):,} rows, all normal={integrity['train_anomaly_count']==0}")
    logger.info(f"  Val:   {len(val_df):,} rows")
    logger.info(f"  Test:  {len(test_df):,} rows")
    for name in ["train", "val", "test"]:
        dist = integrity[f"{name}_label_dist"]
        dist_str = " | ".join(f"class {k}: {v}" for k, v in sorted(dist.items()))
        logger.info(f"  {name.capitalize()} label dist: {dist_str}")
    for col in std_cols:
        logger.info(f"  {col}: train μ={integrity[f'train_{col}_mean']:.4f} "
                     f"σ={integrity[f'train_{col}_std']:.4f}")
    logger.info(f"  Train median time step: {integrity.get('train_median_time_step_s', 'N/A'):.1f}s")
    logger.info(f"  Train max gap: {integrity.get('train_max_gap_s', 'N/A'):.1f}s")
    if integrity["train_anomaly_count"] > 0:
        logger.warning(f"WARNING: {integrity['train_anomaly_count']} anomalous rows in training set!")

    # ------------------------------------------------------------------
    # Step 7: Leakage checks
    # ------------------------------------------------------------------
    leakage = leakage_checks_for_splits(train_df, val_df, test_df)
    logger.info("=" * 40)
    logger.info("LEAKAGE CHECKS")
    logger.info("=" * 40)
    logger.info(f"  Temporal leakage: {leakage['temporal_leak']}")
    logger.info(f"  Duplicate samples (train ∩ val): {leakage['duplicate_samples']}")
    if leakage["temporal_leak"]:
        logger.warning("WARNING: Temporal leakage detected!")
    if leakage.get("duplicate_leak"):
        logger.warning("WARNING: Duplicate samples across splits!")

    # ------------------------------------------------------------------
    # Step 8: Create window sequences
    # ------------------------------------------------------------------
    logger.info("=" * 40)
    logger.info(f"Creating sliding windows (size={args.window_size})")

    result = {}
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X, y_bin, y_multi = create_window_sequences(sdf, args.window_size, std_cols)
        result[f"{name}_X"] = X
        result[f"{name}_y_bin"] = y_bin
        result[f"{name}_y_multi"] = y_multi
        n_windows = X.shape[0]
        n_anom = int(y_bin.sum())
        logger.info(
            f"  {name.capitalize()}: {n_windows:,} windows | "
            f"{n_anom} anomalous ({100*n_anom/max(n_windows,1):.1f}%) | "
            f"shape={X.shape}"
        )

    # ------------------------------------------------------------------
    # Step 9: Save
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        "dataset": "costa",
        "model": "scvae",
        "window_size": args.window_size,
        "conditional_features": CONDITIONAL_FEATURES,
        "target_features": TARGET_FEATURES,
        "all_features": std_cols,
        "standardization": std_stats,
        "mad_retention_pct": args.mad_retention,
        "augmentation": args.augment,
        "data_integrity": {
            k: {str(k2): v2 for k2, v2 in v.items()} if isinstance(v, dict) else v
            for k, v in integrity.items()
        },
        "leakage_checks": leakage,
    }

    npz_path = output_dir / "scvae_sequences.npz"
    np.savez_compressed(
        npz_path,
        train_X=result["train_X"],
        train_y_bin=result["train_y_bin"],
        train_y_multi=result["train_y_multi"],
        val_X=result["val_X"],
        val_y_bin=result["val_y_bin"],
        val_y_multi=result["val_y_multi"],
        test_X=result["test_X"],
        test_y_bin=result["test_y_bin"],
        test_y_multi=result["test_y_multi"],
    )
    logger.success(f"Saved window sequences → {npz_path}")

    meta_path = output_dir / "scvae_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.success(f"Saved metadata → {meta_path}")

    logger.success("=" * 60)
    logger.success("SCVAE Preprocessing Complete")
    logger.success(f"  Output: {output_dir}")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
