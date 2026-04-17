"""
Data ingestion pipeline — multi-dataset support.

Supported datasets (pass --dataset <name>):
  reunion    La Réunion real labeled faults, ~7s sampling  [default]
  costa      Costa et al. real labeled faults, 1 Hz
  mendeley   GPVS-Faults simulated, high-rate

Each loader standardises to a common schema and writes one Parquet file to
data/interim/.  All downstream stages read from there.

Standard output schema
----------------------
  timestamp  : datetime[us, UTC]   — absolute time (or synthetic at 1 Hz for Costa)
  label      : int32                — fault class integer (0 = normal)
  <sensors>  : float64              — dataset-specific electrical / meteorological cols

Usage:
    uv run python -m src.data.ingestion                    # reunion (default)
    uv run python -m src.data.ingestion --dataset costa
    uv run python -m src.data.ingestion --dataset mendeley
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

# ============================================================================
# PATHS
# ============================================================================
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"
# Per-dataset ingestion output directories (mirror data_config.yaml merged_output paths)
_INGEST_DIR = {
    "la_reunion": INTERIM_DIR / "ingestion" / "la_reunion",
    "costa":      INTERIM_DIR / "ingestion" / "costa",
    "mendeley":   INTERIM_DIR / "ingestion" / "mendeley",
}
# Keep OUTPUT_DIR as a fallback alias for legacy callers
OUTPUT_DIR = INTERIM_DIR

REUNION_DIR = RAW_DIR / "University of La Réunion Data"


# ============================================================================
# LA RÉUNION (real, labeled faults, ~7s sampling)
# ============================================================================

def load_reunion(output_path: Path | None = None) -> pl.DataFrame:
    """
    Load La Réunion dt2 (fault-labeled inverter data) and dt1 (meteorological).
    Merges on nearest timestamp (tolerance 30s).
    Saves merged DataFrame to data/interim/reunion_dt2_merged.parquet.

    Returns a Polars DataFrame.
    """
    dt1_path = REUNION_DIR / "dt1_solar_and_meteorological_measurement.csv"
    dt2_path = REUNION_DIR / "dt2_electrical_production_inverter_1_with_faults.csv"

    logger.info("Loading La Réunion dt1 (meteorological)...")
    dt1 = pl.read_csv(dt1_path, try_parse_dates=True)
    logger.info(f"  dt1: {len(dt1):,} rows, columns: {dt1.columns}")

    logger.info("Loading La Réunion dt2 (fault-labeled electrical)...")
    dt2 = pl.read_csv(dt2_path, try_parse_dates=True)
    logger.info(f"  dt2: {len(dt2):,} rows — Fault distribution:")
    logger.info(f"  {dt2['Fault'].value_counts()}")

    # Sort both by timestamp for asof join
    dt1 = dt1.sort("time")
    dt2 = dt2.sort("time")

    # For each dt2 row find the nearest dt1 record within 30s
    merged = dt2.join_asof(dt1, on="time", strategy="nearest", tolerance="30s")
    logger.info(f"After asof join: {len(merged):,} rows, {merged.width} columns")

    if output_path is None:
        output_path = _INGEST_DIR["la_reunion"] / "reunion_dt2_merged.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(output_path)
    logger.success(f"Saved → {output_path}")
    return merged


def load_reunion_dt3(output_path: Path | None = None) -> pl.DataFrame:
    """
    Load La Réunion dt3 (healthy inverter 2 — no faults) + dt1 (meteorological).
    Needed for the differential power signal: ΔP = P_inv1 - α·P_inv2.

    Returns a Polars DataFrame.
    """
    dt1_path = REUNION_DIR / "dt1_solar_and_meteorological_measurement.csv"
    dt3_path = REUNION_DIR / "dt3_electrical_production_inverter_2.csv"

    logger.info("Loading La Réunion dt3 (healthy inverter)...")
    dt3 = pl.read_csv(dt3_path, try_parse_dates=True)
    logger.info(f"  dt3: {len(dt3):,} rows, columns: {dt3.columns}")

    logger.info("Loading La Réunion dt1 (meteorological) for dt3 merge...")
    dt1 = pl.read_csv(dt1_path, try_parse_dates=True)

    dt1 = dt1.sort("time")
    dt3 = dt3.sort("time")

    merged = dt3.join_asof(dt1, on="time", strategy="nearest", tolerance="30s")
    logger.info(f"dt3 after asof join: {len(merged):,} rows, {merged.width} columns")

    if output_path is None:
        output_path = _INGEST_DIR["la_reunion"] / "reunion_dt3_merged.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(output_path)
    logger.success(f"Saved → {output_path}")
    return merged


# ============================================================================
# COSTA (real, 1 Hz, labeled faults — Brazil)
# Source: Costa et al., Sensors 20(17), 4688 (2020)
# GitHub: https://github.com/clayton-h-costa/pv_fault_dataset
#
# Raw files (MATLAB v5 format, loaded with scipy.io.loadmat):
#   dataset_elec.mat  → vdc1, vdc2 (V), idc1, idc2 (A)   shape (1, N)
#   dataset_amb.mat   → irr (W/m²), pvt (°C), f_nv (label) shapes (1,N) / (N,1)
#
# Dataset facts (from probe):
#   N = 1,373,798 samples  ≈ 15.90 days at 1 Hz
#   52% nighttime (irr ≤ 5 W/m²)
#   Labels: 0=Normal, 1=Short-Circuit, 2=Degradation, 3=Open-Circuit, 4=Shadowing
#   Faults 1-3: artificially induced, ~10 min blocks, daytime only
#   Fault 4: NATURAL shadowing (buildings/clouds), 1136 episodes, mostly 1-2s transients
#
# Timestamp reconstruction:
#   No clock is stored in the .mat files. Epoch derived from two cross-checks:
#   (1) Curitiba solar geometry (lon 49.27°W): true solar noon = 15:17 UTC.
#       Mean irr peak over clean days (2-11) = 11.63 h from t0 → t0 ≈ 03:34 UTC.
#   (2) Paper fault schedule (11:30–13:00 BRT) under BRST (UTC-2): Day-13 fault
#       blocks anchor t0 = 03:37–03:38 UTC.
#   Final epoch: 2020-01-01T03:38:00Z ≈ 01:38 BRST. Both methods agree within 4 min.
#   With this epoch: sunrise ≈ 06:30 BRT, solar noon ≈ 12:17 BRT (physically consistent).
#   Cyclic hour features are meaningful; irradiance remains the reliable day/night proxy.
# ============================================================================

def load_costa(output_path: Path | None = None) -> pl.DataFrame:
    """
    Load Costa PV fault dataset from .mat files and save to Parquet.

    Output columns
    --------------
    timestamp : datetime[us, UTC]  — synthetic, 1 Hz, epoch 2020-01-01T03:38:00Z (≈ 01:38 BRST)
    label     : int32              — 0=Normal 1=ShortCircuit 2=Degradation
                                     3=OpenCircuit 4=Shadowing
    vdc1/2    : float64            — String voltage (V)
    idc1/2    : float64            — String current (A)
    pdc1/2    : float64            — String DC power (W)  = vdc * idc
    pdc       : float64            — Total DC power (W)   = pdc1 + pdc2
    irr       : float64            — Irradiance (W/m²), clipped to ≥ 0
    pvt       : float64            — PV module temperature (°C)
    is_daytime: bool               — irr > 5 W/m² (reliable day/night proxy)

    Notes
    -----
    - Very short label transitions (1-6 s) appear at fault boundaries — these are
      measurement artefacts, not separate experiments.
    - Shadowing (label 4) is natural (buildings/clouds), NOT artificially induced.
    - Night data is retained; use is_daytime to filter in downstream tasks.
    """
    import scipy.io

    costa_dir = RAW_DIR / "Costa PV Fault Dataset"
    elec_path = costa_dir / "dataset_elec.mat"
    amb_path  = costa_dir / "dataset_amb.mat"

    for p in (elec_path, amb_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Costa .mat file not found: {p}\n"
                "Clone from https://github.com/clayton-h-costa/pv_fault_dataset "
                "into data/raw/Costa PV Fault Dataset/"
            )

    logger.info("Loading Costa dataset_elec.mat ...")
    elec = scipy.io.loadmat(str(elec_path))

    logger.info("Loading Costa dataset_amb.mat ...")
    amb = scipy.io.loadmat(str(amb_path))

    # Extract and flatten — all arrays shape (1, N) or (N, 1)
    vdc1 = elec["vdc1"].flatten().astype(np.float64)
    vdc2 = elec["vdc2"].flatten().astype(np.float64)
    idc1 = elec["idc1"].flatten().astype(np.float64)
    idc2 = elec["idc2"].flatten().astype(np.float64)
    irr  = amb["irr"].flatten().astype(np.float64)
    pvt  = amb["pvt"].flatten().astype(np.float64)
    f_nv = amb["f_nv"].flatten().astype(np.int32)

    N = len(f_nv)
    logger.info("Costa dataset loaded | N={:,} samples ≈ {:.2f} days at 1 Hz", N, N / 86400)

    # Clip irradiance — sensor noise produces small negatives at night
    irr = np.clip(irr, 0.0, None)

    # Derived power channels
    pdc1 = vdc1 * idc1
    pdc2 = vdc2 * idc2
    pdc  = pdc1 + pdc2

    # Synthetic timestamp: 1 Hz, reconstructed from two independent cross-checks:
    #   (1) Solar geometry: Curitiba lon 49.27°W → true solar noon = 15:17 UTC.
    #       Mean irr peak (days 2-11, clean days) = 11.63 h from t0 → t0 ≈ 03:34 UTC.
    #   (2) Fault schedule anchor: paper reports faults at 11:30, 11:45, 12:05, 12:25,
    #       12:40, 13:00 BRT. Under BRST (UTC-2, in effect 2019 and earlier) the Day-13
    #       fault blocks triangulate t0 ≈ 03:37–03:38 UTC.
    # Both methods agree → t0 = 03:38 UTC = 01:38 BRST (Brazilian Summer Time UTC-2).
    # With this epoch sunrise ≈ 06:30 BRT, solar noon ≈ 12:17 BRT — physically consistent.
    # Polars Datetime("us") stores microseconds since Unix epoch — compute in µs directly.
    epoch_us = pd.Timestamp("2020-01-01 03:38:00", tz="UTC").value // 1_000  # ns → µs
    ts_us = epoch_us + np.arange(N, dtype=np.int64) * 1_000_000  # 1 s = 1,000,000 µs

    df = pl.DataFrame(
        {
            "timestamp": pl.Series(ts_us).cast(pl.Datetime("us", "UTC")),
            "label":     pl.Series(f_nv),
            "vdc1":      pl.Series(vdc1),
            "vdc2":      pl.Series(vdc2),
            "idc1":      pl.Series(idc1),
            "idc2":      pl.Series(idc2),
            "pdc1":      pl.Series(pdc1),
            "pdc2":      pl.Series(pdc2),
            "pdc":       pl.Series(pdc),
            "irr":       pl.Series(irr),
            "pvt":       pl.Series(pvt),
            "is_daytime": pl.Series(irr > 5.0),
        }
    )

    logger.info("Fault distribution (f_nv):")
    fault_names = {0: "Normal", 1: "ShortCircuit", 2: "Degradation",
                   3: "OpenCircuit", 4: "Shadowing"}
    counts = df["label"].value_counts().sort("label")
    for row in counts.iter_rows(named=True):
        lbl = row["label"]
        cnt = row["count"]
        logger.info("  {:d} ({:<13}): {:>9,} ({:.2f}%)",
                    lbl, fault_names.get(lbl, "Unknown"), cnt, 100 * cnt / N)

    daytime_n = int(df["is_daytime"].sum())
    logger.info("Daytime samples (irr>5): {:,} ({:.1f}%)", daytime_n, 100 * daytime_n / N)

    if output_path is None:
        output_path = _INGEST_DIR["costa"] / "costa_merged.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.success(
        "Saved → {} | {:,} rows × {} cols",
        output_path, len(df), df.width,
    )
    return df


# ============================================================================
# MENDELEY / GPVS-FAULTS (lab experiment, high-frequency, grid-connected PV)
# Reference: Bakdi et al., Int. J. Electrical Power & Energy Systems (2020)
# Mendeley Data: https://doi.org/10.17632/n76t439f65.1
#
# Raw files: data/raw/GPVS-Faults/CSV_Files/F{x}{y}.csv  (16 files)
#   x ∈ {0,..,7}: fault class — 0=fault-free, 1-7=seven fault types
#   y ∈ {L, M}  : operation mode — L=IPPT (limited power), M=MPPT (maximum power)
#
# Data facts:
#   ~10 kHz sampling (Ts ≈ 99.99 µs), ~14.4 seconds per experiment
#   ~143 K rows per file → ~2.3 M total before downsampling
#   Columns: Time, Ipv, Vpv, Vdc, ia, ib, ic, va, vb, vc, Iabc, If, Vabc, Vf
#
# Fault injection: faults introduced MANUALLY halfway through each experiment.
#   F0{L,M}: fault-free throughout (label=0).
#   F1-F7:   first half = normal pre-fault operation (label=0),
#             second half = active fault (label=fault_class).
#
# Timestamp: no real clock stored. Synthetic UTC timestamps are assigned by
#   placing each experiment in its own 1-hour window starting 2021-01-01T00:00Z,
#   so gap-based segmentation naturally yields one segment per experiment.
#   sim_time_s retains the original within-experiment simulation time (seconds).
# ============================================================================

def load_mendeley(output_path: Path | None = None, target_hz: float = 1000.0) -> pl.DataFrame:
    """
    Load GPVS-Faults (Mendeley) simulated PV fault dataset from CSV files.

    Parameters
    ----------
    target_hz : float
        Target sampling rate after downsampling (default 1000 Hz).
        Raw data is ~10 kHz; downsampling by ~10× reduces each file from
        ~143 K to ~14 K rows (total ~230 K rows for all 16 experiments).

    Output columns
    --------------
    timestamp   : datetime[us, UTC] — synthetic, experiment i starts at epoch + i*3600s
    label       : int32  — 0=Normal, 1-7=active fault class
    fault_class : int32  — fault class from filename (0-7), regardless of phase
    mode        : str    — "IPPT" (L files) or "MPPT" (M files)
    phase       : str    — "normal" (F0 files) | "pre_fault" | "fault" (F1-F7 files)
    sim_time_s  : float64 — original simulation time within the experiment (seconds)
    Ipv, Vpv, Vdc, ia, ib, ic, va, vb, vc : float64 — raw electrical measurements
    Iabc, If, Vabc, Vf : float64 — positive-sequence phasor estimates

    Notes
    -----
    - For classification, filter phase=="fault" for fault samples and
      fault_class==0 for normal samples to avoid pre-fault contamination.
    - For anomaly detection, use fault_class==0 (F0 files) as the clean normal class.
    - Downsampling uses uniform row-stride (every nth row), not averaging.
    """
    mendeley_dir = RAW_DIR / "GPVS-Faults" / "CSV_Files"
    if not mendeley_dir.exists():
        raise FileNotFoundError(
            f"GPVS-Faults CSV_Files directory not found: {mendeley_dir}\n"
            "Download from https://doi.org/10.17632/n76t439f65.1 and place under "
            "data/raw/GPVS-Faults/CSV_Files/"
        )

    # Epoch: 2021-01-01T00:00:00Z — distinct from Costa (2020) to avoid confusion.
    # Each experiment occupies a 1-hour window so gap-based segmentation splits them.
    base_epoch_us = pd.Timestamp("2021-01-01 00:00:00", tz="UTC").value // 1_000  # ns → µs
    experiment_gap_us = 3_600 * 1_000_000  # 1 hour in µs

    all_frames: list[pl.DataFrame] = []
    experiment_idx = 0

    for fault_class in range(8):          # F0 … F7
        for mode_char in ("L", "M"):      # IPPT, MPPT
            fname = f"F{fault_class}{mode_char}.csv"
            fpath = mendeley_dir / fname
            if not fpath.exists():
                logger.warning("Missing file, skipping: {}", fpath)
                continue

            logger.info("Loading {} (fault={}, mode={}) ...", fname, fault_class, mode_char)
            # Force all columns to Float64 — some files have Iabc/Vabc as integer 1
            # in early rows (F0 experiments at steady state), which causes schema
            # inference to pick Int64 and then fail on fractional values later.
            float_cols = {c: pl.Float64 for c in (
                "Time", "Ipv", "Vpv", "Vdc",
                "ia", "ib", "ic", "va", "vb", "vc",
                "Iabc", "If", "Vabc", "Vf",
            )}
            raw = pl.read_csv(fpath, schema_overrides=float_cols)

            # --- Downsample to target_hz ---
            # Estimate actual sampling rate from first two rows
            t0_s = raw["Time"][0]
            t1_s = raw["Time"][1]
            actual_hz = 1.0 / (t1_s - t0_s)
            stride = max(1, round(actual_hz / target_hz))
            if stride > 1:
                raw = raw[::stride]
            logger.info(
                "  {} rows (stride={}, actual≈{:.0f}Hz → target≈{:.0f}Hz)",
                len(raw), stride, actual_hz, actual_hz / stride,
            )

            # --- Label assignment ---
            # F0: fault-free throughout → label=0, phase="normal"
            # F1-F7: first half pre-fault (label=0), second half active fault (label=fault_class)
            n = len(raw)
            if fault_class == 0:
                labels = np.zeros(n, dtype=np.int32)
                phases = np.full(n, "normal")
            else:
                midpoint = n // 2
                labels = np.zeros(n, dtype=np.int32)
                labels[midpoint:] = fault_class
                phases = np.empty(n, dtype=object)
                phases[:midpoint] = "pre_fault"
                phases[midpoint:] = "fault"

            # --- Synthetic timestamps ---
            # Map sim_time_s (0 … ~14.4 s) into the experiment's UTC window.
            # sim_time_s values in microseconds + experiment offset
            sim_us = (raw["Time"].to_numpy() * 1_000_000).astype(np.int64)
            ts_us = base_epoch_us + experiment_idx * experiment_gap_us + sim_us

            frame = pl.DataFrame({
                "timestamp":   pl.Series(ts_us).cast(pl.Datetime("us", "UTC")),
                "label":       pl.Series(labels),
                "fault_class": pl.Series(np.full(n, fault_class, dtype=np.int32)),
                "mode":        pl.Series(np.full(n, "IPPT" if mode_char == "L" else "MPPT")),
                "phase":       pl.Series(phases.tolist()),
                "sim_time_s":  raw["Time"].cast(pl.Float64),
                "Ipv":  raw["Ipv"].cast(pl.Float64),
                "Vpv":  raw["Vpv"].cast(pl.Float64),
                "Vdc":  raw["Vdc"].cast(pl.Float64),
                "ia":   raw["ia"].cast(pl.Float64),
                "ib":   raw["ib"].cast(pl.Float64),
                "ic":   raw["ic"].cast(pl.Float64),
                "va":   raw["va"].cast(pl.Float64),
                "vb":   raw["vb"].cast(pl.Float64),
                "vc":   raw["vc"].cast(pl.Float64),
                "Iabc": raw["Iabc"].cast(pl.Float64),
                "If":   raw["If"].cast(pl.Float64),
                "Vabc": raw["Vabc"].cast(pl.Float64),
                "Vf":   raw["Vf"].cast(pl.Float64),
            })
            all_frames.append(frame)
            experiment_idx += 1

    if not all_frames:
        raise FileNotFoundError(
            f"No F{{x}}{{y}}.csv files found in {mendeley_dir}. "
            "Check that CSV_Files/ contains F0L.csv … F7M.csv."
        )

    df = pl.concat(all_frames)
    logger.info("Mendeley combined | {:,} rows × {} cols", len(df), df.width)

    # Fault distribution report
    logger.info("Label distribution (active fault phase):")
    fault_names = {
        0: "Normal/FaultFree", 1: "Fault1", 2: "Fault2", 3: "Fault3",
        4: "Fault4", 5: "Fault5", 6: "Fault6", 7: "Fault7",
    }
    counts = df["label"].value_counts().sort("label")
    total = len(df)
    for row in counts.iter_rows(named=True):
        lbl = row["label"]
        cnt = row["count"]
        logger.info("  {:d} ({:<15}): {:>8,} ({:.1f}%)",
                    lbl, fault_names.get(lbl, "Unknown"), cnt, 100 * cnt / total)

    logger.info("Phase distribution:")
    for row in df["phase"].value_counts().iter_rows(named=True):
        logger.info("  {}: {:,}", row["phase"], row["count"])

    if output_path is None:
        output_path = _INGEST_DIR["mendeley"] / "mendeley_merged.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.success("Saved → {} | {:,} rows × {} cols", output_path, len(df), df.width)
    return df


# ============================================================================
# DATASET REGISTRY
# ============================================================================

LOADERS: dict[str, callable] = {
    "la_reunion": load_reunion,
    "costa": load_costa,
    "mendeley": load_mendeley,
}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PV fault dataset to data/interim/")
    parser.add_argument(
        "--dataset",
        choices=list(LOADERS.keys()),
        default="la_reunion",
        help="Which dataset to ingest (default: la_reunion)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(OUTPUT_DIR / "ingestion.log", rotation="10 MB")

    logger.info("=== Stage 0: Data Ingestion — dataset={} ===", args.dataset)
    loader = LOADERS[args.dataset]
    df = loader()
    logger.info("Schema: {}", {k: str(v) for k, v in df.schema.items()})
    logger.info("First row: {}", df.row(0, named=True))

    if args.dataset == "la_reunion":
        logger.info("=== Stage 0b: Ingesting dt3 (healthy reference) ===")
        load_reunion_dt3()

    logger.success("=== Ingestion complete | dataset={} ===", args.dataset)
