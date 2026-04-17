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
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "interim"

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
        output_path = OUTPUT_DIR / "reunion_dt2_merged.parquet"
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
        output_path = OUTPUT_DIR / "reunion_dt3_merged.parquet"
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
# Timestamp limitation:
#   No clock is stored in the .mat files. We reconstruct a synthetic UTC timestamp
#   at 1 Hz starting from 2020-01-01 00:00:00 (matches publication year).
#   The exact recording start time is unknown, so time-of-day cyclic features will
#   be offset from real local time. Irradiance is the reliable day/night proxy.
# ============================================================================

def load_costa(output_path: Path | None = None) -> pl.DataFrame:
    """
    Load Costa PV fault dataset from .mat files and save to Parquet.

    Output columns
    --------------
    timestamp : datetime[us, UTC]  — synthetic, 1 Hz, epoch 2020-01-01T00:00:00Z
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

    # Synthetic timestamp: 1 Hz starting at publication-year epoch (2020-01-01 00:00:00 UTC).
    # The .mat files contain no clock information. The offset is arbitrary but consistent.
    # Polars Datetime("us") stores microseconds since Unix epoch — compute in µs directly.
    epoch_us = pd.Timestamp("2020-01-01 00:00:00", tz="UTC").value // 1_000  # ns → µs
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
        output_path = OUTPUT_DIR / "costa_merged.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.success(
        "Saved → {} | {:,} rows × {} cols",
        output_path, len(df), df.width,
    )
    return df


# ============================================================================
# MENDELEY / GPVS-FAULTS (simulated, MATLAB/Simulink)
# TODO: fill in actual file names once Mendeley CSVs are placed under data/raw/GPVS-Faults/
# Reference: Jovicic et al., Mendeley Data (2020)
# ============================================================================

def load_mendeley(output_path: Path | None = None) -> pl.DataFrame:
    """
    Load GPVS-Faults (Mendeley) simulated PV fault dataset.

    Expected raw files:
      data/raw/GPVS-Faults/<fault_data>.csv (or multiple CSVs per fault class)

    TODO: verify exact CSV structure and column mappings.
    """
    mendeley_dir = RAW_DIR / "GPVS-Faults"
    if not mendeley_dir.exists():
        raise FileNotFoundError(
            f"GPVS-Faults directory not found: {mendeley_dir}\n"
            "Download from https://doi.org/10.17632/n76t439f65.1 and place CSVs there."
        )

    raise NotImplementedError(
        "Mendeley ingestion not yet implemented. "
        "Fill in file names and column mappings once you've inspected the CSVs."
    )


# ============================================================================
# DATASET REGISTRY
# ============================================================================

LOADERS: dict[str, callable] = {
    "reunion": load_reunion,
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
        default="reunion",
        help="Which dataset to ingest (default: reunion)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(OUTPUT_DIR / "ingestion.log", rotation="10 MB")

    logger.info("=== Stage 0: Data Ingestion — dataset={} ===", args.dataset)
    loader = LOADERS[args.dataset]
    df = loader()
    logger.info("Schema: {}", {k: str(v) for k, v in df.schema.items()})
    logger.info("First row: {}", df.row(0, named=True))

    if args.dataset == "reunion":
        logger.info("=== Stage 0b: Ingesting dt3 (healthy reference) ===")
        load_reunion_dt3()

    logger.success("=== Ingestion complete | dataset={} ===", args.dataset)
