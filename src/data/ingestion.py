"""
Data ingestion pipeline — La Réunion dataset.
Loads dt1 (meteorological) + dt2 (fault-labeled electrical), merges on timestamp,
saves to data/interim/ as Parquet.
All downstream stages read from data/interim/.
"""
from __future__ import annotations

from pathlib import Path

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
# MAIN
# ============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(OUTPUT_DIR / "ingestion.log", rotation="10 MB")

    logger.info("=== Stage 0: Data Ingestion — La Réunion ===")
    df = load_reunion()
    print(df.head(3))
    print(f"Schema: {df.schema}")

    logger.info("=== Stage 0b: Ingesting dt3 (healthy reference) ===")
    df3 = load_reunion_dt3()
    print(df3.head(3))
    logger.success("=== Ingestion complete ===")
