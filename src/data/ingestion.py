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
    logger.success("=== Ingestion complete ===")
