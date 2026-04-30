"""
Preprocessing specifically for the LSTM Autoencoder model on the Costa dataset.

Performs:
1. Addition of time-based features: time_of_day_sin, time_of_day_cos
2. Normalization of current and voltage via Short Circuit Current and Open Circuit Voltage
3. MinMax scaling of pvt (temperature)
4. Dividing irr by 1000
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import pvlib

# Global parameters for normalization
SHORT_CIRCUIT_CURRENT = 9.45  # A
OPEN_CIRCUIT_VOLTAGE = 45.6   # V
PEAK_POWER = 4000.0           # W (Published peak power yield)

# Coordinates for Clearness Index (USER TO FILL)
LATITUDE = -25.438686
LONGITUDE = -49.268487
ALTITUDE = 935

def preprocess_for_lstmae(input_parquet: str | Path, output_parquet: str | Path) -> None:
    input_parquet = Path(input_parquet)
    output_parquet = Path(output_parquet)

    logger.info(f"Reading ingestion parquet: {input_parquet}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input file not found: {input_parquet}")

    df = pd.read_parquet(input_parquet)

    # 1. Time of day features
    if "timestamp" in df.columns:
        # Avoid timezone issues by using localized hour if applicable, assuming UTC for Costa
        time_col = df["timestamp"]
        hour_of_day = time_col.dt.hour + time_col.dt.minute / 60.0 + time_col.dt.second / 3600.0
        
        df["time_of_day_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
        df["time_of_day_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)
        logger.info("Added time_of_day_sin and time_of_day_cos features.")
    else:
        logger.warning("No 'timestamp' column found. Time of day features skipped.")

    # 2. Physics-Based Features Calculation (BEFORE Normalization)
    if "irr" in df.columns and "timestamp" in df.columns:
        # Calculate clear sky irradiance using pvlib
        loc = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE)
        times = pd.DatetimeIndex(df["timestamp"])
        
        # Determine the timezone - assume UTC if naive
        if times.tzinfo is None:
            times = times.tz_localize("UTC")
            
        clear_sky = loc.get_clearsky(times)
        
        # Calculate Clearness Index (Measured Irradiance / Clear Sky GHI)
        # Add epsilon to prevent division by zero during nights/eclipses
        df["clearness_index"] = df["irr"].values / (clear_sky["ghi"].values + 1e-6)
        
        # Cap abnormal values (e.g. slight sensor positive reading at night, but clear sky is ~0)
        df["clearness_index"] = df["clearness_index"].clip(lower=0, upper=1.5)
        logger.info("Added physics-based feature: clearness_index.")

    # 3. Normalize currents, voltages, and powers
    for col in ["idc1", "idc2"]:
        if col in df.columns:
            df[col] = df[col] / SHORT_CIRCUIT_CURRENT
            
    for col in ["vdc1", "vdc2"]:
        if col in df.columns:
            df[col] = df[col] / OPEN_CIRCUIT_VOLTAGE
            
    for col in ["pdc1", "pdc2"]:
        if col in df.columns:
            df[col] = df[col] / PEAK_POWER
            
    logger.info(f"Normalized currents ({SHORT_CIRCUIT_CURRENT}A), voltages ({OPEN_CIRCUIT_VOLTAGE}V), and powers ({PEAK_POWER}W).")

    # 4. Normalize irradiance
    if "irr" in df.columns:
        df["irr"] = df["irr"] / 1000.0
        logger.info("Normalized irr by 1000.")

    # 4. MinMax Scale temperature
    if "pvt" in df.columns:
        pvt_min = df["pvt"].min()
        pvt_max = df["pvt"].max()
        if pvt_max > pvt_min:
            df["pvt"] = (df["pvt"] - pvt_min) / (pvt_max - pvt_min)
        else:
            df["pvt"] = 0.0
        logger.info("MinMax scaled pvt (temperature).")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logger.success(f"Saved LSTM-AE preprocessed data to {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Costa dataset for LSTM-AE")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/interim/ingestion/costa/costa_merged.parquet",
        help="Path to ingestion parquet"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed/preprocessed/costa_lstmae/lstmae_preprocessed.parquet",
        help="Path to output preprocessed parquet"
    )
    args = parser.parse_args()
    
    preprocess_for_lstmae(args.input, args.output)
