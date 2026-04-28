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

# Global parameters for normalization
SHORT_CIRCUIT_CURRENT = 9.45  # A
OPEN_CIRCUIT_VOLTAGE = 45.6   # V


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

    # 2. Normalize currents and voltages
    for col in ["idc1", "idc2"]:
        if col in df.columns:
            df[col] = df[col] / SHORT_CIRCUIT_CURRENT
            
    for col in ["vdc1", "vdc2"]:
        if col in df.columns:
            df[col] = df[col] / OPEN_CIRCUIT_VOLTAGE
            
    logger.info(f"Normalized currents by {SHORT_CIRCUIT_CURRENT}A and voltages by {OPEN_CIRCUIT_VOLTAGE}V.")

    # 3. Normalize irradiance
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
