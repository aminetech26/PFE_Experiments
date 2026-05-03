"""
Preprocessing specifically for the SCVAE model on the Costa dataset.

Performs:
1. Division of powers (pdc1, pdc2) by Peak Power
2. Z-score standardization on specific features: pvt, irr, pdc1, pdc2
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# Global parameters for normalization
PEAK_POWER = 2500.0           # W (Published peak power yield)

def preprocess_for_scvae(input_parquet: str | Path, output_parquet: str | Path) -> None:
    input_parquet = Path(input_parquet)
    output_parquet = Path(output_parquet)

    logger.info(f"Reading ingestion parquet: {input_parquet}")
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input file not found: {input_parquet}")

    df = pd.read_parquet(input_parquet)

    # 1. Divide powers by peak power
    for col in ["pdc1", "pdc2"]:
        if col in df.columns:
            df[col] = df[col] / PEAK_POWER
            
    logger.info(f"Divided powers (pdc1, pdc2) by {PEAK_POWER}W.")

    # 2. Z-score standardization for pvt, irr, pdc1, pdc2
    features_to_standardize = ["pvt", "irr", "pdc1", "pdc2"]
    
    for col in features_to_standardize:
        if col in df.columns:
            col_mean = df[col].mean()
            col_std = df[col].std()
            if col_std > 0:
                df[col] = (df[col] - col_mean) / col_std
            else:
                # Fallback if standard deviation is zero mapping all to 0.0
                df[col] = 0.0
                
    logger.info(f"Applied z-score standardization to: {features_to_standardize}.")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logger.success(f"Saved SCVAE preprocessed data to {output_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Costa dataset for SCVAE")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/interim/ingestion/costa/costa_merged.parquet",
        help="Path to ingestion parquet"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed/preprocessed/costa_scvae/scvae_preprocessed.parquet",
        help="Path to output preprocessed parquet"
    )
    args = parser.parse_args()
    
    preprocess_for_scvae(args.input, args.output)
