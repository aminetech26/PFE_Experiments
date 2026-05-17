#!/usr/bin/env python3
"""
GTBAD Row-by-Row Evaluation — iterates over costa_merged.parquet and displays
anomaly score (reconstruction error), decision (ANOMALY/normal), and real label
for every row.

Usage:
    uv run python -m src.evaluation.evaluate_gtbad
    uv run python -m src.evaluation.evaluate_gtbad --start-row 1000 --max-rows 500
    uv run python -m src.evaluation.evaluate_gtbad --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.modeling.anomaly_detection.dl.gtbad_model import GTBADModel, reconstruction_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "ingestion" / "costa" / "costa_merged.parquet"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "checkpoints" / "gtbad" / "gtbad_best.pt"
DEFAULT_THRESHOLD_JSON = PROJECT_ROOT / "experiments" / "metrics" / "gtbad_results.json"

CLASS_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def load_threshold_from_json(json_path: Path) -> float:
    if not json_path.exists():
        raise FileNotFoundError(
            f"Training results not found: {json_path}\n"
            "Run: uv run python -m src.modeling.anomaly_detection.dl.train_gtbad"
        )
    with open(json_path, "r") as f:
        data = json.load(f)
    threshold = data["anomaly_detection"]["threshold_value"]
    logger.info(f"  Loaded threshold from {json_path}: {threshold:.6f}")
    return float(threshold)


@torch.no_grad()
def infer_single(
    model: GTBADModel,
    raw_row: np.ndarray,
    scaler_min: np.ndarray,
    scaler_max: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    scaler_range = scaler_max - scaler_min
    scaler_range[scaler_range < 1e-10] = 1.0

    raw = raw_row.astype(np.float32)
    scaled = (raw - scaler_min) / scaler_range

    X_tensor = torch.from_numpy(scaled[np.newaxis, np.newaxis, :]).float().to(device)
    recon_tensor = model(X_tensor)
    error_val = float(reconstruction_error(X_tensor, recon_tensor).cpu().numpy()[0])

    return {
        "raw": raw.tolist(),
        "scaled": scaled.tolist(),
        "reconstructed": recon_tensor.cpu().numpy()[0, 0, :].tolist(),
        "error": error_val,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GTBAD row-by-row: score, decision, and real label"
    )
    parser.add_argument("--parquet-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--threshold-json", type=str, default=str(DEFAULT_THRESHOLD_JSON))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-row", type=int, default=0,
                        help="Row index to start inference from (0-based)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit rows to process (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── 1. Load data ──────────────────────────────────────────────────────
    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet not found: {parquet_path}\n"
            "Run: uv run python -m src.data.ingestion --dataset costa"
        )
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded: {len(df):,} rows from {parquet_path}")

    if args.start_row > 0:
        df = df.iloc[args.start_row:]
        logger.info(f"  Starting from row index {args.start_row}")

    if args.max_rows is not None:
        df = df.head(args.max_rows)
        logger.info(f"  Limited to {args.max_rows:,} rows")

    # ── 2. Load checkpoint ────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run: uv run dvc pull train_gtbad"
        )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    n_features = checkpoint["n_features"]
    feature_names = checkpoint["feature_names"]
    scaler_min = np.array(checkpoint["scaler_min"], dtype=np.float32)
    scaler_max = np.array(checkpoint["scaler_max"], dtype=np.float32)
    logger.info(f"  Checkpoint: {n_features} features ({feature_names})")

    args_ckpt = checkpoint.get("args", {})
    d_model = args_ckpt.get("d_model", 64)
    nhead = args_ckpt.get("nhead", 2)
    num_encoder_layers = args_ckpt.get("num_encoder_layers", 3)
    lstm_hidden = args_ckpt.get("lstm_hidden", 32)
    dropout = args_ckpt.get("dropout", 0.1)

    model = GTBADModel(
        input_dim=n_features,
        output_dim=n_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        lstm_hidden=lstm_hidden,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── 3. Load threshold ─────────────────────────────────────────────────
    threshold = load_threshold_from_json(Path(args.threshold_json))

    # ── 4. Row-by-row inference ───────────────────────────────────────────
    header = f"{'Row':<8} {'Real Label':<18} {'Score (Error)':>14}  Decision"
    sep = "-" * len(header)

    logger.info("")
    logger.info(f"  Row-by-Row Inference  |  Threshold = {threshold:.6f}")
    print(header)
    print(sep)

    for row_idx, (_, row) in enumerate(df.iterrows(), start=args.start_row + 1):
        label_val = int(row["label"])
        label_name = CLASS_NAMES.get(label_val, f"Unknown({label_val})")
        raw_row = row[feature_names].values.astype(np.float64)

        result = infer_single(model, raw_row, scaler_min, scaler_max, device)
        error = result["error"]
        decision = "ANOMALY" if error > threshold else "normal"

        print(f"{row_idx:<8} {label_name:<18} {error:>14.6f}  {decision}")

    logger.success(f"Done — {len(df):,} rows processed.")


if __name__ == "__main__":
    main()
