#!/usr/bin/env python3
"""
GTBAD Plus-Physics Row-by-Row Evaluation.

Loads the plus_physics checkpoint, applies the same feature engineering to
the raw Costa dataset, then computes row-by-row anomaly scores with the
trained model. Logs evaluation to MLflow → DagsHub.

Usage:
    uv run python -m src.evaluation.evaluate_gtbad_plus_physics
    uv run python -m src.evaluation.evaluate_gtbad_plus_physics --start-row 1000 --max-rows 500
    uv run python -m src.evaluation.evaluate_gtbad_plus_physics --device cuda
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

from src.data.features import add_physics_features
from src.modeling.anomaly_detection.dl.gtbad_model import GTBADModel, reconstruction_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "ingestion" / "costa" / "costa_merged.parquet"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "experiments" / "checkpoints" / "gtbad_plus_physics" / "gtbad_pp_best.pt"
DEFAULT_THRESHOLD_JSON = PROJECT_ROOT / "experiments" / "metrics" / "gtbad_plus_physics_results.json"
DEFAULT_META_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_gtbad_pp" / "gtbad_pp_metadata.json"

CLASS_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}

PLUS_PHYSICS_FLAGS = {
    "enable_delta_temp": False,
    "enable_dP_dt": True,
    "enable_dV_dt": True,
    "enable_dI_dt": True,
    "enable_Vg_normalized": False,
    "enable_power_imbalance": True,
    "enable_current_imbalance": True,
    "enable_voltage_imbalance": True,
    "enable_string_share": False,
    "enable_temp_power_correction": True,
    "temp_ref_c": 25.0,
    "gamma_pmax_pct_per_c": -0.40,
    "temp_power_eps": 1e-8,
    "irr_norm_floor": 1.0,
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
            "Run: uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_plus_physics"
        )
    with open(json_path) as f:
        data = json.load(f)
    threshold = data["anomaly_detection"]["threshold_value"]
    logger.info(f"  Loaded threshold: {threshold:.6f}")
    return float(threshold)


def load_feature_names_from_meta(meta_path: Path) -> list[str]:
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("feature_names", [])
    return []


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
        "scaled": scaled.tolist(),
        "reconstructed": recon_tensor.cpu().numpy()[0, 0, :].tolist(),
        "error": error_val,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GTBAD plus_physics row-by-row")
    parser.add_argument("--parquet-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--threshold-json", type=str, default=str(DEFAULT_THRESHOLD_JSON))
    parser.add_argument("--meta-path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── MLflow ──────────────────────────────────────────────────────────────
    if not args.no_mlflow:
        try:
            from src.mlflow_setup import init_tracking
            import mlflow
            init_tracking("anomaly")
            mlflow.start_run(run_name=f"eval_gtbad_pp_seed{args.seed}")
            mlflow.set_tags({"model": "GTBAD", "variant": "plus_physics", "mode": "evaluation"})
        except Exception as exc:
            logger.warning(f"MLflow init failed (non-fatal): {exc}")

    # ── 1. Load data ────────────────────────────────────────────────────────
    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}\nRun: uv run python -m src.data.ingestion --dataset costa")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    logger.info(f"Loaded: {len(df):,} rows from {parquet_path}")

    # ── 2. Apply plus_physics features ─────────────────────────────────────
    physics_flags = {k: v for k, v in PLUS_PHYSICS_FLAGS.items()
                     if k.startswith("enable_") or k in ("temp_ref_c", "gamma_pmax_pct_per_c", "temp_power_eps", "irr_norm_floor")}
    df = add_physics_features(df, segment_col="segment_id", time_col="timestamp", flags=physics_flags)
    logger.info(f"  Applied plus_physics features: {df.shape[1]} columns")

    # ── 3. Load checkpoint ─────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun: uv run dvc pull train_gtbad_plus_physics")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    feature_names = checkpoint.get("feature_names", [])
    if not feature_names:
        feature_names = load_feature_names_from_meta(Path(args.meta_path))
    n_features = checkpoint.get("n_features", len(feature_names))
    scaler_min = np.array(checkpoint.get("scaler_min", [0.0] * n_features), dtype=np.float32)
    scaler_max = np.array(checkpoint.get("scaler_max", [1.0] * n_features), dtype=np.float32)

    if n_features == 0 and feature_names:
        n_features = len(feature_names)

    # Handle mismatch if features drifted
    available_features = [f for f in feature_names if f in df.columns]
    if len(available_features) < n_features:
        logger.warning(f"Only {len(available_features)}/{n_features} checkpoint features found in data; using available subset")
    feature_names = available_features
    n_features = len(feature_names)

    logger.info(f"  Checkpoint: {n_features} features ({feature_names})")

    args_ckpt = checkpoint.get("args", {})
    d_model = args_ckpt.get("d_model", 64)
    nhead = args_ckpt.get("nhead", 2)
    num_encoder_layers = args_ckpt.get("num_encoder_layers", 3)
    lstm_hidden = args_ckpt.get("lstm_hidden", 32)
    dropout = args_ckpt.get("dropout", 0.1)

    model = GTBADModel(
        input_dim=n_features, output_dim=n_features,
        d_model=d_model, nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        lstm_hidden=lstm_hidden, dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── 4. Load threshold ──────────────────────────────────────────────────
    threshold = load_threshold_from_json(Path(args.threshold_json))

    # If scaler is length-mismatched, resize to match available features
    scaler_min = scaler_min[:n_features]
    scaler_max = scaler_max[:n_features]

    # ── 5. Row-by-row inference ────────────────────────────────────────────
    if args.start_row > 0:
        df = df.iloc[args.start_row:]
        logger.info(f"  Starting from row index {args.start_row}")
    if args.max_rows is not None:
        df = df.head(args.max_rows)
        logger.info(f"  Limited to {args.max_rows:,} rows")

    df_eval = df[feature_names].copy()

    header = f"{'Row':<8} {'Real Label':<18} {'Score (Error)':>14}  Decision"
    sep = "-" * len(header)

    logger.info("")
    logger.info(f"  Row-by-Row Inference  |  Threshold = {threshold:.6f}")
    print(header)
    print(sep)

    total_anomaly = 0
    total_normal = 0

    for row_idx, (idx, row) in enumerate(df_eval.iterrows(), start=args.start_row + 1):
        label_val = int(df.loc[idx, "label"])
        label_name = CLASS_NAMES.get(label_val, f"Unknown({label_val})")
        raw_row = row.values.astype(np.float64)

        result = infer_single(model, raw_row, scaler_min, scaler_max, device)
        error = result["error"]
        decision = "ANOMALY" if error > threshold else "normal"
        if decision == "ANOMALY":
            total_anomaly += 1
        else:
            total_normal += 1

        print(f"{row_idx:<8} {label_name:<18} {error:>14.6f}  {decision}")

    logger.info("")
    logger.info(f"Summary: {total_anomaly} anomalies, {total_normal} normal out of {len(df_eval):,} rows")

    # ── MLflow logging ─────────────────────────────────────────────────────
    if not args.no_mlflow and mlflow.active_run():
        try:
            mlflow.log_metrics({
                "eval_rows_processed": len(df_eval),
                "eval_anomalies_detected": total_anomaly,
                "eval_normal_detected": total_normal,
                "threshold_used": threshold,
            })
            mlflow.end_run()
            logger.success("MLflow evaluation run logged")
        except Exception as exc:
            logger.warning(f"MLflow logging failed (non-fatal): {exc}")

    logger.success(f"Done — {len(df_eval):,} rows processed.")


if __name__ == "__main__":
    main()
