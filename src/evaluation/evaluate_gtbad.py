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
from sklearn.metrics import average_precision_score

from src.modeling.anomaly_detection.dl.gtbad_model import GTBADModel, reconstruction_error
from src.modeling.common.artifact_contract import compute_anomaly_per_class_metrics
from src.modeling.common.episode_metrics import episode_macro_f1_binary

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


def _add_group_columns(df: pd.DataFrame, gap_seconds: int = 300) -> pd.DataFrame:
    """Add episode_id and operating_day_id columns for episode-level metrics."""
    if "episode_id" not in df.columns:
        if df.index.name == "timestamp":
            dt_s = df.index.to_series().diff().dt.total_seconds().fillna(0)
        else:
            dt_s = pd.to_datetime(df["timestamp"]).diff().dt.total_seconds().fillna(0)
        label_change = df["label"].diff().fillna(0) != 0
        df["episode_id"] = ((dt_s > gap_seconds) | label_change).cumsum().astype(int)

    if "operating_day_id" not in df.columns:
        if df.index.name == "timestamp":
            operating_day = df.index.to_series().dt.date.astype(str)
        else:
            operating_day = pd.to_datetime(df["timestamp"]).dt.date.astype(str)
        df["operating_day_id"] = pd.factorize(operating_day, sort=True)[0].astype(int)

    return df


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
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = _add_group_columns(df)
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

    all_errors: list[float] = []
    all_labels: list[int] = []
    all_group_ids: list[str] = []
    group_col = "episode_id" if "episode_id" in df.columns else None

    for row_idx, (_, row) in enumerate(df.iterrows(), start=args.start_row + 1):
        label_val = int(row["label"])
        label_name = CLASS_NAMES.get(label_val, f"Unknown({label_val})")
        raw_row = row[feature_names].values.astype(np.float64)

        result = infer_single(model, raw_row, scaler_min, scaler_max, device)
        error = result["error"]
        decision = "ANOMALY" if error > threshold else "normal"

        all_errors.append(error)
        all_labels.append(label_val)
        if group_col:
            all_group_ids.append(str(row[group_col]))

        print(f"{row_idx:<8} {label_name:<18} {error:>14.6f}  {decision}")

    logger.success(f"Done — {len(df):,} rows processed.")

    # ── 5. Aggregate metrics ──────────────────────────────────────────────
    errors_arr = np.array(all_errors)
    labels_arr = np.array(all_labels)
    true_bin = (labels_arr > 0).astype(int)
    preds_bin = (errors_arr > threshold).astype(int)

    EVALUABLE_CLASSES = [1, 2, 3, 4]

    # Overall metrics
    tp = int(np.sum((preds_bin == 1) & (true_bin == 1)))
    fp = int(np.sum((preds_bin == 1) & (true_bin == 0)))
    fn = int(np.sum((preds_bin == 0) & (true_bin == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    pr_auc = float(average_precision_score(true_bin, errors_arr)) if len(np.unique(true_bin)) > 1 else 0.0

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE METRICS")
    logger.info("=" * 60)
    logger.info(f"  Overall Precision: {precision:.4f}")
    logger.info(f"  Overall Recall:    {recall:.4f}")
    logger.info(f"  Overall F1:        {f1:.4f}")
    logger.info(f"  Overall PR-AUC:    {pr_auc:.4f}")

    # Per-class metrics
    per_class_pr_auc: list[float] = []
    per_class_f1: list[float] = []
    per_class_recall: list[float] = []
    per_class_precision: list[float] = []

    logger.info("\n  ── Per Class ──")
    logger.info(f"  {'Class':<14} {'N':>8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'PR-AUC':>10}")

    for cls in EVALUABLE_CLASSES:
        mask = (labels_arr == cls)
        n_fault = int(mask.sum())
        if n_fault == 0:
            continue

        # Per-class: combine healthy (label==0) with this fault class
        healthy_mask = (labels_arr == 0)
        pc_errors = np.concatenate([errors_arr[healthy_mask], errors_arr[mask]])
        pc_labels = np.concatenate([np.zeros(healthy_mask.sum()), labels_arr[mask]])
        pc_bin = (pc_labels > 0).astype(int)
        pc_preds = (pc_errors > threshold).astype(int)

        pc_tp = int(np.sum((pc_preds == 1) & (pc_bin == 1)))
        pc_fp = int(np.sum((pc_preds == 1) & (pc_bin == 0)))
        pc_fn = int(np.sum((pc_preds == 0) & (pc_bin == 1)))
        pc_prec = pc_tp / (pc_tp + pc_fp) if (pc_tp + pc_fp) > 0 else 0.0
        pc_rec = pc_tp / (pc_tp + pc_fn) if (pc_tp + pc_fn) > 0 else 0.0
        pc_f1 = 2 * pc_prec * pc_rec / (pc_prec + pc_rec) if (pc_prec + pc_rec) > 0 else 0.0
        pc_pr = float(average_precision_score(pc_bin, pc_errors)) if len(np.unique(pc_bin)) > 1 else 0.0

        per_class_pr_auc.append(pc_pr)
        per_class_f1.append(pc_f1)
        per_class_recall.append(pc_rec)
        per_class_precision.append(pc_prec)

        logger.info(
            f"  {CLASS_NAMES[cls]:<14} {n_fault:>8} {pc_prec:>10.4f} {pc_rec:>10.4f} "
            f"{pc_f1:>10.4f} {pc_pr:>10.4f}"
        )

    # Macro averages
    n_cls = len(per_class_pr_auc)
    if n_cls > 0:
        logger.info("\n  ── Macro Averages ──")
        logger.info(f"  Macro PR-AUC:    {np.mean(per_class_pr_auc):.4f}")
        logger.info(f"  Macro F1:        {np.mean(per_class_f1):.4f}")
        logger.info(f"  Macro Recall:    {np.mean(per_class_recall):.4f}")
        logger.info(f"  Macro Precision: {np.mean(per_class_precision):.4f}")

    # ── Standard contract: per-class metrics ──────────────────────────────
    per_class_metrics = compute_anomaly_per_class_metrics(
        labels=labels_arr,
        scores=errors_arr,
        threshold=threshold,
        normal_label=0,
    )
    worst_class_pr_auc = None
    pr_aucs = [
        float(m["pr_auc_vs_normal"])
        for m in per_class_metrics.values()
        if m.get("pr_auc_vs_normal") is not None
    ]
    if pr_aucs:
        worst_class_pr_auc = round(float(min(pr_aucs)), 6)

    logger.info("\n  ── Per-Class (contract) ──")
    for cls_key, m in per_class_metrics.items():
        logger.info(
            f"    {CLASS_NAMES.get(int(cls_key), cls_key):<14} "
            f"PR-AUC={m.get('pr_auc_vs_normal', 'N/A')} "
            f"F1@thr={m.get('f1_at_threshold_vs_normal', 'N/A')} "
            f"Prec@thr={m.get('precision_at_threshold_vs_normal', 'N/A')} "
            f"Rec@thr={m.get('recall_at_threshold_vs_normal', 'N/A')} "
            f"support={m.get('support_fault', 0)}"
        )

    # ── Episode-level macro F1 ────────────────────────────────────────────
    group_ids_arr = np.array(all_group_ids) if all_group_ids else None
    test_episode_macro_f1 = episode_macro_f1_binary(true_bin, preds_bin, group_ids_arr)
    logger.info(f"\n  Test Episode Macro F1: {test_episode_macro_f1:.4f}")

    if worst_class_pr_auc is not None:
        logger.info(f"  Worst-class PR-AUC:    {worst_class_pr_auc:.4f}")

    # Label distribution (frequency)
    logger.info("\n  ── Label Distribution ──")
    for lbl in sorted(CLASS_NAMES.keys()):
        cnt = int((labels_arr == lbl).sum())
        if cnt > 0:
            logger.info(f"  {CLASS_NAMES[lbl]:<14} {cnt:>8}")


if __name__ == "__main__":
    main()
