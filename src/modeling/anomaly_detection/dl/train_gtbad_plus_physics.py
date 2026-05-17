#!/usr/bin/env python3
"""
GTBAD Plus-Physics Training & Evaluation — Costa dataset.

Same GVSAO-Transformer-BiLSTM architecture and hyperparameter optimisation
as the baseline train_gtbad.py, but trained on the plus_physics feature set
(9 base sensors + derivatives, imbalances, temperature-corrected power).

All runs are logged to MLflow → DagsHub.

Usage:
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_plus_physics
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_plus_physics --no-gvsao
    uv run python -m src.modeling.anomaly_detection.dl.train_gtbad_plus_physics --epochs 100 --lr 0.001
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.anomaly_detection.dl.gtbad_model import GTBADModel, reconstruction_error
from src.modeling.anomaly_detection.dl.gvsao import GVSaoConfig, GVSaoResult, run_gvsao
from src.mlflow_setup import init_tracking

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_gtbad_pp" / "gtbad_pp_data.npz"
DEFAULT_META_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_gtbad_pp" / "gtbad_pp_metadata.json"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "gtbad_plus_physics"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "experiments" / "metrics"

FAULT_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}

EVALUABLE_CLASSES = [1, 2, 3, 4]


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)


def load_preprocessed(npz_path: Path, meta_path: Path) -> dict[str, Any]:
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {npz_path}\n"
            "Run: uv run python -m src.data.preprocess_gtbad_plus_physics"
        )
    data = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "X_train": torch.from_numpy(data["X_train"].astype(np.float32)),
        "X_val": torch.from_numpy(data["X_val"].astype(np.float32)),
        "X_test": torch.from_numpy(data["X_test"].astype(np.float32)),
        "labels_train": data["labels_train"].astype(np.int32),
        "labels_val": data["labels_val"].astype(np.int32),
        "labels_test": data["labels_test"].astype(np.int32),
        "n_features": int(data["X_train"].shape[2]),
        "feature_names": meta.get("feature_names", []),
    }


def _make_dataloader(X: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(X, X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, _ in dataloader:
        X_batch = X_batch.to(device)
        optimizer.zero_grad()
        recon = model(X_batch)
        loss = criterion(recon, X_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_reconstruction(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    all_errors: list[np.ndarray] = []
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[start:end].to(device)
        recon = model(X_batch)
        err = reconstruction_error(X_batch, recon).cpu().numpy()
        all_errors.append(err)
    return np.concatenate(all_errors)


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    device: torch.device,
    lr: float,
    batch_size: int,
    epochs: int,
    patience: int = 15,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    train_loader = _make_dataloader(X_train, batch_size, shuffle=True)
    val_loader = _make_dataloader(X_val, batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                recon = model(X_batch)
                val_loss += criterion(recon, X_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            logger.info(f"    Epoch {epoch+1:3d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if epochs_without_improvement >= patience:
            if verbose:
                logger.info(f"    Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    training_info = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val_loss),
        "epochs_trained": len(train_losses),
    }
    return model, training_info


def compute_threshold(errors: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(errors, percentile))


def evaluate_anomaly_detection(
    errors: np.ndarray, labels: np.ndarray, threshold: float,
) -> dict[str, float]:
    preds = (errors > threshold).astype(int)
    true = (labels > 0).astype(int)
    tp = int(np.sum((preds == 1) & (true == 1)))
    fp = int(np.sum((preds == 1) & (true == 0)))
    fn = int(np.sum((preds == 0) & (true == 1)))
    tn = int(np.sum((preds == 0) & (true == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "threshold": threshold,
    }


def build_model(n_features: int, d_model: int, nhead: int, num_encoder_layers: int, lstm_hidden: int, dropout: float) -> GTBADModel:
    return GTBADModel(
        input_dim=n_features, output_dim=n_features,
        d_model=d_model, nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        lstm_hidden=lstm_hidden, dropout=dropout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GTBAD plus_physics on Costa")
    parser.add_argument("--npz-path", type=str, default=str(DEFAULT_NPZ_PATH))
    parser.add_argument("--meta-path", type=str, default=str(DEFAULT_META_PATH))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gvsao-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-gvsao", action="store_true")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--lstm-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ── Init MLflow ──────────────────────────────────────────────────────────
    run_name = f"gtbad_plus_physics_seed{args.seed}"
    if not args.no_mlflow:
        try:
            init_tracking("anomaly")
            mlflow.start_run(run_name=run_name)
            mlflow.set_tags({
                "model": "GTBAD",
                "variant": "plus_physics",
                "dataset": "costa",
                "seed": str(args.seed),
            })
            logger.info("MLflow tracking active")
        except Exception as exc:
            logger.warning(f"MLflow init failed (non-fatal): {exc}")

    # ── Load preprocessed data ──────────────────────────────────────────────
    logger.info("=== Step 1: Load plus_physics preprocessed data ===")
    npz_path = Path(args.npz_path)
    meta_path = Path(args.meta_path)
    tensors = load_preprocessed(npz_path, meta_path)
    n_features = tensors["n_features"]
    feature_names = tensors["feature_names"]
    X_train = tensors["X_train"]
    X_val = tensors["X_val"]
    X_test = tensors["X_test"]
    labels_test = tensors["labels_test"]
    logger.info(f"  n_features: {n_features} ({feature_names[:5]}...)")
    logger.info(f"  X_train: {tuple(X_train.shape)} | X_val: {tuple(X_val.shape)} | X_test: {tuple(X_test.shape)}")

    # ── GVSAO HPO ────────────────────────────────────────────────────────────
    final_lr = args.lr or 0.001
    final_batch_size = args.batch_size or 32
    gvsao_result = None

    if not args.no_gvsao and (args.lr is None or args.batch_size is None):
        logger.info("=== Step 2: GVSAO Hyperparameter Tuning ===")
        gvsao_config = GVSaoConfig(
            population_size=10, max_generations=5,
            lr_bounds=(1e-5, 1e-1), batch_bounds=(16, 128),
            seed=args.seed,
        )

        def fitness_fn(lr: float, batch: int) -> float:
            model = build_model(n_features, args.d_model, args.nhead,
                              args.num_encoder_layers, args.lstm_hidden, args.dropout).to(device)
            effective_batch = min(batch, X_train.shape[0])
            _, info = train_model(model, X_train, X_val, device,
                                 lr=lr, batch_size=effective_batch,
                                 epochs=args.gvsao_epochs, patience=3, verbose=False)
            return info["best_val_loss"]

        gvsao_result = run_gvsao(fitness_fn, gvsao_config, verbose=True)
        final_lr = gvsao_result.best_params["learning_rate"]
        final_batch_size = gvsao_result.best_params["batch_size"]
        final_batch_size = min(final_batch_size, X_train.shape[0])
        logger.success(f"GVSAO best: lr={final_lr:.6f}, batch={final_batch_size}")

    # ── Train final model ───────────────────────────────────────────────────
    logger.info(f"=== Step 3: Train GTBAD (lr={final_lr:.6f}, batch={final_batch_size}) ===")
    model = build_model(n_features, args.d_model, args.nhead,
                       args.num_encoder_layers, args.lstm_hidden, args.dropout).to(device)

    t0 = time.perf_counter()
    model, training_info = train_model(model, X_train, X_val, device,
                                       lr=final_lr, batch_size=final_batch_size,
                                       epochs=args.epochs, patience=args.patience, verbose=True)
    train_time = time.perf_counter() - t0
    logger.info(f"  Training completed in {train_time:.1f}s")

    # ── Checkpoint ──────────────────────────────────────────────────────────
    checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "gtbad_pp_best.pt"
    meta_path_src = Path(args.meta_path)
    scaler_min = []
    scaler_max = []
    if meta_path_src.exists():
        meta = json.loads(meta_path_src.read_text(encoding="utf-8"))
        scaler_min = meta.get("scaler_min", [])
        scaler_max = meta.get("scaler_max", [])
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": n_features,
        "feature_names": feature_names,
        "scaler_min": scaler_min,
        "scaler_max": scaler_max,
        "args": vars(args),
    }, ckpt_path)
    logger.success(f"  Saved checkpoint → {ckpt_path}")

    # ── Anomaly Detection Evaluation ────────────────────────────────────────
    logger.info("=== Step 4: Anomaly Detection Evaluation ===")
    train_errors = evaluate_reconstruction(model, X_train, device, final_batch_size)
    threshold = compute_threshold(train_errors, args.threshold_percentile)
    logger.info(f"  Anomaly threshold ({args.threshold_percentile}th pctl): {threshold:.6f}")

    val_errors = evaluate_reconstruction(model, X_val, device, final_batch_size)
    val_fp = int(np.sum(val_errors > threshold))
    val_result = evaluate_anomaly_detection(val_errors, tensors["labels_val"], threshold)
    logger.info(f"  Val (healthy): FP={val_fp}/{len(val_errors)} ({100*val_fp/len(val_errors):.2f}%)")

    class_results: dict[str, dict] = {}
    for cls in EVALUABLE_CLASSES:
        cls_mask = labels_test == cls
        if int(cls_mask.sum()) == 0:
            logger.warning(f"  No data for fault class {cls}")
            continue
        X_fault = X_test[cls_mask]
        labels_fault = labels_test[cls_mask]
        fault_errors = evaluate_reconstruction(model, X_fault, device, final_batch_size)
        result = evaluate_anomaly_detection(fault_errors, labels_fault, threshold)
        class_results[f"fault_class_{cls}"] = result
        logger.info(
            f"  {FAULT_NAMES[cls]:<14} "
            f"Precision={result['precision']:.4f} "
            f"Recall={result['recall']:.4f} "
            f"F1={result['f1_score']:.4f} "
            f"TP={result['TP']} FP={result['FP']} FN={result['FN']}"
        )

    # ── Overall ─────────────────────────────────────────────────────────────
    overall_result: dict[str, float] = {}
    all_fault_mask = np.isin(labels_test, EVALUABLE_CLASSES)
    if int(all_fault_mask.sum()) > 0:
        X_all_fault = X_test[all_fault_mask]
        labels_all_fault = labels_test[all_fault_mask]
        all_fault_errors = evaluate_reconstruction(model, X_all_fault, device, final_batch_size)
        combined_val = np.concatenate([val_errors, all_fault_errors])
        combined_label = np.concatenate([np.zeros(len(val_errors)), labels_all_fault])
        overall_result = evaluate_anomaly_detection(combined_val, combined_label, threshold)
        logger.info(
            f"  {'OVERALL':<14} "
            f"Precision={overall_result['precision']:.4f} "
            f"Recall={overall_result['recall']:.4f} "
            f"F1={overall_result['f1_score']:.4f} "
            f"TP={overall_result['TP']} FP={overall_result['FP']} FN={overall_result['FN']}"
        )

    # ── Save Results ───────────────────────────────────────────────────────
    metrics_dir = Path(DEFAULT_METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_payload = {
        "model": "GTBAD Plus-Physics (GVSAO-Transformer-BiLSTM)",
        "dataset": "Costa PV Fault Dataset",
        "feature_profile": "plus_physics",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_features": feature_names,
        "n_features": n_features,
        "model_config": {
            "d_model": args.d_model, "nhead": args.nhead,
            "num_encoder_layers": args.num_encoder_layers,
            "lstm_hidden": args.lstm_hidden, "dropout": args.dropout,
        },
        "training": {
            "learning_rate": final_lr,
            "batch_size": final_batch_size,
            "epochs": training_info["epochs_trained"],
            "best_val_loss": training_info["best_val_loss"],
            "train_time_seconds": round(train_time, 1),
            "seed": args.seed,
        },
        "gvsao": {
            "enabled": gvsao_result is not None,
            "best_params": gvsao_result.best_params if gvsao_result else None,
            "best_fitness": gvsao_result.best_fitness if gvsao_result else None,
            "history": gvsao_result.history if gvsao_result else None,
            "n_evals": gvsao_result.n_evals if gvsao_result else 0,
        },
        "anomaly_detection": {
            "threshold_percentile": args.threshold_percentile,
            "threshold_value": threshold,
            "val_healthy_fp_rate": float(val_fp / len(val_errors)) if len(val_errors) > 0 else 0.0,
            "per_class": {k: v for k, v in class_results.items()},
            "overall": overall_result,
        },
    }

    results_path = metrics_dir / "gtbad_plus_physics_results.json"
    results_path.write_text(json.dumps(results_payload, indent=2, default=str), encoding="utf-8")
    logger.success(f"  Results saved → {results_path}")

    # ── MLflow logging ─────────────────────────────────────────────────────
    if not args.no_mlflow and mlflow.active_run():
        try:
            mlflow.log_params({
                "model_type": "GTBAD",
                "variant": "plus_physics",
                "n_features": n_features,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_encoder_layers": args.num_encoder_layers,
                "lstm_hidden": args.lstm_hidden,
                "dropout": args.dropout,
                "learning_rate": final_lr,
                "batch_size": final_batch_size,
                "epochs_trained": training_info["epochs_trained"],
                "patience": args.patience,
                "threshold_percentile": args.threshold_percentile,
                "gvsao_enabled": gvsao_result is not None,
                "seed": args.seed,
                "train_frac": 0.80,
            })

            mlflow_metrics: dict[str, float] = {
                "best_val_loss": training_info["best_val_loss"],
                "train_time_seconds": round(train_time, 1),
                "anomaly_threshold": threshold,
                "val_healthy_fp_rate": float(val_fp / len(val_errors)) if len(val_errors) > 0 else 0.0,
            }
            for cls_name, cls_res in class_results.items():
                mlflow_metrics[f"{cls_name}_precision"] = cls_res["precision"]
                mlflow_metrics[f"{cls_name}_recall"] = cls_res["recall"]
                mlflow_metrics[f"{cls_name}_f1"] = cls_res["f1_score"]
            if overall_result:
                mlflow_metrics["overall_precision"] = overall_result["precision"]
                mlflow_metrics["overall_recall"] = overall_result["recall"]
                mlflow_metrics["overall_f1"] = overall_result["f1_score"]
            mlflow.log_metrics(mlflow_metrics)

            if ckpt_path.exists():
                mlflow.log_artifact(str(ckpt_path))
            if results_path.exists():
                mlflow.log_artifact(str(results_path))

            run_id = mlflow.active_run().info.run_id
            logger.success(f"MLflow run logged: {run_name} [{run_id}]")
            mlflow.end_run()
        except Exception as exc:
            logger.warning(f"MLflow logging failed (non-fatal): {exc}")

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.success("=" * 60)
    logger.success("GTBAD Plus-Physics Training & Evaluation Complete")
    if overall_result:
        logger.success(f"  Overall F1: {overall_result['f1_score']:.4f}")
        logger.success(f"  Overall Precision: {overall_result['precision']:.4f}")
        logger.success(f"  Overall Recall: {overall_result['recall']:.4f}")
    logger.success(f"  Features: {n_features}")
    logger.success(f"  Checkpoint: {ckpt_path}")
    logger.success(f"  Metrics: {results_path}")
    logger.success("=" * 60)


if __name__ == "__main__":
    main()
