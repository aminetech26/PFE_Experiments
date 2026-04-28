"""
Training script for the LSTM Autoencoder model on Costa preprocessed data.

Performs:
1. Load lstmae_preprocessed.parquet
2. Filter the sequences: train ONLY on healthy data (label == 0)
3. Grid Search over [LSTM_UNITS, LATENT_DIM, LEARNING_RATE, LOOKBACK, BATCH_SIZE]
4. Extract 95th percentile MAE threshold on training data
5. Evaluate on unseen tests (healthy) and faulty data using threshold mapping limits
6. Log relevant statistical flags mappings and save best model
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime, timezone
import itertools

from src.modeling.anomaly_detection.dl.lstm_ae_model import build_lstm_ae_model

# Force TensorFlow to only log errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "lstmae"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "experiments" / "metrics"

DEFAULT_INPUT_COLS = ["irr", "idc1", "idc2", "vdc1", "vdc2"]

def create_sequences_with_labels(data: np.ndarray, labels: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences and designate window as faulty if ANY fault exists."""
    X, Y = [], []
    for i in range(len(data) - lookback + 1):
        X.append(data[i:i + lookback])
        # If any point in the lookback window is faulty (>0), label sequence as 1 (faulty)
        seq_label = 1 if np.any(labels[i:i + lookback] > 0) else 0
        Y.append(seq_label)
    return np.array(X), np.array(Y)

def get_sequence_mae(model, X_seq: np.ndarray) -> np.ndarray:
    """Calculates MAE for each sequence mapping across original shapes."""
    preds = model.predict(X_seq, verbose=0)
    # Average across time (dim 1) and features (dim 2)
    return np.mean(np.abs(preds - X_seq), axis=(1, 2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa" / "lstmae_preprocessed.parquet"),
        help="Path to LSTM-AE preprocessed parquet dataset"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per grid search trial")
    parser.add_argument("--test-split", type=float, default=0.2, help="Train/Test split ratio for healthy data")
    args = parser.parse_args()

    # Grid Search space expanded
    lstm_units_grid = [16, 32]
    latent_dim_grid = [8, 16]
    lr_grid = [0.01, 0.001]
    lookback_grid = [10, 20]
    batch_size_grid = [64, 128]
    
    # Check physical paths
    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        logger.error(f"File not found: {parquet_path}")
        logger.warning("Please run: python -m src.data.preprocess_lstmae")
        return

    logger.info(f"Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Feature Selection
    missing = [c for c in DEFAULT_INPUT_COLS + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in preprocess file (ensure 'label' is present): {missing}")

    data_array = df[DEFAULT_INPUT_COLS].values
    labels_array = df["label"].values
    n_features = len(DEFAULT_INPUT_COLS)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    best_score = float('inf')
    best_params = {}
    best_model_path = None
    results_history = []
    
    grid = list(itertools.product(lstm_units_grid, latent_dim_grid, lr_grid, lookback_grid, batch_size_grid))
    
    logger.info(f"Starting Grid Search with {len(grid)} combinations training ONLY on healthy data...")
    
    for i, (units, latent, lr, lookback, batch) in enumerate(grid):
        logger.info(f"Trial {i+1}/{len(grid)} - units:{units}, latent:{latent}, lr:{lr}, lookback:{lookback}, batch:{batch}")
        
        # Create sequences per lookback dynamically
        X_seq, Y_seq = create_sequences_with_labels(data_array, labels_array, lookback)
        
        # Split healthy (0) vs faulty (1)
        healthy_mask = (Y_seq == 0)
        X_healthy = X_seq[healthy_mask]
        X_faulty = X_seq[~healthy_mask]
        
        # Split healthy data into Train and Test sequentially
        n_healthy_samples = len(X_healthy)
        split_idx = int(n_healthy_samples * (1 - args.test_split))
        
        X_train_healthy = X_healthy[:split_idx]
        X_test_healthy = X_healthy[split_idx:]
        
        # Build Model
        model = build_lstm_ae_model(
            lookback=lookback,
            n_features=n_features,
            lstm_units_1=units,
            latent_dim=latent,
            learning_rate=lr
        )
        
        # Train Autoencoder strictly on HEALTHY data
        history = model.fit(
            X_train_healthy, X_train_healthy,
            epochs=args.epochs,
            batch_size=batch,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Learn threshold: 95th percentile of MAE on training data
        train_mae_per_seq = get_sequence_mae(model, X_train_healthy)
        threshold_95 = np.percentile(train_mae_per_seq, 95)
        
        # Evaluate Healthy Reconstruction Metrics on Unseen Test set
        X_test_pred = model.predict(X_test_healthy, verbose=0)
        test_flat = X_test_healthy.reshape(-1, n_features)
        pred_flat = X_test_pred.reshape(-1, n_features)
        
        test_mse = mean_squared_error(test_flat, pred_flat)
        test_mae = mean_absolute_error(test_flat, pred_flat)
        test_r2 = r2_score(test_flat, pred_flat)
        
        # Evaluate Statistical Flags (False Positives vs True Positives) using calculated threshold
        test_mae_per_seq = get_sequence_mae(model, X_test_healthy)
        faulty_mae_per_seq = get_sequence_mae(model, X_faulty)

        # Flag anomalies 
        fp_mask = test_mae_per_seq > threshold_95
        tp_mask = faulty_mae_per_seq > threshold_95
        
        fp_count = np.sum(fp_mask)
        tn_count = len(test_mae_per_seq) - fp_count
        
        tp_count = np.sum(tp_mask)
        fn_count = len(faulty_mae_per_seq) - tp_count
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        logger.info(f"  Result: Test MSE={test_mse:.6f}, Threshold={threshold_95:.6f} | F1={f1:.4f}, TPR(Recall)={recall:.4f}, FPR={fp_count/(fp_count+tn_count):.4f}")
        
        trial_result = {
            "lstm_units": units,
            "latent_dim": latent,
            "learning_rate": lr,
            "lookback": lookback,
            "batch_size": batch,
            "test_healthy_mse": test_mse,
            "test_healthy_mae": test_mae,
            "test_healthy_r2": test_r2,
            "learned_threshold": threshold_95,
            "true_positives": int(tp_count),
            "true_negatives": int(tn_count),
            "false_positives": int(fp_count),
            "false_negatives": int(fn_count),
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        results_history.append(trial_result)
        
        # Prioritize Best Score by combination mapping F1 score to balance false/true detection optimally. 
        # Alternatively sticking to lowering MSE error if F1 models identically 0
        score_val = f1 if f1 > 0 else (-test_mse)
        current_best_val = best_score if best_score != float('inf') else -1000

        if score_val > current_best_val or best_score == float('inf'):
            best_score = score_val
            best_params = trial_result
            
            # Save Keras Model
            checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = checkpoint_dir / "lstmae_best.keras"
            model.save(filepath=best_model_path)
            
    logger.success(f"Grid Search Complete. Best parameters: {best_params}")
    
    # Save Metrics File
    metrics_dir = Path(DEFAULT_METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "lstmae_results.json"
    
    metrics_payload = {
        "model": "LSTM_Autoencoder",
        "dataset": "Costa PV Fault Dataset",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_features": DEFAULT_INPUT_COLS,
        "grid_search_history": results_history,
        "best_parameters": best_params,
        "best_test_metrics": {
            "test_healthy_MSE": best_params["test_healthy_mse"],
            "test_healthy_MAE": best_params["test_healthy_mae"],
            "test_healthy_R2": best_params["test_healthy_r2"],
        },
        "anomaly_statistics": {
            "anomaly_threshold_95_percentile": best_params["learned_threshold"],
            "precision": best_params["precision"],
            "recall_TPR": best_params["recall"],
            "f1_score": best_params["f1_score"],
            "true_positives": best_params["true_positives"],
            "true_negatives": best_params["true_negatives"],
            "false_positives": best_params["false_positives"],
            "false_negatives": best_params["false_negatives"],
        }
    }
    
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    
    logger.success(f"Metrics saved to {metrics_path}")
    logger.success(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    main()
