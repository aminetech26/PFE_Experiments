import argparse
import json
import logging
from pathlib import Path
import os

# Fix matplotlib backend in headless/Colab environments before any keras imports
os.environ['MPLBACKEND'] = 'Agg'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def create_eval_sequences(data: np.ndarray, labels: np.ndarray, fault_labels: np.ndarray, lookback: int):
    """
    Create sliding window sequences. Assigns a sequence label=1 if any point in the window is anomalous.
    Assigns the bounding fault_label based on the prominent fault inside the window.
    """
    X, Y_binary, Y_multi = [], [], []
    for i in range(len(data) - lookback + 1):
        X.append(data[i:i + lookback])
        win_labels = labels[i:i + lookback]
        win_faults = fault_labels[i:i + lookback]
        
        seq_label = 1 if np.any(win_labels > 0) else 0
        
        faults_only = win_faults[win_faults > 0]
        if len(faults_only) > 0:
            vals, counts = np.unique(faults_only, return_counts=True)
            seq_fault_label = int(vals[np.argmax(counts)])
        else:
            seq_fault_label = 0
            
        Y_binary.append(seq_label)
        Y_multi.append(seq_fault_label)
        
    return np.array(X), np.array(Y_binary), np.array(Y_multi)

def print_class_statistics(seq_multi, tp_mask, fn_mask):
    """
    Prints recall and counts for each distinct fault_label class.
    Only evaluates faulty sequences, so TN and FP are excluded.
    """
    unique_classes = [c for c in np.unique(seq_multi) if c != 0]
    
    logger.info("=" * 60)
    logger.info(f"{'Class':<8} | {'Total':<9} | {'Detected (TP)':<14} | {'Missed (FN)':<12} | {'Recall':<7}")
    logger.info("-" * 60)
    
    global_tp = np.sum(tp_mask)
    global_fn = np.sum(fn_mask)
    global_total = global_tp + global_fn
    
    for cls in unique_classes:
        cls_mask = (seq_multi == cls)
        
        c_tp = np.sum(tp_mask & cls_mask)
        c_fn = np.sum(fn_mask & cls_mask)
        c_total = c_tp + c_fn
        
        c_rec = c_tp / c_total if c_total > 0 else 0.0
        
        logger.info(f"{cls:<8} | {c_total:<9} | {c_tp:<14} | {c_fn:<12} | {c_rec:<7.4f}")
        
    logger.info("=" * 60)
    
    g_rec = global_tp / global_total if global_total > 0 else 0.0
    logger.info(f"{'ALL':<8} | {global_total:<9} | {global_tp:<14} | {global_fn:<12} | {g_rec:<7.4f}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(PROJECT_ROOT / "experiments/checkpoints/lstmae/lstmae_best.keras"))
    parser.add_argument("--dataset-path", type=str, default=str(PROJECT_ROOT / "data/processed/preprocessed/costa_lstmae/lstmae_preprocessed.parquet"))
    parser.add_argument("--metrics-path", type=str, default=str(PROJECT_ROOT / "experiments/metrics/lstmae_results.json"))
    args = parser.parse_args()

    logger.info(f"Loading metrics and settings from {args.metrics_path}")
    with open(args.metrics_path, "r") as f:
        metrics = json.load(f)
    
    threshold = metrics["anomaly_statistics"]["anomaly_threshold_95_percentile"]
    input_cols = metrics["input_features"]
    lookback = metrics["best_parameters"]["lookback"]
    logger.info(f"Loaded Threshold: {threshold:.6f}, Lookback: {lookback}")

    logger.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    logger.info(f"Loading dataset from {args.dataset_path}")
    df = pd.read_parquet(args.dataset_path)
    
    data_array = df[input_cols].values
    labels_array = df["label"].values if "label" in df.columns else np.zeros(len(df))
    
    # Check if fault_label is there, else mock it
    if "fault_label" in df.columns:
        fault_labels_array = df["fault_label"].values
    else:
        logger.warning("'fault_label' column missing, defaulting class splits to 'label' column directly.")
        fault_labels_array = labels_array

    logger.info("Creating sequences...")
    X_seq, Y_binary_seq, Y_multi_seq = create_eval_sequences(data_array, labels_array, fault_labels_array, lookback)

    # Keep ONLY faulty data for this evaluation
    faulty_mask = (Y_binary_seq == 1)
    X_seq = X_seq[faulty_mask]
    Y_binary_seq = Y_binary_seq[faulty_mask]
    Y_multi_seq = Y_multi_seq[faulty_mask]

    logger.info(f"Filtered down to {len(X_seq)} purely faulty sequences.")

    logger.info("Running inference...")
    preds = model.predict(X_seq, batch_size=256)
    
    # Calculate MAE over time and feature dimensions
    seq_maes = np.mean(np.abs(preds - X_seq), axis=(1, 2))
    
    logger.info("Calculating statistics...")
    # Threshold masking
    predicted_anomalies = (seq_maes > threshold).astype(int)

    # Global masks (Since all sequences left are actual anomalies/faults, i.e., actual == 1)
    tp_mask = (predicted_anomalies == 1)
    fn_mask = (predicted_anomalies == 0)

    print_class_statistics(Y_multi_seq, tp_mask, fn_mask)


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
