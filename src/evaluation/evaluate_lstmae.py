import argparse
import json
import logging
from pathlib import Path
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

def print_class_statistics(seq_multi, tp_mask, fp_mask, fn_mask, tn_mask):
    """
    Prints precision/recall and counts for each distinct fault_label class 
    by slicing out the respective counts.
    """
    unique_classes = np.unique(seq_multi)
    
    logger.info("=" * 60)
    logger.info(f"{'Class':<8} | {'Type':<8} | {'TP':<6} | {'TN':<6} | {'FP':<6} | {'FN':<6} | {'Prec':<7} | {'Recall':<7} | {'F1':<7}")
    logger.info("-" * 60)
    
    global_tp = np.sum(tp_mask)
    global_tn = np.sum(tn_mask)
    global_fp = np.sum(fp_mask)
    global_fn = np.sum(fn_mask)
    
    for cls in unique_classes:
        cls_mask = (seq_multi == cls)
        
        # For class 0 (Healthy), its success is TN, failure is FP. TP/FN don't apply semantically to 'healthy' in Anomaly Detection as it's the negative class.
        # But we can still count them locally (if seq_multi == 0, true label is 0)
        c_tp = np.sum(tp_mask & cls_mask)
        c_tn = np.sum(tn_mask & cls_mask)
        c_fp = np.sum(fp_mask & cls_mask)
        c_fn = np.sum(fn_mask & cls_mask)
        
        c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
        
        type_str = "Healthy" if cls == 0 else "Fault"
        
        logger.info(f"{cls:<8} | {type_str:<8} | {c_tp:<6} | {c_tn:<6} | {c_fp:<6} | {c_fn:<6} | {c_prec:<7.4f} | {c_rec:<7.4f} | {c_f1:<7.4f}")
        
    logger.info("=" * 60)
    
    g_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    g_rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    g_f1 = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0
    logger.info(f"{'ALL':<8} | {'Overall':<8} | {global_tp:<6} | {global_tn:<6} | {global_fp:<6} | {global_fn:<6} | {g_prec:<7.4f} | {g_rec:<7.4f} | {g_f1:<7.4f}")
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

    logger.info("Running inference...")
    preds = model.predict(X_seq, batch_size=256)
    
    # Calculate MAE over time and feature dimensions
    seq_maes = np.mean(np.abs(preds - X_seq), axis=(1, 2))
    
    logger.info("Calculating statistics...")
    # Threshold masking
    predicted_anomalies = (seq_maes > threshold).astype(int)
    actual_anomalies = Y_binary_seq

    # Global masks
    tp_mask = (predicted_anomalies == 1) & (actual_anomalies == 1)
    tn_mask = (predicted_anomalies == 0) & (actual_anomalies == 0)
    fp_mask = (predicted_anomalies == 1) & (actual_anomalies == 0)
    fn_mask = (predicted_anomalies == 0) & (actual_anomalies == 1)

    print_class_statistics(Y_multi_seq, tp_mask, fp_mask, fn_mask, tn_mask)


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
