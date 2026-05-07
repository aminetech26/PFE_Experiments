import argparse
import json
import logging
from pathlib import Path
import os
import pickle

# Fix matplotlib backend in headless/Colab environments
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.modeling.anomaly_detection.dl.gtbad_model import GTBADModel, reconstruction_error

def print_class_statistics(labels, tp_mask, fn_mask):
    """
    Prints recall and counts for each distinct anomaly class (from labels column).
    Only evaluates faulty sequences (label > 0), so TN and FP are excluded.
    """
    unique_classes = [c for c in np.unique(labels) if c != 0]
    
    logger.info("=" * 60)
    logger.info(f"{'Class':<8} | {'Total':<9} | {'Detected (TP)':<14} | {'Missed (FN)':<12} | {'Recall':<7}")
    logger.info("-" * 60)
    
    global_tp = np.sum(tp_mask)
    global_fn = np.sum(fn_mask)
    global_total = global_tp + global_fn
    
    for cls in unique_classes:
        cls_mask = (labels == cls)
        
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
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_ROOT / "experiments/checkpoints/gtbad"))
    parser.add_argument("--dataset-path", type=str, default=str(PROJECT_ROOT / "data/interim/ingestion/costa/costa_merged.parquet"))
    args = parser.parse_args()

    checkpoint_dir = Path(args.model_dir)

    logger.info("Loading preprocessor and thresholds...")
    with open(checkpoint_dir / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
        
    with open(checkpoint_dir / "threshold.json", "r") as f:
        threshold_data = json.load(f)
        
    threshold = threshold_data["threshold"]
    input_dim = threshold_data["input_dim"]
    output_dim = threshold_data["output_dim"]

    logger.info(f"Loading GTBAD model from {checkpoint_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GTBADModel(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=64,
        nhead=2,
        num_encoder_layers=3,
        lstm_hidden=32,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_dir / "gtbad_best.pt", map_location=device))
    model.eval()

    logger.info(f"Loading dataset from {args.dataset_path}")
    _, ext = os.path.splitext(args.dataset_path)
    if ext.lower() == ".parquet":
        df = pd.read_parquet(args.dataset_path)
    else:
        df = pd.read_csv(args.dataset_path)
        
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Check for label column (actual anomaly labels in dataset)
    if "label" not in df.columns:
        logger.error("Dataset has no 'label' column, cannot evaluate performance.")
        return
    
    logger.info("Filtering dataset to keep only anomalous rows (label > 0) for evaluation...")
    df_anomaly = df[df["label"] > 0].copy()
    df_anomaly.reset_index(drop=True, inplace=True)
    
    if len(df_anomaly) == 0:
        logger.warning("No anomalous samples found in dataset. Evaluation skipped.")
        return
    
    # Get the original labels before preprocessing (to track anomaly classes)
    original_labels = df_anomaly["label"].values
    
    # We apply transform to scale and build windows
    logger.info("Applying GTBAD preprocessing transform...")
    X_full, y_target, mask, df_processed = preprocessor.transform(df_anomaly, "timestamp")

    # Map sliding window indices back to original labels
    # For each window, we track the maximum label value in that window (to classify the window as anomalous)
    window_labels = []
    window_len = preprocessor.window_len
    n_samples = len(X_full)
    
    for idx in range(n_samples):
        # Each sample corresponds to window at position (window_len - 1 + idx)
        window_end = window_len - 1 + idx
        window_start = window_end - window_len + 1
        
        # Get the labels for this window
        window_label_values = original_labels[window_start:window_end + 1]
        # Use max label in window (if any value > 0, window is anomalous)
        max_label = np.max(window_label_values) if len(window_label_values) > 0 else 0
        window_labels.append(max_label)
    
    window_labels = np.array(window_labels)
    
    logger.info(f"Generated {len(X_full)} window samples from {len(df_anomaly)} anomalous timesteps.")
    logger.info(f"Window label distribution: {np.unique(window_labels, return_counts=True)}")

    logger.info("Running inference...")
    
    # We do prediction in chunks to avoid OOM
    batch_size = 256
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for i in range(0, len(X_full), batch_size):
            x_b = torch.tensor(X_full[i:i+batch_size], dtype=torch.float32).to(device)
            # Target is the numeric features (first output_dim columns of X_full)
            y_b = torch.tensor(X_full[i:i+batch_size, :, :output_dim], dtype=torch.float32).to(device)
            
            preds = model(x_b)
            # Reuses the exact reconstruction error function from training
            err = reconstruction_error(preds, y_b)
            all_errors.extend(err.cpu().numpy().tolist())

    all_errors = np.array(all_errors)

    logger.info(f"Evaluating with global threshold: {threshold:.6f}")
    predicted_anomalies = (all_errors > threshold).astype(int)

    # TP: predicted as anomaly and actually anomalous (label > 0)
    # FN: predicted as normal but actually anomalous (label > 0)
    tp_mask = (predicted_anomalies == 1) & (window_labels > 0)
    fn_mask = (predicted_anomalies == 0) & (window_labels > 0)

    print_class_statistics(window_labels, tp_mask, fn_mask)

if __name__ == "__main__":
    main()
