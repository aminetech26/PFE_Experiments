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
    
    if "fault_label" not in df.columns and "label" in df.columns:
        logger.warning("'fault_label' column missing, defaulting class splits to 'label' column directly.")
        df["fault_label"] = df["label"]

    if "label" in df.columns:
        logger.info("Filtering dataset to keep only anomalous rows (label > 0)...")
        df = df[df["label"] > 0].copy()
        df.reset_index(drop=True, inplace=True)
    else:
        logger.error("Dataset has no 'label' column, cannot evaluate performance.")
        return
        
    # We apply transform to scale and build windows
    logger.info("Applying GTBAD preprocessing transform...")
    X_full, y_target, labels_seq, fault_labels_seq = preprocessor.transform(df, "timestamp")

    # Already filtered the dataframe, so all resulting sequences are faulty
    X_seq = X_full
    Y_multi_seq = fault_labels_seq
    
    logger.info(f"Generated {len(X_seq)} purely faulty sequences after filtering.")

    logger.info("Running inference...")
    
    # We do prediction in chunks to avoid OOM
    batch_size = 256
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for i in range(0, len(X_seq), batch_size):
            x_b = torch.tensor(X_seq[i:i+batch_size], dtype=torch.float32).to(device)
            y_b = torch.tensor(X_seq[i:i+batch_size, :, :output_dim], dtype=torch.float32).to(device)
            
            preds = model(x_b)
            # Reuses the exact reconstruction error function from training
            err = reconstruction_error(preds, y_b)
            all_errors.extend(err.cpu().numpy().tolist())

    all_errors = np.array(all_errors)

    logger.info(f"Evaluating with global threshold: {threshold:.6f}")
    predicted_anomalies = (all_errors > threshold).astype(int)

    tp_mask = (predicted_anomalies == 1)
    fn_mask = (predicted_anomalies == 0)

    print_class_statistics(Y_multi_seq, tp_mask, fn_mask)

if __name__ == "__main__":
    main()
