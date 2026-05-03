import argparse
import json
import logging
from pathlib import Path
import os

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.modeling.anomaly_detection.dl.scvae_model import SCVAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_sequence_mae_scvae(model, X_seq_np):
    """
    Given a batch of sequences, pass forward through SCVAE and return
    the MAE across time for each sequence and feature.
    Returned shape: (N_seq, N_features)
    """
    # Scale data contextually exactly like training if necessary.
    # We assume X_seq_np is already scaled or we should scale it.
    # In train_scvae, the whole array was minmax scaled. 
    # For fair evaluation, it should be scaled globally exactly like the training script logic.
    
    X_tensor = torch.tensor(X_seq_np, dtype=torch.float32).unsqueeze(2) # (N, L, 1, F)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    model.eval()
    all_maes = []
    
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            # (batch, seq, feat_dim, input_dim) -> (seq, batch, feat_dim, input_dim)
            X_batch = X_batch.permute(1, 0, 2, 3).to(DEVICE)
            Y_batch = Y_batch.permute(1, 0, 2, 3).to(DEVICE)
            
            # Forward pass to populate self.Xr_mean
            _ = model(X_batch, Y_batch)
            
            # Reconstructions are stacked list of length `seq_len`
            # Each is (batch, input_dim * label_dim) i.e. (batch, 9)
            preds_seq = torch.stack(model.Xr_mean) # Shape: (seq_len, batch_size, input_dim)
            
            # Rearrange back to (batch_size, seq_len, input_dim)
            preds_seq = preds_seq.permute(1, 0, 2).cpu().numpy()
            
            # Original batch input was X_batch.permute(1,0,2,3) -> mapped back
            truth_seq = X_batch.permute(1, 0, 2, 3).squeeze(2).cpu().numpy()
            
            # MAE over sequence length (axis 1) => shape (batch, input_dim)
            mae = np.mean(np.abs(preds_seq - truth_seq), axis=1)
            all_maes.append(mae)
            
    return np.vstack(all_maes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=str(PROJECT_ROOT / "experiments/checkpoints/scvae/scvae_best.pth"))
    parser.add_argument("--dataset-path", type=str, default=str(PROJECT_ROOT / "data/processed/preprocessed/costa_scvae/scvae_preprocessed.parquet"))
    parser.add_argument("--metrics-path", type=str, default=str(PROJECT_ROOT / "experiments/metrics/scvae_results.json"))
    args = parser.parse_args()

    logger.info(f"Loading metrics and settings from {args.metrics_path}")
    with open(args.metrics_path, "r") as f:
        metrics = json.load(f)
    
    input_cols = metrics["input_features"]
    
    # Get hyperparameters chosen during grid search
    best_params = metrics.get("best_parameters", {'h_dim': 64, 'z_dim': 16})
    h_dim = best_params.get("h_dim", 64)
    z_dim = best_params.get("z_dim", 16)
    lookback = 50 # window_size used in train_scvae.py

    logger.info(f"Loading SCVAE model from {args.model_path}")
    model = SCVAE(
        x_dim=1,
        label_dim=1,
        h_dim=h_dim,
        z_dim=z_dim,
        input_dim=len(input_cols),
        device=DEVICE
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    logger.info(f"Loading dataset from {args.dataset_path}")
    df = pd.read_parquet(args.dataset_path)
    
    data_array = df[input_cols].values
    
    # Already preprocessed via preprocess_scvae.py (z-score standardized)
    data_scaled = np.nan_to_num(data_array).astype(np.float32, copy=False)
    
    labels_array = df["label"].values if "label" in df.columns else np.zeros(len(df))
    
    if "fault_label" in df.columns:
        fault_labels_array = df["fault_label"].values
    else:
        logger.warning("'fault_label' column missing, defaulting class splits to 'label' column directly.")
        fault_labels_array = labels_array

    logger.info("Creating sequences...")
    X_seq, Y_binary_seq, Y_multi_seq = create_eval_sequences(data_scaled, labels_array, fault_labels_array, lookback)

    # 1. Compute dynamic thresholds on ONLY HEALTHY data to establish the boundaries
    logger.info("Extracting anomaly bounds from healthy data...")
    healthy_mask = (Y_binary_seq == 0)
    X_healthy = X_seq[healthy_mask]
    
    if len(X_healthy) > 0:
        healthy_maes = get_sequence_mae_scvae(model, X_healthy)
        thresholds_dict = {
            "90th_percentile": np.percentile(healthy_maes, 90, axis=0).tolist(),
            "95th_percentile": np.percentile(healthy_maes, 95, axis=0).tolist(),
            "99th_percentile": np.percentile(healthy_maes, 99, axis=0).tolist(),
            "max": np.max(healthy_maes, axis=0).tolist(),
            "mean_plus_3std": (np.mean(healthy_maes, axis=0) + 3*np.std(healthy_maes, axis=0)).tolist()
        }
    else:
        logger.error("No healthy data available to establish thresholds!")
        return

    # 2. Keep ONLY faulty data for final evaluation prints
    faulty_mask = (Y_binary_seq == 1)
    X_faulty = X_seq[faulty_mask]
    Y_multi_faulty = Y_multi_seq[faulty_mask]

    logger.info(f"Filtered down to {len(X_faulty)} purely faulty sequences.")

    logger.info("Running inference on faulty data...")
    seq_maes_faulty = get_sequence_mae_scvae(model, X_faulty)
    
    logger.info("Calculating statistics for each feature threshold method...")
    for method, thresholds_str in thresholds_dict.items():
        logger.info(f"\n>>>> EVALUATING THRESHOLD METHOD: {method.upper()} <<<<")
        
        # Convert list to numpy array for vector broadcasting
        threshold_arr = np.array(thresholds_str)
        
        # Threshold masking: Anomaly = ANY feature exceeds its specific feature-threshold
        predicted_anomalies = (seq_maes_faulty > threshold_arr).any(axis=1).astype(int)

        # Global masks (Since all sequences left are actual anomalies/faults, i.e., actual == 1)
        tp_mask = (predicted_anomalies == 1)
        fn_mask = (predicted_anomalies == 0)

        print_class_statistics(Y_multi_faulty, tp_mask, fn_mask)

if __name__ == "__main__":
    main()