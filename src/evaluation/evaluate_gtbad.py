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


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from src.modeling.anomaly_detection.dl.gtbad_model import GTBADModel

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
    raw_pft = threshold_data.get("per_feature_thresholds")
    if raw_pft is None or len(raw_pft) != output_dim:
        logger.warning("per_feature_thresholds not found in checkpoint; falling back to global threshold")
        per_feature_thresholds = None
    else:
        per_feature_thresholds = np.array(raw_pft)

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
    
    logger.info("Applying GTBAD preprocessing transform on full dataset...")
    X_full, y_target, mask, df_processed = preprocessor.transform(df, "timestamp")

    # Map sliding window indices back to original labels
    original_labels = df["label"].values
    window_labels = []
    window_len = preprocessor.window_len
    n_samples = len(X_full)
    
    for idx in range(n_samples):
        window_end = window_len - 1 + idx
        window_start = window_end - window_len + 1
        window_label_values = original_labels[window_start:window_end + 1]
        max_label = np.max(window_label_values) if len(window_label_values) > 0 else 0
        window_labels.append(max_label)
    
    window_labels = np.array(window_labels)
    
    n_anom = np.sum(window_labels > 0)
    logger.info(f"Generated {len(X_full)} total window samples ({n_anom} anomalous, {len(X_full) - n_anom} healthy).")
    if n_anom > 0:
        logger.info(f"Anomalous window label distribution: {dict(zip(*np.unique(window_labels[window_labels > 0], return_counts=True)))}")

    logger.info("Running inference...")
    batch_size = 256
    model.eval()
    all_feature_errors = []
    
    with torch.no_grad():
        for i in range(0, len(X_full), batch_size):
            x_b = torch.tensor(X_full[i:i+batch_size], dtype=torch.float32).to(device)
            y_b = torch.tensor(X_full[i:i+batch_size, :, :output_dim], dtype=torch.float32).to(device)
            preds = model(x_b)
            feat_err = ((preds - y_b) ** 2).mean(dim=1)
            all_feature_errors.append(feat_err.cpu().numpy())

    all_feature_errors = np.concatenate(all_feature_errors, axis=0)

    if per_feature_thresholds is not None:
        exceeds_threshold = all_feature_errors > per_feature_thresholds
        n_features_exceeding = exceeds_threshold.sum(axis=1)
        predicted_anomalies = (n_features_exceeding > output_dim / 2).astype(int)
        logger.info(f"Majority-vote decision: anomaly if >{output_dim//2}/{output_dim} features exceed threshold")
    else:
        all_errors = all_feature_errors.mean(axis=1)
        predicted_anomalies = (all_errors > threshold).astype(int)
        logger.info(f"Falling back to global threshold: {threshold:.6f}")

    # Full confusion matrix
    anomaly_mask = window_labels > 0
    tp = (predicted_anomalies == 1) & anomaly_mask
    fp = (predicted_anomalies == 1) & ~anomaly_mask
    tn = (predicted_anomalies == 0) & ~anomaly_mask
    fn = (predicted_anomalies == 0) & anomaly_mask

    logger.info(f"TP={tp.sum()}, FP={fp.sum()}, TN={tn.sum()}, FN={fn.sum()}")
    if fp.sum() + tn.sum() > 0:
        fpr = fp.sum() / (fp.sum() + tn.sum())
        logger.info(f"False Positive Rate: {fpr:.4f}")
    if tp.sum() + fn.sum() > 0:
        rec = tp.sum() / (tp.sum() + fn.sum())
        logger.info(f"Overall Recall: {rec:.4f}")
    if tp.sum() + fp.sum() > 0:
        prec = tp.sum() / (tp.sum() + fp.sum())
        logger.info(f"Overall Precision: {prec:.4f}")
    if tp.sum() + fp.sum() > 0 and tp.sum() + fn.sum() > 0:
        prec = tp.sum() / (tp.sum() + fp.sum())
        rec = tp.sum() / (tp.sum() + fn.sum())
        logger.info(f"Overall F1: {2 * prec * rec / (prec + rec):.4f}")

    print_class_statistics(window_labels, tp, fn)

if __name__ == "__main__":
    main()
