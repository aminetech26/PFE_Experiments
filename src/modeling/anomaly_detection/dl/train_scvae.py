import argparse
import os
import json
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import itertools
from pathlib import Path
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from src.modeling.anomaly_detection.dl.scvae_model import SCVAE

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_lstmae" / "lstmae_preprocessed.parquet"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "checkpoints" / "scvae"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "experiments" / "metrics"
DEFAULT_INPUT_COLS = ["irr", "idc1", "idc2", "vdc1", "vdc2", "pdc1", "pdc2", "clearness_index", "pvt"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_windows(data, window_size):
    """Creates overlapping sliding windows from the time series data."""
    chunks = []
    for t in range(data.shape[0] - window_size + 1):
        chunks.append(data[t : t + window_size, :])
    # output shape: (seq_len, batch_size, input_dim) to match SCVAE expected shape
    return np.stack(chunks).swapaxes(0, 1)

def load_and_prepare_data(data_path, window_size, batch_size):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Filter ONLY healthy data for training the VAE (label == 0) if applicable
    # The current assumption from LSTM-AE logic is to train strictly on healthy.
    # To keep your provided SCVAE purely as written:
    data = df[DEFAULT_INPUT_COLS].values
    
    # Scale data
    data = (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0) + 1e-8)
    data = np.nan_to_num(data)
    
    # Create windows
    windows = create_windows(data, window_size)
    
    # Split Train/Val (80/20)
    split_idx = int(windows.shape[1] * 0.8)
    train_data = windows[:, :split_idx, :]
    val_data = windows[:, split_idx:, :]
    
    # Expand dims to match architecture requirement: (seq_len, Batch, feature_dim, input_dim) 
    train_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(2)
    val_tensor = torch.tensor(val_data, dtype=torch.float32).unsqueeze(2)
    
    # We use X as both input and target for autoencoder
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_tensor), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    train_loader, val_loader = load_and_prepare_data(args.parquet_path, args.window_size, args.batch_size)
    
    # Hyperparameters to search over
    param_grid = {
        'h_dim': [32, 64],
        'z_dim': [8, 16],
        'lr': [1e-3, 5e-4]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_loss = float('inf')
    best_params = None
    best_model_state = None
    results_history = []
    
    logger.info(f"Starting Grid Search securely on {DEVICE}...")

    for idx, params in enumerate(combinations):
        logger.info(f"Testing Combo {idx + 1}/{len(combinations)}: {params}")
        
        # Initialize Model
        model = SCVAE(
            x_dim=1,                 # post-embedding dimension
            label_dim=1,             # post-embedding dimension
            h_dim=params['h_dim'],
            z_dim=params['z_dim'],
            input_dim=len(DEFAULT_INPUT_COLS),
            device=DEVICE
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        
        # Training Loop
        for epoch in range(args.epochs):
            model.train()
            train_loss_total = 0
            
            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.permute(1, 0, 2, 3).to(DEVICE)
                Y_batch = Y_batch.permute(1, 0, 2, 3).to(DEVICE)
                
                optimizer.zero_grad()
                kld_loss, nll_loss = model(X_batch, Y_batch)
                loss = kld_loss + nll_loss
                
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item()
                
            # Validation
            model.eval()
            val_loss_total = 0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch = X_batch.permute(1, 0, 2, 3).to(DEVICE)
                    Y_batch = Y_batch.permute(1, 0, 2, 3).to(DEVICE)
                    kld, nll = model(X_batch, Y_batch)
                    val_loss_total += (kld + nll).item()
                    
            val_loss_avg = val_loss_total / len(val_loader)
            train_loss_avg = train_loss_total / len(train_loader)
            
            logger.info(f"  Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")
        
        trial_result = {**params, "val_loss": val_loss_avg, "train_loss": train_loss_avg}
        results_history.append(trial_result)

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            best_params = params
            best_model_state = model.state_dict()
            
    logger.success(f"Grid Search Complete! Best Val Loss: {best_loss:.4f} with config: {best_params}")
    
    # Checkpoint Dir
    checkpoint_dir = Path(DEFAULT_CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "scvae_best.pth"
    
    # Save best model
    torch.save(best_model_state, str(best_model_path))
    logger.success(f"Best model weights saved to {best_model_path}")

    # Metrics Dir
    metrics_dir = Path(DEFAULT_METRICS_DIR)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "scvae_results.json"
    
    metrics_payload = {
        "model": "SCVAE",
        "dataset": "Costa PV Fault Dataset",
        "input_features": DEFAULT_INPUT_COLS,
        "grid_search_history": results_history,
        "best_parameters": best_params,
        "best_val_loss": best_loss
    }
    
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    logger.success(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()