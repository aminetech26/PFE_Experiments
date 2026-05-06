import os

# Fix matplotlib backend in headless/Colab environments
os.environ['MPLBACKEND'] = 'Agg'

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import warnings
import argparse
import pickle
import json
from pathlib import Path

warnings.filterwarnings("ignore")

from loguru import logger
from src.data.preprocess_gtbad import PVDataPreprocessor
from src.modeling.anomaly_detection.dl.gtbad_model import GTBADModel, reconstruction_error


# -------------------  helper: GVSAO optimizer  -------------------
class GVSAO:
    """
    Improved Snow Ablation Optimizer for hyperparameter tuning.
    Searches for learning rate (log scale) and batch size (integer).
    Implements good-point set initialization, dual-population update,
    melting factor, and periodic oscillation mutation.
    """

    def __init__(
        self, dim, bounds, pop_size=20, max_gen=10, T_period=5, A=0.1, mutation_prob=0.1
    ):
        """
        dim: 2
        bounds: list of (low, high) for each dim (log scale for lr)
        T_period: oscillation period
        A: amplitude
        mutation_prob: probability of applying oscillation mutation
        """
        self.dim = dim
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.T_period = T_period
        self.A = A
        self.mutation_prob = mutation_prob

    def _init_population(self):
        # Good point set initialization with small random perturbation
        pop = np.zeros((self.pop_size, self.dim))
        for d in range(self.dim):
            lb, ub = self.bounds[d]
            # equally spaced points between 0 and 1
            for i in range(self.pop_size):
                base = lb + (ub - lb) * i / (self.pop_size - 1)
                noise = np.random.uniform(-0.1, 0.1) * (ub - lb) / 2
                pop[i, d] = np.clip(base + noise, lb, ub)
        return pop

    def optimize(self, fitness_func, verbose=True):
        """
        fitness_func: function taking (lr, batch_size) and returning fitness (lower is better)
        Returns best solution and best fitness.
        """
        pop = self._init_population()
        fitness = np.full(self.pop_size, np.inf)
        pop_best = copy.deepcopy(pop)
        fit_best = np.full(self.pop_size, np.inf)
        global_best = None
        global_best_fit = np.inf

        # Eval initial population
        for i in range(self.pop_size):
            lr, bs = self._decode(pop[i])
            fitness[i] = fitness_func(lr, bs)
            fit_best[i] = fitness[i]
            pop_best[i] = pop[i]
            if fitness[i] < global_best_fit:
                global_best_fit = fitness[i]
                global_best = pop[i]

        if verbose:
            logger.info(f"Gen 0 Best Fitness: {global_best_fit:.6f}")

        FEs = self.pop_size
        FEs_max = self.pop_size * self.max_gen

        for gen in range(1, self.max_gen):
            # Sort to identify elite and compute population center
            idx_sort = np.argsort(fitness)
            pop_sorted = pop[idx_sort]
            fit_sorted = fitness[idx_sort]
            best_sol = pop_sorted[0]
            pop_center = np.mean(pop, axis=0)

            # Melting factor
            T_ratio = gen / self.max_gen
            DDF = 0.35 + 0.25 * (np.exp(FEs / FEs_max) - 1) / (np.e - 1)
            M = DDF * T_ratio  # melting factor

            new_pop = np.copy(pop)
            for i in range(self.pop_size):
                # random phase selection (exploration vs exploitation)
                if np.random.rand() < 0.5:  # exploration
                    b = np.random.normal(0, 1, self.dim)
                    theta1 = np.random.rand()
                    new_pop[i] = pop[i] + b * (
                        theta1 * (best_sol - pop[i]) + (1 - theta1) * (pop_center - pop[i])
                    )
                else:  # exploitation
                    b = np.random.normal(0, 1, self.dim)
                    theta2 = np.random.rand()
                    new_pop[i] = pop[i] + M * (best_sol - pop[i]) + b * (
                        theta2 * (pop_center - pop[i])
                    )

                # Bound check
                new_pop[i] = np.clip(new_pop[i], self.bounds[:, 0], self.bounds[:, 1])

                # Periodic oscillation mutation
                if np.random.rand() < self.mutation_prob:
                    W = self.A * np.sin(2 * np.pi * gen / self.T_period)
                    direction = best_sol - new_pop[i]
                    new_pop[i] = new_pop[i] + W * direction
                    new_pop[i] = np.clip(new_pop[i], self.bounds[:, 0], self.bounds[:, 1])

            # Evaluate new population
            for i in range(self.pop_size):
                if np.array_equal(new_pop[i], pop[i]):
                    continue  # skip redundant evaluation
                lr, bs = self._decode(new_pop[i])
                fitness[i] = fitness_func(lr, bs)
                FEs += 1
                if fitness[i] < fit_best[i]:
                    fit_best[i] = fitness[i]
                    pop_best[i] = new_pop[i]
                if fitness[i] < global_best_fit:
                    global_best_fit = fitness[i]
                    global_best = new_pop[i]

            pop = new_pop
            if verbose:
                logger.info(f"Gen {gen} Best Fitness: {global_best_fit:.6f}")

        best_lr, best_bs = self._decode(global_best)
        return best_lr, best_bs, global_best_fit

    def _decode(self, x):
        # x[0]: log10(lr), range [log10(1e-5), log10(1e-1)] -> lr
        lr_log = x[0]
        lr = 10 ** lr_log
        bs = int(round(x[1]))
        bs = max(16, min(128, bs))
        return lr, bs


# -------------------  main training script  -------------------
def load_dataset(filepath, timestamp_col="timestamp", label_col="anomaly"):
    _, ext = os.path.splitext(filepath)
    if ext.lower() == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    else:
        raise ValueError(f"Column {timestamp_col} not found.")
    return df


def train_lightweight(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu"):
    """Quick training for GVSAO fitness evaluation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for x_batch, y_batch, mask_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            # y_pred shape (batch, seq_len, output_dim), y_batch is (batch, seq_len, output_dim)
            # We reconstruct the full input and compute loss on all steps.
            loss = criterion(y_pred, y_batch[:, :, :])  # assuming y_batch has same length as x
            if mask_batch is not None:
                loss = (loss * mask_batch.float().unsqueeze(-1).unsqueeze(-1)).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0.0
    n_val = 0
    with torch.no_grad():
        for x_batch, y_batch, mask_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            mask_batch = mask_batch.to(device)
            y_pred = model(x_batch)
            loss = torch.sum((y_pred - y_batch) ** 2, dim=(1, 2))
            if mask_batch is not None:
                loss = loss * mask_batch.float()
            val_loss += loss.sum().item()
            n_val += mask_batch.sum().item() if mask_batch is not None else len(y_batch)
    avg_loss = val_loss / n_val if n_val > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train GTBAD model")
    parser.add_argument("--skip-hpo", action="store_true", help="Skip hyperparameter optimization and use fixed values.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Fixed learning rate (used if --skip-hpo is set)")
    parser.add_argument("--batch-size", type=int, default=32, help="Fixed batch size (used if --skip-hpo is set)")
    args = parser.parse_args()

    # Configuration
    DATA_PATH = "data/interim/ingestion/costa/costa_merged.parquet"
    TIMESTAMP_COL = "timestamp"
    LABEL_COL = "label"  # if exists; otherwise ignore metrics
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    # 1. Load and preprocess
    logger.info(f"Loading data from {DATA_PATH}...")
    df = load_dataset(DATA_PATH, TIMESTAMP_COL, LABEL_COL)
    preprocessor = PVDataPreprocessor(
        window_len=10,
        stride=1,
        power_col="pdc1",
        corr_threshold=0.97,
    )
    X_full, y_target, mask, df_clean = preprocessor.fit_transform(df, TIMESTAMP_COL)
    # X_full: (n_samples, 10, input_dim)  with input_dim = n_selected + 32
    # y_target: (n_samples, 10, n_selected)
    # To train with full sequence reconstruction, we should use the numeric part of X_full (first n_selected features) as target.
    n_selected = len(preprocessor.selected_features)
    X_numeric_full = X_full[:, :, :n_selected].copy()
    # Now X_numeric_full shape (n_samples, 10, n_selected) is both input and target.
    # The input to model will be X_full (with time encodings), target is X_numeric_full.

    # Train/validation/test split
    X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
        X_full, X_numeric_full, mask, test_size=0.2, random_state=42, shuffle=False
    )  # time series, no shuffle
    # further split train into train/val for GVSAO (80% train, 20% val of train)
    X_tr, X_val, y_tr, y_val, mask_tr, mask_val = train_test_split(
        X_train, y_train, mask_train, test_size=0.2, random_state=42, shuffle=False
    )

    # 2. Hyperparameter optimization or load fixed values
    input_dim = X_train.shape[2]
    output_dim = n_selected

    if args.skip_hpo:
        logger.info(f"Skipping HPO, using provided hyperparameters: LR={args.lr}, Batch Size={args.batch_size}")
        best_lr = args.lr
        best_bs = args.batch_size
    else:
        logger.info("Starting GVSAO hyperparameter optimization...")
        def fitness_func(lr, bs):
            bs = int(bs)
            # Build lightweight loaders
            train_dataset = TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32),
                torch.tensor(mask_tr, dtype=torch.float32),
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
                torch.tensor(mask_val, dtype=torch.float32),
            )
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
            model = GTBADModel(
                input_dim=input_dim,
                output_dim=output_dim,
                d_model=64,
                nhead=2,
                num_encoder_layers=3,
                lstm_hidden=32,
                dropout=0.1,
            ).to(DEVICE)
            val_loss = train_lightweight(
                model, train_loader, val_loader, epochs=5, lr=lr, device=DEVICE
            )
            return val_loss

        bounds = [[np.log10(1e-5), np.log10(1e-1)], [16, 128]]  # lr in log, bs linear
        optimizer_gvsao = GVSAO(dim=2, bounds=bounds, pop_size=20, max_gen=10)
        best_lr, best_bs, best_fit = optimizer_gvsao.optimize(fitness_func)
        logger.info(f"Best LR: {best_lr:.6f}, Best batch size: {best_bs}, Fitness: {best_fit:.6f}")

    # 3. Final training with best hyperparameters
    logger.info(f"Starting final training with bs={best_bs}, lr={best_lr}")
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(mask_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=best_bs, shuffle=True)

    model = GTBADModel(
        input_dim=input_dim,
        output_dim=output_dim,
        d_model=64,
        nhead=2,
        num_encoder_layers=3,
        lstm_hidden=32,
        dropout=0.1,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x_b, y_b, m_b in train_loader:
            x_b = x_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            m_b = m_b.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x_b)
            loss = criterion(y_pred, y_b)
            if m_b is not None:
                loss = (loss * m_b.unsqueeze(-1).unsqueeze(-1)).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{EPOCHS} Loss: {avg_loss:.6f}")

    # 4. Evaluation (anomaly detection)
    logger.info("Computing reconstruction errors to find anomaly threshold...")
    model.eval()
    # Compute reconstruction errors on training set to set threshold
    train_errors = []
    with torch.no_grad():
        for x_b, y_b, m_b in DataLoader(train_dataset, batch_size=best_bs, shuffle=False):
            x_b = x_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            y_pred = model(x_b)
            err = reconstruction_error(y_pred, y_b)
            train_errors.extend(err.cpu().numpy().tolist())
    train_errors = np.array(train_errors)
    threshold = np.percentile(train_errors, 95)
    logger.info(f"Anomaly threshold (95th percentile): {threshold:.6f}")

    # --- Save Model and Preprocessor ---
    checkpoint_dir = Path("experiments/checkpoints/gtbad")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = checkpoint_dir / "gtbad_best.pt"
    torch.save(model.state_dict(), model_path)
    
    preprocessor_path = checkpoint_dir / "preprocessor.pkl"
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
        
    threshold_path = checkpoint_dir / "threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"threshold": float(threshold), "input_dim": input_dim, "output_dim": output_dim}, f)
        
    logger.info(f"Model, preprocessor, and threshold saved to {checkpoint_dir}")

    # Test set errors
    logger.info("Computing errors on test set...")
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        torch.tensor(mask_test, dtype=torch.float32),
    )
    test_loader = DataLoader(test_dataset, batch_size=best_bs, shuffle=False)
    test_errors = []
    with torch.no_grad():
        for x_b, y_b, _ in test_loader:
            x_b = x_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            y_pred = model(x_b)
            err = reconstruction_error(y_pred, y_b)
            test_errors.extend(err.cpu().numpy().tolist())
    test_errors = np.array(test_errors)
    pred_anomaly = (test_errors > threshold).astype(int)

    # If true labels are available, compute metrics
    if LABEL_COL in df.columns:
        # need to align labels with test samples (they were created by sliding window)
        # The test set indices are after train_test_split; we can store the original sample indices during preprocessing.
        # For brevity, we assume df has anomaly labels aligned with timestamps and sliding window samples.
        # We will skip exact alignment here due to complexity; a complete implementation would track index mapping.
        logger.warning("True labels available but alignment with sliding windows not implemented in this script.")
    else:
        logger.info("No true labels; anomaly scores saved as 'test_anomaly_scores.csv'")
        np.savetxt("test_anomaly_scores.csv", test_errors, delimiter=",")


if __name__ == "__main__":
    main()
