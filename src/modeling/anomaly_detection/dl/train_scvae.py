# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from src.modeling.anomaly_detection.dl.scvae_model import SCVAE

# Force TensorFlow to only log errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessed" / "costa_scvae" / "scvae_preprocessed.parquet"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments" / "checkpoints" / "scvae" / "scvae_best.pth"

INPUT_FEATURES = ["pvt", "irr"]
OUTPUT_FEATURES = ["pdc1", "pdc2"]


def _resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    if data.shape[0] < window_size:
        raise ValueError("window_size is larger than the number of samples")

    windows = np.lib.stride_tricks.sliding_window_view(data, (window_size, data.shape[1]))
    return windows.reshape(-1, window_size, data.shape[1])


def load_window_sequences(parquet_path: str | Path, window_size: int) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    if "label" in df.columns:
        df = df[df["label"] == 0].copy()

    feature_cols = INPUT_FEATURES + OUTPUT_FEATURES
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")

    timestamp_col = "timestamp"
    if timestamp_col in df.columns:
        df = df.dropna(subset=[timestamp_col] + feature_cols).copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        df = df.dropna(subset=[timestamp_col])
        df = df.sort_values(timestamp_col)
    else:
        df = df.dropna(subset=feature_cols).copy()

    data = df[feature_cols].to_numpy(dtype=np.float32)
    data = np.nan_to_num(data).astype(np.float32, copy=False)
    return create_windows(data, window_size)


def test_one_epoch(dataloader, model, reg, mode, device):
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for _, (data, y_data) in enumerate(dataloader):
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            model(data, y_data)
            if mode == 0:
                loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss
            elif mode == 1:
                loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss + \
                    model.nll_loss_prior + 0 * model.smooth_loss_prior
            elif mode == 2:
                loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss + \
                    model.kld_loss_predict + model.nll_loss_predict
            test_loss += loss.item()
        test_loss /= num_batches
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")
    return test_loss


def predict_one_epoch(dataloader, model, reg, device, batch_size):
    num_batches = len(dataloader)
    predict_loss = 0
    reconstruct_loss = 0
    prior_loss = 0
    with torch.no_grad():
        for _, (data, y_data) in enumerate(dataloader):
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            model(data, y_data)
            loss_predict = model.nll_loss_predict
            loss_reconstruct = model.nll_loss
            loss_prior = model.nll_loss_prior

            predict_loss += loss_predict.item()
            reconstruct_loss += loss_reconstruct.item()
            prior_loss += loss_prior.item()
        predict_loss /= num_batches * batch_size
        reconstruct_loss /= num_batches * batch_size
        prior_loss /= num_batches * batch_size

    print(
        f"predict Error: \n , Avg loss: {predict_loss:>8f} \n reconstruct Error: \n , Avg loss: {reconstruct_loss:>8f} \n prior Error: \n , Avg loss: {prior_loss:>8f} \n")


def train_one_epoch(dataloader, model, optimizer, reg, mode, device):
    train_loss = 0
    for batch_idx, (data, y_data) in enumerate(dataloader):
        data = data.permute(1, 0, 3, 2).to(device)
        y_data = y_data.permute(1, 0, 3, 2).to(device)
        optimizer.zero_grad()
        model(data, y_data)
        if mode == 0:
            loss = model.kld_loss + model.nll_loss + \
                reg * model.smooth_loss
        elif mode == 1:
            loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss + \
                model.nll_loss_prior + 0 * model.smooth_loss_prior
        if mode == 2:
            loss = model.kld_loss + model.nll_loss + reg * model.smooth_loss + \
                model.kld_loss_predict + model.nll_loss_predict
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 2 == 0:
            size = len(data)
            loss, current = loss.item(), batch_idx * size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train(train_loader, test_loader, model, optimizer, reg, mode, model_path, device, n_epochs, batch_size):
    best_test_loss = 0
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model.train()
        train_one_epoch(train_loader, model, optimizer, reg, mode, device)
        model.eval()
        test_loss = test_one_epoch(test_loader, model, reg, mode, device)
        print("test predict")
        predict_one_epoch(test_loader, model, reg, device, batch_size)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print("Saved PyTorch Model State to model.pth")
    print("Done!")


def reconstruct(model, dataloader, device):
    print("*"*20+"reconstruct results")
    with torch.no_grad():
        mus, stds, scores = [], [], []
        for _, (data, y_data) in enumerate(dataloader):
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            mu, std, score = model.reconstruct(data, y_data, is_prior=False)
            mu = np.transpose(mu, (1, 0, 2))
            std = np.transpose(std, (1, 0, 2))
            score = np.transpose(score, (1, 0, 2))

            mus.append(mu)
            stds.append(std)
            scores.append(score)
        mus = np.concatenate(mus, axis=0)
        stds = np.concatenate(stds, axis=0)
        scores = np.concatenate(scores, axis=0)
        print("scores.shape", scores.shape)
        return mus, stds, scores


def predict_withLabel(model, dataloader, device):
    print("*"*20+"predict result")
    with torch.no_grad():
        mus, stds, scores = [], [], []
        for _, (data, y_data) in enumerate(dataloader):
            data = data.permute(1, 0, 3, 2).to(device)
            y_data = y_data.permute(1, 0, 3, 2).to(device)
            mu, std, score = model.predict_withLabel(data, y_data)
            mu = np.transpose(mu, (1, 0, 2))
            std = np.transpose(std, (1, 0, 2))
            score = np.transpose(score, (1, 0, 2))

            mus.append(mu)
            stds.append(std)
            scores.append(score)
        mus = np.concatenate(mus, axis=0)
        stds = np.concatenate(stds, axis=0)
        scores = np.concatenate(scores, axis=0)
        print("scores.shape", scores.shape)

        return mus, stds, scores


def predict(model, dataloader, device):
    print("*"*20+"predict result")
    with torch.no_grad():
        mus, stds = [], []
        for _, (data, y_data) in enumerate(dataloader):
            data = data.permute(1, 0, 3, 2).to(device)
            mu, std = model.predict(data)
            mu = np.transpose(mu, (1, 0, 2))
            std = np.transpose(std, (1, 0, 2))

            mus.append(mu)
            stds.append(std)
        mus = np.concatenate(mus, axis=0)
        stds = np.concatenate(stds, axis=0)
        print("mu shape", mus.shape)

        return mus, stds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, default=str(DEFAULT_DATA_PATH),
                        help="Path to SCVAE preprocessed parquet")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="saved model path")
    parser.add_argument("--reg", type=float, default=0,
                        help="smooth canonical intensity")
    parser.add_argument("--batch_size", type=int,
                        default=256, help="batch size")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="device, e.g. cpu, cuda:0, cuda:1")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-5, help="learning_rate")
    parser.add_argument("--print_every", type=int, default=1,
                        help="the number of iterations between printing the results")
    parser.add_argument("--n_epochs", type=int, default=100000,
                        help="Maximum number of iterations")
    parser.add_argument("--h_dim", type=int, default=512,
                        help="Neural network hidden layer dimension")
    parser.add_argument("--z_dim", type=int, default=128,
                        help="hidden variable dimension of VAEs")
    parser.add_argument("--mode", type=int, default=2,
                        help="the mode when train")
    parser.add_argument("--test_ratio", type=float,
                        default=0.25, help="the test ratio in data_set")
    parser.add_argument("--window_size", type=int, default=50,
                        help="Sequence length for sliding windows")
    opt = parser.parse_args()

    device = _resolve_device(opt.device)

    windows = load_window_sequences(opt.parquet_path, opt.window_size)

    x_pvt = windows[..., 0:1]
    x_irr = windows[..., 1:2]
    y_pdc1 = windows[..., 2:3]
    y_pdc2 = windows[..., 3:4]

    num_windows = windows.shape[0]
    if opt.test_ratio != 1 and num_windows >= 2:
        x_train1, x_test1, x_train2, x_test2, y_train1, y_test1, y_train2, y_test2 = train_test_split(
            x_pvt, x_irr, y_pdc1, y_pdc2, test_size=opt.test_ratio, random_state=42)
    else:
        if num_windows < 2 and opt.test_ratio != 1:
            print("Only one window available; using all data for both train and test. Set --test_ratio 1 to silence this warning.")
        x_train1 = x_test1 = x_pvt
        x_train2 = x_test2 = x_irr
        y_train1 = y_test1 = y_pdc1
        y_train2 = y_test2 = y_pdc2

    multi_x_train = np.stack([x_train1, x_train2], axis=3)
    multi_x_test = np.stack([x_test1, x_test2], axis=3)
    multi_y_train = np.stack([y_train1, y_train2], axis=3)
    multi_y_test = np.stack([y_test1, y_test2], axis=3)

    x_dim = multi_x_train.shape[-1]
    y_dim = multi_y_train.shape[-1]
    input_dim = multi_x_train.shape[-2]
    h_dim = opt.h_dim
    z_dim = opt.z_dim
    n_epochs = opt.n_epochs
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size
    model_path = Path(opt.model_path)

    chunk_torch = torch.FloatTensor(multi_x_train)
    test_torch = torch.FloatTensor(multi_x_test)

    train_loader_ordered = DataLoader(TensorDataset(
        chunk_torch, torch.FloatTensor(multi_y_train)), batch_size=batch_size, shuffle=False)
    test_loader_ordered = DataLoader(TensorDataset(
        test_torch, torch.FloatTensor(multi_y_test)), batch_size=batch_size, shuffle=False)

    model = SCVAE(x_dim, y_dim, h_dim, z_dim, input_dim, 1,
                  device=device, is_prior=False).to(device)

    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    reg = opt.reg

    # Train
    train(train_loader_ordered, test_loader_ordered, model, optimizer, reg, opt.mode, model_path, device, n_epochs, batch_size)
