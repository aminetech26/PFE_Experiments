from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.deployment.load_artifact import load_artifact_bundle


def _load_input(csv_path: Path, feature_names: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise KeyError(f"Missing required features: {missing}")
    return df, df[feature_names].to_numpy(dtype=np.float32)


def _unwrap_model_payload(model_payload):
    if isinstance(model_payload, dict) and "model" in model_payload:
        return model_payload["model"], model_payload
    return model_payload, {}


def _score_anomaly_model(model, model_name: str, x: np.ndarray, df: pd.DataFrame, dm: dict) -> np.ndarray:
    if model_name == "isolation_forest" and hasattr(model, "score_samples"):
        return -model.score_samples(x)
    if model_name == "one_class_svm" and hasattr(model, "decision_function"):
        return -model.decision_function(x)
    if model_name == "bocd" and hasattr(model, "score"):
        features = list(dm.get("feature_names", []))
        group_col = dm.get("group_column")
        label_col = str(dm.get("label_column", "label"))
        score_df = pd.DataFrame(x, columns=features, index=df.index)
        if isinstance(group_col, str) and group_col in df.columns:
            score_df[group_col] = df[group_col].values
        if label_col in df.columns:
            score_df[label_col] = df[label_col].values
        else:
            score_df[label_col] = 0.0
        return model.score(score_df, features, label_col, group_col if isinstance(group_col, str) else None)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return -model.decision_function(x)
    raise RuntimeError(f"Unsupported anomaly model interface for model={model_name}")


def _score_dl_anomaly_model(lit, x: np.ndarray, dm: dict) -> np.ndarray:
    """Run windowed inference on a DL LightningModule with score_windows().

    Slides a window of size W over the T timesteps of x [T, F].
    Each window produces one score assigned to its center timestep.
    Edge timesteps are filled with the nearest available score.
    Returns scores array of shape [T].
    """
    win_size = int(dm.get("window_size", 30))
    T, F = x.shape
    if T < win_size:
        # Pad to at least one window
        pad = np.zeros((win_size - T, F), dtype=np.float32)
        x_padded = np.concatenate([x, pad], axis=0)
    else:
        x_padded = x

    # Build all windows with stride 1
    n_windows = len(x_padded) - win_size + 1
    windows = np.stack([x_padded[i : i + win_size] for i in range(n_windows)], axis=0)  # [N, W, F]
    x_t = torch.from_numpy(windows)

    batch_size = 256
    scores_list: list[np.ndarray] = []
    for start in range(0, n_windows, batch_size):
        batch = x_t[start : start + batch_size]
        scores_list.append(lit.score_windows(batch).cpu().numpy())
    window_scores = np.concatenate(scores_list)  # [N]

    # Assign each window score to its center timestep
    centers = np.arange(n_windows) + win_size // 2  # center index in x_padded
    timestep_scores = np.full(len(x_padded), np.nan, dtype=np.float64)
    for i, c in enumerate(centers):
        if np.isnan(timestep_scores[c]):
            timestep_scores[c] = window_scores[i]

    # Fill NaN edges by forward/backward fill
    valid = np.where(~np.isnan(timestep_scores))[0]
    if len(valid) == 0:
        timestep_scores[:] = 0.0
    else:
        for i in range(len(timestep_scores)):
            if np.isnan(timestep_scores[i]):
                nearest = valid[np.argmin(np.abs(valid - i))]
                timestep_scores[i] = timestep_scores[nearest]

    return timestep_scores[:T].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference from deployment artifact")
    parser.add_argument("artifact_dir")
    parser.add_argument("input_csv")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bundle = load_artifact_bundle(args.artifact_dir)
    dm = bundle["deployment_manifest"]
    model_obj = bundle["model_obj"]
    model_path = bundle["model_path"]

    if model_obj is None:
        raise RuntimeError(f"Could not load model artifact: {model_path}")

    features = list(dm.get("feature_names", []))
    df, x = _load_input(Path(args.input_csv), features)

    scaler = bundle["scaler_obj"]
    if scaler is not None:
        x = scaler.transform(x)

    out: dict[str, object]
    task = str(dm.get("task", ""))
    model_name = str(dm.get("model", ""))

    if task.startswith("anomaly"):
        # DL path: LightningModule with score_windows
        if hasattr(model_obj, "score_windows"):
            scores = _score_dl_anomaly_model(model_obj, x, dm)
        else:
            model, _ = _unwrap_model_payload(model_obj)
            scores = _score_anomaly_model(model, model_name, x, df, dm)
        threshold = float(dm.get("threshold", 0.5))
        preds = (scores >= threshold).astype(int)
        out = {"scores": scores.tolist(), "predictions": preds.tolist(), "threshold": threshold}
    else:
        model, payload = _unwrap_model_payload(model_obj)
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Classification model missing predict_proba")
        proba = model.predict_proba(x)
        pred_indices = np.argmax(proba, axis=1)
        label_encoder = payload.get("label_encoder") if isinstance(payload, dict) else None
        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(pred_indices).tolist()
        else:
            classes = dm.get("classes")
            predictions = [classes[int(i)] for i in pred_indices] if classes else pred_indices.tolist()
        out = {"predictions": predictions, "probabilities": proba.tolist()}

    text = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
