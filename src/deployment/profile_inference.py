from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from src.deployment.load_artifact import load_artifact_bundle


def _infer(model, x: np.ndarray, dm: dict) -> None:
    # DL path: LightningModule with score_windows expects [B, W, F] tensor
    if hasattr(model, "score_windows"):
        win_size = int(dm.get("window_size", 30))
        B = x.shape[0]
        F = x.shape[1]
        x_t = torch.from_numpy(
            np.random.randn(B, win_size, F).astype(np.float32)
        )
        _ = model.score_windows(x_t)
        return

    model_name = str(dm.get("model", ""))
    if model_name == "isolation_forest" and hasattr(model, "score_samples"):
        _ = -model.score_samples(x)
    elif model_name == "one_class_svm" and hasattr(model, "decision_function"):
        _ = -model.decision_function(x)
    elif model_name == "bocd" and hasattr(model, "score"):
        features = list(dm.get("feature_names", []))
        label_col = str(dm.get("label_column", "label"))
        df = pd.DataFrame(x, columns=features)
        df[label_col] = 0.0
        _ = model.score(df, features, label_col, None)
    elif hasattr(model, "predict_proba"):
        _ = model.predict_proba(x)
    elif hasattr(model, "decision_function"):
        _ = model.decision_function(x)
    else:
        _ = model.predict(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile inference latency from artifact")
    parser.add_argument("artifact_dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    bundle = load_artifact_bundle(args.artifact_dir)
    dm = bundle["deployment_manifest"]
    model_obj = bundle["model_obj"]
    if model_obj is None:
        raise RuntimeError(f"Could not load model artifact: {bundle['model_path']}")

    model = model_obj["model"] if isinstance(model_obj, dict) and "model" in model_obj else model_obj
    n_features = len(dm.get("feature_names", []))
    if n_features <= 0:
        raise ValueError("deployment_manifest.feature_names is empty")

    x = np.random.randn(int(args.batch_size), n_features).astype(np.float32)

    for _ in range(int(args.warmup)):
        _infer(model, x, dm)

    lat_ms: list[float] = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        _infer(model, x, dm)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(lat_ms, dtype=float)
    report = {
        "batch_size": int(args.batch_size),
        "runs": int(args.runs),
        "latency_ms": {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
