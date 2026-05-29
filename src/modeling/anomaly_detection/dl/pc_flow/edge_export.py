"""PC-Flow edge deployment: ONNX export, parity check, latency benchmark.

Usage:
    python -m src.modeling.anomaly_detection.dl.pc_flow.edge_export \
        --checkpoint experiments/anomaly/pc_flow/<ts>/checkpoints/best.ckpt \
        --n-features 15 --n-context 2 \
        --output experiments/anomaly/pc_flow/<ts>/pc_flow.onnx

Outputs:
    <output>.onnx             — ONNX model (opset 17)
    deployment_metrics.json   — param count, latency stats, parity check result
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.modeling.anomaly_detection.dl.pc_flow.model import PCFlowModel


def export_onnx(
    model: PCFlowModel,
    output_path: Path,
    n_features: int,
    n_context: int,
    opset: int = 17,
) -> None:
    model.eval()
    dummy_x = torch.randn(1, n_features)
    dummy_c = torch.randn(1, n_context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_x, dummy_c),
        str(output_path),
        input_names=["x", "c"],
        output_names=["z", "log_det"],
        dynamic_axes={"x": {0: "batch"}, "c": {0: "batch"},
                      "z": {0: "batch"}, "log_det": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        # Legacy TorchScript exporter: stable for this static MLP graph, honors
        # dynamic_axes, and avoids the onnxscript dependency the dynamo exporter
        # (torch>=2.x default) requires. Needs only `onnx` at export time.
        dynamo=False,
    )


def parity_check(
    model: PCFlowModel,
    onnx_path: Path,
    n_features: int,
    n_context: int,
    n_samples: int = 1000,
    tol: float = 1e-4,
) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"status": "skipped", "reason": "onnxruntime not installed"}

    model.eval()
    x = torch.randn(n_samples, n_features)
    c = torch.randn(n_samples, n_context)

    with torch.no_grad():
        pt_scores = model.anomaly_score(x, c).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["z", "log_det"], {"x": x.numpy(), "c": c.numpy()})
    z_ort, logdet_ort = ort_out
    # Recompute NLL from ONNX outputs
    import math
    D = n_features
    log_p_z = -0.5 * (z_ort ** 2).sum(axis=-1) - 0.5 * D * math.log(2 * math.pi)
    ort_scores = -(log_p_z + logdet_ort)

    max_diff = float(np.abs(pt_scores - ort_scores).max())
    mean_diff = float(np.abs(pt_scores - ort_scores).mean())
    passed = max_diff < tol
    return {
        "status": "passed" if passed else "failed",
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "tolerance": tol,
        "n_samples": n_samples,
    }


def benchmark_latency(
    model: PCFlowModel,
    n_features: int,
    n_context: int,
    n_warmup: int = 100,
    n_bench: int = 1000,
    batch_size: int = 1,
) -> dict:
    """Per-sample latency at batch_size=1 (online inference)."""
    model.eval()
    x = torch.randn(batch_size, n_features)
    c = torch.randn(batch_size, n_context)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.anomaly_score(x, c)

        times = []
        for _ in range(n_bench):
            t0 = time.perf_counter()
            _ = model.anomaly_score(x, c)
            times.append(time.perf_counter() - t0)

    times_us = [t * 1e6 for t in times]  # microseconds
    return {
        "batch_size": batch_size,
        "n_bench": n_bench,
        "latency_median_us": float(np.median(times_us)),
        "latency_p99_us": float(np.percentile(times_us, 99)),
        "latency_mean_us": float(np.mean(times_us)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="PC-Flow ONNX export + edge benchmarking")
    p.add_argument("--checkpoint", default=None, help="Path to Lightning .ckpt (optional; builds from config if not given)")
    p.add_argument("--n-features", type=int, required=True, help="Number of non-context flow input features")
    p.add_argument("--n-context", type=int, default=2, help="Number of context features (irr, pvt)")
    p.add_argument("--n-coupling-layers", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--output", default=None, help="Path for ONNX file (default: alongside checkpoint)")
    p.add_argument("--no-onnx", action="store_true", help="Skip ONNX export (latency benchmark only)")
    args = p.parse_args()

    model = PCFlowModel(
        n_features=args.n_features,
        n_context=args.n_context,
        n_coupling_layers=args.n_coupling_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        # Extract model state dict from Lightning checkpoint
        state = ckpt.get("state_dict", ckpt)
        # Strip "model." prefix if present (Lightning wraps the model)
        model_state = {k.replace("model.", "", 1): v for k, v in state.items() if k.startswith("model.")}
        if model_state:
            model.load_state_dict(model_state, strict=True)
        else:
            model.load_state_dict(state, strict=False)

    n_params = model.n_params
    print(f"PC-Flow parameters: {n_params:,}")

    results: dict = {"n_params": n_params, "n_features": args.n_features, "n_context": args.n_context}

    # Latency benchmark
    latency = benchmark_latency(model, args.n_features, args.n_context)
    results["latency"] = latency
    print(f"Latency (batch=1): median={latency['latency_median_us']:.1f} µs  p99={latency['latency_p99_us']:.1f} µs")

    if not args.no_onnx:
        onnx_path = Path(args.output) if args.output else (
            Path(args.checkpoint).parent.parent / "pc_flow.onnx" if args.checkpoint
            else Path("pc_flow.onnx")
        )
        export_onnx(model, onnx_path, args.n_features, args.n_context)
        print(f"ONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")
        results["onnx_path"] = str(onnx_path)
        results["onnx_size_kb"] = round(onnx_path.stat().st_size / 1024, 1)

        parity = parity_check(model, onnx_path, args.n_features, args.n_context)
        results["parity_check"] = parity
        print(f"Parity check: {parity['status']} (max_diff={parity.get('max_abs_diff', 'n/a')})")

    # Save deployment metrics
    out_dir = Path(args.checkpoint).parent.parent if args.checkpoint else Path(".")
    dm_path = out_dir / "deployment_metrics.json"
    dm_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Deployment metrics saved: {dm_path}")


if __name__ == "__main__":
    main()
