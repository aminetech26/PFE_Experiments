from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ckpt(model_path: Path, model_name: str) -> Any:
    """Load a PyTorch Lightning checkpoint for the given DL model name."""
    if model_name == "maat":
        from src.modeling.anomaly_detection.dl.maat.trainer import MAATLightningModule  # noqa: PLC0415
        lit = MAATLightningModule.load_from_checkpoint(str(model_path), map_location="cpu")
    elif model_name == "dlssm":
        from src.modeling.anomaly_detection.dl.dlssm.trainer import DLSSMLightningModule  # noqa: PLC0415
        lit = DLSSMLightningModule.load_from_checkpoint(str(model_path), map_location="cpu")
    else:
        raise ValueError(f"Unknown DL model '{model_name}' for .ckpt loading")
    lit.eval()
    return lit


def load_artifact_bundle(artifact_dir: str | Path) -> dict[str, Any]:
    root = Path(artifact_dir)
    manifest_path = root / "deployment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing deployment manifest: {manifest_path}")
    deployment_manifest = load_json(manifest_path)
    model_rel = deployment_manifest.get("model_artifact", "")
    model_path = root / model_rel if model_rel else None
    scaler_rel = deployment_manifest.get("scaler_artifact")
    scaler_path = root / scaler_rel if scaler_rel else None

    if model_path is None:
        raise ValueError("deployment_manifest.json missing non-empty model_artifact")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact referenced by manifest does not exist: {model_path}")
    if scaler_path is not None and not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact referenced by manifest does not exist: {scaler_path}")

    model_obj = None
    if model_path.suffix == ".joblib":
        model_obj = joblib.load(model_path)
    elif model_path.suffix == ".ckpt":
        model_obj = _load_ckpt(model_path, str(deployment_manifest.get("model", "")))

    scaler_obj = None
    if scaler_path and scaler_path.exists() and scaler_path.suffix == ".joblib":
        scaler_obj = joblib.load(scaler_path)

    return {
        "artifact_dir": root,
        "deployment_manifest": deployment_manifest,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "model_obj": model_obj,
        "scaler_obj": scaler_obj,
    }
