from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch


def _coerce_stats(raw: Any) -> dict[str, tuple[float, float]] | None:
    if not isinstance(raw, dict):
        return None
    out: dict[str, tuple[float, float]] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            out[str(k)] = (float(v[0]), float(v[1]))
    return out if out else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dlssm_ckpt(model_path: Path, root: Path, deployment_manifest: dict[str, Any]) -> Any:
    """Load pure DLS-SSM checkpoint by reconstructing module explicitly.

    Avoids reliance on Lightning's load_from_checkpoint constructor inference,
    and remains stable if hyperparameters were not serialized in old checkpoints.
    """
    from src.modeling.anomaly_detection.dl.dlssm.model import DeepLatentStateSpaceModel  # noqa: PLC0415, I001
    from src.modeling.anomaly_detection.dl.dlssm.trainer import DLSSMLightningModule  # noqa: PLC0415, I001

    run_params_path = root / "run_params.json"
    run_params: dict[str, Any] = load_json(run_params_path) if run_params_path.exists() else {}

    feature_names = [str(x) for x in deployment_manifest.get("feature_names", [])]
    n_features = len(feature_names)
    if n_features <= 0:
        raise ValueError("deployment_manifest.feature_names must be non-empty for DLSSM ckpt load")

    scaler_path = root / str(deployment_manifest.get("scaler_artifact", ""))
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler artifact required for DLSSM load: {scaler_path}")
    scaler = joblib.load(scaler_path)
    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    feature_idx = {name: i for i, name in enumerate(feature_names)}

    model = DeepLatentStateSpaceModel(
        n_features=n_features,
        win_size=int(run_params.get("win_size", deployment_manifest.get("window_size", 30))),
        hidden_dim=int(run_params.get("hidden_dim", 64)),
        latent_dim=int(run_params.get("latent_dim", 16)),
        encoder_dim=int(run_params.get("encoder_dim", 64)),
        decoder_dim=int(run_params.get("decoder_dim", 64)),
        n_gru_layers=int(run_params.get("n_gru_layers", 1)),
        dropout=float(run_params.get("dropout", 0.1)),
        se_reduction_ratio=int(run_params.get("se_reduction_ratio", 4)),
    )
    lit = DLSSMLightningModule(
        model=model,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_idx=feature_idx,
        learning_rate=float(run_params.get("learning_rate", 1e-3)),
        weight_decay=float(run_params.get("weight_decay", 1e-5)),
        gradient_clip_val=float(run_params.get("gradient_clip_val", 1.0)),
        max_epochs=int(run_params.get("max_epochs_actual", 1)),
        total_steps=1,
        warmup_steps=0,
        min_lr_ratio=0.01,
        beta_kl=float(run_params.get("beta_kl", 0.1)),
        kl_warmup_epochs=int(run_params.get("kl_warmup_epochs", 5)),
        lambda_kl_score=float(run_params.get("lambda_kl_score", 0.1)),
        score_reduction=str(run_params.get("score_reduction", "center")),
        free_bits=float(run_params.get("free_bits", 0.0)),
        latent_dim=int(run_params.get("latent_dim", 16)),
        score_instability_p95=1e6,
        score_instability_max=1e7,
        non_finite_warn_limit_per_epoch=5,
        max_abs_base_score_term=1e5,
        hpo_mode=False,
    )

    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    lit.load_state_dict(state_dict, strict=True)
    return lit


def _load_ckpt(model_path: Path, model_name: str, root: Path, deployment_manifest: dict[str, Any]) -> Any:
    """Load a PyTorch Lightning checkpoint for the given DL model name."""
    if model_name == "maat":
        from src.modeling.anomaly_detection.dl.maat.model import MambaAnomalyTransformer  # noqa: PLC0415, I001
        from src.modeling.anomaly_detection.dl.maat.trainer import MAATLightningModule  # noqa: PLC0415, I001

        cfg_path = root / "resolved_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing resolved config required for MAAT ckpt load: {cfg_path}")
        resolved = load_json(cfg_path)
        maat_cfg = resolved.get("maat", {})
        training_cfg = resolved.get("training", {})
        feature_names = [str(x) for x in deployment_manifest.get("feature_names", [])]
        n_features = len(feature_names)
        if n_features <= 0:
            raise ValueError("deployment_manifest.feature_names must be non-empty for MAAT ckpt load")

        drift_feature_idx = feature_names.index("pdc_temp_corrected_norm_irr") if "pdc_temp_corrected_norm_irr" in feature_names else None
        voltage_drop_feature_indices = [
            int(feature_names.index(c))
            for c in ("vdc1", "vdc2")
            if c in feature_names
        ]
        power_imbalance_feature_idx = (
            int(feature_names.index("power_imbalance")) if "power_imbalance" in feature_names else None
        )
        current_imbalance_feature_idx = (
            int(feature_names.index("current_imbalance")) if "current_imbalance" in feature_names else None
        )
        voltage_imbalance_feature_idx = (
            int(feature_names.index("voltage_imbalance")) if "voltage_imbalance" in feature_names else None
        )

        model = MambaAnomalyTransformer(
            win_size=int(maat_cfg.get("win_size", deployment_manifest.get("window_size", 30))),
            enc_in=n_features,
            c_out=n_features,
            d_model=int(maat_cfg.get("d_model", 64)),
            n_heads=int(maat_cfg.get("n_heads", 2)),
            e_layers=int(maat_cfg.get("e_layers", 2)),
            d_ff=int(maat_cfg.get("d_ff", 256)),
            dropout=float(maat_cfg.get("dropout", 0.1)),
            block_size=int(maat_cfg.get("block_size", 10)),
        )
        lit = MAATLightningModule(
            model=model,
            k=float(maat_cfg.get("k", 3.0)),
            temperature=float(maat_cfg.get("temperature", 50.0)),
            learning_rate=float(training_cfg.get("lr", 1.0e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1.0e-2)),
            gradient_clip_val=float(training_cfg.get("gradient_clip_val", 1.0)),
            max_epochs=int(training_cfg.get("max_epochs", 1)),
            total_steps=1,
            warmup_steps=int(training_cfg.get("warmup_steps", 0)),
            min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.01)),
            score_reduction=str(maat_cfg.get("score_reduction", "max")),
            drift_feature_idx=drift_feature_idx,
            voltage_drop_feature_indices=voltage_drop_feature_indices,
            power_imbalance_feature_idx=power_imbalance_feature_idx,
            current_imbalance_feature_idx=current_imbalance_feature_idx,
            voltage_imbalance_feature_idx=voltage_imbalance_feature_idx,
        )
        ckpt = torch.load(model_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        lit.load_state_dict(state_dict, strict=True)
        if isinstance(ckpt, dict):
            lit.on_load_checkpoint(ckpt)
    elif model_name == "dlssm":
        lit = _load_dlssm_ckpt(model_path, root, deployment_manifest)
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
    calibration_rel = deployment_manifest.get("score_calibration_artifact")
    if not calibration_rel:
        calibration_rel = deployment_manifest.get("extras", {}).get("score_calibration_artifact")
    calibration_path = root / calibration_rel if calibration_rel else None

    if model_path is None:
        raise ValueError("deployment_manifest.json missing non-empty model_artifact")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact referenced by manifest does not exist: {model_path}")
    if scaler_path is not None and not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact referenced by manifest does not exist: {scaler_path}")
    if calibration_path is not None and not calibration_path.exists():
        raise FileNotFoundError(
            f"Score calibration artifact referenced by manifest does not exist: {calibration_path}"
        )

    model_obj = None
    if model_path.suffix == ".joblib":
        model_obj = joblib.load(model_path)
    elif model_path.suffix == ".ckpt":
        model_obj = _load_ckpt(model_path, str(deployment_manifest.get("model", "")), root, deployment_manifest)

    if model_obj is not None and calibration_path is not None and calibration_path.exists():
        calibration = load_json(calibration_path)
        pv_stats = _coerce_stats(calibration.get("pv_score_stats"))
        if pv_stats is not None and hasattr(model_obj, "_pv_score_stats"):
            model_obj._pv_score_stats = pv_stats
        thr = calibration.get("threshold")
        if thr is not None and hasattr(model_obj, "val_threshold"):
            model_obj.val_threshold = float(thr)

    scaler_obj = None
    if scaler_path and scaler_path.exists() and scaler_path.suffix == ".joblib":
        scaler_obj = joblib.load(scaler_path)

    return {
        "artifact_dir": root,
        "deployment_manifest": deployment_manifest,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "calibration_path": calibration_path,
        "model_obj": model_obj,
        "scaler_obj": scaler_obj,
    }
