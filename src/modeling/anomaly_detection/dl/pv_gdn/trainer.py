from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from src.mlflow_setup import init_tracking
from src.modeling.anomaly_detection.dl.dataset import TimeSeriesDataset
from src.modeling.anomaly_detection.dl.pv_gdn.model import PVGDN
from src.modeling.common.artifact_contract import (
    build_candidate_per_true_class_thresholds,
    build_deployment_manifest,
    build_run_manifest,
    build_score_calibration_payload,
    compute_anomaly_per_class_metrics,
    compute_macro_per_class_pr_auc,
    write_json,
)
from src.modeling.common.episode_metrics import episode_macro_f1_binary
from src.modeling.common.feature_loader import load_features_for_task
from src.utils.paths import get_experiments_root

PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _default_comparison_records_path() -> Path:
    return get_experiments_root() / "metrics" / "anomaly_comparison_records.jsonl"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PV-GDN anomaly detection")
    p.add_argument("--task", default="anomaly_semisup")
    p.add_argument("--dataset", default="costa")
    p.add_argument("--split-path", default="path_a")
    p.add_argument("--profile", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-type", default="baseline")
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--comparison-records-path", default=str(_default_comparison_records_path()))
    p.add_argument("--win-size", type=int, default=None, help="Override win_size from config (Run A=30, Run B=60)")
    p.add_argument(
        "--score-reduction",
        default=None,
        choices=["max_group", "mean_group", "global_only", "voltage_only", "power_only", "string_only"],
        help="Override score_reduction from config (Run C=voltage_only)",
    )
    return p.parse_args()


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _calibrate_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError(
            "Validation threshold calibration requires both normal and fault windows; "
            f"got labels={unique_labels.tolist()}"
        )
    prec, rec, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Validation threshold calibration produced no PR thresholds")
    denom = prec[:-1] + rec[:-1]
    f1_vals = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / np.maximum(denom, 1e-10), 0.0)
    best_idx = int(np.argmax(f1_vals))
    return float(thresholds[best_idx]), float(f1_vals[best_idx]), float(prec[best_idx]), float(rec[best_idx])


def _feature_groups(features: list[str]) -> dict[str, list[int]]:
    idx = {f: i for i, f in enumerate(features)}

    def _pick(names: list[str]) -> list[int]:
        return [idx[n] for n in names if n in idx]

    voltage = _pick(["vdc1", "vdc2", "voltage_imbalance", "dV_dt"])
    power = _pick(
        [
            "pdc",
            "pdc1",
            "pdc2",
            "pdc_temp_corrected_norm_irr",
            "power_imbalance",
            "dP_dt",
            "dI_dt",
        ]
    )
    # String-balance group: cross-string imbalance signatures — key for class 1/3 detection
    string = _pick(["voltage_imbalance", "current_imbalance", "power_imbalance"])
    assigned = set(voltage) | set(power) | set(string)
    other = [i for i in range(len(features)) if i not in assigned]
    groups = {
        "global": list(range(len(features))),
        "voltage": voltage,
        "power": power,
        "string": string,
        "other": other,
    }
    resolved = {k: v for k, v in groups.items() if v}
    if len(resolved.get("voltage", [])) < 2:
        raise ValueError(
            "PV-GDN requires at least 2 resolved voltage features; "
            f"got {len(resolved.get('voltage', []))} from features={features}"
        )
    if len(resolved.get("string", [])) < 2:
        raise ValueError(
            "PV-GDN requires at least 2 resolved string-balance features; "
            f"got {len(resolved.get('string', []))} from features={features}"
        )
    return resolved


def _compute_group_scores(residuals: np.ndarray, groups: dict[str, list[int]]) -> dict[str, np.ndarray]:
    group_scores: dict[str, np.ndarray] = {}
    for g, ids in groups.items():
        group_scores[g] = residuals[:, ids].mean(axis=1)
    return group_scores


def _reduce_scores(group_scores: dict[str, np.ndarray], mode: str) -> np.ndarray:
    if not group_scores:
        return np.array([])
    if mode in {"global", "global_only"}:
        if "global" not in group_scores:
            raise ValueError("score_reduction='global' requires a non-empty 'global' group")
        return group_scores["global"]
    if mode == "voltage_only":
        if "voltage" not in group_scores:
            raise ValueError("score_reduction='voltage_only' requires a non-empty 'voltage' group")
        return group_scores["voltage"]
    if mode == "power_only":
        if "power" not in group_scores:
            raise ValueError("score_reduction='power_only' requires a non-empty 'power' group")
        return group_scores["power"]
    if mode == "string_only":
        if "string" not in group_scores:
            raise ValueError("score_reduction='string_only' requires a non-empty 'string' group")
        return group_scores["string"]
    # All named groups participate in max/mean — global anchors large-amplitude faults (class 4),
    # specialized groups capture relational violations (classes 1/2/3).
    # "context" = mean |Z-scored context| catches sustained faults next-step prediction misses.
    all_groups = [
        group_scores[name]
        for name in ("global", "voltage", "power", "string", "other", "context")
        if name in group_scores
    ]
    if mode == "mean_group":
        if not all_groups:
            raise ValueError("score_reduction='mean_group' requires at least one group")
        return np.vstack(all_groups).mean(axis=0)
    if mode == "max_group":
        if not all_groups:
            raise ValueError("score_reduction='max_group' requires at least one group")
        return np.vstack(all_groups).max(axis=0)
    raise ValueError(f"Unsupported score_reduction mode: {mode}")


def _safe_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        return None
    if np.unique(labels_arr).size < 2:
        return None
    return float(average_precision_score(labels_arr, scores))


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels_arr = np.asarray(labels)
    if labels_arr.size == 0:
        return None
    if np.unique(labels_arr).size < 2:
        return None
    return float(roc_auc_score(labels_arr, scores))


def _resolve_inference_device(accelerator: str | None) -> torch.device:
    accelerator_str = str(accelerator or "auto").lower()
    if accelerator_str in {"auto", "gpu", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _has_any_normal_windows(ds: TimeSeriesDataset) -> bool:
    return any(int(ds[idx][1].item()) == 0 for idx in range(len(ds)))


def _fit_residual_scale(residuals: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    median = np.median(residuals, axis=0)
    mad = np.median(np.abs(residuals - median), axis=0)
    return np.maximum(mad, eps)


def _apply_residual_scale(residuals: np.ndarray, residual_scale: np.ndarray | None) -> np.ndarray:
    if residual_scale is None:
        return residuals
    return residuals / residual_scale[np.newaxis, :]


def _extract_top_graph_edges(model: PVGDN, features: list[str], max_edges: int = 20) -> list[dict[str, float | str]]:
    with torch.no_grad():
        adjacency = model._adjacency().detach().cpu().numpy()
    edges: list[dict[str, float | str]] = []
    for target_idx, target_name in enumerate(features):
        for source_idx, source_name in enumerate(features):
            weight = float(adjacency[target_idx, source_idx])
            if not np.isfinite(weight) or weight <= 0:
                continue
            edges.append({"source": source_name, "target": target_name, "weight": weight})
    edges.sort(key=lambda item: item["weight"], reverse=True)
    return edges[:max_edges]


class PVGDNLightning(pl.LightningModule):
    def __init__(
        self,
        model: PVGDN,
        learning_rate: float,
        weight_decay: float,
        groups: dict[str, list[int]] | None = None,
        score_reduction: str = "max_group",
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.loss_fn = nn.SmoothL1Loss()
        self._groups = groups or {}
        self._score_reduction = score_reduction
        self._val_residuals: list[np.ndarray] = []
        self._val_context_dev: list[np.ndarray] = []
        self._val_labels_bin: list[np.ndarray] = []
        self._val_labels_orig: list[np.ndarray] = []

    def training_step(self, batch, batch_idx: int):
        x, _ = batch
        context = x[:, :-1, :]
        target = x[:, -1, :]
        x_hat, _ = self.model(context)
        loss = self.loss_fn(x_hat, target)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x = batch[0]
        y_bin = batch[1]
        y_orig = batch[2].detach().cpu().numpy() if len(batch) > 2 else y_bin.detach().cpu().numpy().copy()
        context = x[:, :-1, :]
        target = x[:, -1, :]
        x_hat, _ = self.model(context)
        loss = self.loss_fn(x_hat, target)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        residuals = (x_hat - target).abs().detach().cpu().numpy()
        # context_dev: mean |Z-scored context| per feature — catches sustained fault distributions
        context_dev = context.abs().mean(dim=1).detach().cpu().numpy()  # [B, F]
        self._val_residuals.append(residuals)
        self._val_context_dev.append(context_dev)
        self._val_labels_bin.append(y_bin.detach().cpu().numpy().astype(int))
        self._val_labels_orig.append(y_orig)

        normal_mask = y_bin == 0
        if torch.any(normal_mask):
            normal_loss = self.loss_fn(x_hat[normal_mask], target[normal_mask])
            self.log(
                "val_normal_loss",
                normal_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=int(normal_mask.sum().item()),
            )

    def on_validation_epoch_end(self) -> None:
        if not self._val_residuals:
            return
        residuals = np.concatenate(self._val_residuals)
        context_dev = np.concatenate(self._val_context_dev)
        labels_bin = np.concatenate(self._val_labels_bin)
        labels_orig = np.concatenate(self._val_labels_orig)
        self._val_residuals.clear()
        self._val_context_dev.clear()
        self._val_labels_bin.clear()
        self._val_labels_orig.clear()

        # Always log val_macro_per_class_pr_auc — Lightning's ModelCheckpoint/EarlyStopping
        # crash during the pre-training sanity check if the monitored metric is absent even once.
        val_pr_auc_val = 0.0
        macro_pr_auc_val = 0.0
        per_class: dict = {}
        if self._groups and np.unique(labels_bin).size >= 2:
            # Use unscaled residuals + context_dev for monitoring (MAD scale fit after full training).
            # context group catches sustained faults where next-step residual is small.
            group_scores = _compute_group_scores(residuals, self._groups)
            group_scores["context"] = context_dev.mean(axis=1)
            val_scores = _reduce_scores(group_scores, self._score_reduction)
            val_pr_auc_val = _safe_pr_auc(labels_bin, val_scores) or 0.0
            macro = compute_macro_per_class_pr_auc(labels=labels_orig, scores=val_scores)
            macro_pr_auc_val = macro.get("macro_per_class_pr_auc") or 0.0
            per_class = macro.get("per_class_pr_auc_vs_normal", {})
        self.log("val_pr_auc", val_pr_auc_val, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_macro_per_class_pr_auc", macro_pr_auc_val, prog_bar=True, on_step=False, on_epoch=True)
        for k in ("1", "2", "3", "4"):
            v = per_class.get(k)
            if v is not None:
                self.log(f"val_class{k}_pr_auc_vs_normal", v, prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


@torch.no_grad()
def _infer_residuals(
    model: PVGDN,
    dl: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (residuals, labels_bin, labels_orig, group_ids, context_dev).

    context_dev [N, F]: mean |Z-scored context| per feature per window.
    For normal windows ≈ 0.8 (expected |N(0,1)|); for sustained faults >> 0.8.
    Used as an additional scoring group to catch faults that next-step prediction misses.
    """
    model.eval()
    all_residuals: list[np.ndarray] = []
    all_context_dev: list[np.ndarray] = []
    all_labels_bin: list[np.ndarray] = []
    all_labels_orig: list[np.ndarray] = []
    all_group_ids: list[np.ndarray] = []

    for batch in dl:
        x = batch[0].to(device)
        y_bin = batch[1].detach().cpu().numpy().astype(int)
        y_orig = batch[2].detach().cpu().numpy() if len(batch) > 2 else y_bin.copy()
        group_ids = np.asarray(batch[3]) if len(batch) > 3 else np.array(["row"] * len(y_bin), dtype=object)

        context = x[:, :-1, :]
        target = x[:, -1, :]
        x_hat, _ = model(context)
        residuals = (x_hat - target).abs().detach().cpu().numpy()
        context_dev = context.abs().mean(dim=1).detach().cpu().numpy()  # [B, F]

        all_residuals.append(residuals)
        all_context_dev.append(context_dev)
        all_labels_bin.append(y_bin)
        all_labels_orig.append(y_orig)
        all_group_ids.append(group_ids)

    residuals = np.concatenate(all_residuals) if all_residuals else np.array([])
    context_dev = np.concatenate(all_context_dev) if all_context_dev else np.array([])
    labels_bin = np.concatenate(all_labels_bin) if all_labels_bin else np.array([])
    labels_orig = np.concatenate(all_labels_orig) if all_labels_orig else np.array([])
    group_ids = np.concatenate(all_group_ids) if all_group_ids else np.array([])
    return residuals, labels_bin, labels_orig, group_ids, context_dev


def run_pv_gdn(config: dict | None = None) -> None:
    args = _parse_args()
    if config is None:
        config = _load_config()

    dl_cfg = config["anomaly_detection"]["dl"]["models"]["pv_gdn"]
    train_cfg = config.get("training", {})
    seed = int(args.seed)
    pl.seed_everything(seed, workers=True)

    train_df, val_df, test_df, manifest, resolved_run_dir = load_features_for_task(
        task=args.task,
        profile=args.profile,
        run_dir=args.run_dir,
        run_id=args.run_id,
        dataset=args.dataset,
        split_path=args.split_path,
    )
    features: list[str] = manifest.get("final_features", [])
    label_col: str = str(manifest.get("label_column", "label"))

    y_train = (train_df[label_col].to_numpy() != 0).astype(int)
    if int(y_train.sum()) > 0:
        logger.warning("Train contains non-normal rows; semisup expected normal-only train")

    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])

    win_size = int(args.win_size or dl_cfg.get("win_size", 30))
    train_stride = int(dl_cfg.get("train_stride", 5))
    eval_stride = int(dl_cfg.get("eval_stride", 1))
    batch_size = int(dl_cfg.get("batch_size", train_cfg.get("batch_size", 512)))
    lr = float(dl_cfg.get("learning_rate", 1e-3))
    wd = float(dl_cfg.get("weight_decay", 1e-5))
    max_epochs = 3 if args.smoke else int(train_cfg.get("max_epochs", 100))
    patience = 3 if args.smoke else int(train_cfg.get("patience", 10))
    score_reduction = str(args.score_reduction or dl_cfg.get("score_reduction", "max_group"))

    ds_train = TimeSeriesDataset(
        train_df,
        features,
        label_col,
        win_size,
        stride=train_stride,
        normal_only=True,
        target_position="last",
    )
    ds_val = TimeSeriesDataset(
        val_df,
        features,
        label_col,
        win_size,
        stride=eval_stride,
        normal_only=False,
        return_original_label=True,
        return_group_id=True,
        target_position="last",
    )
    ds_test = TimeSeriesDataset(
        test_df,
        features,
        label_col,
        win_size,
        stride=eval_stride,
        normal_only=False,
        return_original_label=True,
        return_group_id=True,
        target_position="last",
    )
    groups = _feature_groups(features)
    group_membership = {k: [features[i] for i in ids] for k, ids in groups.items()}
    _reduce_scores({name: np.zeros(1, dtype=float) for name in groups}, score_reduction)

    if len(ds_train) == 0:
        raise ValueError("PV-GDN train dataset produced zero windows; adjust win_size/stride or input splits")
    if len(ds_val) == 0:
        raise ValueError("PV-GDN val dataset produced zero windows; adjust win_size/stride or input splits")
    if len(ds_test) == 0:
        raise ValueError("PV-GDN test dataset produced zero windows; adjust win_size/stride or input splits")
    if not _has_any_normal_windows(ds_val):
        raise ValueError(
            "PV-GDN val dataset produced no normal-centered windows; cannot monitor val_normal_loss"
        )

    train_dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)
    logger.info("Windows | train={} val={} test={}", len(ds_train), len(ds_val), len(ds_test))

    model = PVGDN(
        n_features=len(features),
        d_model=int(dl_cfg.get("d_model", 64)),
        temporal_kernel_size=int(dl_cfg.get("temporal_kernel_size", 3)),
        temporal_layers=int(dl_cfg.get("temporal_layers", 2)),
        graph_top_k=int(dl_cfg.get("graph_top_k", 4)),
        dropout=float(dl_cfg.get("dropout", 0.1)),
    )
    lit = PVGDNLightning(
        model=model, learning_rate=lr, weight_decay=wd,
        groups=groups, score_reduction=score_reduction,
    )

    ckpt_cb = ModelCheckpoint(monitor="val_macro_per_class_pr_auc", mode="max", save_top_k=1)
    es_cb = EarlyStopping(monitor="val_macro_per_class_pr_auc", mode="max", patience=patience)
    accelerator = dl_cfg.get("accelerator", train_cfg.get("accelerator", "auto"))
    devices = dl_cfg.get("devices", train_cfg.get("devices", "auto"))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[ckpt_cb, es_cb],
        deterministic=False,
        enable_progress_bar=True,
        accelerator=accelerator,
        devices=devices,
        precision=dl_cfg.get("precision", train_cfg.get("precision", "32-true")),
        gradient_clip_val=float(train_cfg.get("gradient_clip_val", 1.0)),
        log_every_n_steps=50,
    )

    t0 = time.perf_counter()
    trainer.fit(lit, train_dataloaders=train_dl, val_dataloaders=val_dl)
    fit_time = time.perf_counter() - t0

    best_path = ckpt_cb.best_model_path
    if best_path:
        best_model = PVGDN(
            n_features=len(features),
            d_model=int(dl_cfg.get("d_model", 64)),
            temporal_kernel_size=int(dl_cfg.get("temporal_kernel_size", 3)),
            temporal_layers=int(dl_cfg.get("temporal_layers", 2)),
            graph_top_k=int(dl_cfg.get("graph_top_k", 4)),
            dropout=float(dl_cfg.get("dropout", 0.1)),
        )
        lit = PVGDNLightning.load_from_checkpoint(
            best_path, model=best_model, learning_rate=lr, weight_decay=wd,
            groups=groups, score_reduction=score_reduction,
        )

    trainer_device = getattr(trainer.strategy, "root_device", None)
    inferred_device = trainer_device if isinstance(trainer_device, torch.device) else _resolve_inference_device(accelerator)
    lit = lit.to(inferred_device)
    lit.model = lit.model.to(inferred_device)
    logger.info("PV-GDN groups: {}", group_membership)
    train_residuals, _, _, _, train_context_dev = _infer_residuals(lit.model, train_dl, inferred_device)
    residual_scale = _fit_residual_scale(train_residuals)
    context_dev_scale = _fit_residual_scale(train_context_dev)
    val_residuals, val_labels_bin, val_labels_orig, val_group_ids, val_context_dev = _infer_residuals(
        lit.model, val_dl, inferred_device
    )
    test_residuals, test_labels_bin, test_labels_orig, test_group_ids, test_context_dev = _infer_residuals(
        lit.model, test_dl, inferred_device
    )

    train_residuals_scaled = _apply_residual_scale(train_residuals, residual_scale)
    val_residuals_scaled = _apply_residual_scale(val_residuals, residual_scale)
    test_residuals_scaled = _apply_residual_scale(test_residuals, residual_scale)
    train_context_dev_scaled = _apply_residual_scale(train_context_dev, context_dev_scale)
    val_context_dev_scaled = _apply_residual_scale(val_context_dev, context_dev_scale)
    test_context_dev_scaled = _apply_residual_scale(test_context_dev, context_dev_scale)

    train_group_scores = _compute_group_scores(train_residuals_scaled, groups)
    val_group_scores = _compute_group_scores(val_residuals_scaled, groups)
    test_group_scores = _compute_group_scores(test_residuals_scaled, groups)
    # "context" group: mean MAD-scaled |Z-scored context| — catches sustained faults
    train_group_scores["context"] = train_context_dev_scaled.mean(axis=1)
    val_group_scores["context"] = val_context_dev_scaled.mean(axis=1)
    test_group_scores["context"] = test_context_dev_scaled.mean(axis=1)

    val_scores = _reduce_scores(val_group_scores, score_reduction)
    test_scores = _reduce_scores(test_group_scores, score_reduction)

    threshold, val_f1, val_prec, val_rec = _calibrate_threshold(val_scores, val_labels_bin)
    val_preds = (val_scores >= threshold).astype(int)
    test_preds = (test_scores >= threshold).astype(int)

    val_pr_auc = _safe_pr_auc(val_labels_bin, val_scores)
    test_pr_auc = _safe_pr_auc(test_labels_bin, test_scores)
    val_roc_auc = _safe_roc_auc(val_labels_bin, val_scores)
    test_roc_auc = _safe_roc_auc(test_labels_bin, test_scores)
    val_acc = float(accuracy_score(val_labels_bin, val_preds))
    test_acc = float(accuracy_score(test_labels_bin, test_preds))
    test_f1 = float(f1_score(test_labels_bin, test_preds, zero_division=0))
    test_prec = float(precision_score(test_labels_bin, test_preds, zero_division=0))
    test_rec = float(recall_score(test_labels_bin, test_preds, zero_division=0))

    val_episode_macro_f1 = episode_macro_f1_binary(val_labels_bin, val_preds, val_group_ids)
    test_episode_macro_f1 = episode_macro_f1_binary(test_labels_bin, test_preds, test_group_ids)

    val_macro = compute_macro_per_class_pr_auc(labels=val_labels_orig, scores=val_scores)
    test_macro = compute_macro_per_class_pr_auc(labels=test_labels_orig, scores=test_scores)

    metrics = {
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "val_f1_at_threshold": float(val_f1),
        "val_accuracy_at_threshold": val_acc,
        "val_precision_at_threshold": float(val_prec),
        "val_recall_at_threshold": float(val_rec),
        "val_episode_macro_f1": float(val_episode_macro_f1),
        "val_macro_per_class_pr_auc": val_macro.get("macro_per_class_pr_auc"),
        "val_worst_class_pr_auc": val_macro.get("worst_class_pr_auc"),
        "val_selection_score": val_macro.get("macro_per_class_pr_auc") or 0.0,
        "threshold": float(threshold),
        "threshold_policy": "validation_pr_curve_f1",
        "score_reduction": score_reduction,
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_f1_at_threshold": test_f1,
        "test_accuracy_at_threshold": test_acc,
        "test_precision_at_threshold": test_prec,
        "test_recall_at_threshold": test_rec,
        "test_episode_macro_f1": float(test_episode_macro_f1),
        "test_macro_per_class_pr_auc": test_macro.get("macro_per_class_pr_auc"),
        "test_worst_class_pr_auc": test_macro.get("worst_class_pr_auc"),
        "test_class1_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("1"),
        "test_class2_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("2"),
        "test_class3_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("3"),
        "test_class4_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("4"),
        "n_train_total": int(len(train_df)),
        "n_train_windows": int(len(ds_train)),
        "fit_time_s": round(fit_time, 2),
    }

    cls1_mask = (test_labels_orig == 0) | (test_labels_orig == 1)
    cls1_y = (test_labels_orig[cls1_mask] == 1).astype(int)
    cls2_mask = (test_labels_orig == 0) | (test_labels_orig == 2)
    cls2_y = (test_labels_orig[cls2_mask] == 2).astype(int)
    for g_name, g_scores in test_group_scores.items():
        metrics[f"test_pr_auc_group_{g_name}"] = _safe_pr_auc(test_labels_bin, g_scores)
        metrics[f"test_class1_pr_auc_group_{g_name}"] = _safe_pr_auc(cls1_y, g_scores[cls1_mask])
        metrics[f"test_class2_pr_auc_group_{g_name}"] = _safe_pr_auc(cls2_y, g_scores[cls2_mask])

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else get_experiments_root() / "anomaly" / "pv_gdn" / f"pv_gdn_{ts}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.ckpt"
    scaler_path = artifacts_dir / "scaler.joblib"
    residual_scale_path = artifacts_dir / "residual_scale.json"
    global_metrics_path = artifacts_dir / "global_metrics.json"
    per_class_metrics_path = artifacts_dir / "per_class_metrics.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    deployment_manifest_path = artifacts_dir / "deployment_manifest.json"
    features_manifest_path = artifacts_dir / "features_manifest.json"
    score_calibration_path = artifacts_dir / "score_calibration.json"
    group_diagnostics_path = artifacts_dir / "group_diagnostics.json"
    graph_edges_path = artifacts_dir / "graph_top_edges.json"

    if best_path:
        model_path.write_bytes(Path(best_path).read_bytes())
    joblib.dump(scaler, scaler_path)
    write_json(
        residual_scale_path,
        {
            "feature_names": features,
            "method": "mad",
            "epsilon_floor": 1e-6,
            "residual_scale": {f: float(s) for f, s in zip(features, residual_scale, strict=False)},
            "context_dev_scale": {f: float(s) for f, s in zip(features, context_dev_scale, strict=False)},
        },
    )
    write_json(
        graph_edges_path,
        {
            "top_edges": _extract_top_graph_edges(lit.model, features),
            "graph_top_k": int(dl_cfg.get("graph_top_k", 4)),
        },
    )

    per_class_metrics = compute_anomaly_per_class_metrics(
        labels=test_labels_orig,
        scores=test_scores,
        threshold=threshold,
        val_labels=val_labels_orig,
        val_scores=val_scores,
    )
    candidate_thresholds = build_candidate_per_true_class_thresholds(per_class_metrics, normal_label=0)

    run_name = f"anomaly_pv_gdn_{ts}"
    run_manifest = build_run_manifest(
        task=args.task,
        model="pv_gdn",
        model_family="anomaly_dl",
        dataset=args.dataset,
        split_path=args.split_path,
        feature_profile=str(args.profile),
        feature_run_dir=str(resolved_run_dir),
        seed=seed,
        run_type="smoke" if args.smoke else args.run_type,
        extras={"run_name": run_name, "win_size": win_size, "score_reduction": score_reduction},
    )
    deployment_manifest = build_deployment_manifest(
        task=args.task,
        model="pv_gdn",
        model_family="anomaly_dl",
        model_artifact=model_path.name,
        scaler_artifact=scaler_path.name,
        feature_names=features,
        label_column=label_col,
        threshold=threshold,
        window_size=win_size,
        score_direction="higher_is_more_anomalous",
        classes=[str(c) for c in sorted(np.unique(test_labels_orig).tolist())],
        extras={
            "threshold_policy": "validation_pr_curve_f1",
            "score_calibration_artifact": score_calibration_path.name,
            "score_reduction": score_reduction,
            "group_diagnostics_artifact": group_diagnostics_path.name,
            "residual_scale_artifact": residual_scale_path.name,
            "target_position": "last",
            "context_window_size": win_size - 1,
            "prediction_horizon": 1,
        },
    )
    write_json(
        score_calibration_path,
        build_score_calibration_payload(
            threshold=threshold,
            threshold_policy="validation_pr_curve_f1",
            threshold_quantile=None,
            candidate_per_true_class_thresholds=candidate_thresholds,
        ),
    )
    write_json(global_metrics_path, metrics)
    write_json(per_class_metrics_path, per_class_metrics)
    write_json(
        group_diagnostics_path,
        {
            "group_membership": group_membership,
            "score_reduction": score_reduction,
            "target_position": "last",
            "context_window_size": win_size - 1,
            "train_group_score_median": {
                g: float(np.median(s)) for g, s in train_group_scores.items()
            },
            "val_group_pr_auc": {g: _safe_pr_auc(val_labels_bin, s) for g, s in val_group_scores.items()},
            "test_group_pr_auc": {g: _safe_pr_auc(test_labels_bin, s) for g, s in test_group_scores.items()},
            "test_class1_group_pr_auc": {
                g: _safe_pr_auc(cls1_y, s[cls1_mask]) for g, s in test_group_scores.items()
            },
            "test_class2_group_pr_auc": {
                g: _safe_pr_auc(cls2_y, s[cls2_mask]) for g, s in test_group_scores.items()
            },
        },
    )
    write_json(run_manifest_path, run_manifest)
    write_json(deployment_manifest_path, deployment_manifest)
    write_json(features_manifest_path, manifest)

    try:
        init_tracking("anomaly")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"task": args.task, "dataset": args.dataset, "split_path": args.split_path, "profile": str(args.profile), "model": "pv_gdn"})
            mlflow.log_params(
                {
                    "n_features": len(features),
                    "win_size": win_size,
                    "train_stride": train_stride,
                    "eval_stride": eval_stride,
                    "d_model": int(dl_cfg.get("d_model", 64)),
                    "temporal_kernel_size": int(dl_cfg.get("temporal_kernel_size", 3)),
                    "temporal_layers": int(dl_cfg.get("temporal_layers", 2)),
                    "graph_top_k": int(dl_cfg.get("graph_top_k", 4)),
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "seed": seed,
                    "score_reduction": score_reduction,
                    "target_position": "last",
                    "context_window_size": win_size - 1,
                    "prediction_horizon": 1,
                }
            )
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)})
            for cls_str, m in per_class_metrics.items():
                if m.get("pr_auc_vs_normal") is not None:
                    mlflow.log_metric(f"test_pr_auc_class{cls_str}_vs_normal", m["pr_auc_vs_normal"])
                if m.get("per_class_threshold") is not None:
                    mlflow.log_metric(f"test_f1_class{cls_str}_per_class", m["f1_at_per_class_threshold"])
                    mlflow.log_metric(f"test_f1_class{cls_str}_global", m["f1_at_threshold_vs_normal"])
                    mlflow.log_metric(f"per_class_threshold_class{cls_str}", m["per_class_threshold"])
            for p in (
                global_metrics_path,
                per_class_metrics_path,
                group_diagnostics_path,
                run_manifest_path,
                deployment_manifest_path,
                features_manifest_path,
                score_calibration_path,
                scaler_path,
                residual_scale_path,
                graph_edges_path,
                model_path,
            ):
                if p.exists():
                    mlflow.log_artifact(str(p))

            comparison_record = {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "run_name": run_name,
                "task": args.task,
                "dataset": args.dataset,
                "split_path": args.split_path,
                "model": "pv_gdn",
                "model_family": "anomaly_dl",
                "run_type": "smoke" if args.smoke else args.run_type,
                "seed": seed,
                "feature_profile": str(args.profile),
                "feature_run_dir": str(resolved_run_dir),
                "n_features": len(features),
                "fit_time_s": round(fit_time, 2),
                "threshold": threshold,
                "score_reduction": score_reduction,
                "context_window_size": win_size - 1,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "val_f1_at_threshold": float(val_f1),
                "val_macro_per_class_pr_auc": val_macro.get("macro_per_class_pr_auc"),
                "val_worst_class_pr_auc": val_macro.get("worst_class_pr_auc"),
                "test_pr_auc": test_pr_auc,
                "test_roc_auc": test_roc_auc,
                "test_macro_per_class_pr_auc": test_macro.get("macro_per_class_pr_auc"),
                "test_worst_class_pr_auc": test_macro.get("worst_class_pr_auc"),
                "test_class1_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("1"),
                "test_class2_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("2"),
                "test_class3_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("3"),
                "test_class4_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("4"),
                "test_f1_at_threshold": test_f1,
                "test_precision_at_threshold": test_prec,
                "test_recall_at_threshold": test_rec,
                "test_accuracy_at_threshold": test_acc,
                "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
            }
            for g_name in test_group_scores:
                comparison_record[f"test_pr_auc_group_{g_name}"] = metrics.get(f"test_pr_auc_group_{g_name}")
                comparison_record[f"test_class2_pr_auc_group_{g_name}"] = metrics.get(
                    f"test_class2_pr_auc_group_{g_name}"
                )
            for k in ("1", "2", "3", "4"):
                pcm = per_class_metrics.get(k, {})
                comparison_record[f"test_class{k}_f1_at_per_class_threshold"] = pcm.get("f1_at_per_class_threshold")
                comparison_record[f"test_class{k}_f1_at_global_threshold"] = pcm.get("f1_at_threshold_vs_normal")
                comparison_record[f"test_class{k}_per_class_threshold"] = pcm.get("per_class_threshold")
            records_path = Path(args.comparison_records_path)
            records_path.parent.mkdir(parents=True, exist_ok=True)
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(comparison_record, default=str) + "\n")
            mlflow.log_artifact(str(records_path))
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): {}", exc)


if __name__ == "__main__":
    run_pv_gdn()
