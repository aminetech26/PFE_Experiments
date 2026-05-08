from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
from torch.utils.data import DataLoader

from src.mlflow_setup import init_tracking
from src.modeling.anomaly_detection.dl.dataset import TimeSeriesDataset
from src.modeling.anomaly_detection.dl.dlssm.losses import (
    compute_anomaly_scores,
    kl_divergence,
    physics_consistency_loss,
    prediction_loss,
    reconstruction_loss,
    self_paced_weights,
)
from src.modeling.anomaly_detection.dl.dlssm.model import DeepLatentStateSpaceModel
from src.modeling.common.feature_loader import load_features_for_task
from src.utils.paths import get_experiments_root

# trainer.py is under dl/dlssm/, so PROJECT_ROOT is 5 levels up
# path: dlssm/ → dl/ → anomaly_detection/ → modeling/ → src/ → PFE_Experiments/
PROJECT_ROOT = Path(__file__).resolve().parents[5]

torch.set_float32_matmul_precision("medium")


def _default_comparison_records_path() -> Path:
    return get_experiments_root() / "metrics" / "anomaly_comparison_records.jsonl"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PI-SP-DLS-SSM anomaly detection")
    p.add_argument("--task", default="anomaly_semisup")
    p.add_argument("--dataset", default="costa")
    p.add_argument("--split-path", default="path_a")
    p.add_argument("--profile", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--smoke", action="store_true", help="Smoke test: 1 epoch, large stride")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-type", default="baseline", help="baseline | ablation | final | smoke")
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--comparison-records-path", default=str(_default_comparison_records_path()))
    p.add_argument("--physics", action="store_true", help="Enable physics consistency loss")
    p.add_argument("--self-paced", action="store_true", help="Enable self-paced curriculum weighting")
    return p.parse_args()


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _calibrate_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
    """Return (threshold, best_f1, precision, recall) by maximising F1 on PR curve."""
    prec, rec, thresholds = precision_recall_curve(labels, scores)
    denom = prec[:-1] + rec[:-1]
    safe_denom = np.where(denom > 0, denom, 1.0)
    f1_vals = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / safe_denom, 0.0)
    best_idx = int(np.argmax(f1_vals))
    return (
        float(thresholds[best_idx]),
        float(f1_vals[best_idx]),
        float(prec[best_idx]),
        float(rec[best_idx]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class DLSSMLightningModule(pl.LightningModule):
    """PI-SP-DLS-SSM training module.

    Uses standard automatic optimization (single ELBO loss with physics and
    self-paced extensions) — no manual backward like MAAT.
    """

    def __init__(
        self,
        model: DeepLatentStateSpaceModel,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
        feature_idx: dict[str, int],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        gradient_clip_val: float = 1.0,
        max_epochs: int = 30,
        beta_kl: float = 0.1,
        kl_warmup_epochs: int = 5,
        lambda_phys: float = 0.05,
        enable_string_power: bool = True,
        enable_imbalance: bool = True,
        self_paced_enabled: bool = True,
        physics_enabled: bool = True,
        tau_start: float = 0.1,
        tau_end: float = 1.0,
        w_min: float = 0.2,
        lambda_kl_score: float = 0.1,
        lambda_phys_score: float = 0.0,
        alpha_pred: float = 0.1,
        lambda_pred_score: float = 0.0,
        score_reduction: str = "center",
        free_bits: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)
        self.feature_idx = feature_idx

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.max_epochs = max_epochs
        self.beta_kl = beta_kl
        self.kl_warmup_epochs = kl_warmup_epochs
        self.lambda_phys = lambda_phys
        self.enable_string_power = enable_string_power
        self.enable_imbalance = enable_imbalance
        self.physics_enabled = physics_enabled
        self.self_paced_enabled = self_paced_enabled
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.w_min = w_min
        self.lambda_kl_score = lambda_kl_score
        self.lambda_phys_score = lambda_phys_score
        self.alpha_pred = alpha_pred
        self.lambda_pred_score = lambda_pred_score
        self.score_reduction = score_reduction
        self.free_bits = free_bits

        self._val_outputs: list[dict] = []
        self._test_outputs: list[dict] = []
        self.best_val_pr_auc: float = 0.0
        self.val_threshold: float = 0.5
        self._val_scores_np: np.ndarray | None = None
        self._val_labels_np: np.ndarray | None = None
        self._test_scores_np: np.ndarray | None = None
        self._test_labels_np: np.ndarray | None = None

    def _physics_loss(self, x_hat: torch.Tensor) -> torch.Tensor:
        if not self.physics_enabled:
            return torch.zeros(x_hat.size(0), device=x_hat.device, dtype=x_hat.dtype)
        return physics_consistency_loss(
            x_hat,
            self.scaler_mean,
            self.scaler_scale,
            self.feature_idx,
            enable_string_power=self.enable_string_power,
            enable_imbalance=self.enable_imbalance,
        )

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch  # [B, W, F]
        out = self.model(x)

        # Per-window loss components [B]
        recon_w = reconstruction_loss(out["x_hat"], x).mean(dim=1)
        pred_w = prediction_loss(out["x_pred"], x).mean(dim=1)
        kl_w = kl_divergence(
            out["q_mu"], out["q_logvar"], out["p_mu"], out["p_logvar"],
            free_bits=self.free_bits,
        ).mean(dim=1)
        phys_w = self._physics_loss(out["x_hat"])

        # KL warmup: ramp beta_kl from 0 to beta_kl over kl_warmup_epochs
        beta = self.beta_kl * min(1.0, (self.current_epoch + 1) / max(self.kl_warmup_epochs, 1))

        # SPL tau schedule: anneal from tau_start (focus easy) to tau_end (include hard)
        progress = self.current_epoch / max(self.max_epochs - 1, 1)
        tau = self.tau_start * (self.tau_end / max(self.tau_start, 1e-6)) ** progress

        # Self-paced weights
        if self.self_paced_enabled:
            w = self_paced_weights(recon_w, phys_w, self.lambda_phys, tau, self.w_min)
        else:
            w = torch.ones_like(recon_w)

        lambda_phys = self.lambda_phys if self.physics_enabled else 0.0
        per_window = recon_w + self.alpha_pred * pred_w + beta * kl_w + lambda_phys * phys_w
        loss = (w * per_window).mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon", recon_w.mean(), on_step=False, on_epoch=True)
        self.log("train_pred", pred_w.mean(), on_step=False, on_epoch=True)
        self.log("train_kl", kl_w.mean(), on_step=False, on_epoch=True)
        self.log("train_phys", phys_w.mean(), on_step=False, on_epoch=True)
        self.log("train_beta_kl", beta, on_step=False, on_epoch=True)
        self.log("train_tau", tau, on_step=False, on_epoch=True)
        self.log("train_w_mean", w.mean(), on_step=False, on_epoch=True)
        self.log("train_w_std", w.std(), on_step=False, on_epoch=True)
        self.log("train_w_min", w.min(), on_step=False, on_epoch=True)
        self.log("train_w_max", w.max(), on_step=False, on_epoch=True)
        # Effective sample fraction: fraction of windows with weight above mean
        eff_frac = (w > w.mean()).float().mean()
        self.log("train_w_eff_frac", eff_frac, on_step=False, on_epoch=True)
        return loss

    def _score_batch(self, x: torch.Tensor, out: dict) -> torch.Tensor:
        return compute_anomaly_scores(
            x, out["x_hat"], out["q_mu"], out["q_logvar"],
            out["p_mu"], out["p_logvar"],
            lambda_kl_score=self.lambda_kl_score,
            score_reduction=self.score_reduction,
            lambda_phys_score=self.lambda_phys_score,
            scaler_mean=self.scaler_mean if self.lambda_phys_score > 0.0 else None,
            scaler_scale=self.scaler_scale if self.lambda_phys_score > 0.0 else None,
            feature_idx=self.feature_idx if self.lambda_phys_score > 0.0 else None,
            enable_string_power=self.enable_string_power,
            enable_imbalance=self.enable_imbalance,
            x_pred=out["x_pred"],
            lambda_pred_score=self.lambda_pred_score,
        )

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, labels = batch
        with torch.no_grad():
            out = self.model(x)
        scores = self._score_batch(x, out).cpu()
        self._val_outputs.append({"scores": scores, "labels": labels.cpu()})

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return
        all_scores = torch.cat([o["scores"] for o in self._val_outputs]).numpy()
        all_labels = torch.cat([o["labels"] for o in self._val_outputs]).numpy()
        self._val_outputs.clear()

        if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
            self.log("val_pr_auc", 0.0, prog_bar=True)
            return

        val_pr_auc = float(average_precision_score(all_labels, all_scores))
        val_roc_auc = float(roc_auc_score(all_labels, all_scores))
        threshold, val_f1, val_prec, val_rec = _calibrate_threshold(all_scores, all_labels)
        val_preds = (all_scores >= threshold).astype(int)
        val_acc = float(accuracy_score(all_labels, val_preds))

        self.best_val_pr_auc = max(self.best_val_pr_auc, val_pr_auc)
        self.val_threshold = threshold
        self._val_scores_np = all_scores
        self._val_labels_np = all_labels

        self.log("val_pr_auc", val_pr_auc, prog_bar=True)
        self.log("val_roc_auc", val_roc_auc)
        self.log("val_f1_at_threshold", val_f1)
        self.log("val_precision_at_threshold", val_prec)
        self.log("val_recall_at_threshold", val_rec)
        self.log("val_accuracy_at_threshold", val_acc)
        self.log("val_threshold", threshold)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, labels = batch
        with torch.no_grad():
            out = self.model(x)
        scores = self._score_batch(x, out).cpu()
        self._test_outputs.append({"scores": scores, "labels": labels.cpu()})

    def on_test_epoch_end(self) -> None:
        if not self._test_outputs:
            return
        all_scores = torch.cat([o["scores"] for o in self._test_outputs]).numpy()
        all_labels = torch.cat([o["labels"] for o in self._test_outputs]).numpy()
        self._test_outputs.clear()
        self._test_scores_np = all_scores
        self._test_labels_np = all_labels

        if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
            self.log("test_pr_auc", 0.0)
            return

        test_pr_auc = float(average_precision_score(all_labels, all_scores))
        test_roc_auc = float(roc_auc_score(all_labels, all_scores))
        test_preds = (all_scores >= self.val_threshold).astype(int)
        test_f1 = float(f1_score(all_labels, test_preds, zero_division=0))
        test_acc = float(accuracy_score(all_labels, test_preds))
        test_prec = float(precision_score(all_labels, test_preds, zero_division=0))
        test_rec = float(recall_score(all_labels, test_preds, zero_division=0))

        self.log("test_pr_auc", test_pr_auc, prog_bar=True)
        self.log("test_roc_auc", test_roc_auc)
        self.log("test_f1_at_threshold", test_f1)
        self.log("test_accuracy_at_threshold", test_acc)
        self.log("test_precision_at_threshold", test_prec)
        self.log("test_recall_at_threshold", test_rec)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=self.learning_rate * 0.01
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_pr_curve(
    val_scores: np.ndarray, val_labels: np.ndarray,
    test_scores: np.ndarray, test_labels: np.ndarray,
    path: Path, model_name: str = "DLS-SSM",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for scores, labels, split in [
        (val_scores, val_labels, "Val"),
        (test_scores, test_labels, "Test"),
    ]:
        prec, rec, _ = precision_recall_curve(labels, scores)
        auc = average_precision_score(labels, scores)
        ax.plot(rec, prec, label=f"{split} (PR-AUC={auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_score_histogram(
    val_scores: np.ndarray, val_labels: np.ndarray,
    threshold: float, path: Path, model_name: str = "DLS-SSM",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(val_scores[val_labels == 0], bins=60, alpha=0.5, label="Val — normal", density=True)
    if val_labels.sum() > 0:
        ax.hist(val_scores[val_labels == 1], bins=60, alpha=0.5, label="Val — fault", density=True)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.set_title(f"Anomaly Score Distribution — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_score_timeline(
    test_scores: np.ndarray, test_labels: np.ndarray,
    threshold: float, path: Path, model_name: str = "DLS-SSM",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = np.where(test_labels == 1, "red", "steelblue")
    ax.scatter(np.arange(len(test_scores)), test_scores, c=colors, s=2, alpha=0.5, rasterized=True)
    ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Test window index")
    ax.set_ylabel("Anomaly score")
    ax.set_title(f"Score Timeline (Test) — {model_name}  |  blue=normal  red=fault")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _save_prediction_residual_plot(
    lit: "DLSSMLightningModule",
    test_dl: DataLoader,
    features: list[str],
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    path: Path,
    model_name: str = "DLS-SSM",
) -> None:
    """Visualize x_pred vs x_hat vs x for the highest-scoring fault window.

    The point: show that the GRU state predicts what *should* be happening
    (x_pred), and the latent residual (x_hat - x_pred) carries the deviation
    that flags the anomaly. This is the qualitative interpretability story
    that recon-only autoencoders cannot tell.
    """
    fault_idx = np.where(test_labels == 1)[0]
    if len(fault_idx) == 0:
        logger.warning("No fault windows in test set — skipping prediction/residual plot")
        return

    target_idx = int(fault_idx[np.argmax(test_scores[fault_idx])])

    # Iterate test_dl (no shuffle) to find the matching window
    target_window: torch.Tensor | None = None
    counter = 0
    for x_batch, _ in test_dl:
        bs = x_batch.size(0)
        if counter + bs > target_idx:
            target_window = x_batch[target_idx - counter : target_idx - counter + 1]
            break
        counter += bs
    if target_window is None:
        logger.warning("Failed to locate target fault window — skipping plot")
        return

    device = next(lit.model.parameters()).device
    target_window = target_window.to(device)
    lit.model.eval()
    with torch.no_grad():
        out = lit.model(target_window)
    x_np = target_window[0].cpu().numpy()
    x_pred_np = out["x_pred"][0].cpu().numpy()
    x_hat_np = out["x_hat"][0].cpu().numpy()

    # Inverse-standardize for physical interpretability
    mean = lit.scaler_mean.cpu().numpy()
    scale = lit.scaler_scale.cpu().numpy()
    x_phys = x_np * scale + mean
    x_pred_phys = x_pred_np * scale + mean
    x_hat_phys = x_hat_np * scale + mean

    # Choose key PV features if present; otherwise fall back to first 4
    preferred = ["pdc1", "vdc1", "idc1", "power_imbalance",
                 "pdc2", "vdc2", "idc2", "current_imbalance", "voltage_imbalance"]
    chosen = [f for f in preferred if f in features][:4]
    if len(chosen) < 4:
        chosen = features[:4]

    # Skip t=0 — h_prev[:, 0] = 0 makes the first prediction degenerate
    t = np.arange(1, x_phys.shape[0])

    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 2.5 * len(chosen)), sharex=True)
    if len(chosen) == 1:
        axes = [axes]

    for ax, feat in zip(axes, chosen, strict=False):
        idx = features.index(feat)
        ax.plot(t, x_phys[1:, idx], color="black", linewidth=1.8, label="observed x")
        ax.plot(t, x_pred_phys[1:, idx], color="steelblue", linestyle="--",
                linewidth=1.5, alpha=0.9, label="x_pred (state-only)")
        ax.plot(t, x_hat_phys[1:, idx], color="firebrick", linestyle=":",
                linewidth=1.5, alpha=0.9, label="x_hat (state + residual)")
        ax.set_ylabel(feat)
        ax.grid(True, alpha=0.3)

    score = float(test_scores[target_idx])
    axes[0].set_title(
        f"Prediction / Residual Decomposition — {model_name}\n"
        f"Highest-scoring fault window (test_idx={target_idx}, score={score:.3f}). "
        f"Gap between observed and x_pred is what z_t must explain."
    )
    axes[0].legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("Timestep within window")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_dlssm(config: dict | None = None) -> None:
    args = _parse_args()
    if config is None:
        config = _load_config()

    dlssm_cfg: dict = config["anomaly_detection"]["dl"]["models"]["dlssm"]
    seed: int = args.seed
    is_smoke: bool = args.smoke

    physics_enabled: bool = args.physics or bool(dlssm_cfg.get("physics_enabled", False))
    self_paced_enabled: bool = args.self_paced or bool(dlssm_cfg.get("self_paced", False))

    run_type = "smoke" if is_smoke else args.run_type

    # Derive a human-readable variant name for logging
    if physics_enabled and self_paced_enabled:
        variant = "PI-SP-DLS-SSM"
    elif physics_enabled:
        variant = "PI-DLS-SSM"
    elif self_paced_enabled:
        variant = "SP-DLS-SSM"
    else:
        variant = "DLS-SSM"

    logger.info(
        "Loading features | task={} dataset={} split_path={} profile={}",
        args.task, args.dataset, args.split_path, args.profile,
    )
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
    n_features = len(features)
    feature_idx = {name: i for i, name in enumerate(features)}

    y_train = (train_df[label_col].to_numpy() != 0).astype(int)
    y_val = (val_df[label_col].to_numpy() != 0).astype(int)
    y_test = (test_df[label_col].to_numpy() != 0).astype(int)

    if y_train.sum():
        logger.warning("Train contains {} non-normal rows — expected all-normal for semisup.", y_train.sum())
    logger.info(
        "Rows — train: {:,}  val: {:,} (faults: {:,})  test: {:,} (faults: {:,}) | features: {}",
        len(train_df), len(val_df), int(y_val.sum()), len(test_df), int(y_test.sum()), n_features,
    )

    # Scale — fit on train only, apply to all splits
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])

    scaler_mean_t = torch.tensor(scaler.mean_, dtype=torch.float32)
    scaler_scale_t = torch.tensor(scaler.scale_, dtype=torch.float32)

    # Window config
    win_size = int(dlssm_cfg["win_size"])
    train_stride = int(dlssm_cfg.get("train_stride", 1))
    eval_stride = int(dlssm_cfg.get("eval_stride", 1))
    batch_size = int(dlssm_cfg.get("batch_size", 256))
    max_epochs = int(dlssm_cfg.get("max_epochs", 30))
    patience = 10

    if is_smoke:
        train_stride = max(train_stride, win_size // 2)
        eval_stride = max(eval_stride, win_size // 2)
        max_epochs = 1
        patience = 1
        logger.info("Smoke mode: 1 epoch, large strides")

    def _make_dataloaders(stride_train: int, stride_eval: int, bs: int):
        ds_train = TimeSeriesDataset(train_df, features, label_col, win_size, stride_train, normal_only=True)
        ds_val = TimeSeriesDataset(val_df, features, label_col, win_size, stride_eval, normal_only=False)
        ds_test = TimeSeriesDataset(test_df, features, label_col, win_size, stride_eval, normal_only=False)
        kw = dict(drop_last=False, num_workers=0)
        return (
            DataLoader(ds_train, batch_size=bs, shuffle=True, **kw),
            DataLoader(ds_val, batch_size=bs, shuffle=False, **kw),
            DataLoader(ds_test, batch_size=bs, shuffle=False, **kw),
        )

    train_dl, val_dl, test_dl = _make_dataloaders(train_stride, eval_stride, batch_size)
    logger.info(
        "Windows — train: {:,}  val: {:,}  test: {:,}",
        len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset),
    )

    # Build model and Lightning module
    pl.seed_everything(seed, workers=True)

    physics_components = dlssm_cfg.get("physics_components", {})
    model = DeepLatentStateSpaceModel(
        n_features=n_features,
        win_size=win_size,
        hidden_dim=int(dlssm_cfg.get("hidden_dim", 64)),
        latent_dim=int(dlssm_cfg.get("latent_dim", 16)),
        encoder_dim=int(dlssm_cfg.get("encoder_dim", 64)),
        decoder_dim=int(dlssm_cfg.get("decoder_dim", 64)),
        n_gru_layers=int(dlssm_cfg.get("n_gru_layers", 1)),
        dropout=float(dlssm_cfg.get("dropout", 0.1)),
    )

    lit = DLSSMLightningModule(
        model=model,
        scaler_mean=scaler_mean_t,
        scaler_scale=scaler_scale_t,
        feature_idx=feature_idx,
        learning_rate=float(dlssm_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(dlssm_cfg.get("weight_decay", 1e-5)),
        gradient_clip_val=float(dlssm_cfg.get("gradient_clip_val", 1.0)),
        max_epochs=max_epochs,
        beta_kl=float(dlssm_cfg.get("beta_kl", 0.1)),
        kl_warmup_epochs=int(dlssm_cfg.get("kl_warmup_epochs", 5)),
        lambda_phys=float(dlssm_cfg.get("lambda_phys", 0.05)),
        enable_string_power=bool(physics_components.get("string_power", True)),
        enable_imbalance=bool(physics_components.get("imbalance", True)),
        physics_enabled=physics_enabled,
        self_paced_enabled=self_paced_enabled,
        tau_start=float(dlssm_cfg.get("self_paced_tau_start", 0.1)),
        tau_end=float(dlssm_cfg.get("self_paced_tau_end", 1.0)),
        w_min=float(dlssm_cfg.get("self_paced_w_min", 0.2)),
        lambda_kl_score=float(dlssm_cfg.get("lambda_kl_score", 0.1)),
        lambda_phys_score=float(dlssm_cfg.get("lambda_phys_score", 0.0)),
        alpha_pred=float(dlssm_cfg.get("alpha_pred", 0.1)),
        lambda_pred_score=float(dlssm_cfg.get("lambda_pred_score", 0.0)),
        score_reduction=str(dlssm_cfg.get("score_reduction", "center")),
        free_bits=float(dlssm_cfg.get("free_bits", 0.0)),
    )

    # Artifact dir
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        artifacts_dir = get_experiments_root() / "anomaly" / "dlssm" / f"{variant.lower()}_{ts}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = artifacts_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Train
    callbacks = [
        EarlyStopping(monitor="val_pr_auc", patience=patience, mode="max", verbose=False),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor="val_pr_auc",
            mode="max",
            save_top_k=1,
        ),
    ]

    accelerator = str(dlssm_cfg.get("accelerator", "auto"))
    precision_cfg = dlssm_cfg.get("precision", None)  # e.g. "16-mixed" or None
    trainer_kwargs: dict = dict(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=True,
        logger=False,
        deterministic=False,
        gradient_clip_val=float(dlssm_cfg.get("gradient_clip_val", 1.0)),
        accelerator=accelerator,
    )
    if precision_cfg is not None:
        trainer_kwargs["precision"] = precision_cfg
    trainer = pl.Trainer(**trainer_kwargs)

    logger.info("Training {} (physics={}, self_paced={})…", variant, physics_enabled, self_paced_enabled)
    t0 = time.perf_counter()
    trainer.fit(lit, train_dataloaders=train_dl, val_dataloaders=val_dl)
    fit_time = time.perf_counter() - t0
    logger.info("Training done in {:.1f}s", fit_time)

    # Find best checkpoint
    ckpt_cb = next((c for c in callbacks if isinstance(c, ModelCheckpoint)), None)
    best_ckpt_path: str | None = ckpt_cb.best_model_path if (ckpt_cb and ckpt_cb.best_model_path) else None

    # Re-validate with best checkpoint
    if best_ckpt_path:
        logger.info("Re-validating with best checkpoint: {}", best_ckpt_path)
        pl.Trainer(enable_progress_bar=False, enable_model_summary=False, logger=False).validate(
            lit, dataloaders=val_dl, ckpt_path=best_ckpt_path
        )
    else:
        logger.warning("No checkpoint found — val scores from last epoch.")

    # Test
    pl.Trainer(enable_progress_bar=False, enable_model_summary=False, logger=False).test(
        lit, dataloaders=test_dl, ckpt_path=best_ckpt_path
    )

    val_scores = lit._val_scores_np
    val_labels = lit._val_labels_np
    test_scores = lit._test_scores_np
    test_labels = lit._test_labels_np

    if val_scores is None or test_scores is None:
        logger.error("Score arrays not populated — aborting artifact save.")
        return

    threshold = lit.val_threshold
    val_pr_auc = float(average_precision_score(val_labels, val_scores))
    val_roc_auc = float(roc_auc_score(val_labels, val_scores))
    _, val_f1, val_prec, val_rec = _calibrate_threshold(val_scores, val_labels)
    val_preds = (val_scores >= threshold).astype(int)
    val_acc = float(accuracy_score(val_labels, val_preds))

    test_pr_auc = float(average_precision_score(test_labels, test_scores))
    test_roc_auc = float(roc_auc_score(test_labels, test_scores))
    test_preds = (test_scores >= threshold).astype(int)
    test_f1 = float(f1_score(test_labels, test_preds, zero_division=0))
    test_acc = float(accuracy_score(test_labels, test_preds))
    test_prec = float(precision_score(test_labels, test_preds, zero_division=0))
    test_rec = float(recall_score(test_labels, test_preds, zero_division=0))

    logger.info(
        "Test — PR-AUC={:.4f}  ROC-AUC={:.4f}  F1={:.4f}  Prec={:.4f}  Rec={:.4f}",
        test_pr_auc, test_roc_auc, test_f1, test_prec, test_rec,
    )

    metrics: dict = {
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "val_f1_at_threshold": val_f1,
        "val_accuracy_at_threshold": val_acc,
        "val_precision_at_threshold": val_prec,
        "val_recall_at_threshold": val_rec,
        "threshold": threshold,
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_f1_at_threshold": test_f1,
        "test_accuracy_at_threshold": test_acc,
        "test_precision_at_threshold": test_prec,
        "test_recall_at_threshold": test_rec,
        "n_train_windows": len(train_dl.dataset),
        "n_features": n_features,
        "fit_time_s": round(fit_time, 2),
    }

    # Save artifacts
    run_params = {
        "variant": variant,
        "physics_enabled": physics_enabled,
        "self_paced_enabled": self_paced_enabled,
        **{k: v for k, v in dlssm_cfg.items() if not isinstance(v, dict)},
        "max_epochs_actual": max_epochs,
        "n_features": n_features,
        "seed": seed,
    }

    metrics_path = artifacts_dir / "metrics.json"
    params_path = artifacts_dir / "run_params.json"
    scaler_path = artifacts_dir / "scaler.joblib"
    manifest_path = artifacts_dir / "features_manifest.json"
    pr_curve_path = artifacts_dir / "pr_curve.png"
    histogram_path = artifacts_dir / "score_histogram.png"
    timeline_path = artifacts_dir / "score_timeline.png"
    pred_residual_path = artifacts_dir / "prediction_residual_decomposition.png"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    params_path.write_text(json.dumps(run_params, indent=2, default=str), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    joblib.dump(scaler, scaler_path)

    _save_pr_curve(val_scores, val_labels, test_scores, test_labels, pr_curve_path, model_name=variant)
    _save_score_histogram(val_scores, val_labels, threshold, histogram_path, model_name=variant)
    _save_score_timeline(test_scores, test_labels, threshold, timeline_path, model_name=variant)
    try:
        _save_prediction_residual_plot(
            lit, test_dl, features, test_scores, test_labels,
            pred_residual_path, model_name=variant,
        )
    except Exception as exc:
        logger.warning("Prediction/residual plot failed (non-fatal): {}", exc)

    logger.info("Artifacts saved → {}", artifacts_dir)

    # MLflow
    run_name = f"anomaly_dlssm_{variant.lower().replace('-', '_')}_{ts}"
    try:
        init_tracking("anomaly")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "task": args.task,
                "dataset": args.dataset,
                "split_path": args.split_path,
                "profile": str(args.profile),
                "model": "dlssm",
                "model_family": "anomaly_dl",
                "variant": variant,
                "physics_informed": str(physics_enabled),
                "self_paced": str(self_paced_enabled),
            })
            mlflow.log_params({
                k: v for k, v in run_params.items()
                if not isinstance(v, (dict, list))
            })
            mlflow.log_metrics(metrics)
            for p in (metrics_path, params_path, scaler_path, manifest_path,
                      pr_curve_path, histogram_path, timeline_path, pred_residual_path):
                if p.exists():
                    mlflow.log_artifact(str(p))
            if best_ckpt_path and Path(best_ckpt_path).exists():
                mlflow.log_artifact(best_ckpt_path, artifact_path="checkpoints")

            comparison_record = {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "run_name": run_name,
                "task": args.task,
                "dataset": args.dataset,
                "split_path": args.split_path,
                "model": "dlssm",
                "variant": variant,
                "model_family": "anomaly_dl",
                "run_type": run_type,
                "seed": seed,
                "feature_profile": str(args.profile),
                "feature_run_dir": str(resolved_run_dir),
                "physics_enabled": physics_enabled,
                "self_paced_enabled": self_paced_enabled,
                "n_features": n_features,
                "win_size": win_size,
                "hidden_dim": int(dlssm_cfg.get("hidden_dim", 64)),
                "latent_dim": int(dlssm_cfg.get("latent_dim", 16)),
                "lambda_phys": float(dlssm_cfg.get("lambda_phys", 0.05)),
                "threshold": threshold,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "val_f1_at_threshold": val_f1,
                "val_accuracy_at_threshold": val_acc,
                "test_pr_auc": test_pr_auc,
                "test_roc_auc": test_roc_auc,
                "test_f1_at_threshold": test_f1,
                "test_accuracy_at_threshold": test_acc,
                "test_precision_at_threshold": test_prec,
                "test_recall_at_threshold": test_rec,
                "fit_time_s": round(fit_time, 2),
                "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
            }
            records_path = Path(args.comparison_records_path)
            records_path.parent.mkdir(parents=True, exist_ok=True)
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(comparison_record, default=str) + "\n")
            # Skip logging the shared comparison records file — it's cumulative
            # across all runs and may contain non-cp1252 characters on Windows.

        logger.info("MLflow run logged: {}", run_name)
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): {}", exc)

    ocsvm_baseline = 0.92
    cmp = "beats" if test_pr_auc > ocsvm_baseline else "below"
    logger.info(
        "{} {} OC-SVM baseline: {:.4f} {} {:.2f}",
        variant, cmp, test_pr_auc, ">" if cmp == "beats" else "<=", ocsvm_baseline,
    )


if __name__ == "__main__":
    run_dlssm()
