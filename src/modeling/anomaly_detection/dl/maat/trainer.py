from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
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
from src.modeling.anomaly_detection.dl.maat.losses import (
    association_losses,
    compute_maat_scores,
)
from src.modeling.anomaly_detection.dl.maat.model import MambaAnomalyTransformer
from src.modeling.common.feature_loader import load_features_for_task
from src.modeling.common.hyperparameter_optimizer import (
    HPOStageConfig,
    run_staged_optuna,
    suggest_params_from_space,
)
from src.utils.paths import get_experiments_root

# trainer.py lives under dl/maat/, so PROJECT_ROOT is 5 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[5]

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_float32_matmul_precision("medium")  # use Tensor Cores on Ampere+


def _hpo_config_fingerprint(maat_cfg: dict, seed: int) -> str:
    """8-char hash of HPO-relevant config fields.

    Same config+seed resumes existing study; any change creates a new one.
    Covers: architecture defaults, search spaces, seed, score_reduction.
    Does NOT include train_stride/eval_stride (no effect on trial comparisons).
    score_reduction is included because it changes the trial objective value
    via _reduce_scores(); mixing studies across reduction modes contaminates results.
    """
    relevant = {
        k: maat_cfg.get(k)
        for k in ("win_size", "block_size", "d_model", "n_heads", "e_layers",
                  "d_ff", "dropout", "k", "temperature", "score_reduction")
    }
    relevant["hpo_stage1"] = maat_cfg.get("hpo_stage1", {})
    relevant["hpo_stage2"] = maat_cfg.get("hpo_stage2", {})
    relevant["seed"] = seed
    raw = json.dumps(relevant, sort_keys=True, default=str).encode()
    return hashlib.sha1(raw).hexdigest()[:8]


def _default_comparison_records_path() -> Path:
    return get_experiments_root() / "metrics" / "anomaly_comparison_records.jsonl"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAAT anomaly detection — Mamba Adaptive Anomaly Transformer")
    p.add_argument("--task", default="anomaly_semisup")
    p.add_argument("--dataset", default="costa")
    p.add_argument("--split-path", default="path_a")
    p.add_argument("--profile", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--hpo", action="store_true", help="Run two-stage Optuna HPO")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 3 epochs, large stride, no HPO")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--artifacts-dir", default=None)
    p.add_argument("--comparison-records-path", default=str(_default_comparison_records_path()))
    p.add_argument("--run-type", default="baseline", help="baseline | hpo | final | smoke")
    p.add_argument(
        "--best-params",
        default=None,
        help=(
            "JSON string or path to a JSON file of pre-found HPO params. "
            "Skips HPO entirely and merges these params directly into the final run. "
            'Example: \'{"learning_rate": 3.2e-4, "k": 4.77, "temperature": 10}\''
        ),
    )
    return p.parse_args()


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _calibrate_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
    """Return (threshold, best_f1, precision, recall) by maximising F1 on the PR curve."""
    prec, rec, thresholds = precision_recall_curve(labels, scores)
    denom = prec[:-1] + rec[:-1]
    # Evaluate division only where denom > 0 to silence RuntimeWarning.
    # np.where evaluates both branches eagerly, so guard the denominator explicitly.
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

class MAATLightningModule(pl.LightningModule):
    """MAAT with minimax loss training.

    Uses manual optimization (automatic_optimization=False) for the two-pass
    minimax: loss1 = rec - k*series (minimize), loss2 = rec + k*prior (minimize).
    """

    automatic_optimization = False

    def __init__(
        self,
        model: MambaAnomalyTransformer,
        k: float = 3.0,
        temperature: float = 50.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        gradient_clip_val: float | None = 1.0,
        max_epochs: int = 100,
        score_reduction: str = "center",
    ) -> None:
        super().__init__()
        self.model = model
        self.k = k
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.max_epochs = max_epochs
        self.score_reduction = score_reduction

        self._val_outputs: list[dict] = []
        self._test_outputs: list[dict] = []
        self.best_val_pr_auc: float = 0.0
        self.val_threshold: float = 0.5
        self._val_scores_np: np.ndarray | None = None
        self._val_labels_np: np.ndarray | None = None
        self._test_scores_np: np.ndarray | None = None
        self._test_labels_np: np.ndarray | None = None
        self._nan_count: int = 0

    def _compute_step(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_hat, series_list, prior_list, _ = self.model(x)
        rec_loss = nn.functional.mse_loss(x_hat, x)
        s_loss, p_loss = association_losses(series_list, prior_list)
        loss1 = rec_loss - self.k * s_loss
        loss2 = rec_loss + self.k * p_loss
        return rec_loss, s_loss, p_loss, loss1, loss2

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, _ = batch
        opt = self.optimizers()

        rec_loss, s_loss, p_loss, loss1, loss2 = self._compute_step(x)

        if not (torch.isfinite(loss1) and torch.isfinite(loss2)):
            self._nan_count += 1
            self.log("nan_detected", 1.0, prog_bar=False)
            return

        opt.zero_grad(set_to_none=True)
        self.manual_backward(loss1, retain_graph=True)
        self.manual_backward(loss2)

        if self.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.gradient_clip_val
            )

        grad_norm = 0.0
        params_with_grad = [p for p in self.model.parameters() if p.grad is not None]
        if params_with_grad:
            grad_norm = float(
                torch.norm(torch.stack([p.grad.norm() for p in params_with_grad]))
            )
            if not math.isfinite(grad_norm):
                self._nan_count += 1
                self.log("nan_detected", 1.0, prog_bar=False)
                opt.zero_grad(set_to_none=True)
                return

        opt.step()

        self.log("train_rec_loss", rec_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_series_kl", s_loss, on_step=False, on_epoch=True)
        self.log("train_prior_kl", p_loss, on_step=False, on_epoch=True)
        self.log("train_loss1", loss1, on_step=False, on_epoch=True)
        self.log("train_loss2", loss2, on_step=False, on_epoch=True)
        self.log("grad_norm", grad_norm, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        try:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log("learning_rate", current_lr)
        except Exception:
            pass

    def _reduce_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Reduce [B, W] scores to [B] per-window scalars."""
        if self.score_reduction == "mean":
            return scores.mean(dim=1)
        if self.score_reduction == "max":
            return scores.max(dim=1).values
        # default: center
        return scores[:, scores.size(1) // 2]

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, labels = batch
        with torch.no_grad():
            x_hat, series_list, prior_list, _ = self.model(x)

        recon_error = ((x - x_hat) ** 2).mean(dim=-1)  # [B, W]
        scores = compute_maat_scores(series_list, prior_list, recon_error, self.temperature)
        center_scores = self._reduce_scores(scores).cpu()
        center_labels = labels.cpu()

        rec_loss = nn.functional.mse_loss(x_hat, x)
        s_loss, p_loss = association_losses(series_list, prior_list)

        self._val_outputs.append({
            "scores": center_scores,
            "labels": center_labels,
            "rec_loss": rec_loss.item(),
            "s_loss": s_loss.item(),
            "p_loss": p_loss.item(),
        })

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return

        all_scores = torch.cat([o["scores"] for o in self._val_outputs]).numpy()
        all_labels = torch.cat([o["labels"] for o in self._val_outputs]).numpy()
        mean_rec = float(np.mean([o["rec_loss"] for o in self._val_outputs]))
        mean_s = float(np.mean([o["s_loss"] for o in self._val_outputs]))
        mean_p = float(np.mean([o["p_loss"] for o in self._val_outputs]))
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
        self.log("val_rec_loss", mean_rec)
        self.log("val_series_kl", mean_s)
        self.log("val_prior_kl", mean_p)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, labels = batch
        with torch.no_grad():
            x_hat, series_list, prior_list, _ = self.model(x)

        recon_error = ((x - x_hat) ** 2).mean(dim=-1)
        scores = compute_maat_scores(series_list, prior_list, recon_error, self.temperature)
        self._test_outputs.append({
            "scores": self._reduce_scores(scores).cpu(),
            "labels": labels.cpu(),
        })

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
            opt,
            T_max=self.max_epochs,
            eta_min=self.learning_rate * 0.01,
        )
        # With automatic_optimization=False, Lightning does not step the scheduler
        # automatically. We step it manually in on_train_epoch_end.
        return [opt], [sch]


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers (identical contract to OC-SVM)
# ─────────────────────────────────────────────────────────────────────────────

def _save_pr_curve(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    path: Path,
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
    ax.set_title("PR Curve — MAAT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_score_histogram(
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(val_scores[val_labels == 0], bins=60, alpha=0.5, label="Val — normal", density=True)
    if val_labels.sum() > 0:
        ax.hist(val_scores[val_labels == 1], bins=60, alpha=0.5, label="Val — fault", density=True)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution — MAAT")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_score_timeline(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = np.where(test_labels == 1, "red", "steelblue")
    ax.scatter(np.arange(len(test_scores)), test_scores, c=colors, s=2, alpha=0.5, rasterized=True)
    ax.axhline(threshold, color="orange", linestyle="--", linewidth=1.5, label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("Test window index")
    ax.set_ylabel("Anomaly score")
    ax.set_title("Score Timeline (Test) — MAAT  |  blue=normal  red=fault")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Optuna pruning callback
# ─────────────────────────────────────────────────────────────────────────────

class _OptunaPruningCallback(pl.Callback):
    """Reports val_pr_auc to Optuna each epoch and raises TrialPruned when pruner fires."""

    def __init__(self, trial: optuna.Trial, warmup_epochs: int = 3) -> None:
        self.trial = trial
        self.warmup_epochs = warmup_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch < self.warmup_epochs:
            return
        val_pr_auc = trainer.callback_metrics.get("val_pr_auc")
        if val_pr_auc is None:
            return
        self.trial.report(float(val_pr_auc), step=trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(maat_cfg: dict, n_features: int) -> MambaAnomalyTransformer:
    return MambaAnomalyTransformer(
        win_size=int(maat_cfg["win_size"]),
        enc_in=n_features,
        c_out=n_features,
        d_model=int(maat_cfg["d_model"]),
        n_heads=int(maat_cfg["n_heads"]),
        e_layers=int(maat_cfg["e_layers"]),
        d_ff=int(maat_cfg["d_ff"]),
        dropout=float(maat_cfg["dropout"]),
        block_size=int(maat_cfg["block_size"]),
    )


def _build_lightning_module(
    model: MambaAnomalyTransformer,
    maat_cfg: dict,
    training_cfg: dict,
    max_epochs: int,
) -> MAATLightningModule:
    return MAATLightningModule(
        model=model,
        k=float(maat_cfg.get("k", 3.0)),
        temperature=float(maat_cfg.get("temperature", 50.0)),
        learning_rate=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-2)),
        gradient_clip_val=float(training_cfg.get("gradient_clip_val", 1.0)),
        max_epochs=max_epochs,
        score_reduction=str(maat_cfg.get("score_reduction", "center")),
    )


def _train_and_eval(
    maat_cfg: dict,
    training_cfg: dict,
    train_dl: DataLoader,
    val_dl: DataLoader,
    n_features: int,
    max_epochs: int,
    patience: int,
    seed: int,
    ckpt_dir: Path | None = None,
    trial: optuna.Trial | None = None,
) -> tuple[MAATLightningModule, str | None]:
    pl.seed_everything(seed, workers=True)

    model = _build_model(maat_cfg, n_features)
    lit = _build_lightning_module(model, maat_cfg, training_cfg, max_epochs)

    ckpt_callback: ModelCheckpoint | None = None
    callbacks: list = [
        EarlyStopping(monitor="val_pr_auc", patience=patience, mode="max", verbose=False),
    ]
    if ckpt_dir is not None:
        ckpt_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor="val_pr_auc",
            mode="max",
            save_top_k=1,
        )
        callbacks.append(ckpt_callback)
    if trial is not None:
        callbacks.append(_OptunaPruningCallback(trial, warmup_epochs=3))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=(ckpt_dir is not None),
        logger=False,
        deterministic=False,
    )
    trainer.fit(lit, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_ckpt_path: str | None = None
    if ckpt_callback is not None and ckpt_callback.best_model_path:
        best_ckpt_path = ckpt_callback.best_model_path

    return lit, best_ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_maat(config: dict | None = None) -> None:
    args = _parse_args()
    if config is None:
        config = _load_config()

    maat_cfg: dict = config["anomaly_detection"]["dl"]["models"]["maat"]
    hpo_cfg: dict = config["anomaly_detection"]["dl"]["hpo"]
    training_cfg: dict = config.get("training", {})

    seed: int = args.seed
    is_smoke: bool = args.smoke

    # Pre-load best params if provided — skips HPO entirely.
    injected_params: dict = {}
    if args.best_params:
        raw = args.best_params.strip()
        try:
            # Treat as a file path only if it doesn't start with '{' (i.e. not inline JSON).
            if not raw.startswith("{") and Path(raw).exists():
                injected_params = json.loads(Path(raw).read_text(encoding="utf-8"))
                logger.info(f"Loaded best params from file: {raw}")
            else:
                injected_params = json.loads(raw)
                logger.info("Loaded best params from --best-params JSON string")
        except Exception as exc:
            raise ValueError(f"--best-params could not be parsed: {exc}") from exc

    run_hpo: bool = args.hpo and not is_smoke and not injected_params

    # ── Load features ──────────────────────────────────────────────────────────
    logger.info(
        f"Loading features | task={args.task} dataset={args.dataset} "
        f"split_path={args.split_path} profile={args.profile}"
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

    y_train = (train_df[label_col].to_numpy() != 0).astype(int)
    y_val = (val_df[label_col].to_numpy() != 0).astype(int)
    y_test = (test_df[label_col].to_numpy() != 0).astype(int)

    non_normal = int(y_train.sum())
    if non_normal:
        logger.warning(f"Train contains {non_normal} non-normal rows — expected all-normal for semisup.")

    logger.info(
        f"Rows — train: {len(train_df):,}  val: {len(val_df):,} (faults: {y_val.sum():,})  "
        f"test: {len(test_df):,} (faults: {y_test.sum():,}) | features: {n_features}"
    )

    # ── Scale ──────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])

    # ── Window config ──────────────────────────────────────────────────────────
    win_size = int(maat_cfg["win_size"])
    train_stride = int(maat_cfg.get("train_stride", 1))
    eval_stride = int(maat_cfg.get("eval_stride", 1))
    batch_size = int(training_cfg.get("batch_size", 256))
    max_epochs = int(training_cfg.get("max_epochs", 100))
    patience = int(training_cfg.get("patience", 15))

    if is_smoke:
        train_stride = max(train_stride, win_size // 2)
        eval_stride = max(eval_stride, win_size // 2)
        max_epochs = 3
        patience = 3
        run_type = "smoke"
        logger.info("Smoke mode: 3 epochs, large strides, no HPO")
    else:
        run_type = args.run_type

    def _make_dataloaders(cfg: dict, stride_train: int, stride_eval: int, bs: int) -> tuple:
        ds_train = TimeSeriesDataset(
            train_df, features, label_col, cfg["win_size"], stride_train, normal_only=True
        )
        ds_val = TimeSeriesDataset(
            val_df, features, label_col, cfg["win_size"], stride_eval, normal_only=False
        )
        ds_test = TimeSeriesDataset(
            test_df, features, label_col, cfg["win_size"], stride_eval, normal_only=False
        )
        # num_workers=0: avoids DataLoader multiprocessing teardown races under
        # Lightning/Optuna/Colab (Python 3.12 AssertionError: can only test a child process).
        # persistent_workers requires num_workers > 0, so also forced off.
        train_dl = DataLoader(ds_train, batch_size=bs, shuffle=True, drop_last=False, num_workers=0)
        val_dl = DataLoader(ds_val, batch_size=bs, shuffle=False, drop_last=False, num_workers=0)
        test_dl = DataLoader(ds_test, batch_size=bs, shuffle=False, drop_last=False, num_workers=0)
        return train_dl, val_dl, test_dl

    train_dl, val_dl, test_dl = _make_dataloaders(maat_cfg, train_stride, eval_stride, batch_size)
    logger.info(
        f"Windows — train: {len(train_dl.dataset):,}  val: {len(val_dl.dataset):,}  "
        f"test: {len(test_dl.dataset):,}"
    )

    # ── HPO ────────────────────────────────────────────────────────────────────
    best_params: dict = injected_params
    stage_results = []

    if injected_params:
        logger.info(f"Skipping HPO — using injected params: {injected_params}")

    if run_hpo:
        trial_budget: dict = hpo_cfg.get("trial_budget", {})
        n_stage1 = int(trial_budget.get("stage1_training", 20))
        n_stage2 = int(trial_budget.get("stage2_architecture", 40))
        hpo_epochs = max(5, max_epochs // 5)
        hpo_patience = max(3, patience // 3)

        fixed_arch = {
            k: maat_cfg[k]
            for k in ("win_size", "block_size", "d_model", "n_heads", "e_layers", "d_ff")
        }

        def objective_builder(stage: HPOStageConfig, frozen_params: dict):
            def objective(trial: optuna.Trial) -> float:
                trial_p = suggest_params_from_space(trial, stage.search_space)
                merged = {**fixed_arch, **frozen_params, **trial_p}

                ws = int(merged.get("win_size", maat_cfg["win_size"]))
                bs_ = int(merged.get("block_size", maat_cfg["block_size"]))
                dm = int(merged.get("d_model", maat_cfg["d_model"]))
                nh = int(merged.get("n_heads", maat_cfg["n_heads"]))
                if ws % bs_ != 0 or dm % nh != 0:
                    raise optuna.TrialPruned()

                trial_maat_cfg = {**maat_cfg, **merged}
                trial_training_cfg = {**training_cfg, **{
                    k: merged[k] for k in ("weight_decay", "batch_size", "gradient_clip_val")
                    if k in merged
                }}
                if "learning_rate" in merged:
                    trial_training_cfg["lr"] = float(merged["learning_rate"])

                trial_bs = int(trial_training_cfg.get("batch_size", min(batch_size, 128)))
                t_dl, v_dl, _ = _make_dataloaders(
                    trial_maat_cfg, train_stride, eval_stride, trial_bs
                )

                try:
                    lit, _ = _train_and_eval(
                        trial_maat_cfg, trial_training_cfg, t_dl, v_dl,
                        n_features, hpo_epochs, hpo_patience, seed, trial=trial,
                    )
                    pr_auc = lit.best_val_pr_auc
                except optuna.exceptions.TrialPruned:
                    raise
                except Exception as exc:
                    logger.warning(f"HPO trial real exception (will prune): {type(exc).__name__}: {exc}")
                    raise optuna.TrialPruned()

                if not math.isfinite(pr_auc) or pr_auc <= 0.0:
                    raise optuna.TrialPruned()
                return pr_auc

            return objective

        stage1_space = dict(maat_cfg.get("hpo_stage1", {}))
        stage2_space = dict(maat_cfg.get("hpo_stage2", {}))

        stages = [
            HPOStageConfig(
                name="stage1_training",
                search_space=stage1_space,
                n_trials=n_stage1,
                direction="maximize",
                sampler=str(hpo_cfg.get("sampler", "tpe")),
                pruner=str(hpo_cfg.get("pruner", "hyperband")),
            ),
            HPOStageConfig(
                name="stage2_architecture",
                search_space=stage2_space,
                n_trials=n_stage2,
                direction="maximize",
                sampler=str(hpo_cfg.get("sampler", "tpe")),
                pruner=str(hpo_cfg.get("pruner", "hyperband")),
            ),
        ]

        profile_str = str(args.profile or "default").replace("/", "_").replace("\\", "_")
        cfg_hash = _hpo_config_fingerprint(maat_cfg, seed)
        fingerprinted_prefix = (
            f"{hpo_cfg.get('study_name_prefix', 'anomaly_dl_maat')}"
            f"_{args.dataset}_{args.task}_{args.split_path}_{profile_str}_{cfg_hash}"
        )

        logger.info(f"Running two-stage HPO | stage1={n_stage1} trials  stage2={n_stage2} trials")
        best_params, stage_results = run_staged_optuna(
            stages=stages,
            objective_builder=objective_builder,
            seed=seed,
            storage_url=hpo_cfg.get("storage_url"),
            study_name_prefix=fingerprinted_prefix,
        )
        logger.info(f"HPO best params: {best_params}")
    else:
        best_params = {}
        logger.info("HPO skipped — using config defaults")

    # ── Merge best params into final config ────────────────────────────────────
    final_maat_cfg = dict(maat_cfg)
    final_training_cfg = dict(training_cfg)
    for k, v in best_params.items():
        if k in ("win_size", "block_size", "d_model", "n_heads", "e_layers", "d_ff",
                 "dropout", "k", "temperature"):
            final_maat_cfg[k] = v
        elif k == "learning_rate":
            final_training_cfg["lr"] = v
        elif k in ("weight_decay", "batch_size", "gradient_clip_val"):
            final_training_cfg[k] = v

    final_batch_size = int(final_training_cfg.get("batch_size", batch_size))
    train_dl, val_dl, test_dl = _make_dataloaders(
        final_maat_cfg, train_stride, eval_stride, final_batch_size
    )

    # ── Final training ─────────────────────────────────────────────────────────
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        artifacts_dir = get_experiments_root() / "anomaly" / "maat" / ts
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = artifacts_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training final MAAT model…")
    t0 = time.perf_counter()
    final_lit, best_ckpt_path = _train_and_eval(
        final_maat_cfg, final_training_cfg, train_dl, val_dl,
        n_features=n_features,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        ckpt_dir=ckpt_dir,
    )
    fit_time = time.perf_counter() - t0
    logger.info(f"Training done in {fit_time:.1f}s | best_ckpt={best_ckpt_path}")

    # ── Re-validate with best checkpoint to get scores/threshold from best epoch ──
    if best_ckpt_path is not None:
        logger.info(f"Re-validating with best checkpoint: {best_ckpt_path}")
        trainer_val = pl.Trainer(
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer_val.validate(final_lit, dataloaders=val_dl, ckpt_path=best_ckpt_path)
    else:
        logger.warning("No best checkpoint found — val scores are from last training epoch.")

    logger.info(f"val_pr_auc (best ckpt)={final_lit.best_val_pr_auc:.4f}")

    # ── Test evaluation ────────────────────────────────────────────────────────
    trainer_test = pl.Trainer(
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer_test.test(
        final_lit,
        dataloaders=test_dl,
        ckpt_path=best_ckpt_path,
    )

    val_scores = final_lit._val_scores_np
    val_labels = final_lit._val_labels_np
    test_scores = final_lit._test_scores_np
    test_labels = final_lit._test_labels_np

    if val_scores is None or test_scores is None:
        logger.error("Score arrays not populated — test may not have run.")
        return

    threshold = final_lit.val_threshold
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
        f"Test — PR-AUC={test_pr_auc:.4f}  ROC-AUC={test_roc_auc:.4f}  "
        f"F1={test_f1:.4f}  Prec={test_prec:.4f}  Rec={test_rec:.4f}"
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
        "nan_count": final_lit._nan_count,
    }

    # ── Save artifacts ──────────────────────────────────────────────────────────
    metrics_path = artifacts_dir / "metrics.json"
    scaler_path = artifacts_dir / "scaler.joblib"
    params_path = artifacts_dir / "best_params.json"
    config_path = artifacts_dir / "resolved_config.json"
    pr_curve_path = artifacts_dir / "pr_curve.png"
    histogram_path = artifacts_dir / "score_histogram.png"
    timeline_path = artifacts_dir / "score_timeline.png"
    manifest_path = artifacts_dir / "features_manifest.json"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(scaler, scaler_path)
    params_path.write_text(json.dumps(best_params, indent=2, default=str), encoding="utf-8")
    config_path.write_text(
        json.dumps({"maat": final_maat_cfg, "training": final_training_cfg}, indent=2, default=str),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    _save_pr_curve(val_scores, val_labels, test_scores, test_labels, pr_curve_path)
    _save_score_histogram(val_scores, val_labels, threshold, histogram_path)
    _save_score_timeline(test_scores, test_labels, threshold, timeline_path)

    if stage_results:
        for sr in stage_results:
            trials_df = sr.study.trials_dataframe()
            trials_df.to_csv(artifacts_dir / f"hpo_trials_{sr.name}.csv", index=False)

    logger.info(f"Artifacts saved → {artifacts_dir}")

    # ── MLflow ──────────────────────────────────────────────────────────────────
    run_name = f"anomaly_maat_{ts}"
    try:
        init_tracking("anomaly")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "task": args.task,
                "dataset": args.dataset,
                "split_path": args.split_path,
                "profile": str(args.profile),
                "model": "maat",
                "model_family": "anomaly_dl",
                "hpo_mode": "auto_two_stage" if run_hpo else ("injected" if injected_params else "disabled"),
            })
            mlflow.log_params({
                **{k: v for k, v in final_maat_cfg.items() if not isinstance(v, dict)},
                "n_features": n_features,
                "seed": seed,
                "hpo_enabled": run_hpo,
                "run_type": run_type,
                "score_reduction": final_maat_cfg.get("score_reduction", "center"),
            })
            mlflow.log_metrics(metrics)
            for p in (metrics_path, scaler_path, params_path, config_path,
                      pr_curve_path, histogram_path, timeline_path, manifest_path):
                if p.exists():
                    mlflow.log_artifact(str(p))
            if stage_results:
                for sr in stage_results:
                    csv_path = artifacts_dir / f"hpo_trials_{sr.name}.csv"
                    if csv_path.exists():
                        mlflow.log_artifact(str(csv_path))

            comparison_record = {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "run_name": run_name,
                "task": args.task,
                "dataset": args.dataset,
                "split_path": args.split_path,
                "model": "maat",
                "model_family": "anomaly_dl",
                "run_type": run_type,
                "seed": seed,
                "feature_profile": str(args.profile),
                "feature_run_dir": str(resolved_run_dir),
                "hpo_enabled": run_hpo,
                "hpo_stage_budgets": hpo_cfg.get("trial_budget", {}),
                "best_params": best_params,
                "n_features": n_features,
                "win_size": int(final_maat_cfg["win_size"]),
                "block_size": int(final_maat_cfg["block_size"]),
                "score_reduction": final_maat_cfg.get("score_reduction", "center"),
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
            mlflow.log_artifact(str(records_path))

        logger.info(f"MLflow run logged: {run_name}")
    except Exception as exc:
        logger.warning(f"MLflow logging failed (non-fatal): {exc}")

    ocsvm_baseline = 0.92
    if test_pr_auc > ocsvm_baseline:
        logger.info(f"MAAT beats OC-SVM baseline: {test_pr_auc:.4f} > {ocsvm_baseline}")
    else:
        logger.info(f"MAAT below OC-SVM baseline: {test_pr_auc:.4f} <= {ocsvm_baseline}")


if __name__ == "__main__":
    run_maat()
