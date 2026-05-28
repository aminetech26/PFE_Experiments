"""PC-Flow trainer: normalizing flow with NLL anomaly scoring.

Mirrors pc_ae/trainer.py structure exactly so the k-fold orchestrator,
MLflow logging, artifact contract, and GPD threshold calibration all
drop in unchanged.

Key differences vs PC-AE:
- Training loss = mean negative log-likelihood on normal data (not MSE).
- Anomaly score = -log p(x | c) (not reconstruction MSE).
- No c-invariance regularizer needed: physics conditioning is built into
  the flow's coupling MLPs. The NLL directly captures the density under
  the conditional distribution p(x | irr, pvt).
"""
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
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from loguru import logger
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
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
from src.modeling.anomaly_detection.dl.pc_flow.model import PCFlowModel
from src.modeling.common.artifact_contract import (
    build_candidate_per_true_class_thresholds,
    build_deployment_manifest,
    build_run_manifest,
    build_score_calibration_payload,
    compute_anomaly_per_class_metrics,
    compute_episode_level_pr_auc,
    compute_macro_per_class_pr_auc,
    write_json,
)
from src.modeling.common.dl_training_utils import (
    _build_warmup_cosine_scheduler,
    _resolve_loader_runtime,
    _trainer_runtime_kwargs,
)
from src.modeling.common.episode_metrics import episode_macro_f1_binary
from src.modeling.common.feature_loader import load_features_for_task
from src.modeling.common.fold_override import add_fold_override_args, apply_fold_overrides
from src.modeling.common.hyperparameter_optimizer import (
    _build_pruner,
    _build_sampler,
    suggest_params_from_space,
)
from src.modeling.common.threshold_calibration import (
    calibrate_threshold,
    calibrate_threshold_quantile,
    load_threshold_config,
    threshold_policy_str as _threshold_policy_str,
)
from src.utils.paths import get_experiments_root


PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _default_comparison_records_path() -> Path:
    return get_experiments_root() / "metrics" / "anomaly_comparison_records.jsonl"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PC-Flow physics-conditioned normalizing flow anomaly detection")
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
    p.add_argument("--hpo", action="store_true")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--best-params", default=None)
    add_fold_override_args(p)
    return p.parse_args()


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class PCFlowLightningModule(pl.LightningModule):
    """PC-Flow training module: conditional NF with NLL loss."""

    def __init__(
        self,
        model: PCFlowModel,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
        feature_idx: dict[str, int],
        context_feature_indices: list[int],
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        gradient_clip_val: float = 1.0,
        max_epochs: int = 50,
        total_steps: int = 1,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
        hpo_mode: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)
        self.feature_idx = feature_idx
        self.context_feature_indices: list[int] = list(context_feature_indices)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.max_epochs = max_epochs
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.hpo_mode = hpo_mode

        self._val_outputs: list[dict] = []
        self._test_outputs: list[dict] = []
        self.best_val_pr_auc: float = 0.0
        self.val_threshold: float = 0.5
        self._val_scores_np: np.ndarray | None = None
        self._val_labels_np: np.ndarray | None = None
        self._val_original_labels_np: np.ndarray | None = None
        self._val_group_ids_np: np.ndarray | None = None
        self._test_scores_np: np.ndarray | None = None
        self._test_labels_np: np.ndarray | None = None
        self._test_original_labels_np: np.ndarray | None = None
        self._test_group_ids_np: np.ndarray | None = None

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.squeeze(1)
        return x

    def _extract_context(self, x_flat: torch.Tensor) -> torch.Tensor:
        idx = torch.tensor(self.context_feature_indices, device=x_flat.device, dtype=torch.long)
        return x_flat.index_select(1, idx)

    def _get_x_and_c(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract non-context features and context vector."""
        c = self._extract_context(x_flat)
        # Build boolean mask for non-context features
        all_indices = set(range(x_flat.shape[1]))
        ctx_set = set(self.context_feature_indices)
        non_ctx_idx = sorted(all_indices - ctx_set)
        idx_t = torch.tensor(non_ctx_idx, device=x_flat.device, dtype=torch.long)
        x_feat = x_flat.index_select(1, idx_t)
        return x_feat, c

    def _unpack_batch(self, batch: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list | None]:
        x = batch[0]
        labels = batch[1]
        original_labels = batch[2] if len(batch) > 2 else labels
        group_ids = list(batch[3]) if len(batch) > 3 else None
        return x, labels, original_labels, group_ids

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, _, _, _ = self._unpack_batch(batch)
        x_flat = self._flatten(x)
        x_feat, c = self._get_x_and_c(x_flat)
        nll = -self.model.log_prob(x_feat, c)    # [B]
        # Clip extreme values for numerical stability at early epochs
        nll = torch.clamp(nll, max=1e6)
        loss = nll.mean()
        self.log("train_nll", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def _score_batch(self, x_flat: torch.Tensor) -> torch.Tensor:
        x_feat, c = self._get_x_and_c(x_flat)
        scores = self.model.anomaly_score(x_feat, c)   # [B], higher = more anomalous
        return torch.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=-1e6)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, labels, original_labels, group_ids = self._unpack_batch(batch)
        x_flat = self._flatten(x)
        scores = self._score_batch(x_flat).detach().cpu()
        self._val_outputs.append({
            "scores": scores,
            "labels": labels.detach().cpu(),
            "original_labels": original_labels.detach().cpu(),
            "group_ids": group_ids,
        })

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return
        all_scores = torch.cat([o["scores"] for o in self._val_outputs]).numpy()
        all_labels = torch.cat([o["labels"] for o in self._val_outputs]).numpy()
        all_original_labels = torch.cat([o["original_labels"] for o in self._val_outputs]).numpy()
        all_group_ids = None
        if self._val_outputs[0].get("group_ids") is not None:
            all_group_ids = np.array(
                [gid for o in self._val_outputs for gid in (o.get("group_ids") or [])],
                dtype=object,
            )
        self._val_outputs.clear()

        if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
            self.log("val_macro_per_class_pr_auc", 0.0, prog_bar=True)
            self.log("val_macro_per_class_pr_auc_monitor", 0.0, prog_bar=True)
            self.log("val_worst_class_pr_auc", 0.0, prog_bar=True)
            self.log("val_pr_auc", 0.0, prog_bar=True)
            self.log("val_roc_auc", 0.0, prog_bar=True)
            return

        val_pr_auc = float(average_precision_score(all_labels, all_scores))
        val_roc_auc = float(roc_auc_score(all_labels, all_scores))
        per_class_pr = compute_macro_per_class_pr_auc(labels=all_original_labels, scores=all_scores)
        val_macro_pc = float(per_class_pr.get("macro_per_class_pr_auc") or 0.0)
        val_worst_pc = float(per_class_pr.get("worst_class_pr_auc") or 0.0)
        per_class_detail = per_class_pr.get("per_class_pr_auc_vs_normal", {})

        self.best_val_pr_auc = max(self.best_val_pr_auc, val_macro_pc)
        self._val_scores_np = all_scores
        self._val_labels_np = all_labels
        self._val_original_labels_np = all_original_labels
        self._val_group_ids_np = all_group_ids

        self.log("val_macro_per_class_pr_auc", val_macro_pc, prog_bar=True)
        self.log("val_macro_per_class_pr_auc_monitor", val_macro_pc, prog_bar=True)
        self.log("val_worst_class_pr_auc", val_worst_pc, prog_bar=True)
        self.log("val_pr_auc", val_pr_auc, prog_bar=True)
        self.log("val_roc_auc", val_roc_auc, prog_bar=True)

        pc_parts = "  ".join(f"cls{k}={v:.3f}" for k, v in sorted(per_class_detail.items()))
        logger.info(
            "Epoch {:3d} | macro_pc={:.4f}  worst={:.4f}  binary={:.4f}  "
            "score_p95={:.4f}  score_max={:.4f} | {}",
            self.current_epoch, val_macro_pc, val_worst_pc, val_pr_auc,
            float(np.percentile(all_scores, 95)), float(np.max(all_scores)), pc_parts,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        x, labels, original_labels, group_ids = self._unpack_batch(batch)
        x_flat = self._flatten(x)
        scores = self._score_batch(x_flat).detach().cpu()
        self._test_outputs.append({
            "scores": scores,
            "labels": labels.detach().cpu(),
            "original_labels": original_labels.detach().cpu(),
            "group_ids": group_ids,
        })

    def on_test_epoch_end(self) -> None:
        if not self._test_outputs:
            return
        all_scores = torch.cat([o["scores"] for o in self._test_outputs]).numpy()
        all_labels = torch.cat([o["labels"] for o in self._test_outputs]).numpy()
        all_original_labels = torch.cat([o["original_labels"] for o in self._test_outputs]).numpy()
        all_group_ids = None
        if self._test_outputs[0].get("group_ids") is not None:
            all_group_ids = np.array(
                [gid for o in self._test_outputs for gid in (o.get("group_ids") or [])],
                dtype=object,
            )
        self._test_outputs.clear()
        self._test_scores_np = all_scores
        self._test_labels_np = all_labels
        self._test_original_labels_np = all_original_labels
        self._test_group_ids_np = all_group_ids

    @torch.no_grad()
    def collect_train_scores(self, train_dl: DataLoader) -> np.ndarray:
        self.eval()
        chunks: list[np.ndarray] = []
        for batch in train_dl:
            x = batch[0].to(self.device)
            x_flat = self._flatten(x)
            scores = self._score_batch(x_flat).detach().cpu().numpy()
            chunks.append(scores)
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        sch = _build_warmup_cosine_scheduler(
            opt, warmup_steps=self.warmup_steps,
            total_steps=self.total_steps, min_lr_ratio=self.min_lr_ratio,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}


# ─────────────────────────────────────────────────────────────────────────────
# Optuna pruning callback
# ─────────────────────────────────────────────────────────────────────────────

class _OptunaPruningCallback(Callback):
    def __init__(self, trial: optuna.Trial, monitor: str = "val_macro_per_class_pr_auc_monitor", warmup_epochs: int = 3) -> None:
        self._trial = trial
        self._monitor = monitor
        self._warmup_epochs = int(warmup_epochs)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking or trainer.current_epoch < self._warmup_epochs:
            return
        logged = trainer.callback_metrics.get(self._monitor)
        if logged is None:
            return
        value = float(logged.detach().cpu().item()) if hasattr(logged, "detach") else float(logged)
        self._trial.report(value, step=trainer.current_epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()


# ─────────────────────────────────────────────────────────────────────────────
# HPO
# ─────────────────────────────────────────────────────────────────────────────

def _run_hpo(
    base_cfg: dict,
    hpo_cfg: dict,
    training_cfg: dict,
    train_dl: DataLoader,
    val_dl: DataLoader,
    scaler_mean_t: torch.Tensor,
    scaler_scale_t: torch.Tensor,
    feature_idx: dict,
    n_features: int,
    n_non_context: int,
    context_feature_indices: list[int],
    seed: int,
    n_trials_override: int | None,
) -> tuple[dict, int]:
    search_space = hpo_cfg.get("search_space", {})
    n_trials = n_trials_override or int(hpo_cfg.get("n_trials", 30))
    timeout = hpo_cfg.get("timeout_seconds", None)
    full_epochs = int(base_cfg.get("max_epochs", training_cfg.get("max_epochs", 50)))
    hpo_epochs = max(
        int(hpo_cfg.get("min_hpo_epochs", 5)),
        int(full_epochs * float(hpo_cfg.get("max_epochs_fraction", 0.30))),
    )
    hpo_patience = max(
        int(hpo_cfg.get("min_hpo_patience", 3)),
        int(int(training_cfg.get("patience", 10)) * float(hpo_cfg.get("patience_fraction", 0.40))),
    )
    pruning_warmup = int(hpo_cfg.get("pruning_warmup_epochs", 3))

    def objective(trial: optuna.Trial) -> float:
        suggested = suggest_params_from_space(trial, search_space)
        cfg = {**base_cfg, **suggested}
        hpo_model = PCFlowModel(
            n_features=n_non_context,
            n_context=len(context_feature_indices),
            n_coupling_layers=int(cfg.get("n_coupling_layers", 4)),
            hidden_dim=int(cfg.get("hidden_dim", 32)),
            dropout=float(suggested.get("dropout", cfg.get("dropout", 0.0))),
        )
        hpo_lit = PCFlowLightningModule(
            model=hpo_model,
            scaler_mean=scaler_mean_t,
            scaler_scale=scaler_scale_t,
            feature_idx=feature_idx,
            context_feature_indices=context_feature_indices,
            learning_rate=float(suggested.get("learning_rate", cfg.get("learning_rate", 3e-4))),
            weight_decay=float(suggested.get("weight_decay", cfg.get("weight_decay", 1e-5))),
            gradient_clip_val=float(cfg.get("gradient_clip_val", 1.0)),
            max_epochs=hpo_epochs,
            total_steps=max(1, len(train_dl) * hpo_epochs),
            warmup_steps=int(training_cfg.get("warmup_steps", 0)),
            min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.01)),
            hpo_mode=True,
        )
        hpo_trainer = pl.Trainer(
            max_epochs=hpo_epochs,
            callbacks=[
                EarlyStopping(monitor="val_macro_per_class_pr_auc_monitor", patience=hpo_patience, mode="max"),
                _OptunaPruningCallback(trial, warmup_epochs=pruning_warmup),
            ],
            enable_progress_bar=False, enable_model_summary=False,
            enable_checkpointing=False, logger=False, deterministic=False,
            gradient_clip_val=float(cfg.get("gradient_clip_val", 1.0)),
            **_trainer_runtime_kwargs(training_cfg, cfg),
        )
        try:
            hpo_trainer.fit(hpo_lit, train_dataloaders=train_dl, val_dataloaders=val_dl)
        except optuna.exceptions.TrialPruned:
            raise
        except (ValueError, RuntimeError) as exc:
            if any(s in str(exc).lower() for s in ["nan", "inf", "input contains"]):
                raise optuna.exceptions.TrialPruned() from exc
            raise
        return float(hpo_lit.best_val_pr_auc)

    sampler = _build_sampler(str(hpo_cfg.get("sampler", "tpe")), seed=seed)
    pruner = _build_pruner(str(hpo_cfg.get("pruner", "median")))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    _start = time.time(); _done = 0
    def _objective_with_progress(trial: optuna.Trial) -> float:
        nonlocal _done
        try:
            return objective(trial)
        finally:
            _done += 1
            elapsed = time.time() - _start
            rate = _done / elapsed if elapsed > 0 else 0
            if n_trials:
                eta = (n_trials - _done) / rate if rate > 0 else 0
                logger.info("HPO trial {}/{} | ETA {:.0f}s", _done, n_trials, eta)

    study.optimize(_objective_with_progress, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_completed == 0:
        raise RuntimeError("HPO finished with zero completed trials.")
    best = study.best_trial
    logger.info("HPO done: best={:.4f} | params={}", best.value, best.params)
    return best.params, n_completed


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers (same as PC-AE)
# ─────────────────────────────────────────────────────────────────────────────

def _save_pr_curve(val_s, val_l, test_s, test_l, path: Path, model_name: str = "PC-Flow") -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for scores, labels, split in [(val_s, val_l, "val"), (test_s, test_l, "test")]:
        p, r, _ = precision_recall_curve(labels, scores)
        auc = average_precision_score(labels, scores)
        ax.plot(r, p, label=f"{split} (PR-AUC={auc:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} PR Curve"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def _save_score_histogram(val_s, val_l, threshold: float, path: Path, model_name: str = "PC-Flow") -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(val_s[val_l == 0], bins=80, alpha=0.6, label="Normal", density=True)
    if val_l.sum() > 0:
        ax.hist(val_s[val_l == 1], bins=80, alpha=0.6, label="Fault", density=True)
    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.4f}")
    ax.set_xlabel("NLL score"); ax.set_ylabel("Density")
    ax.set_title(f"{model_name} Score Distribution"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


def _save_score_timeline(test_s, test_l, threshold: float, path: Path, model_name: str = "PC-Flow") -> None:
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(test_s, lw=0.5, alpha=0.7, label="NLL score")
    ax.axhline(threshold, color="red", linestyle="--", lw=1.0, label="Threshold")
    fault_idx = np.where(test_l == 1)[0]
    if len(fault_idx):
        ax.scatter(fault_idx, test_s[fault_idx], s=2, c="red", alpha=0.3, label="Fault")
    ax.set_xlabel("Sample"); ax.set_ylabel("NLL score")
    ax.set_title(f"{model_name} Score Timeline"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_pc_flow(config: dict | None = None) -> None:
    args = _parse_args()
    if config is None:
        config = _load_config()

    pc_flow_cfg: dict = config["anomaly_detection"]["dl"]["models"]["pc_flow"]
    training_cfg: dict = config.get("training", {})
    threading_cfg: dict = training_cfg.get("threading", {})
    seed: int = args.seed
    is_smoke: bool = args.smoke
    run_type = "smoke" if is_smoke else args.run_type
    variant = "PC-Flow"

    logger.info("Loading features | task={} dataset={} split_path={} profile={}",
                args.task, args.dataset, args.split_path, args.profile)
    train_df, val_df, test_df, manifest, resolved_run_dir = load_features_for_task(
        task=args.task, profile=args.profile, run_dir=args.run_dir,
        run_id=args.run_id, dataset=args.dataset, split_path=args.split_path,
    )
    val_df, test_df = apply_fold_overrides(args, val_df, test_df)

    features: list[str] = manifest.get("final_features", [])
    label_col: str = str(manifest.get("label_column", "label"))
    n_features = len(features)
    feature_idx = {name: i for i, name in enumerate(features)}

    context_feature_names: list[str] = pc_flow_cfg.get("context_features", ["irr", "pvt"])
    context_feature_indices = [feature_idx[f] for f in context_feature_names if f in feature_idx]
    if not context_feature_indices:
        raise RuntimeError(f"None of context_features {context_feature_names} found in features {features}")
    logger.info("Context features {} → indices {}", context_feature_names, context_feature_indices)

    # Non-context feature indices (what the flow models)
    all_idx = set(range(n_features))
    non_ctx_idx = sorted(all_idx - set(context_feature_indices))
    n_non_context = len(non_ctx_idx)
    logger.info("Flow input dim: {} (total {} − context {})", n_non_context, n_features, len(context_feature_indices))

    scaler = StandardScaler()
    train_df = train_df.copy(); val_df = val_df.copy(); test_df = test_df.copy()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])
    scaler_mean_t = torch.tensor(scaler.mean_, dtype=torch.float32)
    scaler_scale_t = torch.tensor(scaler.scale_, dtype=torch.float32)

    win_size = int(pc_flow_cfg.get("win_size", 1))
    train_stride = int(pc_flow_cfg.get("train_stride", 1))
    eval_stride = int(pc_flow_cfg.get("eval_stride", 1))
    batch_size = int(pc_flow_cfg.get("batch_size", 4096))
    max_epochs = int(pc_flow_cfg.get("max_epochs", training_cfg.get("max_epochs", 50)))
    patience = int(pc_flow_cfg.get("patience", training_cfg.get("patience", 10)))

    if is_smoke:
        max_epochs = 1; patience = 1
        logger.info("Smoke mode: 1 epoch")

    loader_runtime = _resolve_loader_runtime(threading_cfg, hpo_mode=bool(getattr(args, "hpo", False)))

    def _make_dataloaders(bs: int):
        ds_train = TimeSeriesDataset(train_df, features, label_col, win_size, train_stride, normal_only=True)
        ds_val = TimeSeriesDataset(val_df, features, label_col, win_size, eval_stride,
                                   normal_only=False, return_original_label=True, return_group_id=True)
        ds_test = TimeSeriesDataset(test_df, features, label_col, win_size, eval_stride,
                                    normal_only=False, return_original_label=True, return_group_id=True)
        kw = {"drop_last": False, "num_workers": loader_runtime["num_workers"],
              "pin_memory": loader_runtime["pin_memory"],
              "persistent_workers": loader_runtime["persistent_workers"]}
        if loader_runtime["num_workers"] > 0:
            kw["prefetch_factor"] = loader_runtime["prefetch_factor"]
        return (DataLoader(ds_train, batch_size=bs, shuffle=True, **kw),
                DataLoader(ds_val, batch_size=bs, shuffle=False, **kw),
                DataLoader(ds_test, batch_size=bs, shuffle=False, **kw))

    train_dl, val_dl, test_dl = _make_dataloaders(batch_size)
    train_dl_seq = DataLoader(
        TimeSeriesDataset(train_df, features, label_col, win_size, train_stride, normal_only=True),
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=loader_runtime["num_workers"], pin_memory=loader_runtime["pin_memory"],
        persistent_workers=loader_runtime["persistent_workers"],
        **({"prefetch_factor": loader_runtime["prefetch_factor"]} if loader_runtime["num_workers"] > 0 else {}),
    )
    logger.info("Samples — train: {:,}  val: {:,}  test: {:,}",
                len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))

    # HPO or inject best_params
    hpo_params: dict = {}
    _hpo_n_completed: int | None = None
    if args.best_params:
        hpo_params = json.loads(args.best_params)
        logger.info("Applying best_params from CLI: {}", hpo_params)
    elif getattr(args, "hpo", False):
        hpo_cfg = pc_flow_cfg.get("hpo", {})
        logger.info("Starting HPO sweep ({} trials)...", args.n_trials or hpo_cfg.get("n_trials", 30))
        hpo_params, _hpo_n_completed = _run_hpo(
            base_cfg=pc_flow_cfg, hpo_cfg=hpo_cfg, training_cfg=training_cfg,
            train_dl=train_dl, val_dl=val_dl,
            scaler_mean_t=scaler_mean_t, scaler_scale_t=scaler_scale_t,
            feature_idx=feature_idx, n_features=n_features,
            n_non_context=n_non_context,
            context_feature_indices=context_feature_indices,
            seed=seed, n_trials_override=args.n_trials,
        )

    if hpo_params:
        pc_flow_cfg = {**pc_flow_cfg, **hpo_params}

    pl.seed_everything(seed, workers=True)
    model = PCFlowModel(
        n_features=n_non_context,
        n_context=len(context_feature_indices),
        n_coupling_layers=int(pc_flow_cfg.get("n_coupling_layers", 4)),
        hidden_dim=int(pc_flow_cfg.get("hidden_dim", 32)),
        dropout=float(pc_flow_cfg.get("dropout", 0.0)),
    )
    logger.info("PC-Flow params: {:,}", model.n_params)

    lit = PCFlowLightningModule(
        model=model, scaler_mean=scaler_mean_t, scaler_scale=scaler_scale_t,
        feature_idx=feature_idx, context_feature_indices=context_feature_indices,
        learning_rate=float(pc_flow_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(pc_flow_cfg.get("weight_decay", 1e-5)),
        gradient_clip_val=float(pc_flow_cfg.get("gradient_clip_val", 1.0)),
        max_epochs=max_epochs,
        total_steps=max(1, len(train_dl) * max_epochs),
        warmup_steps=int(training_cfg.get("warmup_steps", 0)),
        min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.01)),
    )

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else get_experiments_root() / "anomaly" / "pc_flow" / ts
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = artifacts_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_macro_per_class_pr_auc_monitor", patience=patience, mode="max"),
        ModelCheckpoint(dirpath=str(ckpt_dir), filename="best",
                        monitor="val_macro_per_class_pr_auc_monitor", mode="max",
                        save_top_k=1, save_last=True),
    ]
    trainer = pl.Trainer(
        max_epochs=max_epochs, callbacks=callbacks,
        enable_progress_bar=False, enable_model_summary=False,
        enable_checkpointing=True, logger=False, deterministic=False,
        gradient_clip_val=float(pc_flow_cfg.get("gradient_clip_val", 1.0)),
        **_trainer_runtime_kwargs(training_cfg, pc_flow_cfg),
    )
    logger.info("Training {}", variant)
    t0 = time.perf_counter()
    trainer.fit(lit, train_dataloaders=train_dl, val_dataloaders=val_dl)
    fit_time = time.perf_counter() - t0
    logger.info("Training done in {:.1f}s", fit_time)

    ckpt_cb = next((c for c in callbacks if isinstance(c, ModelCheckpoint)), None)
    best_ckpt_path: str | None = ckpt_cb.best_model_path if (ckpt_cb and ckpt_cb.best_model_path) else None

    if best_ckpt_path:
        pl.Trainer(enable_progress_bar=False, enable_model_summary=False, logger=False).validate(
            lit, dataloaders=val_dl, ckpt_path=best_ckpt_path
        )

    pl.Trainer(enable_progress_bar=False, enable_model_summary=False, logger=False).test(
        lit, dataloaders=test_dl, ckpt_path=best_ckpt_path
    )

    # Threshold calibration on TRAIN NLL scores (normal only)
    thr_cfg = load_threshold_config(config, pc_flow_cfg)
    train_scores_np = lit.collect_train_scores(train_dl_seq)
    threshold, threshold_diag = calibrate_threshold(train_scores_np, **thr_cfg)
    lit.val_threshold = threshold
    threshold_policy_str_val = _threshold_policy_str(threshold_diag)
    logger.info("Threshold ({} strategy) = {:.6f}", threshold_diag.get("strategy"), threshold)

    val_scores = lit._val_scores_np; val_labels = lit._val_labels_np
    val_original_labels = lit._val_original_labels_np; val_group_ids = lit._val_group_ids_np
    test_scores = lit._test_scores_np; test_labels = lit._test_labels_np
    test_original_labels = lit._test_original_labels_np; test_group_ids = lit._test_group_ids_np
    if val_scores is None or test_scores is None:
        logger.error("Scores not populated."); return

    val_pr_auc = float(average_precision_score(val_labels, val_scores))
    val_roc_auc = float(roc_auc_score(val_labels, val_scores))
    val_preds = (val_scores >= threshold).astype(int)
    val_f1 = float(f1_score(val_labels, val_preds, zero_division=0))
    val_prec = float(precision_score(val_labels, val_preds, zero_division=0))
    val_rec = float(recall_score(val_labels, val_preds, zero_division=0))
    val_acc = float(accuracy_score(val_labels, val_preds))

    test_pr_auc = float(average_precision_score(test_labels, test_scores))
    test_roc_auc = float(roc_auc_score(test_labels, test_scores))
    test_preds = (test_scores >= threshold).astype(int)
    test_f1 = float(f1_score(test_labels, test_preds, zero_division=0))
    test_prec = float(precision_score(test_labels, test_preds, zero_division=0))
    test_rec = float(recall_score(test_labels, test_preds, zero_division=0))
    test_acc = float(accuracy_score(test_labels, test_preds))

    val_episode_macro_f1 = episode_macro_f1_binary(val_labels, val_preds, val_group_ids)
    test_episode_macro_f1 = episode_macro_f1_binary(test_labels, test_preds, test_group_ids)

    val_pc_source = val_original_labels if val_original_labels is not None else val_labels
    test_pc_source = test_original_labels if test_original_labels is not None else test_labels
    val_macro = compute_macro_per_class_pr_auc(labels=val_pc_source, scores=val_scores)
    test_macro = compute_macro_per_class_pr_auc(labels=test_pc_source, scores=test_scores)

    val_episode_metrics = {}; test_episode_metrics = {}
    if val_group_ids is not None:
        val_episode_metrics = compute_episode_level_pr_auc(
            labels=val_labels, scores=val_scores,
            original_labels=val_pc_source, group_ids=val_group_ids, agg="p95",
        )
    if test_group_ids is not None:
        test_episode_metrics = compute_episode_level_pr_auc(
            labels=test_labels, scores=test_scores,
            original_labels=test_pc_source, group_ids=test_group_ids, agg="p95",
        )

    logger.info("Test — PR-AUC={:.4f}  ROC-AUC={:.4f}  F1={:.4f}  Prec={:.4f}  Rec={:.4f}",
                test_pr_auc, test_roc_auc, test_f1, test_prec, test_rec)
    logger.info("Test — macro_pc_pr_auc={:.4f}  worst={:.4f}",
                test_macro.get("macro_per_class_pr_auc") or 0.0,
                test_macro.get("worst_class_pr_auc") or 0.0)
    if test_episode_metrics:
        logger.info("Test EPISODE — binary={:.4f}  macro_pc={:.4f}  worst={:.4f}  n_episodes={}",
                    test_episode_metrics.get("episode_binary_pr_auc") or 0.0,
                    test_episode_metrics.get("episode_macro_per_class_pr_auc") or 0.0,
                    test_episode_metrics.get("episode_worst_class_pr_auc") or 0.0,
                    test_episode_metrics.get("n_episodes") or 0)

    metrics = {
        "score_name": "pc_flow_nll",
        "score_components": ["negative_log_likelihood"],
        "threshold_policy": threshold_policy_str_val,
        "selection_metric": "val_macro_per_class_pr_auc_monitor",
        "threshold": threshold,
        "val_pr_auc": val_pr_auc, "val_roc_auc": val_roc_auc,
        "val_f1_at_threshold": val_f1, "val_precision_at_threshold": val_prec,
        "val_recall_at_threshold": val_rec, "val_accuracy_at_threshold": val_acc,
        "val_episode_macro_f1": float(val_episode_macro_f1),
        "val_macro_per_class_pr_auc": val_macro.get("macro_per_class_pr_auc"),
        "val_worst_class_pr_auc": val_macro.get("worst_class_pr_auc"),
        "val_selection_score": val_macro.get("macro_per_class_pr_auc"),
        "test_pr_auc": test_pr_auc, "test_roc_auc": test_roc_auc,
        "test_f1_at_threshold": test_f1, "test_precision_at_threshold": test_prec,
        "test_recall_at_threshold": test_rec, "test_accuracy_at_threshold": test_acc,
        "test_episode_macro_f1": float(test_episode_macro_f1),
        "test_macro_per_class_pr_auc": test_macro.get("macro_per_class_pr_auc"),
        "test_worst_class_pr_auc": test_macro.get("worst_class_pr_auc"),
        "test_class1_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("1"),
        "test_class2_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("2"),
        "test_class3_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("3"),
        "test_class4_pr_auc_vs_normal": test_macro.get("per_class_pr_auc_vs_normal", {}).get("4"),
        "val_episode_binary_pr_auc": val_episode_metrics.get("episode_binary_pr_auc"),
        "val_episode_macro_per_class_pr_auc": val_episode_metrics.get("episode_macro_per_class_pr_auc"),
        "val_episode_worst_class_pr_auc": val_episode_metrics.get("episode_worst_class_pr_auc"),
        "test_episode_binary_pr_auc": test_episode_metrics.get("episode_binary_pr_auc"),
        "test_episode_macro_per_class_pr_auc": test_episode_metrics.get("episode_macro_per_class_pr_auc"),
        "test_episode_worst_class_pr_auc": test_episode_metrics.get("episode_worst_class_pr_auc"),
        "test_episode_class1_pr_auc": test_episode_metrics.get("episode_per_class_pr_auc_vs_normal", {}).get("1"),
        "test_episode_class2_pr_auc": test_episode_metrics.get("episode_per_class_pr_auc_vs_normal", {}).get("2"),
        "test_episode_class3_pr_auc": test_episode_metrics.get("episode_per_class_pr_auc_vs_normal", {}).get("3"),
        "test_episode_class4_pr_auc": test_episode_metrics.get("episode_per_class_pr_auc_vs_normal", {}).get("4"),
        "n_train_samples": len(train_dl.dataset),
        "n_features": n_features, "n_non_context": n_non_context,
        "n_context_features": len(context_feature_indices),
        "n_params": model.n_params, "fit_time_s": round(fit_time, 2),
    }

    run_name = f"pc_flow_{ts}"
    run_params = {
        "variant": variant,
        **{k: v for k, v in pc_flow_cfg.items() if not isinstance(v, dict)},
        "max_epochs_actual": max_epochs, "n_features": n_features,
        "n_non_context": n_non_context, "n_params": model.n_params,
        "seed": seed, "context_features": context_feature_names,
    }

    # Save artifacts
    global_metrics_path = artifacts_dir / "global_metrics.json"
    per_class_metrics_path = artifacts_dir / "per_class_metrics.json"
    episode_metrics_path = artifacts_dir / "episode_metrics.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    deployment_manifest_path = artifacts_dir / "deployment_manifest.json"
    params_path = artifacts_dir / "run_params.json"
    hpo_params_path = artifacts_dir / "hpo_best_params.json"
    scaler_path = artifacts_dir / "scaler.joblib"
    manifest_path = artifacts_dir / "features_manifest.json"
    calibration_path = artifacts_dir / "score_calibration.json"
    pr_curve_path = artifacts_dir / "pr_curve.png"
    histogram_path = artifacts_dir / "score_histogram.png"
    timeline_path = artifacts_dir / "score_timeline.png"

    write_json(global_metrics_path, metrics)
    params_path.write_text(json.dumps(run_params, indent=2, default=str), encoding="utf-8")
    if hpo_params:
        hpo_params_path.write_text(json.dumps(hpo_params, indent=2, default=str), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    joblib.dump(scaler, scaler_path)

    per_class_metrics = compute_anomaly_per_class_metrics(
        labels=test_pc_source, scores=test_scores, threshold=threshold,
        val_labels=val_pc_source, val_scores=val_scores,
    )
    candidate_thresholds = build_candidate_per_true_class_thresholds(per_class_metrics, normal_label=0)
    calibration_payload = build_score_calibration_payload(
        threshold=threshold, threshold_policy=threshold_policy_str_val,
        threshold_quantile=float(thr_cfg.get("percentile", 95.0)) / 100.0,
        candidate_per_true_class_thresholds=candidate_thresholds,
        score_stats={"score_name": "pc_flow_nll", "score_components": ["negative_log_likelihood"]},
    )
    calibration_payload["threshold_diagnostics"] = threshold_diag
    write_json(calibration_path, calibration_payload)
    write_json(per_class_metrics_path, per_class_metrics)
    write_json(episode_metrics_path, {"val": val_episode_metrics, "test": test_episode_metrics, "agg": "p95"})

    run_manifest = build_run_manifest(
        task=args.task, model="pc_flow", model_family="anomaly_dl",
        dataset=args.dataset, split_path=args.split_path,
        feature_profile=str(args.profile), feature_run_dir=str(resolved_run_dir),
        seed=seed, run_type=run_type,
        extras={"run_name": run_name, "variant": variant,
                "contribution": "conditional_normalizing_flow_nll_anomaly_scoring",
                "n_params": model.n_params},
    )
    deployment_manifest = build_deployment_manifest(
        task=args.task, model="pc_flow", model_family="anomaly_dl",
        model_artifact="checkpoints/best.ckpt" if best_ckpt_path else "checkpoints/last.ckpt",
        scaler_artifact=scaler_path.name,
        feature_names=features, label_column=label_col,
        threshold=threshold, window_size=win_size,
        score_direction="higher_is_more_anomalous",
        classes=[str(c) for c in sorted(np.unique(test_pc_source).tolist())],
        extras={
            "checkpoint_available": bool(best_ckpt_path),
            "threshold_policy": threshold_policy_str_val,
            "selection_metric": "val_macro_per_class_pr_auc_monitor",
            "score_name": "pc_flow_nll",
            "context_features": context_feature_names,
            "n_params": model.n_params,
            "n_coupling_layers": int(pc_flow_cfg.get("n_coupling_layers", 4)),
            "onnx_exportable": True,
        },
    )
    write_json(run_manifest_path, run_manifest)
    write_json(deployment_manifest_path, deployment_manifest)

    _save_pr_curve(val_scores, val_labels, test_scores, test_labels, pr_curve_path, variant)
    _save_score_histogram(val_scores, val_labels, threshold, histogram_path, variant)
    _save_score_timeline(test_scores, test_labels, threshold, timeline_path, variant)
    logger.info("Artifacts saved → {}", artifacts_dir)

    # MLflow / DagsHub
    try:
        init_tracking("anomaly")
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({
                "task": args.task, "dataset": args.dataset, "split_path": args.split_path,
                "profile": str(args.profile), "model": "pc_flow", "model_family": "anomaly_dl",
                "variant": variant, "contribution": "conditional_normalizing_flow_nll_anomaly_scoring",
                "n_params": str(model.n_params),
            })
            mlflow.log_params({k: v for k, v in run_params.items() if not isinstance(v, (dict, list))})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float, np.integer, np.floating))
                                and not isinstance(v, bool) and v is not None})
            for cls_str, m in per_class_metrics.items():
                if m.get("pr_auc_vs_normal") is not None:
                    mlflow.log_metric(f"test_pr_auc_class{cls_str}_vs_normal", m["pr_auc_vs_normal"])
            for p in (global_metrics_path, per_class_metrics_path, episode_metrics_path,
                      run_manifest_path, deployment_manifest_path, params_path,
                      hpo_params_path, scaler_path, manifest_path, calibration_path,
                      pr_curve_path, histogram_path, timeline_path):
                if p.exists():
                    mlflow.log_artifact(str(p))
            if best_ckpt_path and Path(best_ckpt_path).exists():
                mlflow.log_artifact(best_ckpt_path, artifact_path="checkpoints")

            comparison_record = {
                "timestamp_utc": datetime.now(UTC).isoformat(), "run_name": run_name,
                "task": args.task, "dataset": args.dataset, "split_path": args.split_path,
                "model": "pc_flow", "variant": variant, "model_family": "anomaly_dl",
                "run_type": run_type, "seed": seed, "feature_profile": str(args.profile),
                "feature_run_dir": str(resolved_run_dir), "n_features": n_features,
                "n_non_context": n_non_context, "n_params": model.n_params,
                "win_size": win_size,
                "threshold": threshold, "threshold_policy": threshold_policy_str_val,
                "selection_metric": "val_macro_per_class_pr_auc_monitor",
                "score_name": "pc_flow_nll", "score_components": ["negative_log_likelihood"],
                **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
                "fit_time_s": round(fit_time, 2),
                "hpo_enabled": bool(getattr(args, "hpo", False)),
                "best_params_injected": bool(args.best_params),
                "hpo_n_trials_completed": _hpo_n_completed,
                "hpo_best_params": json.dumps(hpo_params, default=str) if hpo_params else None,
                "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
            }
            records_path = Path(args.comparison_records_path)
            records_path.parent.mkdir(parents=True, exist_ok=True)
            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(comparison_record, default=str) + "\n")
        logger.info("MLflow run logged: {}", run_name)
    except Exception as exc:
        logger.warning("MLflow logging failed (non-fatal): {}", exc)


if __name__ == "__main__":
    run_pc_flow()
