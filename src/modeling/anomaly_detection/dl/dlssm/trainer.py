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
from src.modeling.anomaly_detection.dl.dlssm.losses import (
    apply_pv_safe_augmentation,
    combine_anomaly_score_components,
    compute_anomaly_score_components,
    consistency_loss,
    expected_power_loss,
    kl_divergence,
    one_class_loss,
    physics_consistency_loss,
    prediction_loss,
    reconstruction_loss,
)
from src.modeling.anomaly_detection.dl.dlssm.model import DeepLatentStateSpaceModel
from src.modeling.common.artifact_contract import (
    build_candidate_per_true_class_thresholds,
    build_deployment_manifest,
    build_run_manifest,
    build_score_calibration_payload,
    compute_anomaly_per_class_metrics,
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
from src.modeling.common.hyperparameter_optimizer import (
    _build_pruner,
    _build_sampler,
    suggest_params_from_space,
)
from src.utils.paths import get_experiments_root

# trainer.py is under dl/dlssm/, so PROJECT_ROOT is 5 levels up
# path: dlssm/ → dl/ → anomaly_detection/ → modeling/ → src/ → PFE_Experiments/
PROJECT_ROOT = Path(__file__).resolve().parents[5]

torch.set_float32_matmul_precision("medium")


def _default_comparison_records_path() -> Path:
    return get_experiments_root() / "metrics" / "anomaly_comparison_records.jsonl"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PI-DLS-SSM anomaly detection")
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
    p.add_argument("--one-class", action="store_true", help="Deep SVDD-style latent compactness loss")
    p.add_argument("--consistency", action="store_true", help="Consistency regularization on q_mu under PV-safe augmentation")
    p.add_argument("--cvae", action="store_true", help="Conditional VAE: condition on operating-regime features (irr, pvt, ...)")
    p.add_argument("--hpo", action="store_true", help="Run Optuna HPO sweep then retrain best config")
    p.add_argument("--n-trials", type=int, default=None, help="Override number of HPO trials")
    p.add_argument("--best-params", default=None, help="JSON string of best params to apply directly (skips HPO search)")
    return p.parse_args()


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _calibrate_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
    """Return (threshold, f1, precision, recall) using q95 of normal-class scores.

    Deploy-realistic: calibrated from the 95th-percentile of the score distribution
    over normal-class (label==0) windows — no fault-label leakage at calibration
    time. F1/precision/recall are reported AT this threshold for monitoring only,
    not for model selection (selection uses val_macro_per_class_pr_auc).
    """
    normal_scores = scores[labels == 0]
    threshold = float(np.percentile(normal_scores if len(normal_scores) > 0 else scores, 95))
    preds = (scores >= threshold).astype(int)
    f1 = float(f1_score(labels, preds, zero_division=0))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    return threshold, f1, prec, rec


# ─────────────────────────────────────────────────────────────────────────────
# HPO helpers
# ─────────────────────────────────────────────────────────────────────────────

class _OptunaPruningCallback(Callback):
    """Minimal ASHA pruning callback — no optuna-integration dependency."""

    def __init__(
        self, trial: optuna.Trial, monitor: str = "val_macro_per_class_pr_auc", warmup_epochs: int = 3
    ) -> None:
        self._trial = trial
        self._monitor = monitor
        self._warmup_epochs = int(warmup_epochs)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if trainer.current_epoch < self._warmup_epochs:
            return
        logged = trainer.callback_metrics.get(self._monitor)
        if logged is None:
            return
        value = float(logged.detach().cpu().item()) if hasattr(logged, "detach") else float(logged)
        self._trial.report(value, step=trainer.current_epoch)
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned()

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
    physics_enabled: bool,
    oc_enabled: bool,
    consistency_enabled: bool,
    seed: int,
    n_trials_override: int | None,
    cvae_enabled: bool = False,
    condition_idx: list[int] | None = None,
    condition_dim: int = 0,
    recon_mask: torch.Tensor | None = None,
    # DLSSM-FDD params
    slow_context_enabled: bool = False,
    slow_hidden_dim: int = 0,
    n_attention_heads: int = 4,
    expected_power_enabled: bool = False,
    expected_power_indices: list[int] | None = None,
    lambda_expected: float = 0.5,
    lambda_pr_score: float = 0.1,
    oc_center_max_norm: float = 50.0,
    score_instability_p95: float = 1e6,
    score_instability_max: float = 1e7,
    non_finite_warn_limit_per_epoch: int = 5,
    max_abs_oc_score_term: float | None = 1e5,
    max_abs_phys_score_term: float | None = 1e5,
    max_abs_pr_score_term: float | None = 1e5,
    max_abs_base_score_term: float | None = 1e5,
    max_abs_pred_score_term: float | None = 1e5,
) -> dict:
    """Run Optuna TPE + ASHA sweep. Returns best params dict."""
    search_space = hpo_cfg.get("search_space", {})
    n_trials = n_trials_override or int(hpo_cfg.get("n_trials", 40))
    timeout = hpo_cfg.get("timeout_seconds", None)
    full_epochs = int(base_cfg.get("max_epochs", training_cfg.get("max_epochs", 100)))
    hpo_epochs = max(
        int(hpo_cfg.get("min_hpo_epochs", 10)),
        int(full_epochs * float(hpo_cfg.get("max_epochs_fraction", 0.25))),
    )
    hpo_patience = max(
        int(hpo_cfg.get("min_hpo_patience", 5)),
        int(int(training_cfg.get("patience", 15)) * float(hpo_cfg.get("patience_fraction", 0.33))),
    )
    pruning_warmup_epochs = int(hpo_cfg.get("pruning_warmup_epochs", 3))

    def objective(trial: optuna.Trial) -> float:
        suggested = suggest_params_from_space(trial, search_space)
        cfg = {**base_cfg, **suggested}

        hpo_model = DeepLatentStateSpaceModel(
            n_features=n_features,
            win_size=int(cfg["win_size"]),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            latent_dim=int(cfg.get("latent_dim", 32)),
            encoder_dim=int(cfg.get("encoder_dim", 256)),
            decoder_dim=int(cfg.get("decoder_dim", 256)),
            n_gru_layers=int(cfg.get("n_gru_layers", 1)),
            dropout=float(suggested.get("dropout", cfg.get("dropout", 0.1))),
            condition_dim=condition_dim,
            n_attention_heads=n_attention_heads,
            slow_hidden_dim=slow_hidden_dim if slow_context_enabled else 0,
            expected_power_indices=expected_power_indices if expected_power_enabled else None,
        )
        physics_components = cfg.get("physics_components", {})
        hpo_lit = DLSSMLightningModule(
            model=hpo_model,
            scaler_mean=scaler_mean_t,
            scaler_scale=scaler_scale_t,
            feature_idx=feature_idx,
            learning_rate=float(suggested.get("learning_rate", cfg.get("learning_rate", 1e-3))),
            weight_decay=float(suggested.get("weight_decay", cfg.get("weight_decay", 1e-5))),
            gradient_clip_val=float(cfg.get("gradient_clip_val", 1.0)),
            max_epochs=hpo_epochs,
            total_steps=max(1, len(train_dl) * hpo_epochs),
            warmup_steps=int(training_cfg.get("warmup_steps", 0)),
            min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.01)),
            beta_kl=float(cfg.get("beta_kl", 0.1)),
            kl_warmup_epochs=int(cfg.get("kl_warmup_epochs", 5)),
            lambda_phys=float(suggested.get("lambda_phys", cfg.get("lambda_phys", 0.1))),
            enable_string_power=bool(physics_components.get("string_power", True)),
            enable_imbalance=bool(physics_components.get("imbalance", True)),
            physics_enabled=physics_enabled,
            lambda_kl_score=float(suggested.get("lambda_kl_score", cfg.get("lambda_kl_score", 0.1))),
            lambda_phys_score=float(suggested.get("lambda_phys_score", cfg.get("lambda_phys_score", 0.0))),
            alpha_pred=float(cfg.get("alpha_pred", 0.1)),
            lambda_pred_score=float(suggested.get("lambda_pred_score", cfg.get("lambda_pred_score", 0.0))),
            score_reduction=str(cfg.get("score_reduction", "max")),
            free_bits=float(suggested.get("free_bits", cfg.get("free_bits", 0.05))),
            oc_enabled=oc_enabled,
            consistency_enabled=consistency_enabled,
            lambda_oc=float(suggested.get("lambda_oc", cfg.get("lambda_oc", 0.1))),
            lambda_oc_score=float(suggested.get("lambda_oc_score", cfg.get("lambda_oc_score", 0.0))),
            oc_warmup_epochs=int(cfg.get("oc_warmup_epochs", 3)),
            lambda_cons=float(suggested.get("lambda_cons", cfg.get("lambda_cons", 0.1))),
            aug_noise_std=float(suggested.get("aug_noise_std", cfg.get("aug_noise_std", 0.05))),
            aug_feature_dropout=float(suggested.get("aug_feature_dropout", cfg.get("aug_feature_dropout", 0.1))),
            latent_dim=int(cfg.get("latent_dim", 32)),
            cvae_enabled=cvae_enabled,
            condition_idx=condition_idx or [],
            recon_mask=recon_mask,
            slow_context_enabled=slow_context_enabled,
            expected_power_enabled=expected_power_enabled,
            expected_power_indices=expected_power_indices,
            lambda_expected=float(suggested.get("lambda_expected", cfg.get("lambda_expected", lambda_expected))),
            lambda_pr_score=float(suggested.get("lambda_pr_score", cfg.get("lambda_pr_score", lambda_pr_score))),
            oc_center_max_norm=oc_center_max_norm,
            score_instability_p95=score_instability_p95,
            score_instability_max=score_instability_max,
            non_finite_warn_limit_per_epoch=non_finite_warn_limit_per_epoch,
            max_abs_oc_score_term=max_abs_oc_score_term,
            max_abs_phys_score_term=max_abs_phys_score_term,
            max_abs_pr_score_term=max_abs_pr_score_term,
            max_abs_base_score_term=max_abs_base_score_term,
            max_abs_pred_score_term=max_abs_pred_score_term,
            hpo_mode=True,
        )

        hpo_trainer = pl.Trainer(
            max_epochs=hpo_epochs,
            callbacks=[
                EarlyStopping(monitor="val_macro_per_class_pr_auc", patience=hpo_patience, mode="max", verbose=False),
                _OptunaPruningCallback(
                    trial, monitor="val_macro_per_class_pr_auc", warmup_epochs=pruning_warmup_epochs
                ),
            ],
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            deterministic=False,
            gradient_clip_val=float(cfg.get("gradient_clip_val", 1.0)),
            **_trainer_runtime_kwargs(training_cfg, cfg),
        )
        try:
            hpo_trainer.fit(hpo_lit, train_dataloaders=train_dl, val_dataloaders=val_dl)
        except optuna.exceptions.TrialPruned:
            raise
        except ValueError as exc:
            # Some unstable trials can emit NaN/Inf validation scores that break sklearn metrics.
            # Mark such trials as pruned instead of aborting the whole sweep.
            if "Input contains NaN" in str(exc) or "infinity" in str(exc).lower():
                logger.warning("Pruning unstable HPO trial due to non-finite validation scores: {}", exc)
                raise optuna.exceptions.TrialPruned() from exc
            raise

        if hpo_lit.non_finite_events > 0:
            reason = hpo_lit.stability_reason or f"{hpo_lit.non_finite_events} non-finite events"
            logger.warning("Pruning unstable HPO trial: {}", reason)
            raise optuna.exceptions.TrialPruned()
        if hpo_lit.stability_unstable:
            reason = hpo_lit.stability_reason or "score/center instability"
            logger.warning("Pruning unstable HPO trial: {}", reason)
            raise optuna.exceptions.TrialPruned()
        return float(hpo_lit.best_val_pr_auc)

    sampler = _build_sampler(str(hpo_cfg.get("sampler", "tpe")), seed=seed)
    pruner = _build_pruner(str(hpo_cfg.get("pruner", "asha")))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Progress tracker for remaining-trials indicator
    _hpo_start = time.time()
    _hpo_done = 0

    def _objective_with_progress(trial: optuna.Trial) -> float:
        nonlocal _hpo_done
        try:
            return objective(trial)
        finally:
            _hpo_done += 1
            elapsed = time.time() - _hpo_start
            rate = _hpo_done / elapsed if elapsed > 0 else 0
            if n_trials:
                remaining = n_trials - _hpo_done
                eta_s = remaining / rate if rate > 0 else 0
                logger.info(
                    "HPO trial {}/{} ({:.1f}s/trial) | ETA {:.0f}s ({:.1f}m)",
                    _hpo_done, n_trials, 1.0 / rate if rate > 0 else 0,
                    eta_s, eta_s / 60,
                )
            else:
                logger.info(
                    "HPO trial {} ({:.1f}s/trial) | no limit",
                    _hpo_done, 1.0 / rate if rate > 0 else 0,
                )

    study.optimize(_objective_with_progress, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_completed == 0:
        raise RuntimeError(
            "HPO finished with zero completed trials (all trials pruned/failed). "
            "Relax stability constraints or narrow search space around known-stable params."
        )
    best = study.best_trial
    logger.info(
        "HPO done: best val_macro_per_class_pr_auc={:.4f} | completed={} total={} | params={}",
        best.value, n_completed, len(study.trials), best.params,
    )
    return best.params, n_completed


# ─────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ─────────────────────────────────────────────────────────────────────────────

class DLSSMLightningModule(pl.LightningModule):
    """PI-DLS-SSM training module.

    Uses standard automatic optimization (single ELBO loss with optional
    regularization terms) — no manual backward like MAAT.
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
        total_steps: int = 1,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
        beta_kl: float = 0.1,
        kl_warmup_epochs: int = 5,
        lambda_phys: float = 0.05,
        enable_string_power: bool = True,
        enable_imbalance: bool = True,
        physics_enabled: bool = True,
        lambda_kl_score: float = 0.1,
        lambda_phys_score: float = 0.0,
        alpha_pred: float = 0.1,
        lambda_pred_score: float = 0.0,
        score_reduction: str = "center",
        free_bits: float = 0.0,
        oc_enabled: bool = False,
        consistency_enabled: bool = False,
        lambda_oc: float = 0.1,
        lambda_oc_score: float = 0.0,
        oc_warmup_epochs: int = 3,
        lambda_cons: float = 0.1,
        aug_noise_std: float = 0.05,
        aug_feature_dropout: float = 0.1,
        latent_dim: int = 16,
        cvae_enabled: bool = False,
        condition_idx: list[int] | None = None,
        recon_mask: torch.Tensor | None = None,
        # DLSSM-FDD additions
        slow_context_enabled: bool = False,
        expected_power_enabled: bool = False,
        expected_power_indices: list[int] | None = None,
        lambda_expected: float = 0.5,
        lambda_pr_score: float = 0.1,
        # Stability controls
        oc_center_max_norm: float = 50.0,
        score_instability_p95: float = 1e6,
        score_instability_max: float = 1e7,
        non_finite_warn_limit_per_epoch: int = 5,
        max_abs_oc_score_term: float | None = 1e5,
        max_abs_phys_score_term: float | None = 1e5,
        max_abs_pr_score_term: float | None = 1e5,
        max_abs_base_score_term: float | None = 1e5,
        max_abs_pred_score_term: float | None = 1e5,
        hpo_mode: bool = False,
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
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.beta_kl = beta_kl
        self.kl_warmup_epochs = kl_warmup_epochs
        self.lambda_phys = lambda_phys
        self.enable_string_power = enable_string_power
        self.enable_imbalance = enable_imbalance
        self.physics_enabled = physics_enabled
        self.lambda_kl_score = lambda_kl_score
        self.lambda_phys_score = lambda_phys_score
        self.alpha_pred = alpha_pred
        self.lambda_pred_score = lambda_pred_score
        self.score_reduction = score_reduction
        self.free_bits = free_bits

        # One-class (Deep SVDD) and consistency regularization
        self.oc_enabled = oc_enabled
        self.consistency_enabled = consistency_enabled
        self.lambda_oc = lambda_oc
        self.lambda_oc_score = lambda_oc_score
        self.oc_warmup_epochs = oc_warmup_epochs
        self.lambda_cons = lambda_cons
        self.aug_noise_std = aug_noise_std
        self.aug_feature_dropout = aug_feature_dropout

        # Normal latent center: initialized after oc_warmup_epochs from the
        # mean of q_mu over the train set. Until then, OC loss stays at 0.
        self.register_buffer("c", torch.zeros(latent_dim))
        self.register_buffer("c_initialized", torch.tensor(False))

        # CVAE conditioning
        self.cvae_enabled = cvae_enabled
        self.condition_idx = list(condition_idx) if condition_idx else []
        if recon_mask is not None:
            self.register_buffer("recon_mask", recon_mask.float())
        else:
            self.recon_mask = None

        # DLSSM-FDD: slow context + expected-power head
        self.slow_context_enabled = slow_context_enabled
        self.expected_power_enabled = expected_power_enabled
        self.expected_power_indices: list[int] = list(expected_power_indices or [])
        self.lambda_expected = lambda_expected
        self.lambda_pr_score = lambda_pr_score
        self.oc_center_max_norm = float(oc_center_max_norm)
        self.score_instability_p95 = float(score_instability_p95)
        self.score_instability_max = float(score_instability_max)
        self.non_finite_warn_limit_per_epoch = int(non_finite_warn_limit_per_epoch)
        self.max_abs_oc_score_term = float(max_abs_oc_score_term) if max_abs_oc_score_term is not None else None
        self.max_abs_phys_score_term = float(max_abs_phys_score_term) if max_abs_phys_score_term is not None else None
        self.max_abs_pr_score_term = float(max_abs_pr_score_term) if max_abs_pr_score_term is not None else None
        self.max_abs_base_score_term = float(max_abs_base_score_term) if max_abs_base_score_term is not None else None
        self.max_abs_pred_score_term = float(max_abs_pred_score_term) if max_abs_pred_score_term is not None else None
        self.hpo_mode = hpo_mode

        self._val_outputs: list[dict] = []
        self._test_outputs: list[dict] = []
        self.best_val_pr_auc: float = 0.0
        self.val_threshold: float = 0.5
        self.non_finite_events: int = 0
        self.stability_unstable: bool = False
        self.stability_reason: str = ""
        self._non_finite_warned_batches: int = 0
        self._non_finite_windows_epoch: int = 0
        self._component_warned_batches: int = 0
        self._val_scores_np: np.ndarray | None = None
        self._val_labels_np: np.ndarray | None = None
        self._val_original_labels_np: np.ndarray | None = None
        self._val_group_ids_np: np.ndarray | None = None
        self._test_scores_np: np.ndarray | None = None
        self._test_labels_np: np.ndarray | None = None
        self._test_original_labels_np: np.ndarray | None = None
        self._test_group_ids_np: np.ndarray | None = None

    def _unpack_batch(
        self, batch: tuple
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, list | None]:
        """Unpack batch tuple → (x, x_slow, labels, original_labels, group_ids).

        Handles both slow-context and legacy batch shapes transparently.
        """
        if self.slow_context_enabled:
            # (x, x_slow, label[, original_label[, group_id]])
            x, x_slow = batch[0], batch[1]
            labels = batch[2]
            original_labels = batch[3] if len(batch) > 3 else labels
            group_ids = list(batch[4]) if len(batch) > 4 else None
        else:
            # legacy: (x, label[, original_label[, group_id]])
            x, x_slow = batch[0], None
            labels = batch[1]
            original_labels = batch[2] if len(batch) > 2 else labels
            group_ids = list(batch[3]) if len(batch) > 3 else None
        return x, x_slow, labels, original_labels, group_ids

    def _extract_condition(self, x: torch.Tensor) -> torch.Tensor | None:
        """Slice the conditioning sub-vector c_t from x. Returns None if CVAE off."""
        if not self.cvae_enabled or not self.condition_idx:
            return None
        return x[:, :, self.condition_idx]

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

    def _in_sanity_check(self) -> bool:
        return bool(getattr(self.trainer, "sanity_checking", False))

    def _reset_stability_state(self) -> None:
        """Clear transient stability flags before re-evaluating a checkpoint."""
        self.non_finite_events = 0
        self.stability_unstable = False
        self.stability_reason = ""
        self._non_finite_warned_batches = 0
        self._non_finite_windows_epoch = 0
        self._component_warned_batches = 0

    def _record_non_finite_event(self, message: str, *args) -> None:
        """Record a real instability event while ignoring Lightning sanity checks."""
        logger.warning(message, *args)
        if not self._in_sanity_check():
            self.non_finite_events += 1
            self.stability_unstable = True
            self.stability_reason = message.format(*args) if args else message

    def _check_model_outputs_finite(self, out: dict, stage: str, batch_idx: int) -> None:
        """Detect non-finite model outputs before score sanitization can hide them."""
        names = ("x_hat", "x_pred", "q_mu", "q_logvar", "p_mu", "p_logvar", "p_expected")
        bad_parts: list[str] = []
        for name in names:
            value = out.get(name)
            if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
                bad_parts.append(name)
        if bad_parts:
            self._record_non_finite_event(
                "{} batch {} has non-finite model outputs: {}",
                stage,
                batch_idx,
                ",".join(bad_parts),
            )

    def _check_raw_score_components(
        self, components: dict[str, torch.Tensor], stage: str, batch_idx: int
    ) -> None:
        """Use raw pre-clamp score components for instability decisions."""
        score_guard_active = bool(self.c_initialized) or (self.current_epoch >= self.oc_warmup_epochs)
        bad_parts: list[str] = []
        for name, value in components.items():
            detached = value.detach()
            finite = torch.isfinite(detached)
            has_non_finite = not bool(finite.all())
            abs_value = detached.abs()
            finite_abs = abs_value[finite]
            if finite_abs.numel() > 0:
                comp_p95 = float(torch.quantile(finite_abs.float(), 0.95).item())
                comp_max = float(finite_abs.max().item())
            else:
                comp_p95 = float("inf")
                comp_max = float("inf")
            too_large = score_guard_active and (
                comp_p95 >= self.score_instability_p95 or comp_max >= self.score_instability_max
            )
            if has_non_finite or too_large:
                bad_parts.append(
                    f"{name}(nonfinite={has_non_finite},p95={comp_p95:.4g},max={comp_max:.4g})"
                )

        if not bad_parts:
            return

        if not self._in_sanity_check():
            self.stability_unstable = True
            self.stability_reason = f"raw_score_components: {'; '.join(bad_parts)}"
            if any("nonfinite=True" in part for part in bad_parts):
                self.non_finite_events += 1
            if self.hpo_mode:
                self.trainer.should_stop = True

        if self._component_warned_batches < self.non_finite_warn_limit_per_epoch:
            logger.warning(
                "{} batch {} raw score component instability: {}",
                stage,
                batch_idx,
                "; ".join(bad_parts),
            )
            self._component_warned_batches += 1

    def _check_training_loss_finite(
        self,
        components: dict[str, torch.Tensor],
        loss: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor | None:
        """Check training loss components for non-finite values before the optimizer step.

        Returns a safe dummy loss if any component is non-finite, None otherwise.
        """
        bad_parts: list[str] = []
        for name, value in components.items():
            if not bool(torch.isfinite(value).all()):
                bad_parts.append(name)
        if not bool(torch.isfinite(loss).all()):
            bad_parts.append("final_loss")
        if not bad_parts:
            return None
        self._record_non_finite_event(
            "Training batch {} non-finite loss components: {}",
            batch_idx,
            ",".join(bad_parts),
        )
        if self.hpo_mode and not self._in_sanity_check():
            self.trainer.should_stop = True
        return torch.tensor(1.0, device=loss.device, dtype=loss.dtype)

    @torch.no_grad()
    def _init_one_class_center(self) -> None:
        """Set self.c to mean(q_mu) over the train loader after warmup.

        Anti-collapse: any |c_i| < 1e-3 is floored to ±1e-3 to avoid the
        trivial c → 0, q_mu → 0 solution that Deep SVDD warns about.
        """
        try:
            train_dl = self.trainer.train_dataloader
        except Exception:
            train_dl = None
        if train_dl is None:
            logger.warning("OC center init: no train dataloader available — skipping")
            return

        was_training = self.training
        self.eval()
        n_seen = 0
        running = torch.zeros_like(self.c)
        n_good = 0
        for batch in train_dl:
            n_seen += 1
            x, x_slow, _, _, _ = self._unpack_batch(
                tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
            )
            c = self._extract_condition(x)
            out = self.model(x, c, x_slow)
            q_mean = out["q_mu"].mean(dim=(0, 1)).detach()
            if not bool(torch.isfinite(q_mean).all()):
                continue
            running = running + q_mean
            n_good += 1
        if n_seen == 0 or n_good == 0:
            logger.warning(
                "OC center init: no valid batches (seen={}, valid={}) — skipping",
                n_seen,
                n_good,
            )
            if n_seen > 0 and not bool(getattr(self.trainer, "sanity_checking", False)):
                self.stability_unstable = True
                self.stability_reason = "oc_center_init: no valid batches"
            if was_training:
                self.train()
            return
        c_new = running / n_good

        if not bool(torch.isfinite(c_new).all()):
            logger.warning("OC center init produced non-finite center — skipping this epoch")
            if not self._in_sanity_check():
                self.stability_unstable = True
                self.stability_reason = "oc_center_init: non-finite center"
            if was_training:
                self.train()
            return

        c_norm = float(c_new.norm().item())
        if c_norm > self.oc_center_max_norm:
            scale = self.oc_center_max_norm / max(c_norm, 1e-12)
            c_new = c_new * scale
            logger.warning(
                "OC center norm {:.4f} exceeded max {:.4f}; clipping",
                c_norm,
                self.oc_center_max_norm,
            )

        # Anti-collapse floor
        small = c_new.abs() < 1e-3
        sign = torch.where(c_new == 0, torch.ones_like(c_new), torch.sign(c_new))
        c_new = torch.where(small, sign * 1e-3, c_new)

        self.c.copy_(c_new)
        self.c_initialized.fill_(True)
        if was_training:
            self.train()
        logger.info("Initialized one-class center | ||c||={:.4f}", float(self.c.norm()))

    def on_train_epoch_start(self) -> None:
        if (
            self.oc_enabled
            and not bool(self.c_initialized)
            and self.current_epoch >= self.oc_warmup_epochs
        ):
            self._init_one_class_center()

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, x_slow, _, _, _ = self._unpack_batch(batch)
        c = self._extract_condition(x)
        out = self.model(x, c, x_slow)

        # Per-window loss components [B]
        recon_w = reconstruction_loss(out["x_hat"], x, recon_mask=self.recon_mask).mean(dim=1)
        pred_w = prediction_loss(out["x_pred"], x, recon_mask=self.recon_mask).mean(dim=1)
        kl_w = kl_divergence(
            out["q_mu"], out["q_logvar"], out["p_mu"], out["p_logvar"],
            free_bits=self.free_bits,
        ).mean(dim=1)
        phys_w = self._physics_loss(out["x_hat"])

        # KL warmup: ramp beta_kl from 0 to beta_kl over kl_warmup_epochs
        beta = self.beta_kl * min(1.0, (self.current_epoch + 1) / max(self.kl_warmup_epochs, 1))

        lambda_phys = self.lambda_phys if self.physics_enabled else 0.0

        # One-class compactness (only after c is initialized)
        if self.oc_enabled and bool(self.c_initialized):
            oc_w = one_class_loss(out["q_mu"], self.c)
            lambda_oc_eff = self.lambda_oc
        else:
            oc_w = torch.zeros_like(recon_w)
            lambda_oc_eff = 0.0

        # Consistency regularization: extra forward pass on perturbed input.
        # When CVAE is on, the conditioning columns are protected from
        # augmentation so c_t stays the same — q(z|x,c) ≈ q(z|x_aug,c).
        if self.consistency_enabled:
            x_aug = apply_pv_safe_augmentation(
                x, self.aug_noise_std, self.aug_feature_dropout,
                protect_idx=self.condition_idx if self.cvae_enabled else None,
            )
            c_aug = self._extract_condition(x_aug)  # equals c if protected
            out_aug = self.model(x_aug, c_aug, x_slow)
            cons_w = consistency_loss(out["q_mu"], out_aug["q_mu"])
            lambda_cons_eff = self.lambda_cons
        else:
            cons_w = torch.zeros_like(recon_w)
            lambda_cons_eff = 0.0

        # Expected-power head loss (IEC 61724 Performance Ratio training signal)
        if self.expected_power_enabled and out.get("p_expected") is not None:
            exp_w = expected_power_loss(out["p_expected"], x, self.expected_power_indices)
            lambda_exp_eff = self.lambda_expected
        else:
            exp_w = torch.zeros_like(recon_w)
            lambda_exp_eff = 0.0

        per_window = (
            recon_w
            + self.alpha_pred * pred_w
            + beta * kl_w
            + lambda_phys * phys_w
            + lambda_oc_eff * oc_w
            + lambda_cons_eff * cons_w
            + lambda_exp_eff * exp_w
        )
        loss = per_window.mean()

        # Non-finite guard: check all loss components before optimizer consumes them
        loss_components = {
            "recon": recon_w,
            "pred": pred_w,
            "kl": kl_w,
            "phys": phys_w,
            "oc": oc_w,
            "cons": cons_w,
            "exp": exp_w,
        }
        dummy = self._check_training_loss_finite(loss_components, loss, batch_idx)
        if dummy is not None:
            return dummy

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon", recon_w.mean(), on_step=False, on_epoch=True)
        self.log("train_pred", pred_w.mean(), on_step=False, on_epoch=True)
        self.log("train_kl", kl_w.mean(), on_step=False, on_epoch=True)
        self.log("train_phys", phys_w.mean(), on_step=False, on_epoch=True)
        self.log("train_oc", oc_w.mean(), on_step=False, on_epoch=True)
        self.log("train_cons", cons_w.mean(), on_step=False, on_epoch=True)
        self.log("train_expected", exp_w.mean(), on_step=False, on_epoch=True)
        self.log("train_c_norm", self.c.norm(), on_step=False, on_epoch=True)
        self.log("train_beta_kl", beta, on_step=False, on_epoch=True)
        return loss

    def _score_batch(self, x: torch.Tensor, out: dict) -> torch.Tensor:
        scores, _ = self._score_batch_with_components(x, out)
        return scores

    def _score_batch_with_components(
        self, x: torch.Tensor, out: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = compute_anomaly_score_components(
            x=x,
            x_hat=out["x_hat"],
            q_mu=out["q_mu"],
            q_logvar=out["q_logvar"],
            p_mu=out["p_mu"],
            p_logvar=out["p_logvar"],
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
            c=self.c if (self.oc_enabled and bool(self.c_initialized)) else None,
            lambda_oc_score=self.lambda_oc_score,
            recon_mask=self.recon_mask,
            p_expected=out.get("p_expected"),
            lambda_pr_score=self.lambda_pr_score if self.expected_power_enabled else 0.0,
            expected_power_indices=self.expected_power_indices if self.expected_power_enabled else None,
        )
        scores = combine_anomaly_score_components(
            components,
            lambda_pred_score=self.lambda_pred_score,
            lambda_oc_score=self.lambda_oc_score,
            lambda_phys_score=self.lambda_phys_score,
            lambda_pr_score=self.lambda_pr_score if self.expected_power_enabled else 0.0,
            max_abs_base_score_term=self.max_abs_base_score_term,
            max_abs_pred_score_term=self.max_abs_pred_score_term,
            max_abs_oc_score_term=self.max_abs_oc_score_term,
            max_abs_phys_score_term=self.max_abs_phys_score_term,
            max_abs_pr_score_term=self.max_abs_pr_score_term,
        )
        return scores, components

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, x_slow, labels, original_labels, group_ids = self._unpack_batch(batch)
        with torch.no_grad():
            out = self.model(x, self._extract_condition(x), x_slow)
        self._check_model_outputs_finite(out, "Validation", batch_idx)
        scores, components = self._score_batch_with_components(x, out)
        self._check_raw_score_components(components, "Validation", batch_idx)
        scores = scores.cpu()
        non_finite = ~torch.isfinite(scores)
        if bool(non_finite.any()):
            n_bad = int(non_finite.sum().item())
            if not self._in_sanity_check():
                self.non_finite_events += 1
                self.stability_unstable = True
            self._non_finite_windows_epoch += n_bad
            if self._non_finite_warned_batches < self.non_finite_warn_limit_per_epoch:
                logger.warning(
                    "Validation batch {} contains {} non-finite scores; sanitizing with nan_to_num",
                    batch_idx,
                    n_bad,
                )
                self._non_finite_warned_batches += 1
            scores = torch.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=-1e6)
        self._val_outputs.append(
            {
                "scores": scores,
                "labels": labels.cpu(),
                "original_labels": original_labels.cpu(),
                "group_ids": group_ids,
            }
        )

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

        if self._non_finite_windows_epoch > 0 and self._non_finite_warned_batches >= self.non_finite_warn_limit_per_epoch:
            logger.warning(
                "Validation epoch {} had {} non-finite windows across many batches (log capped at {})",
                self.current_epoch,
                self._non_finite_windows_epoch,
                self.non_finite_warn_limit_per_epoch,
            )
        if self._component_warned_batches > 0:
            logger.warning(
                "Validation epoch {}: raw score component instability detected in {} batches "
                "(component-level warnings capped per batch)",
                self.current_epoch,
                self._component_warned_batches,
            )
        self._non_finite_warned_batches = 0
        self._non_finite_windows_epoch = 0
        self._component_warned_batches = 0

        finite_mask = np.isfinite(all_scores)
        n_non_finite = int((~finite_mask).sum())
        if n_non_finite > 0:
            if not self._in_sanity_check():
                self.non_finite_events += 1
                self.stability_unstable = True
            logger.warning(
                "Validation epoch {} has {} non-finite aggregated scores; sanitizing",
                self.current_epoch,
                n_non_finite,
            )
            all_scores = np.nan_to_num(all_scores, nan=1e6, posinf=1e6, neginf=-1e6)

        if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
            # Degenerate batch (sanity check or all-normal segment): log zeros with
            # identical kwargs to the normal path so Lightning doesn't reject a kwarg mismatch.
            self.log("val_macro_per_class_pr_auc", 0.0, prog_bar=True)
            self.log("val_worst_class_pr_auc", 0.0, prog_bar=True)
            self.log("val_pr_auc", 0.0)
            self.log("val_roc_auc", 0.0)
            self.log("val_f1_at_threshold", 0.0)
            self.log("val_precision_at_threshold", 0.0)
            self.log("val_recall_at_threshold", 0.0)
            self.log("val_accuracy_at_threshold", 0.0)
            self.log("val_threshold", 0.0)
            return

        val_pr_auc = float(average_precision_score(all_labels, all_scores))
        val_roc_auc = float(roc_auc_score(all_labels, all_scores))
        threshold, val_f1, val_prec, val_rec = _calibrate_threshold(all_scores, all_labels)
        val_preds = (all_scores >= threshold).astype(int)
        val_acc = float(accuracy_score(all_labels, val_preds))

        per_class_pr = compute_macro_per_class_pr_auc(
            labels=all_original_labels, scores=all_scores
        )
        val_macro_pc_pr_auc = float(per_class_pr.get("macro_per_class_pr_auc") or 0.0)
        val_worst_pc_pr_auc = float(per_class_pr.get("worst_class_pr_auc") or 0.0)
        per_class_detail: dict = per_class_pr.get("per_class_pr_auc_vs_normal", {})

        self.best_val_pr_auc = max(self.best_val_pr_auc, val_macro_pc_pr_auc)
        self.val_threshold = threshold
        self._val_scores_np = all_scores
        self._val_labels_np = all_labels
        self._val_original_labels_np = all_original_labels
        self._val_group_ids_np = all_group_ids

        self.log("val_macro_per_class_pr_auc", val_macro_pc_pr_auc, prog_bar=True)
        self.log("val_worst_class_pr_auc", val_worst_pc_pr_auc, prog_bar=True)
        self.log("val_pr_auc", val_pr_auc)
        self.log("val_roc_auc", val_roc_auc)
        self.log("val_f1_at_threshold", val_f1)
        self.log("val_precision_at_threshold", val_prec)
        self.log("val_recall_at_threshold", val_rec)
        self.log("val_accuracy_at_threshold", val_acc)
        self.log("val_threshold", threshold)

        # Per-class PR-AUC breakdown for visibility during training
        pc_parts = "  ".join(
            f"cls{k}={v:.3f}" for k, v in sorted(per_class_detail.items())
        ) if per_class_detail else "n/a"
        logger.info(
            "Epoch {:3d} | macro_pc={:.4f}  worst={:.4f}  binary={:.4f}  thr={:.4f}  "
            "score_p95={:.4f}  score_max={:.4f}  c_norm={:.4f} | {}",
            self.current_epoch,
            val_macro_pc_pr_auc, val_worst_pc_pr_auc, val_pr_auc, threshold,
            float(np.percentile(all_scores, 95)),
            float(np.max(all_scores)),
            float(self.c.norm()) if bool(self.c_initialized) else 0.0,
            pc_parts,
        )

        score_p95 = float(np.percentile(all_scores, 95))
        score_max = float(np.max(all_scores))
        c_norm = float(self.c.norm()) if bool(self.c_initialized) else 0.0
        c_bad = bool(self.c_initialized) and (not np.isfinite(c_norm) or c_norm > self.oc_center_max_norm)
        score_guard_active = bool(self.c_initialized) or (self.current_epoch >= self.oc_warmup_epochs)
        score_bad = score_guard_active and (
            (not np.isfinite(score_p95))
            or (not np.isfinite(score_max))
            or (score_p95 >= self.score_instability_p95)
            or (score_max >= self.score_instability_max)
        )
        if c_bad or score_bad:
            self.stability_unstable = True
            parts = []
            if c_bad:
                parts.append(f"c_norm={c_norm:.4g}")
            if score_bad:
                parts.append(f"score_p95={score_p95:.4g}")
                parts.append(f"score_max={score_max:.4g}")
            self.stability_reason = "|".join(parts)
            logger.warning(
                "Instability detected epoch {}: {}",
                self.current_epoch,
                "; ".join(parts),
            )
            if self.hpo_mode and getattr(self.trainer, "sanity_checking", False) is False:
                self.trainer.should_stop = True

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        x, x_slow, labels, original_labels, group_ids = self._unpack_batch(batch)
        with torch.no_grad():
            out = self.model(x, self._extract_condition(x), x_slow)
        self._check_model_outputs_finite(out, "Test", batch_idx)
        scores, components = self._score_batch_with_components(x, out)
        self._check_raw_score_components(components, "Test", batch_idx)
        scores = scores.cpu()
        non_finite = ~torch.isfinite(scores)
        if bool(non_finite.any()):
            n_bad = int(non_finite.sum().item())
            self._record_non_finite_event(
                "Test batch {} contains {} non-finite scores; sanitizing with nan_to_num",
                batch_idx,
                n_bad,
            )
            scores = torch.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=-1e6)
        self._test_outputs.append(
            {
                "scores": scores,
                "labels": labels.cpu(),
                "original_labels": original_labels.cpu(),
                "group_ids": group_ids,
            }
        )

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

    @torch.no_grad()
    def score_windows(
        self,
        x: torch.Tensor,
        x_slow: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score a batch of windows [B, W, F] → anomaly scores [B].

        When *return_diagnostics=True* returns ``(scores, raw_components)``
        where raw_components contains pre-sanitization score terms for each
        component (reconstruction, kl, physics, oc, prediction, pr).
        """
        out = self.model(x, self._extract_condition(x), x_slow)
        if not return_diagnostics:
            return self._score_batch(x, out)
        scores, components = self._score_batch_with_components(x, out)
        return scores, components

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        sch = _build_warmup_cosine_scheduler(
            opt,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr_ratio=self.min_lr_ratio,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}


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
    lit: DLSSMLightningModule,
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
    target_x_slow: torch.Tensor | None = None
    counter = 0
    for batch in test_dl:
        x_batch = batch[0]
        bs = x_batch.size(0)
        if counter + bs > target_idx:
            i = target_idx - counter
            target_window = x_batch[i : i + 1]
            if lit.slow_context_enabled and len(batch) >= 2 and isinstance(batch[1], torch.Tensor):
                target_x_slow = batch[1][i : i + 1]
            break
        counter += bs
    if target_window is None:
        logger.warning("Failed to locate target fault window — skipping plot")
        return

    device = next(lit.model.parameters()).device
    target_window = target_window.to(device)
    target_x_slow = target_x_slow.to(device) if target_x_slow is not None else None
    lit.model.eval()
    with torch.no_grad():
        c_target = lit._extract_condition(target_window)
        out = lit.model(target_window, c_target, target_x_slow)
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
    training_cfg: dict = config.get("training", {})
    threading_cfg: dict = config.get("training", {}).get("threading", {})
    seed: int = args.seed
    is_smoke: bool = args.smoke

    physics_enabled: bool = args.physics or bool(dlssm_cfg.get("physics_enabled", False))
    oc_enabled: bool = args.one_class or bool(dlssm_cfg.get("one_class", False))
    consistency_enabled: bool = args.consistency or bool(dlssm_cfg.get("consistency", False))
    cvae_enabled: bool = args.cvae or bool(dlssm_cfg.get("cvae", False))

    run_type = "smoke" if is_smoke else args.run_type

    # Derive a human-readable variant name for logging (may be overridden to DLSSM-FDD below)
    parts: list[str] = []
    if cvae_enabled:
        parts.append("C")
    if physics_enabled:
        parts.append("PI")
    if oc_enabled:
        parts.append("OC")
    if consistency_enabled:
        parts.append("Cons")
    variant = ("-".join(parts) + "-DLS-SSM") if parts else "DLS-SSM"

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
    batch_size = int(dlssm_cfg.get("batch_size", training_cfg.get("batch_size", 256)))
    max_epochs = int(dlssm_cfg.get("max_epochs", training_cfg.get("max_epochs", 100)))
    patience = int(dlssm_cfg.get("patience", training_cfg.get("patience", 15)))

    if is_smoke:
        train_stride = max(train_stride, win_size // 2)
        eval_stride = max(eval_stride, win_size // 2)
        max_epochs = 1
        patience = 1
        logger.info("Smoke mode: 1 epoch, large strides")

    loader_runtime = _resolve_loader_runtime(
        threading_cfg, hpo_mode=bool(getattr(args, "hpo", False))
    )
    logger.info(
        "DataLoader runtime | workers={} pin_memory={} persistent_workers={} prefetch_factor={}",
        loader_runtime["num_workers"],
        loader_runtime["pin_memory"],
        loader_runtime["persistent_workers"],
        loader_runtime["prefetch_factor"],
    )

    # Slow-context config (needed before _make_dataloaders; no CVAE dependency)
    fdd_cfg = dlssm_cfg.get("fdd", {}) or {}
    slow_context_enabled: bool = bool(fdd_cfg.get("slow_context_enabled", True))
    slow_context_seconds: int = int(fdd_cfg.get("slow_context_seconds", 3600))
    slow_stride: int = int(fdd_cfg.get("slow_stride", 60))
    slow_hidden_dim: int = int(fdd_cfg.get("slow_hidden_dim", 64))
    n_attention_heads: int = int(fdd_cfg.get("n_attention_heads", 4))

    if slow_context_enabled:
        try:
            from mamba_ssm import Mamba as _Mamba  # noqa: F401
        except ImportError:
            logger.warning(
                "mamba_ssm not available — slow-context branch disabled. "
                "Run on Colab A100 with `pip install mamba-ssm` for full DLSSM-FDD."
            )
            slow_context_enabled = False
            slow_hidden_dim = 0

    def _make_dataloaders(stride_train: int, stride_eval: int, bs: int):
        slow_kw = dict(
            return_slow_context=slow_context_enabled,
            slow_context_samples=slow_context_seconds,
            slow_stride=slow_stride,
        )
        ds_train = TimeSeriesDataset(
            train_df, features, label_col, win_size, stride_train, normal_only=True, **slow_kw
        )
        ds_val = TimeSeriesDataset(
            val_df,
            features,
            label_col,
            win_size,
            stride_eval,
            normal_only=False,
            return_original_label=True,
            return_group_id=True,
            **slow_kw,
        )
        ds_test = TimeSeriesDataset(
            test_df,
            features,
            label_col,
            win_size,
            stride_eval,
            normal_only=False,
            return_original_label=True,
            return_group_id=True,
            **slow_kw,
        )
        kw = {
            "drop_last": False,
            "num_workers": loader_runtime["num_workers"],
            "pin_memory": loader_runtime["pin_memory"],
            "persistent_workers": loader_runtime["persistent_workers"],
        }
        if loader_runtime["num_workers"] > 0:
            kw["prefetch_factor"] = loader_runtime["prefetch_factor"]
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

    # CVAE conditioning: resolve indices from feature names (used by HPO + final build)
    cond_cfg = dlssm_cfg.get("conditioning", {}) or {}
    cond_feats = [f for f in cond_cfg.get("features", []) if f in feature_idx]
    missing_cond = [f for f in cond_cfg.get("features", []) if f not in feature_idx]
    if cvae_enabled and missing_cond:
        logger.warning("CVAE: missing condition features in manifest: {}", missing_cond)
    if cvae_enabled and not cond_feats:
        logger.warning("CVAE requested but no condition features available — disabling CVAE")
        cvae_enabled = False
    condition_idx = [feature_idx[f] for f in cond_feats] if cvae_enabled else []
    condition_dim = len(condition_idx)

    skip_feats = [f for f in cond_cfg.get("recon_skip_features", []) if f in feature_idx]
    skip_idx = [feature_idx[f] for f in skip_feats] if cvae_enabled else []
    if cvae_enabled and skip_idx:
        recon_mask = torch.ones(n_features, dtype=torch.float32)
        recon_mask[skip_idx] = 0.0
        logger.info(
            "CVAE: condition_features={} | recon_skip_features={} ({} kept of {})",
            cond_feats, skip_feats, n_features - len(skip_idx), n_features,
        )
    else:
        recon_mask = None
        if cvae_enabled:
            logger.info("CVAE: condition_features={} (no recon mask)", cond_feats)

    # DLSSM-FDD: expected-power configuration (needs condition_dim from CVAE block above)
    expected_power_enabled: bool = bool(fdd_cfg.get("expected_power_enabled", True))
    expected_power_features: list[str] = fdd_cfg.get("expected_power_features", ["pdc1", "pdc2"])
    expected_power_indices: list[int] = [
        feature_idx[f] for f in expected_power_features if f in feature_idx
    ]
    if expected_power_enabled and not expected_power_indices:
        logger.warning(
            "DLSSM-FDD: expected_power_enabled=True but none of {} found in features — disabling",
            expected_power_features,
        )
        expected_power_enabled = False
    if expected_power_enabled and condition_dim == 0:
        logger.warning(
            "DLSSM-FDD: expected_power_enabled=True requires CVAE (condition_dim>0) — disabling"
        )
        expected_power_enabled = False

    lambda_expected: float = float(fdd_cfg.get("lambda_expected", 0.5))
    lambda_pr_score: float = float(fdd_cfg.get("lambda_pr_score", 0.1))

    stability_cfg = dlssm_cfg.get("stability", {}) or {}
    oc_center_max_norm: float = float(stability_cfg.get("oc_center_max_norm", 50.0))
    score_instability_p95: float = float(stability_cfg.get("score_instability_p95", 1e6))
    score_instability_max: float = float(stability_cfg.get("score_instability_max", 1e7))
    non_finite_warn_limit_per_epoch: int = int(
        stability_cfg.get("non_finite_warn_limit_per_epoch", 5)
    )
    _raw_oc = stability_cfg.get("max_abs_oc_score_term")
    max_abs_oc_score_term: float | None = float(_raw_oc) if _raw_oc is not None else 1e5
    _raw_phys = stability_cfg.get("max_abs_phys_score_term")
    max_abs_phys_score_term: float | None = float(_raw_phys) if _raw_phys is not None else 1e5
    _raw_pr = stability_cfg.get("max_abs_pr_score_term")
    max_abs_pr_score_term: float | None = float(_raw_pr) if _raw_pr is not None else 1e5
    _raw_base = stability_cfg.get("max_abs_base_score_term")
    max_abs_base_score_term: float | None = float(_raw_base) if _raw_base is not None else 1e5
    _raw_pred = stability_cfg.get("max_abs_pred_score_term")
    max_abs_pred_score_term: float | None = float(_raw_pred) if _raw_pred is not None else 1e5

    # Override variant to DLSSM-FDD when all stacked components are active
    if slow_context_enabled and expected_power_enabled and cvae_enabled:
        variant = "DLSSM-FDD"

    # Apply best params from --best-params JSON or from HPO sweep
    hpo_params: dict = {}
    _hpo_n_completed: int | None = None
    if args.best_params:
        hpo_params = json.loads(args.best_params)
        logger.info("Applying best_params from CLI: {}", hpo_params)
    elif getattr(args, "hpo", False):
        hpo_cfg = dlssm_cfg.get("hpo", {})
        logger.info("Starting HPO sweep ({} trials)...", args.n_trials or hpo_cfg.get("n_trials", 40))
        hpo_params, _hpo_n_completed = _run_hpo(
            base_cfg=dlssm_cfg,
            hpo_cfg=hpo_cfg,
            training_cfg=training_cfg,
            cvae_enabled=cvae_enabled,
            condition_idx=condition_idx,
            condition_dim=condition_dim,
            recon_mask=recon_mask,
            train_dl=train_dl,
            val_dl=val_dl,
            scaler_mean_t=scaler_mean_t,
            scaler_scale_t=scaler_scale_t,
            feature_idx=feature_idx,
            n_features=n_features,
            physics_enabled=physics_enabled,
            oc_enabled=oc_enabled,
            consistency_enabled=consistency_enabled,
            seed=seed,
            n_trials_override=args.n_trials,
            slow_context_enabled=slow_context_enabled,
            slow_hidden_dim=slow_hidden_dim,
            n_attention_heads=n_attention_heads,
            expected_power_enabled=expected_power_enabled,
            expected_power_indices=expected_power_indices,
            lambda_expected=lambda_expected,
            lambda_pr_score=lambda_pr_score,
            oc_center_max_norm=oc_center_max_norm,
            score_instability_p95=score_instability_p95,
            score_instability_max=score_instability_max,
            non_finite_warn_limit_per_epoch=non_finite_warn_limit_per_epoch,
            max_abs_oc_score_term=max_abs_oc_score_term,
            max_abs_phys_score_term=max_abs_phys_score_term,
            max_abs_pr_score_term=max_abs_pr_score_term,
            max_abs_base_score_term=max_abs_base_score_term,
            max_abs_pred_score_term=max_abs_pred_score_term,
        )
        logger.info("HPO best params: {}", json.dumps(hpo_params, default=str))

    # Merge hpo_params into dlssm_cfg (non-destructive — only affects this run)
    if hpo_params:
        dlssm_cfg = {**dlssm_cfg, **hpo_params}

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
        condition_dim=condition_dim,
        n_attention_heads=n_attention_heads,
        slow_hidden_dim=slow_hidden_dim if slow_context_enabled else 0,
        expected_power_indices=expected_power_indices if expected_power_enabled else None,
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
        total_steps=max(1, len(train_dl) * max_epochs),
        warmup_steps=int(training_cfg.get("warmup_steps", 0)),
        min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.01)),
        beta_kl=float(dlssm_cfg.get("beta_kl", 0.1)),
        kl_warmup_epochs=int(dlssm_cfg.get("kl_warmup_epochs", 5)),
        lambda_phys=float(dlssm_cfg.get("lambda_phys", 0.05)),
        enable_string_power=bool(physics_components.get("string_power", True)),
        enable_imbalance=bool(physics_components.get("imbalance", True)),
        physics_enabled=physics_enabled,
        lambda_kl_score=float(dlssm_cfg.get("lambda_kl_score", 0.1)),
        lambda_phys_score=float(dlssm_cfg.get("lambda_phys_score", 0.0)),
        alpha_pred=float(dlssm_cfg.get("alpha_pred", 0.1)),
        lambda_pred_score=float(dlssm_cfg.get("lambda_pred_score", 0.0)),
        score_reduction=str(dlssm_cfg.get("score_reduction", "center")),
        free_bits=float(dlssm_cfg.get("free_bits", 0.0)),
        oc_enabled=oc_enabled,
        consistency_enabled=consistency_enabled,
        lambda_oc=float(dlssm_cfg.get("lambda_oc", 0.1)),
        lambda_oc_score=float(dlssm_cfg.get("lambda_oc_score", 0.0)),
        oc_warmup_epochs=int(dlssm_cfg.get("oc_warmup_epochs", 3)),
        lambda_cons=float(dlssm_cfg.get("lambda_cons", 0.1)),
        aug_noise_std=float(dlssm_cfg.get("aug_noise_std", 0.05)),
        aug_feature_dropout=float(dlssm_cfg.get("aug_feature_dropout", 0.1)),
        latent_dim=int(dlssm_cfg.get("latent_dim", 16)),
        cvae_enabled=cvae_enabled,
        condition_idx=condition_idx,
        recon_mask=recon_mask,
        slow_context_enabled=slow_context_enabled,
        expected_power_enabled=expected_power_enabled,
        expected_power_indices=expected_power_indices,
        lambda_expected=lambda_expected,
        lambda_pr_score=lambda_pr_score,
        oc_center_max_norm=oc_center_max_norm,
        score_instability_p95=score_instability_p95,
        score_instability_max=score_instability_max,
        non_finite_warn_limit_per_epoch=non_finite_warn_limit_per_epoch,
        max_abs_oc_score_term=max_abs_oc_score_term,
        max_abs_phys_score_term=max_abs_phys_score_term,
        max_abs_pr_score_term=max_abs_pr_score_term,
        max_abs_base_score_term=max_abs_base_score_term,
        max_abs_pred_score_term=max_abs_pred_score_term,
        hpo_mode=False,
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
        EarlyStopping(monitor="val_macro_per_class_pr_auc", patience=patience, mode="max", verbose=False),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor="val_macro_per_class_pr_auc",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
    ]

    trainer_kwargs: dict = dict(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=True,
        logger=False,
        deterministic=False,
        gradient_clip_val=float(dlssm_cfg.get("gradient_clip_val", 1.0)),
        **_trainer_runtime_kwargs(training_cfg, dlssm_cfg),
    )
    trainer = pl.Trainer(**trainer_kwargs)

    logger.info(
        "Training {} (cvae={}, physics={}, one_class={}, consistency={})",
        variant, cvae_enabled, physics_enabled, oc_enabled, consistency_enabled,
    )
    t0 = time.perf_counter()
    trainer.fit(lit, train_dataloaders=train_dl, val_dataloaders=val_dl)
    fit_time = time.perf_counter() - t0
    logger.info("Training done in {:.1f}s", fit_time)

    # Find best checkpoint
    ckpt_cb = next((c for c in callbacks if isinstance(c, ModelCheckpoint)), None)
    best_ckpt_path: str | None = ckpt_cb.best_model_path if (ckpt_cb and ckpt_cb.best_model_path) else None

    if lit.stability_unstable or lit.non_finite_events > 0:
        reason = lit.stability_reason or "unknown"
        if not best_ckpt_path:
            raise RuntimeError(
                "Training aborted due to numerical instability and no best checkpoint available. "
                f"Reason: {reason} "
                f"(stability_unstable={lit.stability_unstable}, non_finite_events={lit.non_finite_events})."
            )
        logger.warning(
            "Numerical instability detected (reason={}, stability_unstable={}, non_finite_events={}) — "
            "validating best checkpoint before continuing",
            reason, lit.stability_unstable, lit.non_finite_events,
        )
        lit._reset_stability_state()

    # Re-validate with best checkpoint
    if best_ckpt_path:
        logger.info("Re-validating with best checkpoint: {}", best_ckpt_path)
        pl.Trainer(enable_progress_bar=False, enable_model_summary=False, logger=False).validate(
            lit, dataloaders=val_dl, ckpt_path=best_ckpt_path
        )
        if lit.stability_unstable or lit.non_finite_events > 0:
            reason = lit.stability_reason or "unknown"
            raise RuntimeError(
                "Best checkpoint failed stability validation. "
                f"Reason: {reason} "
                f"(stability_unstable={lit.stability_unstable}, non_finite_events={lit.non_finite_events})."
            )
    else:
        logger.warning("No checkpoint found — val scores from last epoch.")

    # Test
    pl.Trainer(enable_progress_bar=False, enable_model_summary=False, logger=False).test(
        lit, dataloaders=test_dl, ckpt_path=best_ckpt_path
    )
    if lit.stability_unstable or lit.non_finite_events > 0:
        reason = lit.stability_reason or "unknown"
        raise RuntimeError(
            "Best checkpoint failed stability test. "
            f"Reason: {reason} "
            f"(stability_unstable={lit.stability_unstable}, non_finite_events={lit.non_finite_events})."
        )

    val_scores = lit._val_scores_np
    val_labels = lit._val_labels_np
    val_original_labels = lit._val_original_labels_np
    val_group_ids = lit._val_group_ids_np
    test_scores = lit._test_scores_np
    test_labels = lit._test_labels_np
    test_original_labels = lit._test_original_labels_np
    test_group_ids = lit._test_group_ids_np

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
    val_episode_macro_f1 = episode_macro_f1_binary(val_labels, val_preds, val_group_ids)
    test_episode_macro_f1 = episode_macro_f1_binary(test_labels, test_preds, test_group_ids)

    val_per_class_source = val_original_labels if val_original_labels is not None else val_labels
    test_per_class_source = test_original_labels if test_original_labels is not None else test_labels
    val_macro = compute_macro_per_class_pr_auc(labels=val_per_class_source, scores=val_scores)
    test_macro = compute_macro_per_class_pr_auc(labels=test_per_class_source, scores=test_scores)

    logger.info(
        "Test — PR-AUC={:.4f}  ROC-AUC={:.4f}  F1={:.4f}  Prec={:.4f}  Rec={:.4f}",
        test_pr_auc, test_roc_auc, test_f1, test_prec, test_rec,
    )

    score_components = ["reconstruction", "kl", "prediction"]
    if physics_enabled:
        score_components.append("physics")
    if oc_enabled:
        score_components.append("one_class")
    if expected_power_enabled:
        score_components.append("expected_power_pr")

    metrics: dict = {
        "score_name": "dlssm_anomaly_score",
        "score_components": score_components,
        "score_reduction": str(dlssm_cfg.get("score_reduction", "center")),
        "threshold_policy": "q95_normal_val",
        "selection_metric": "val_macro_per_class_pr_auc",
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "val_f1_at_threshold": val_f1,
        "val_accuracy_at_threshold": val_acc,
        "val_precision_at_threshold": val_prec,
        "val_recall_at_threshold": val_rec,
        "val_episode_macro_f1": float(val_episode_macro_f1),
        "val_macro_per_class_pr_auc": val_macro.get("macro_per_class_pr_auc"),
        "val_worst_class_pr_auc": val_macro.get("worst_class_pr_auc"),
        "val_selection_score": val_macro.get("macro_per_class_pr_auc"),
        "threshold": threshold,
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
        "n_train_windows": len(train_dl.dataset),
        "n_features": n_features,
        "fit_time_s": round(fit_time, 2),
    }
    run_name = f"anomaly_dlssm_{variant.lower().replace('-', '_')}_{ts}"

    # Save artifacts
    run_params = {
        "variant": variant,
        "physics_enabled": physics_enabled,
        "oc_enabled": oc_enabled,
        "consistency_enabled": consistency_enabled,
        "cvae_enabled": cvae_enabled,
        "condition_features": cond_feats if cvae_enabled else [],
        "recon_skip_features": skip_feats if cvae_enabled else [],
        **{k: v for k, v in dlssm_cfg.items() if not isinstance(v, dict)},
        "max_epochs_actual": max_epochs,
        "n_features": n_features,
        "seed": seed,
        "c_norm": float(lit.c.norm()) if bool(lit.c_initialized) else 0.0,
    }

    global_metrics_path = artifacts_dir / "global_metrics.json"
    per_class_metrics_path = artifacts_dir / "per_class_metrics.json"
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
    pred_residual_path = artifacts_dir / "prediction_residual_decomposition.png"

    write_json(global_metrics_path, metrics)
    params_path.write_text(json.dumps(run_params, indent=2, default=str), encoding="utf-8")
    if hpo_params:
        hpo_params_path.write_text(json.dumps(hpo_params, indent=2, default=str), encoding="utf-8")
        logger.info("Best HPO params saved → {}  (pass with --best-params for multi-seed)", hpo_params_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    joblib.dump(scaler, scaler_path)
    per_class_metrics = compute_anomaly_per_class_metrics(
        labels=test_per_class_source,
        scores=test_scores,
        threshold=threshold,
        val_labels=val_per_class_source,
        val_scores=val_scores,
    )
    candidate_thresholds = build_candidate_per_true_class_thresholds(per_class_metrics, normal_label=0)
    write_json(
        calibration_path,
        build_score_calibration_payload(
            threshold=threshold,
            threshold_policy="q95_normal_val",
            threshold_quantile=0.95,
            candidate_per_true_class_thresholds=candidate_thresholds,
            score_stats={
                "score_name": "dlssm_anomaly_score",
                "score_reduction": str(dlssm_cfg.get("score_reduction", "center")),
                "lambda_kl_score": float(dlssm_cfg.get("lambda_kl_score", 0.1)),
                "lambda_pred_score": float(dlssm_cfg.get("lambda_pred_score", 0.0)),
                "lambda_phys_score": float(dlssm_cfg.get("lambda_phys_score", 0.0)),
                "lambda_oc_score": float(dlssm_cfg.get("lambda_oc_score", 0.0)),
                "lambda_pr_score": lambda_pr_score,
                "score_components": score_components,
            },
        ),
    )
    run_manifest = build_run_manifest(
        task=args.task,
        model="dlssm",
        model_family="anomaly_dl",
        dataset=args.dataset,
        split_path=args.split_path,
        feature_profile=str(args.profile),
        feature_run_dir=str(resolved_run_dir),
        seed=seed,
        run_type=run_type,
        extras={"run_name": run_name, "variant": variant},
    )
    model_artifact_rel = "checkpoints/best.ckpt" if best_ckpt_path else "checkpoints/last.ckpt"
    deployment_manifest = build_deployment_manifest(
        task=args.task,
        model="dlssm",
        model_family="anomaly_dl",
        model_artifact=model_artifact_rel,
        scaler_artifact=scaler_path.name,
        feature_names=features,
        label_column=label_col,
        threshold=threshold,
        window_size=win_size,
        score_direction="higher_is_more_anomalous",
        classes=[str(c) for c in sorted(np.unique(test_per_class_source).tolist())],
        extras={
            "checkpoint_available": bool(best_ckpt_path),
            "threshold_policy": "q95_normal_val",
            "selection_metric": "val_macro_per_class_pr_auc",
            "score_name": "dlssm_anomaly_score",
            "score_components": score_components,
            "score_reduction": str(dlssm_cfg.get("score_reduction", "center")),
            "score_calibration_artifact": calibration_path.name,
        },
    )
    write_json(per_class_metrics_path, per_class_metrics)
    write_json(run_manifest_path, run_manifest)
    write_json(deployment_manifest_path, deployment_manifest)

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
                "one_class": str(oc_enabled),
                "consistency": str(consistency_enabled),
                "cvae": str(cvae_enabled),
            })
            mlflow.log_params({
                k: v for k, v in run_params.items()
                if not isinstance(v, (dict, list))
            })
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                }
            )
            for cls_str, m in per_class_metrics.items():
                if m.get("pr_auc_vs_normal") is not None:
                    mlflow.log_metric(f"test_pr_auc_class{cls_str}_vs_normal", m["pr_auc_vs_normal"])
            for p in (
                global_metrics_path,
                per_class_metrics_path,
                run_manifest_path,
                deployment_manifest_path,
                params_path,
                hpo_params_path,
                scaler_path,
                manifest_path,
                calibration_path,
                pr_curve_path,
                histogram_path,
                timeline_path,
                pred_residual_path,
            ):
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
                "oc_enabled": oc_enabled,
                "consistency_enabled": consistency_enabled,
                "cvae_enabled": cvae_enabled,
                "condition_features": ",".join(cond_feats) if cvae_enabled else "",
                "recon_skip_features": ",".join(skip_feats) if cvae_enabled else "",
                "lambda_oc": float(dlssm_cfg.get("lambda_oc", 0.1)),
                "lambda_cons": float(dlssm_cfg.get("lambda_cons", 0.1)),
                "c_norm": float(lit.c.norm()) if bool(lit.c_initialized) else 0.0,
                "n_features": n_features,
                "win_size": win_size,
                "hidden_dim": int(dlssm_cfg.get("hidden_dim", 64)),
                "latent_dim": int(dlssm_cfg.get("latent_dim", 16)),
                "lambda_phys": float(dlssm_cfg.get("lambda_phys", 0.05)),
                "threshold": threshold,
                "threshold_policy": "q95_normal_val",
                "selection_metric": "val_macro_per_class_pr_auc",
                "val_selection_score": val_macro.get("macro_per_class_pr_auc"),
                "score_name": "dlssm_anomaly_score",
                "score_components": score_components,
                "score_reduction": str(dlssm_cfg.get("score_reduction", "center")),
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "val_f1_at_threshold": val_f1,
                "val_accuracy_at_threshold": val_acc,
                "val_episode_macro_f1": float(val_episode_macro_f1),
                "val_macro_per_class_pr_auc": val_macro.get("macro_per_class_pr_auc"),
                "val_worst_class_pr_auc": val_macro.get("worst_class_pr_auc"),
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
                "fit_time_s": round(fit_time, 2),
                "hpo_enabled": bool(getattr(args, "hpo", False)),
                "best_params_injected": bool(args.best_params),
                "hpo_n_trials_requested": (args.n_trials or dlssm_cfg.get("hpo", {}).get("n_trials")) if getattr(args, "hpo", False) else None,
                "hpo_n_trials_completed": _hpo_n_completed,
                "hpo_best_params": json.dumps(hpo_params, default=str) if hpo_params else None,
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
