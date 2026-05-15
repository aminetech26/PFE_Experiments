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
    compute_association_discrepancy,
    compute_maat_scores,
)
from src.modeling.anomaly_detection.dl.maat.model import MambaAnomalyTransformer
from src.modeling.common.artifact_contract import (
    build_deployment_manifest,
    build_run_manifest,
    build_score_calibration_payload,
    compute_anomaly_per_class_metrics,
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
    HPOStageConfig,
    run_staged_optuna,
    suggest_params_from_space,
)
from src.utils.paths import get_experiments_root

# trainer.py lives under dl/maat/, so PROJECT_ROOT is 5 levels up
PROJECT_ROOT = Path(__file__).resolve().parents[5]

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_float32_matmul_precision("high")  # TF32 matmul on Ampere; fp32 accumulators preserved
torch.backends.cudnn.benchmark = True       # fixed input shapes → cuDNN autotunes once and caches


def _hpo_config_fingerprint(maat_cfg: dict, hpo_cfg: dict, seed: int) -> str:
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
    """Return (threshold, best_f1, precision, recall) by maximising F1 on the PR curve.

    Uses 5-point rolling-mean smoothing when the positive count is ≥ 50 to reduce
    argmax sensitivity on jagged PR curves from sparse-positive validation sets.
    """
    prec, rec, thresholds = precision_recall_curve(labels, scores)
    denom = prec[:-1] + rec[:-1]
    # Evaluate division only where denom > 0 to silence RuntimeWarning.
    # np.where evaluates both branches eagerly, so guard the denominator explicitly.
    safe_denom = np.where(denom > 0, denom, 1.0)
    f1_vals = np.where(denom > 0, 2 * prec[:-1] * rec[:-1] / safe_denom, 0.0)

    n_positives = int(labels.sum())
    if n_positives >= 50 and len(f1_vals) >= 7:
        kernel = np.ones(5) / 5.0
        f1_smooth = np.convolve(f1_vals, kernel, mode="same")
        # Restore raw values at edges — convolve pads with zeros there, biasing argmax.
        f1_smooth[:2] = f1_vals[:2]
        f1_smooth[-2:] = f1_vals[-2:]
        best_idx = int(np.argmax(f1_smooth))
    else:
        best_idx = int(np.argmax(f1_vals))

    return (
        float(thresholds[best_idx]),
        float(f1_vals[best_idx]),
        float(prec[best_idx]),
        float(rec[best_idx]),
    )


def _safe_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0
    if not np.all(np.isfinite(scores)):
        return 0.0
    return float(average_precision_score(labels, scores))


def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.0
    if not np.all(np.isfinite(scores)):
        return 0.0
    return float(roc_auc_score(labels, scores))


def _score_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    return {
        "pr_auc": _safe_pr_auc(labels, scores),
        "roc_auc": _safe_roc_auc(labels, scores),
        "f1_at_threshold": float(f1_score(labels, preds, zero_division=0)),
        "accuracy_at_threshold": float(accuracy_score(labels, preds)),
        "precision_at_threshold": float(precision_score(labels, preds, zero_division=0)),
        "recall_at_threshold": float(recall_score(labels, preds, zero_division=0)),
        "threshold": float(threshold),
    }


def _episode_macro_f1_binary(
    labels: np.ndarray,
    preds: np.ndarray,
    group_ids: np.ndarray | None,
) -> float:
    result = episode_macro_f1_binary(labels, preds, group_ids)
    if group_ids is None or len(group_ids) != len(labels):
        logger.warning("episode_macro_f1 unavailable: missing or misaligned group_ids")
    return result


def _zscore_from_val(val_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(val_scores))
    std = float(np.std(val_scores))
    if std <= 1e-12 or not math.isfinite(std):
        std = 1.0
    return (val_scores - mean) / std, (test_scores - mean) / std


def _zscore_with_stats(scores: np.ndarray, mean: float, std: float) -> np.ndarray:
    safe_std = std if std > 1e-12 and math.isfinite(std) else 1.0
    return (scores - mean) / safe_std


def _fit_normal_stats(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    normal_scores = scores[labels == 0]
    if len(normal_scores) == 0:
        normal_scores = scores
    mean = float(np.mean(normal_scores))
    std = float(np.std(normal_scores))
    if not math.isfinite(std) or std <= 1e-12:
        std = 1.0
    return mean, std


def _logsumexp_multi(arrays: list[np.ndarray], tau: float = 1.0) -> np.ndarray:
    stacked = np.stack([a / tau for a in arrays], axis=0)
    max_vals = np.max(stacked, axis=0)
    return tau * (max_vals + np.log(np.exp(stacked - max_vals).sum(axis=0)))


def _macro_fault_pr_auc(
    original_labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    values: list[float] = []
    for fault_label in sorted(float(x) for x in np.unique(original_labels) if float(x) != 0.0):
        mask = (original_labels == fault_label) | (original_labels == 0)
        labels = (original_labels[mask] == fault_label).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        values.append(_safe_pr_auc(labels, scores[mask]))
    return float(np.mean(values)) if values else 0.0


def _logsumexp_pair(a: np.ndarray, b: np.ndarray, tau: float) -> np.ndarray:
    stacked = np.stack([a / tau, b / tau], axis=0)
    max_vals = np.max(stacked, axis=0)
    return tau * (max_vals + np.log(np.exp(stacked - max_vals).sum(axis=0)))


def _is_finite_array(arr: np.ndarray | None) -> bool:
    return arr is not None and np.all(np.isfinite(arr))


def _add_score_entry(
    report: dict,
    name: str,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    val_original_labels: np.ndarray,
    test_original_labels: np.ndarray,
) -> None:
    threshold, _, _, _ = _calibrate_threshold(val_scores, val_labels)
    report["scores"][name] = {
        "threshold_policy": "diagnostic_pr_curve_f1",
        "val_macro_fault_pr_auc": _macro_fault_pr_auc(val_original_labels, val_scores),
        "test_macro_fault_pr_auc": _macro_fault_pr_auc(test_original_labels, test_scores),
        "val": _score_metrics(val_scores, val_labels, threshold),
        "test": _score_metrics(test_scores, test_labels, threshold),
    }


def _build_score_decomposition_report(
    val_parts: dict[str, np.ndarray],
    test_parts: dict[str, np.ndarray],
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    val_original_labels: np.ndarray,
    test_original_labels: np.ndarray,
) -> tuple[dict, dict | None]:
    _best_tracker: dict = {}
    report: dict = {
        "version": 2,
        "naming": {
            "channels": {
                "product": "MAAT product score",
                "reconstruction": "MAAT reconstruction score",
                "association_discrepancy": "MAAT association discrepancy score",
                "association_affinity": "MAAT association affinity score",
                "dc_voltage_drop": "Physics DC voltage-drop score (vdc1/vdc2)",
                "mismatch_imbalance": "Physics mismatch/imbalance score",
                "drift_efficiency": "Drift/efficiency score",
            },
            "fusions": {
                "max_z_*": "Element-wise max of z-scored components",
                "logsumexp_z_*": "Smooth max (logsumexp) over z-scored components",
            },
        },
        "scores": {},
    }

    def _track_add(name: str, val_s: np.ndarray, test_s: np.ndarray) -> None:
        _add_score_entry(
            report, name, val_s, test_s,
            val_labels, test_labels, val_original_labels, test_original_labels,
        )
        entry_f1 = report["scores"][name]["val"]["f1_at_threshold"]
        entry_thr = report["scores"][name]["val"]["threshold"]
        if entry_f1 > _best_tracker.get("val_f1", -1.0):
            _best_tracker.update({
                "name": name,
                "val_scores": val_s,
                "test_scores": test_s,
                "threshold": entry_thr,
                "val_f1": entry_f1,
            })

    for name in ("product", "reconstruction", "association_discrepancy", "association_affinity"):
        _track_add(name, val_parts[name], test_parts[name])

    val_recon_z, test_recon_z = _zscore_from_val(
        val_parts["reconstruction"], test_parts["reconstruction"]
    )
    val_product_z, test_product_z = _zscore_from_val(val_parts["product"], test_parts["product"])
    val_clean_max = np.maximum(val_recon_z, val_product_z)
    test_clean_max = np.maximum(test_recon_z, test_product_z)
    _track_add("max_z_recon_product", val_clean_max, test_clean_max)
    for tau in (0.5, 1.0):
        _track_add(
            f"logsumexp_z_recon_product_tau_{tau:g}",
            _logsumexp_pair(val_recon_z, val_product_z, tau=tau),
            _logsumexp_pair(test_recon_z, test_product_z, tau=tau),
        )

    if "drift_efficiency" in val_parts and "drift_efficiency" in test_parts:
        _track_add("drift_efficiency", val_parts["drift_efficiency"], test_parts["drift_efficiency"])
        val_drift_z, test_drift_z = _zscore_from_val(
            val_parts["drift_efficiency"], test_parts["drift_efficiency"]
        )
        _track_add(
            "max_z_product_drift",
            np.maximum(val_product_z, val_drift_z),
            np.maximum(test_product_z, test_drift_z),
        )
        _track_add(
            "max_z_recon_product_drift",
            np.maximum(np.maximum(val_recon_z, val_product_z), val_drift_z),
            np.maximum(np.maximum(test_recon_z, test_product_z), test_drift_z),
        )
        for tau in (0.5, 1.0):
            _track_add(
                f"logsumexp_z_product_drift_tau_{tau:g}",
                _logsumexp_pair(val_product_z, val_drift_z, tau=tau),
                _logsumexp_pair(test_product_z, test_drift_z, tau=tau),
            )

    if "dc_voltage_drop" in val_parts and "dc_voltage_drop" in test_parts:
        _track_add("dc_voltage_drop", val_parts["dc_voltage_drop"], test_parts["dc_voltage_drop"])
        val_vdrop_z, test_vdrop_z = _zscore_from_val(
            val_parts["dc_voltage_drop"], test_parts["dc_voltage_drop"]
        )
        _track_add(
            "max_z_product_voltage",
            np.maximum(val_product_z, val_vdrop_z),
            np.maximum(test_product_z, test_vdrop_z),
        )
        _track_add(
            "max_z_recon_product_voltage",
            np.maximum(np.maximum(val_recon_z, val_product_z), val_vdrop_z),
            np.maximum(np.maximum(test_recon_z, test_product_z), test_vdrop_z),
        )
        for tau in (0.5, 1.0):
            _track_add(
                f"logsumexp_z_product_voltage_tau_{tau:g}",
                _logsumexp_pair(val_product_z, val_vdrop_z, tau=tau),
                _logsumexp_pair(test_product_z, test_vdrop_z, tau=tau),
            )

    if (
        "mismatch_imbalance" in val_parts
        and "mismatch_imbalance" in test_parts
        and "dc_voltage_drop" in val_parts
        and "dc_voltage_drop" in test_parts
    ):
        _track_add(
            "mismatch_imbalance",
            val_parts["mismatch_imbalance"],
            test_parts["mismatch_imbalance"],
        )
        val_mismatch_z, test_mismatch_z = _zscore_from_val(
            val_parts["mismatch_imbalance"], test_parts["mismatch_imbalance"]
        )
        _track_add(
            "max_z_recon_product_dc_voltage_mismatch",
            np.maximum(np.maximum(np.maximum(val_recon_z, val_product_z), val_vdrop_z), val_mismatch_z),
            np.maximum(np.maximum(np.maximum(test_recon_z, test_product_z), test_vdrop_z), test_mismatch_z),
        )
        for tau in (0.5, 1.0):
            val_dvm = _logsumexp_pair(
                _logsumexp_pair(val_recon_z, val_product_z, tau=tau),
                _logsumexp_pair(val_vdrop_z, val_mismatch_z, tau=tau),
                tau=tau,
            )
            test_dvm = _logsumexp_pair(
                _logsumexp_pair(test_recon_z, test_product_z, tau=tau),
                _logsumexp_pair(test_vdrop_z, test_mismatch_z, tau=tau),
                tau=tau,
            )
            _track_add(f"logsumexp_z_recon_product_dc_voltage_mismatch_tau_{tau:g}", val_dvm, test_dvm)

    by_fault: dict[str, dict] = {}
    for fault_label in sorted(float(x) for x in np.unique(test_original_labels) if float(x) != 0.0):
        mask = (test_original_labels == fault_label) | (test_original_labels == 0)
        labels = (test_original_labels[mask] == fault_label).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        by_fault[f"{fault_label:g}"] = {
            "n_windows": int(mask.sum()),
            "n_fault_windows": int(labels.sum()),
            "product_pr_auc": _safe_pr_auc(labels, test_parts["product"][mask]),
            "reconstruction_pr_auc": _safe_pr_auc(labels, test_parts["reconstruction"][mask]),
            "association_discrepancy_pr_auc": _safe_pr_auc(
                labels, test_parts["association_discrepancy"][mask]
            ),
            "association_affinity_pr_auc": _safe_pr_auc(
                labels, test_parts["association_affinity"][mask]
            ),
            "max_z_recon_product_pr_auc": _safe_pr_auc(labels, test_clean_max[mask]),
        }
        if "drift_efficiency" in test_parts:
            by_fault[f"{fault_label:g}"]["drift_efficiency_pr_auc"] = _safe_pr_auc(
                labels, test_parts["drift_efficiency"][mask]
            )
        if "dc_voltage_drop" in test_parts:
            by_fault[f"{fault_label:g}"]["dc_voltage_drop_pr_auc"] = _safe_pr_auc(
                labels, test_parts["dc_voltage_drop"][mask]
            )
        if "mismatch_imbalance" in test_parts:
            by_fault[f"{fault_label:g}"]["mismatch_imbalance_pr_auc"] = _safe_pr_auc(
                labels, test_parts["mismatch_imbalance"][mask]
            )
    report["test_by_fault_vs_normal"] = by_fault
    return report, (_best_tracker if _best_tracker else None)


# Lookup table used by the override block to reconstruct fusion metadata from
# the decomposition score name, without re-parsing the string.
_FUSION_METADATA: dict[str, tuple[str, list[str], "float | None"]] = {
    "product":                                     ("raw",         ["product"],                                                                    None),
    "reconstruction":                              ("raw",         ["reconstruction"],                                                             None),
    "association_discrepancy":                     ("raw",         ["association_discrepancy"],                                                    None),
    "association_affinity":                        ("raw",         ["association_affinity"],                                                       None),
    "dc_voltage_drop":                             ("raw",         ["dc_voltage_drop"],                                                            None),
    "mismatch_imbalance":                          ("raw",         ["mismatch_imbalance"],                                                         None),
    "drift_efficiency":                            ("raw",         ["drift_efficiency"],                                                           None),
    "max_z_recon_product":                         ("max_z",       ["reconstruction", "product"],                                                  None),
    "logsumexp_z_recon_product_tau_0.5":           ("logsumexp_z", ["reconstruction", "product"],                                                  0.5),
    "logsumexp_z_recon_product_tau_1":             ("logsumexp_z", ["reconstruction", "product"],                                                  1.0),
    "max_z_product_drift":                         ("max_z",       ["product", "drift_efficiency"],                                                None),
    "max_z_recon_product_drift":                   ("max_z",       ["reconstruction", "product", "drift_efficiency"],                              None),
    "logsumexp_z_product_drift_tau_0.5":           ("logsumexp_z", ["product", "drift_efficiency"],                                                0.5),
    "logsumexp_z_product_drift_tau_1":             ("logsumexp_z", ["product", "drift_efficiency"],                                                1.0),
    "max_z_product_voltage":                       ("max_z",       ["product", "dc_voltage_drop"],                                                 None),
    "max_z_recon_product_voltage":                 ("max_z",       ["reconstruction", "product", "dc_voltage_drop"],                               None),
    "logsumexp_z_product_voltage_tau_0.5":         ("logsumexp_z", ["product", "dc_voltage_drop"],                                                 0.5),
    "logsumexp_z_product_voltage_tau_1":           ("logsumexp_z", ["product", "dc_voltage_drop"],                                                 1.0),
    "max_z_recon_product_dc_voltage_mismatch":     ("max_z",       ["reconstruction", "product", "dc_voltage_drop", "mismatch_imbalance"],         None),
    "logsumexp_z_recon_product_dc_voltage_mismatch_tau_0.5": (
        "logsumexp_z", ["reconstruction", "product", "dc_voltage_drop", "mismatch_imbalance"],  0.5),
    "logsumexp_z_recon_product_dc_voltage_mismatch_tau_1": (
        "logsumexp_z", ["reconstruction", "product", "dc_voltage_drop", "mismatch_imbalance"],  1.0),
}


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
        total_steps: int = 1,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
        score_reduction: str = "center",
        drift_feature_idx: int | None = None,
        voltage_drop_feature_indices: list[int] | None = None,
        power_imbalance_feature_idx: int | None = None,
        current_imbalance_feature_idx: int | None = None,
        voltage_imbalance_feature_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.k = k
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.max_epochs = max_epochs
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.score_reduction = score_reduction
        self.drift_feature_idx = drift_feature_idx
        self.voltage_drop_feature_indices = voltage_drop_feature_indices or []
        self.power_imbalance_feature_idx = power_imbalance_feature_idx
        self.current_imbalance_feature_idx = current_imbalance_feature_idx
        self.voltage_imbalance_feature_idx = voltage_imbalance_feature_idx

        self._val_outputs: list[dict] = []
        self._test_outputs: list[dict] = []
        self.best_val_pr_auc: float = 0.0
        self.best_val_pv_maat_f1: float = 0.0
        self.best_val_episode_macro_f1: float = 0.0
        self.best_val_selection_score: float = 0.0
        self.val_threshold: float = 0.5
        self._pv_score_stats: dict[str, tuple[float, float]] | None = None
        self._diag_score_stats: dict[str, dict[str, tuple[float, float]]] = {}
        self._val_scores_np: np.ndarray | None = None
        self._val_product_scores_np: np.ndarray | None = None
        self._val_labels_np: np.ndarray | None = None
        self._val_recon_scores_np: np.ndarray | None = None
        self._val_assoc_scores_np: np.ndarray | None = None
        self._val_assoc_affinity_scores_np: np.ndarray | None = None
        self._val_drift_scores_np: np.ndarray | None = None
        self._val_voltage_drop_scores_np: np.ndarray | None = None
        self._val_mismatch_scores_np: np.ndarray | None = None
        self._val_original_labels_np: np.ndarray | None = None
        self._val_group_ids_np: np.ndarray | None = None
        self._test_scores_np: np.ndarray | None = None
        self._test_product_scores_np: np.ndarray | None = None
        self._test_labels_np: np.ndarray | None = None
        self._test_recon_scores_np: np.ndarray | None = None
        self._test_assoc_scores_np: np.ndarray | None = None
        self._test_assoc_affinity_scores_np: np.ndarray | None = None
        self._test_drift_scores_np: np.ndarray | None = None
        self._test_voltage_drop_scores_np: np.ndarray | None = None
        self._test_mismatch_scores_np: np.ndarray | None = None
        self._test_original_labels_np: np.ndarray | None = None
        self._test_group_ids_np: np.ndarray | None = None
        self._nan_count: int = 0

    def _compute_step(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        x_hat, series_list, prior_list, _ = self.model(x)
        rec_loss = nn.functional.mse_loss(x_hat, x)
        s_loss, p_loss = association_losses(series_list, prior_list)
        loss1 = rec_loss - self.k * s_loss
        loss2 = rec_loss + self.k * p_loss
        return {
            "rec_loss": rec_loss,
            "series_loss": s_loss,
            "prior_loss": p_loss,
            "loss1": loss1,
            "loss2": loss2,
        }

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, _ = batch
        opt = self.optimizers()

        step = self._compute_step(x)
        rec_loss = step["rec_loss"]
        s_loss = step["series_loss"]
        p_loss = step["prior_loss"]
        loss1 = step["loss1"]
        loss2 = step["loss2"]

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
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        self.log("train_rec_loss", rec_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_series_kl", s_loss, on_step=False, on_epoch=True)
        self.log("train_prior_kl", p_loss, on_step=False, on_epoch=True)
        self.log("train_loss1", loss1, on_step=False, on_epoch=True)
        self.log("train_loss2", loss2, on_step=False, on_epoch=True)
        self.log("grad_norm", grad_norm, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
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
        if self.score_reduction == "median":
            return scores.median(dim=1).values
        # default: center
        return scores[:, scores.size(1) // 2]

    def _diag_window_reductions(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_scores = scores.max(dim=1).values
        q95_scores = torch.quantile(scores, 0.95, dim=1)
        topk = max(1, int(math.ceil(scores.size(1) * 0.10)))
        top10_scores = scores.topk(k=topk, dim=1).values.mean(dim=1)
        return max_scores, q95_scores, top10_scores

    def _reduce_scores_by_mode(self, scores: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "max":
            return scores.max(dim=1).values
        if mode == "q95":
            return torch.quantile(scores, 0.95, dim=1)
        if mode == "top10mean":
            topk = max(1, int(math.ceil(scores.size(1) * 0.10)))
            return scores.topk(k=topk, dim=1).values.mean(dim=1)
        raise ValueError(f"Unknown reduction mode: {mode}")

    def _component_scores_by_mode(
        self,
        x: torch.Tensor,
        product_scores: torch.Tensor,
        mode: str,
    ) -> dict[str, torch.Tensor]:
        product = self._reduce_scores_by_mode(product_scores, mode)

        voltage = None
        if self.voltage_drop_feature_indices:
            voltage_parts: list[torch.Tensor] = []
            for idx in self.voltage_drop_feature_indices:
                series = torch.clamp(-x[:, :, idx], min=0.0)
                voltage_parts.append(self._reduce_scores_by_mode(series, mode))
            voltage = torch.stack(voltage_parts, dim=1).max(dim=1).values

        mismatch = None
        mismatch_parts: list[torch.Tensor] = []
        if self.power_imbalance_feature_idx is not None:
            series = torch.clamp(x[:, :, self.power_imbalance_feature_idx], min=0.0)
            mismatch_parts.append(self._reduce_scores_by_mode(series, mode))
        if self.current_imbalance_feature_idx is not None:
            series = torch.clamp(x[:, :, self.current_imbalance_feature_idx], min=0.0)
            mismatch_parts.append(self._reduce_scores_by_mode(series, mode))
        if self.voltage_imbalance_feature_idx is not None:
            series = torch.clamp(x[:, :, self.voltage_imbalance_feature_idx], min=0.0)
            mismatch_parts.append(self._reduce_scores_by_mode(series, mode))
        if mismatch_parts:
            mismatch = torch.stack(mismatch_parts, dim=1).max(dim=1).values

        return {
            "product": product,
            "dc_voltage_drop": voltage,
            "mismatch_imbalance": mismatch,
        }

    def _fuse_components_np(
        self,
        components: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
        product = components["product"]
        voltage = components.get("dc_voltage_drop")
        mismatch = components.get("mismatch_imbalance")
        if voltage is None or mismatch is None:
            p_mu, p_sd = _fit_normal_stats(product, labels)
            fused = _zscore_with_stats(product, p_mu, p_sd)
            return fused, {"product": (p_mu, p_sd)}

        p_mu, p_sd = _fit_normal_stats(product, labels)
        v_mu, v_sd = _fit_normal_stats(voltage, labels)
        m_mu, m_sd = _fit_normal_stats(mismatch, labels)
        fused = _logsumexp_multi(
            [
                _zscore_with_stats(product, p_mu, p_sd),
                _zscore_with_stats(voltage, v_mu, v_sd),
                _zscore_with_stats(mismatch, m_mu, m_sd),
            ],
            tau=1.0,
        )
        return fused, {
            "product": (p_mu, p_sd),
            "dc_voltage_drop": (v_mu, v_sd),
            "mismatch_imbalance": (m_mu, m_sd),
        }

    def _fuse_components_with_stats_np(
        self,
        components: dict[str, np.ndarray],
        stats: dict[str, tuple[float, float]],
    ) -> np.ndarray:
        if "product" not in stats:
            return components["product"]
        product = _zscore_with_stats(components["product"], *stats["product"])
        if "dc_voltage_drop" not in stats or "mismatch_imbalance" not in stats:
            return product
        voltage = _zscore_with_stats(components["dc_voltage_drop"], *stats["dc_voltage_drop"])
        mismatch = _zscore_with_stats(components["mismatch_imbalance"], *stats["mismatch_imbalance"])
        return _logsumexp_multi([product, voltage, mismatch], tau=1.0)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            x, labels, original_labels, group_ids = batch
        elif len(batch) == 3:
            x, labels, original_labels = batch
            group_ids = None
        else:
            x, labels = batch
            original_labels = labels
            group_ids = None
        with torch.no_grad():
            x_hat, series_list, prior_list, _ = self.model(x)

        recon_error = ((x - x_hat) ** 2).mean(dim=-1)  # [B, W]
        assoc_dis = compute_association_discrepancy(series_list, prior_list)  # [B, W]
        assoc_affinity = torch.softmax(-assoc_dis * self.temperature, dim=-1)
        drift_scores = None
        if self.drift_feature_idx is not None:
            drift_series = torch.clamp(-x[:, :, self.drift_feature_idx], min=0.0)
            drift_scores = drift_series.mean(dim=1)
        voltage_drop_scores = None
        voltage_parts: list[torch.Tensor] = []
        for idx in self.voltage_drop_feature_indices:
            voltage_parts.append(torch.clamp(-x[:, :, idx], min=0.0).max(dim=1).values)
        if voltage_parts:
            voltage_drop_scores = torch.stack(voltage_parts, dim=1).max(dim=1).values
        product_scores = compute_maat_scores(series_list, prior_list, recon_error, self.temperature)
        product_window_scores = self._reduce_scores(product_scores)
        diag_max = self._component_scores_by_mode(x, product_scores, "max")
        diag_q95 = self._component_scores_by_mode(x, product_scores, "q95")
        diag_top10 = self._component_scores_by_mode(x, product_scores, "top10mean")
        center_scores = product_window_scores.cpu()
        mismatch_scores = None
        mismatch_parts: list[torch.Tensor] = []
        if self.power_imbalance_feature_idx is not None:
            mismatch_parts.append(torch.clamp(x[:, :, self.power_imbalance_feature_idx], min=0.0).max(dim=1).values)
        if self.current_imbalance_feature_idx is not None:
            mismatch_parts.append(torch.clamp(x[:, :, self.current_imbalance_feature_idx], min=0.0).max(dim=1).values)
        if self.voltage_imbalance_feature_idx is not None:
            mismatch_parts.append(torch.clamp(x[:, :, self.voltage_imbalance_feature_idx], min=0.0).max(dim=1).values)
        if mismatch_parts:
            mismatch_scores = torch.stack(mismatch_parts, dim=1).max(dim=1).values
        center_labels = labels.cpu()

        rec_loss = nn.functional.mse_loss(x_hat, x)
        s_loss, p_loss = association_losses(series_list, prior_list)

        self._val_outputs.append({
            "scores": center_scores,
            "product_scores": product_window_scores.cpu(),
            "recon_scores": self._reduce_scores(recon_error).cpu(),
            "assoc_scores": self._reduce_scores(assoc_dis).cpu(),
            "assoc_affinity_scores": self._reduce_scores(assoc_affinity).cpu(),
            "drift_scores": drift_scores.cpu() if drift_scores is not None else None,
            "voltage_drop_scores": (
                voltage_drop_scores.cpu() if voltage_drop_scores is not None else None
            ),
            "mismatch_scores": mismatch_scores.cpu() if mismatch_scores is not None else None,
            "labels": center_labels,
            "original_labels": original_labels.cpu(),
            "group_ids": list(group_ids) if group_ids is not None else None,
            "diag_max_product": diag_max["product"].cpu(),
            "diag_max_voltage": diag_max["dc_voltage_drop"].cpu() if diag_max["dc_voltage_drop"] is not None else None,
            "diag_max_mismatch": diag_max["mismatch_imbalance"].cpu() if diag_max["mismatch_imbalance"] is not None else None,
            "diag_q95_product": diag_q95["product"].cpu(),
            "diag_q95_voltage": diag_q95["dc_voltage_drop"].cpu() if diag_q95["dc_voltage_drop"] is not None else None,
            "diag_q95_mismatch": diag_q95["mismatch_imbalance"].cpu() if diag_q95["mismatch_imbalance"] is not None else None,
            "diag_top10_product": diag_top10["product"].cpu(),
            "diag_top10_voltage": diag_top10["dc_voltage_drop"].cpu() if diag_top10["dc_voltage_drop"] is not None else None,
            "diag_top10_mismatch": diag_top10["mismatch_imbalance"].cpu() if diag_top10["mismatch_imbalance"] is not None else None,
            "rec_loss": rec_loss.item(),
            "s_loss": s_loss.item(),
            "p_loss": p_loss.item(),
        })

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return

        all_scores = torch.cat([o["scores"] for o in self._val_outputs]).numpy()
        all_product_scores = torch.cat([o["product_scores"] for o in self._val_outputs]).numpy()
        all_recon_scores = torch.cat([o["recon_scores"] for o in self._val_outputs]).numpy()
        all_assoc_scores = torch.cat([o["assoc_scores"] for o in self._val_outputs]).numpy()
        all_assoc_affinity_scores = torch.cat(
            [o["assoc_affinity_scores"] for o in self._val_outputs]
        ).numpy()
        drift_present = self._val_outputs[0].get("drift_scores") is not None
        vdrop_present = self._val_outputs[0].get("voltage_drop_scores") is not None
        mismatch_present = self._val_outputs[0].get("mismatch_scores") is not None
        all_drift_scores = (
            torch.cat([o["drift_scores"] for o in self._val_outputs]).numpy() if drift_present else None
        )
        all_voltage_drop_scores = (
            torch.cat([o["voltage_drop_scores"] for o in self._val_outputs]).numpy()
            if vdrop_present else None
        )
        all_mismatch_scores = (
            torch.cat([o["mismatch_scores"] for o in self._val_outputs]).numpy()
            if mismatch_present else None
        )
        all_labels = torch.cat([o["labels"] for o in self._val_outputs]).numpy()
        all_original_labels = torch.cat([o["original_labels"] for o in self._val_outputs]).numpy()
        all_diag_max_product = torch.cat([o["diag_max_product"] for o in self._val_outputs]).numpy()
        all_diag_q95_product = torch.cat([o["diag_q95_product"] for o in self._val_outputs]).numpy()
        all_diag_top10_product = torch.cat([o["diag_top10_product"] for o in self._val_outputs]).numpy()
        all_diag_max_voltage = (
            torch.cat([o["diag_max_voltage"] for o in self._val_outputs]).numpy()
            if self._val_outputs[0].get("diag_max_voltage") is not None
            else None
        )
        all_diag_q95_voltage = (
            torch.cat([o["diag_q95_voltage"] for o in self._val_outputs]).numpy()
            if self._val_outputs[0].get("diag_q95_voltage") is not None
            else None
        )
        all_diag_top10_voltage = (
            torch.cat([o["diag_top10_voltage"] for o in self._val_outputs]).numpy()
            if self._val_outputs[0].get("diag_top10_voltage") is not None
            else None
        )
        all_diag_max_mismatch = (
            torch.cat([o["diag_max_mismatch"] for o in self._val_outputs]).numpy()
            if self._val_outputs[0].get("diag_max_mismatch") is not None
            else None
        )
        all_diag_q95_mismatch = (
            torch.cat([o["diag_q95_mismatch"] for o in self._val_outputs]).numpy()
            if self._val_outputs[0].get("diag_q95_mismatch") is not None
            else None
        )
        all_diag_top10_mismatch = (
            torch.cat([o["diag_top10_mismatch"] for o in self._val_outputs]).numpy()
            if self._val_outputs[0].get("diag_top10_mismatch") is not None
            else None
        )
        all_group_ids = None
        if self._val_outputs[0].get("group_ids") is not None:
            all_group_ids = np.array(
                [gid for o in self._val_outputs for gid in (o.get("group_ids") or [])],
                dtype=object,
            )
        mean_rec = float(np.mean([o["rec_loss"] for o in self._val_outputs]))
        mean_s = float(np.mean([o["s_loss"] for o in self._val_outputs]))
        mean_p = float(np.mean([o["p_loss"] for o in self._val_outputs]))
        self._val_outputs.clear()

        if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
            self.log("val_pr_auc", 0.0)
            self.log("val_pv_maat_f1", 0.0, prog_bar=True)
            self.log("val_episode_macro_f1", 0.0)
            self.log("val_selection_score", 0.0, prog_bar=True)
            return

        val_pr_auc = _safe_pr_auc(all_labels, all_scores)
        val_roc_auc = _safe_roc_auc(all_labels, all_scores)

        self.best_val_pr_auc = max(self.best_val_pr_auc, val_pr_auc)
        self._val_scores_np = all_scores
        self._val_product_scores_np = all_product_scores
        self._val_labels_np = all_labels
        self._val_recon_scores_np = all_recon_scores
        self._val_assoc_scores_np = all_assoc_scores
        self._val_assoc_affinity_scores_np = all_assoc_affinity_scores
        self._val_drift_scores_np = all_drift_scores
        self._val_voltage_drop_scores_np = all_voltage_drop_scores
        self._val_mismatch_scores_np = all_mismatch_scores
        self._val_original_labels_np = all_original_labels
        self._val_group_ids_np = all_group_ids

        if all_voltage_drop_scores is not None and all_mismatch_scores is not None:
            if (
                not _is_finite_array(all_product_scores)
                or not _is_finite_array(all_voltage_drop_scores)
                or not _is_finite_array(all_mismatch_scores)
            ):
                self.log("val_pv_maat_f1", 0.0, prog_bar=True)
                self.log("val_episode_macro_f1", 0.0)
                self.log("val_selection_score", 0.0, prog_bar=True)
                return
            p_mu, p_sd = _fit_normal_stats(all_product_scores, all_labels)
            v_mu, v_sd = _fit_normal_stats(all_voltage_drop_scores, all_labels)
            m_mu, m_sd = _fit_normal_stats(all_mismatch_scores, all_labels)
            pv_scores = _logsumexp_multi(
                [
                    _zscore_with_stats(all_product_scores, p_mu, p_sd),
                    _zscore_with_stats(all_voltage_drop_scores, v_mu, v_sd),
                    _zscore_with_stats(all_mismatch_scores, m_mu, m_sd),
                ],
                tau=1.0,
            )
            self._pv_score_stats = {
                "product": (p_mu, p_sd),
                "dc_voltage_drop": (v_mu, v_sd),
                "mismatch_imbalance": (m_mu, m_sd),
            }
        else:
            if not _is_finite_array(all_product_scores):
                self.log("val_pv_maat_f1", 0.0, prog_bar=True)
                self.log("val_episode_macro_f1", 0.0)
                self.log("val_selection_score", 0.0, prog_bar=True)
                return
            pv_scores = all_product_scores
            p_mu, p_sd = _fit_normal_stats(all_product_scores, all_labels)
            self._pv_score_stats = {"product": (p_mu, p_sd)}

        if not _is_finite_array(pv_scores):
            self.log("val_pv_maat_f1", 0.0, prog_bar=True)
            self.log("val_episode_macro_f1", 0.0)
            self.log("val_selection_score", 0.0, prog_bar=True)
            return

        threshold, val_f1, val_prec, val_rec = _calibrate_threshold(pv_scores, all_labels)
        if not math.isfinite(threshold):
            self.log("val_pv_maat_f1", 0.0, prog_bar=True)
            self.log("val_episode_macro_f1", 0.0)
            self.log("val_selection_score", 0.0, prog_bar=True)
            return

        val_preds = (pv_scores >= threshold).astype(int)
        val_episode_macro_f1 = _episode_macro_f1_binary(all_labels, val_preds, all_group_ids)
        val_selection_score = val_f1
        val_acc = float(accuracy_score(all_labels, val_preds))
        val_pv_pr_auc = _safe_pr_auc(all_labels, pv_scores)

        self.best_val_pv_maat_f1 = max(self.best_val_pv_maat_f1, val_f1)
        self.best_val_episode_macro_f1 = max(self.best_val_episode_macro_f1, val_episode_macro_f1)
        self.best_val_selection_score = max(self.best_val_selection_score, val_selection_score)
        self.val_threshold = threshold
        self._val_scores_np = pv_scores
        max_fused, max_stats = self._fuse_components_np(
            {
                "product": all_diag_max_product,
                "dc_voltage_drop": all_diag_max_voltage,
                "mismatch_imbalance": all_diag_max_mismatch,
            },
            all_labels,
        )
        q95_fused, q95_stats = self._fuse_components_np(
            {
                "product": all_diag_q95_product,
                "dc_voltage_drop": all_diag_q95_voltage,
                "mismatch_imbalance": all_diag_q95_mismatch,
            },
            all_labels,
        )
        top10_fused, top10_stats = self._fuse_components_np(
            {
                "product": all_diag_top10_product,
                "dc_voltage_drop": all_diag_top10_voltage,
                "mismatch_imbalance": all_diag_top10_mismatch,
            },
            all_labels,
        )
        self._val_diag_scores = {"max": max_fused, "q95": q95_fused, "top10mean": top10_fused}
        self._diag_score_stats = {"max": max_stats, "q95": q95_stats, "top10mean": top10_stats}

        self.log("val_pr_auc", val_pr_auc)
        self.log("val_pv_maat_pr_auc", val_pv_pr_auc)
        self.log("val_pv_maat_f1", val_f1, prog_bar=True)
        self.log("val_episode_macro_f1", val_episode_macro_f1)
        self.log("val_selection_score", val_selection_score, prog_bar=True)
        self.log("val_roc_auc", val_roc_auc)
        self.log("val_pv_maat_precision", val_prec)
        self.log("val_pv_maat_recall", val_rec)
        self.log("val_pv_maat_accuracy", val_acc)
        self.log("val_pv_maat_threshold", threshold)
        self.log("val_rec_loss", mean_rec)
        self.log("val_series_kl", mean_s)
        self.log("val_prior_kl", mean_p)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        if len(batch) == 4:
            x, labels, original_labels, group_ids = batch
        elif len(batch) == 3:
            x, labels, original_labels = batch
            group_ids = None
        else:
            x, labels = batch
            original_labels = labels
            group_ids = None
        with torch.no_grad():
            x_hat, series_list, prior_list, _ = self.model(x)

        recon_error = ((x - x_hat) ** 2).mean(dim=-1)
        assoc_dis = compute_association_discrepancy(series_list, prior_list)
        assoc_affinity = torch.softmax(-assoc_dis * self.temperature, dim=-1)
        drift_scores = None
        if self.drift_feature_idx is not None:
            drift_series = torch.clamp(-x[:, :, self.drift_feature_idx], min=0.0)
            drift_scores = drift_series.mean(dim=1)
        voltage_drop_scores = None
        voltage_parts: list[torch.Tensor] = []
        for idx in self.voltage_drop_feature_indices:
            voltage_parts.append(torch.clamp(-x[:, :, idx], min=0.0).max(dim=1).values)
        if voltage_parts:
            voltage_drop_scores = torch.stack(voltage_parts, dim=1).max(dim=1).values
        product_scores = compute_maat_scores(series_list, prior_list, recon_error, self.temperature)
        product_window_scores = self._reduce_scores(product_scores)
        diag_max = self._component_scores_by_mode(x, product_scores, "max")
        diag_q95 = self._component_scores_by_mode(x, product_scores, "q95")
        diag_top10 = self._component_scores_by_mode(x, product_scores, "top10mean")
        window_scores = product_window_scores
        mismatch_scores = None
        mismatch_parts: list[torch.Tensor] = []
        if self.power_imbalance_feature_idx is not None:
            mismatch_parts.append(torch.clamp(x[:, :, self.power_imbalance_feature_idx], min=0.0).max(dim=1).values)
        if self.current_imbalance_feature_idx is not None:
            mismatch_parts.append(torch.clamp(x[:, :, self.current_imbalance_feature_idx], min=0.0).max(dim=1).values)
        if self.voltage_imbalance_feature_idx is not None:
            mismatch_parts.append(torch.clamp(x[:, :, self.voltage_imbalance_feature_idx], min=0.0).max(dim=1).values)
        if mismatch_parts:
            mismatch_scores = torch.stack(mismatch_parts, dim=1).max(dim=1).values
        self._test_outputs.append({
            "scores": window_scores.cpu(),
            "product_scores": product_window_scores.cpu(),
            "recon_scores": self._reduce_scores(recon_error).cpu(),
            "assoc_scores": self._reduce_scores(assoc_dis).cpu(),
            "assoc_affinity_scores": self._reduce_scores(assoc_affinity).cpu(),
            "drift_scores": drift_scores.cpu() if drift_scores is not None else None,
            "voltage_drop_scores": (
                voltage_drop_scores.cpu() if voltage_drop_scores is not None else None
            ),
            "mismatch_scores": mismatch_scores.cpu() if mismatch_scores is not None else None,
            "labels": labels.cpu(),
            "original_labels": original_labels.cpu(),
            "group_ids": list(group_ids) if group_ids is not None else None,
            "diag_max_product": diag_max["product"].cpu(),
            "diag_max_voltage": diag_max["dc_voltage_drop"].cpu() if diag_max["dc_voltage_drop"] is not None else None,
            "diag_max_mismatch": diag_max["mismatch_imbalance"].cpu() if diag_max["mismatch_imbalance"] is not None else None,
            "diag_q95_product": diag_q95["product"].cpu(),
            "diag_q95_voltage": diag_q95["dc_voltage_drop"].cpu() if diag_q95["dc_voltage_drop"] is not None else None,
            "diag_q95_mismatch": diag_q95["mismatch_imbalance"].cpu() if diag_q95["mismatch_imbalance"] is not None else None,
            "diag_top10_product": diag_top10["product"].cpu(),
            "diag_top10_voltage": diag_top10["dc_voltage_drop"].cpu() if diag_top10["dc_voltage_drop"] is not None else None,
            "diag_top10_mismatch": diag_top10["mismatch_imbalance"].cpu() if diag_top10["mismatch_imbalance"] is not None else None,
        })

    def on_test_epoch_end(self) -> None:
        if not self._test_outputs:
            return

        all_scores = torch.cat([o["scores"] for o in self._test_outputs]).numpy()
        all_product_scores = torch.cat([o["product_scores"] for o in self._test_outputs]).numpy()
        all_recon_scores = torch.cat([o["recon_scores"] for o in self._test_outputs]).numpy()
        all_assoc_scores = torch.cat([o["assoc_scores"] for o in self._test_outputs]).numpy()
        all_assoc_affinity_scores = torch.cat(
            [o["assoc_affinity_scores"] for o in self._test_outputs]
        ).numpy()
        drift_present = self._test_outputs[0].get("drift_scores") is not None
        vdrop_present = self._test_outputs[0].get("voltage_drop_scores") is not None
        mismatch_present = self._test_outputs[0].get("mismatch_scores") is not None
        all_drift_scores = (
            torch.cat([o["drift_scores"] for o in self._test_outputs]).numpy() if drift_present else None
        )
        all_voltage_drop_scores = (
            torch.cat([o["voltage_drop_scores"] for o in self._test_outputs]).numpy()
            if vdrop_present else None
        )
        all_mismatch_scores = (
            torch.cat([o["mismatch_scores"] for o in self._test_outputs]).numpy()
            if mismatch_present else None
        )
        all_labels = torch.cat([o["labels"] for o in self._test_outputs]).numpy()
        all_original_labels = torch.cat([o["original_labels"] for o in self._test_outputs]).numpy()
        all_diag_max_product = torch.cat([o["diag_max_product"] for o in self._test_outputs]).numpy()
        all_diag_q95_product = torch.cat([o["diag_q95_product"] for o in self._test_outputs]).numpy()
        all_diag_top10_product = torch.cat([o["diag_top10_product"] for o in self._test_outputs]).numpy()
        all_diag_max_voltage = (
            torch.cat([o["diag_max_voltage"] for o in self._test_outputs]).numpy()
            if self._test_outputs[0].get("diag_max_voltage") is not None
            else None
        )
        all_diag_q95_voltage = (
            torch.cat([o["diag_q95_voltage"] for o in self._test_outputs]).numpy()
            if self._test_outputs[0].get("diag_q95_voltage") is not None
            else None
        )
        all_diag_top10_voltage = (
            torch.cat([o["diag_top10_voltage"] for o in self._test_outputs]).numpy()
            if self._test_outputs[0].get("diag_top10_voltage") is not None
            else None
        )
        all_diag_max_mismatch = (
            torch.cat([o["diag_max_mismatch"] for o in self._test_outputs]).numpy()
            if self._test_outputs[0].get("diag_max_mismatch") is not None
            else None
        )
        all_diag_q95_mismatch = (
            torch.cat([o["diag_q95_mismatch"] for o in self._test_outputs]).numpy()
            if self._test_outputs[0].get("diag_q95_mismatch") is not None
            else None
        )
        all_diag_top10_mismatch = (
            torch.cat([o["diag_top10_mismatch"] for o in self._test_outputs]).numpy()
            if self._test_outputs[0].get("diag_top10_mismatch") is not None
            else None
        )
        all_group_ids = None
        if self._test_outputs[0].get("group_ids") is not None:
            all_group_ids = np.array(
                [gid for o in self._test_outputs for gid in (o.get("group_ids") or [])],
                dtype=object,
            )
        self._test_outputs.clear()

        if (
            self._pv_score_stats is not None
            and all_voltage_drop_scores is not None
            and all_mismatch_scores is not None
        ):
            p_mu, p_sd = self._pv_score_stats["product"]
            v_mu, v_sd = self._pv_score_stats["dc_voltage_drop"]
            m_mu, m_sd = self._pv_score_stats["mismatch_imbalance"]
            all_scores = _logsumexp_multi(
                [
                    _zscore_with_stats(all_product_scores, p_mu, p_sd),
                    _zscore_with_stats(all_voltage_drop_scores, v_mu, v_sd),
                    _zscore_with_stats(all_mismatch_scores, m_mu, m_sd),
                ],
                tau=1.0,
            )
        else:
            all_scores = all_product_scores

        self._test_scores_np = all_scores
        self._test_product_scores_np = all_product_scores
        self._test_labels_np = all_labels
        self._test_recon_scores_np = all_recon_scores
        self._test_assoc_scores_np = all_assoc_scores
        self._test_assoc_affinity_scores_np = all_assoc_affinity_scores
        self._test_drift_scores_np = all_drift_scores
        self._test_voltage_drop_scores_np = all_voltage_drop_scores
        self._test_mismatch_scores_np = all_mismatch_scores
        self._test_original_labels_np = all_original_labels
        self._test_group_ids_np = all_group_ids
        self._test_diag_scores = {
            "max": self._fuse_components_with_stats_np(
                {
                    "product": all_diag_max_product,
                    "dc_voltage_drop": all_diag_max_voltage,
                    "mismatch_imbalance": all_diag_max_mismatch,
                },
                self._diag_score_stats.get("max", self._pv_score_stats or {}),
            ),
            "q95": self._fuse_components_with_stats_np(
                {
                    "product": all_diag_q95_product,
                    "dc_voltage_drop": all_diag_q95_voltage,
                    "mismatch_imbalance": all_diag_q95_mismatch,
                },
                self._diag_score_stats.get("q95", self._pv_score_stats or {}),
            ),
            "top10mean": self._fuse_components_with_stats_np(
                {
                    "product": all_diag_top10_product,
                    "dc_voltage_drop": all_diag_top10_voltage,
                    "mismatch_imbalance": all_diag_top10_mismatch,
                },
                self._diag_score_stats.get("top10mean", self._pv_score_stats or {}),
            ),
        }

        if not _is_finite_array(all_scores) or not math.isfinite(self.val_threshold):
            self.log("test_pv_maat_pr_auc", 0.0)
            self.log("test_pv_maat_f1", 0.0)
            self.log("test_episode_macro_f1", 0.0)
            return

        if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
            self.log("test_pv_maat_pr_auc", 0.0)
            return

        test_pr_auc = _safe_pr_auc(all_labels, all_scores)
        test_roc_auc = _safe_roc_auc(all_labels, all_scores)
        test_preds = (all_scores >= self.val_threshold).astype(int)
        test_f1 = float(f1_score(all_labels, test_preds, zero_division=0))
        test_episode_macro_f1 = _episode_macro_f1_binary(all_labels, test_preds, all_group_ids)
        test_acc = float(accuracy_score(all_labels, test_preds))
        test_prec = float(precision_score(all_labels, test_preds, zero_division=0))
        test_rec = float(recall_score(all_labels, test_preds, zero_division=0))

        self.log("test_pv_maat_pr_auc", test_pr_auc, prog_bar=True)
        self.log("test_roc_auc", test_roc_auc)
        self.log("test_pv_maat_f1", test_f1)
        self.log("test_episode_macro_f1", test_episode_macro_f1)
        self.log("test_pv_maat_accuracy", test_acc)
        self.log("test_pv_maat_precision", test_prec)
        self.log("test_pv_maat_recall", test_rec)

    @torch.no_grad()
    def score_windows(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of windows [B, W, F] → anomaly scores [B]."""
        x_hat, series_list, prior_list, _ = self.model(x)
        recon_error = ((x - x_hat) ** 2).mean(dim=-1)
        product_scores = compute_maat_scores(series_list, prior_list, recon_error, self.temperature)
        product = product_scores.max(dim=1).values.detach().cpu().numpy()

        voltage = None
        voltage_parts: list[torch.Tensor] = []
        for idx in self.voltage_drop_feature_indices:
            voltage_parts.append(torch.clamp(-x[:, :, idx], min=0.0).max(dim=1).values)
        if voltage_parts:
            voltage = torch.stack(voltage_parts, dim=1).max(dim=1).values

        mismatch = None
        mismatch_parts: list[torch.Tensor] = []
        if self.power_imbalance_feature_idx is not None:
            mismatch_parts.append(torch.clamp(x[:, :, self.power_imbalance_feature_idx], min=0.0).max(dim=1).values)
        if self.current_imbalance_feature_idx is not None:
            mismatch_parts.append(torch.clamp(x[:, :, self.current_imbalance_feature_idx], min=0.0).max(dim=1).values)
        if self.voltage_imbalance_feature_idx is not None:
            mismatch_parts.append(torch.clamp(x[:, :, self.voltage_imbalance_feature_idx], min=0.0).max(dim=1).values)
        if mismatch_parts:
            mismatch = torch.stack(mismatch_parts, dim=1).max(dim=1).values

        if self._pv_score_stats is None:
            raise ValueError("MAAT score calibration stats are missing; cannot compute pv_maat_score")

        if voltage is None or mismatch is None:
            if (
                "dc_voltage_drop" in self._pv_score_stats
                or "mismatch_imbalance" in self._pv_score_stats
            ):
                raise ValueError(
                    "MAAT physics features unavailable at inference but fused calibration requires them"
                )
            p_mu, p_sd = self._pv_score_stats["product"]
            fused = _zscore_with_stats(product, p_mu, p_sd)
        else:
            if "dc_voltage_drop" not in self._pv_score_stats or "mismatch_imbalance" not in self._pv_score_stats:
                raise ValueError("MAAT physics calibration stats missing for fused pv_maat_score")
            voltage_np = voltage.detach().cpu().numpy()
            mismatch_np = mismatch.detach().cpu().numpy()
            p_mu, p_sd = self._pv_score_stats["product"]
            v_mu, v_sd = self._pv_score_stats["dc_voltage_drop"]
            m_mu, m_sd = self._pv_score_stats["mismatch_imbalance"]
            fused = _logsumexp_multi(
                [
                    _zscore_with_stats(product, p_mu, p_sd),
                    _zscore_with_stats(voltage_np, v_mu, v_sd),
                    _zscore_with_stats(mismatch_np, m_mu, m_sd),
                ],
                tau=1.0,
            )
        return torch.from_numpy(fused).to(device=x.device, dtype=x.dtype)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["pv_score_stats"] = self._pv_score_stats
        checkpoint["diag_score_stats"] = self._diag_score_stats

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        loaded = checkpoint.get("pv_score_stats")
        if isinstance(loaded, dict):
            self._pv_score_stats = {
                str(k): (float(v[0]), float(v[1])) for k, v in loaded.items() if isinstance(v, (list, tuple)) and len(v) == 2
            }
        loaded_diag = checkpoint.get("diag_score_stats")
        if isinstance(loaded_diag, dict):
            casted: dict[str, dict[str, tuple[float, float]]] = {}
            for mode, stats in loaded_diag.items():
                if not isinstance(stats, dict):
                    continue
                casted[str(mode)] = {
                    str(k): (float(v[0]), float(v[1]))
                    for k, v in stats.items()
                    if isinstance(v, (list, tuple)) and len(v) == 2
                }
            self._diag_score_stats = casted

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
        # With automatic_optimization=False, Lightning does not step the scheduler
        # automatically. We step it manually after each optimizer step.
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
        if not np.all(np.isfinite(scores)) or labels.sum() in (0, len(labels)):
            continue
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
    """Reports monitored validation metric to Optuna and raises TrialPruned."""

    def __init__(
        self,
        trial: optuna.Trial,
        warmup_epochs: int = 3,
        monitor: str = "val_selection_score",
    ) -> None:
        self.trial = trial
        self.warmup_epochs = warmup_epochs
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch < self.warmup_epochs:
            return
        monitored = trainer.callback_metrics.get(self.monitor)
        if monitored is None:
            return
        self.trial.report(float(monitored), step=trainer.current_epoch)
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
    total_steps: int,
    drift_feature_idx: int | None,
    voltage_drop_feature_indices: list[int],
    power_imbalance_feature_idx: int | None,
    current_imbalance_feature_idx: int | None,
    voltage_imbalance_feature_idx: int | None,
) -> MAATLightningModule:
    return MAATLightningModule(
        model=model,
        k=float(maat_cfg.get("k", 3.0)),
        temperature=float(maat_cfg.get("temperature", 50.0)),
        learning_rate=float(training_cfg.get("lr", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-2)),
        gradient_clip_val=float(training_cfg.get("gradient_clip_val", 1.0)),
        max_epochs=max_epochs,
        total_steps=total_steps,
        warmup_steps=int(training_cfg.get("warmup_steps", 0)),
        min_lr_ratio=float(training_cfg.get("min_lr_ratio", 0.01)),
        score_reduction=str(maat_cfg.get("score_reduction", "max")),
        drift_feature_idx=drift_feature_idx,
        voltage_drop_feature_indices=voltage_drop_feature_indices,
        power_imbalance_feature_idx=power_imbalance_feature_idx,
        current_imbalance_feature_idx=current_imbalance_feature_idx,
        voltage_imbalance_feature_idx=voltage_imbalance_feature_idx,
    )


def _train_and_eval(
    maat_cfg: dict,
    training_cfg: dict,
    train_dl: DataLoader,
    val_dl: DataLoader,
    n_features: int,
    drift_feature_idx: int | None,
    voltage_drop_feature_indices: list[int],
    power_imbalance_feature_idx: int | None,
    current_imbalance_feature_idx: int | None,
    voltage_imbalance_feature_idx: int | None,
    max_epochs: int,
    patience: int,
    total_steps: int,
    seed: int,
    pruning_warmup_epochs: int = 3,
    ckpt_dir: Path | None = None,
    trial: optuna.Trial | None = None,
) -> tuple[MAATLightningModule, str | None]:
    pl.seed_everything(seed, workers=True)

    model = _build_model(maat_cfg, n_features)
    lit = _build_lightning_module(
        model=model,
        maat_cfg=maat_cfg,
        training_cfg=training_cfg,
        max_epochs=max_epochs,
        total_steps=total_steps,
        drift_feature_idx=drift_feature_idx,
        voltage_drop_feature_indices=voltage_drop_feature_indices,
        power_imbalance_feature_idx=power_imbalance_feature_idx,
        current_imbalance_feature_idx=current_imbalance_feature_idx,
        voltage_imbalance_feature_idx=voltage_imbalance_feature_idx,
    )

    ckpt_callback: ModelCheckpoint | None = None
    callbacks: list = [
        EarlyStopping(monitor="val_selection_score", patience=patience, mode="max", verbose=False),
    ]
    if ckpt_dir is not None:
        ckpt_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor="val_selection_score",
            mode="max",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(ckpt_callback)
    if trial is not None:
        callbacks.append(
            _OptunaPruningCallback(
                trial,
                warmup_epochs=pruning_warmup_epochs,
                monitor="val_selection_score",
            )
        )

    trainer = pl.Trainer(
        min_epochs=max(10, pruning_warmup_epochs + 5),
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=(ckpt_dir is not None),
        logger=False,
        deterministic=False,
        benchmark=True,
        **_trainer_runtime_kwargs(training_cfg),
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
    drift_feature_name = "pdc_temp_corrected_norm_irr"
    drift_feature_idx: int | None = None
    if drift_feature_name in features:
        drift_feature_idx = int(features.index(drift_feature_name))
        logger.info(f"Drift channel enabled on feature: {drift_feature_name} (idx={drift_feature_idx})")
    else:
        logger.info(f"Drift channel disabled: missing feature '{drift_feature_name}' in manifest")

    voltage_drop_feature_indices: list[int] = []
    for voltage_col in ("vdc1", "vdc2"):
        if voltage_col in features:
            voltage_drop_feature_indices.append(int(features.index(voltage_col)))
    if voltage_drop_feature_indices:
        logger.info(
            "Voltage-drop channel enabled on features: {}",
            [features[idx] for idx in voltage_drop_feature_indices],
        )
    else:
        logger.info("Voltage-drop channel disabled: missing both 'vdc1' and 'vdc2' in manifest")

    power_imbalance_feature_idx: int | None = None
    current_imbalance_feature_idx: int | None = None
    voltage_imbalance_feature_idx: int | None = None
    if "power_imbalance" in features:
        power_imbalance_feature_idx = int(features.index("power_imbalance"))
    if "current_imbalance" in features:
        current_imbalance_feature_idx = int(features.index("current_imbalance"))
    if "voltage_imbalance" in features:
        voltage_imbalance_feature_idx = int(features.index("voltage_imbalance"))

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

    loader_runtime = _resolve_loader_runtime(
        training_cfg.get("threading", {}), hpo_mode=run_hpo
    )
    logger.info(
        "DataLoader runtime | workers={} pin_memory={} persistent_workers={} prefetch_factor={}",
        loader_runtime["num_workers"],
        loader_runtime["pin_memory"],
        loader_runtime["persistent_workers"],
        loader_runtime["prefetch_factor"],
    )

    def _make_dataloaders(cfg: dict, stride_train: int, stride_eval: int, bs: int) -> tuple:
        ds_train = TimeSeriesDataset(
            train_df, features, label_col, cfg["win_size"], stride_train, normal_only=True
        )
        ds_val = TimeSeriesDataset(
            val_df, features, label_col, cfg["win_size"], stride_eval,
            normal_only=False, return_original_label=True, return_group_id=True,
        )
        ds_test = TimeSeriesDataset(
            test_df, features, label_col, cfg["win_size"], stride_eval,
            normal_only=False, return_original_label=True, return_group_id=True,
        )
        dl_common = {
            "drop_last": False,
            "num_workers": loader_runtime["num_workers"],
            "pin_memory": loader_runtime["pin_memory"],
            "persistent_workers": loader_runtime["persistent_workers"],
        }
        if loader_runtime["num_workers"] > 0:
            dl_common["prefetch_factor"] = loader_runtime["prefetch_factor"]

        train_dl = DataLoader(ds_train, batch_size=bs, shuffle=True, **dl_common)
        val_dl = DataLoader(ds_val, batch_size=bs, shuffle=False, **dl_common)
        test_dl = DataLoader(ds_test, batch_size=bs, shuffle=False, **dl_common)
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
        hpo_epochs = max(
            int(hpo_cfg.get("min_hpo_epochs", 10)),
            int(max_epochs * float(hpo_cfg.get("max_epochs_fraction", 0.25))),
        )
        hpo_patience = max(
            int(hpo_cfg.get("min_hpo_patience", 5)),
            int(patience * float(hpo_cfg.get("patience_fraction", 0.33))),
        )
        pruning_warmup_epochs = int(hpo_cfg.get("pruning_warmup_epochs", 3))

        fixed_arch = {
            k: maat_cfg[k]
            for k in ("win_size", "block_size", "d_model", "n_heads", "e_layers", "d_ff")
        }

        def objective_builder(stage: HPOStageConfig, frozen_params: dict):
            def objective(trial: optuna.Trial) -> float:
                trial_p = suggest_params_from_space(trial, stage.search_space)
                merged = {**fixed_arch, **frozen_params, **trial_p}
                merged["score_reduction"] = "max"

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
                trial_stride = int(merged.get("train_stride", train_stride))

                try:
                    t_dl, v_dl, _ = _make_dataloaders(
                        trial_maat_cfg, trial_stride, eval_stride, trial_bs
                    )
                    lit, _ = _train_and_eval(
                        trial_maat_cfg, trial_training_cfg, t_dl, v_dl,
                        n_features,
                        drift_feature_idx,
                        voltage_drop_feature_indices,
                        power_imbalance_feature_idx,
                        current_imbalance_feature_idx,
                        voltage_imbalance_feature_idx,
                        hpo_epochs,
                        hpo_patience,
                        total_steps=max(1, len(t_dl) * hpo_epochs),
                        seed=seed,
                        pruning_warmup_epochs=pruning_warmup_epochs,
                        trial=trial,
                    )
                    selection_score = float(lit.best_val_selection_score)
                except optuna.exceptions.TrialPruned:
                    raise
                except Exception as exc:
                    logger.warning(f"HPO trial real exception (will prune): {type(exc).__name__}: {exc}")
                    raise optuna.TrialPruned()

                if not math.isfinite(selection_score) or selection_score <= 0.0:
                    raise optuna.TrialPruned()
                return selection_score

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
        cfg_hash = _hpo_config_fingerprint(maat_cfg, hpo_cfg, seed)
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
        logger.info(f"HPO selection-score winner: {best_params}")

        # Validation-only selection: among each stage's top-decile selection-score trials,
        # pick the simplest / most-regularized one. Fights val-overfit without ever
        # touching the test set.
        def _simplicity_score(t: optuna.trial.FrozenTrial) -> tuple:
            p = t.params
            sr_rank = {"center": 2, "mean": 1, "max": 0}.get(p.get("score_reduction", "max"), 0)
            return (
                -int(p.get("d_model", 64)),                 # smaller architecture
                -int(p.get("e_layers", 2)),
                -int(p.get("d_ff", 256)),
                -float(p.get("learning_rate", 1.0e-4)),     # lower lr (less aggressive)
                 int(p.get("batch_size", 256)),             # larger batch (more stable)
                -sr_rank,                                   # simpler score reduction
            )

        validation_selected: dict = {}
        for sr in stage_results:
            completed = [
                t for t in sr.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
                and math.isfinite(t.value)
            ]
            if not completed:
                continue
            top_n = max(1, len(completed) // 10)
            top_decile = sorted(completed, key=lambda t: t.value, reverse=True)[:top_n]
            picked = max(top_decile, key=_simplicity_score)
            logger.info(
                f"[{sr.name}] val-only select | top-decile={top_n}/{len(completed)} | "
                f"picked selection_score={picked.value:.4f} (stage max was {sr.best_value:.4f}) | "
                f"params={picked.params}"
            )
            validation_selected.update(picked.params)

        if validation_selected:
            best_params = validation_selected
        logger.info(f"HPO validation-only selected params: {best_params}")
    else:
        if not injected_params:
            best_params = {}
            logger.info("HPO skipped — using config defaults")

    # ── Merge best params into final config ────────────────────────────────────
    final_maat_cfg = dict(maat_cfg)
    final_training_cfg = dict(training_cfg)
    for k, v in best_params.items():
        if k in (
            "win_size",
            "block_size",
            "d_model",
            "n_heads",
            "e_layers",
            "d_ff",
            "dropout",
            "k",
            "temperature",
            "train_stride",
        ):
            final_maat_cfg[k] = v
        elif k == "learning_rate":
            final_training_cfg["lr"] = v
        elif k in ("weight_decay", "batch_size", "gradient_clip_val"):
            final_training_cfg[k] = v

    final_batch_size = int(final_training_cfg.get("batch_size", batch_size))
    final_maat_cfg["score_reduction"] = "max"
    final_train_stride = int(final_maat_cfg.get("train_stride", train_stride))
    train_dl, val_dl, test_dl = _make_dataloaders(
        final_maat_cfg, final_train_stride, eval_stride, final_batch_size
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
        drift_feature_idx=drift_feature_idx,
        voltage_drop_feature_indices=voltage_drop_feature_indices,
        power_imbalance_feature_idx=power_imbalance_feature_idx,
        current_imbalance_feature_idx=current_imbalance_feature_idx,
        voltage_imbalance_feature_idx=voltage_imbalance_feature_idx,
        max_epochs=max_epochs,
        patience=patience,
        total_steps=max(1, len(train_dl) * max_epochs),
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
            precision="32-true",  # scoring/PR-AUC stays fp32 regardless of training precision
        )
        trainer_val.validate(final_lit, dataloaders=val_dl, ckpt_path=best_ckpt_path)
    else:
        logger.warning("No best checkpoint found — val scores are from last training epoch.")

    logger.info(
        "val_pv_maat_f1 (best ckpt)={:.4f} | val_episode_macro_f1={:.4f} | val_selection_score={:.4f}",
        final_lit.best_val_pv_maat_f1,
        final_lit.best_val_episode_macro_f1,
        final_lit.best_val_selection_score,
    )

    # ── Test evaluation ────────────────────────────────────────────────────────
    trainer_test = pl.Trainer(
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        precision="32-true",  # scoring/PR-AUC stays fp32 regardless of training precision
    )
    trainer_test.test(
        final_lit,
        dataloaders=test_dl,
        ckpt_path=best_ckpt_path,
    )

    val_scores = final_lit._val_scores_np
    val_product_scores = final_lit._val_product_scores_np
    val_labels = final_lit._val_labels_np
    test_scores = final_lit._test_scores_np
    test_product_scores = final_lit._test_product_scores_np
    test_labels = final_lit._test_labels_np
    val_recon_scores = final_lit._val_recon_scores_np
    val_assoc_scores = final_lit._val_assoc_scores_np
    val_assoc_affinity_scores = final_lit._val_assoc_affinity_scores_np
    val_original_labels = final_lit._val_original_labels_np
    test_recon_scores = final_lit._test_recon_scores_np
    test_assoc_scores = final_lit._test_assoc_scores_np
    test_assoc_affinity_scores = final_lit._test_assoc_affinity_scores_np
    val_drift_scores = final_lit._val_drift_scores_np
    test_drift_scores = final_lit._test_drift_scores_np
    val_voltage_drop_scores = final_lit._val_voltage_drop_scores_np
    test_voltage_drop_scores = final_lit._test_voltage_drop_scores_np
    val_mismatch_scores = final_lit._val_mismatch_scores_np
    test_mismatch_scores = final_lit._test_mismatch_scores_np
    test_original_labels = final_lit._test_original_labels_np
    val_group_ids = final_lit._val_group_ids_np
    test_group_ids = final_lit._test_group_ids_np
    val_diag_scores = getattr(final_lit, "_val_diag_scores", None)
    test_diag_scores = getattr(final_lit, "_test_diag_scores", None)

    if val_scores is None or test_scores is None:
        logger.error("Score arrays not populated — test may not have run.")
        return

    threshold = final_lit.val_threshold
    val_pr_auc = _safe_pr_auc(val_labels, val_scores)
    val_roc_auc = _safe_roc_auc(val_labels, val_scores)
    val_preds = (val_scores >= threshold).astype(int)
    val_f1 = float(f1_score(val_labels, val_preds, zero_division=0))
    val_episode_macro_f1 = _episode_macro_f1_binary(val_labels, val_preds, val_group_ids)
    val_selection_score = val_f1
    val_prec = float(precision_score(val_labels, val_preds, zero_division=0))
    val_rec = float(recall_score(val_labels, val_preds, zero_division=0))
    val_acc = float(accuracy_score(val_labels, val_preds))
    product_val_pr_auc = _safe_pr_auc(val_labels, val_product_scores)

    test_pr_auc = _safe_pr_auc(test_labels, test_scores)
    test_roc_auc = _safe_roc_auc(test_labels, test_scores)
    test_preds = (test_scores >= threshold).astype(int)
    test_f1 = float(f1_score(test_labels, test_preds, zero_division=0))
    test_episode_macro_f1 = _episode_macro_f1_binary(test_labels, test_preds, test_group_ids)
    test_acc = float(accuracy_score(test_labels, test_preds))
    test_prec = float(precision_score(test_labels, test_preds, zero_division=0))
    test_rec = float(recall_score(test_labels, test_preds, zero_division=0))
    product_test_pr_auc = _safe_pr_auc(test_labels, test_product_scores)

    logger.info(
        f"Test — PR-AUC={test_pr_auc:.4f}  ROC-AUC={test_roc_auc:.4f}  "
        f"F1={test_f1:.4f}  Prec={test_prec:.4f}  Rec={test_rec:.4f}"
    )

    _val_n_pos = int(val_labels.sum()) if val_labels is not None else 0
    _threshold_method = (
        "smoothed_argmax_5"
        if _val_n_pos >= 50 and len(np.unique(val_scores)) >= 7
        else "raw_argmax"
    )

    metrics: dict = {
        "score_name": "pv_maat_score",
        "score_fusion": "logsumexp_z",
        "score_components": ["product", "dc_voltage_drop", "mismatch_imbalance"],
        "score_tau": 1.0,
        "threshold_policy": "validation_pr_curve_f1",
        "selection_metric": "val_pv_maat_f1",
        "official_score_reduction": "max",
        "score_reduction_scope": "all_components_temporal",
        "diagnostic_score_reductions": ["q95", "top10mean"],
        "val_positive_count": _val_n_pos,
        "threshold_selection_method": _threshold_method,
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "val_pv_maat_f1": val_f1,
        "val_pv_maat_accuracy": val_acc,
        "val_pv_maat_precision": val_prec,
        "val_pv_maat_recall": val_rec,
        "val_episode_macro_f1": val_episode_macro_f1,
        "val_selection_score": val_selection_score,
        "threshold": threshold,
        "product_val_pr_auc": product_val_pr_auc,
        "test_pv_maat_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "product_test_pr_auc": product_test_pr_auc,
        "test_pv_maat_f1": test_f1,
        "test_pv_maat_accuracy": test_acc,
        "test_pv_maat_precision": test_prec,
        "test_pv_maat_recall": test_rec,
        "test_episode_macro_f1": test_episode_macro_f1,
        "n_train_windows": len(train_dl.dataset),
        "n_features": n_features,
        "fit_time_s": round(fit_time, 2),
        "nan_count": final_lit._nan_count,
    }
    run_name = f"anomaly_maat_{ts}"

    # ── Save artifacts ──────────────────────────────────────────────────────────
    global_metrics_path = artifacts_dir / "global_metrics.json"
    per_class_metrics_path = artifacts_dir / "per_class_metrics.json"
    run_manifest_path = artifacts_dir / "run_manifest.json"
    deployment_manifest_path = artifacts_dir / "deployment_manifest.json"
    scaler_path = artifacts_dir / "scaler.joblib"
    params_path = artifacts_dir / "best_params.json"
    config_path = artifacts_dir / "resolved_config.json"
    pr_curve_path = artifacts_dir / "pr_curve.png"
    histogram_path = artifacts_dir / "score_histogram.png"
    timeline_path = artifacts_dir / "score_timeline.png"
    manifest_path = artifacts_dir / "features_manifest.json"
    decomposition_path = artifacts_dir / "score_decomposition_metrics.json"
    calibration_path = artifacts_dir / "score_calibration.json"
    diag_reductions_path = artifacts_dir / "diagnostic_score_reductions.json"

    write_json(global_metrics_path, metrics)
    joblib.dump(scaler, scaler_path)
    params_path.write_text(json.dumps(best_params, indent=2, default=str), encoding="utf-8")
    config_path.write_text(
        json.dumps({"maat": final_maat_cfg, "training": final_training_cfg}, indent=2, default=str),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    calibration_payload = {
        **build_score_calibration_payload(
            threshold=threshold,
            threshold_policy="validation_pr_curve_f1",
            threshold_quantile=None,
        ),
        "score_name": "pv_maat_score",
        "score_fusion": "logsumexp_z",
        "score_components": ["product", "dc_voltage_drop", "mismatch_imbalance"],
        "official_score_reduction": "max",
        "score_reduction_scope": "all_components_temporal",
        "pv_score_stats": final_lit._pv_score_stats,
        "diag_score_stats": final_lit._diag_score_stats,
    }
    write_json(calibration_path, calibration_payload)
    per_class_labels = test_original_labels if test_original_labels is not None else test_labels
    per_class_metrics = compute_anomaly_per_class_metrics(
        labels=per_class_labels,
        scores=test_scores,
        threshold=threshold,
    )
    run_manifest = build_run_manifest(
        task=args.task,
        model="maat",
        model_family="anomaly_dl",
        dataset=args.dataset,
        split_path=args.split_path,
        feature_profile=str(args.profile),
        feature_run_dir=str(resolved_run_dir),
        seed=seed,
        run_type=run_type,
        extras={"run_name": run_name},
    )
    model_artifact_rel = "checkpoints/best.ckpt" if best_ckpt_path else "checkpoints/last.ckpt"
    deployment_manifest = build_deployment_manifest(
        task=args.task,
        model="maat",
        model_family="anomaly_dl",
        model_artifact=model_artifact_rel,
        scaler_artifact=scaler_path.name,
        feature_names=features,
        label_column=label_col,
        threshold=threshold,
        window_size=int(final_maat_cfg["win_size"]),
        score_direction="higher_is_more_anomalous",
        classes=[str(c) for c in sorted(np.unique(per_class_labels).tolist())],
        extras={
            "checkpoint_available": bool(best_ckpt_path),
            "score_name": "pv_maat_score",
            "score_fusion": "logsumexp_z",
            "score_components": ["product", "dc_voltage_drop", "mismatch_imbalance"],
            "score_tau": 1.0,
            "threshold_policy": "validation_pr_curve_f1",
            "selection_metric": "val_pv_maat_f1",
            "official_score_reduction": "max",
            "score_reduction_scope": "all_components_temporal",
            "diagnostic_score_reductions": ["q95", "top10mean"],
            "score_calibration_artifact": calibration_path.name,
        },
    )
    write_json(per_class_metrics_path, per_class_metrics)
    write_json(run_manifest_path, run_manifest)
    write_json(deployment_manifest_path, deployment_manifest)

    _save_pr_curve(val_scores, val_labels, test_scores, test_labels, pr_curve_path)
    _save_score_histogram(val_scores, val_labels, threshold, histogram_path)
    _save_score_timeline(test_scores, test_labels, threshold, timeline_path)

    decomposition_report = None
    if all(
        x is not None for x in (
            val_recon_scores,
            val_assoc_scores,
            val_assoc_affinity_scores,
            val_original_labels,
            test_recon_scores,
            test_assoc_scores,
            test_assoc_affinity_scores,
            val_mismatch_scores,
            test_mismatch_scores,
            val_voltage_drop_scores,
            test_voltage_drop_scores,
            test_original_labels,
        )
    ):
        val_parts = {
            "product": val_product_scores,
            "reconstruction": val_recon_scores,
            "association_discrepancy": val_assoc_scores,
            "association_affinity": val_assoc_affinity_scores,
            "dc_voltage_drop": val_voltage_drop_scores,
            "mismatch_imbalance": val_mismatch_scores,
        }
        test_parts = {
            "product": test_product_scores,
            "reconstruction": test_recon_scores,
            "association_discrepancy": test_assoc_scores,
            "association_affinity": test_assoc_affinity_scores,
            "dc_voltage_drop": test_voltage_drop_scores,
            "mismatch_imbalance": test_mismatch_scores,
        }
        if val_drift_scores is not None and test_drift_scores is not None:
            val_parts["drift_efficiency"] = val_drift_scores
            test_parts["drift_efficiency"] = test_drift_scores

        decomposition_report, _decomp_best = _build_score_decomposition_report(
            val_parts=val_parts,
            test_parts=test_parts,
            val_labels=val_labels,
            test_labels=test_labels,
            val_original_labels=val_original_labels,
            test_original_labels=test_original_labels,
        )
        decomposition_path.write_text(
            json.dumps(decomposition_report, indent=2, default=str), encoding="utf-8"
        )
        max_recon_product = decomposition_report["scores"]["max_z_recon_product"]
        logger.info(
            "Score decomposition — max_z_recon_product test_pr_auc={:.4f} test_macro_fault_pr_auc={:.4f}",
            max_recon_product["test"]["pr_auc"], max_recon_product["test_macro_fault_pr_auc"],
        )

        # If any decomposition variant has a better val F1 than the current headline
        # (pv_maat_score), override the headline with that variant. Selection is
        # purely val-based, so benchmark protocol remains intact.
        if _decomp_best and _decomp_best["val_f1"] > val_f1:
            _alt_val_s = _decomp_best["val_scores"]
            _alt_test_s = _decomp_best["test_scores"]
            _alt_thr = _decomp_best["threshold"]
            _alt_name = _decomp_best["name"]
            _alt_val_preds = (_alt_val_s >= _alt_thr).astype(int)
            _alt_test_preds = (_alt_test_s >= _alt_thr).astype(int)

            # Resolve fusion method, component names, and tau from the winner name.
            _alt_fusion_method, _alt_components, _alt_tau = _FUSION_METADATA.get(
                _alt_name, ("unknown", [_alt_name], None)
            )
            # Per-component z-score stats needed to reproduce the alt fusion at deploy time.
            # Std fallback matches _zscore_from_val exactly: <= 1e-12 or non-finite → 1.0.
            def _safe_std(arr: np.ndarray) -> float:
                s = float(np.std(arr))
                return s if s > 1e-12 and math.isfinite(s) else 1.0

            _alt_zscore_stats = {
                comp: {
                    "mean": float(np.mean(val_parts[comp])),
                    "std": _safe_std(val_parts[comp]),
                }
                for comp in _alt_components
                if comp in val_parts
            }
            # Threshold method for the alt score.
            # len(f1_vals) in _calibrate_threshold equals len(np.unique(scores)) (sklearn
            # precision_recall_curve appends a trailing precision=1/recall=0 point, so
            # len(prec[:-1]) == len(thresholds) == len(np.unique(scores))).
            _alt_threshold_method = (
                "smoothed_argmax_5"
                if _val_n_pos >= 50 and len(np.unique(_alt_val_s)) >= 7
                else "raw_argmax"
            )

            logger.info(
                f"Decomp-best fusion '{_alt_name}' val_f1={_decomp_best['val_f1']:.4f} "
                f"> pv_maat val_f1={val_f1:.4f} — overriding headline metrics."
            )

            val_f1 = _decomp_best["val_f1"]
            val_prec = float(precision_score(val_labels, _alt_val_preds, zero_division=0))
            val_rec = float(recall_score(val_labels, _alt_val_preds, zero_division=0))
            val_acc = float(accuracy_score(val_labels, _alt_val_preds))
            val_episode_macro_f1 = _episode_macro_f1_binary(val_labels, _alt_val_preds, val_group_ids)
            val_pr_auc = _safe_pr_auc(val_labels, _alt_val_s)
            val_roc_auc = _safe_roc_auc(val_labels, _alt_val_s)
            val_selection_score = val_f1
            threshold = _alt_thr

            test_f1 = float(f1_score(test_labels, _alt_test_preds, zero_division=0))
            test_pr_auc = _safe_pr_auc(test_labels, _alt_test_s)
            test_roc_auc = _safe_roc_auc(test_labels, _alt_test_s)
            test_prec = float(precision_score(test_labels, _alt_test_preds, zero_division=0))
            test_rec = float(recall_score(test_labels, _alt_test_preds, zero_division=0))
            test_acc = float(accuracy_score(test_labels, _alt_test_preds))
            test_episode_macro_f1 = _episode_macro_f1_binary(
                test_labels, _alt_test_preds, test_group_ids
            )

            for _k, _v in {
                "score_name": _alt_name, "headline_fusion": _alt_name,
                "score_fusion": _alt_fusion_method,
                "score_components": _alt_components,
                "score_tau": _alt_tau,
                "threshold_selection_method": _alt_threshold_method,
                "val_pr_auc": val_pr_auc, "val_roc_auc": val_roc_auc,
                "val_pv_maat_f1": val_f1, "val_pv_maat_accuracy": val_acc,
                "val_pv_maat_precision": val_prec, "val_pv_maat_recall": val_rec,
                "val_episode_macro_f1": val_episode_macro_f1, "val_selection_score": val_selection_score,
                "threshold": threshold,
                "test_pv_maat_pr_auc": test_pr_auc, "test_roc_auc": test_roc_auc,
                "test_pv_maat_f1": test_f1, "test_pv_maat_accuracy": test_acc,
                "test_pv_maat_precision": test_prec, "test_pv_maat_recall": test_rec,
                "test_episode_macro_f1": test_episode_macro_f1,
            }.items():
                metrics[_k] = _v
            write_json(global_metrics_path, metrics)

            # Re-generate all reporting artifacts with the winning fusion.
            _alt_per_class_labels = (
                test_original_labels if test_original_labels is not None else test_labels
            )
            write_json(
                per_class_metrics_path,
                compute_anomaly_per_class_metrics(
                    labels=_alt_per_class_labels, scores=_alt_test_s, threshold=_alt_thr,
                ),
            )
            _save_pr_curve(_alt_val_s, val_labels, _alt_test_s, test_labels, pr_curve_path)
            _save_score_histogram(_alt_val_s, val_labels, _alt_thr, histogram_path)
            _save_score_timeline(_alt_test_s, test_labels, _alt_thr, timeline_path)

            # Calibration: update threshold + record headline fusion. pv_score_stats
            # is retained as-is (it documents the base model's scoring pipeline).
            # alt_fusion_zscore_stats carries per-component (mean, std) from val so
            # the alt fusion can be reproduced exactly at inference without re-fitting.
            write_json(calibration_path, {
                **build_score_calibration_payload(
                    threshold=_alt_thr,
                    threshold_policy="validation_pr_curve_f1",
                    threshold_quantile=None,
                ),
                "score_name": _alt_name,
                "score_fusion": _alt_fusion_method,
                "score_components": _alt_components,
                "score_tau": _alt_tau,
                "official_score_reduction": "max",
                "score_reduction_scope": "all_components_temporal",
                "headline_fusion": _alt_name,
                "headline_threshold": _alt_thr,
                "alt_fusion_zscore_stats": _alt_zscore_stats,
                "pv_score_stats": final_lit._pv_score_stats,
                "diag_score_stats": final_lit._diag_score_stats,
            })

            # Deployment manifest: update threshold and score identity.
            write_json(deployment_manifest_path, build_deployment_manifest(
                task=args.task,
                model="maat",
                model_family="anomaly_dl",
                model_artifact=model_artifact_rel,
                scaler_artifact=scaler_path.name,
                feature_names=features,
                label_column=label_col,
                threshold=_alt_thr,
                window_size=int(final_maat_cfg["win_size"]),
                score_direction="higher_is_more_anomalous",
                classes=[str(c) for c in sorted(np.unique(_alt_per_class_labels).tolist())],
                extras={
                    "checkpoint_available": bool(best_ckpt_path),
                    "score_name": _alt_name,
                    "score_fusion": _alt_fusion_method,
                    "score_components": _alt_components,
                    "score_tau": _alt_tau,
                    "alt_fusion_zscore_stats": _alt_zscore_stats,
                    "threshold_policy": "validation_pr_curve_f1",
                    "selection_metric": "val_pv_maat_f1",
                    "official_score_reduction": "max",
                    "score_reduction_scope": "all_components_temporal",
                    "score_calibration_artifact": calibration_path.name,
                },
            ))

    if stage_results:
        for sr in stage_results:
            trials_df = sr.study.trials_dataframe()
            trials_df.to_csv(artifacts_dir / f"hpo_trials_{sr.name}.csv", index=False)

    if val_diag_scores is not None and test_diag_scores is not None:
        diag_report: dict[str, dict[str, float]] = {}
        for name in ("q95", "top10mean"):
            val_s = val_diag_scores[name]
            test_s = test_diag_scores[name]
            normal_val = val_s[val_labels == 0]
            diag_thr = float(np.quantile(normal_val, 0.95)) if len(normal_val) else float(np.quantile(val_s, 0.95))
            diag_report[name] = {
                "threshold": diag_thr,
                "val_f1": float(f1_score(val_labels, (val_s >= diag_thr).astype(int), zero_division=0)),
                "test_f1": float(f1_score(test_labels, (test_s >= diag_thr).astype(int), zero_division=0)),
                "val_pr_auc": _safe_pr_auc(val_labels, val_s),
                "test_pr_auc": _safe_pr_auc(test_labels, test_s),
            }
        write_json(diag_reductions_path, diag_report)

    logger.info(f"Artifacts saved → {artifacts_dir}")

    # ── MLflow ──────────────────────────────────────────────────────────────────
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
            for p in (
                global_metrics_path,
                per_class_metrics_path,
                run_manifest_path,
                deployment_manifest_path,
                scaler_path,
                params_path,
                config_path,
                pr_curve_path,
                histogram_path,
                timeline_path,
                manifest_path,
                decomposition_path,
                calibration_path,
                diag_reductions_path,
            ):
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
                "score_name": metrics.get("score_name", "pv_maat_score"),
                "score_fusion": metrics.get("score_fusion", "logsumexp_z"),
                "score_components": metrics.get("score_components", ["product", "dc_voltage_drop", "mismatch_imbalance"]),
                "score_tau": metrics.get("score_tau", 1.0),
                "headline_fusion": metrics.get("headline_fusion", "pv_maat_score"),
                "threshold_policy": "validation_pr_curve_f1",
                "selection_metric": "val_pv_maat_f1",
                "score_reduction_scope": "all_components_temporal",
                "threshold": threshold,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "val_pv_maat_f1": val_f1,
                "val_episode_macro_f1": val_episode_macro_f1,
                "val_pv_maat_accuracy": val_acc,
                "test_pv_maat_pr_auc": test_pr_auc,
                "test_roc_auc": test_roc_auc,
                "test_pv_maat_f1": test_f1,
                "test_episode_macro_f1": test_episode_macro_f1,
                "test_pv_maat_accuracy": test_acc,
                "test_pv_maat_precision": test_prec,
                "test_pv_maat_recall": test_rec,
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

    logger.info(
        "Primary MAAT result — test_f1={:.4f} | secondary test_pr_auc={:.4f}",
        test_f1,
        test_pr_auc,
    )


if __name__ == "__main__":
    run_maat()
