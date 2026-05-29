"""Uniform operating-point calibration + temporal hysteresis for Task A detectors.

Every anomaly detector (DL and ML) calls `compute_operating_points` with its
per-sample scores so the reported *decision* metrics are produced identically and
the cross-model comparison stays apples-to-apples.

Reported operating points (all thresholds calibrated on TRAIN normal scores only
— blindfolded, no fault labels, consistent with the semi-supervised setting):

  - ``gpd_baseline``        : GPD-POT at a conservative FPR (default 5%). The
                              standard, conservative reference.
  - ``sensitive``           : GPD-POT at a loose FPR (default 20%). Deliberately
                              sensitive — recovers recall.
  - ``sensitive_hysteresis``: the sensitive threshold + N-consecutive confirmation
                              (temporal debounce). Hysteresis claws back the
                              precision a sensitive threshold gives up.

The sensitive FPR and the hysteresis N are *a-priori design choices* (N is a
detection-latency budget), never tuned on test. Threshold-free PR-AUC remains the
primary metric; these are the operating-point F1/precision/recall numbers.

Design rationale for the system: hysteresis suppresses the transient false alarms
a sensitive per-sample threshold admits, so the detector can run a more sensitive
cutoff than a stand-alone threshold safely could — recovering recall while the
N-consecutive rule restores precision.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.modeling.common.threshold_calibration import calibrate_threshold


def apply_persistence_filter(
    preds: np.ndarray, group_ids: np.ndarray | None, n_consecutive: int
) -> np.ndarray:
    """Debounce per-sample alarms with an N-consecutive rule (canonical home).

    A sample is flagged only if it and the preceding (n_consecutive - 1) samples
    *within the same segment* are all raw alarms. Segment boundaries reset the run
    counter so persistence never spans two episodes. Delays detection by
    (n_consecutive - 1) samples. Blindfolded: uses no fault labels.

    ``n_consecutive <= 1`` (or missing group_ids) returns the raw alarms.
    """
    preds = np.asarray(preds).astype(np.int8)
    if n_consecutive <= 1 or group_ids is None:
        return preds.copy()
    group_ids = np.asarray(group_ids)
    out = np.zeros_like(preds)
    groups: dict = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    for idxs in groups.values():
        run = 0
        for i in idxs:  # appearance order == temporal order within a segment
            run = run + 1 if preds[i] == 1 else 0
            out[i] = 1 if run >= n_consecutive else 0
    return out


def _decision_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    labels = np.asarray(labels).astype(np.int8)
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "alarm_rate": float(np.mean(preds)),
    }


def compute_operating_points(
    *,
    calib_normal_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    test_group_ids: np.ndarray | None,
    pot_quantile: float = 0.90,
    baseline_fpr: float = 0.05,
    sensitive_fpr: float = 0.20,
    hysteresis_n: int = 10,
) -> dict:
    """Compute the three uniform operating points from held-out normal scores.

    Calibration uses VAL-normal scores (held out from training) — identical
    source across every detector. Val-normal is preferred over train-normal
    because the model is fit to minimize error on train-normal, so train-normal
    scores are optimistically low and would bias the threshold toward false
    alarms on unseen normal. Held-out calibration removes that bias.

    Args:
        calib_normal_scores: anomaly scores on held-out (val) NORMAL samples.
        test_labels:    binary test labels (1 = fault).
        test_scores:    anomaly scores on test.
        test_group_ids: per-sample segment ids (temporal order within segment)
                        for the hysteresis filter; None disables hysteresis.
        pot_quantile:   GPD peaks-over-threshold anchor quantile.
        baseline_fpr / sensitive_fpr: target normal-exceedance rates.
        hysteresis_n:   N-consecutive confirmation (latency budget, samples).

    Returns a dict with one entry per operating point, each carrying its
    threshold and decision metrics, plus the policy strings.
    """
    calib_normal_scores = np.asarray(calib_normal_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)

    t_base, _ = calibrate_threshold(
        calib_normal_scores, strategy="gpd", pot_quantile=pot_quantile, target_tail_prob=baseline_fpr
    )
    t_sens, _ = calibrate_threshold(
        calib_normal_scores, strategy="gpd", pot_quantile=pot_quantile, target_tail_prob=sensitive_fpr
    )

    base_pred = (test_scores >= t_base).astype(np.int8)
    sens_pred = (test_scores >= t_sens).astype(np.int8)
    deb_pred = apply_persistence_filter(sens_pred, test_group_ids, hysteresis_n)

    return {
        "gpd_baseline": {
            "threshold": float(t_base),
            "target_fpr": float(baseline_fpr),
            "policy": f"gpd_pot{pot_quantile:g}_fpr{baseline_fpr:g}",
            **_decision_metrics(test_labels, base_pred),
        },
        "sensitive": {
            "threshold": float(t_sens),
            "target_fpr": float(sensitive_fpr),
            "policy": f"gpd_pot{pot_quantile:g}_fpr{sensitive_fpr:g}",
            **_decision_metrics(test_labels, sens_pred),
        },
        "sensitive_hysteresis": {
            "threshold": float(t_sens),
            "target_fpr": float(sensitive_fpr),
            "hysteresis_n": int(hysteresis_n),
            "detection_latency_samples": int(max(0, hysteresis_n - 1)),
            "policy": f"gpd_pot{pot_quantile:g}_fpr{sensitive_fpr:g}_hyst{hysteresis_n}",
            **_decision_metrics(test_labels, deb_pred),
        },
    }


def flatten_operating_points(ops: dict, split: str = "test") -> dict:
    """Flatten the nested operating-point dict to scalar metrics for MLflow.

    Keys look like ``test_op_sensitive_hysteresis_f1``.
    """
    flat: dict = {}
    for name, payload in ops.items():
        for k, v in payload.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                flat[f"{split}_op_{name}_{k}"] = float(v)
    return flat
