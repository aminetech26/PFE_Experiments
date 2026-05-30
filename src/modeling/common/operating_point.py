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
  - ``conformal``           : conformal p-value calibration at level alpha —
                              distribution-free, finite-sample guarantee
                              P(flag|normal) <= alpha (no GPD tail-shape assumption).
  - ``fdr_bh``              : Benjamini-Hochberg on conformal p-values — controls
                              the false-discovery rate among alarms <= q
                              (Bates et al. 2023, Annals of Statistics).
  - ``cusum``               : one-sided CUSUM (Page 1954), minimax-optimal sequential
                              change detector (Lorden 1971) — principled replacement
                              for the ad-hoc hysteresis temporal layer.

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


def conformal_pvalues(
    calib_normal_scores: np.ndarray, test_scores: np.ndarray
) -> np.ndarray:
    """Marginal (inductive) conformal p-values for anomaly scores.

    p(s) = (1 + #{calib_normal >= s}) / (n_calib + 1), higher score = more
    anomalous. Distribution-free, finite-sample guarantee: under exchangeability
    of normal data, P(p(s) <= alpha | normal) <= alpha — no tail-shape assumption
    (unlike GPD). Calibrated on held-out val-normal scores.
    """
    calib = np.sort(np.asarray(calib_normal_scores, dtype=float))
    test = np.asarray(test_scores, dtype=float)
    n = len(calib)
    if n == 0:
        return np.ones_like(test)
    # #{calib >= s} = n - (index of first calib element >= s)
    ge = n - np.searchsorted(calib, test, side="left")
    return (1.0 + ge) / (n + 1.0)


def _bh_reject(pvals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg reject mask at FDR level q (boolean array).

    Controls the false-discovery rate among flagged points. Conformal p-values
    are PRDS under exchangeability, so BH gives valid finite-sample FDR control
    (Bates et al. 2023, Annals of Statistics).
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    line = q * (np.arange(1, m + 1) / m)
    below = np.where(sorted_p <= line)[0]
    reject = np.zeros(m, dtype=bool)
    if below.size:
        k = below.max()  # largest index passing the BH line
        reject[order[: k + 1]] = True
    return reject


def _cusum_calibrate_h(
    calib_normal_scores: np.ndarray, mu: float, sigma: float, k: float, target_fpr: float
) -> float:
    """Calibrate the CUSUM control limit h on normal scores to a target FPR.

    Run the one-sided CUSUM recursion over standardized normal scores and take the
    (1 - target_fpr) quantile of the statistic, so P(S_t > h | normal) ≈ target_fpr.
    Blindfolded (normal-only).
    """
    z = (np.asarray(calib_normal_scores, dtype=float) - mu) / (sigma if sigma > 0 else 1.0)
    s = 0.0
    stats = np.empty(len(z))
    for i in range(len(z)):
        s = max(0.0, s + z[i] - k)
        stats[i] = s
    return float(np.quantile(stats, 1.0 - target_fpr)) if len(z) else float("inf")


def _cusum_predict(
    scores: np.ndarray, group_ids: np.ndarray | None, mu: float, sigma: float, k: float, h: float
) -> np.ndarray:
    """One-sided upper CUSUM (Page 1954), reset per segment.

    z_t = (x_t - mu)/sigma ; S_t = max(0, S_{t-1} + z_t - k) ; alarm when S_t > h.
    Minimax-optimal sequential change detector (Lorden 1971; Moustakides 1986) —
    the principled replacement for ad-hoc N-consecutive hysteresis. Accumulates
    evidence of a sustained upward shift in the anomaly score.
    """
    scores = np.asarray(scores, dtype=float)
    z = (scores - mu) / (sigma if sigma > 0 else 1.0)
    out = np.zeros(len(scores), dtype=np.int8)
    if group_ids is None:
        group_ids = np.zeros(len(scores), dtype=int)
    groups: dict = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    for idxs in groups.values():
        s = 0.0
        for i in idxs:  # temporal order within segment
            s = max(0.0, s + z[i] - k)
            out[i] = 1 if s > h else 0
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


def _episode_decision_metrics(
    preds: np.ndarray, labels: np.ndarray, group_ids: np.ndarray | None
) -> dict:
    """Event-level RECALL: fraction of fault episodes the detector fires on (>=1 sample).

    The operator-meaningful "did the dashboard alarm at some point during this fault
    event" — higher than per-sample recall for a sequential detector (it confirms the
    event even if it misses onset samples).

    Deliberately reports ONLY recall, not precision/F1. Point-adjusted *precision* is
    pathological: a single stray FP sample condemns a whole normal episode, so even a
    0.9998-sample-precision detector shows ~0.5 episode precision (Kim et al. 2022).
    Use the per-sample precision from `_decision_metrics` for the precision story.
    """
    if group_ids is None:
        return {}
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    groups: dict = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    n_fault = 0
    n_fault_detected = 0
    for idxs in groups.values():
        idxs_a = np.asarray(idxs)
        if labels[idxs_a].max() > 0:                    # fault episode
            n_fault += 1
            if preds[idxs_a].max() > 0:                 # alarmed at least once during it
                n_fault_detected += 1
    return {
        "episode_recall": float(n_fault_detected / n_fault) if n_fault else 0.0,
        "n_fault_episodes": int(n_fault),
        "n_episodes": int(len(groups)),
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
    conformal_alpha: float = 0.05,
    fdr_q: float = 0.10,
    op_cfg: dict | None = None,
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

    # Conformal calibration on held-out normal scores (distribution-free guarantee).
    pvals = conformal_pvalues(calib_normal_scores, test_scores)
    conf_pred = (pvals <= conformal_alpha).astype(np.int8)   # P(flag|normal) <= alpha
    fdr_pred = _bh_reject(pvals, fdr_q).astype(np.int8)       # FDR among alarms <= q

    # CUSUM (sequential, per-segment) — principled temporal layer.
    _cfg = op_cfg or {}
    cusum_k = float(_cfg.get("cusum_k", 0.5))
    _mu = float(np.mean(calib_normal_scores)) if np.asarray(calib_normal_scores).size else 0.0
    _sigma = float(np.std(calib_normal_scores)) if np.asarray(calib_normal_scores).size else 1.0
    _h = _cusum_calibrate_h(calib_normal_scores, _mu, _sigma, cusum_k, baseline_fpr)
    cusum_pred = _cusum_predict(test_scores, test_group_ids, _mu, _sigma, cusum_k, _h)

    return {
        "gpd_baseline": {
            "threshold": float(t_base),
            "target_fpr": float(baseline_fpr),
            "policy": f"gpd_pot{pot_quantile:g}_fpr{baseline_fpr:g}",
            **_decision_metrics(test_labels, base_pred),
            **_episode_decision_metrics(base_pred, test_labels, test_group_ids),
        },
        "sensitive": {
            "threshold": float(t_sens),
            "target_fpr": float(sensitive_fpr),
            "policy": f"gpd_pot{pot_quantile:g}_fpr{sensitive_fpr:g}",
            **_decision_metrics(test_labels, sens_pred),
            **_episode_decision_metrics(sens_pred, test_labels, test_group_ids),
        },
        "sensitive_hysteresis": {
            "threshold": float(t_sens),
            "target_fpr": float(sensitive_fpr),
            "hysteresis_n": int(hysteresis_n),
            "detection_latency_samples": int(max(0, hysteresis_n - 1)),
            "policy": f"gpd_pot{pot_quantile:g}_fpr{sensitive_fpr:g}_hyst{hysteresis_n}",
            **_decision_metrics(test_labels, deb_pred),
            **_episode_decision_metrics(deb_pred, test_labels, test_group_ids),
        },
        "conformal": {
            "alpha": float(conformal_alpha),
            "n_calib": int(np.asarray(calib_normal_scores).size),
            "policy": f"conformal_p<={conformal_alpha:g}",  # distribution-free FPR<=alpha
            **_decision_metrics(test_labels, conf_pred),
            **_episode_decision_metrics(conf_pred, test_labels, test_group_ids),
        },
        "fdr_bh": {
            "q": float(fdr_q),
            "policy": f"bh_fdr<={fdr_q:g}",  # FDR among alarms <= q (Bates et al. 2023)
            **_decision_metrics(test_labels, fdr_pred),
            **_episode_decision_metrics(fdr_pred, test_labels, test_group_ids),
        },
        "cusum": {
            "k": float(cusum_k),
            "h": float(_h),
            "policy": f"cusum_k{cusum_k:g}_fpr{baseline_fpr:g}",  # Page/Lorden optimal sequential
            **_decision_metrics(test_labels, cusum_pred),
            **_episode_decision_metrics(cusum_pred, test_labels, test_group_ids),
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
