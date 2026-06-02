"""Deployment-characterization metrics for the winning Task A detector.

This is *not* a model-selection metric (that is threshold-free macro/worst per-class
PR-AUC, reported in the tournament table). This module produces the thesis
"deployment" table that characterizes the **single deployed escalating alarm
system** at each of its design tiers — the system is not a set of competing
classifiers, so the tiers are described, never averaged.

Tiers (ISA-18.2 three-level escalation):
    P3 Advisory  — GPD/sensitive + hysteresis : first alarm, recall-oriented.
    P2 High      — conformal p<=alpha          : distribution-free FPR<=alpha escalation.
    P1 Critical  — one-sided CUSUM (Page 1954) : sequential confirmation of a sustained shift.

For the memoryless threshold tiers (P3, P2) we report sample precision/recall,
episode recall, and a per-sample false-alarm rate on normal data. For the
sequential CUSUM tier, per-sample F1 is meaningless (a stateful detector fires and
holds); we instead report the literature-standard sequential metrics:
    - detection delay (ARL1 proxy): samples from fault onset to first alarm,
    - ARL0 proxy: mean samples between false-alarm onsets on normal data,
    - episode escalation rate: fraction of fault episodes that reach P1.

All inputs are consumed from the ``_predictions`` payload of
``compute_operating_points(..., return_predictions=True)`` so thresholds have a
single source of truth and are never recomputed here.
"""
from __future__ import annotations

import numpy as np


def _segment_index_lists(group_ids: np.ndarray) -> list[np.ndarray]:
    """Group sample indices by segment id, preserving temporal (appearance) order."""
    groups: dict = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    return [np.asarray(idxs) for idxs in groups.values()]


def _detection_delays(
    preds: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, float | int | None]:
    """Per fault-episode onset->first-alarm delay (samples).

    Onset = first faulty sample within a fault segment (temporal order). Delay =
    samples from onset to the first alarm at or after onset. Episodes never
    alarmed are misses (excluded from the delay distribution but counted in the
    detection rate). Returns median/p90/mean delay over *detected* episodes.
    """
    preds = np.asarray(preds).astype(np.int8)
    labels = np.asarray(labels).astype(np.int8)
    delays: list[int] = []
    n_fault = 0
    n_detected = 0
    for idxs in _segment_index_lists(group_ids):
        seg_labels = labels[idxs]
        if seg_labels.max() <= 0:
            continue  # normal segment
        n_fault += 1
        onset_pos = int(np.argmax(seg_labels > 0))  # first faulty position
        seg_preds = preds[idxs]
        post = np.where(seg_preds[onset_pos:] == 1)[0]
        if post.size:
            n_detected += 1
            delays.append(int(post[0]))  # samples after onset
    if not delays:
        return {
            "median_detection_delay_samples": None,
            "p90_detection_delay_samples": None,
            "mean_detection_delay_samples": None,
            "n_fault_episodes": int(n_fault),
            "n_detected_episodes": int(n_detected),
            "episode_detection_rate": float(n_detected / n_fault) if n_fault else 0.0,
        }
    arr = np.asarray(delays, dtype=float)
    return {
        "median_detection_delay_samples": float(np.median(arr)),
        "p90_detection_delay_samples": float(np.percentile(arr, 90)),
        "mean_detection_delay_samples": float(np.mean(arr)),
        "n_fault_episodes": int(n_fault),
        "n_detected_episodes": int(n_detected),
        "episode_detection_rate": float(n_detected / n_fault) if n_fault else 0.0,
    }


def _false_alarm_stats(
    preds: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray | None,
) -> dict[str, float | int | None]:
    """False-alarm rate and ARL0 (mean samples between false-alarm onsets) on normal data.

    FAR is per-sample on normal-labeled samples. ARL0 counts *rising edges*
    (0->1 transitions) within normal segments as distinct false alarms — correct
    for a sequential detector that holds its alarm across consecutive samples —
    and divides the normal-sample count by the onset count.
    """
    preds = np.asarray(preds).astype(np.int8)
    labels = np.asarray(labels).astype(np.int8)
    normal_mask = labels == 0
    n_normal = int(normal_mask.sum())
    if n_normal == 0:
        return {"false_alarm_rate": None, "arl0_samples": None, "n_normal_samples": 0}

    far = float(preds[normal_mask].mean())

    # Count false-alarm onsets within normal-only segments (rising edges).
    n_onsets = 0
    if group_ids is not None:
        for idxs in _segment_index_lists(np.asarray(group_ids)):
            if labels[idxs].max() > 0:
                continue  # only pure-normal segments contribute clean false alarms
            seg = preds[idxs]
            prev = 0
            for v in seg:
                if v == 1 and prev == 0:
                    n_onsets += 1
                prev = v
    else:
        # No segments: count global rising edges on normal samples
        seq = preds[normal_mask]
        n_onsets = int(np.sum((seq[1:] == 1) & (seq[:-1] == 0))) + int(seq[:1].sum())

    # None (not inf) when there are no false alarms — keeps the JSON strict-valid;
    # false_alarm_rate=0 + n_false_alarm_onsets=0 already convey "no false alarms".
    arl0 = float(n_normal / n_onsets) if n_onsets > 0 else None
    return {
        "false_alarm_rate": far,
        "arl0_samples": arl0,
        "n_false_alarm_onsets": int(n_onsets),
        "n_normal_samples": n_normal,
    }


def compute_deployment_metrics(
    *,
    operating_points: dict,
    test_labels_binary: np.ndarray,
    test_group_ids: np.ndarray | None,
    macro_per_class_pr_auc: float | None,
    worst_class_pr_auc: float | None,
    worst_class: str | None = None,
    worst_class_support: int | None = None,
    seconds_per_sample: float | None = None,
    selection_metric: str = "val_macro_per_class_pr_auc",
) -> dict:
    """Build the thesis-ready deployment-characterization payload.

    Requires ``operating_points`` produced with ``return_predictions=True``.
    """
    preds_payload = operating_points.get("_predictions")
    if preds_payload is None:
        raise ValueError(
            "operating_points has no '_predictions' — call compute_operating_points("
            "..., return_predictions=True)"
        )
    thr = preds_payload["thresholds"]
    labels = np.asarray(test_labels_binary).astype(np.int8)
    gids = np.asarray(test_group_ids) if test_group_ids is not None else None

    def _tier(op_key: str) -> dict:
        op = operating_points[op_key]
        preds = preds_payload[op_key]
        delays = _detection_delays(preds, labels, gids) if gids is not None else {}
        fa = _false_alarm_stats(preds, labels, gids)
        tier = {
            "policy": op.get("policy"),
            "sample_precision": op.get("precision"),
            "sample_recall": op.get("recall"),
            "sample_f1": op.get("f1"),
            "episode_recall": op.get("episode_recall"),
            **delays,
            **fa,
        }
        if seconds_per_sample and delays.get("median_detection_delay_samples") is not None:
            tier["median_detection_delay_seconds"] = (
                delays["median_detection_delay_samples"] * float(seconds_per_sample)
            )
        return tier

    # P3 Advisory = sensitive + hysteresis (the deployed first-alarm tier)
    p3 = _tier("sensitive_hysteresis")
    p3["threshold"] = thr["sensitive"]
    p3["hysteresis_n"] = operating_points["sensitive_hysteresis"].get("hysteresis_n")

    # P2 High = conformal (distribution-free escalation)
    p2 = _tier("conformal")
    p2["conformal_alpha"] = thr["conformal_alpha"]
    p2["fpr_guarantee"] = thr["conformal_alpha"]  # distribution-free upper bound

    # P1 Critical = CUSUM (sequential confirmation). Per-sample F1 dropped as
    # meaningless for a stateful detector; sequential metrics lead instead.
    cusum_op = operating_points["cusum"]
    cusum_preds = preds_payload["cusum"]
    cusum_delays = _detection_delays(cusum_preds, labels, gids) if gids is not None else {}
    cusum_fa = _false_alarm_stats(cusum_preds, labels, gids)
    p1 = {
        "policy": cusum_op.get("policy"),
        "cusum_k": thr["cusum_k"],
        "cusum_h": thr["cusum_h"],
        "episode_escalation_rate": cusum_op.get("episode_recall"),  # fraction reaching P1
        **cusum_delays,
        **cusum_fa,
        "sample_precision": cusum_op.get("precision"),  # kept for completeness, not headline
    }
    if seconds_per_sample and cusum_delays.get("median_detection_delay_samples") is not None:
        p1["median_detection_delay_seconds"] = (
            cusum_delays["median_detection_delay_samples"] * float(seconds_per_sample)
        )

    return {
        "headline": {
            "selection_metric": selection_metric,
            "macro_per_class_pr_auc": macro_per_class_pr_auc,
            "worst_class_pr_auc": worst_class_pr_auc,
            "worst_class": worst_class,
            "worst_class_support": worst_class_support,
            "note": (
                "Threshold-free; this is the model-selection / tournament metric. "
                "PR-AUC baseline equals class prevalence, which differs per class — "
                "report per-class support alongside."
            ),
        },
        "tiers": {
            "P3_advisory": p3,
            "P2_high": p2,
            "P1_critical": p1,
        },
        "notes": {
            "tiers_are_sequential": (
                "P3 -> P2 -> P1 is one escalating system, not competing detectors. "
                "Tier metrics are a characterization and must not be averaged."
            ),
            "cusum_metric_choice": (
                "CUSUM is a stateful sequential detector; per-sample F1 is not "
                "reported. Detection delay (ARL1 proxy), ARL0 (mean samples between "
                "false-alarm onsets), and episode escalation rate are the "
                "literature-standard characterization."
            ),
            "sample_f1_caveat": (
                "Sample precision/recall/F1 reflect the artificial test-set fault "
                "prevalence under a ~97% normal class; lead with episode recall."
            ),
        },
    }
