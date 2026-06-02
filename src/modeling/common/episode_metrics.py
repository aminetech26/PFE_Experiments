from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def episode_macro_f1_binary(
    labels: np.ndarray,
    preds: np.ndarray,
    group_ids: np.ndarray | None,
) -> float:
    """Episode-level macro F1 for binary anomaly detection.

    Each group (episode/segment/day) is labeled anomalous if any row within it
    is true-positive, and predicted anomalous if any row within it is predicted
    positive. F1 is then computed macro-averaged over episodes.

    Returns 0.0 if group_ids is None, misaligned, or empty.
    """
    if group_ids is None or len(group_ids) != len(labels):
        return 0.0
    by_group: dict[str, dict[str, int]] = {}
    for y, p, g in zip(labels.astype(int), preds.astype(int), group_ids.astype(str), strict=False):
        row = by_group.setdefault(g, {"pos_true": 0, "pos_pred": 0})
        row["pos_true"] += int(y == 1)
        row["pos_pred"] += int(p == 1)
    if not by_group:
        return 0.0
    ep_true = np.array([1 if v["pos_true"] > 0 else 0 for v in by_group.values()], dtype=int)
    ep_pred = np.array([1 if v["pos_pred"] > 0 else 0 for v in by_group.values()], dtype=int)
    return float(f1_score(ep_true, ep_pred, average="macro", zero_division=0))


def episode_macro_pr_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray | None,
) -> float:
    """Episode-level PR AUC using max-aggregated anomaly scores per episode.

    For each episode, the score is the maximum row-level score and the label is 1
    if any row is anomalous. PR AUC is computed on these episode-level aggregates.

    Returns 0.0 if group_ids is None, misaligned, empty, or all episodes share
    the same label.
    """
    if group_ids is None or len(group_ids) != len(labels):
        return 0.0
    by_group: dict[str, dict[str, float | int]] = {}
    for s, y, g in zip(scores.astype(float), labels.astype(int), group_ids.astype(str), strict=False):
        row = by_group.setdefault(g, {"max_score": -float("inf"), "has_anomaly": 0})
        row["max_score"] = max(float(row["max_score"]), float(s))
        row["has_anomaly"] = max(int(row["has_anomaly"]), int(y == 1))
    if not by_group:
        return 0.0
    ep_true = np.array([v["has_anomaly"] for v in by_group.values()], dtype=int)
    ep_scores = np.array([v["max_score"] for v in by_group.values()], dtype=float)
    if len(np.unique(ep_true)) < 2:
        return 0.0
    return float(average_precision_score(ep_true, ep_scores))
