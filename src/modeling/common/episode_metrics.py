from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


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
