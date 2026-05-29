"""Blindfolded threshold strategies and operating-point diagnostics.

All functions here treat the detector's per-sample anomaly score as given and
analyse how the *decision* (threshold + post-processing) is made. The headline
strategy — temporal persistence — is "blindfolded": it uses no fault labels,
only the alarm stream and segment ids, plus a latency budget chosen a priori.

The label-using helpers (`f1_optimal_threshold`) are provided ONLY to bound the
gap between the deployable operating point and the oracle one (the "price of
deployability"); they must never be used to *set* the deployed threshold without
disclosing that supervision was used.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# Single source of truth for the persistence filter (pipeline + analysis share it).
from src.modeling.common.operating_point import apply_persistence_filter

__all__ = [
    "apply_persistence_filter",
    "persistence_sweep",
    "f1_optimal_threshold",
    "metrics_at_threshold",
    "operating_point_nll_diagnostic",
]


def persistence_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    threshold: float,
    n_values: tuple[int, ...] = (1, 3, 5, 10),
) -> list[dict]:
    """Sweep the N-consecutive persistence filter at a fixed threshold.

    Returns one record per N with precision/recall/F1 against per-sample labels
    plus the realized alarm rate. N=1 is the raw (no-debounce) baseline.
    """
    raw = (np.asarray(scores) >= threshold).astype(np.int8)
    labels = np.asarray(labels).astype(np.int8)
    rows: list[dict] = []
    for n in n_values:
        deb = apply_persistence_filter(raw, group_ids, n)
        rows.append(
            {
                "n_consecutive": int(n),
                "precision": float(precision_score(labels, deb, zero_division=0)),
                "recall": float(recall_score(labels, deb, zero_division=0)),
                "f1": float(f1_score(labels, deb, zero_division=0)),
                "alarm_rate": float(deb.mean()),
                "detection_latency_samples": int(n - 1),
            }
        )
    return rows


def f1_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold on the PR curve that maximizes F1 (USES fault labels).

    For the oracle upper bound (on test) or the val-tuned operating point. The
    returned F1 is the best achievable at any single global threshold — what the
    PR-AUC "promises". Never use to set the deployed threshold silently.
    """
    p, r, thr = precision_recall_curve(labels, scores)
    f1 = 2.0 * p * r / (p + r + 1e-12)
    f1 = f1[:-1]  # drop the (P,R)=(1,0) sentinel with no threshold
    if f1.size == 0:
        return float("inf"), 0.0
    best = int(np.nanargmax(f1))
    return float(thr[best]), float(f1[best])


def metrics_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> dict:
    preds = (np.asarray(scores) >= threshold).astype(np.int8)
    labels = np.asarray(labels).astype(np.int8)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def operating_point_nll_diagnostic(
    normal_scores: np.ndarray,
    irr: np.ndarray,
    pvt: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Does the normal anomaly score vary with the operating point?

    If yes, the flow's conditioning is incomplete and a conditional threshold
    t(irr, pvt) would help. If the per-bin mean score is roughly flat, the
    conditioning already absorbs the operating point and a global threshold is
    fine. Blindfolded: normal samples only.

    Returns per-context Pearson correlation, binned mean score, and a
    `bin_spread_ratio` = std(per-bin mean) / std(all scores). A large ratio
    (e.g. > ~0.3) signals the score still depends on the operating point.
    """
    s = np.asarray(normal_scores, dtype=float)
    s_std = float(np.std(s)) or 1.0
    out: dict = {}
    for name, raw_x in [("irr", irr), ("pvt", pvt)]:
        x = np.asarray(raw_x, dtype=float)
        corr = (
            float(np.corrcoef(x, s)[0, 1])
            if np.std(x) > 1e-9 and np.std(s) > 1e-9
            else 0.0
        )
        edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
        binned = []
        bin_means = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x <= hi)
            if m.sum() > 0:
                bm = float(s[m].mean())
                bin_means.append(bm)
                binned.append({"center": float((lo + hi) / 2), "mean_score": bm, "n": int(m.sum())})
        spread = float(np.std(bin_means)) / s_std if bin_means else 0.0
        out[name] = {
            "pearson_corr": corr,
            "bin_spread_ratio": spread,
            "binned_mean_score": binned,
        }
    return out
