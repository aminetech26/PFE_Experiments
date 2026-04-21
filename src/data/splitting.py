"""
Splitting utilities for PV fault detection experiments.

This module provides temporal-stratified splitting that respects:
1. Segment boundaries (no segment split across train/val/test)
2. Temporal order (train < val < test chronologically within each class)
3. Class stratification (fault classes distributed proportionally)

Functions:
    temporal_stratified_split: Main split for supervised learning (Task A/B)
    hybrid_semisup_split: Train on normal-only, val/test have faults (Task A)
    filter_to_evaluable_classes: Filter to classes with enough segments (Task B)

Classes:
    SegmentTimeSeriesCV: Expanding window CV respecting segments
    PerClassSegmentTimeSeriesCV: Per-class temporal CV for imbalanced data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitArtifacts:
    """Container for split results."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    dropped: pd.DataFrame
    manifest: dict


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _segment_summary(
    df: pd.DataFrame,
    segment_col: str,
    label_col: str,
    time_col: str,
) -> pd.DataFrame:
    """Build summary of each segment with start time, labels, and row count."""
    summary = (
        df.groupby(segment_col)
        .agg(
            start=(time_col, "min"),
            end=(time_col, "max"),
            n_rows=(segment_col, "size"),
            labels=(label_col, lambda x: sorted(pd.Series(x).dropna().unique().tolist())),
        )
        .reset_index(drop=False)
        .sort_values("start")
        .reset_index(drop=True)
    )
    return summary


def _get_primary_fault(labels: list) -> float | None:
    """Get the primary fault class from a list of labels (first non-zero)."""
    faults = [l for l in labels if l != 0.0]
    return faults[0] if faults else None


# =============================================================================
# MAIN SPLIT FUNCTIONS
# =============================================================================


def temporal_stratified_split(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    label_col: str = "label",
    time_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    embargo_seconds: int = 0,  # Not used in segment-stratified (segment boundaries provide isolation)
) -> SplitArtifacts:
    """
    Temporal-stratified split: respects BOTH temporal order AND class stratification.

    For each fault class:
      - Sort segments by start time
      - Assign first N to train, next M to val, rest to test
      - This ensures train < val < test temporally within each class

    Note: embargo_seconds is accepted but NOT applied. In segment-stratified splits,
    different classes have overlapping time ranges (e.g., fault class A and B may
    both have data in Feb-March). Global embargo would incorrectly drop data.
    Segment boundaries (gaps > 5 min) provide sufficient isolation.

    Parameters
    ----------
    df : DataFrame
        Input data with segment_id, label, and timestamp columns
    segment_col : str
        Column name for segment IDs
    label_col : str
        Column name for class labels
    time_col : str
        Column name for timestamps
    train_ratio : float
        Fraction of data for training (default 0.70)
    val_ratio : float
        Fraction of data for validation (default 0.15)
    embargo_seconds : int
        Ignored for segment-stratified splits (kept for API compatibility)

    Returns
    -------
    SplitArtifacts
        Container with train, val, test DataFrames and manifest
    """
    seg_summary = _segment_summary(df, segment_col, label_col, time_col)

    seg_summary["primary_fault"] = seg_summary["labels"].apply(_get_primary_fault)
    seg_summary["has_fault"] = seg_summary["primary_fault"].notna()

    fault_segments = seg_summary[seg_summary["has_fault"]].copy()
    normal_segments = seg_summary[~seg_summary["has_fault"]].copy()

    train_segs, val_segs, test_segs = [], [], []
    class_temporal_info = {}

    # Distribute fault segments per class (temporal order)
    for fault_class in sorted(fault_segments["primary_fault"].unique()):
        class_seg_df = fault_segments[fault_segments["primary_fault"] == fault_class].sort_values(
            "start"
        )
        class_segs = class_seg_df[segment_col].tolist()

        n = len(class_segs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))

        if n <= 2:
            # Too few segments: all go to train
            train_segs.extend(class_segs)
            class_temporal_info[fault_class] = {
                "n_segments": n,
                "train": class_segs,
                "val": [],
                "test": [],
                "note": "Too few segments for val/test",
            }
        else:
            train_class = class_segs[:n_train]
            val_class = class_segs[n_train : n_train + n_val]
            test_class = class_segs[n_train + n_val :]

            train_segs.extend(train_class)
            val_segs.extend(val_class)
            test_segs.extend(test_class)

            class_temporal_info[fault_class] = {
                "n_segments": n,
                "train": train_class,
                "val": val_class,
                "test": test_class,
            }

    # Distribute normal segments (temporal order)
    normal_seg_df = normal_segments.sort_values("start")
    normal_seg_ids = normal_seg_df[segment_col].tolist()

    n_normal = len(normal_seg_ids)
    n_train_normal = int(n_normal * train_ratio)
    n_val_normal = int(n_normal * val_ratio)

    train_segs.extend(normal_seg_ids[:n_train_normal])
    val_segs.extend(normal_seg_ids[n_train_normal : n_train_normal + n_val_normal])
    test_segs.extend(normal_seg_ids[n_train_normal + n_val_normal :])

    # Create DataFrames
    train_df = df[df[segment_col].isin(train_segs)].copy()
    val_df = df[df[segment_col].isin(val_segs)].copy()
    test_df = df[df[segment_col].isin(test_segs)].copy()

    # For segment-stratified splits, segment boundaries provide isolation
    # No global embargo needed (would drop valid data due to interleaved classes)
    dropped = pd.DataFrame()

    manifest = {
        "split_type": "temporal_stratified",
        "n_rows": len(df),
        "n_segments": len(seg_summary),
        "n_fault_segments": len(fault_segments),
        "n_normal_segments": len(normal_segments),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "dropped_rows": len(dropped),
        "train_segments": len(train_segs),
        "val_segments": len(val_segs),
        "test_segments": len(test_segs),
        "train_class_counts": train_df[label_col].value_counts().sort_index().to_dict(),
        "val_class_counts": val_df[label_col].value_counts().sort_index().to_dict(),
        "test_class_counts": test_df[label_col].value_counts().sort_index().to_dict(),
        "fault_segments_by_class": fault_segments.groupby("primary_fault")[segment_col]
        .count()
        .to_dict(),
        "class_temporal_info": {str(k): v for k, v in class_temporal_info.items()},
        "note": "Segment boundaries (>300s gaps) provide temporal isolation. No global embargo applied.",
    }

    return SplitArtifacts(train_df, val_df, test_df, dropped, manifest)


def hybrid_semisup_split(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    label_col: str = "label",
    time_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    embargo_seconds: int = 0,  # Not used (segment boundaries provide isolation)
) -> SplitArtifacts:
    """
    Temporal hybrid split for semi-supervised anomaly detection.

    Train: Normal-only segments (chronologically first 70%)
    Val/Test: Normal + fault segments (temporal order preserved)

    Use for one-class learning (Isolation Forest, One-Class SVM, Autoencoder).

    Note: embargo_seconds is accepted but NOT applied. Segment boundaries (>5 min gaps)
    provide sufficient temporal isolation between data points.
    """
    seg_summary = _segment_summary(df, segment_col, label_col, time_col)

    seg_summary["has_fault"] = seg_summary["labels"].apply(lambda x: any(l != 0.0 for l in x))
    seg_summary["primary_fault"] = seg_summary["labels"].apply(_get_primary_fault)

    fault_segments = seg_summary[seg_summary["has_fault"]].copy()
    normal_segments = seg_summary[~seg_summary["has_fault"]].copy()

    # Train: normal-only (temporal order)
    normal_seg_df = normal_segments.sort_values("start")
    normal_seg_ids = normal_seg_df[segment_col].tolist()

    n_normal = len(normal_seg_ids)
    n_train = int(n_normal * train_ratio)
    n_val = int(n_normal * val_ratio)

    train_normal = normal_seg_ids[:n_train]
    val_normal = normal_seg_ids[n_train : n_train + n_val]
    test_normal = normal_seg_ids[n_train + n_val :]

    # Val/Test: add fault segments (temporal order, 50/50 split)
    val_fault, test_fault = [], []

    for fault_class in sorted(fault_segments["primary_fault"].unique()):
        class_seg_df = fault_segments[fault_segments["primary_fault"] == fault_class].sort_values(
            "start"
        )
        class_segs = class_seg_df[segment_col].tolist()

        n = len(class_segs)
        n_val_fault = max(1, n // 2)

        if n == 1:
            test_fault.extend(class_segs)
        else:
            val_fault.extend(class_segs[:n_val_fault])
            test_fault.extend(class_segs[n_val_fault:])

    train_segs = train_normal
    val_segs = val_normal + val_fault
    test_segs = test_normal + test_fault

    train_df = df[df[segment_col].isin(train_segs)].copy()
    val_df = df[df[segment_col].isin(val_segs)].copy()
    test_df = df[df[segment_col].isin(test_segs)].copy()

    # Segment boundaries provide isolation - no global embargo needed
    dropped = pd.DataFrame()

    manifest = {
        "split_type": "hybrid_semisup_temporal",
        "n_rows": len(df),
        "n_segments": len(seg_summary),
        "n_fault_segments": len(fault_segments),
        "n_normal_segments": len(normal_segments),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "dropped_rows": len(dropped),
        "train_segments": len(train_segs),
        "val_segments": len(val_segs),
        "test_segments": len(test_segs),
        "train_fault_segments": 0,
        "val_fault_segments": len(val_fault),
        "test_fault_segments": len(test_fault),
        "train_class_counts": train_df[label_col].value_counts().sort_index().to_dict(),
        "val_class_counts": val_df[label_col].value_counts().sort_index().to_dict(),
        "test_class_counts": test_df[label_col].value_counts().sort_index().to_dict(),
        "note": "Train=normal-only (temporal). Val/Test=normal+faults. Segment boundaries provide isolation.",
    }

    return SplitArtifacts(train_df, val_df, test_df, dropped, manifest)


def filter_to_evaluable_classes(
    df: pd.DataFrame,
    evaluable_classes: list = [3.1, 3.2, 4.0],
    label_col: str = "label",
) -> pd.DataFrame:
    """Filter DataFrame to only evaluable fault classes (for Task B classification)."""
    return df[df[label_col].isin(evaluable_classes)].copy()


# Alias for backward compatibility
segment_stratified_split = temporal_stratified_split


# =============================================================================
# CROSS-VALIDATION FOR HYPERPARAMETER TUNING
# =============================================================================


class SegmentTimeSeriesCV:
    """
    Segment-aware temporal cross-validation (expanding window).

    For each fold:
      - Train on first N segments (chronologically)
      - Validate on next segment(s)

    Respects temporal order AND segment boundaries.

    Example
    -------
    >>> cv = SegmentTimeSeriesCV(n_splits=3, min_train_segments=2)
    >>> for train_idx, val_idx in cv.split(X, groups=segment_ids):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[val_idx], y[val_idx])
    """

    def __init__(
        self,
        n_splits: int = 3,
        min_train_segments: int = 1,
        gap_segments: int = 0,
    ):
        self.n_splits = n_splits
        self.min_train_segments = min_train_segments
        self.gap_segments = gap_segments

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/val indices for each fold."""
        if groups is None:
            raise ValueError("groups (segment_ids) must be provided")

        groups = np.array(groups)
        unique_segments = []
        segment_indices = {}

        for idx, seg in enumerate(groups):
            if seg not in segment_indices:
                unique_segments.append(seg)
                segment_indices[seg] = []
            segment_indices[seg].append(idx)

        n_segments = len(unique_segments)

        if n_segments < self.min_train_segments + 1:
            raise ValueError(
                f"Need at least {self.min_train_segments + 1} segments, got {n_segments}"
            )

        segments_per_fold = max(1, (n_segments - self.min_train_segments) // self.n_splits)

        for fold in range(self.n_splits):
            train_end = self.min_train_segments + fold * segments_per_fold
            train_segments = unique_segments[:train_end]

            val_start = train_end + self.gap_segments
            val_end = min(val_start + segments_per_fold, n_segments)

            if val_start >= n_segments:
                break

            val_segments = unique_segments[val_start:val_end]

            train_idx = [i for seg in train_segments for i in segment_indices[seg]]
            val_idx = [i for seg in val_segments for i in segment_indices[seg]]

            yield np.array(train_idx), np.array(val_idx)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class PerClassSegmentTimeSeriesCV:
    """
    Per-class segment-aware temporal CV for imbalanced classification.

    Performs expanding window CV separately for each class, then combines.
    Useful when different classes have different temporal distributions.
    """

    def __init__(self, n_splits: int = 3, min_train_segments: int = 1):
        self.n_splits = n_splits
        self.min_train_segments = min_train_segments

    def split(
        self,
        X,
        y,
        groups,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/val indices with per-class temporal stratification."""
        y = np.array(y)
        groups = np.array(groups)
        unique_classes = np.unique(y)

        for fold in range(self.n_splits):
            train_idx_all = []
            val_idx_all = []

            for cls in unique_classes:
                cls_mask = y == cls
                cls_indices = np.where(cls_mask)[0]
                cls_groups = groups[cls_mask]

                unique_segs = []
                seg_to_idx = {}
                for idx, seg in zip(cls_indices, cls_groups):
                    if seg not in seg_to_idx:
                        unique_segs.append(seg)
                        seg_to_idx[seg] = []
                    seg_to_idx[seg].append(idx)

                n_segs = len(unique_segs)
                if n_segs < 2:
                    train_idx_all.extend(cls_indices.tolist())
                    continue

                train_end = min(self.min_train_segments + fold, n_segs - 1)
                train_segs = unique_segs[:train_end]
                val_segs = unique_segs[train_end : train_end + 1]

                for seg in train_segs:
                    train_idx_all.extend(seg_to_idx[seg])
                for seg in val_segs:
                    val_idx_all.extend(seg_to_idx[seg])

            if val_idx_all:
                yield np.array(train_idx_all), np.array(val_idx_all)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
