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
    identify_fault_episodes: Find contiguous fault periods (Task C helper)
    prediction_episode_split: Episode-based split for fault prediction (Task C)

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
        class_seg_df = fault_segments[fault_segments["primary_fault"] == fault_class].sort_values("start")
        class_segs = class_seg_df[segment_col].tolist()
        
        n = len(class_segs)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        
        if n <= 2:
            # Too few segments: all go to train
            train_segs.extend(class_segs)
            class_temporal_info[fault_class] = {
                "n_segments": n, "train": class_segs, "val": [], "test": [],
                "note": "Too few segments for val/test"
            }
        else:
            train_class = class_segs[:n_train]
            val_class = class_segs[n_train:n_train + n_val]
            test_class = class_segs[n_train + n_val:]
            
            train_segs.extend(train_class)
            val_segs.extend(val_class)
            test_segs.extend(test_class)
            
            class_temporal_info[fault_class] = {
                "n_segments": n, "train": train_class, "val": val_class, "test": test_class
            }
    
    # Distribute normal segments (temporal order)
    normal_seg_df = normal_segments.sort_values("start")
    normal_seg_ids = normal_seg_df[segment_col].tolist()
    
    n_normal = len(normal_seg_ids)
    n_train_normal = int(n_normal * train_ratio)
    n_val_normal = int(n_normal * val_ratio)
    
    train_segs.extend(normal_seg_ids[:n_train_normal])
    val_segs.extend(normal_seg_ids[n_train_normal:n_train_normal + n_val_normal])
    test_segs.extend(normal_seg_ids[n_train_normal + n_val_normal:])
    
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
        "fault_segments_by_class": fault_segments.groupby("primary_fault")[segment_col].count().to_dict(),
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
    val_normal = normal_seg_ids[n_train:n_train + n_val]
    test_normal = normal_seg_ids[n_train + n_val:]
    
    # Val/Test: add fault segments (temporal order, 50/50 split)
    val_fault, test_fault = [], []
    
    for fault_class in sorted(fault_segments["primary_fault"].unique()):
        class_seg_df = fault_segments[fault_segments["primary_fault"] == fault_class].sort_values("start")
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
# TASK C: FAULT PREDICTION (EPISODE-BASED SPLIT)
# =============================================================================

def identify_fault_episodes(
    df: pd.DataFrame,
    label_col: str = "label",
    time_col: str = "timestamp",
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """
    Identify contiguous fault episodes in the data.
    
    A fault episode is a contiguous period where fault labels are present.
    Episodes are bounded by:
      - Transition from normal to fault
      - Transition from fault to normal
      - Segment boundaries (>5 min gaps)
    
    Returns DataFrame with columns:
      - episode_id: Unique identifier
      - fault_class: The fault label
      - start: Episode start timestamp
      - end: Episode end timestamp
      - duration: Episode duration
      - n_rows: Number of data points
      - segment_id: Which segment contains this episode
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    
    # Mark fault rows
    df_sorted["is_fault"] = df_sorted[label_col] != 0.0
    
    # Detect episode boundaries (fault start or segment change during fault)
    shifted_fault = df_sorted["is_fault"].shift(1)
    shifted_fault = shifted_fault.where(shifted_fault.notna(), False).astype(bool)
    df_sorted["fault_start"] = (
        (df_sorted["is_fault"] & ~shifted_fault) |
        (df_sorted["is_fault"] & (df_sorted[segment_col] != df_sorted[segment_col].shift(1)))
    )
    
    # Assign episode IDs (only to fault rows)
    df_sorted["episode_id"] = df_sorted["fault_start"].cumsum()
    df_sorted.loc[~df_sorted["is_fault"], "episode_id"] = -1
    
    # Aggregate episode info
    fault_rows = df_sorted[df_sorted["is_fault"]].copy()
    
    if len(fault_rows) == 0:
        return pd.DataFrame(columns=[
            "episode_id", "fault_class", "start", "end", "duration", "n_rows", "segment_id"
        ])
    
    episodes = fault_rows.groupby("episode_id").agg({
        label_col: "first",
        time_col: ["min", "max", "count"],
        segment_col: "first",
    })
    episodes.columns = ["fault_class", "start", "end", "n_rows", "segment_id"]
    episodes["duration"] = episodes["end"] - episodes["start"]
    episodes = episodes.reset_index()
    
    return episodes.sort_values("start").reset_index(drop=True)


def prediction_episode_split(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    label_col: str = "label",
    time_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    forecast_horizon_seconds: int = 1800,  # 30 minutes
    min_episodes_for_eval: int = 3,
) -> SplitArtifacts:
    """
    Episode-based split for fault prediction (Task C).
    
    Splits data ensuring fault episodes are atomic units:
    - Episodes assigned to train/test based on temporal order within each class
    - Pre-fault zones (within forecast_horizon of episode start) go with their episode
    - Normal data split temporally (excluding pre-fault zones)
    - Val set contains only normal data (no fault episodes for clean threshold tuning)
    
    Parameters
    ----------
    df : DataFrame
        Input data with segment_id, label, and timestamp columns
    forecast_horizon_seconds : int
        How far ahead to predict (default 1800 = 30 min)
    min_episodes_for_eval : int
        Minimum episodes needed to put class in test (default 3)
    
    Returns
    -------
    SplitArtifacts
        train, val, test DataFrames with additional columns:
        - episode_id: Which episode this row belongs to (-1 for normal)
        - in_prefault_zone: Whether this row is in a pre-fault window
        - zone_episode_id: Which episode's pre-fault zone this belongs to
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    
    # Step 1: Identify all fault episodes
    episodes = identify_fault_episodes(df, label_col, time_col, segment_col)
    n_episodes = len(episodes)
    
    if n_episodes == 0:
        raise ValueError("No fault episodes found in data")
    
    # Step 2: Assign episodes to train/test
    episode_assignments = {}
    episodes_by_class = {}
    
    for fault_class in sorted(episodes["fault_class"].unique()):
        class_episodes = episodes[episodes["fault_class"] == fault_class].copy()
        class_episodes = class_episodes.sort_values("start")
        ep_ids = class_episodes["episode_id"].tolist()
        n = len(ep_ids)
        
        episodes_by_class[fault_class] = {
            "n_episodes": n,
            "episode_ids": ep_ids,
        }
        
        if n < min_episodes_for_eval:
            # Too few episodes - all go to train
            for ep_id in ep_ids:
                episode_assignments[ep_id] = "train"
            episodes_by_class[fault_class]["train"] = ep_ids
            episodes_by_class[fault_class]["test"] = []
            episodes_by_class[fault_class]["evaluable"] = False
        else:
            # Split temporally: first ~70% train, last ~30% test
            n_train = max(1, int(n * train_ratio))
            train_eps = ep_ids[:n_train]
            test_eps = ep_ids[n_train:]
            
            for ep_id in train_eps:
                episode_assignments[ep_id] = "train"
            for ep_id in test_eps:
                episode_assignments[ep_id] = "test"
            
            episodes_by_class[fault_class]["train"] = train_eps
            episodes_by_class[fault_class]["test"] = test_eps
            episodes_by_class[fault_class]["evaluable"] = True
    
    # Step 3: Mark pre-fault zones
    forecast_delta = pd.Timedelta(seconds=forecast_horizon_seconds)
    
    # Initialize columns
    df["episode_id"] = -1
    df["in_prefault_zone"] = False
    df["zone_episode_id"] = -1
    df["episode_set"] = "normal"  # Will be "train", "test", or "normal"
    
    # Mark fault rows with their episode
    df_sorted_temp = df.sort_values(time_col).reset_index(drop=True)
    df_sorted_temp["is_fault"] = df_sorted_temp[label_col] != 0.0
    shifted_fault_temp = df_sorted_temp["is_fault"].shift(1)
    shifted_fault_temp = shifted_fault_temp.where(shifted_fault_temp.notna(), False).astype(bool)
    df_sorted_temp["fault_start"] = (
        (df_sorted_temp["is_fault"] & ~shifted_fault_temp) |
        (df_sorted_temp["is_fault"] & (df_sorted_temp[segment_col] != df_sorted_temp[segment_col].shift(1)))
    )
    df_sorted_temp["temp_episode_id"] = df_sorted_temp["fault_start"].cumsum()
    df_sorted_temp.loc[~df_sorted_temp["is_fault"], "temp_episode_id"] = -1
    
    df["episode_id"] = df_sorted_temp["temp_episode_id"].values
    
    # Mark episode assignments
    for ep_id, assignment in episode_assignments.items():
        mask = df["episode_id"] == ep_id
        df.loc[mask, "episode_set"] = assignment
    
    # Mark pre-fault zones for each episode
    for _, ep_row in episodes.iterrows():
        ep_id = ep_row["episode_id"]
        ep_start = ep_row["start"]
        ep_segment = ep_row["segment_id"]
        assignment = episode_assignments[ep_id]
        
        # Pre-fault zone: same segment, within forecast_horizon before episode start
        prefault_start = ep_start - forecast_delta
        prefault_mask = (
            (df[segment_col] == ep_segment) &
            (df[time_col] >= prefault_start) &
            (df[time_col] < ep_start) &
            (df[label_col] == 0.0)  # Only normal rows
        )
        
        df.loc[prefault_mask, "in_prefault_zone"] = True
        df.loc[prefault_mask, "zone_episode_id"] = ep_id
        df.loc[prefault_mask, "episode_set"] = assignment  # Pre-fault goes with episode
    
    # Step 4: Split normal data (not in any pre-fault zone, not fault)
    normal_mask = (df[label_col] == 0.0) & (~df["in_prefault_zone"])
    normal_df = df[normal_mask].copy()
    
    # Temporal split of normal data
    normal_df = normal_df.sort_values(time_col)
    n_normal = len(normal_df)
    n_train_normal = int(n_normal * train_ratio)
    n_val_normal = int(n_normal * val_ratio)
    
    normal_indices = normal_df.index.tolist()
    train_normal_idx = set(normal_indices[:n_train_normal])
    val_normal_idx = set(normal_indices[n_train_normal:n_train_normal + n_val_normal])
    test_normal_idx = set(normal_indices[n_train_normal + n_val_normal:])
    
    # Apply assignments
    df.loc[df.index.isin(train_normal_idx), "episode_set"] = "train"
    df.loc[df.index.isin(val_normal_idx), "episode_set"] = "val"
    df.loc[df.index.isin(test_normal_idx), "episode_set"] = "test"
    
    # Step 5: Create split DataFrames
    train_df = df[df["episode_set"] == "train"].copy()
    val_df = df[df["episode_set"] == "val"].copy()
    test_df = df[df["episode_set"] == "test"].copy()
    
    # Count episodes and pre-fault samples per set
    train_episodes = train_df[train_df["episode_id"] != -1]["episode_id"].nunique()
    test_episodes = test_df[test_df["episode_id"] != -1]["episode_id"].nunique()
    train_prefault = train_df["in_prefault_zone"].sum()
    test_prefault = test_df["in_prefault_zone"].sum()
    
    manifest = {
        "split_type": "prediction_episode_based",
        "n_rows": len(df),
        "n_fault_episodes": n_episodes,
        "forecast_horizon_seconds": forecast_horizon_seconds,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_fault_episodes": train_episodes,
        "test_fault_episodes": test_episodes,
        "val_fault_episodes": 0,
        "train_prefault_samples": int(train_prefault),
        "test_prefault_samples": int(test_prefault),
        "train_fault_samples": int((train_df[label_col] != 0).sum()),
        "test_fault_samples": int((test_df[label_col] != 0).sum()),
        "train_normal_samples": int((train_df[label_col] == 0).sum()),
        "val_normal_samples": int((val_df[label_col] == 0).sum()),
        "test_normal_samples": int((test_df[label_col] == 0).sum()),
        "episodes_by_class": {
            str(k): {
                "n_episodes": v["n_episodes"],
                "train": v["train"],
                "test": v["test"],
                "evaluable": v["evaluable"],
            }
            for k, v in episodes_by_class.items()
        },
        "note": "Episode-based split for Task C. Pre-fault zones go with their episode. Val=normal only.",
    }
    
    return SplitArtifacts(train_df, val_df, test_df, pd.DataFrame(), manifest)


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
            raise ValueError(f"Need at least {self.min_train_segments + 1} segments, got {n_segments}")
        
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
                val_segs = unique_segs[train_end:train_end + 1]
                
                for seg in train_segs:
                    train_idx_all.extend(seg_to_idx[seg])
                for seg in val_segs:
                    val_idx_all.extend(seg_to_idx[seg])
            
            if val_idx_all:
                yield np.array(train_idx_all), np.array(val_idx_all)
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
