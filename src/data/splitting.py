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

from collections.abc import Iterator
from dataclasses import dataclass

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
    faults = [v for v in labels if v != 0.0]
    return faults[0] if faults else None


def _ratio_error(actual: float, target: float) -> float:
    """Absolute deviation between achieved and target ratio."""
    return abs(actual - target)


def _choose_temporal_boundaries(
    ordered_ids: list,
    ordered_rows: list[int],
    train_ratio: float,
    val_ratio: float,
    allow_empty_train: bool = False,
) -> tuple[list, list, list, dict]:
    """Choose chronology-preserving boundaries using both episode and row ratios."""
    n = len(ordered_ids)
    min_train_units = 0 if allow_empty_train else 1
    if n == 0:
        return [], [], [], {"n_units": 0, "status": "empty"}
    if n == 1:
        if min_train_units == 0:
            return [], ordered_ids, [], {"n_units": 1, "status": "single_unit_empty_train"}
        return ordered_ids, [], [], {"n_units": 1, "status": "single_unit"}
    if n == 2:
        if min_train_units == 0:
            return (
                [],
                [ordered_ids[0]],
                [ordered_ids[1]],
                {"n_units": 2, "status": "two_units_empty_train"},
            )
        return [ordered_ids[0]], [], [ordered_ids[1]], {"n_units": 2, "status": "two_units"}

    total_rows = int(sum(ordered_rows))
    cumulative_rows = np.cumsum(ordered_rows)
    target_test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    best: tuple[float, int, int] | None = None

    if allow_empty_train and train_ratio == 0.0:
        train_end_values = [0]
    else:
        train_end_values = range(min_train_units, n - 1)

    for train_end in train_end_values:
        for val_end in range(train_end + 1, n):
            train_count = train_end
            val_count = val_end - train_end
            test_count = n - val_end
            if val_count <= 0 or test_count <= 0:
                continue

            train_rows = int(cumulative_rows[train_end - 1])
            val_rows = int(cumulative_rows[val_end - 1] - cumulative_rows[train_end - 1])
            test_rows = int(total_rows - cumulative_rows[val_end - 1])

            # Rows carry more learning signal than episode counts, so row errors get 2x
            # weight. The 1x episode term still penalises extreme episode imbalance (e.g.
            # 1 val episode vs 10 test episodes) without overriding honest row coverage.
            score = (
                2.0 * _ratio_error(train_rows / total_rows, train_ratio)
                + 2.0 * _ratio_error(val_rows / total_rows, val_ratio)
                + 2.0 * _ratio_error(test_rows / total_rows, target_test_ratio)
                + 1.0 * _ratio_error(train_count / n, train_ratio)
                + 1.0 * _ratio_error(val_count / n, val_ratio)
                + 1.0 * _ratio_error(test_count / n, target_test_ratio)
            )

            candidate = (score, train_end, val_end)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        train_end = max(min_train_units, int(n * train_ratio))
        val_end = min(n - 1, train_end + max(1, int(n * val_ratio)))
    else:
        _, train_end, val_end = best

    train_ids = ordered_ids[:train_end]
    val_ids = ordered_ids[train_end:val_end]
    test_ids = ordered_ids[val_end:]

    train_rows = int(sum(ordered_rows[:train_end]))
    val_rows = int(sum(ordered_rows[train_end:val_end]))
    test_rows = int(sum(ordered_rows[val_end:]))

    diagnostics = {
        "n_units": n,
        "total_rows": total_rows,
        "target_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": target_test_ratio,
        },
        "achieved_episode_ratios": {
            "train": len(train_ids) / n,
            "val": len(val_ids) / n,
            "test": len(test_ids) / n,
        },
        "achieved_row_ratios": {
            "train": train_rows / total_rows if total_rows else None,
            "val": val_rows / total_rows if total_rows else None,
            "test": test_rows / total_rows if total_rows else None,
        },
        "episode_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "row_counts": {
            "train": train_rows,
            "val": val_rows,
            "test": test_rows,
        },
        "status": "optimized_jointly_for_episode_and_row_ratios",
    }
    return train_ids, val_ids, test_ids, diagnostics


def _build_split_support_report(
    df: pd.DataFrame,
    segment_col: str,
    label_col: str,
    split_frames: dict[str, pd.DataFrame],
) -> dict:
    """Report achieved support by class in both episode and row space."""
    episode_labels = (
        df.groupby(segment_col, observed=True)[label_col].first().reset_index(drop=False)
    )
    all_labels = sorted(df[label_col].dropna().unique().tolist())
    report: dict[str, dict] = {}

    for label in all_labels:
        total_rows = int((df[label_col] == label).sum())
        total_episodes = int((episode_labels[label_col] == label).sum())
        per_split = {}
        for split_name, split_df in split_frames.items():
            split_episode_labels = (
                split_df.groupby(segment_col, observed=True)[label_col]
                .first()
                .reset_index(drop=False)
            )
            split_rows = int((split_df[label_col] == label).sum())
            split_episodes = int((split_episode_labels[label_col] == label).sum())
            per_split[split_name] = {
                "rows": split_rows,
                "row_ratio_within_class": split_rows / total_rows if total_rows else None,
                "episodes": split_episodes,
                "episode_ratio_within_class": (
                    split_episodes / total_episodes if total_episodes else None
                ),
            }
        report[str(label)] = {
            "total_rows": total_rows,
            "total_episodes": total_episodes,
            "splits": per_split,
        }
    return report


def _build_multilabel_unit_support_report(
    df: pd.DataFrame,
    segment_col: str,
    label_col: str,
    split_frames: dict[str, pd.DataFrame],
) -> dict:
    """Report row support plus unit-presence support for mixed-label grouped units."""
    unit_labels = df.groupby(segment_col, observed=True)[label_col].apply(
        lambda x: set(pd.Series(x).dropna().tolist())
    )
    all_labels = sorted(df[label_col].dropna().unique().tolist())
    report: dict[str, dict] = {}

    for label in all_labels:
        total_rows = int((df[label_col] == label).sum())
        total_units = int(sum(label in labels for labels in unit_labels))
        per_split = {}
        for split_name, split_df in split_frames.items():
            split_unit_labels = split_df.groupby(segment_col, observed=True)[label_col].apply(
                lambda x: set(pd.Series(x).dropna().tolist())
            )
            split_rows = int((split_df[label_col] == label).sum())
            split_units = int(sum(label in labels for labels in split_unit_labels))
            per_split[split_name] = {
                "rows": split_rows,
                "row_ratio_within_class": split_rows / total_rows if total_rows else None,
                "units": split_units,
                "unit_ratio_within_class": split_units / total_units if total_units else None,
            }
        report[str(label)] = {
            "total_rows": total_rows,
            "total_units": total_units,
            "splits": per_split,
        }
    return report


def _apply_forward_purge(
    train_ids: list,
    val_ids: list,
    test_ids: list,
    purge_units: int,
) -> tuple[list, list, list, list]:
    """Drop the earliest units of downstream partitions to create explicit boundary gaps."""
    if purge_units <= 0:
        return train_ids, val_ids, test_ids, []

    drop_val = val_ids[: min(purge_units, len(val_ids))]
    drop_test = test_ids[: min(purge_units, len(test_ids))]
    kept_val = val_ids[len(drop_val) :]
    kept_test = test_ids[len(drop_test) :]
    dropped_ids = [*drop_val, *drop_test]
    return train_ids, kept_val, kept_test, dropped_ids


def _forward_fractional_boundary_purge(
    frame: pd.DataFrame,
    segment_col: str,
    time_col: str,
    purge_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Drop the earliest fraction of elapsed time from the first grouped unit in a partition."""
    if frame.empty or purge_fraction <= 0:
        return (
            frame,
            pd.DataFrame(columns=frame.columns),
            {
                "applied": False,
                "purged_rows": 0,
                "purged_group": None,
                "purge_fraction": purge_fraction,
                "purge_basis": "elapsed_time_within_group",
                "cutoff_timestamp": None,
            },
        )

    first_group = frame.sort_values([segment_col, time_col])[segment_col].iloc[0]
    first_group_df = frame[frame[segment_col] == first_group].sort_values(time_col)
    start_ts = first_group_df[time_col].iloc[0]
    end_ts = first_group_df[time_col].iloc[-1]
    elapsed_seconds = (end_ts - start_ts).total_seconds()

    if elapsed_seconds <= 0:
        n_drop = int(np.floor(len(first_group_df) * purge_fraction))
        if n_drop <= 0 and len(first_group_df) > 1:
            n_drop = 1
        drop_index = first_group_df.index[:n_drop] if n_drop > 0 else first_group_df.index[:0]
        cutoff_ts = first_group_df[time_col].iloc[n_drop - 1] if n_drop > 0 else None
    else:
        cutoff_ts = start_ts + pd.to_timedelta(elapsed_seconds * purge_fraction, unit="s")
        drop_index = first_group_df.index[first_group_df[time_col] <= cutoff_ts]

    if len(drop_index) <= 0:
        return (
            frame,
            pd.DataFrame(columns=frame.columns),
            {
                "applied": False,
                "purged_rows": 0,
                "purged_group": int(first_group)
                if isinstance(first_group, (int, np.integer))
                else first_group,
                "purge_fraction": purge_fraction,
                "purge_basis": "elapsed_time_within_group",
                "cutoff_timestamp": cutoff_ts.isoformat() if cutoff_ts is not None else None,
            },
        )

    purged = frame.loc[drop_index].copy()
    kept = frame.drop(index=drop_index).copy()
    return (
        kept,
        purged,
        {
            "applied": True,
            "purged_rows": int(len(purged)),
            "purged_group": int(first_group)
            if isinstance(first_group, (int, np.integer))
            else first_group,
            "purge_fraction": float(purge_fraction),
            "purge_basis": "elapsed_time_within_group",
            "cutoff_timestamp": cutoff_ts.isoformat() if cutoff_ts is not None else None,
        },
    )


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
        class_rows = class_seg_df["n_rows"].astype(int).tolist()

        train_class, val_class, test_class, split_diag = _choose_temporal_boundaries(
            class_segs,
            class_rows,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        train_segs.extend(train_class)
        val_segs.extend(val_class)
        test_segs.extend(test_class)

        class_temporal_info[fault_class] = {
            "n_segments": len(class_segs),
            "train": train_class,
            "val": val_class,
            "test": test_class,
            "support_diagnostics": split_diag,
        }

    # Distribute normal segments (temporal order)
    # Normal class: large-n naive slicing adequate; evaluability concern is on fault classes.
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

    support_report = _build_multilabel_unit_support_report(
        df,
        segment_col,
        label_col,
        {"train": train_df, "val": val_df, "test": test_df},
    )

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
        "support_report": support_report,
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

    seg_summary["has_fault"] = seg_summary["labels"].apply(lambda x: any(v != 0.0 for v in x))
    seg_summary["primary_fault"] = seg_summary["labels"].apply(_get_primary_fault)

    fault_segments = seg_summary[seg_summary["has_fault"]].copy()
    normal_segments = seg_summary[~seg_summary["has_fault"]].copy()

    # Train: normal-only (temporal order)
    # Normal class: large-n naive slicing adequate; evaluability concern is on fault classes.
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

    fault_support_info = {}
    for fault_class in sorted(fault_segments["primary_fault"].unique()):
        class_seg_df = fault_segments[fault_segments["primary_fault"] == fault_class].sort_values(
            "start"
        )
        class_segs = class_seg_df[segment_col].tolist()
        class_rows = class_seg_df["n_rows"].astype(int).tolist()

        _, val_class, test_class, split_diag = _choose_temporal_boundaries(
            class_segs,
            class_rows,
            train_ratio=0.0,
            val_ratio=0.5,
            allow_empty_train=True,
        )
        if not val_class and test_class:
            val_class = test_class[:1]
            test_class = test_class[1:]
        elif not test_class and val_class:
            test_class = val_class[-1:]
            val_class = val_class[:-1]

        val_fault.extend(val_class)
        test_fault.extend(test_class)
        fault_support_info[str(fault_class)] = split_diag

    train_segs = train_normal
    val_segs = val_normal + val_fault
    test_segs = test_normal + test_fault

    train_df = df[df[segment_col].isin(train_segs)].copy()
    val_df = df[df[segment_col].isin(val_segs)].copy()
    test_df = df[df[segment_col].isin(test_segs)].copy()

    # Segment boundaries provide isolation - no global embargo needed
    dropped = pd.DataFrame()

    support_report = _build_multilabel_unit_support_report(
        df,
        segment_col,
        label_col,
        {"train": train_df, "val": val_df, "test": test_df},
    )

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
        "fault_support_diagnostics": fault_support_info,
        "train_class_counts": train_df[label_col].value_counts().sort_index().to_dict(),
        "val_class_counts": val_df[label_col].value_counts().sort_index().to_dict(),
        "test_class_counts": test_df[label_col].value_counts().sort_index().to_dict(),
        "support_report": support_report,
        "note": "Train=normal-only (temporal). Val/Test=normal+faults. Segment boundaries provide isolation.",
    }

    return SplitArtifacts(train_df, val_df, test_df, dropped, manifest)


def blocked_temporal_split(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    label_col: str = "label",
    time_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_units: int = 1,
    boundary_purge_fraction: float = 0.0,
    unit_name: str = "days",
) -> SplitArtifacts:
    """Chronological blocked split with forward purge between partitions."""
    seg_summary = _segment_summary(df, segment_col, label_col, time_col)
    ordered_ids = seg_summary[segment_col].tolist()
    ordered_rows = seg_summary["n_rows"].astype(int).tolist()

    train_ids, val_ids, test_ids, split_diag = _choose_temporal_boundaries(
        ordered_ids,
        ordered_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_ids, val_ids, test_ids, purged_ids = _apply_forward_purge(
        train_ids,
        val_ids,
        test_ids,
        purge_units=purge_units,
    )

    train_df = df[df[segment_col].isin(train_ids)].copy()
    val_df = df[df[segment_col].isin(val_ids)].copy()
    test_df = df[df[segment_col].isin(test_ids)].copy()
    dropped = df[df[segment_col].isin(purged_ids)].copy()

    val_df, purged_val_rows, val_boundary_purge = _forward_fractional_boundary_purge(
        val_df,
        segment_col=segment_col,
        time_col=time_col,
        purge_fraction=boundary_purge_fraction,
    )
    test_df, purged_test_rows, test_boundary_purge = _forward_fractional_boundary_purge(
        test_df,
        segment_col=segment_col,
        time_col=time_col,
        purge_fraction=boundary_purge_fraction,
    )
    dropped = pd.concat(
        [dropped, purged_val_rows, purged_test_rows], ignore_index=False
    ).sort_index()

    support_report = _build_multilabel_unit_support_report(
        df,
        segment_col,
        label_col,
        {"train": train_df, "val": val_df, "test": test_df},
    )

    manifest = {
        "split_type": "blocked_temporal_purged",
        "unit_name": unit_name,
        "n_rows": len(df),
        "n_segments": len(seg_summary),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "dropped_rows": len(dropped),
        "train_segments": len(train_ids),
        "val_segments": len(val_ids),
        "test_segments": len(test_ids),
        "purged_segments": len(purged_ids),
        "purge_units": purge_units,
        "boundary_purge_fraction": boundary_purge_fraction,
        "boundary_purge": {
            "val": val_boundary_purge,
            "test": test_boundary_purge,
        },
        "global_temporal_diagnostics": split_diag,
        "train_class_counts": train_df[label_col].value_counts().sort_index().to_dict(),
        "val_class_counts": val_df[label_col].value_counts().sort_index().to_dict(),
        "test_class_counts": test_df[label_col].value_counts().sort_index().to_dict(),
        "support_report": support_report,
        "note": f"Pure chronological blocked split over {unit_name} with forward purge.",
    }
    return SplitArtifacts(train_df, val_df, test_df, dropped, manifest)


def blocked_semisup_split(
    df: pd.DataFrame,
    segment_col: str = "segment_id",
    label_col: str = "label",
    time_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_units: int = 1,
    boundary_purge_fraction: float = 0.0,
    unit_name: str = "days",
) -> SplitArtifacts:
    """Blocked chronological split for semi-supervised learning with normal-only train rows."""
    seg_summary = _segment_summary(df, segment_col, label_col, time_col)
    ordered_ids = seg_summary[segment_col].tolist()
    ordered_rows = seg_summary["n_rows"].astype(int).tolist()

    train_ids, val_ids, test_ids, split_diag = _choose_temporal_boundaries(
        ordered_ids,
        ordered_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_ids, val_ids, test_ids, purged_ids = _apply_forward_purge(
        train_ids,
        val_ids,
        test_ids,
        purge_units=purge_units,
    )

    train_all = df[df[segment_col].isin(train_ids)].copy()
    train_df = train_all[train_all[label_col] == 0.0].copy()
    train_fault_rows = train_all[train_all[label_col] != 0.0].copy()
    val_df = df[df[segment_col].isin(val_ids)].copy()
    test_df = df[df[segment_col].isin(test_ids)].copy()
    purged_df = df[df[segment_col].isin(purged_ids)].copy()

    val_df, purged_val_rows, val_boundary_purge = _forward_fractional_boundary_purge(
        val_df,
        segment_col=segment_col,
        time_col=time_col,
        purge_fraction=boundary_purge_fraction,
    )
    test_df, purged_test_rows, test_boundary_purge = _forward_fractional_boundary_purge(
        test_df,
        segment_col=segment_col,
        time_col=time_col,
        purge_fraction=boundary_purge_fraction,
    )
    dropped = pd.concat([purged_df, train_fault_rows], ignore_index=False).sort_index()
    dropped = pd.concat(
        [dropped, purged_val_rows, purged_test_rows], ignore_index=False
    ).sort_index()

    support_report = _build_multilabel_unit_support_report(
        df,
        segment_col,
        label_col,
        {"train": train_df, "val": val_df, "test": test_df},
    )

    manifest = {
        "split_type": "blocked_semisup_purged",
        "unit_name": unit_name,
        "n_rows": len(df),
        "n_segments": len(seg_summary),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "dropped_rows": len(dropped),
        "train_segments": len(train_ids),
        "val_segments": len(val_ids),
        "test_segments": len(test_ids),
        "purged_segments": len(purged_ids),
        "purge_units": purge_units,
        "boundary_purge_fraction": boundary_purge_fraction,
        "boundary_purge": {
            "val": val_boundary_purge,
            "test": test_boundary_purge,
        },
        "train_fault_rows_removed": len(train_fault_rows),
        "global_temporal_diagnostics": split_diag,
        "train_class_counts": train_df[label_col].value_counts().sort_index().to_dict(),
        "val_class_counts": val_df[label_col].value_counts().sort_index().to_dict(),
        "test_class_counts": test_df[label_col].value_counts().sort_index().to_dict(),
        "support_report": support_report,
        "note": (
            f"Pure chronological blocked split over {unit_name}; train retains only normal rows, "
            "and boundary-adjacent downstream units are purged."
        ),
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
        x,
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

    def get_n_splits(self, x=None, y=None, groups=None) -> int:
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
        x,
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

    def get_n_splits(self, x=None, y=None, groups=None) -> int:
        return self.n_splits
