from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Core responsibility: Search and select a valid, temporally consistent split with class-coverage constraints.

@dataclass(frozen=True)
class SplitArtifacts:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    dropped: pd.DataFrame
    manifest: dict


def _segment_summary(df: pd.DataFrame, segment_col: str, label_col: str, time_col: str) -> pd.DataFrame:
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


def _slice_by_segments(
    df: pd.DataFrame,
    segment_order: list,
    train_end_idx: int,
    val_end_idx: int,
    gap_segments: int,
    segment_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_segments = segment_order[:train_end_idx]

    drop_after_train = segment_order[train_end_idx : min(train_end_idx + gap_segments, len(segment_order))]
    val_start_idx = train_end_idx + gap_segments
    val_segments = segment_order[val_start_idx:val_end_idx]

    drop_after_val = segment_order[val_end_idx : min(val_end_idx + gap_segments, len(segment_order))]
    test_start_idx = val_end_idx + gap_segments
    test_segments = segment_order[test_start_idx:]

    dropped_segments = set(drop_after_train + drop_after_val)

    train_df = df[df[segment_col].isin(train_segments)].copy()
    val_df = df[df[segment_col].isin(val_segments)].copy()
    test_df = df[df[segment_col].isin(test_segments)].copy()
    dropped_df = df[df[segment_col].isin(dropped_segments)].copy()

    return train_df, val_df, test_df, dropped_df


def _apply_time_embargo(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str,
    embargo_seconds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if embargo_seconds <= 0:
        empty = train_df.iloc[0:0].copy()
        return train_df, val_df, test_df, empty

    emb = pd.to_timedelta(embargo_seconds, unit="s")

    b1_train_end = train_df[time_col].max()
    b1_val_start = val_df[time_col].min()
    b2_val_end = val_df[time_col].max()
    b2_test_start = test_df[time_col].min()

    train_keep = train_df[time_col] <= (b1_train_end - emb)
    val_keep_left = val_df[time_col] >= (b1_val_start + emb)
    val_keep_right = val_df[time_col] <= (b2_val_end - emb)
    val_keep = val_keep_left & val_keep_right
    test_keep = test_df[time_col] >= (b2_test_start + emb)

    dropped = pd.concat(
        [
            train_df[~train_keep],
            val_df[~val_keep],
            test_df[~test_keep],
        ],
        axis=0,
        ignore_index=False,
    ).sort_values(time_col)

    return (
        train_df[train_keep].copy(),
        val_df[val_keep].copy(),
        test_df[test_keep].copy(),
        dropped.copy(),
    )


def _class_support_ok(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    min_train_count: int,
    min_val_count: int,
) -> bool:
    train_counts = train_df[label_col].value_counts()
    val_counts = val_df[label_col].value_counts()

    all_classes = sorted(pd.concat([train_df[label_col], val_df[label_col]]).dropna().unique().tolist())
    for c in all_classes:
        if train_counts.get(c, 0) < min_train_count:
            return False
        if val_counts.get(c, 0) < min_val_count:
            return False
    return True


def segment_aware_temporal_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    label_col: str,
    segment_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    gap_segments: int,
    embargo_seconds: int = 0,
    min_train_count: int = 1,
    min_val_count: int = 1,
) -> SplitArtifacts:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    ordered = df.sort_values(time_col).reset_index(drop=True)
    seg_df = _segment_summary(ordered, segment_col=segment_col, label_col=label_col, time_col=time_col)
    segments = seg_df[segment_col].tolist()

    if len(segments) < (4 + 2 * gap_segments):
        raise ValueError(
            f"Not enough contiguous segments ({len(segments)}) for split with gap_segments={gap_segments}."
        )

    total_rows = len(ordered)
    target_train = train_ratio * total_rows
    target_val = val_ratio * total_rows

    best = None
    best_score = float("inf")

    # Boundaries are segment indices to preserve natural gaps.
    for train_end_idx in range(1, len(segments) - 2):
        min_val_start = train_end_idx + gap_segments + 1
        if min_val_start >= len(segments) - 1:
            break

        for val_end_idx in range(min_val_start + 1, len(segments)):
            train_df, val_df, test_df, dropped_df = _slice_by_segments(
                ordered,
                segments,
                train_end_idx,
                val_end_idx,
                gap_segments,
                segment_col,
            )

            if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
                continue

            # Enforce time-based purge/embargo around split boundaries.
            train_df, val_df, test_df, embargo_dropped = _apply_time_embargo(
                train_df,
                val_df,
                test_df,
                time_col=time_col,
                embargo_seconds=embargo_seconds,
            )

            if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
                continue

            if not _class_support_ok(
                train_df,
                val_df,
                label_col=label_col,
                min_train_count=min_train_count,
                min_val_count=min_val_count,
            ):
                continue

            if len(test_df[label_col].dropna().unique()) == 0:
                continue

            dropped_df = pd.concat([dropped_df, embargo_dropped], axis=0, ignore_index=False)
            dropped_df = dropped_df.sort_values(time_col)

            train_err = abs(len(train_df) - target_train) / total_rows
            val_err = abs(len(val_df) - target_val) / total_rows
            drop_penalty = len(dropped_df) / total_rows
            score = train_err + val_err + 0.25 * drop_penalty

            if score < best_score:
                best = (train_df, val_df, test_df, dropped_df, train_end_idx, val_end_idx)
                best_score = score

    if best is None:
        raise RuntimeError(
            "Could not find a feasible temporal split satisfying class support thresholds. "
            "Relax min_train_count/min_val_count or reduce gap_segments."
        )

    train_df, val_df, test_df, dropped_df, train_end_idx, val_end_idx = best

    manifest = {
        "n_rows": total_rows,
        "n_segments": len(segments),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "dropped_rows": int(len(dropped_df)),
        "gap_segments": int(gap_segments),
        "embargo_seconds": int(embargo_seconds),
        "min_train_count": int(min_train_count),
        "min_val_count": int(min_val_count),
        "boundary_train_end_segment_idx": int(train_end_idx),
        "boundary_val_end_segment_idx": int(val_end_idx),
        "train_start": str(train_df[time_col].min()),
        "train_end": str(train_df[time_col].max()),
        "val_start": str(val_df[time_col].min()),
        "val_end": str(val_df[time_col].max()),
        "test_start": str(test_df[time_col].min()),
        "test_end": str(test_df[time_col].max()),
        "train_class_counts": train_df[label_col].value_counts().sort_index().to_dict(),
        "val_class_counts": val_df[label_col].value_counts().sort_index().to_dict(),
        "test_class_counts": test_df[label_col].value_counts().sort_index().to_dict(),
    }

    return SplitArtifacts(
        train=train_df,
        val=val_df,
        test=test_df,
        dropped=dropped_df,
        manifest=manifest,
    )
