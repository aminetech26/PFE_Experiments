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
    class_requirements: dict,
) -> bool:
    train_counts = train_df[label_col].value_counts()
    val_counts = val_df[label_col].value_counts()

    for c, req in class_requirements.items():
        if bool(req["is_rare"]):
            continue
        if train_counts.get(c, 0) < int(req["required_train"]):
            return False
        if val_counts.get(c, 0) < int(req["required_val"]):
            return False
    return True


def _build_class_requirements(
    df: pd.DataFrame,
    *,
    label_col: str,
    min_train_count: int,
    min_val_count: int,
    train_min_frac: float,
    val_min_frac: float,
    rare_class_threshold: int,
) -> dict:
    counts = df[label_col].value_counts(dropna=True).to_dict()
    requirements = {}
    for cls, total in counts.items():
        total_i = int(total)
        is_rare = total_i < int(rare_class_threshold)
        req_train = max(int(min_train_count), int(np.ceil(float(train_min_frac) * total_i)))
        req_val = max(int(min_val_count), int(np.ceil(float(val_min_frac) * total_i)))
        requirements[cls] = {
            "total": total_i,
            "is_rare": bool(is_rare),
            "required_train": int(req_train),
            "required_val": int(req_val),
        }
    return requirements


def _class_risk_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    label_col: str,
    class_requirements: dict,
) -> dict:
    train_counts = train_df[label_col].value_counts().to_dict()
    val_counts = val_df[label_col].value_counts().to_dict()
    test_counts = test_df[label_col].value_counts().to_dict()

    classes = sorted(class_requirements.keys())
    rare_classes = [c for c in classes if bool(class_requirements[c]["is_rare"])]
    enforced_classes = [c for c in classes if not bool(class_requirements[c]["is_rare"])]
    unseen_in_train = [
        c for c in classes if int(train_counts.get(c, 0)) == 0 and int(val_counts.get(c, 0)) + int(test_counts.get(c, 0)) > 0
    ]
    unseen_in_val = [
        c for c in classes if int(val_counts.get(c, 0)) == 0 and int(train_counts.get(c, 0)) + int(test_counts.get(c, 0)) > 0
    ]

    return {
        "rare_classes": [float(c) if isinstance(c, (int, float, np.floating, np.integer)) else str(c) for c in rare_classes],
        "enforced_classes": [float(c) if isinstance(c, (int, float, np.floating, np.integer)) else str(c) for c in enforced_classes],
        "unseen_in_train": [float(c) if isinstance(c, (int, float, np.floating, np.integer)) else str(c) for c in unseen_in_train],
        "unseen_in_val": [float(c) if isinstance(c, (int, float, np.floating, np.integer)) else str(c) for c in unseen_in_val],
    }


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
    train_min_frac: float = 0.01,
    val_min_frac: float = 0.005,
    rare_class_threshold: int = 30,
) -> SplitArtifacts:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    ordered = df.sort_values(time_col).reset_index(drop=True)
    seg_df = _segment_summary(ordered, segment_col=segment_col, label_col=label_col, time_col=time_col)
    segments = seg_df[segment_col].tolist()
    n_seg = len(segments)

    if n_seg < (4 + 2 * gap_segments):
        raise ValueError(
            f"Not enough contiguous segments ({n_seg}) for split with gap_segments={gap_segments}."
        )

    total_rows = len(ordered)
    target_train = train_ratio * total_rows
    target_val = val_ratio * total_rows
    class_requirements = _build_class_requirements(
        ordered,
        label_col=label_col,
        min_train_count=min_train_count,
        min_val_count=min_val_count,
        train_min_frac=train_min_frac,
        val_min_frac=val_min_frac,
        rare_class_threshold=rare_class_threshold,
    )

    # Pre-compute cumulative row counts: cum_rows[i] = total rows in the first i segments.
    # This lets us compute train/val/test sizes in O(1) instead of O(rows) per iteration.
    seg_rows = seg_df["n_rows"].values
    cum_rows = np.concatenate([[0], np.cumsum(seg_rows)]).astype(np.int64)

    # Pre-compute cumulative class counts for enforced (non-rare) classes only.
    # One vectorized groupby → O(rows + n_seg * n_cls) instead of O(N² * rows).
    enforced_classes = [c for c, req in class_requirements.items() if not req["is_rare"]]
    n_cls = len(enforced_classes)
    if n_cls > 0:
        seg_class_raw = (
            ordered.groupby(segment_col)[label_col]
            .value_counts()
            .unstack(fill_value=0)
            .reindex(segments, fill_value=0)
        )
        seg_class_enf = seg_class_raw.reindex(columns=enforced_classes, fill_value=0).to_numpy(dtype=np.int64)
        # cum_class[i] = total class counts for the first i segments; shape (n_seg+1, n_cls)
        cum_class = np.vstack(
            [np.zeros((1, n_cls), dtype=np.int64), np.cumsum(seg_class_enf, axis=0)]
        )
        required_train_arr = np.array(
            [int(class_requirements[c]["required_train"]) for c in enforced_classes], dtype=np.int64
        )
        required_val_arr = np.array(
            [int(class_requirements[c]["required_val"]) for c in enforced_classes], dtype=np.int64
        )
    else:
        cum_class = None
        required_train_arr = required_val_arr = None

    best_boundary = None
    best_score = float("inf")

    # Search over segment boundaries using only O(1) cumulative-count lookups.
    for train_end_idx in range(1, n_seg - 2):
        val_start_idx = train_end_idx + gap_segments
        if val_start_idx + 1 >= n_seg:
            break

        train_rows = int(cum_rows[train_end_idx])
        if train_rows == 0:
            continue

        # Early-exit: if this train boundary already fails class support, skip.
        if n_cls > 0 and np.any(cum_class[train_end_idx] < required_train_arr):
            continue

        for val_end_idx in range(val_start_idx + 1, n_seg):
            test_start_idx = val_end_idx + gap_segments
            if test_start_idx >= n_seg:
                continue

            val_rows = int(cum_rows[val_end_idx] - cum_rows[val_start_idx])
            test_rows = int(cum_rows[n_seg] - cum_rows[test_start_idx])

            if val_rows == 0 or test_rows == 0:
                continue

            if n_cls > 0:
                val_cls_counts = cum_class[val_end_idx] - cum_class[val_start_idx]
                if np.any(val_cls_counts < required_val_arr):
                    continue

            train_err = abs(train_rows - target_train) / total_rows
            val_err = abs(val_rows - target_val) / total_rows
            score = train_err + val_err

            if score < best_score:
                best_boundary = (train_end_idx, val_end_idx)
                best_score = score

    if best_boundary is None:
        raise RuntimeError(
            "Could not find a feasible temporal split satisfying class support thresholds. "
            "Relax class support parameters or reduce gap_segments."
        )

    train_end_idx, val_end_idx = best_boundary

    # Apply full DataFrame operations exactly once for the selected boundaries.
    train_df, val_df, test_df, dropped_df = _slice_by_segments(
        ordered, segments, train_end_idx, val_end_idx, gap_segments, segment_col
    )
    train_df, val_df, test_df, embargo_dropped = _apply_time_embargo(
        train_df, val_df, test_df, time_col=time_col, embargo_seconds=embargo_seconds
    )
    dropped_df = pd.concat(
        [dropped_df, embargo_dropped], axis=0, ignore_index=False
    ).sort_values(time_col)

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
        "train_min_frac": float(train_min_frac),
        "val_min_frac": float(val_min_frac),
        "rare_class_threshold": int(rare_class_threshold),
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
        "class_requirements": {
            str(c): {
                "total": int(req["total"]),
                "is_rare": bool(req["is_rare"]),
                "required_train": int(req["required_train"]),
                "required_val": int(req["required_val"]),
            }
            for c, req in class_requirements.items()
        },
    }

    risk = _class_risk_report(
        train_df,
        val_df,
        test_df,
        label_col=label_col,
        class_requirements=class_requirements,
    )
    risk["rare_class_threshold"] = int(rare_class_threshold)
    manifest["class_risk"] = risk

    return SplitArtifacts(
        train=train_df,
        val=val_df,
        test=test_df,
        dropped=dropped_df,
        manifest=manifest,
    )
