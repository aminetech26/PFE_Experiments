"""Episode-stratified K-fold CV over a val+test pool.

Train is held fixed (semi-supervised: normal-only, scaler/threshold tied to it).
Only val/test rotate. Per class, episodes are partitioned into K disjoint groups;
in fold k, group k becomes the fold's TEST set, the remaining (K-1) groups become
the fold's VAL set. Each episode lands in test exactly once across the K folds.

Design rationale:
- We rotate at the EPISODE level (not row level) so an episode never appears in
  both val and test of the same fold — that would inflate per-class scores.
- Stratification is per class: each class's episode count is divided evenly across
  folds, so every fold has a representative sample of every class.
- (K-1)/K of the pool is the val set, used for early stopping & HPO-config
  selection. This gives early stopping enough signal even on rare classes.
- Test = 1/K. Across the K folds, every fault episode is scored exactly once on
  some test fold, so the per-class aggregate uses the *full* episode support
  (just distributed over folds), with K independent variance samples.

Public API:
    make_episode_stratified_folds(val_df, test_df, n_folds, seed, label_col, group_col)
        → list of (val_fold_df, test_fold_df) pairs, length n_folds
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class FoldAssignment:
    """One fold's episode-id sets."""
    fold_idx: int
    val_episode_ids: list
    test_episode_ids: list


def _episode_class_table(df: pd.DataFrame, label_col: str, group_col: str) -> pd.DataFrame:
    """One row per episode: episode_id, class (mode of label_col)."""
    rows = []
    for ep_id, sub in df.groupby(group_col):
        cls = int(sub[label_col].mode().iat[0])
        rows.append({"episode_id": ep_id, "class": cls, "n_rows": len(sub)})
    return pd.DataFrame(rows)


def _partition_episodes_for_class(
    episode_ids: list, n_folds: int, rng: np.random.Generator
) -> list[list]:
    """Shuffle then split into n_folds near-equal groups."""
    eps = list(episode_ids)
    rng.shuffle(eps)
    # np.array_split handles uneven counts gracefully
    groups = [list(g) for g in np.array_split(np.array(eps, dtype=object), n_folds)]
    return groups


def assign_folds(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_folds: int,
    seed: int,
    label_col: str = "label",
    group_col: str = "episode_id",
) -> list[FoldAssignment]:
    """Compute the K fold assignments at episode level.

    Returns one FoldAssignment per fold. Episode IDs are unique across val+test
    in the pool; each id appears in TEST exactly once (across the K folds) and
    in VAL K-1 times.
    """
    pool = pd.concat([val_df, test_df], ignore_index=True)
    if group_col not in pool.columns:
        raise RuntimeError(f"Episode column '{group_col}' not found in val/test DataFrames")
    if label_col not in pool.columns:
        raise RuntimeError(f"Label column '{label_col}' not found in val/test DataFrames")

    table = _episode_class_table(pool, label_col, group_col)
    classes = sorted(table["class"].unique().tolist())
    logger.info("Episode pool: {} unique episodes across {} classes", len(table), len(classes))
    for cls in classes:
        logger.info("  class {} → {} episodes", cls, int((table["class"] == cls).sum()))

    rng = np.random.default_rng(seed)
    # Per class: split episodes into n_folds groups
    per_class_groups: dict[int, list[list]] = {}
    for cls in classes:
        ep_ids = table.loc[table["class"] == cls, "episode_id"].tolist()
        per_class_groups[cls] = _partition_episodes_for_class(ep_ids, n_folds, rng)

    # Compose folds: fold k's TEST = group k from every class; VAL = the rest
    folds: list[FoldAssignment] = []
    for k in range(n_folds):
        test_ids: list = []
        val_ids: list = []
        for cls in classes:
            groups = per_class_groups[cls]
            for j, g in enumerate(groups):
                if j == k:
                    test_ids.extend(g)
                else:
                    val_ids.extend(g)
        folds.append(FoldAssignment(fold_idx=k, val_episode_ids=val_ids, test_episode_ids=test_ids))
        # Sanity log
        test_cls = table[table["episode_id"].isin(test_ids)].groupby("class")["episode_id"].nunique().to_dict()
        val_cls = table[table["episode_id"].isin(val_ids)].groupby("class")["episode_id"].nunique().to_dict()
        logger.info("Fold {}: test_episodes={} val_episodes={} per-class test={} val={}",
                    k, len(test_ids), len(val_ids), test_cls, val_cls)

    return folds


def materialize_fold(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold: FoldAssignment,
    group_col: str = "episode_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the pool by the fold's episode-id sets.

    Returns (val_fold_df, test_fold_df).
    """
    pool = pd.concat([val_df, test_df], ignore_index=True)
    val_mask = pool[group_col].isin(fold.val_episode_ids)
    test_mask = pool[group_col].isin(fold.test_episode_ids)
    val_fold = pool[val_mask].reset_index(drop=True)
    test_fold = pool[test_mask].reset_index(drop=True)
    return val_fold, test_fold


def make_episode_stratified_folds(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_folds: int,
    seed: int = 42,
    label_col: str = "label",
    group_col: str = "episode_id",
) -> list[tuple[pd.DataFrame, pd.DataFrame, FoldAssignment]]:
    """Convenience wrapper: returns (val_fold_df, test_fold_df, FoldAssignment) per fold."""
    assignments = assign_folds(val_df, test_df, n_folds, seed, label_col, group_col)
    out = []
    for fa in assignments:
        v, t = materialize_fold(val_df, test_df, fa, group_col)
        out.append((v, t, fa))
    return out
