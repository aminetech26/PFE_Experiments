"""Per-fold val/test DataFrame override helper.

All baseline trainers (PC-AE, PC-DLSSM, MAAT, OC-SVM, IForest, BOCD) call
load_features_for_task to load (train_df, val_df, test_df). When the K-fold
orchestrator invokes a trainer for one fold, it passes --val-parquet-override
and --test-parquet-override flags pointing to per-fold parquet slices. This
helper applies those overrides uniformly and validates schema compatibility.

Train is never overridden — semi-supervised constraint requires a fixed,
normal-only training set so the scaler and threshold are stable across folds.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger


def apply_fold_overrides(
    args: Any,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """If args carries --val/test-parquet-override, swap val/test DataFrames.

    Validates that the override parquet has all columns present in the original
    val/test DataFrames; selects only those columns (in the original order).
    """
    val_override = getattr(args, "val_parquet_override", None)
    test_override = getattr(args, "test_parquet_override", None)
    fold_id = getattr(args, "fold_id", None)

    if val_override:
        new_val = pd.read_parquet(val_override)
        missing = set(val_df.columns) - set(new_val.columns)
        if missing:
            raise RuntimeError(f"val_parquet_override missing columns: {missing}")
        logger.info("FOLD OVERRIDE: val rows {} → {} (from {})",
                    len(val_df), len(new_val), val_override)
        val_df = new_val[list(val_df.columns)].reset_index(drop=True)

    if test_override:
        new_test = pd.read_parquet(test_override)
        missing = set(test_df.columns) - set(new_test.columns)
        if missing:
            raise RuntimeError(f"test_parquet_override missing columns: {missing}")
        logger.info("FOLD OVERRIDE: test rows {} → {} (from {})",
                    len(test_df), len(new_test), test_override)
        test_df = new_test[list(test_df.columns)].reset_index(drop=True)

    if fold_id is not None:
        logger.info("FOLD ID: {}", fold_id)

    return val_df, test_df


def add_fold_override_args(parser) -> None:
    """Register --val-parquet-override / --test-parquet-override / --fold-id on parser."""
    parser.add_argument("--val-parquet-override", default=None,
                        help="Path to a parquet file whose rows replace val_df (k-fold orchestration)")
    parser.add_argument("--test-parquet-override", default=None,
                        help="Path to a parquet file whose rows replace test_df (k-fold orchestration)")
    parser.add_argument("--fold-id", type=int, default=None,
                        help="Fold index for logging / artifact naming (k-fold orchestration)")
