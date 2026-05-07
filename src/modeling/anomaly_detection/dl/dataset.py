from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_GROUP_COL_FALLBACK = ("episode_id", "segment_id", "operating_day_id")


def _resolve_group_col(df: pd.DataFrame) -> str | None:
    for col in _GROUP_COL_FALLBACK:
        if col in df.columns and df[col].notna().any():
            return col
    return None


class TimeSeriesDataset(Dataset):
    """Grouped sliding-window dataset for anomaly DL.

    Windows never cross group boundaries (episode_id → segment_id → operating_day_id).
    normal_only=True: filters rows to label==0 (for semisup train split).
    Returns (x: Tensor[W, F], label: scalar int) where label is the center-step label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        win_size: int,
        stride: int = 1,
        normal_only: bool = False,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.stride = stride

        if normal_only:
            n_before = len(df)
            df = df[df[label_col] == 0].copy()
            n_after = len(df)
            n_dropped = n_before - n_after
            if n_dropped > 0:
                import warnings
                warnings.warn(
                    f"TimeSeriesDataset (normal_only=True): dropped {n_dropped} non-normal rows "
                    f"from {n_before} total.",
                    stacklevel=2,
                )

        group_col = _resolve_group_col(df)

        self._features: np.ndarray = df[feature_cols].to_numpy(dtype=np.float32)
        self._labels: np.ndarray = (df[label_col].to_numpy() != 0).astype(np.int64)

        # Build (start_idx, end_idx) window indices that don't cross group boundaries
        self._windows: list[tuple[int, int]] = []
        if group_col is not None:
            group_values = df[group_col].to_numpy()
            # Find contiguous same-group segments
            str_vals = group_values.astype(str)
            boundaries = np.where(str_vals[1:] != str_vals[:-1])[0] + 1
            segment_starts = np.concatenate([[0], boundaries])
            segment_ends = np.concatenate([boundaries, [len(df)]])
        else:
            # No group column — treat entire dataframe as one segment
            segment_starts = np.array([0])
            segment_ends = np.array([len(df)])

        for seg_start, seg_end in zip(segment_starts, segment_ends):
            seg_len = seg_end - seg_start
            if seg_len < win_size:
                continue
            for start in range(0, seg_len - win_size + 1, stride):
                self._windows.append((int(seg_start + start), int(seg_start + start + win_size)))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self._windows[idx]
        x = torch.from_numpy(self._features[start:end])  # [W, F]
        center = start + (end - start) // 2
        label = torch.tensor(self._labels[center], dtype=torch.long)
        return x, label
