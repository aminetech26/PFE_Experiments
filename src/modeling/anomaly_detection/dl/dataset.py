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
    Returns (x: Tensor[W, F], label: scalar int) where label is taken from the
    configured target position (center or last step of the window).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
        win_size: int,
        stride: int = 1,
        normal_only: bool = False,
        return_original_label: bool = False,
        return_group_id: bool = False,
        target_position: str = "center",
        return_slow_context: bool = False,
        slow_context_samples: int = 3600,
        slow_stride: int = 60,
    ) -> None:
        super().__init__()
        self.win_size = win_size
        self.stride = stride
        self.return_original_label = return_original_label
        self.return_group_id = return_group_id
        if target_position not in {"center", "last"}:
            raise ValueError(f"Unsupported target_position: {target_position}")
        self.target_position = target_position
        self.return_slow_context = return_slow_context
        self.slow_context_samples = slow_context_samples
        self.slow_stride = max(1, slow_stride)

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
        self._group_col = group_col

        self._features: np.ndarray = df[feature_cols].to_numpy(dtype=np.float32)
        self._original_labels: np.ndarray = df[label_col].to_numpy(dtype=np.float64)
        self._labels: np.ndarray = (self._original_labels != 0).astype(np.int64)
        self._group_ids: np.ndarray | None = None
        if group_col is not None:
            self._group_ids = df[group_col].astype(str).to_numpy()

        # Build (start_idx, end_idx) window indices that don't cross group boundaries
        self._windows: list[tuple[int, int]] = []
        self._window_seg_starts: list[int] = []  # segment start for each window (slow context)
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
                self._window_seg_starts.append(int(seg_start))

    def __len__(self) -> int:
        return len(self._windows)

    def _get_slow_context(self, idx: int, start: int) -> torch.Tensor:
        """Extract preceding slow-rate context for the window starting at `start`.

        Uses rows strictly before `start` (i.e. ctx_start:start) so the slow context
        never includes any timestep in the fast window being scored — preserving causal
        prior/prediction semantics. Downsampled by `slow_stride`. When history is shorter
        than target_len (including the segment-start case where no history exists at all),
        pads the front with zeros in standardized space (zero == per-feature mean, neutral
        placeholder with no connection to the current observation).
        """
        seg_start = self._window_seg_starts[idx]
        ctx_start = max(seg_start, start - self.slow_context_samples)
        raw = self._features[ctx_start:start:self.slow_stride]  # [T_raw, F]
        target_len = max(1, self.slow_context_samples // self.slow_stride)
        n_features = self._features.shape[1]
        if len(raw) < target_len:
            # Zero-pad in standardized space (zero == per-feature mean, neutral placeholder).
            # Never use seg_start row: when start == seg_start that row is inside the fast
            # window being scored, which would leak the current observation into the prior.
            pad = np.zeros((target_len - len(raw), n_features), dtype=raw.dtype if len(raw) > 0 else self._features.dtype)
            raw = np.concatenate([pad, raw], axis=0) if len(raw) > 0 else pad
        return torch.from_numpy(raw[-target_len:])  # [target_len, F]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        start, end = self._windows[idx]
        x = torch.from_numpy(self._features[start:end])  # [W, F]
        target_idx = end - 1 if self.target_position == "last" else start + (end - start) // 2
        label = torch.tensor(self._labels[target_idx], dtype=torch.long)

        extras: list[torch.Tensor | str] = []
        if self.return_original_label:
            extras.append(torch.tensor(self._original_labels[target_idx], dtype=torch.float64))
        if self.return_group_id:
            extras.append(self._group_ids[target_idx] if self._group_ids is not None else "row")

        if self.return_slow_context:
            x_slow = self._get_slow_context(idx, start)
            return (x, x_slow, label, *extras)
        return (x, label, *extras)
