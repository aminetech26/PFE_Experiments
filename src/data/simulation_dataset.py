#!/usr/bin/env python3
"""
Build a simulation dataset for production-style inference testing.

Loads the ingested Costa parquet and produces a synthetic stream of:

  normal → fault episode → normal → fault episode → normal → ...

Original chronological order is **not** preserved. Normal rows are split
into chunks that are interleaved with fault episodes to mimic a realistic
inference stream: long stretches of healthy behaviour punctuated by short
fault episodes.

Each fault row receives an ``episode_id`` (integer ≥ 1). Normal rows get 0.
Synthetic 1 Hz timestamps are generated starting from 2024-01-01T00:00:00Z.

Usage:
    uv run python -m src.data.simulation_dataset
    uv run python -m src.data.simulation_dataset --start-row 50000 --episode-length 20
    uv run python -m src.data.simulation_dataset --episode-length 20 --episodes-per-class 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "ingestion" / "costa" / "costa_merged.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "simulation" / "costa_simulation.parquet"

EVALUABLE_CLASSES = [1, 2, 3, 4]
CLASS_NAMES = {0: "Normal", 1: "ShortCircuit", 2: "Degradation", 3: "OpenCircuit", 4: "Shadowing"}


def _find_segments(series: pd.Series, label: int) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) inclusive for every contiguous block of *label*."""
    mask = (series.values == label)
    segments: list[tuple[int, int]] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _extract_episodes(
    df: pd.DataFrame,
    segments: list[tuple[int, int]],
    episode_length: int,
    max_episodes: int,
    seed: int,
) -> list[pd.DataFrame]:
    """Extract non-overlapping *episode_length*-row windows from segments.

    Windows that don't fit (segment < episode_length) are discarded.
    If more than *max_episodes* are available we shuffle and pick *max_episodes*.
    """
    rng = np.random.default_rng(seed)
    candidates: list[pd.DataFrame] = []

    for start, end in segments:
        seg_len = end - start + 1
        n_full = seg_len // episode_length
        for e in range(n_full):
            win_start = start + e * episode_length
            win_end = win_start + episode_length
            candidates.append(df.iloc[win_start:win_end].copy())

    if len(candidates) == 0:
        logger.warning(f"    No episode of length {episode_length} found in {len(segments)} segments")
        return []

    if max_episodes > 0 and len(candidates) > max_episodes:
        indices = rng.choice(len(candidates), size=max_episodes, replace=False)
        candidates = [candidates[i] for i in sorted(indices)]

    return candidates


def build_simulation(
    input_path: Path,
    episode_length: int = 10,
    episodes_per_class: int = 10,
    normal_rows: int = 0,
    normal_chunk_size: int = 0,
    start_row: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the simulation dataset.

    Args:
        input_path: Path to ``costa_merged.parquet``.
        episode_length: Rows per fault episode.
        episodes_per_class: Max episodes per fault class (0 = all available).
        normal_rows: Number of normal rows in the pool, taken from the **latest**
            original timestamps (0 = all normal rows).
        normal_chunk_size: Normal rows between each episode (0 = auto: evenly
            distributed across all inter-episode gaps).
        start_row: Row index to start reading from the source parquet (0-based).
        seed: Random seed for episode selection, shuffling, and chunk sampling.

    Returns:
        DataFrame with synthetic 1 Hz timestamps and ``episode_id`` column.
    """
    rng = np.random.default_rng(seed)

    logger.info(f"Loading: {input_path}")
    df = pd.read_parquet(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    if start_row > 0:
        df = df.iloc[start_row:]
        logger.info(f"  Starting from row index {start_row}")
    logger.info(f"  {len(df):,} rows  |  labels: {sorted(df['label'].unique())}")

    # ── Normal pool ───────────────────────────────────────────────────────
    normal_pool = df[df["label"] == 0].copy()
    normal_pool = normal_pool[(normal_pool["irr"] > 200) & (normal_pool["pdc"] > 100)]
    if normal_rows > 0 and normal_rows < len(normal_pool):
        normal_pool = normal_pool.iloc[-normal_rows:]
    logger.info(f"  Normal pool: {len(normal_pool):,} rows  (daytime filter: irr>200 & pdc>100)")

    # ── Fault episodes ─────────────────────────────────────────────────────
    fault_episodes: list[pd.DataFrame] = []
    for cls in EVALUABLE_CLASSES:
        segments = _find_segments(df["label"], cls)
        episodes = _extract_episodes(df, segments, episode_length, episodes_per_class, seed + cls)
        fault_episodes.extend(episodes)
        logger.info(
            f"  {CLASS_NAMES[cls]:<14}: {len(segments)} segments → "
            f"{len(episodes)} episodes x {episode_length} rows"
        )

    # Shuffle episodes so fault types are mixed in the stream
    rng.shuffle(fault_episodes)
    n_episodes = len(fault_episodes)
    logger.info(f"  Total fault episodes: {n_episodes}")

    if n_episodes == 0:
        logger.warning("No fault episodes extracted — output is normal-only")
        normal_pool["episode_id"] = 0
        return normal_pool

    # ── Split normal pool into chunks ──────────────────────────────────────
    if normal_chunk_size > 0:
        chunk_size = normal_chunk_size
    else:
        chunk_size = max(1, len(normal_pool) // (n_episodes + 1))

    normal_indices = rng.permutation(len(normal_pool))
    normal_chunks: list[pd.DataFrame] = []
    pos = 0
    for _ in range(n_episodes + 1):
        end = min(pos + chunk_size, len(normal_pool))
        chunk_idx = normal_indices[pos:end]
        normal_chunks.append(
            normal_pool.iloc[sorted(chunk_idx)].copy() if len(chunk_idx) > 0
            else pd.DataFrame(columns=normal_pool.columns)
        )
        pos = end
    logger.info(f"  Normal chunk size: {chunk_size}  (→ {n_episodes + 1} chunks)")

    # ── Interleave ─────────────────────────────────────────────────────────
    interleaved: list[pd.DataFrame] = []
    ep_id = 0

    for i in range(n_episodes):
        nc = normal_chunks[i]
        if len(nc) > 0:
            nc["episode_id"] = 0
            interleaved.append(nc)

        ep = fault_episodes[i].copy()
        ep_id += 1
        ep["episode_id"] = ep_id
        interleaved.append(ep)

    nc = normal_chunks[-1]
    if len(nc) > 0:
        nc["episode_id"] = 0
        interleaved.append(nc)

    combined = pd.concat(interleaved, axis=0)

    # ── Synthetic timestamps (1 Hz) ────────────────────────────────────────
    start = pd.Timestamp("2024-01-01T00:00:00", tz="UTC")
    combined.index = pd.date_range(start=start, periods=len(combined), freq="s", tz="UTC")
    combined["episode_id"] = combined["episode_id"].astype(int)

    logger.info(f"  Output: {len(combined):,} rows  |  {ep_id} fault episodes")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Build Costa simulation dataset for inference")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--episode-length", type=int, default=10,
                        help="Consecutive rows per fault episode (default: 10)")
    parser.add_argument("--episodes-per-class", type=int, default=10,
                        help="Max episodes per fault class (0 = all available)")
    parser.add_argument("--normal-rows", type=int, default=0,
                        help="Normal rows in the pool (0 = all, latest timestamps)")
    parser.add_argument("--normal-chunk-size", type=int, default=0,
                        help="Normal rows between episodes (0 = auto: evenly distributed)")
    parser.add_argument("--start-row", type=int, default=0,
                        help="Row index to start reading from the source parquet (0-based)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Building Costa simulation dataset (interleaved)")
    logger.info(f"  Episode length:      {args.episode_length} rows")
    logger.info(f"  Episodes per class:  {args.episodes_per_class} (0=all)")
    logger.info(f"  Normal rows pool:    {args.normal_rows} (0=all)")
    logger.info(f"  Normal chunk size:   {args.normal_chunk_size} (0=auto)")
    logger.info(f"  Start row:           {args.start_row} (0=beginning)")
    logger.info(f"  Seed:                {args.seed}")

    sim = build_simulation(
        input_path=Path(args.input),
        episode_length=args.episode_length,
        episodes_per_class=args.episodes_per_class,
        normal_rows=args.normal_rows,
        normal_chunk_size=args.normal_chunk_size,
        start_row=args.start_row,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sim.to_parquet(output_path)
    logger.success(f"Saved → {output_path}")

    # Per-episode summary
    logger.info("\nEpisode breakdown:")
    for eid in sorted(sim["episode_id"].unique()):
        sub = sim[sim["episode_id"] == eid]
        if eid == 0:
            logger.info(f"  episode 0 (Normal):   {len(sub):,} rows")
        else:
            lbl = int(sub["label"].iloc[0])
            t0 = sub.index[0]
            t1 = sub.index[-1]
            logger.info(f"  episode {eid:3d} ({CLASS_NAMES[lbl]:<14}): {len(sub)} rows  [{t0} → {t1}]")


if __name__ == "__main__":
    main()
