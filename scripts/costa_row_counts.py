#!/usr/bin/env python3
"""Read raw Costa .mat files and print row counts (total and per class).

Usage:
    uv run python scripts/costa_row_counts.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COSTA_DIR = PROJECT_ROOT / "data" / "raw" / "Costa PV Fault Dataset"

CLASS_NAMES: dict[int, str] = {
    0: "Normal",
    1: "ShortCircuit",
    2: "Degradation",
    3: "OpenCircuit",
    4: "Shadowing",
}


def main() -> None:
    elec_path = COSTA_DIR / "dataset_elec.mat"
    amb_path = COSTA_DIR / "dataset_amb.mat"

    for p in (elec_path, amb_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    elec = scipy.io.loadmat(str(elec_path))
    amb = scipy.io.loadmat(str(amb_path))

    f_nv = amb["f_nv"].flatten().astype(np.int32)
    irr = amb["irr"].flatten().astype(np.float64)
    n_total = len(f_nv)

    print(f"Total rows (raw): {n_total:,}")
    print(f"≈ {n_total / 86400:.2f} days at 1 Hz")
    print()

    print(f"{'Class':<16} {'Label':>5} {'Rows (raw)':>12} {'Rows (irr>=100)':>16}")
    print("-" * 53)

    mask_day = irr >= 100.0
    for label, name in sorted(CLASS_NAMES.items()):
        raw_cnt = int((f_nv == label).sum())
        day_cnt = int((f_nv[mask_day] == label).sum()) if label == 0 else int((f_nv == label).sum())
        print(f"{name:<16} {label:>5} {raw_cnt:>12,} {day_cnt:>16,}")

    print("-" * 53)
    print(f"{'TOTAL':<16} {'':>5} {n_total:>12,} {mask_day.sum():>16,}")
    print()
    print("(irr>=100 filter matches ingestion default)")


if __name__ == "__main__":
    main()
