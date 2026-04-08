from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CpuResources:
    logical_cores: int
    physical_cores: int | None


def detect_cpu_resources() -> CpuResources:
    logical = os.cpu_count() or 1
    physical = None

    try:
        import psutil  # type: ignore

        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None

    return CpuResources(logical_cores=logical, physical_cores=physical)


def compute_thread_budget(
    cpu: CpuResources,
    *,
    prefer_physical_cores: bool = True,
    reserve_cores: int = 1,
    max_threads: int | None = None,
) -> int:
    base = cpu.physical_cores if prefer_physical_cores and cpu.physical_cores else cpu.logical_cores
    threads = max(1, int(base) - int(reserve_cores))

    if max_threads is not None:
        threads = min(threads, max(1, int(max_threads)))

    return max(1, threads)
