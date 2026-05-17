from __future__ import annotations

import math
import os

import torch


def _resolve_loader_runtime(threading_cfg: dict, *, hpo_mode: bool) -> dict:
    cpu_count = max(1, int(os.cpu_count() or 1))
    configured_workers = threading_cfg.get("dl_num_workers", "auto")
    if isinstance(configured_workers, str) and configured_workers.lower() == "auto":
        cap = 2 if hpo_mode else 8
        num_workers = max(0, min(cap, cpu_count - 1))
    else:
        num_workers = max(0, int(configured_workers))

    pin_memory_cfg = threading_cfg.get("dl_pin_memory", "auto")
    if isinstance(pin_memory_cfg, str) and pin_memory_cfg.lower() == "auto":
        pin_memory = bool(torch.cuda.is_available())
    else:
        pin_memory = bool(pin_memory_cfg)

    persistent_default = bool(threading_cfg.get("dl_persistent_workers", True))
    # In HPO each trial rebuilds dataloaders; persistent workers leak across trials → off.
    persistent_workers = bool(num_workers > 0 and persistent_default and not hpo_mode)
    prefetch_factor = int(threading_cfg.get("dl_prefetch_factor", 2))
    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_lr_ratio, float(step + 1) / float(warmup_steps))
        progress_den = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(progress_den)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _trainer_runtime_kwargs(training_cfg: dict, model_cfg: dict | None = None) -> dict:
    """Build pl.Trainer kwargs for accelerator/devices/precision.

    model_cfg overrides training_cfg when provided (DLSSM pattern).
    When model_cfg is None only training_cfg is used (MAAT pattern).
    """
    cfg = model_cfg or {}
    kwargs = {
        "accelerator": str(cfg.get("accelerator", training_cfg.get("accelerator", "auto"))),
    }
    devices = cfg.get("devices", training_cfg.get("devices", "auto"))
    if devices != "auto":
        kwargs["devices"] = devices
    precision_cfg = cfg.get("precision", training_cfg.get("precision"))
    if precision_cfg is not None:
        kwargs["precision"] = precision_cfg
    return kwargs
