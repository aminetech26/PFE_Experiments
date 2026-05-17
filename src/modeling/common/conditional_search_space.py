"""Hierarchical / conditional Optuna parameter sampling for MAAT HPO.

Each trial samples architecture first, then training params conditioned on arch,
so validity constraints (d_model % n_heads == 0, win_size % block_size == 0) are
satisfied by construction — no TrialPruned budget waste.
"""
from __future__ import annotations

import re

import optuna


def normalize_conditional_params(params: dict) -> dict:
    """Map branch-specific Optuna param names back to canonical keys.

    Supports mixed-schema studies where older trials may already have
    canonical keys.
    """
    out = dict(params)
    _branch_to_canonical = [
        (re.compile(r"^n_heads_for_d_model_\d+$"), "n_heads"),
        (re.compile(r"^block_size_for_win_size_\d+$"), "block_size"),
        (re.compile(r"^batch_size_for_d_model_\d+$"), "batch_size"),
    ]
    for pattern, canonical_key in _branch_to_canonical:
        matches = [k for k in out if pattern.fullmatch(k)]
        if matches:
            out[canonical_key] = out.pop(matches[0])
    return out


# n_heads choices that are always valid divisors of d_model
_N_HEADS_FOR_D_MODEL: dict[int, list[int]] = {
    32:  [2, 4],
    64:  [2, 4, 8],
    128: [2, 4, 8],
}

# block_size choices that are always valid divisors of win_size (and ≤ 30)
_BLOCK_SIZE_FOR_WIN_SIZE: dict[int, list[int]] = {
    30:  [5, 10, 15, 30],
    60:  [5, 10, 15, 20, 30],
    90:  [5, 10, 15, 18, 30],
    120: [5, 10, 15, 20, 24, 30],
}

# dropout upper bound scales with depth (more layers → more regularisation needed)
_DROPOUT_RANGE_FOR_E_LAYERS: dict[int, tuple[float, float]] = {
    1: (0.0,  0.10),
    2: (0.0,  0.20),
    3: (0.05, 0.30),
    4: (0.10, 0.35),
}

# lr upper bound shrinks with d_model (larger model → smaller safe lr)
_LR_RANGE_FOR_D_MODEL: dict[int, tuple[float, float]] = {
    32:  (5e-5, 1e-3),
    64:  (5e-5, 8e-4),
    128: (3e-5, 5e-4),
}

# batch_size choices: large d_model needs smaller batch to fit memory
_BATCH_SIZE_FOR_D_MODEL: dict[int, list[int]] = {
    32:  [128, 256],
    64:  [128, 256],
    128: [64, 128],
}


def suggest_conditional_params(trial: optuna.Trial) -> dict:
    """Sample all MAAT HPO params for one trial with hard validity guarantees."""
    # ── Architecture ──────────────────────────────────────────────────────────
    e_layers  = trial.suggest_int("e_layers", 1, 4)
    d_model   = trial.suggest_categorical("d_model", [32, 64, 128])
    # NOTE:
    # Optuna does not allow changing a parameter's categorical choices across
    # trials in the same study. Using a single "n_heads" name with
    # d_model-dependent choices triggers:
    # "CategoricalDistribution does not support dynamic value space".
    #
    # We avoid this by using branch-specific parameter names with fixed choice
    # sets, then projecting to canonical returned keys.
    n_heads = trial.suggest_categorical(f"n_heads_for_d_model_{d_model}", _N_HEADS_FOR_D_MODEL[d_model])
    d_ff      = trial.suggest_categorical("d_ff", [128, 256, 512])
    win_size  = trial.suggest_categorical("win_size", [30, 60, 90, 120])
    block_size = trial.suggest_categorical(
        f"block_size_for_win_size_{win_size}",
        _BLOCK_SIZE_FOR_WIN_SIZE[win_size],
    )

    # ── Training — conditioned on arch ────────────────────────────────────────
    dr_lo, dr_hi = _DROPOUT_RANGE_FOR_E_LAYERS[e_layers]
    dropout = trial.suggest_float("dropout", dr_lo, dr_hi)

    lr_lo, lr_hi = _LR_RANGE_FOR_D_MODEL[d_model]
    learning_rate = trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True)

    weight_decay      = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical(
        f"batch_size_for_d_model_{d_model}",
        _BATCH_SIZE_FOR_D_MODEL[d_model],
    )
    gradient_clip_val = trial.suggest_categorical("gradient_clip_val", [0.5, 1.0, 2.0])
    k                 = trial.suggest_float("k", 2.0, 6.0)
    temperature       = trial.suggest_categorical("temperature", [10, 25, 50, 100])

    return {
        "e_layers": e_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "win_size": win_size,
        "block_size": block_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "gradient_clip_val": gradient_clip_val,
        "k": k,
        "temperature": temperature,
    }
