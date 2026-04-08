from __future__ import annotations

from collections.abc import Callable
from numbers import Number

import optuna


def _is_int_range(value: list) -> bool:
    return len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value)


def suggest_params_from_space(trial: optuna.Trial, search_space: dict) -> dict:
    params: dict = {}
    for name, spec in search_space.items():
        if isinstance(spec, dict):
            spec_type = spec.get("type", "categorical")
            if spec_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    int(spec["low"]),
                    int(spec["high"]),
                    step=int(spec.get("step", 1)),
                    log=bool(spec.get("log", False)),
                )
            elif spec_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    step=float(spec["step"]) if "step" in spec else None,
                    log=bool(spec.get("log", False)),
                )
            elif spec_type == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unsupported HPO spec type for '{name}': {spec_type}")
            continue

        if isinstance(spec, (list, tuple)):
            if _is_int_range(list(spec)):
                params[name] = trial.suggest_int(name, int(spec[0]), int(spec[1]))
            elif len(spec) == 2 and all(isinstance(v, Number) for v in spec):
                params[name] = trial.suggest_float(name, float(spec[0]), float(spec[1]))
            else:
                params[name] = trial.suggest_categorical(name, list(spec))
            continue

        params[name] = spec

    return params


def midpoint_params_from_space(search_space: dict) -> dict:
    params: dict = {}
    for name, spec in search_space.items():
        if isinstance(spec, dict):
            spec_type = spec.get("type", "categorical")
            if spec_type == "int":
                params[name] = int((int(spec["low"]) + int(spec["high"])) / 2)
            elif spec_type == "float":
                params[name] = float((float(spec["low"]) + float(spec["high"])) / 2)
            elif spec_type == "categorical":
                choices = spec["choices"]
                params[name] = choices[0]
            else:
                raise ValueError(f"Unsupported HPO spec type for '{name}': {spec_type}")
            continue

        if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(v, Number) for v in spec):
            params[name] = int((spec[0] + spec[1]) / 2) if _is_int_range(list(spec)) else float((spec[0] + spec[1]) / 2)
            continue

        if isinstance(spec, (list, tuple)) and len(spec) > 0:
            params[name] = spec[0]
            continue

        params[name] = spec

    return params


def run_optuna(
    objective: Callable[[optuna.Trial], float],
    *,
    search_space: dict,
    n_trials: int,
    direction: str = "maximize",
    seed: int = 42,
    n_jobs: int = 1,
    timeout_seconds: int | None = None,
    on_trial_complete: Callable[[optuna.Study, optuna.FrozenTrial], None] | None = None,
) -> tuple[dict, optuna.Study]:
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=max(1, int(n_jobs)),
        timeout=timeout_seconds,
        callbacks=[on_trial_complete] if on_trial_complete else None,
    )

    best_params = suggest_params_from_space(optuna.trial.FixedTrial(study.best_params), search_space)
    return best_params, study
