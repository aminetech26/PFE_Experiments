from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Number

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler


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

        if (
            isinstance(spec, (list, tuple))
            and len(spec) == 2
            and all(isinstance(v, Number) for v in spec)
        ):
            params[name] = (
                int((spec[0] + spec[1]) / 2)
                if _is_int_range(list(spec))
                else float((spec[0] + spec[1]) / 2)
            )
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
    sampler_name: str = "tpe",
    pruner_name: str | None = None,
    storage_url: str | None = None,
    study_name: str | None = None,
    load_if_exists: bool = True,
    on_trial_complete: Callable[[optuna.Study, optuna.FrozenTrial], None] | None = None,
) -> tuple[dict, optuna.Study]:
    sampler = _build_sampler(sampler_name=sampler_name, seed=seed)
    pruner = _build_pruner(pruner_name=pruner_name)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage_url,
        study_name=study_name,
        load_if_exists=load_if_exists,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=max(1, int(n_jobs)),
        timeout=timeout_seconds,
        callbacks=[on_trial_complete] if on_trial_complete else None,
    )

    best_params = suggest_params_from_space(
        optuna.trial.FixedTrial(study.best_params), search_space
    )
    return best_params, study


def _build_sampler(sampler_name: str, seed: int) -> BaseSampler:
    name = str(sampler_name).strip().lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    raise ValueError(f"Unsupported Optuna sampler: {sampler_name}")


def _build_pruner(pruner_name: str | None) -> BasePruner:
    if pruner_name is None:
        return optuna.pruners.NopPruner()

    name = str(pruner_name).strip().lower()
    if name in {"", "none", "off", "disabled"}:
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    if name in {"successive_halving", "sha", "asha"}:
        return optuna.pruners.SuccessiveHalvingPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    raise ValueError(f"Unsupported Optuna pruner: {pruner_name}")


@dataclass(slots=True)
class HPOStageConfig:
    name: str
    search_space: dict
    n_trials: int
    direction: str = "maximize"
    timeout_seconds: int | None = None
    sampler: str = "tpe"
    pruner: str | None = None


@dataclass(slots=True)
class HPOStageResult:
    name: str
    best_params: dict
    best_value: float
    study: optuna.Study


def run_staged_optuna(
    *,
    stages: list[HPOStageConfig],
    objective_builder: Callable[[HPOStageConfig, dict], Callable[[optuna.Trial], float]],
    seed: int = 42,
    n_jobs: int = 1,
    storage_url: str | None = None,
    study_name_prefix: str = "staged_hpo",
    load_if_exists: bool = True,
    on_trial_complete: Callable[[optuna.Study, optuna.FrozenTrial], None] | None = None,
) -> tuple[dict, list[HPOStageResult]]:
    """Run sequential HPO stages without manual stage triggering.

    Stage N receives merged best params from previous stages as frozen context.
    """
    merged_best: dict = {}
    stage_results: list[HPOStageResult] = []

    for stage in stages:
        if stage.n_trials <= 0:
            continue
        objective = objective_builder(stage, dict(merged_best))
        best_params, study = run_optuna(
            objective,
            search_space=stage.search_space,
            n_trials=stage.n_trials,
            direction=stage.direction,
            seed=seed,
            n_jobs=n_jobs,
            timeout_seconds=stage.timeout_seconds,
            sampler_name=stage.sampler,
            pruner_name=stage.pruner,
            storage_url=storage_url,
            study_name=f"{study_name_prefix}__{stage.name}",
            load_if_exists=load_if_exists,
            on_trial_complete=on_trial_complete,
        )
        merged_best.update(best_params)
        stage_results.append(
            HPOStageResult(
                name=stage.name,
                best_params=best_params,
                best_value=float(study.best_value),
                study=study,
            )
        )

    return merged_best, stage_results
