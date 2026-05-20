"""
GVSAO v2 — Generalised Good-point-set Vectorised Snow Ablation Optimiser.

Extends the original GVSAO (gvsao.py) from a fixed 2-parameter (lr, batch)
to an N-parameter search with support for:
  - Continuous log-scale parameters
  - Continuous linear-scale parameters
  - Discrete candidate-list parameters

All parameters are internally normalised to [0, 1] and decoded at evaluation
time. The underlying SAO algorithm (good-point-set init, dual-population,
periodic oscillation mutation) is unchanged.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from loguru import logger


@dataclass
class ParamDef:
    """Definition of one search parameter."""

    name: str
    kind: str  # "continuous_log" | "continuous_linear" | "discrete"
    candidates: list[Any] = field(default_factory=list)
    bounds: tuple[float, float] | None = None


@dataclass
class GVSaoV2Config:
    param_defs: list[ParamDef] = field(default_factory=list)
    population_size: int = 20
    max_generations: int = 10
    mutation_rate: float = 0.15
    oscillation_period: int = 5
    oscillation_amplitude: float = 0.3
    perturb_strength: float = 0.5
    seed: int = 42

    @property
    def n_dims(self) -> int:
        return len(self.param_defs)


@dataclass
class GVSaoV2Result:
    best_params: dict[str, Any] = field(default_factory=dict)
    best_fitness: float = float("inf")
    history: list[dict] = field(default_factory=list)
    n_evals: int = 0
    elapsed_seconds: float = 0.0


# ── Encoding / Decoding ──────────────────────────────────────────────────────


def _decode_param(value_01: float, pd_: ParamDef) -> Any:
    """Map [0, 1] to the parameter's real value."""
    if pd_.kind == "continuous_log":
        lo, hi = pd_.bounds
        return float(10.0 ** (np.log10(lo) + value_01 * (np.log10(hi) - np.log10(lo))))
    elif pd_.kind == "continuous_linear":
        lo, hi = pd_.bounds
        return float(lo + value_01 * (hi - lo))
    elif pd_.kind == "discrete":
        idx = int(round(value_01 * (len(pd_.candidates) - 1)))
        idx = max(0, min(idx, len(pd_.candidates) - 1))
        return pd_.candidates[idx]
    raise ValueError(f"Unknown parameter kind: {pd_.kind}")


def _encode_param(value: Any, pd_: ParamDef) -> float:
    """Map a real parameter value back to [0, 1]."""
    if pd_.kind == "continuous_log":
        lo, hi = pd_.bounds
        return float((np.log10(max(value, 1e-15)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)))
    elif pd_.kind == "continuous_linear":
        lo, hi = pd_.bounds
        return float((value - lo) / (hi - lo))
    elif pd_.kind == "discrete":
        try:
            idx = pd_.candidates.index(value)
        except ValueError:
            idx = 0
        if len(pd_.candidates) == 1:
            return 0.0
        return float(idx / (len(pd_.candidates) - 1))
    raise ValueError(f"Unknown parameter kind: {pd_.kind}")


def decode_individual(individual: np.ndarray, param_defs: list[ParamDef]) -> dict[str, Any]:
    """Map [0,1]^D vector to a dict of real hyperparameter values."""
    result: dict[str, Any] = {}
    for i, pd_ in enumerate(param_defs):
        result[pd_.name] = _decode_param(float(individual[i]), pd_)
    return result


def encode_individual(params: dict[str, Any], param_defs: list[ParamDef]) -> np.ndarray:
    """Map a dict of real hyperparameter values back to [0,1]^D."""
    return np.array([_encode_param(params[pd_.name], pd_) for pd_ in param_defs])


# ── Good Point Set Initialisation ────────────────────────────────────────────


def _good_point_set(
    n: int,
    dim: int,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    perturb_strength: float = 0.5,
) -> np.ndarray:
    """Generate initial population via good point set + small perturbation."""
    population = np.zeros((n, dim))
    for d in range(dim):
        lb, ub = lower[d], upper[d]
        uniform = lb + (ub - lb) * np.arange(n) / max(n - 1, 1)
        eps = perturb_strength * rng.uniform(0, 1, n) * (ub - lb) / 10.0
        population[:, d] = uniform + eps
    return np.clip(population, lower, upper)


# ── Melting Factor ───────────────────────────────────────────────────────────


def _melting_factor(fes: int, fes_max: int, t: float, t1: float = 0.0) -> float:
    t_ratio = fes / max(fes_max, 1)
    ddf = 0.35 + 0.25 * (math.exp(t_ratio) - 1.0) / (math.e - 1.0)
    return ddf * (t - t1)


# ── Main Optimiser ───────────────────────────────────────────────────────────


def run_gvsao_v2(
    fitness_fn: Callable[[dict[str, Any]], float],
    config: GVSaoV2Config,
    verbose: bool = True,
) -> GVSaoV2Result:
    """
    Run generalised GVSAO to find optimal parameters.

    fitness_fn : callable
        Takes a dict of decoded parameters → fitness value (lower is better).
    config : GVSaoV2Config
    verbose : bool
    """
    rng = np.random.default_rng(config.seed)
    N = config.population_size
    G = config.max_generations
    D = config.n_dims
    lb = np.zeros(D)
    ub = np.ones(D)

    population = _good_point_set(N, D, lb, ub, rng, config.perturb_strength)
    fitness = np.full(N, np.inf)
    best_global_fitness = np.inf
    best_global = population[0].copy()
    history: list[dict] = []
    n_evals = 0

    t_start = time.perf_counter()

    for gen in range(G):
        exploration_size = max(1, int(N * (0.8 - 0.3 * gen / max(G - 1, 1))))
        elite_size = max(1, int(N * 0.15))

        # Evaluate any unevaluated individuals
        for i in range(N):
            if np.isinf(fitness[i]):
                params = decode_individual(population[i], config.param_defs)
                fitness[i] = fitness_fn(params)
                n_evals += 1
                if verbose:
                    f1 = -fitness[i] if fitness[i] <= 0 else 0.0
                    logger.info(f"    eval {n_evals:3d}: F1={f1:.4f} | " +
                                "  ".join(f"{k}={v}" for k, v in params.items()))

        sorted_idx = np.argsort(fitness)
        elite_indices = sorted_idx[:elite_size].tolist()

        best_pop_fitness = fitness[sorted_idx[0]]
        if best_pop_fitness < best_global_fitness:
            best_global_fitness = best_pop_fitness
            best_global = population[sorted_idx[0]].copy()

        z_star = population[sorted_idx[0]].copy()
        z_bar = np.mean(population, axis=0)

        b_t = rng.normal(0, 1, (N, D))
        theta1 = rng.uniform(0, 1, N)
        ddf_dynamic = 0.35 + 0.25 * (math.exp(gen / max(G - 1, 1)) - 1.0) / (math.e - 1.0)
        m_t = ddf_dynamic * 2.5

        new_population = population.copy()

        for i in range(N):
            if i in elite_indices:
                continue

            if i < exploration_size:
                direction = (
                    theta1[i] * (z_star - population[i])
                    + (1.0 - theta1[i]) * (z_bar - population[i])
                )
                new_population[i] = population[i] + b_t[i] * direction
            else:
                theta2 = rng.uniform(0, 1)
                direction = (z_star - population[i]) + theta2 * (z_bar - population[i])
                new_population[i] = population[i] + m_t * direction

        # Periodic oscillation mutation
        for i in range(N):
            if i in elite_indices:
                continue
            if rng.random() < config.mutation_rate:
                k = gen
                T_period = config.oscillation_period
                W = config.oscillation_amplitude * math.sin(2 * math.pi * k / max(T_period, 1))
                new_population[i] = new_population[i] + W * (z_star - new_population[i])

        # Gaussian perturbation
        for i in range(N):
            if i in elite_indices:
                continue
            if rng.random() < 0.1:
                new_population[i] += rng.normal(0, 0.02, D)

        population = np.clip(new_population, lb, ub)
        fitness = np.full(N, np.inf)

        if verbose:
            best_p = decode_individual(best_global, config.param_defs)
            gen_fitness = fitness[sorted_idx]
            best_f1 = -gen_fitness[0] if gen_fitness[0] <= 0 else 0.0
            mean_f1 = -float(np.mean(gen_fitness)) if np.mean(gen_fitness) <= 0 else 0.0
            logger.info(
                f"  GVSAO gen {gen+1}/{G} | "
                f"best_F1={best_f1:.4f} (mean={mean_f1:.4f}) | "
                + "  ".join(f"{k}={v}" for k, v in best_p.items())
            )

        history.append({
            "generation": gen + 1,
            "best_fitness": float(best_global_fitness),
            "best_params": decode_individual(best_global, config.param_defs),
        })

    elapsed = time.perf_counter() - t_start
    best_params = decode_individual(best_global, config.param_defs)

    return GVSaoV2Result(
        best_params=best_params,
        best_fitness=float(best_global_fitness),
        history=history,
        n_evals=n_evals,
        elapsed_seconds=elapsed,
    )
