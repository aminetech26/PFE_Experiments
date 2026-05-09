"""
GVSAO — Good-point-set Vectorised Snow Ablation Optimiser.

Implements the improved Snow Ablation Optimisation algorithm from:
  Wu et al. (2024) "Optimization of Bi-LSTM photovoltaic power
  prediction based on improved snow ablation optimization algorithm"
  Energies 17(17):4434.

Used for offline hyperparameter tuning of the GTBAD model:
  - Good point set initialisation (uniform coverage of search space)
  - Dual-population mechanism (exploration / exploitation)
  - Melting / sublimation / evaporation phases
  - Periodic oscillation mutation (escape local optima)

Target: jointly optimise learning rate (α) and batch size (B).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GVSaoConfig:
    population_size: int = 20
    max_generations: int = 10
    lr_bounds: tuple[float, float] = (1e-5, 1e-1)
    batch_bounds: tuple[float, float] = (16, 128)
    mutation_rate: float = 0.15
    oscillation_period: int = 5
    oscillation_amplitude: float = 0.3
    perturb_strength: float = 0.5
    seed: int = 42


@dataclass
class GVSaoResult:
    best_params: dict[str, float]
    best_fitness: float
    history: list[dict] = field(default_factory=list)
    n_evals: int = 0
    elapsed_seconds: float = 0.0


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


def _decode_individual(
    individual: np.ndarray,
    lr_bounds: tuple[float, float],
    batch_bounds: tuple[float, float],
) -> tuple[float, int]:
    """Map [0,1]^2 parameters to real hyperparameter values."""
    lr_log = individual[0]
    lr = 10.0 ** (np.log10(lr_bounds[0]) + lr_log * (np.log10(lr_bounds[1]) - np.log10(lr_bounds[0])))
    batch = int(round(batch_bounds[0] + individual[1] * (batch_bounds[1] - batch_bounds[0])))
    batch = max(4, batch)
    return lr, batch


def _encode_individual(
    lr: float,
    batch: int,
    lr_bounds: tuple[float, float],
    batch_bounds: tuple[float, float],
) -> np.ndarray:
    """Map real hyperparameters back to [0,1]^2."""
    lr_log = (np.log10(max(lr, 1e-15)) - np.log10(lr_bounds[0])) / (
        np.log10(lr_bounds[1]) - np.log10(lr_bounds[0])
    )
    batch_norm = (batch - batch_bounds[0]) / (batch_bounds[1] - batch_bounds[0])
    return np.array([lr_log, batch_norm])


def _melting_factor(
    fes: int,
    fes_max: int,
    t: float,
    t1: float = 0.0,
) -> float:
    """Compute melting rate M(t) = DDF(t) * (T - T1)."""
    t_ratio = fes / max(fes_max, 1)
    ddf = 0.35 + 0.25 * (math.exp(t_ratio) - 1.0) / (math.e - 1.0)
    return ddf * (t - t1)


def run_gvsao(
    fitness_fn: Callable[[float, int], float],
    config: GVSaoConfig | None = None,
    verbose: bool = True,
) -> GVSaoResult:
    """
    Run GVSAO to find optimal learning rate and batch size.

    Parameters
    ----------
    fitness_fn : callable
        Takes (learning_rate, batch_size) → validation_loss (lower is better).
    config : GVSaoConfig, optional
    verbose : bool

    Returns
    -------
    GVSaoResult with best_params, best_fitness, history.
    """
    cfg = config or GVSaoConfig()
    rng = np.random.default_rng(cfg.seed)

    N = cfg.population_size
    G = cfg.max_generations
    D = 2
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    population = _good_point_set(N, D, lb, ub, rng, cfg.perturb_strength)
    fitness = np.full(N, np.inf)
    best_global_fitness = np.inf
    best_global = population[0].copy()
    history: list[dict] = []
    n_evals = 0

    t_start = time.perf_counter()

    for gen in range(G):
        exploration_size = max(1, int(N * (0.8 - 0.3 * gen / max(G - 1, 1))))
        exploitation_size = N - exploration_size

        elite_size = max(1, int(N * 0.15))
        elite_indices: list[int] = []
        fitness_evaluated = np.full(N, False)

        for i in range(N):
            if np.isinf(fitness[i]):
                lr, batch = _decode_individual(population[i], cfg.lr_bounds, cfg.batch_bounds)
                fitness[i] = fitness_fn(lr, batch)
                n_evals += 1
                fitness_evaluated[i] = True

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

        for i in range(N):
            if i in elite_indices:
                continue
            if rng.random() < cfg.mutation_rate:
                k = gen
                T_period = cfg.oscillation_period
                W = cfg.oscillation_amplitude * math.sin(2 * math.pi * k / max(T_period, 1))
                new_population[i] = new_population[i] + W * (z_star - new_population[i])

        for i in range(N):
            if i in elite_indices:
                continue
            if rng.random() < 0.1:
                new_population[i] += rng.normal(0, 0.02, D)

        population = np.clip(new_population, lb, ub)
        fitness = np.full(N, np.inf)

        if verbose:
            lr_best, batch_best = _decode_individual(best_global, cfg.lr_bounds, cfg.batch_bounds)
            print(
                f"  GVSAO gen {gen+1}/{G} | "
                f"best_loss={best_global_fitness:.6f} | "
                f"lr={lr_best:.6f} | "
                f"batch={batch_best}"
            )

        history.append({
            "generation": gen + 1,
            "best_fitness": float(best_global_fitness),
            "best_lr": float(_decode_individual(best_global, cfg.lr_bounds, cfg.batch_bounds)[0]),
            "best_batch": int(_decode_individual(best_global, cfg.lr_bounds, cfg.batch_bounds)[1]),
        })

    elapsed = time.perf_counter() - t_start
    best_lr, best_batch = _decode_individual(best_global, cfg.lr_bounds, cfg.batch_bounds)

    return GVSaoResult(
        best_params={"learning_rate": best_lr, "batch_size": int(best_batch)},
        best_fitness=float(best_global_fitness),
        history=history,
        n_evals=n_evals,
        elapsed_seconds=elapsed,
    )
