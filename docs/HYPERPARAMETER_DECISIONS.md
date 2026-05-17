# Hyperparameter Decisions and Rationale

## Purpose

This document explains the hyperparameter strategy for the PV fault detection project, with emphasis on:

- why each decision exists,
- what is currently implemented,
- what is planned but not yet active,
- how to reason about tradeoffs before launching expensive experiments.

This is a reasoning document, not just an execution checklist.

---

## Context and constraints

The hyperparameter strategy is shaped by the project realities documented in [SPLIT_DECISIONS.md](SPLIT_DECISIONS.md):

1. Fault classes are temporally concentrated and segment-imbalanced.
2. Leakage control is non-negotiable.
3. Compute budget is finite and must be shared with feature experiments.
4. Class imbalance means default metrics can be misleading.

Because of this, hyperparameter optimization (HPO) is treated as a controlled experimental process, not brute-force search.

---

## Current implementation scope

Current active HPO path is for Task B classification with LightGBM.

Files involved:

- runtime config: `PFE_Experiments/configs/model_config.yaml`
- training entrypoint: `PFE_Experiments/src/modeling/classification/ml/lightgbm_model.py`
- generic HPO helpers: `PFE_Experiments/src/modeling/common/hyperparameter_optimizer.py`
- CPU/thread planner: `PFE_Experiments/src/modeling/common/system_resources.py`

### What is implemented now

1. Optuna objective-driven search with configurable sampler (default: TPE).
2. Config-driven search space parsing.
3. Midpoint deterministic baseline mode (`--no-optuna`).
4. Trial count and timeout control.
5. Parallel Optuna jobs with CPU-aware thread budgeting.
6. Optional Optuna study persistence via storage URL (SQLite-ready for Colab resume).
6. Logging of best score, selected parameters, and trial dataframe artifact.

### What is not implemented yet

1. Stage-aware DL HPO orchestration (Stage 1 training recipe / Stage 2 architecture) is not yet active.
2. Config-driven pruner selection for all trainers beyond current LightGBM path.
3. Segment-aware temporal CV inside Optuna objective for all trainers.
4. Multi-model HPO execution from the same trainer (LightGBM is active model today).

---

## Primary HPO objective choice

### Decision

Use weighted F1 on validation as the current Optuna objective for Task B.

### Why

1. Task B is multi-class with significant class imbalance.
2. Accuracy can be high while minority classes fail.
3. Weighted F1 balances precision and recall per class while accounting for support.

### Caveat

Weighted F1 can still mask poor minority behavior if one class dominates strongly. For final reporting, pair it with:

- macro F1,
- per-class F1,
- confusion matrix,
- PR-AUC (if probability-quality interpretation is required).

---

## Search space design philosophy

### Decision pattern

Use bounded, physics-aware, model-stability-oriented ranges rather than very wide unconstrained ranges.

Current LightGBM search space includes:

- `n_estimators`,
- `learning_rate`,
- `num_leaves`,
- `subsample`,
- `colsample_bytree`,
- `reg_alpha`,
- `reg_lambda`,
- `min_child_samples`.

### Why this shape

1. The space is broad enough to capture bias-variance tradeoff.
2. It avoids obviously unstable regions for this dataset size and class structure.
3. It keeps search efficient under moderate trial budgets.

### Midpoint mode rationale

The midpoint baseline is intentionally deterministic and fast. It is useful for:

- pipeline preflight,
- regression checks after refactors,
- quick sanity runs before expensive HPO.

---

## Holdout vs CV for HPO

### Current behavior

Optuna evaluates each trial on a fixed train/val holdout by default (`validation_mode: holdout`).

### Why this is currently used

1. Faster iteration and lower compute.
2. Easier debugging and reproducibility during active pipeline refactoring.
3. Compatible with immediate experimental throughput needs.

### Strategic limitation

This is weaker than segment-aware temporal CV for robust model selection under temporal structure.

### Alignment with split rationale

Split decisions explicitly call for segment-ordered temporal CV for tuning rigor. Therefore:

- current implementation is acceptable for iterative development,
- but publication-grade claims should clearly state holdout-based tuning unless CV mode is implemented and used.

---

## Sampler decision

### Current runtime decision

TPE sampler is the default; sampler and pruner are now configurable from runtime config.

### Why TPE is a sensible default here

1. Better sample efficiency than random search under moderate budgets.
2. Works well on mixed integer/float/categorical spaces.
3. Strong practical default for tabular model tuning.

### Remaining gap

Config currently includes a sampler key, but runtime does not yet switch sampler by that key. This should be implemented to avoid configuration ambiguity.

---

## Parallelism and resource policy

### Decision

Use explicit CPU-aware threading policy rather than relying on library defaults.

### Mechanism

1. Detect logical and physical cores.
2. Prefer physical cores by default.
3. Reserve a safety margin (`reserve_cores`).
4. Allocate:
   - Optuna parallel trial workers,
   - per-trial model threads.

### Why

1. Avoid oversubscription and unstable runtime.
2. Keep workstation responsive during long searches.
3. Make experiments repeatable across machines.

### Tradeoff

Increasing parallel trials can reduce per-trial thread count, which may hurt each model fit. Optimal setting depends on objective complexity and hardware.

---

## Logging and observability decisions

### Decision

HPO must produce both local and remote evidence of what happened.

### Current outputs

1. MLflow params/metrics/tags.
2. Best-trial metric summary.
3. Trials dataframe artifact.
4. Feature manifest and threading plan artifacts.
5. Local metrics/model artifacts for offline reproducibility.
6. Leakage report artifact from integrated leakage suite.
7. Persistent comparison records JSONL for post-run statistical testing.

### Why

1. Post-hoc auditability is required for thesis-quality reporting.
2. Debugging failed or unstable experiments needs detailed traceability.
3. Reproducibility requires explicit linkage: feature run -> config -> best params -> metrics.

---

## Decision rules before launching real experiments

Use this flow:

1. Is the goal pipeline validation?
- Use `--no-optuna` midpoint baseline.

2. Is the goal quick ranking of feature profiles?
- Use low trial count holdout HPO and fixed run ids.

3. Is the goal final model claim or thesis table?
- Prefer temporal segment-aware CV HPO (once implemented),
- or clearly label results as holdout-tuned if CV mode is not yet active.

4. Is runtime unstable or too slow?
- reduce trial count,
- reduce parallel trials,
- cap max threads,
- keep deterministic seed and fixed run id.

---

## Recommended immediate improvements

Priority order:

1. Implement sampler selection from config (`tpe`, `random`, optional CMA-ES).
2. Implement optional pruner selection (`median`, `hyperband`).
3. Add objective mode switch (`holdout`, `segment_temporal_cv`).
4. Persist Optuna study in sqlite for resume/restart support.
5. Track per-trial summary JSONL in addition to MLflow table artifact.

### Two-stage DL orchestration policy (transparent runtime)

The DL two-stage policy is intended to be automatic from one run command, not manually triggered per stage.

- Stage 1 (training recipe): tune `learning_rate`, `batch_size`, `weight_decay` with fixed reference architecture.
- Stage 2 (architecture): tune depth/width/components while freezing Stage 1 recipe.
- Stage 3 (optional): small joint local search around Stage 1+2 winner.

Operational requirement:

- users choose one top-level HPO mode (`auto_two_stage`) and optional `run_stage3` flag,
- runtime orchestrator launches stages sequentially and carries best params forward.

### Completed wiring update

The following are now implemented:

- leakage suite execution in classification training,
- leakage report persistence per run,
- run-level comparison records persistence,
- dedicated comparison script for Wilcoxon + effect size:
   - `python -m src.evaluation.compare_classification_runs`

---

## Reporting guidance

When writing thesis or reports, include:

1. Objective metric used for optimization and why.
2. Search space bounds and rationale.
3. Trial budget and timeout.
4. Sampler and pruning policy.
5. Validation scheme used during HPO (holdout vs temporal CV).
6. Compute policy (threads, parallel trials, hardware context).
7. Reproducibility anchor (feature run id and config snapshot).

This prevents overstating rigor and makes comparisons scientifically defensible.

---

## Final position

Current HPO design is practical, reproducible, and good for fast iteration. It is not yet the strongest possible methodological setup for temporal generalization claims. The next step is not more trials; it is better validation structure (segment-aware temporal CV) combined with explicit sampler/pruner configurability.
