# Modeling Decisions

## Purpose of this document

This document explains the current Task B classification modeling stack in detail, including:

- why each design choice exists,
- how data and configuration flow through the system,
- what each module is responsible for,
- which files are produced,
- and the exact call chain from CLI entrypoint to artifacts and remote tracking.

The objective is to help you reason about the system, challenge assumptions, and safely evolve it, not just execute commands.

---

## Scope and current status

Current implementation scope:

- implemented and production-usable: Task B classification baseline with LightGBM + optional Optuna + MLflow to DagsHub,
- task-aware feature run selection (profile/latest, run id, or explicit path),
- CPU threading plan and Optuna parallelism controls,
- local artifact writing + remote experiment tracking.

Not yet implemented in this trainer:

- non-LightGBM model execution (even if listed in config),
- CV-based robust model selection (current flow uses train/val tuning and final train+val fit),
- anomaly and additional classification models under their dispatchers are still being expanded.

Task A anomaly-ML note (current phase):

- the actively used baseline is One-Class SVM under `src/modeling/anomaly_detection/ml/one_class_svm_model.py`,
- intended primary split/task pairing is `anomaly_semisup` (normal-only train),
- and current Costa-first profile recommendation is `plus_physics` as the compact baseline before richer temporal/spectral ablations.

---

## Costa dataset nature — important for interpreting baseline results

Costa is a **real PV installation** with **artificially induced faults**, not a simulated dataset. Mendeley/GPVS-Faults is the simulated dataset (PSIM/Simulink circuit model output).

This distinction matters when interpreting near-ceiling classification results on Costa:

- Artificially induced faults produce cleaner, more consistent signatures than naturally occurring faults because the fault is physically applied at a known type, severity, and timing.
- The four fault classes are physically distinct electrical failure modes with clearly separable voltage/current fingerprints.
- At 1 Hz under stable induced conditions, consecutive fault samples are near-identical — the classifier recognizes a consistent electrical state.

**Implication for thesis reporting:** near-ceiling f1_weighted on Costa is an expected result given the controlled induction protocol, not evidence of leakage or overfit. Always note in any results table that Costa uses artificially induced faults and that La Réunion (naturally occurring faults in the field) is the harder generalization target.

---

## Task A (OCSVM) decisions, challenges, and resolutions

### 1) Semisupervised split compatibility

- Challenge: OCSVM requires normal-only training, but accidental runs on non-semisup artifacts remain possible.
- Decision: treat `anomaly_semisup` as the canonical OCSVM route for Costa baselines; non-normal train content is a warning condition that must be inspected before accepting a run.
- Resolution outcome: experiment protocol now anchors OCSVM on `anomaly_semisup` and interprets runs through novelty-detection semantics.

### 2) Train-cap practicality vs representativeness

- Challenge: OCSVM becomes expensive on full normal train sets; naive row-random subsampling is fast but can under-cover operating regimes.
- Decision: keep capped training (`max_train_samples`) but upgrade sampling policy from uniform row-random to group-aware quota sampling with within-group temporal spacing.
- Resolution outcome: capped runs remain computationally feasible while improving coverage across Costa group structure and reducing dense autocorrelation bias.

### 3) Sampling observability

- Challenge: prior runs logged sample size only, not sample-selection structure.
- Decision: log sampling strategy metadata (`strategy`, `group column`, `number of groups`) alongside run metrics/params.
- Resolution outcome: OCSVM runs are now easier to audit and compare when seed/cap are varied.

### 4) Kernel handling policy

- Challenge: kernel choice is conceptually a hyperparameter, but implementation uses kernel-specific search spaces.
- Decision: treat `kernel` as a top-level experiment branch (`rbf`/`poly`) and run separate studies per kernel; Optuna tunes parameters within that branch.
- Resolution outcome: search spaces stay coherent and run comparisons remain interpretable.

### 5) Metric posture under class imbalance

- Challenge: Costa label skew (and natural shadowing prevalence) can make threshold metrics noisy.
- Decision: keep PR-AUC as the primary ranking metric; threshold calibration on validation remains operational but is interpreted as secondary to ranking quality.
- Resolution outcome: baseline reporting aligns with anomaly-detection best practice under imbalance.

---

## Big picture architecture

### Upstream data lineage (for classification training)

1. `ingest` stage builds interim merged parquet files.
2. `split` stage creates task-specific splits.
3. `preprocess` stage applies cleaning/transforms post-split.
4. `featurize` stage builds task/profile-specific feature runs under `data/processed/features/<task>/runs/<run_id>` and updates `latest_runs.json` pointer(s).
5. `train_classification` stage reads a selected feature run and trains/evaluates/logs a model.

### Why train from feature runs, not from interim splits

Decision: training consumes frozen feature run artifacts, not raw/interim splits.

Reasoning:

- strict reproducibility: model run is tied to a concrete feature run id with a manifest,
- no hidden feature drift between tuning and final train,
- clear ownership boundary: featurization is a complete upstream step with explicit outputs and metadata,
- easier rollback/comparison across feature profiles.

---

## DVC orchestration contract

`dvc.yaml` for classification currently declares:

- command: `uv run python -m src.modeling.classification.ml.run`,
- dependencies:
  - trainer module,
  - feature loader,
  - generic optimizer module,
  - model config,
  - `data/processed/features/latest_runs.json` pointer,
- metrics output:
  - `experiments/metrics/classification_results.json` (cache false),
- model output:
  - `experiments/checkpoints/classification/lightgbm_model.pkl`.

Design implication:

- DVC knows the training stage depends on feature pointers and code/config state.
- historical feature run directories are intentionally not declared as dynamic DVC outs from this stage (to preserve compatibility with your DVC behavior decisions and avoid cleanup side effects).

---

## Module responsibilities

### 1) `src/modeling/classification/ml/run.py` + `src/modeling/classification/ml/lightgbm_model.py` (entrypoint/orchestrator)

Responsibilities:

- parse CLI controls,
- load modeling config,
- compute CPU/threading plan,
- resolve and load selected feature run,
- prepare X/y using feature manifest schema,
- initialize MLflow tracking experiment,
- execute one of two training branches:
  - no-optuna midpoint baseline,
  - Optuna-guided parameter search,
- train final model on train+val,
- evaluate on test,
- write local metrics/model artifacts,
- log metrics/params/artifacts to MLflow.

### 2) `src/modeling/common/feature_loader.py` (feature-run resolver + reader)

Responsibilities:

- resolve which run directory to use,
- read manifest and required split files,
- return dataframes + manifest + resolved run dir.

Resolution paths:

- explicit run id (highest specificity): `data/processed/features/<task>/runs/<run_id>`,
- explicit run dir path,
- profile lookup in `latest_runs.json` under `latest_by_task_profile`,
- fallback to `latest_by_task`.

Namespace behavior:

- feature runs are now resolved under dataset-aware roots and split-path roots (`path_a` / `path_b`), not only a flat task root.

### 3) `src/modeling/common/hyperparameter_optimizer.py` (generic Optuna helper)

Responsibilities:

- convert search-space specs to Optuna suggestions,
- derive deterministic midpoint parameter sets,
- run Optuna optimization for any objective callable,
- support parallel trial workers (`n_jobs`) and callback hooks.

Key design decision:

- optimizer is generic and model-agnostic; model-specific logic lives in the trainer objective function.

### 4) `src/modeling/common/system_resources.py` (CPU planning)

Responsibilities:

- detect logical and physical core counts,
- compute effective thread budget with safety reserve and optional cap.

Design intent:

- avoid accidental CPU oversubscription,
- preserve workstation responsiveness,
- give explicit control for throughput vs stability tradeoff.

### 5) `src/mlflow_setup.py` and `src/modeling/common/experiment_tracker.py`

Responsibilities:

- map task names to experiment names,
- load DagsHub credentials from environment/.env,
- initialize DagsHub MLflow tracking backend,
- set active experiment before logging starts.

Current behavior:

- classification maps to experiment `Task_B_Classification`.

---

## Detailed call chain

### CLI to artifact call chain

1. User invokes:
   - `uv run python -m src.modeling.classification.ml.run ...`
2. `main()` parses flags and loads `configs/model_config.yaml`.
3. `_resolve_threading()` computes execution plan from:
   - detected CPU resources,
   - `training.threading` config,
   - optional CLI overrides.
4. `load_features_for_task()` resolves feature run and loads:
   - `train.parquet`, `val.parquet`, `test.parquet`,
   - `features_manifest.json`.
5. Trainer reads schema from manifest:
   - `final_features` list,
   - `label_column`.
6. Label encoder maps raw labels to numeric classes.
7. `init_tracking("classification")`:
   - validates credentials,
   - initializes DagsHub MLflow,
   - sets experiment.
8. `mlflow.start_run(...)` opens run context.
9. Trainer logs run metadata, threading plan numbers, and feature selection metadata.
10. Branch:
    - if `--no-optuna`: build midpoint params from search space,
    - else: run Optuna objective loop, capture best params and trials artifact.
11. Final model fits on concatenated train+val.
12. Predictions on test produce:
    - accuracy,
    - weighted F1,
    - macro F1,
    - weighted PR-AUC.
13. Leakage suite runs on train/val with the fitted model and logs a structured report.
14. Local artifacts written:
    - metrics JSON,
   - leakage report JSON,
   - comparison record JSONL append,
   - joblib model package.
15. MLflow logs:
    - metrics,
    - params,
    - tags,
    - artifacts,
    - feature manifest,
    - threading plan.
16. run completes and is visible in DagsHub MLflow UI.

### Evaluation and statistical comparison integration

The training runtime now emits two evaluation-focused outputs per run:

- `experiments/metrics/classification_leakage_report.json`
- `experiments/metrics/classification_comparison_records.jsonl`

These outputs support post-run significance testing with:

- `python -m src.evaluation.compare_classification_runs`

The comparison script performs paired Wilcoxon testing and Cohen's d using run records grouped by a selected field (for example `feature_profile`).

---

## Feature-run selection semantics

The system supports three ways to select feature inputs.

### A) `--run-id` (recommended for strict reproducibility)

Use when you want an exact historical feature snapshot.

- deterministic selection,
- strongest auditability,
- ideal for papers/reporting and reruns.

### B) `--run-dir`

Use when a run is outside default resolver logic or in custom location.

- explicit filesystem control,
- bypasses pointer logic.

### C) `--profile` with latest pointers

Use when you want "most recent run for this profile" convenience.

- reads `latest_runs.json`,
- convenient for iterative workflow,
- less strict than run id pinning if new runs are generated later.

Decision summary:

- experiments that must be reproducible across time should always pin `--run-id`.

---

## Configuration semantics (`configs/model_config.yaml`)

### Fields actively consumed by current trainer

- `experiment.seed`
- `classification.active_model`
- `classification.r1_models.lightgbm` search space
- `classification.hpo.direction`
- `classification.hpo.n_trials_phase1`
- `classification.hpo.timeout_seconds`
- `training.threading.*`

### Fields not actively consumed by current trainer (yet)

- many anomaly/forecasting/multitask sections,
- deep-learning section placeholders,
- generic training optimizer/scheduler fields.

Design implication:

- config currently mixes active execution config with future roadmap config.

Recommendation:

- keep this file for now, but eventually split into:
  - `model_runtime_config.yaml` (strictly consumed keys),
  - `model_research_roadmap.yaml` (future experiments),
- this reduces ambiguity and stale-key risk.

---

## Hyperparameter optimization design

### Why generic Optuna wrapper

The wrapper is intentionally model-agnostic to avoid coupling optimization framework to one estimator.

- objective callable defines model-specific training/evaluation,
- search-space parser supports compact list/range specs and explicit typed specs,
- midpoint fallback can build deterministic baseline params without Optuna.

### Branch behavior

No Optuna (`--no-optuna`):

- fastest sanity/preflight path,
- uses midpoint parameterization,
- good for pipeline validation and smoke checks.

With Optuna:

- evaluates candidate params on train/val split,
- logs per-trial completion signal (number, value, best-so-far),
- stores trials dataframe as artifact.

### Alignment with split strategy rationale

Based on the project split rationale in [../docs/SPLIT_DECISIONS.md](../docs/SPLIT_DECISIONS.md):

- intended methodology for tuning is segment-ordered temporal CV inside training,
- current implementation is a temporal holdout tuning scheme (single fixed train/val).

Interpretation:

- the current approach is leakage-aware and operationally valid,
- but it does not yet satisfy the stricter segment-ordered temporal CV objective described in split decisions.

Implication for reporting:

- if running with current code, document this as holdout-based temporal tuning,
- do not claim segment-ordered temporal CV was executed during Optuna search.

---

## CPU threading and parallelism decisions

### Two layers of parallelism

1. model-level threads (LightGBM `n_jobs` and `num_threads`),
2. Optuna trial-level concurrency (`n_jobs` in study optimize).

### Planning algorithm

- detect cores,
- choose base (physical preferred by default),
- reserve system cores,
- cap with max threads if requested,
- divide remaining budget across trial workers.

### Why this matters

- prevents hidden oversubscription,
- makes performance tunable and predictable,
- stabilizes workstation responsiveness under HPO loads.

### Example interpretation

For a plan like:

- logical=16,
- physical=8,
- budget=7,
- optuna trials=2,
- threads per trial=3,

you get roughly 6 training threads active across parallel trials plus headroom.

---

## Metrics, artifacts, and observability

### Local artifacts

- metrics json: `experiments/metrics/classification_results.json`,
- serialized model package: `experiments/checkpoints/classification/lightgbm_model.pkl`.

### MLflow artifacts and metadata

Logged per run:

- key params and metrics,
- feature run metadata (`feature_run_dir`, optional run id),
- full `features_manifest.json`,
- `threading_plan.json`,
- Optuna trials csv (if Optuna enabled).

### Runtime logs (console)

Current logs include:

- thread plan,
- selected feature run details,
- dataset shapes and classes,
- Optuna trial completion events,
- final metrics summary and artifact paths.

### Known noise and mitigation

- LightGBM can emit many split warnings on some parameter/data regimes.
- default `verbosity=-1` is set in model params to keep logs actionable.

---

## Data contract from featurization to modeling

The model trainer assumes the feature run directory contains:

- `train.parquet`, `val.parquet`, `test.parquet`,
- `features_manifest.json`.

Manifest contract used by trainer:

- `final_features`: exact input columns,
- `label_column`: target column name.

Why this contract is critical:

- protects against accidental column mismatch,
- decouples modeling from featurizer internals,
- allows historical feature experiments to remain self-describing.

---

## Failure modes and diagnostics

### 1) `ModuleNotFoundError: src`

Cause:

- command launched outside project root/package context.

Fix:

- run via uv with project directory set, for example:
  - `uv run --directory <PFE_Experiments> python -m src.modeling.classification.ml.run ...`

### 2) missing DagsHub credentials

Cause:

- environment missing `DAGSHUB_USERNAME`, `DAGSHUB_REPO`, `DAGSHUB_USER_TOKEN`.

Fix:

- set in `.env` or shell environment.

### 3) missing feature files or manifest keys

Cause:

- selected run path invalid or incomplete,
- featurization output contract changed.

Fix:

- inspect chosen run directory,
- verify manifest contains `final_features` and `label_column`.

### 4) Optuna slows workstation heavily

Cause:

- high trial concurrency + high per-trial threads.

Fix:

- reduce `optuna_parallel_trials`,
- lower `max_threads`,
- increase `reserve_cores`.

---

## Recommended operating modes

### Preflight verification mode

Use for fast sanity checks before expensive experiments.

- set `--no-optuna`,
- pin a run id,
- keep moderate threads,
- confirm metrics/model artifact write and MLflow run visibility.

### Research tuning mode

Use for model search.

- enable Optuna,
- tune trial count + timeout,
- explicitly define threading plan,
- monitor trial logs and resource pressure.

### Reproducible reporting mode

Use for final thesis reporting figures/tables.

- always pin `--run-id`,
- persist model config snapshot with run metadata,
- archive resulting run ids and metrics artifacts.

---

## Architectural rationale summary

Core decisions and why they were chosen:

1. Train from feature runs, not raw splits:
   reproducibility and schema stability.

2. Generic optimizer utility:
   future model flexibility without rewiring optimization plumbing.

3. Manifest-driven schema:
   explicit input contract and defensive failures.

4. Explicit CPU planning:
   predictable performance and reduced contention.

5. MLflow + DagsHub first-class integration:
   shared, remote, auditable experiment history.

6. Local artifacts + remote logs dual strategy:
   resilient workflow for both offline inspection and centralized tracking.

---

## Current limitations and next improvements

Limitations:

- trainer currently enforces `active_model == lightgbm`,
- no CV loop yet for robust hyperparameter evaluation,
- config still contains non-runtime roadmap sections.

High-value next steps:

1. implement model registry in trainer for catboost/extra_trees,
2. add CV objective option with time-aware folds,
3. split runtime config from roadmap config,
4. add structured per-trial JSONL logs for deeper observability,
5. add confusion matrix and class-wise plots as artifacts.

---

## Quick command reference

Thread plan only:

- `uv run python -m src.modeling.classification.ml.run --show-thread-plan`

Preflight (deterministic run id, no HPO):

- `uv run python -m src.modeling.classification.ml.run --dataset costa --task classification --split-path path_a --run-id plus_physics__196472d26dd0 --no-optuna --threads 8`

HPO run with explicit concurrency:

- `uv run python -m src.modeling.classification.ml.run --dataset costa --task classification --split-path path_a --run-id plus_physics__196472d26dd0 --n-trials 50 --threads 8 --optuna-jobs 2`

---

## Mental model to keep in mind

Think of the system as three boundaries:

1. feature boundary: featurizer emits a self-contained run package,
2. training boundary: trainer consumes package + config and emits model/metrics,
3. tracking boundary: MLflow captures enough metadata to reconstruct what happened.

If each boundary keeps a strict contract, experimentation stays fast while remaining explainable and reproducible.
