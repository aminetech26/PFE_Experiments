# Next Session Guide - Costa-First Research Track

**Last Updated:** April 24, 2026  
**Active Dataset:** `costa`  
**Status:** Costa is now the canonical vertical benchmark across config, ingestion, EDA, and DVC.

---

## What Changed

- Thesis scope is now explicitly `fault detection + fault classification`.
- Forecasting is **not** a separate task anymore; residual analysis is treated as one anomaly-detection family, not its own pipeline task.
- `Costa` is the active dataset through config via `PFE_Experiments/configs/data_config.yaml`.
- Costa ingestion now trims to `irr >= 100 W/m²` using a permissive `>=` rule.
- Costa timestamps remain synthetic but physically coherent and solar-aligned.
- Costa EDA code was cleaned to reflect ingest-time trimming; old `is_daytime`-based legacy branches were removed.
- DVC now follows `active_dataset` instead of hardcoding dataset names.
- Prediction-related config and split code were removed from the active pipeline.

---

## Current Ground Truth

### Thesis / scope

- Primary dataset: `Costa`
- Primary tasks:
  - `Task A`: fault detection / anomaly detection
  - `Task B`: fault classification
- Deployment remains mandatory, but not the main scientific novelty.

### Costa ingestion assumptions

- Raw Costa files do not provide trustworthy absolute timestamps.
- Synthetic epoch is kept at `2020-04-06T03:38:00Z`.
- Epoch is justified by solar coherence with Curitiba, not by exact paper schedule reconstruction.
- Ingestion trims Costa to `irr >= 100 W/m²`.
- Resulting ingested Costa size is `515,958` rows, which closely matches the widely cited paper-scale subset.

### Config / pipeline state

- `active_dataset: costa` is defined in `PFE_Experiments/configs/data_config.yaml`.
- These scripts now default to the active dataset:
  - `src/data/ingestion.py`
  - `src/data/split_pipeline.py`
  - `src/data/preprocess_pipeline.py`
  - `src/data/featurize_pipeline.py`
  - `src/data/eda_pipeline.py`
  - `src/data/eda_visualization.py`
- `PFE_Experiments/dvc.yaml` now uses `${active_dataset}`.

---

## Important Files To Read First

- `docs/THESIS_POSITIONING.md`
- `docs/RESEARCH_QUESTIONS.md`
- `docs/COSTA_VERTICAL_PLAN.md`
- `PFE_Experiments/configs/data_config.yaml`
- `PFE_Experiments/dvc.yaml`
- `PFE_Experiments/src/data/ingestion.py`
- `PFE_Experiments/src/data/eda_pipeline.py`

---

## Commands To Resume Work

Run from `PFE_Experiments/`.

```bash
uv run python -m src.data.ingestion
uv run python -m src.data.split_pipeline
uv run python -m src.data.preprocess_pipeline
uv run python -m src.data.eda_pipeline

# Task B (classification) - current clean runner
uv run python -m src.data.featurize_pipeline --dataset costa --task classification --split-path path_a --profile plus_physics
uv run python -m src.modeling.classification.ml.run --dataset costa --task classification --split-path path_a --profile plus_physics

# Task A (detection) Path B continuity-oriented baseline (Matrix Profile)
uv run python -m src.data.featurize_pipeline --dataset costa --task anomaly_semisup --split-path path_b --profile baseline_raw
uv run python -m src.modeling.anomaly_detection.ml.run --model matrix_profile --dataset costa --task anomaly_semisup --split-path path_b --profile baseline_raw
```

If needed, you can still override the active dataset explicitly:

```bash
uv run python -m src.data.ingestion --dataset la_reunion
```

---

## Most Likely Next Tasks

### 1. Rebuild Costa artifacts end-to-end

Because ingestion and EDA assumptions changed, regenerate:

- ingested Costa parquet
- Costa splits
- Costa preprocessing outputs
- Costa EDA artifacts
- Costa feature runs

### 2. Audit preprocessing under the new Costa subset

Main question:

- do the minimal preprocessing outputs and manifests stay coherent across both `Path A` and `Path B` after the preprocessing scope reduction?

Irradiance-aware normalization is no longer a preprocessing concern for Costa; it now belongs to feature engineering.

### 3. Start Costa vertical benchmarking

Treat Costa vertical work as two parallel benchmark tracks:

- `Track A - fault detection`
- `Track B - fault classification`

Priority order:

1. establish a reproducible baseline for both detection and classification
2. strengthen tabular baselines for both tracks
3. run feature-family and window-size ablations for both tracks where relevant
4. add calibration and thresholding analysis, especially for detection
5. try deep baselines only after the tabular phase is strong

### 4. Track vertical Costa progress explicitly

Use this progression as the working ladder:

1. `Stage V0 - pipeline integrity`
   - ingestion, splits, preprocessing, EDA, and featurization all rerun cleanly
   - manifests and row counts are coherent across stages
   - Costa remains at `515,958` ingested rows after `irr >= 100 W/m²` trimming

2. `Stage V1 - reproducible baseline`
   - run the first detection baseline and the first classification baseline with the current feature stack
   - log metrics, confusion matrices, leakage report, thresholds, and exact feature profile
   - establish the true starting point before optimization

3. `Stage V2 - paper benchmark chase`
   - target the Costa paper detection reference `93.09%`
   - target the Costa paper classification reference `95.44%`
   - do not broaden scope before this stage is credible

4. `Stage V3 - explanatory ablations`
   - feature-family ablations
   - window-size / segmentation sensitivity
   - calibration and threshold checks
   - identify what actually drives gains rather than only reporting a best score

5. `Stage V4 - model family expansion`
   - after tabular baselines are strong, compare with `1D CNN`, `TCN`, then recurrent models if needed
   - keep deployment constraints visible while comparing

6. `Stage V5 - deployment candidate selection`
   - choose one model based on performance + deployability
   - freeze the Costa vertical winner before moving to horizontal generalization

### 5. Clean remaining docs/examples

Some docs may still reflect:

- old `la_reunion` defaults,
- old 3-task language,
- or outdated preprocessing / split narratives.

---

## Recommended Immediate Focus

The next serious session should do this in order:

1. finalize feature-engineering infrastructure for `Path A` / `Path B`
2. harden handcrafted features with EDA-guided pre-drop + train-only pruning
3. implement handcrafted statistics in phases: rolling -> windows -> multi-scale windows
4. keep spectral/signal-processing additions restricted to `Path B`
5. run first Costa detection/classification benchmarks with the hardened handcrafted stack
6. compare against the paper targets `93.09%` and `95.44%`
7. log the result as the current vertical stage (`V1`, `V2`, etc.)

If there is extra time after that:

8. formalize the thresholding/calibration protocol for Costa detection
9. start representation-learning comparator planning (TS2Vec / autoencoder) after handcrafted baselines stabilize

---

## Vertical Progress Snapshot Template

When resuming work, update this mentally or in notes:

- `Current stage:` `V0 | V1 | V2 | V3 | V4 | V5`
- `Best classification result:` `...`
- `Best detection result:` `...`
- `Best detection thresholding setup:` `...`
- `Best feature profile:` `...`
- `Best model family so far:` `...`
- `Main blocker:` `...`
- `Next experiment:` `...`

---

## Sanity Checks To Keep In Mind

- Costa is intentionally trimmed at ingestion; do not reintroduce legacy day/night filtering logic.
- Residual forecasting belongs inside anomaly-method comparison only; do not recreate a standalone `prediction` task.
- Keep `active_dataset` as the source of truth for pipeline defaults.
- Preserve the current epoch unless new evidence is substantially stronger than solar-coherence evidence.

---

## Quick Reminder For Future You

The project is no longer “generic industrial pipeline first.”

It is now:

- `Costa-first`
- `research-first`
- `benchmark-driven`
- `deployment-required but not thesis-defining`

The immediate mission is simple: make Costa excellent before expanding scope.
