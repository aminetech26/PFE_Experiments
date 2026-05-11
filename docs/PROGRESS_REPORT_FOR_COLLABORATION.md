# Progress Report and Publication Readiness Brief

## Project Context
This report summarizes the current advancement of our PV fault intelligence project and clarifies where we are already operational versus where we are in active experimentation.

Objective of the project:
- Build a reproducible and scientifically defensible pipeline for:
  - Task A: anomaly detection
  - Task B: fault classification
  - Task C: fault prediction (early warning)
- Validate all steps on real plant data from La Reunion.
- Extend toward cross-climate and cross-plant robustness for publication-grade results.

Current reference dataset in production workflow:
- University of La Reunion data (dt1, dt2, dt3), around 51 million rows across meteorological and inverter signals.

## 1. Strong Experimental Setup and Reproducibility Backbone

### 1.1 End-to-end experiment orchestration
The workflow is formalized as DVC stages with explicit dependencies and outputs:
- ingest -> split -> preprocess -> featurize -> train

This creates:
- deterministic lineage from raw data to model artifacts,
- reproducible re-runs,

### 1.2 Experiment tracking and audit trail
We use MLflow (connected to DagsHub) for:
- run parameters,
- metrics,
- artifacts,
- feature-manifest logging,
- model and leakage-report traceability.

This means each model result is linked to a concrete feature run and config state.

### 1.3 Data lineage and feature run versioning
Feature engineering outputs are versioned as task-aware runs under:
- data/processed/features/<task>/runs/<profile>

Each run includes:
- train/val/test engineered data,
- features_manifest.json,
- resolved_config.json,
- latest pointer updates per task/profile.

This allows exact reconstruction of any reported result.

## 2. Data Pipeline and EDA-to-Engineering Methodology

### 2.1 Real-data ingestion and synchronization
From La Reunion:
- dt2 (faulty inverter with labels),
- dt3 (healthy inverter),
- dt1 (meteorology).

Core merge design:
- as-of temporal join between electrical and meteorological streams,
- tolerance control to prevent stale matches,

### 2.2 Task-aware splitting strategy
Because faults are temporally concentrated, naive global temporal splitting is invalid for evaluation.

Implemented split strategy:
- Task A semi-supervised: normal-only training, mixed val/test.
- Task A supervised: temporal-stratified split by segments.
- Task B classification: evaluable classes only (3.1, 3.2, 4.0), with train-only classes retained for boundary learning in Task A.
- Task C prediction: episode-based split logic defined to avoid pre-fault leakage.

This directly addresses temporal leakage and class support constraints.

### 2.3 Preprocessing decisions grounded in EDA
The preprocessing policy is explicit and justified:
- Missing values:
  - currently disabled for Costa because the retained post-ingestion subset does not justify imputation overhead.
- Outliers:
  - conservative IQR (3x) winsorize strategy on normal data only,
  - fault outliers preserved as potential signatures.
- Shift-aware transforms:
  - no longer part of preprocessing for Costa,
  - moved to feature engineering as explicit handcrafted representations.

## 3. Data Challenge: Imbalance and Temporal Bias Handling

The dataset is highly imbalanced (normal class dominant), with strong temporal concentration of fault events.

### 3.1 Implemented imbalance controls
Implemented and active in the current Task B baseline:
- class weighting,
- segment-aware temporal stratification,
- metric choice aligned with imbalance (weighted F1, macro F1, weighted PR-AUC).

### 3.2 Active advanced imbalance workstream
The following methods are part of our active experimentation roadmap and design notes:
- stratified temporal undersampling,
- instance hardness thresholding,
- focal loss (for deep models),
- self-paced learning style curricula,
- comparative protocol across multiple resampling strategies.


## 4. Feature Engineering Progress (Structured by Families)

Our feature design follows a family-based approach, with ablation-ready toggles.

### 4.1 Temporal features
- cyclic hour encoding,
- derivatives (segment aware) per unit time (dI/dt, dV/dt, dP/dt when enabled and valid).

### 4.2 Physics-informed features
- delta temperature (TPV - TA),
- irradiance-normalized channels (from feature engineering stage),
- performance-ratio style constructs and normalization logic.

### 4.3 Signal/statistical enrichment
- optional wavelet-denoised signal feature path,
- tsfresh segment-level descriptors (minimal/extensive modes),
- controlled top-k tsfresh selection for compute efficiency.

### 4.4 EDA-guarded feature selection
To prevent amplifying hidden redundancy:
- EDA-based prior anchors,
- optional EDA pre-drop of redundant candidates,
- train-only correlation pruning,
- train-only VIF pruning,
- protected anchor retention policy where possible.

## 5. Modeling Progress by Task

## 5.1 Task A - Anomaly detection
Current status:
- split and evaluation design are formalized,
- model families and threshold calibration protocol are documented,

Model architecture and training:
- Tested architecture: LSTM Autoencoder (LSTM-AE).
- Encoder: stacked LSTM layers.
- Decoder: stacked LSTM layers.

Results:
- The trained model achieved a reconstruction error of 0.045 Mean Absolute Error (MAE) on the test dataset, indicating satisfactory learning of normal behavior and practical anomaly detection capability.

Planned design highlights:
- semi-supervised and supervised variants,
- PR-driven threshold calibration,
- focal-loss compatible deep variants.

## 5.2 Task B - Fault classification (implemented baseline)

- LightGBM trainer
Representative run progression on La Reunion evaluable classes:
- tuned runs (50 TPE trials): weighted F1 around 0.773 to 0.775, weighted PR-AUC around 0.962 to 0.965.

Interpretation:
- strong gain from structured feature profiles and TPE-based tuning,
- remaining class-wise imbalance challenge visible in macro behavior, which motivates the advanced imbalance methods listed above.

## 5.3 Task C - Fault prediction
Current status:
- episode-based formulation and split protocol are defined,
- leakage-avoidance strategy around pre-fault windows is documented,
- full trainer implementation is in progress.

## 5.4 Multi-task architecture direction
We are building toward:
- shared encoder with task-specific heads,
- flagged anomaly -> fault classification pipeline contract,
- joint optimization with robust task-weighting strategies.

This architecture is designed for operational relevance (detect, diagnose, anticipate).

## 6. Hyperparameter Optimization and Training Strategy

Our protocol is intentionally staged:
- Phase 1: broad TPE-guided search for family-level candidate quality.
- Phase 2: focused refinement around winners (especially for deep/multi-task setups).

This staged HPO design improves compute efficiency.

## 7. Evaluation and Leakage-Control Pipeline

Evaluation is treated as a first-class module, not a post-hoc check.

Integrated checks include:
- label-shuffle stress test,
- duplicate overlap check,
- feature-importance audit,
- suspicious-performance sanity checks,
- bootstrap confidence interval estimation.

Why this matters:
- prevents overclaiming,
- identifies fragile or potentially leaky setups early,
- strengthens credibility of any final publication table.

Current practical note:
- the leakage suite is active and currently flags suspicious conditions, and is being used as a hard gate before claiming final performance.

## 8. Perspective

## 8.1 Current limitation of the present benchmark
La Reunion provides strong real-world grounding, but still represents a limited climate and plant context.

Known generalization risk:
- cross-climate shift,
- cross-plant operating regimes,
- degradation and domain drift effects over time.

## 8.3 Why local dataset would be high-value
Your dataset can enable:
- external validation beyond the La Reunion domain,
- quantification of domain shift impact,
- stronger claims on model robustness and generalization,
