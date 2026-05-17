# PFE Master Plan - PV Fault Detection and Diagnosis
## ESI x CDER · Ahmed Amine GUERRAICHE · 2025/2026

> **Goal:** Build a research-grade, Costa-centered PV fault detection and fault classification pipeline, select a deployable winner, and complete edge deployment with GUI. Expand to generalization only after the Costa benchmark is strong.

---

## 0. Strategic Framing (Read This First)

Your supervisor's advice is correct and critical: **results first, research gaps second**.

The deliverable has a fixed spec:
1. Fault detection and fault classification on electrical + meteorological data
2. Edge deployment on Jetson Nano with real-time inference
3. GUI on Jetson Nano

The thesis is now explicitly **research-first** while still honoring the deployment obligation.

The active structure is:

- **Lane A - Must deliver:** Costa benchmark strength, reproducible evaluation, deployable winner, edge GUI.
- **Lane B - Research expansion:** La Reunion and Mendeley generalization, domain adaptation, cross-climate and cross-plant studies.

Rule: Lane B starts only after Lane A is strong.

**Mental model for the next 2 months:** You are not exploring the entire space of ML algorithms. You are running a **structured benchmark tournament** on Costa that produces a defensible final model.

---

## 1. Canonical Scope

### Primary scientific tasks

1. **Fault detection**
2. **Fault classification**

### Mandatory engineering deliverables

1. **Edge deployment**
2. **Operational GUI**

### Optional method, not a primary task

Forecasting is retained only as **residual-based anomaly analysis**. It is not treated as direct fault-onset prediction because public PV fault datasets contain artificially induced faults without defensible pre-fault signatures.

### Dataset priority

1. **Costa** - primary vertical benchmark
2. **La Reunion** - real-world transfer and anomaly-focused extension
3. **Mendeley / GPVS-Faults** - simulated source for transfer and adaptation studies

---

## 2. Program Phases

### Phase A - Costa vertical excellence

- stabilize data, splits, metrics, and leakage checks,
- build strong tabular baselines,
- run feature and window ablations,
- test priority deep models,
- choose a deployable winner.

### Phase B - Deployment completion

- deploy the selected model on Jetson Nano,
- provide GUI-based inference and visualization,
- measure latency and runtime practicality.

### Phase C - Horizontal research expansion

- evaluate transfer to La Reunion and Mendeley,
- characterize distribution shift,
- test adaptation methods such as DANN.

---

## 3. Immediate Priorities

1. Freeze the thesis positioning around Costa-first FDD.
2. Beat or strongly challenge the published Costa results.
3. Keep all experiments leakage-safe and statistically defensible.
4. Select one model with both accuracy and deployment viability.
5. Complete edge deployment and GUI.
6. Only then activate the broader generalization track.

---

## 4. Environment & Tooling Stack

### 1.1 Package Management
```
uv                          # Package manager (fast, lockfile-based)
Python 3.11                 # Stable, broad ecosystem support
```

### 1.2 Core ML / DL
```
torch >= 2.3                # PyTorch (with CUDA support)
pytorch-lightning >= 2.3    # Training loop abstraction (clean, reproducible)
torchmetrics                # Unified metric computation
scikit-learn >= 1.4         # Classical ML, preprocessing, validation
xgboost >= 2.0              # XGBoost v2 (GPU-accelerated)
lightgbm                    # LightGBM
catboost                    # CatBoost
```

### 1.3 Data Engineering
```
pandas >= 2.2               # DataFrames
polars                      # Fast columnar ops for large La Réunion files (~51M rows)
numpy >= 1.26               # Numerical ops
pyarrow                     # Parquet format (replace CSV for large files)
tsfresh                     # Automated time-series feature extraction
openpyxl / xlrd             # Excel ingestion (Sonalgaz)
```

### 1.4 Signal Processing / Feature Engineering
```
scipy                       # FFT, filters, statistical tests
PyWavelets (pywt)           # Wavelet decomposition
EMD-signal (emd)            # CEEMDAN (replaces CEEMD — more stable)
stumpy                      # Matrix profile (efficient motif/anomaly detection)
```

### 1.5 Experiment Tracking & Reproducibility
```
mlflow                      # Experiment tracking, model registry, artifact logging
dvc                         # Data version control + pipeline DAG
optuna >= 3.6               # Hyperparameter optimization (Bayesian + TPE)
```

### 1.6 Statistical Testing & Validation
```
scipy.stats                 # t-test, Wilcoxon, ANOVA
statsmodels                 # Two-way ANOVA, Granger causality, ADF test
pingouin                    # Clean API for effect size, power analysis
```

### 1.7 Online Learning / Drift Detection
```
river                       # Online learning + ADWIN / Page-Hinkley / KSWIN drift detection
```

### 1.8 Edge Deployment
```
onnx + onnxruntime          # Model export (architecture-agnostic)
torch.onnx                  # PyTorch → ONNX export
# On Jetson Nano (separate env):
tensorrt                    # ONNX → TensorRT engine
pycuda                      # GPU inference bindings
```

### 1.9 GUI
```
PyQt6 / PySide6             # Desktop GUI for Jetson Nano dashboard
pyqtgraph                   # Real-time plotting inside Qt
```

### 1.10 Code Quality
```
ruff                        # Linter + formatter (replaces flake8 + black)
pytest                      # Unit & integration tests
pre-commit                  # Git hooks
```

---

## 2. Repository Structure

```
pfe-experiments/
├── PFE_Experiments/
│   ├── data/
│   │   ├── raw/                # Symlinks or DVC pointers to data-sources/
│   │   ├── interim/            # Cleaned, pivoted, merged DataFrames (Parquet)
│   │   └── processed/          # Feature-engineered, windowed, ready-to-train
│   ├── notebooks/
│   │   ├── 01_eda_mendeley.ipynb
│   │   ├── 02_eda_sonalgaz.ipynb
│   │   ├── 03_eda_reunion.ipynb
│   │   ├── 04_feature_engineering.ipynb
│   │   └── 05_baseline_models.ipynb
│   ├── src/
│   │   ├── data/
│   │   │   ├── ingestion.py    # Dataset loaders
│   │   │   ├── preprocessing.py
│   │   │   ├── features.py     # Feature engineering
│   │   │   └── windows.py      # Sliding window creation
│   │   ├── models/
│   │   │   ├── anomaly/        # Anomaly detection models
│   │   │   ├── classification/ # Fault classification models
│   │   │   ├── forecasting/    # Fault forecasting models
│   │   │   └── multitask/      # MTL shared encoder
│   │   ├── training/
│   │   │   ├── trainer.py      # Lightning trainer config
│   │   │   └── callbacks.py
│   │   ├── evaluation/
│   │   │   ├── metrics.py
│   │   │   ├── leakage_checks.py   # Leakage prevention suite
│   │   │   └── statistical_tests.py
│   │   ├── deployment/
│   │   │   ├── export_onnx.py
│   │   │   ├── quantize.py
│   │   │   └── jetson_runtime.py
│   │   └── gui/
│   │       └── dashboard.py
│   ├── configs/                # YAML configs for each experiment
│   │   ├── data_config.yaml
│   │   ├── model_config.yaml
│   │   └── deploy_config.yaml
│   ├── experiments/            # MLflow run artifacts
│   ├── tests/
│   ├── dvc.yaml                # DVC pipeline stages
│   ├── pyproject.toml          # uv project file
│   └── .pre-commit-config.yaml
```

---

## 3. Two-Month Execution Plan

### MARCH: Data → Baseline

| Week | Focus | Key Deliverables |
|------|-------|-----------------|
| **W1 (Mar 2–8)** | Data Auditing & Ingestion | Parquet files for all 3 datasets, DVC pipeline stage 0 |
| **W2 (Mar 9–15)** | EDA + Feature Engineering | Correlation analysis done, feature set frozen, preprocessing pipeline coded |
| **W3 (Mar 16–22)** | ML Baselines (all 3 tasks) | Benchmark table: 5 ML models × 3 tasks with proper CV |
| **W4 (Mar 23–31)** | DL Models (core) | LSTM/GRU + 1D-CNN trained, comparison with ML baseline |

### APRIL: DL Advanced + Deployment

| Week | Focus | Key Deliverables |
|------|-------|-----------------|
| **W5 (Apr 1–7)** | Signal Processing Integration | CEEMDAN + Wavelet denoising layer, evaluate impact via ablation |
| **W6 (Apr 8–14)** | Multi-task Learning | Shared encoder MTL model, compare vs separate models (overhead vs gain) |
| **W7 (Apr 15–21)** | Edge Deployment | ONNX export + TensorRT on Jetson Nano, latency benchmark |
| **W8 (Apr 22–30)** | GUI + Integration + Freeze | Qt dashboard live, full end-to-end test, model frozen |

> **May 1:** Enter thesis writing mode. Code is frozen (bug fixes only).

---

## 4. Task-by-Task Model Selection Strategy

### The Rule: Funnel, Don't Explore Everything Simultaneously

Each task goes through 3 rounds:
- **Round 1** (R1): Fast ML baselines — picks the best family
- **Round 2** (R2): DL models competing with R1 winner
- **Round 3** (R3): Best model refined via HPO + ablation

---

### 4.1 Task A — Anomaly Detection

**Input:** Time-window of electrical + meteorological features  
**Output:** {Normal, Anomaly} binary (or anomaly score)  
**Primary dataset:** La Réunion dt2 (Fault=0 vs Fault≠0) + Mendeley (F0 vs F1-F7)

| Round | Models to Test | Rationale |
|-------|---------------|-----------|
| R1 | Isolation Forest, One-Class SVM, LOF | Proven, minimal labeled data needed |
| R2 | LSTM Autoencoder (reconstruction error), 1D-CNN Autoencoder | Learns temporal structure |
| R3 | Best R2 + threshold calibration (PR curve, not ROC) | Class imbalance: 97% normal |

**Key choices:**
- Use **PR-AUC** (not ROC-AUC) as primary metric — data is 97% normal
- Calibrate alarm threshold on a held-out validation set
- **Skip:** Gaussian Process for anomaly (GPR is for regression; use it only as a reference model baseline, computational overhead too high for real-time)

---

### 4.2 Task B — Fault Classification

**Input:** Anomaly-flagged windows  
**Output:** {F0, F1, F2, F3, F4, F5, F6, F7} or {Normal, Shading, Open-Circuit, Short-Circuit, Inverter, Arc, Ground}  
**Primary dataset:** Mendeley (8 labeled fault types) + La Réunion (5 fault codes)

| Round | Models to Test | Rationale |
|-------|---------------|-----------|
| R1 | LightGBM, CatBoost, Extra Trees | Fast, interpretable, strong baselines |
| R2 | 1D-CNN, GRU, Temporal CNN (TCN) | Temporal hierarchy matters for fault signatures |
| R3 | Best + SHAP explanations | Explainability for PFE defense and publication |

**Key choices:**
- Use **stratified k-fold** (not random) because class imbalance
- Use **Focal Loss** (not cross-entropy) for DL models — handles imbalance inherently
- **Skip for now:** RBM, DBM, DeepBoltzmann — outdated architectures, not competitive with modern CNNs/GRUs on time-series, no maintained PyTorch implementation
- **Skip for now:** KAN (Kolmogorov-Arnold Networks) — very new (2024), promising but unstable training, limited tabular benchmarks. Add to bonus track.
- **Skip for now:** SwinTransformer — designed for vision, not 1D time-series. Use vanilla Transformer encoder or TCN instead.

---

### 4.3 Task C — Fault Forecasting (5–10 min horizon)

**Input:** Sliding window of past observations  
**Output:** Fault probability at t+5min and t+10min  
**Dataset:** La Réunion dt2 (temporal continuity required)

| Round | Models to Test | Rationale |
|-------|---------------|-----------|
| R1 | XGBoost with lag features, ARIMA variant | Strong baselines, fast |
| R2 | LSTM, GRU, TCN | Sequence modeling |
| R3 | Time-MoE or TimesFM (zero-shot evaluation) | Foundation models as comparison point |

**On Time-MoE and TimesFM:**
- Both are worth evaluating in **zero-shot mode** as upper bounds / comparison benchmarks
- Time-MoE (2.4B): Run inference only, don't fine-tune (too expensive without A100+)
- TimesFM (Google): Similarly, use as zero-shot reference
- **They will NOT be your deployed model** on Jetson Nano — they're too large (GBs)
- Your deployed forecasting model will be a fine-tuned GRU or TCN
- Document this properly: "Foundation models serve as performance upper bounds"

---

## 5. Feature Engineering — Refined Strategy

### 5.1 Correlation Analysis — Which to Use

| Method | When to Use | Priority |
|--------|------------|---------|
| **Spearman** | Default for non-linear, non-Gaussian features | **HIGH — use always** |
| **Pearson** | Only as secondary reference for linear features | Medium |
| **Autocorrelation (ACF/PACF)** | Lag selection for forecasting (AR order) | **HIGH for Task C only** |
| **Granger Causality** | Meteorological → electrical causal direction | **HIGH for feature selection justification** |
| Rolling correlation | Checking non-stationarity / drift over time | Medium (use during EDA, not as a feature itself) |

**Skip:** Mutual Information is more informative than Pearson for non-linear but use scipy.stats.spearmanr as default.

### 5.2 Feature Engineering Priority List

**Physical features (high value, domain-justified):**
```
Performance Ratio (PR) = P_measured / (G/G_ref × P_STC)
Fill Factor (FF) = Pmax / (Voc × Isc)    # Mendeley only (has V, I)
Power ratio = P_actual / P_expected_from_irradiance
dP/dt, dI/dt, dV/dt                       # Rate of change (transient fault signatures)
ΔT = T_module - T_ambient                 # Thermal stress indicator
Normalized Vpv = Vpv / Vpv_rolling_mean   # Remove diurnal trend
Current Imbalance (Mendeley): max(ia,ib,ic) - min(ia,ib,ic)
THD proxy: std(ia,ib,ic) over window
```

**Statistical window features (apply sliding window of size W):**
```
mean, std, min, max, skewness, kurtosis   # Per channel per window
zero-crossing rate                         # For AC current channels
energy, RMS                                # Electrical energy in window
```

**Automated:** Run `tsfresh` (minimal feature set, not full) on a representative 10K sample first to find top-20 discriminative features before running on full dataset.

### 5.3 Signal Processing Layer — What to Use

| Method | Verdict | Rationale |
|--------|---------|-----------|
| **CEEMDAN** | **Use for Mendeley** (10kHz data) | Better than CEEMD (more stable). Decompose Vpv/Ipv into IMFs, use mean/energy of each IMF as features. The 10kHz Mendeley data is ideal for this. |
| **Wavelet (db4, level 3–5)** | **Use for La Réunion** (7s sampling) | Good for non-stationary signals at multiple scales |
| **FFT / periodogram** | Use as feature only | For Mendeley: spectral content of fault signals |
| Hilbert-Huang Transform | Skip for now | CEEMDAN + Hilbert is the same pipeline but adds complexity; beneficial only if CEEMDAN alone insufficient |
| Kalman Filter | Use for denoising Sonalgaz data | Smooth noisy daily operational data before feeding to models |

**On Gramian Angular Summation Field (GASF) + CNN:**
- Interesting and publishable, but computationally heavy
- **Strategy:** Implement it only for classification (Task B) on Mendeley dataset as a comparison, not the primary approach. Good figure for the paper.

### 5.4 Dimensionality Reduction — Which to Use

| Method | When |
|--------|------|
| **PCA** | After feature engineering to compress feature matrix (keep 95% variance) |
| **Linear Discriminant Analysis (LDA)** | Supervised DR for classification — compare with PCA |
| **Autoencoder (learned DR)** | When training DL model: encoder naturally does this |

**Skip:** SOM (Self-Organizing Maps) — useful for visualization but not competitive as a feature reducer; use UMAP instead for any visualization needs. **Skip:** Quadratic DA (computationally expensive, not needed).

---

## 6. Hyperparameter Optimization Protocol

**Use Optuna with TPE (Tree-structured Parzen Estimator) as default.** It is the pragmatic gold standard.

**Strategy: Two-Phase HPO**
```
Phase 1 — Wide search (50–100 trials): Random sampler to cover the space
Phase 2 — Narrow search (50 trials): TPE bayesian on promising region
```

**On the paper you mentioned (HPO with surrogate models + simulated annealing + cooperative coevolution):**
- The ideas are solid academically (multi-level hierarchy + progressive freezing is smart)
- **For your timeline:** too complex to implement from scratch
- **Practical compromise:** Use Optuna's built-in pruning (Hyperband/Successive Halving) which achieves similar early stopping benefits without custom implementation
- If you have extra time in Week 7, implement the hierarchical HPO for the classification head only as a comparison experiment — good material for a paper section.

**Skip:** Genetic algorithms / evolutionary HPO — overkill given Optuna's bayesian alternatives.

**On Muon Optimizer:**
- Very new (2024), proven mainly on LLM pre-training, not on time-series classification
- **Use AdamW with cosine annealing warmup as default**
- Only test Muon if AdamW results are plateauing and you have extra experiment budget

---

## 7. The Multi-Task Learning (MTL) Question

**Recommendation: Start separate, then merge.**

```
Phase 1 (Week 3–4): Train Task A, B, C as fully separate models
Phase 2 (Week 6):   Build MTL model with shared encoder
Phase 3:            Compare: MTL vs separate on (accuracy, inference time, model size)
```

**MTL architecture to try:**
```
Input → Shared Encoder (3–4 layer GRU or TCN)
              ↓
    ┌─────────┬──────────┬──────────────┐
    │Anomaly  │ Fault    │  Forecasting │
    │Head     │ Classif  │  Head        │
    │(sigmoid)│ Head     │  (sigmoid /  │
    │         │(softmax) │   linear)    │
    └─────────┴──────────┴──────────────┘
```

**Loss:** `L_total = α·L_anomaly + β·L_classif + γ·L_forecast`
Start with equal weights (α=β=γ=1), then tune.

**Why this is publishable:** Most papers treat these as separate problems. A unified MTL model with a principled loss weighting is a legitimate research contribution.

---

## 8. Rigorous Scientific Protocol (Anti-Leakage Checklist)

This maps directly to the image you shared.

### 8.1 Data Split Strategy

```
TimeSeriesSplit (sklearn) — NEVER shuffle time-series data

Split protocol:
├── 70% Training
├── 15% Validation (HPO, early stopping)
└── 15% Test (touch ONCE, at the very end)

For cross-validation: Blocked Time-Series CV (gap between folds)
```

**Visual:**
```
|---train---|gap|val|gap|---train---||---train---|gap|val|gap|test|
```

### 8.2 Leakage Prevention Checklist (per the shared table)

| Check | When | How |
|-------|------|-----|
| **Label Shuffle Test** | After training | Retrain with shuffled labels → accuracy should drop to random |
| **Duplicate Sample Check** | Before split | `df.duplicated()` across the split boundary |
| **Time-Split Validation** | Always | NEVER use random split on time-series |
| **Feature Importance Audit** | After R1 | If a feature has >60% importance, inspect it for leakage |
| **Preprocessing Inside CV** | Always | Fit scaler on train fold ONLY, transform val/test |
| **Nested CV / HPO Audit** | HPO phase | Inner loop = HPO, outer loop = generalization estimate |
| **External Dataset Validation** | After model freeze | Validate La Réunion model on Mendeley (and vice versa) |
| **Feature Engineering Timeline Check** | During engineering | No future data can contribute to a feature at time t |
| **Performance Sanity Check** | After each run | >98% accuracy on tabular is suspicious — investigate |
| **Bootstrap / Resample Validation** | Final model | 1000-sample bootstrap CI on F1, AUC |

### 8.3 Evaluation Metrics by Task

| Task | Primary Metric | Secondary | Rationale |
|------|---------------|-----------|-----------|
| Anomaly Detection | PR-AUC, F1 (anomaly class) | ROC-AUC | 97% normal class imbalance |
| Classification | Weighted F1, per-class F1 | Confusion matrix | Multi-class imbalance |
| Forecasting | MAE, RMSE | CRPS (probabilistic) | Lead time matters |

### 8.4 Statistical Significance

After getting all model results:
1. **Wilcoxon signed-rank test** (paired) between best model and second-best: p < 0.05
2. **Two-way ANOVA:** Factor A = model architecture, Factor B = dataset. This directly answers "is the improvement consistent across datasets or just on one?"
3. **Effect size (Cohen's d or η²):** Report alongside p-value
4. **Confidence intervals:** Always report as `mean ± std (95% CI)` not just mean

### 8.5 Ablation Study Template

For your final model, run:
```
Full model (all components)          → baseline score
  − Signal denoising (CEEMDAN/Wavelet)  → Δ impact of denoising
  − Physics features (PR, FF, ΔT...)    → Δ impact of domain features
  − Automated features (tsfresh)        → Δ impact of automation
  − HPO (use default params instead)    → Δ impact of optimization
  − MTL (use separate heads instead)    → Δ impact of multi-tasking
```

This table becomes a core section of both your PFE and your paper.

---

## 9. Edge Deployment Pipeline (Jetson Nano)

### 9.1 Quantization Strategy

```
Training (PC/Cloud):
  Float32 training → Float16 TorchScript → ONNX (opset 17)

On Jetson Nano (TensorRT):
  ONNX → TensorRT FP16 engine (latency target: <50ms/inference)
  If insufficient: INT8 with calibration dataset (post-training quantization)
```

**Layer-wise quantization (blockwise):**
- Keep first and last layers in FP16 (sensitive to precision loss)
- Quantize middle layers to INT8

### 9.2 Model Size Budget for Jetson Nano (4GB RAM)

| Component | Max Size | Reason |
|-----------|----------|--------|
| Shared encoder | <5 MB | Multiple inferences per second |
| Each task head | <1 MB | Modular loading |
| Total system (with GUI) | <500 MB RAM | Leave room for OS + Qt |

### 9.3 Inference Benchmark Targets

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Latency | < 50 ms | `torch.cuda.Event` timing |
| Throughput | > 20 samples/sec | Batch inference benchmark |
| Power | < 10W | `tegrastats` on Jetson |

### 9.4 GUI Specification (Qt Dashboard)

```
MainWindow:
├── LivePlotWidget            # Real-time electrical signals (pyqtgraph)
├── AnomalyIndicator          # Red/green LED + anomaly score
├── FaultClassLabel           # Current fault prediction + confidence bar
├── ForecastWidget            # 5 and 10-min probability bars
├── AlertLog                  # Timestamped fault history
└── SystemStatusBar           # Model loaded, inference speed, GPU usage
```

---

## 10. Online Learning & Drift Detection (Bonus Track)

**When to implement:** After the core model is working (end of Week 6 at earliest).

```python
# Recommended river pipeline
from river import drift, anomaly

detector = drift.ADWIN(delta=0.002)   # Adaptive Windowing — statistical
# OR
detector = drift.PageHinkley()         # Cumulative sum — simpler, faster

# Workflow:
for sample in stream:
    prediction = model.predict(sample)
    detector.update(prediction_error)
    if detector.drift_detected:
        # Flag for re-training or domain adaptation
        trigger_retraining_pipeline()
```

**Domain adaptation (research track, post-May):**
- Maximum Mean Discrepancy (MMD) to measure La Réunion → Algeria distribution shift
- STC normalization using plant datasheet parameters before feeding to model

---

## 11. Week-by-Week Checklist

### Week 1 (Mar 2–8): Data Ingestion & Audit
- [ ] Set up uv project, pyproject.toml, pre-commit, DVC init
- [ ] MLflow server running locally
- [ ] Write `ingestion.py`: load all 3 datasets → Parquet
- [ ] Explore Sonalgaz `Situation des perturbations` sheets (potential fault labels!)
- [ ] Run `analyze_data.py` extended version → full statistics per dataset
- [ ] Document: data quality issues, missing values, temporal gaps

### Week 2 (Mar 9–15): EDA & Feature Engineering
- [ ] Correlation analysis (Spearman matrix + Granger causality for forecasting)
- [ ] ACF/PACF plots on La Réunion for lag determination
- [ ] Implement physics features (PR, Fill Factor, dP/dt, ΔT)
- [ ] Implement sliding window function with configurable size + step
- [ ] Run tsfresh (minimal mode) on 10K sample → identify top-20 features
- [ ] Freeze final feature set → write to `feature_config.yaml`
- [ ] **Leakage check:** verify no future data in features

### Week 3 (Mar 16–22): ML Baselines (All 3 Tasks)
- [ ] Implement `TimeSeriesSplit` with gap for all experiments
- [ ] Fit scaler inside CV fold (StandardScaler or RobustScaler — use Robust for fault data)
- [ ] Task A: Isolation Forest, One-Class SVM — log to MLflow
- [ ] Task B: LightGBM, CatBoost, Extra Trees — log to MLflow
- [ ] Task C: XGBoost with lag features — log to MLflow
- [ ] Generate benchmark table (metric ± std from CV)
- [ ] Run label shuffle test on best R1 models

### Week 4 (Mar 23–31): DL Models
- [ ] Implement `PVDataset` (PyTorch Dataset) + `DataModule` (Lightning)
- [ ] Task A: LSTM Autoencoder — reconstruction error → anomaly score
- [ ] Task B: 1D-CNN, GRU — compare with LightGBM winner
- [ ] Task C: LSTM/GRU forecaster — 5 and 10-min horizon
- [ ] Optuna HPO sweep on best DL architecture (50 trials Phase 1)
- [ ] Compare DL vs ML baseline — update benchmark table

### Week 5 (Apr 1–7): Signal Processing Integration
- [ ] Implement CEEMDAN on Mendeley (Vpv, Ipv channels)
- [ ] Implement Wavelet denoising on La Réunion data
- [ ] Ablation: model accuracy WITH vs WITHOUT denoising layer
- [ ] (Optional) GASF image encoding for Task B classification comparison

### Week 6 (Apr 8–14): Multi-Task Learning
- [ ] Implement shared GRU/TCN encoder + 3 task heads (Lightning Module)
- [ ] Train MTL model, compare vs separate models
- [ ] MTL loss weight tuning with Optuna
- [ ] Run two-way ANOVA: factor = {MTL vs separate} × {dataset}
- [ ] Write statistical significance report (Wilcoxon + p-values)
- [ ] Final ablation study table

### Week 7 (Apr 15–21): Edge Deployment
- [ ] Export best model to ONNX (torch.onnx.export)
- [ ] TensorRT engine build on Jetson Nano
- [ ] INT8 calibration if FP16 too slow
- [ ] Latency + throughput benchmark (target <50ms)
- [ ] `tegrastats` power measurement

### Week 8 (Apr 22–30): GUI + Integration + Freeze
- [ ] Qt dashboard: live plot, anomaly indicator, fault label, forecast bars
- [ ] End-to-end test: sensor reading → inference → GUI update
- [ ] External dataset validation (La Réunion model → tested on Mendeley)
- [ ] Documentation: README, config files, experiment logs
- [ ] **CODE FREEZE — May 1**

---

## 12. Techniques to Eliminate (Save Time)

| Technique | Decision | Reason |
|-----------|----------|--------|
| Restricted Boltzmann Machines (RBM) | **Drop** | Outdated, hard to train, outperformed by modern autoencoders |
| Deep Boltzmann Machines (DBM) | **Drop** | Same — no maintained PyTorch implementation |
| Swin Transformer | **Drop** | Vision transformer, no benefit over TCN for 1D time-series |
| QDA (Quadratic Discriminant Analysis) | **Drop** | Not competitive, same family as LDA but worse |
| SOM (Self-Organizing Maps) | **Keep for visualization only** | Not useful as a feature extractor vs PCA/UMAP |
| Muon Optimizer | **Bonus only** | Not proven on time-series; use AdamW + cosine schedule |
| Full GASF + CNN pipeline | **Keep as comparison** | But not as primary; one experiment for publication value |
| Genetic Algorithm HPO | **Drop** | Optuna Bayesian is faster and better |
| Prophet for forecasting | **Drop** | Prophet is for business time-series trend forecasting, not fault signals |
| Direct ARIMA deployment | **Drop** | Use as univariate baseline only; not deployable on Jetson |
| Time-MoE / TimesFM as final model | **Keep as benchmark** | Zero-shot evaluation only; document as upper bound |
| Granger causality as feature | **Drop** | Use for lag selection only, not as a model input |
| Full tsfresh feature set | **Drop** | Minimal mode + top-20 selection only — full set = 800 features, expensive |

---

## 13. Publication Strategy (Bonus — if core works by mid-April)

**Target venue:** IEEE Transactions on Industrial Electronics, or Applied Energy, or Renewable Energy.

**Paper angle:** 
> "A Multi-Task Deep Learning System for Simultaneous Anomaly Detection, Fault Classification, and Short-Horizon Prediction in PV Systems Deployed on Edge Devices"

**What makes it publishable:**
1. MTL model unifying 3 tasks (novelty in unified formulation)
2. Rigorous statistical evaluation (most papers skip this)
3. Real edge deployment benchmark (Jetson Nano latency/power)
4. Multi-dataset validation (3 datasets, 2 climate zones)
5. Leakage-free protocol explicitly documented (rare in PV literature)

**Bonus novelty if time allows:**
- CEEMDAN + GASF hybrid feature encoding
- Domain adaptation via MMD for cross-climate generalization
- Adaptive loss weighting in MTL via meta-gradient

> Write the paper in parallel with the thesis (May–June). 80% overlap in content.

---

## 14. Daily Discipline Protocol

```
Morning (30 min): Review yesterday's MLflow runs, pick today's next experiment
Coding block: Implement, run, log to MLflow (every experiment, no exceptions)
Evening (15 min): Update checklist above, note blockers
Weekly: Commit all code + DVC push, review benchmark table
```

**Golden rule:** If it's not in MLflow, it didn't happen. Every run gets logged.

**When blocked:** Spend max 2 hours debugging. If still stuck, move to the next task and come back. Document the blocker in a `blockers.md` file.

---

*Last updated: March 1, 2026*
