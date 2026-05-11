# Costa Vertical Plan

**Author:** Ahmed Amine GUERRAICHE  
**Date:** April 2026  
**Status:** Active execution plan

---

## 1. Why Costa Is the Anchor Dataset

Costa is the first dataset on which the thesis must become strong.

It is the best vertical benchmark because it is:

- real rather than simulated,
- large enough for meaningful experimentation (~500k samples),
- diverse enough to support multiple fault classes,
- and tied to a published reference with explicit numbers to beat.

That makes Costa the right place to validate the pipeline, sharpen the methodology, and establish a thesis-quality baseline before any broader generalization claims.

---

## 2. Benchmark Targets

The reference paper reports the following Costa results:

- **Fault detection accuracy:** 93.09%
- **Fault classification accuracy:** 95.44% (best ANN result)

These numbers define the first benchmark target.

The thesis goal is not merely to cite them; it is to:

1. reproduce a credible evaluation setup,
2. match or exceed those results where possible,
3. and provide a more rigorous comparison with richer metrics and stronger experimental controls.

---

## 3. Costa Task Definitions

### 3.1 Task A - Fault detection

Binary decision:

- `0` = normal
- `1` = fault

This task answers: **Is the PV plant operating abnormally right now?**

### 3.2 Task B - Fault classification

Multi-class decision over Costa fault labels.

This task answers: **If operation is abnormal, which fault type is present?**

### 3.3 Optional analysis - Residual-based anomaly signal

This is not a primary task. It is a side analysis in which forecasting is used only to estimate expected normal behavior, with residuals acting as anomaly evidence.

---

## 4. Core Metrics

### Detection metrics

For paper comparison:

- accuracy

For thesis-quality evaluation:

- PR-AUC
- F1-score
- precision
- recall
- confusion matrix

### Classification metrics

For paper comparison:

- accuracy

For thesis-quality evaluation:

- macro-F1
- weighted-F1
- per-class F1
- confusion matrix
- calibration quality when probabilities are used operationally

### Deployment metrics

- inference latency
- model size
- memory footprint
- runtime stability on edge hardware

---

## 5. Execution Ladder

Costa vertical execution must advance on **both** primary tasks, not only classification.

- `Track A:` fault detection
- `Track B:` fault classification

### Stage 1 - Reproducible baseline

Goal:

- ensure ingestion, split generation, preprocessing, featurization, and evaluation are stable and leakage-safe on Costa.

Deliverables:

- locked Costa data path,
- reproducible split artifacts,
- baseline metrics scripts for detection and classification,
- comparison-ready experiment logging.

### Stage 2 - Strong tabular baselines

Priority models:

- LightGBM
- XGBoost
- CatBoost
- ExtraTrees

Questions answered:

- how far the current pipeline gets on detection and classification before deep learning,
- whether feature engineering already beats naive baselines,
- and which models are strongest under class imbalance.

### Stage 3 - Feature and window ablations

Ablation axes:

- EDA-guided pre-drop and train-only pruning hardening,
- handcrafted physical feature families,
- handcrafted statistics (rolling -> windows -> multi-scale windows),
- tsfresh automated statistics as a comparator branch,
- temporal context as optional (Path B-oriented) rather than canonical Path A default,
- signal-processing / spectral additions (Path B only),
- window size / step size,
- calibration.

This stage is critical because it turns the thesis from a benchmarking exercise into an explanatory study.

Stage 3 now runs in two explicit tracks:

- Track 1: handcrafted hardening (primary)
- Track 2: automatic representation-learning comparators (secondary)

For detection specifically, this stage must also examine:

- threshold selection,
- calibration,
- PR-AUC behavior under class imbalance,
- and the sensitivity of results to labeling and segment definitions.

### Stage 4 - Deep models

Priority order:

1. 1D CNN
2. TCN
3. GRU / LSTM

Deep models should not be introduced earlier unless the tabular phase has already produced a trustworthy benchmark.

### Stage 5 - Deployment-driven model selection

Choose the final candidate by balancing:

- Costa detection/classification performance,
- stability across repeated runs,
- calibration quality,
- model size,
- and edge latency constraints.

### Stage 6 - Horizontal expansion

Only after the Costa result is strong:

- evaluate on La Reunion,
- evaluate on Mendeley transfer setups,
- test domain adaptation,
- and analyze cross-climate / cross-plant behavior.

---

## 6. Go / No-Go Gates

### G1 - Costa pipeline readiness

Pass when:

- data ingestion is stable,
- split logic is frozen,
- leakage checks pass,
- baseline metrics are reproducible.

### G2 - Costa tabular credibility

Pass when:

- at least one strong and stable tabular result exists for detection,
- at least one strong and stable tabular result exists for classification,
- performance is competitive with the literature,
- and ablations are technically trustworthy.

### G3 - Costa benchmark success

Pass when at least one model:

- matches or exceeds the paper on detection and/or classification with clear rigor,
- or falls slightly short but is better justified through stronger metrics and evaluation rigor.

### G4 - Deployable winner chosen

Pass when one candidate is selected for edge deployment based on both performance and runtime practicality.

### G5 - Horizontal research unlocked

Pass when Costa results are thesis-worthy and the deployable path is credible.

Only then should transfer, DANN, and broader generalization become active priorities.

---

## 7. Method Priority on Costa

### High priority now

- leakage-safe splits
- anomaly-detection baselines
- boosted trees
- feature-family ablations
- window-size tuning
- calibration / thresholding
- 1D CNN / TCN

### Medium priority after vertical strength

- wavelet features
- learned representations
- residual-based anomaly analysis

### Low priority for the first phase

- advanced interpolation methods
- synthetic oversampling with heavy generative models
- domain adaptation before Costa is strong
- state-space models as the immediate final target

---

## 8. Exit Criteria Before Moving Beyond Costa

Do not expand the thesis scope until most of the following are true:

- Costa benchmark is well understood,
- at least one result is clearly thesis-worthy,
- the final deployment candidate is identified,
- ablations support a coherent story,
- and the experiment stack is reproducible enough to support transfer studies.

Costa is the proving ground. Everything else is built on top of it.
