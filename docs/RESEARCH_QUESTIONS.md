# Research Questions and Hypotheses

**Author:** Ahmed Amine GUERRAICHE  
**Date:** April 2026  
**Status:** Active research framing

---

## 1. Purpose

This document defines the research questions that govern the thesis. They are ordered by execution priority.

The rule is simple:

- first answer the **vertical Costa questions**,
- then move to **generalization and adaptation questions**,
- and keep **deployment** as a mandatory validation track throughout.

---

## 2. Primary Research Questions

### RQ1 - Costa benchmark strength

Which leakage-safe pipeline for photovoltaic fault detection and fault classification performs best on the Costa dataset?

### RQ2 - Published baseline comparison

Can the proposed Costa-centered pipeline match or exceed the published Costa benchmarks:

- **93.09%** fault detection accuracy,
- **95.44%** fault classification accuracy?

### RQ3 - Feature contribution

Which feature families contribute most to Costa performance:

- physics-informed features,
- temporal features,
- statistical / tsfresh features,
- signal-processing features,
- or learned representations?

### RQ4 - Model family tradeoff

Which model family provides the best balance between predictive performance and deployment suitability:

- boosted trees,
- shallow neural networks,
- 1D CNN / TCN,
- or recurrent sequence models?

### RQ5 - Robustness of conclusions

Are observed improvements robust under leakage-safe splits, statistical comparison, and calibration analysis rather than artifacts of randomness or evaluation bias?

---

## 3. Secondary Research Questions

### RQ6 - Cross-dataset transfer

How well do modeling choices validated on Costa transfer to La Reunion and Mendeley?

### RQ7 - Domain shift characterization

What types of dataset shift most strongly degrade cross-dataset performance:

- climate,
- sensor set,
- sampling rate,
- or simulated-versus-real acquisition?

### RQ8 - Adaptation benefit

Can domain adaptation techniques such as DANN or distribution alignment reduce the generalization gap across datasets?

### RQ9 - Residual-based anomaly signal

When forecasting is used only as normal-behavior modeling, does residual-based anomaly analysis provide useful complementary signal beyond the main detection pipeline?

---

## 4. Working Hypotheses

### H1

A leakage-safe Costa pipeline with strong feature engineering and boosted-tree baselines should reach or exceed published benchmark performance before deep models become necessary.

### H2

On Costa, data handling choices - especially split rigor, feature design, and windowing - will affect final performance more than model complexity in the early phase.

### H3

Among deep models, TCN and 1D CNN architectures will provide the strongest performance-to-deployment tradeoff.

### H4

Cross-dataset performance will drop substantially without explicit adaptation, especially for simulated-to-real transfer.

### H5

Residual-based anomaly analysis may be useful as a complementary method, but it will not replace the core detection/classification pipeline on current public datasets.

---

## 5. Null Results That Are Still Valuable

The thesis remains strong even if some advanced directions fail. The following outcomes are scientifically useful:

- deep learning does not outperform well-tuned boosted trees on Costa,
- domain adaptation helps only marginally,
- residual forecasting adds little beyond direct anomaly detection,
- or deployment constraints force selection of a simpler model.

These are still publishable-quality findings if they are supported by careful evaluation.

---

## 6. Decision Order

The research questions should be answered in this order:

1. `RQ1-RQ5` on Costa,
2. deployment-oriented model selection,
3. `RQ6-RQ8` on transfer and adaptation,
4. `RQ9` only if time and bandwidth remain.
