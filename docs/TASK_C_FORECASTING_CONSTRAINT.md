# Task C: Fault Forecasting vs. Performance Forecasting

**Analysis Date:** April 2026  
**Status:** Constraint accepted - forecasting demoted to optional residual analysis  
**Relevance:** Explains why forecasting is not a primary thesis task

---

## Executive Summary

**Finding:** Fault forecasting (predicting fault onset before it occurs) is theoretically impossible on all available publicly labeled PV fault datasets because **all faults are artificially induced** rather than naturally occurring through degradation processes.

**Implication:** direct fault forecasting is not retained as a primary thesis task. Forecasting is kept only as an optional **performance-forecasting / residual-based anomaly analysis** method.

**Thesis Impact:** This is a strength, not a weakness. It sharpens the project around two primary tasks - fault detection and fault classification - while preserving forecasting only where it is scientifically defensible.

**Project note:** forecasting was initially considered seriously because a commercial partner indicated that long-term annotated operational data might become available. That dataset did not arrive, so the thesis scope now follows the realities of the public data.

---

## The Constraint: All Faults Are Artificially Induced

### Costa PV Fault Dataset

**Source:** Costa et al., "A Monitoring System for Online Fault Detection and Classification in Photovoltaic Plants," *Sensors* 20(17), 4688 (2020). [PMC Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC7506914/)

**Methodology for fault creation:**

| Fault Type | Creation Method | Duration | Onset |
|---|---|---|---|
| **Short-circuit** | Physical sockets connecting module terminals | 10 minutes | Instant step function |
| **Open-circuit** | Auxiliary circuit breakers disconnecting strings | 10 minutes | Instant step function |
| **Degradation** | Resistors inserted in series with strings | 10 minutes | Instant step function |
| **Shadowing** | "Physically blocking solar radiation using diverse opaque objects" | 10 minutes | Instant step function |

**Critical detail:** The researchers **explicitly excluded cloudy and rainy days** from the 16-day test period to avoid confusing natural shading with experimental shading. This reveals the binary nature: either 100% controlled fault or 100% natural conditions.

**Pre-fault signal:** None. Faults appear as step functions at predetermined moments.

---

### La Reunion Dataset

**Source:** Kbidi, F. et al., "Electrical production data of a domestic grid-connected rooftop PV plant in normal and shading faults conditions associated with solar and meteorological data in a tropical climate," *Data in Brief* 46, 108723 (2023). [HAL](https://laas.hal.science/ENERGY-LAB/hal-04126493v1)

**Methodology:**

Described explicitly as **"experimentally induced faults"** in associated diagnostic papers (Lebreton et al., *Entropy* 2022, 24(9), 1311).

**Implementation:**
- Shading created by placing opaque materials on specific PV modules
- Fault classes are experimental configurations:
  - Class 0: No fault (normal operation)
  - Class 1.0: Uniform shading (1 module fully shaded)
  - Class 2.1-2.3: Constant partial shading on 1/2/3 specific modules
  - Class 3.1-3.2: Constant partial shading on 1/3 or 2/3 of a specific module
  - Class 4.0: Intermittent partial shading (static, not weather-driven)

**Key insight:** Even though La Reunion is an outdoor installation in a tropical climate with real weather, the **fault labels correspond to experimentally applied configurations**, not natural cloud transients. The research team manually placed and removed opaque objects to generate each labeled fault condition.

**Pre-fault signal:** None in the electrical channels. The meteorological data (GTI, DTI) may show gradual changes as clouds naturally pass, but the **labeled fault events correspond to manually applied shading**, not cloud approach.

---

### GPVS-Faults Dataset

**Source:** Jovicic, A. et al., GPVS-Faults: Experimental Data for fault scenarios in grid-connected PV systems under MPPT and IPPT modes. *Mendeley Data* (2020). [Dataset](https://data.mendeley.com/datasets/n76t439f65/1)

**Methodology:** Paper explicitly states **"faults were introduced manually halfway during the experiments."**

Lab setting. No pre-fault signal.

---

### Mendeley Simulated Dataset

**Type:** MATLAB/Simulink simulation at 10 kHz. By definition, no natural degradation process.

---

## Why This Breaks Task C (Original Formulation)

The original Task C objective was:
> "Given a window of N minutes of historical data, predict: will a fault occur in the next 5/10/30 minutes?"

### The Problem

For artificially induced faults:
- The fault is **not caused by any observable precursor in the electrical or meteorological signals**
- It is caused by **external human action** (placing an object, inserting a resistor, flipping a breaker)
- The electrical system has **zero information** about when this action will occur
- Any model trained to "predict" the fault would be learning noise, not physics

**Analogy:** Imagine predicting when a light switch will be flipped based on the light's behavior. The light doesn't "know" the switch will be flipped. Neither does the PV system know an opaque object will be placed on it.

### Why "Pre-Fault Signal" Doesn't Exist

Even if we observe the fault in real-time (the moment opaque material touches a module), there is no **temporal lag** or **transition period** to exploit:

```
t < T:  Normal operation (all signals nominal)
t = T:  Opaque object placed (instant step function)
t > T:  Faulty operation (power drops immediately)

Interval [T-5min, T): Signal is identical to [T-30min, T-5min)
No information distinguishes "fault in 5 minutes" from "normal forever"
```

---

## Legitimate Task C: Performance Forecasting

Rather than predicting faults, predict **expected electrical output** under current meteorological conditions. This is:
- **Solvable** — the relationship between irradiance/temperature and power is physical and observable
- **Operationally useful** — deviations between predicted and actual power trigger alerts
- **Well-supported by datasets** — SKIPP'D (Stanford), HK HKUST, DKASC, NIST, Costa/La Reunion normal periods
- **Established in literature** — solar power forecasting is a mature field with published benchmarks

### Architecture

```
Input:  [X_{t-W}, ..., X_t]  (electrical + meteorological history)
Model:  Learns P_normal = f(irr, temp, time_of_day, ...)
Output: P_predicted_{t+h}    (for horizon h ∈ {5, 10, 30} minutes)

On faulty periods:
residual(t) = |P_actual(t) - P_predicted(t)|

If residual exceeds threshold → anomaly alarm
```

**Why this works even with artificially induced faults:**
- The model learns "normal" physics from normal-operation data
- When a fault occurs (at any moment), actual power deviates from predicted
- The residual captures the fault signature **in hindsight**
- This enables post-hoc evaluation: "If we deployed this forecasting model, would it detect the fault?"

---

## Datasets Supporting Legitimate Task C

### Primary (With Fault Labels for Evaluation)

| Dataset | Role | Horizons | Resolution |
|---|---|---|---|
| **Costa** | Train on label-0, evaluate residuals on labels 1-4 | 5, 10 min | 1 Hz |
| **La Reunion** | Train on normal periods, evaluate on labeled fault periods | 5, 10, 30 min | ~0.14 Hz (7s) |

### Secondary (Benchmarking / Multi-Site Validation)

| Dataset | Role | Horizons | Resolution |
|---|---|---|---|
| **SKIPP'D (Stanford)** | Establish baseline power forecasting performance | 5, 15, 30, 60 min | 1 min |
| **HK HKUST** | Multi-site validation (60 rooftop stations, 3 years) | 5, 15, 30 min | 5 min |
| **DKASC** | Long-term climate diversity (15 years, arid climate) | 15, 30, 60 min | Sub-hourly |
| **NIST** | Highest resolution comparison (1-second data) | 5, 10 min | 1 sec / 1 min avg |

---

## Experimental Design: Task C (Reformulated)

### Experiment C1: Performance Forecasting Baseline
**Goal:** Establish what a state-of-the-art power forecasting model can achieve.

```
Data:      Costa label-0 (normal operation only) + La Reunion normal periods
Models:    Persistence → XGBoost-lag → LSTM → TCN → (optional) Foundation model
Metrics:   MAE, RMSE, MAPE on held-out normal test data
Horizons:  5, 10, 30 minutes
Baseline:  Persistence (assume future = present)
Comparison: SKIPP'D published benchmarks (if available)
```

### Experiment C2: Residual-Based Fault Detection
**Goal:** Prove that performance forecasting enables anomaly detection via residual monitoring.

```
Use best model from C1 (trained on normal data)
Evaluate on Costa fault periods and La Reunion labeled fault periods

For each fault:
  residual(t) = |P_actual(t) - P_predicted(t)|
  
Metrics on fault detection:
  - PR-AUC (primary: imbalanced fault class)
  - Latency: how many minutes after fault onset before residual exceeds threshold
  - F1-score at multiple thresholds
  
Compare to Task A (pure anomaly detection)
Question: "Does forecasting provide added value beyond statistical anomaly detection?"
```

### Experiment C3: Cross-Site Generalization (If Time Permits)
**Goal:** Does a model trained on one site detect faults at another?

```
Train on La Reunion normal data
Test on Costa normal periods (check baseline accuracy)
Test on Costa fault periods (check detection latency)

Train on Costa normal data
Test on La Reunion normal periods
Test on La Reunion fault periods

Question: "Can forecasting models transfer across 
climates (tropical vs. temperate) and hardware?"
```

---

## Thesis Narrative

### What to Say

In your thesis, include a section like:

> **Task C: Performance Forecasting for Anomaly Detection**
>
> While operational PV systems require fault prediction (anticipating problems before they occur), this capability depends on datasets where faults emerge through natural degradation with observable precursor signatures. All publicly available fault-labeled PV electrical time-series datasets (Costa et al., 2020; Lebreton et al., 2022) employ artificially induced faults introduced instantaneously for research purposes. These datasets do not contain the temporal dynamics needed to train predictive models of fault onset.
>
> We therefore reformulate Task C as **performance forecasting**: predicting the electrical output a PV system should produce given current meteorological conditions. When actual output deviates from predicted values, this residual serves as a complementary anomaly detection signal. This approach is:
> 1. **Theoretically sound:** It leverages observable physics (irradiance → power)
> 2. **Operationally meaningful:** It enables real-time monitoring via expected-vs-actual comparison
> 3. **Demonstrably useful:** It will be evaluated against Task A to measure incremental value
>
> This reformulation reflects a broader research insight: the distinction between what can be predicted *from data* (performance under known conditions) versus what cannot (externally imposed events). The honesty about this distinction strengthens the research contribution.

### Why This Is Honest & Valuable

- You're not claiming to do something impossible
- You're identifying and solving a **realistic** problem (performance forecasting is how operators actually detect faults in production)
- You're demonstrating **research maturity** by constraints-aware problem formulation
- The jury will respect this clarity

---

## References

1. Costa, C.H. et al. "A Monitoring System for Online Fault Detection and Classification in Photovoltaic Plants." *Sensors* 20(17), 4688 (2020). DOI: [10.3390/s20174688](https://doi.org/10.3390/s20174688)

2. Kbidi, F. et al. "Electrical production data of a domestic grid-connected rooftop PV plant in normal and shading faults conditions associated with solar and meteorological data in a tropical climate." *Data in Brief* 46, 108723 (2023). DOI: [10.1016/j.dib.2022.108723](https://doi.org/10.1016/j.dib.2022.108723)

3. Lebreton, C.; Kbidi, F.; Graillet, A.; Jegado, T.; Alicalapa, F.; Benne, M.; Damour, C. "PV System Failures Diagnosis Based on Multiscale Dispersion Entropy." *Entropy* 2022, 24, 1311. DOI: [10.3390/e24101311](https://doi.org/10.3390/e24101311)

4. Jovicic, A. et al. "GPVS-Faults: Experimental Data for fault scenarios in grid-connected PV systems under MPPT and IPPT modes." *Mendeley Data* (2020). DOI: [10.17632/n76t439f65.1](https://doi.org/10.17632/n76t439f65.1)

5. Nie, Y. et al. "SKIPP'D: a SKy Images and Photovoltaic Power Generation Dataset for Short-term Solar Forecasting." *Solar Energy* (2023). DOI: [10.1016/j.solener.2023.03.043](https://doi.org/10.1016/j.solener.2023.03.043)

---

## Decision Checklist for Task C

- [ ] **Accept the constraint** — fault forecasting is not possible on these datasets
- [ ] **Adopt the reformulation** — pivot to performance forecasting
- [ ] **Design Experiments C1-C3** — following the experimental design above
- [ ] **Update TECHNICAL_DESIGN.md** — reflect the reformulation in system description
- [ ] **Document in thesis** — include the constraint analysis in thesis Chapter 3 (Data & Methods)
- [ ] **Assign to teammate** — if splitting work, make Task C + Task A form one person's domain (they're now tightly coupled via residuals)

---

## Alternative: Drop Task C Entirely

If performance forecasting doesn't align with your research interests:

**Option:** Invest the effort instead in **deeper exploration of Tasks A and B**:
- Multi-site anomaly detection (train on La Reunion, test on Costa; vice versa)
- Fine-grained transfer learning analysis across climates and sampling rates
- Sim-to-real adaptation comparison (Mendeley → Costa → La Reunion)
- Ensemble methods combining multiple anomaly detection approaches

A system with two exceptional tasks is stronger than one with three tasks where one is compromised.

**Decision needed by:** Before detailed experimental design begins.
