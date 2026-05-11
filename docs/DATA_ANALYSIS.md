# Data Analysis — La Réunion PV Fault Detection Dataset

> **Source:** University of La Réunion — real operational PV installation  
> **Period:** October 2021 – April 2023 (~18 months)  
> **Timezone:** UTC+4 (La Réunion / RET)  
> **Total rows:** ~51 million across 3 datasets

---

## 1. Dataset Overview

| Dataset | File | Rows | Columns | Sampling | Start Date | Description |
|---------|------|------|---------|----------|------------|-------------|
| **DT1** | `dt1_solar_and_meteorological_measurement.csv` | 45,123,826 | 5 | 1 Hz (1s) | 2021-10-28 | Solar & meteorological sensors |
| **DT2** | `dt2_electrical_production_inverter_1_with_faults.csv` | 2,988,814 | 9 | 0.2 Hz (5s) | 2021-10-01 | **Faulty inverter** — fault labels included |
| **DT3** | `dt3_electrical_production_inverter_2.csv` | 2,988,814 | 8 | 0.2 Hz (5s) | 2021-10-01 | **Control inverter** — healthy, no faults produced |

**Key structural difference:** DT2 has a `Fault` column; DT3 does not. DT2 and DT3 share the same 8 electrical columns. DT1 provides environmental context.

---

## 2. DT1 — Meteorological Variables

| Variable | Description | Unit | Sensor |
|----------|-------------|------|--------|
| `GTI` | Global Tilted Irradiance | W/m² | SPN1 Delta-T Devices |
| `DTI` | Diffuse Tilted Irradiance | W/m² | SPN1 Delta-T Devices |
| `TA` | Ambient Temperature | °C | Thermocouple type K |
| `TPV` | PV Module Back-Surface Temperature | °C | Thermocouple type T |

- Sampled at **1 Hz** by a data logger
- Starts **27 days later** than DT2/DT3 (Oct 28 vs Oct 1) — first 27 days of electrical data have no meteorological counterpart

---

## 3. DT2 & DT3 — Electrical Variables

| Variable | Description | Unit |
|----------|-------------|------|
| `Eg` | Energy injected to the grid | kWh |
| `Pg` | Power injected to the grid | W |
| `Ia` | Current produced by PV plant | A |
| `Ig` | Current injected to the grid | A |
| `Va` | PV plant voltage | V |
| `Vg` | Grid voltage | V |
| `Fg` | Grid frequency | Hz |
| `Fault` | Fault type label (**DT2 only**) | category |

- Sampled at **0.2 Hz** (official spec, ~5s intervals; observed ~7s gaps in practice)
- Collected from two separate inverters via Energrid datalogger

---

## 4. DT2 vs DT3 — Key Differences

### 4.1 Structural
- **DT2 (Inverter 1):** The inverter on which **faults were deliberately caused** — has `Fault` column with ground-truth labels
- **DT3 (Inverter 2):** The **control inverter** — PV panels functioning normally, no faults produced (confirmed by original paper). No `Fault` column because no faults exist to label

### 4.2 Statistical Comparison (first 500k rows)

| Metric | DT2 Healthy (Fault=0) | DT2 Faulty (Fault≠0) | DT3 (all) |
|--------|----------------------|----------------------|-----------|
| Pg mean / std | 630.5 / 449.1 | 512.4 / 354.9 | 645.4 / 482.2 |
| Va mean / std | 182.1 / 9.7 | **148.5 / 38.9** | 175.6 / 19.0 |
| Ia mean / std | 3.7 / 2.8 | 3.7 / 2.8 | 3.7 / 2.8 |

**Observations:**
- DT3 statistics closely match DT2-healthy — consistent with its role as healthy reference
- DT2-faulty shows **degraded voltage** (Va: mean 148.5 vs 182.1) and **lower power** (Pg: mean 512.4 vs 630.5)
- Voltage (Va) is the most discriminative feature for fault detection

---

## 5. Fault Taxonomy (DT2)

All faults are **shading faults** deliberately induced on the PV panels:

| Fault Code | Description | Detail | Count | % |
|------------|-------------|--------|-------|---|
| **0.0** | No-fault condition | Normal operation | 2,904,289 | 97.17% |
| **1.0** | Uniform shading | Full uniform shading | 6,318 | 0.21% |
| **2.1** | Constant partial shading — entire module | 1 shaded PV module | 2,148 | 0.07% |
| **2.2** | Constant partial shading — entire module | 2 shaded PV modules | 2,460 | 0.08% |
| **2.3** | Constant partial shading — entire module | 3 shaded PV modules | 3,287 | 0.11% |
| **3.1** | Constant partial shading — portion of module | 1/3 shaded module | 2,717 | 0.09% |
| **3.2** | Constant partial shading — portion of module | 2/3 shaded module | 6,292 | 0.21% |
| **4.0** | Intermittent partial shading — static | | 61,297 | 2.05% |
| *null* | Missing labels | | 6 | ~0% |

### Critical Observations
- **Severe class imbalance:** 97.17% normal vs 2.83% faulty
- **Fault type 4** dominates faults (72% of all fault samples)
- **Hierarchical taxonomy:** fault codes 2.x and 3.x have sub-categories (severity levels)
- **All faults are shading-related** — not electrical failures, degradation, or hotspots

---

## 6. Temporal Synchronization (DT1 ↔ DT2)

### The Problem
DT1 (meteorological) samples at **1 Hz** and DT2/DT3 (electrical) at **0.2 Hz**. Their timestamps almost never align exactly. A regular join on `time` would drop nearly all rows.

### The Solution: `join_asof` (as-of join)
```python
merged = dt2.join_asof(dt1, on="time", strategy="nearest", tolerance="30s")
```

For each DT2 row, finds the DT1 row with the **closest timestamp** within a 30-second window.

### Why This Works
1. **Preserves all DT2 rows** — the fault-labeled electrical data drives the join; every labeled sample is kept
2. **`tolerance="30s"`** — if no meteorological reading exists within 30s, meteo columns become `null` (prevents stale data leaking in)
3. **`strategy="nearest"`** — uses closest reading in either direction. Since DT1 samples 5× faster, the nearest reading is typically 0–2 seconds away — physically negligible for irradiance/temperature
4. **DT2 drives the join** — 3M rows (DT2) joined against 45M rows (DT1) without blowing up the dataset

### Why Meteorological Data Matters
A PV panel's power output depends heavily on irradiance (GTI) and temperature (TA, TPV). Without these, a model cannot distinguish "low power because it's cloudy" from "low power because there's a shading fault." The meteo features provide the **expected operating conditions** that let the model detect anomalies.

### Gap: First 27 Days
DT1 starts on Oct 28, DT2/DT3 start on Oct 1. The first ~27 days of electrical data will have **null meteorological values** after the asof join. This needs handling downstream (drop or impute).

---

## 7. Pipeline Decision: Why Only DT2 + DT1?

The ingestion pipeline (`src/data/ingestion.py`) uses **only DT2 merged with DT1**:

| Decision | Rationale |
|----------|-----------|
| **Use DT2** | Only dataset with ground-truth fault labels → required for supervised learning |
| **Merge with DT1** | Environmental context (irradiance, temperature) needed to distinguish faults from normal weather variation |
| **Skip DT3** | No fault labels → cannot be used for supervised classification. Could serve as healthy reference for comparison-based or unsupervised methods in future work |

---

## 8. Preprocessing Considerations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| 97.17% class imbalance | Models biased toward predicting "normal" | SMOTE, class weights, focal loss, or undersampling |
| 27-day meteo gap | Null GTI/DTI/TA/TPV for Oct 1–27 | Drop these rows or use electrical-only features for that period |
| Irregular sampling (~5–7s gaps) | Not exactly 0.2 Hz | Resample to fixed interval or use time-aware models |
| 6 null Fault labels | Ambiguous samples | Drop (negligible: 6 out of 3M) |
| Hierarchical fault codes (2.x, 3.x) | Multi-class vs binary decision | Binary (fault/no-fault) for detection; multi-class for diagnosis |
| Shading-only faults | Scope limitation | Acknowledge in thesis — results generalize to shading faults, not all PV faults |
