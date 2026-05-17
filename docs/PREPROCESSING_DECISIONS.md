# Preprocessing Decisions — PV Fault Detection

**Author:** Ahmed Amine GUERRAICHE  
**Date:** April 2, 2026 — Updated April 23, 2026  
**Status:** Complete

---

## Executive Summary

This document records the **data-driven preprocessing decisions** for the PV fault detection system. All decisions are grounded in EDA findings and are designed to define the minimal leakage-safe cleaning layer before feature engineering.

**Key principle**: The model should learn to detect **deviations from expected behavior**, not absolute thresholds. Preprocessing should preserve this signal while staying as lean as possible.

Important current Costa scope:

- Costa preprocessing is now intentionally minimal.
- Missing-value handling is disabled for Costa because the retained post-ingestion dataset has no missing-value problem that justifies runtime overhead.
- Irradiance normalization and related shift-aware transforms are no longer preprocessing; they are feature-engineering choices.
- Therefore the active Costa preprocessing layer is currently just outlier handling.
- Costa preprocessing now operates on primary measured channels only (`vdc1`, `vdc2`, `idc1`, `idc2`, `irr`, `pvt`).
- Derived power channels (`pdc1`, `pdc2`, `pdc`) are restored after preprocessing from cleaned primaries: `pdc1=vdc1*idc1`, `pdc2=vdc2*idc2`, `pdc=pdc1+pdc2`.

---

## 1. Missing Value Strategy

### 1.1 EDA Findings

| Data Type | Total Null Episodes | Median Duration | Distribution |
|-----------|---------------------|-----------------|--------------|
| **Meteorological** (post Nov-15) | 137 | 0.5 min | 133 < 1min, 4 in 1-60min, 0 > 60min |
| **Electrical** | 10 | 0.0 min | All < 1min |

**Critical finding**: Some null episodes overlap with fault labels.

| Category | Safe to Impute | Must DROP (fault overlap) |
|----------|----------------|---------------------------|
| Meteorological | 120 episodes | 17 episodes |
| Electrical | 5 episodes | 5 episodes |

### 1.2 Why We Can't Impute on Fault Data

Imputing missing values during fault periods would:
1. **Fabricate readings** where real anomalies existed
2. **Erase fault signatures** — the very signal we're trying to detect
3. **Create false patterns** that could confuse the model

**The only honest option:** Drop rows with missing values during fault periods.

### 1.3 Decision: Tiered Imputation Strategy

| Gap Duration | Strategy | Scope | Rationale |
|--------------|----------|-------|-----------|
| **< 1 minute** | Forward-fill | Normal data only | Short gaps are sensor hiccups; last value is reasonable proxy |
| **1–5 minutes** | Linear interpolation | Normal data only | Moderate gaps; interpolation won't fabricate unrealistic dynamics |
| **> 5 minutes** | DROP rows | All data | Too long to interpolate safely without creating fake patterns |
| **Any gap** | DROP rows | Fault data | Never impute on fault periods |

**Alternatives considered:**

| Approach | Why Rejected |
|----------|--------------|
| Interpolation for all gaps | Fails for long gaps (33-day straight line is meaningless) |
| Forward-fill for all | "500W at 3pm" repeated at midnight is absurd |
| Mean/median imputation | Creates fake "normal" data that could mask patterns |
| KNN imputation | Expensive; risk of test data leakage |
| Seasonal imputation | Overkill for short gaps; parameters hard to tune |

### 1.4 Implementation Notes

```python
def handle_missing_values(df, config, segment_col='segment_id', label_col='Fault'):
    """
    Process missing values within each segment independently.
    
    Rules:
    1. Identify null episodes (contiguous NaN blocks)
    2. For each episode:
       - If overlaps fault label → DROP
       - If < 1 min → forward-fill
       - If 1-5 min → linear interpolate
       - If > 5 min → DROP
    3. Never interpolate across segment boundaries
    """
```

Current Costa implementation note:

- this strategy remains documented for datasets that actually require it,
- but Costa sets missing-value handling to disabled in config and skips the imputation stage entirely.

---

## 2. Outlier Treatment

### 2.1 EDA Findings

- Electrical measurements show extreme values during fault periods
- These extremes are **fault signatures**, not sensor errors
- Outliers in normal data are likely sensor errors or transient anomalies

### 2.2 Decision: Scope-Limited Outlier Clipping

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Method** | IQR × 3 (far outliers) | Conservative; only catches truly extreme values |
| **Scope** | Normal data only | Fault data outliers are expected signatures |
| **Action** | Clip/winsorize to bounds | Preserves row; caps extreme sensor errors |

**Why IQR × 3 (not 1.5)?**

- IQR × 1.5 catches "mild outliers" — may clip legitimate variability
- IQR × 3 catches "far outliers" — truly extreme sensor errors
- Reduces risk of removing real signal

**Alternatives considered:**

| Approach | Why Rejected |
|----------|--------------|
| Z-score (3σ) | Sensitive to existing outliers (mean/std get pulled) |
| Isolation Forest | Introduces model dependency before modeling; complex |
| Domain thresholds only | Requires extensive domain knowledge for each feature |
| No treatment | Sensor errors could create fake fault patterns |

### 2.3 Implementation Notes

```python
def clip_outliers_iqr(df, feature_cols, label_col='Fault', multiplier=3.0):
    """
    Clip outliers using IQR method on normal data only.
    
    Steps:
    1. Compute Q1, Q3, IQR on normal data (label == 0)
    2. Set bounds: lower = Q1 - 3*IQR, upper = Q3 + 3*IQR
    3. Clip normal data to bounds
    4. Leave fault data untouched
    """
    normal_mask = df[label_col] == 0
    
    for col in feature_cols:
        Q1, Q3 = df.loc[normal_mask, col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
        
        # Clip only normal data
        df.loc[normal_mask, col] = df.loc[normal_mask, col].clip(lower, upper)
    
    return df
```

Costa invariant note:

- Because `pdc1`, `pdc2`, and `pdc` are deterministic functions of primary channels at ingestion, they are not winsorized independently in preprocessing.
- Independent clipping of derived channels can break physical identities and create redundant distortion.
- The pipeline therefore clips primary channels only and then recomputes power channels to preserve `pdc = pdc1 + pdc2` exactly in preprocessed artifacts.

---

## 3. Physics Normalization

Status note:

- the reasoning in this section is still useful,
- but for Costa these transforms are no longer executed in preprocessing,
- they now belong to feature engineering, where they are interpreted as explicit handcrafted features for normal-manifold distribution-shift awareness rather than preprocessing fixes for stationarity.

### 3.1 EDA Findings: Why "Stationarity Correction" is the Wrong Frame

ADF/KPSS tests flag all power, current, temperature, and irradiance features as non-stationary. However:

- **Stationarity is a forecasting requirement**, not a classification or anomaly detection requirement
- Tree-based models (LightGBM) are distribution-free — they do not assume stationarity
- The real concern for anomaly detection is **normal-manifold time-variation**: if the distribution of healthy operation shifts across the day, detectors trained on morning data will raise false alarms in the afternoon

The correct question is not "is this feature stationary?" but "does this feature's diurnal drift cause normal-manifold shift that the model cannot absorb through conditioning features?"

Important scope update for the current Costa vertical benchmark:

- the full physics-normalization discussion below records the rationale for irradiance-conditioned representations,
- but these transforms are now treated as feature-engineering choices rather than preprocessing steps for Costa,
- so the active Costa preprocessing layer is limited to outlier handling, with missing-value handling disabled because Costa has no missing-value problem after ingestion.

### 3.2 Feature-Level Decision

| Feature | Diurnal swing | Root driver | Transform | Output |
|---------|--------------|-------------|-----------|--------|
| pdc, pdc1, pdc2 | 0 → 4000 W | Irradiance (ρ=0.93) | ÷ irr | `pdc_norm`, `pdc1_norm`, `pdc2_norm` |
| idc1, idc2 | 0 → 8 A | Irradiance (ρ=0.94) | ÷ irr | `idc1_norm`, `idc2_norm` |
| Pg, Ig, Eg, Fg, Ia | Diurnal | GTI (ρ≈0.95) | ÷ GTI | `Pg_norm`, `Ig_norm`, etc. |
| pvt | 23 → 55 °C | Irradiance-driven thermal loading (ρ=0.70) | OLS residual on irr | `pvt_irr_residual` |
| TA, TPV | Thermal drift | GTI-driven | OLS residual on GTI | `TA_irr_residual`, `TPV_irr_residual` |
| vdc1, vdc2, Vg | ±16 V on 272 V | Weak | None — drift small vs fault signature | Keep raw |
| irr / GTI | Solar arc | Solar geometry | None — used as conditioning variable | Keep raw |

### 3.3 Irradiance Normalization of Power and Current

`feature / irr ≈ efficiency or yield coefficient` — physically meaningful, approximately constant under healthy operation, removes the irradiance-driven diurnal swing from power and current channels.

**Why include idc1/idc2:** Current has the same irradiance coupling as power (ρ=0.94). Normalizing only power but leaving raw current in the feature set is inconsistent — the same diurnal manifold shift applies.

**Safety:** denominator floored at 5 W/m² to avoid near-zero division at dawn/dusk edges.

### 3.4 Irradiance-Conditioned Residualization of Temperature

**Why not polynomial detrending:** Empirically confirmed insufficient. Costa EDA stationarity tests on `pvt_detrend` (polynomial degree=1): ADF p=0.293, KPSS p=0.01 — still non-stationary post-detrend. Root cause: temperature is non-stationary because it co-varies with irradiance (ρ=0.70), not because of a polynomial time-index trend. Detrending time removes the wrong driver.

**Decision:** Fit `feature = β·irr + α` via OLS on normal-class train rows only. Use residual `feature − (β·irr + α)` as the physics-normalized temperature feature. This:
- Removes the irradiance-driven thermal loading at source
- Preserves temperature deviations from the physics-predicted baseline (fault-caused thermal anomalies survive as non-zero residuals)
- Is leakage-safe: β and α fitted on train normal rows, applied to val/test using stored params

**Why temperature carries fault signal:** Costa EDA Mann-Whitney: normal pvt mean=40.0°C vs fault mean=33.7°C (rank-biserial=−0.33, p<0.001). Aggressive detrending that removes this level difference would erase the fault signature. The residualization preserves it because fault-caused temperature depression is not explained by irradiance.

### 3.5 Alternatives Rejected

| Approach | Why Rejected |
|----------|--------------|
| Polynomial detrend (temp) | Empirically insufficient (ADF p=0.29 post-detrend); wrong driver (time not irradiance) |
| STL decomposition | Segments too short (< 2 seasonal cycles); not viable |
| First-order differencing | Destroys sustained fault-level shifts (10-min fault blocks become only onset/offset spikes) |
| Clear-sky index (irr/irr_clearsky) | No clear-sky model available; dataset collected on clear-sky days so CSI≡1 anyway |
| Day-level z-normalization | Requires full-day look-ahead — inference-inadmissible |
| Correcting all features | ADF flags everything; transforms on vdc/voltage destroy signal with negligible manifold benefit |

---

## 4. Preprocessing Pipeline Design

### 4.1 Step Sequence

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: LOAD SPLIT DATA                                            │
│  Input: data/interim/splits/{task}/train.parquet                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: HANDLE MISSING VALUES                                      │
│  • Dataset-dependent                                                 │
│  • Disabled for Costa                                                │
│  • Retained only where missingness is a real problem                │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: OUTLIER TREATMENT                                          │
│  • Compute IQR bounds on normal data                                │
│  • Clip normal data to bounds (fault data untouched)                │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: PHYSICS NORMALIZATION / SHIFT-AWARE TRANSFORMS             │
│  • No longer part of Costa preprocessing                             │
│  • Executed later in feature engineering where applicable            │
└───────────────────────────────┬─────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: SAVE & MANIFEST                                            │
│  Output: data/processed/preprocessed/{task}/train.parquet           │
│  Manifest: preprocess_manifest.json (stats, rows dropped, etc.)     │
└─────────────────────────────────────────────────────────────────────┘
```

Current Costa pipeline interpretation:

- preprocessing output = minimal cleaned signal layer,
- feature engineering output = handcrafted shift-aware representations,
- spectral and continuity-dependent transforms are decided downstream under Path A / Path B rules.

### 4.2 Output Directory Structure

```
data/processed/preprocessed/<dataset>/
├── anomaly_semisup/
│   ├── train.parquet
│   ├── val.parquet
│   ├── test.parquet
│   └── preprocess_manifest.json
├── anomaly_supervised/
│   └── ...
└── classification/
    └── ...
```

### 4.3 Manifest Contents

```json
{
  "version": 1,
  "created_at": "2026-04-02T...",
  "config_used": { ... },
  "statistics": {
    "input_rows": 2012710,
    "output_rows": 2010500,
    "rows_dropped": {
      "missing_fault_overlap": 150,
      "missing_long_gap": 2060,
      "total": 2210
    },
    "outliers_clipped": {
      "Pg": 1234,
      "Ig": 567,
      ...
    },
    "features_created": ["pdc_norm", "pdc1_norm", "pdc2_norm", "idc1_norm", "idc2_norm", "pvt_irr_residual"],
  "irr_residual_params": {"pvt": [0.032, 15.4]}
}
```

---

## 5. Configuration

### 5.1 Config Extension (`configs/data_config.yaml`)

```yaml
preprocessing:
  missing_values:
    ffill_max_gap_seconds: 60        # < 1 min → forward-fill
    interp_max_gap_seconds: 300      # 1-5 min → linear interpolation
    # > 5 min → drop
  
  outliers:
    method: iqr
    iqr_multiplier: 3.0              # 3× IQR (far outliers only)
    action: clip                      # winsorize to bounds
    scope: normal_only               # don't touch fault data
  
  physics_normalization:
    irradiance_normalize:
      features: [pdc, pdc1, pdc2, idc1, idc2]   # Costa; La Réunion uses [Pg, Ig, Eg, Fg, Ia]
      denominator: irr                            # Costa; La Réunion uses GTI
      suffix: _norm
    irr_residualize:
      features: [pvt]                             # Costa; La Réunion uses [TA, TPV]
      irr_col: irr                                # Costa; La Réunion uses GTI
      suffix: _irr_residual
```

---

## 6. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-02 | Tiered missing value strategy | Balances data preservation with honesty about gaps |
| 2026-04-02 | DROP any nulls on fault data | Cannot fabricate fault readings |
| 2026-04-02 | IQR 3× for outliers | Conservative; only catches extreme sensor errors |
| 2026-04-02 | Outliers on normal data only | Fault outliers are expected signatures |
| 2026-04-02 | STL not applicable | Segments too short (< 2 seasonal cycles) |
| 2026-04-02 | Irradiance normalization for power | Physics-grounded; removes irradiance-driven manifold shift |
| 2026-04-23 | Add idc1/idc2 to irr normalization | Current has same irradiance coupling (ρ=0.94) as power; omission was inconsistent |
| 2026-04-23 | Drop polynomial detrend for temperature | Empirically insufficient (ADF p=0.29 post-detrend); wrong driver (time vs irradiance) |
| 2026-04-23 | Irr-conditioned OLS residualization for temp | Removes irradiance-driven thermal loading at source; preserves fault-caused thermal anomalies |
| 2026-04-23 | Rename "stationarity correction" → "physics normalization" | Goal is manifold stability for anomaly detection, not strict stationarity; stationarity is a forecasting assumption irrelevant to FDD |

---

## 7. Honest Reporting Requirements

### In Thesis/Paper

1. **Document imputation limits:**
   > "Missing values on fault-labeled data were dropped rather than imputed to avoid fabricating anomaly signatures."

2. **Acknowledge STL inapplicability:**
   > "STL decomposition was considered but not applied due to segment lengths (median 11h) being shorter than the required 2 seasonal cycles."

3. **Justify normalization choice:**
   > "Irradiance normalization (P/GTI) was used as the primary stationarity correction, transforming absolute power into instantaneous efficiency which is theoretically constant under normal operation."

---

## References

- EDA analysis: `notebooks/eda_reunion.ipynb` (Sections 3, 9, 10)
- Split decisions: `SPLIT_DECISIONS.md`
- Config: `configs/data_config.yaml`
- Session guide: `NEXT_SESSION_GUIDE.md`

---

*This document is the authoritative reference for preprocessing decisions.*
