# Session Findings — April 8, 2026

## Grill Session: Day/Night Boundaries & Windowing Constraints

**Status:** Findings documented, corrections pending in EDA notebook

---

## 0. Verification Query Results (Pre-EDA-Fix Analysis)

### 0.1 Null Pattern Analysis

**Meteo nulls breakdown (50,992 total):**
| Category | Count | Explanation |
|----------|-------|-------------|
| Before Oct 28 (DT1 start) | 41,777 | **Expected**: DT1 didn't exist yet |
| After Oct 28 | 9,215 | **True missing data** (MAR) |

**Oct 28 special case:** DT1 started recording at 16:37 local time, so 7,750 rows that day have null meteo.

**Post-Oct-28 null distribution:**
- Spread across 75 days (not concentrated)
- Uniform across hours (not time-of-day dependent)
- Pattern: **MAR (Missing At Random)** - equipment/sensor issues, not systematic

### 0.2 Intra-Day Gap Analysis

**Total intra-day gaps (>60s, same day): 72**

| Gap Duration | Count |
|--------------|-------|
| 1-5 min | 49 |
| 5-30 min | 22 |
| 30-60 min | 0 |
| 60+ min | 1 (63 min on 2022-03-01) |

**Large gaps (>5 min) - 23 total:**
- Most occur near day boundaries (6am, 17-18pm)
- All during normal operation (Fault=0)
- Likely equipment restarts, maintenance

**Conclusion:** True intra-day gaps are **rare and short** (mostly <5 min).

### 0.3 Recording Window Verification

**442 operating days** (not 465 segments!)

**Start hour distribution:**
| Hour | Days | % |
|------|------|---|
| 05:00 | 90 | 20.4% |
| 06:00 | 271 | **61.3%** |
| 07:00 | 62 | 14.0% |
| 08:00+ | 19 | 4.3% |

**End hour distribution:**
| Hour | Days | % |
|------|------|---|
| 16:xx | 27 | 6.1% |
| 17:xx | 220 | **49.8%** |
| 18:xx | 190 | 43.0% |

**Late-start days (9am+): 16 days**
- Some have very few rows (e.g., Feb 28 2022: only 78 rows starting 18:00)
- These are partial recording days (equipment issues, not weather)

### 0.4 Key Findings Summary

1. **Recording window is ~05:00-19:00 local** (not fixed, varies with sunrise/sunset + equipment)
2. **Typical day: 06:00-17:xx or 06:00-18:xx** (>90% of days)
3. **Meteo nulls before Oct 28: Expected** (DT1 hadn't started)
4. **Meteo nulls after Oct 28: True MAR** (sensor issues, 9,215 rows across 75 days)
5. **Intra-day gaps: Rare** (72 total, mostly <5 min)
6. **Late-start days exist: 16 days** start at 9am or later (partial recording)

---

## 1. Critical EDA Corrections Needed

### 1.1 Recording Pattern Clarification

**Previous understanding (INCORRECT):**
- "465 segments with 5+ min gaps" interpreted as data quality issues
- Gaps treated as missing data problems

**Corrected understanding:**
- **DT1 (Meteorological):** Records **24 hours/day continuously** (~1.88M rows per hour)
- **DT2 (Electrical):** Records **only during daytime** (~05:00-19:00 local time)
- The "465 segments" are actually **~442 operating days**, not data fragments
- Overnight gaps are **expected behavior**, not missing data

### 1.2 Verified Hour Distribution

**DT2 (Electrical) - Local Time (UTC+4):**
```
Hour 05:00:     8,621 rows   (edge - early sunrise)
Hour 06:00:   131,600 rows   (ramp up)
Hour 07:00:   249,806 rows   
...
Hour 15:00:   264,704 rows   (peak)
Hour 16:00:   263,603 rows   
Hour 17:00:   201,995 rows   (ramp down)
Hour 18:00:    38,872 rows   (edge - sunset)

Night (0-5h, 19-23h): 8,460 rows (0.3%)
Day (6-18h): 2,929,362 rows (99.7%)
```

**DT1 (Meteorological) - Uniform 24h:**
```
Every hour: ~1.88M rows (24/7 continuous recording)
```

### 1.3 Daily Recording Bounds

Recording start times vary based on actual sunrise/conditions:
- **05:00 start:** 90 days (20%)
- **06:00 start:** 271 days (62%) ← most common
- **07:00+ start:** 77 days (18%)

Recording end times vary based on actual sunset/conditions:
- **17:xx end:** 220 days (50%)
- **18:xx end:** 190 days (43%)
- **Earlier end:** 32 days (7%)

---

## 2. Decisions Made

### 2.1 Segment Definition
**Decision:** A segment = one solar operating day (sunrise to sunset)

- Overnight gaps (~12 hours) are by design, not missing data
- The GTI > 10 filter is redundant (electrical data already excludes night)
- No need for `astral` library — inverter naturally determines daytime

### 2.2 Time-of-Day Features
**Decision:** Add cyclic time encoding to all features

```python
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
```

**Rationale:** 
- Diurnal patterns are strong (readings at 8am ≠ readings at 2pm)
- Let the model learn time-of-day dependencies
- sin/cos captures cyclicity (hour 23 close to hour 0)

### 2.3 Windowing Constraints
**Decision:** Windows NEVER span overnight gaps

```python
# WINDOWING RULES

1. Each operating day (segment) is an isolated unit
2. Windows cannot cross segment boundaries
3. If segment has < window_size samples, skip entirely
```

**Rationale:**
- Physical discontinuity across 12h overnight gaps
- Rate-of-change features (dP/dt, dI/dt) meaningless across gaps
- Temporal models (LSTM, GRU, TCN) expect continuous sequences

### 2.4 Cold-Start Limitation
**Decision:** Accept that first `window_size` samples each day have no prediction capability

```python
# Per segment:
# - First window_size samples: NO Task C prediction possible
# - These samples CAN still be used for Task A (anomaly) and Task B (classification)
```

**Rationale:**
- This is an honest, defensible limitation
- ~7% of daily operating time is "unpredictable"
- Better than fabricating predictions with insufficient history

**Documentation for thesis:**
> "Fault prediction is not available for approximately the first 60 minutes after daily plant startup, due to insufficient historical context. This is an inherent limitation of sliding-window forecasting approaches."

---

## 3. Implications for Existing Decisions

### 3.1 Stationarity Analysis
**No change needed:** The finding that segments are ~11h (operating days) still supports the conclusion that STL decomposition is not applicable.

### 3.2 Missing Value Analysis
**Needs correction:** The analysis of "null episodes" and "gaps" needs to distinguish between:
1. **Overnight boundaries:** Expected, not data quality issues
2. **True intra-day gaps:** Actual missing data (equipment issues, etc.)

**TODO in EDA:** Re-run gap analysis filtering to only flag intra-day gaps (same date, gap > threshold).

### 3.3 Split Strategy
**Minor update:** The segment-stratified split is still correct, but documentation should clarify:
- "Segment" = "Operating day"
- Temporal ordering within segments preserved
- No windows cross overnight boundaries

### 3.4 Preprocessing Pipeline
**Update needed:** Add time-of-day features (hour_sin, hour_cos) in feature engineering step.

---

## 4. TODO: EDA Corrections

The following corrections should be applied in `notebooks/eda_reunion.ipynb`:

### 4.1 Gap Analysis Section
- [ ] Separate overnight gaps from intra-day gaps
- [ ] Report count of true intra-day gaps (> 60 seconds, same date)
- [ ] Document that overnight gaps are expected behavior

### 4.2 Segment Analysis Section  
- [ ] Clarify "segment" = "operating day"
- [ ] Report number of operating days (not "segments with gaps")
- [ ] Add visualization of daily recording windows

### 4.3 Add Time-of-Day Analysis
- [ ] Plot hour distribution (already have this data)
- [ ] Show recording start/end time distribution across days
- [ ] Visualize diurnal patterns in key features (Pg, GTI by hour)

### 4.4 Windowing Constraints Documentation
- [ ] Add section documenting windowing rules
- [ ] Explain cold-start limitation
- [ ] Update any sample counts that assumed continuous data

---

## 5. Updated Feature List

### Original Features
```yaml
electrical: [Ia, Ig, Eg, Fg, Pg, Va, Vg]
meteorological: [GTI, DTI, TA, TPV]
physics: [performance_ratio, delta_temp, dP_dt, dV_dt, dI_dt, Vg_normalized]
```

### New Time Features (ADD)
```yaml
temporal:
  - hour_sin    # sin(2π * hour / 24)
  - hour_cos    # cos(2π * hour / 24)
```

---

## 6. Summary of Session Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Segment definition | Segment = operating day | Overnight gaps are expected |
| GTI > 10 filter | Keep but note redundancy | Electrical data already daytime-only |
| Time-of-day features | Add sin/cos encoding | Model learns diurnal patterns |
| Window overnight boundary | NEVER span overnight | Physical discontinuity |
| Cold-start | Accept limitation | Honest, defensible |
| Intra-day gaps | Need to verify in EDA | Separate from overnight |

---

*Document created: April 8, 2026*
*Next step: Apply corrections in EDA notebook*
