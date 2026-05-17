# Split Strategy Decisions

**Author:** Ahmed Amine GUERRAICHE  
**Revision:** 2.0 - April 2026  
**Status:** Active reference for dataset-aware split policy

This document records the split strategy decisions for the current thesis scope.

Current scope:

- `Task A` - fault detection / anomaly detection
- `Task B` - fault classification

Forecasting / prediction is no longer a primary thesis task and is therefore not part of the active split policy below.

---

## 1. Global Anti-Leakage Rules

These rules apply to all datasets unless a dataset section explicitly tightens them further.

### 1.1 Split Before Windowing

Sliding windows must be generated only after train/validation/test partitions are assigned.

Why:

- overlapping windows share most of their samples
- windowing before split creates severe leakage even if row-level timestamps differ

Required order:

1. define atomic grouped units
2. split grouped units into train/val/test
3. generate windows independently within each partition

### 1.2 Atomic Split Units

Rows are not valid split units for PV fault time series.

Splits must operate on grouped units such as:

- contiguous gap-based segments
- label-transition-aware episodes
- or another dataset-justified event unit

No grouped unit may be fragmented across train/val/test.

### 1.3 Train-Only Fitting

Anything that estimates parameters from data must be fit on train only, then applied to val/test.

Examples:

- normalization/scaling
- outlier bounds
- differential calibration constants
- feature-selection thresholds when learned from train data

### 1.4 Purge Is Optional, Not Primary Protection

The primary leakage defense is atomic grouped-unit splitting plus post-split windowing.

Purge/embargo is only an additional safeguard when adjacent grouped units remain suspiciously dependent across boundaries.

### 1.5 Honest Evaluability Rule

A class may contribute to training but still be non-evaluable for independent reporting if it lacks enough grouped support to distribute honestly across splits.

---

## 2. Dataset-Aware Structure

Each dataset section follows the same decision pattern:

1. critical structural finding
2. split implications
3. task-specific split policy
4. leakage and evaluability notes
5. implementation status

---

## 3. La Reunion Decisions

La Reunion remains an important source of split-design knowledge because it exposed how naive temporal splitting can make evaluation impossible.

### 3.1 Critical Structural Finding

All fault events occur in a restricted historical portion of the dataset, while a large later interval is fault-free.

Implication:

- strict global temporal splitting makes classification evaluation impossible
- task-specific strategies are required

### 3.2 Core Split Lesson Learned

The key lesson from La Reunion is not a single fixed split recipe.

The actual lesson is:

- split strategy must be task-specific
- temporal ordering must be preserved where scientifically meaningful
- grouped segments must be atomic units
- evaluability constraints can override naive temporal purity

### 3.3 Task A - Detection

#### A1. Semi-Supervised

- `Train`: normal-only grouped segments
- `Val/Test`: grouped segments containing normal and faults
- rationale: one-class anomaly learning with realistic mixed evaluation

#### A2. Supervised

- `Train/Val/Test`: temporal-stratified grouped segments with normal + faults
- rationale: binary supervised learning while preserving temporal order within classes

### 3.4 Task B - Classification

- classification operates on fault classes only
- only evaluable classes are reported independently
- classes with insufficient grouped support may still contribute to Task A training but are not reported as fully evaluable Task B classes

### 3.5 Leakage Policy

- split unit: contiguous grouped segments
- windowing: after split only
- no global purge used as the main defense
- grouped boundaries are treated as sufficient primary isolation

### 3.6 Why This Section Still Matters

Although Costa is now the canonical vertical benchmark, La Reunion remains the historical proof that split design must be driven by dataset structure rather than convenience.

### 3.7 Implementation Status

- La Reunion logic informed the current split codebase
- some implementation details in `PFE_Experiments/src/data/split_pipeline.py` still reflect this older segment-centric design

---

## 4. Costa Decisions

Costa is the current canonical benchmark and therefore needs its own split policy rather than a direct reuse of the La Reunion design.

### 4.1 Critical Structural Findings from Costa EDA

Costa EDA established the following facts:

1. the ingested Costa dataset is complete after `irr >= 100` trimming
2. the retained sequence is mostly continuous, but a small number of important intra-day gaps remain
3. dense 1 Hz rows are highly dependent, so row-wise reasoning is overconfident
4. fault morphology differs strongly by class:
   - labels `1` and `2` often form long induced episodes
   - labels `3` and `4` frequently form many short episodes
5. imbalance must be assessed at both row and episode level

Implication:

- Costa splitting must be continuity-aware and event-aware
- row-level splitting is not acceptable
- window generation must occur after split and inside grouped boundaries only

### 4.2 Costa Atomic Units

Two grouping concepts matter in Costa:

#### A. Gap-Based Segment

- defined by major temporal discontinuity after retained-data trimming
- useful for continuity control

#### B. Episode

- defined by major gap or label transition
- better aligned with fault-event identity
- more meaningful for supervised classification logic

Current EDA conclusion:

- gap-aware grouping is the minimum safe unit
- episode-aware grouping is the preferred unit for supervised fault classification
- Costa implementation should treat `episode_id` as the canonical atomic split unit and preserve continuity-only segments as supporting metadata

### 4.3 Costa Task A - Detection

Both semi-supervised and supervised detection remain scientifically valid on Costa.

#### A1. Semi-Supervised Detection

Recommended policy:

- `Train`: normal-only grouped units
- `Val/Test`: grouped units containing normal + faults
- preserve temporal order within grouped assignment as much as possible

Rationale:

- this matches the anomaly-detection framing
- Costa has enough normal and fault material to evaluate mixed val/test partitions honestly

Preferred split unit:

- minimum: continuity-aware grouped segment
- acceptable stronger option: episode-aware grouping

#### A2. Supervised Detection

Recommended policy:

- `Train/Val/Test`: temporal-stratified grouped split over binary normal/fault labels

Rationale:

- enables supervised anomaly baselines
- still requires grouped-unit integrity because dense rows are highly dependent

Preferred split unit:

- minimum: continuity-aware grouped segment
- recommended for consistency: episode-aware grouped unit if it does not create pathological fragmentation for normal data

### 4.4 Costa Task B - Classification

Classification is fault-only and should operate on grouped fault events, not rows.

Recommended policy:

- `Train/Val/Test`: temporal-stratified grouped split over fault classes only
- normal class is excluded from Task B targets

Preferred split unit:

- `episode_id` or equivalent label-transition-plus-gap grouped unit

Why episode-aware grouping is preferred:

- class identity is tied to contiguous fault runs, not arbitrary gap blocks
- many Costa fault classes have short transient episodes
- label-pure windows are easier to guarantee when episode boundaries are respected

### 4.5 Costa Evaluability Policy

Current Costa EDA indicates that labels `1`, `2`, `3`, and `4` are all presently evaluable at the grouped level.

However, evaluability must still be interpreted honestly:

- label `4` is large by both rows and episodes
- labels `1`, `2`, and `3` are minority classes and require per-class reporting
- if a later grouped split reveals that a class cannot be distributed across train/val/test with enough support, that class should be downgraded from independently evaluable to train-only support

### 4.6 Costa Leakage Policy

For Costa, leakage-safe splitting means:

1. define grouped units first
2. assign whole grouped units to partitions
3. generate windows afterward, separately within each partition
4. never let a window cross a grouped-unit boundary

Purge policy:

- no purge is required as the first-line defense
- grouped-unit splitting plus post-split windowing is the primary anti-leakage mechanism
- a small purge may be considered later only if adjacent grouped boundaries still show suspicious dependence in practice

### 4.7 Costa Time-Clipping Decision

Fixed clock-based trimming was considered as a way to remove intra-day gaps.

Decision:

- reject fixed time clipping as the primary solution

Reason:

- it removes continuity problems but distorts the class distribution, especially shadowing (`label 4`)
- continuity-aware segment/episode-aware windowing preserves the scientifically meaningful irradiance-trimmed subset more honestly

### 4.8 Costa Working Split Recommendation

Current working recommendation before final implementation:

- `Task A semi-supervised`
  - train on normal-only grouped units
  - evaluate on mixed grouped units
- `Task A supervised`
  - grouped temporal-stratified binary split
- `Task B classification`
  - grouped temporal-stratified fault-only split using episode-aware units
- `Windowing`
  - always after split
  - always within grouped boundaries only

### 4.9 Costa Two-Path Fork

Costa now retains two explicitly different split paths for different methodological purposes.

#### Path A — Canonical Episode-Based Benchmark

- atomic unit: `episode_id`
- status: primary benchmark path
- purpose: maximize leakage safety and event-level validity
- admissible downstream methods: local or grouped-safe transforms only

This is the path that should support the main Costa benchmark claims.

#### Path B — Day-Based Purged Comparison

- atomic unit: retained `operating_day`
- assignment: pure chronological day blocks
- purge: forward half-day purge on the first validation and first test days; full-day purge remains optional but is not the Costa default because retained day count is small
- status: secondary methodological comparison path

Path B exists to test continuity-dependent spectral or signal-processing feature engineering methods that are hard to justify under Path A.

#### Path B continuity assumption

- Costa is not treated as perfectly continuous
- instead, retained same-day samples are treated as sufficiently quasi-continuous at the day level
- residual intra-day gaps are tolerated by assumption and documented as a limitation rather than used to split days further

#### Path interpretation rule

- `Path A` answers: what is the strongest leakage-safe event-aware Costa benchmark?
- `Path B` answers: do continuity-dependent spectral or signal-processing features add value when a day-level continuity assumption is explicitly allowed?

#### Current comparison finding

- `Path A` remains preferable for Task B classification because episode-based assignment preserves fault-event trainability.
- `Path B` can improve minority-class validation/test support, but day-based grouping can also starve some fault classes in train.
- Therefore `Path B` is currently most useful as a secondary path for Task A and for continuity-dependent spectral or signal-processing feature-engineering comparisons.

### 4.10 Costa Implementation Status

Important status note:

- Costa split generation now supports `episode_id` as the canonical atomic split unit via config
- continuity-aware grouping remains available as `continuity_segment_id` for diagnostics and supporting analyses
- Costa Path A remains the default split output under `data/interim/splits/costa/`
- Costa Path B comparison outputs live under `data/interim/splits/costa/path_b/`
- Costa policy should therefore be read as implemented split behavior, not only as a future design target

---

## 5. Mendeley Decisions

Mendeley is not yet the active vertical benchmark, but it should still follow the same design logic once promoted.

Current provisional policy:

- use grouped contiguous units as atomic split objects
- preserve temporal ordering where relevant
- evaluate whether simulated fault generation creates artificial regularity that would make random splitting misleadingly easy

This section remains provisional until Mendeley-specific EDA and split analysis are completed.

---

## 6. Practical Rules for the Thesis

When writing the thesis, the split strategy should be described using the following logic:

1. dataset structure was inspected first
2. grouped atomic units were defined to prevent leakage
3. task-specific split logic was adopted rather than a single global split rule
4. window generation was performed only after split
5. evaluability was defined honestly at the grouped-event level, not only by row count

Recommended wording pattern:

> We do not split PV time series at the row level. Instead, contiguous grouped units are treated as atomic objects for train/validation/test assignment. Sliding windows are generated only after split, within each partition separately, to prevent leakage from overlapping temporal context.

For Costa-specific wording:

> Because Costa remains mostly continuous after irradiance trimming but still contains a small number of important continuity breaks, and because fault classes are naturally organized as contiguous episodes, split assignment is performed on grouped event-level units rather than individual rows.

---

## 7. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-01 | La Reunion requires task-specific split strategies | Strict temporal split made evaluation impossible |
| 2026-04-02 | Segment/grouped units adopted as atomic split objects | Prevent row-level temporal leakage |
| 2026-04-02 | Windowing should occur after split | Overlapping windows leak if created first |
| 2026-04-22 | Split policy document restructured per dataset | Costa became canonical benchmark; one global split narrative was no longer enough |
| 2026-04-22 | Costa adopts continuity-aware and episode-aware split logic | Costa EDA showed grouped event structure is the relevant leakage-safe unit |
| 2026-04-22 | Costa rejects fixed time clipping as primary continuity fix | It reduces intra-day gaps but distorts class composition, especially label 4 |
| 2026-04-23 | Costa split logic forked into Path A / Path B | Needed to compare event-safe and continuity-oriented methods without weakening the main benchmark |

---

## 8. Authoritative References

- EDA methodology: `docs/EDA_PROTOCOL_AND_DECISIONS.md`
- system architecture: `docs/TECHNICAL_DESIGN.md`
- current split implementation: `PFE_Experiments/src/data/split_pipeline.py`
- current Costa EDA artifacts: `PFE_Experiments/data/interim/eda/costa/`

This document is the authoritative reference for dataset-aware split policy.
