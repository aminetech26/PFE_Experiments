# EDA Protocol and Decisions

This document records the exploratory data analysis protocol used in the project, the hypotheses each analysis is meant to test, and the dataset-specific conclusions that feed downstream preprocessing, splitting, and feature-engineering decisions.

The goal is to keep the EDA layer thesis-ready and dataset-aware: a shared backbone is reused across datasets, then each dataset adds only the diagnostics required by its own structure and risks.

---

## 1. Purpose of the EDA Layer

The EDA layer is not only descriptive. It is decision-oriented.

Its job is to answer the following questions before modeling:

1. Is the dataset complete enough to model safely?
2. Are there temporal continuity issues that affect segmentation or windowing?
3. Is the label distribution usable for the target task?
4. Which features appear discriminative, and are those conclusions robust after accounting for temporal dependence?
5. Which features are redundant or structurally collinear?
6. Are there dataset-specific anomalies that require targeted treatment or caution?
7. Do the observations justify downstream design choices for preprocessing, splitting, and feature selection?

The EDA layer therefore produces both:

- descriptive artifacts: reports, figures, statistical summaries
- decision artifacts: hypotheses confirmed/rejected, constraints, and downstream recommendations

---

## 2. Shared Dataset-Aware EDA Template

For each dataset, the EDA section should follow the same template:

### 2.1 Dataset Context

- data source
- sampling rate
- label structure
- known acquisition constraints
- known domain caveats

### 2.2 EDA Objectives

- what risks must be verified for this dataset
- what uncertainties must be reduced before preprocessing and splitting

### 2.3 Hypotheses

Each dataset defines explicit hypotheses, for example:

- data completeness hypothesis
- continuity/windowing hypothesis
- class imbalance hypothesis
- feature redundancy hypothesis
- fault separability hypothesis
- dataset-specific anomaly hypothesis

### 2.4 Protocol

The protocol must state:

- which tests/diagnostics were run
- why they were added
- what each one is supposed to confirm or falsify

### 2.5 Findings

- concise evidence-backed results
- row-level and grouped-level perspectives when relevant

### 2.6 Decisions

- what is adopted downstream
- what is rejected
- what remains open and must be revisited later

---

## 3. Shared EDA Backbone Across Datasets

The following EDA components form the common backbone and should be reused across Costa, La Reunion, and Mendeley whenever applicable.

### 3.1 Data Integrity and Structural Checks

- verify dataset row count after ingestion
- verify expected sensor columns and label column availability
- compute missing-value counts by column and by row
- inspect impossible values or nonphysical ranges when relevant

Why this exists:

- to determine whether imputation is needed
- to detect ingestion problems early
- to prevent misleading downstream preprocessing assumptions

### 3.2 Temporal Continuity Checks

- nominal sampling interval
- full gap distribution
- thresholded counts of large gaps
- intra-day vs inter-day gap characterization when timestamps support it
- per-day continuity summaries when continuity is operationally important

Why this exists:

- to determine whether continuity-aware segmentation or windowing is required
- to distinguish real acquisition breaks from filtering-induced retained-data gaps

### 3.3 Label Structure Checks

- class distribution by rows
- class distribution by grouped events/episodes
- minimum grouped support for evaluability

Why this exists:

- to detect class imbalance early
- to decide whether row-level summaries hide severe event-level imbalance

### 3.4 Distribution and Discriminative-Signal Checks

- Mann-Whitney normal vs fault
- per-class normal vs each fault type
- Kruskal-Wallis multiclass global test
- effect sizes, not only p-values
- multiple-testing correction using FDR

Why this exists:

- to determine whether features show meaningful separation
- to avoid overclaiming significance in large datasets

### 3.5 Dependence-Aware Confirmation

- grouped aggregation by segment or episode median
- repeat key tests at grouped level

Why this exists:

- to reduce row-wise overconfidence caused by dense time-series dependence
- to separate robust feature signal from repeated-sample inflation

### 3.6 Redundancy and Relevance Checks

- Spearman correlation
- VIF
- mutual information

Why this exists:

- to identify likely redundant features
- to prevent exact or near-exact collinearity from distorting models and interpretation

### 3.7 Dataset-Specific Diagnostic Extensions

Only add these when required by the dataset. Examples:

- La Reunion: null-episode analysis, imputation overlap with faults, natural gap structure
- Costa: irradiance-trimming gap effects, episode morphology, regime-binned correlation, targeted VDC1 anomaly localization
- Mendeley: simulation regularity, domain-shift realism checks, label-generation artifact checks

---

## 4. Costa EDA Objectives

Costa is the current canonical vertical benchmark.

Its EDA layer is designed to answer the following:

1. Is the ingested Costa dataset clean enough for modeling without missing-value imputation?
2. Does irradiance trimming create continuity issues that affect segmentation and windowing?
3. Is row-wise significance reliable, or does dense 1 Hz dependence exaggerate it?
4. Are some sensor channels structurally redundant rather than merely trend-correlated?
5. Is the class imbalance severe only by rows, or also by grouped events?
6. Are there localized sensor behaviors that require targeted attention before preprocessing?
7. Is pooled correlation vulnerable to spurious trend correlation from irradiance and operating regime?

---

## 5. Costa Hypotheses

### H1. Data completeness

After ingestion and daytime trimming, Costa is complete enough to proceed without missing-value imputation.

### H2. Continuity

Irradiance trimming preserves a mostly continuous daytime sequence but may leave a small number of operationally important intra-day breaks.

### H3. Row-wise overconfidence

Because Costa is dense 1 Hz time-series data, row-wise statistical significance will overstate discriminative evidence unless confirmed at grouped level.

### H4. Structural redundancy

Some electrical features, especially power-derived channels, are structurally redundant and should not all be retained by default.

### H5. Imbalance must be checked beyond rows

Class imbalance must be evaluated at both row and grouped-event level, because row counts alone may hide event imbalance.

### H6. Targeted sensor anomaly

`vdc1` may contain systematic abnormal behavior during nominally normal periods and therefore deserves dedicated localization.

### H7. Spurious pooled correlation

Global pooled correlation may be inflated by shared irradiance/operating-regime effects, so conditional correlation checks are needed before concluding redundancy.

### H8. Stationarity caution

Raw Costa signals, and possibly the first round of transformed signals, may remain non-stationary enough that pooled correlation must be interpreted cautiously.

---

## 6. Costa EDA Protocol

The implemented Costa EDA protocol follows the sequence below.

### 6.1 Integrity and Basic Structure

Tests run:

- row count verification
- sensor column availability check
- missing-value analysis by row and by column
- retained irradiance range inspection after ingestion

Purpose:

- confirm whether Costa requires a La Reunion-like missing-value strategy
- verify the modeling parquet is structurally valid before deeper analysis

### 6.2 Temporal Continuity Analysis

Tests run:

- nominal sampling interval estimation
- full gap distribution
- thresholded gap counts
- intra-day gap analysis
- per-day intra-day continuity summary

Purpose:

- determine whether windowing can operate on the retained sequence safely
- distinguish expected inter-day retained-data gaps from problematic intra-day breaks

### 6.3 Episode Definition and Label Structure

Tests run:

- class distribution by rows
- grouped distribution using `episode_id`
- episode morphology summary: count, median duration, tail behavior, short-episode proportion

Definition used:

- `episode_id` is a contiguous run broken by either a major time gap or a label transition

Purpose:

- represent Costa in a way that respects event boundaries rather than only raw rows
- provide a meaningful unit for event-aware diagnostics and later split decisions

### 6.4 Outlier Diagnostics

Tests run:

- global extreme outlier scan using IQR x 3
- targeted localization for `vdc1` normal-period outliers by day, hour, and episode

Purpose:

- distinguish generic extreme values from localized structured behavior
- decide whether a feature needs targeted inspection before preprocessing policy is finalized

### 6.5 Discriminative Signal Tests

Tests run:

- Mann-Whitney: normal vs all faults merged
- Mann-Whitney: normal vs each fault class separately
- Kruskal-Wallis across all classes
- FDR correction using Benjamini-Hochberg

Purpose:

- test whether features show separability under binary and multiclass views
- avoid relying on raw p-values alone in a large dataset

### 6.6 Dependence-Aware Confirmation

Tests run:

- aggregate rows to episode-level medians
- rerun the main discriminative tests on aggregated episode observations

Purpose:

- reduce inflation caused by dense, highly dependent row sequences
- retain only feature evidence that survives grouped analysis

### 6.7 Redundancy and Relevance Diagnostics

Tests run:

- pooled Spearman correlation
- normal-only Spearman correlation
- normal-only irradiance-binned Spearman correlation
- VIF
- mutual information for binary and multiclass targets

Purpose:

- separate likely structural redundancy from simple pooled trend correlation
- verify whether high-correlation pairs remain high within more controlled operating regimes

### 6.8 Stationarity Diagnostics

Tests run:

- ADF and KPSS on a fixed panel of long normal episodes
- comparison of raw and first-round transformed variables across episodes

Purpose:

- evaluate whether signals remain strongly non-stationary under a stable multi-episode protocol
- use the result as a cautionary layer for interpretation of pooled dependence statistics

Important note:

- stationarity results are treated as supporting evidence for interpretation, not as the sole determinant of preprocessing or split decisions
- inconclusive ADF/KPSS results are acceptable if preprocessing still improves interpretability and downstream split-robust modeling behavior

---

## 7. Why Each Costa Test Was Added

This subsection is meant to be thesis-friendly: each test is tied to a hypothesis.

| Hypothesis | Test Added | Why It Was Needed |
|-----------|------------|-------------------|
| H1 | Missing-value analysis | To verify whether Costa shares La Reunion-style completeness problems |
| H2 | Intra-day gap analysis | To detect trimming-induced discontinuities that could break windowing |
| H3 | Episode-aware aggregated tests | To avoid overconfident row-wise significance from dense 1 Hz dependence |
| H4 | VIF + normal-only + regime-binned correlation | To check whether redundancy is structural or only pooled-trend driven |
| H5 | Row vs episode imbalance summary | To distinguish row imbalance from event imbalance |
| H6 | Targeted `vdc1` localization | To determine whether outliers are random noise or structured behavior |
| H7 | Normal-only irradiance-binned correlation | To reduce spurious correlation bias from shared irradiance trend |
| H8 | ADF + KPSS on a fixed panel of normal episodes | To quantify how strongly non-stationarity remains after first-round transforms without over-relying on a single episode |

---

## 8. Costa Findings and Decisions

### 8.1 Data Completeness

Finding:

- the ingested Costa parquet contains no missing values in timestamp, label, or sensor columns

Decision:

- Costa does not require a La Reunion-style missing-value imputation policy at this stage

### 8.2 Continuity After Irradiance Trimming

Finding:

- inter-day retained-data gaps are expected after `irr >= 100`
- only a small number of intra-day gaps remain above operational thresholds
- these gaps are few but important for windowing

Decision:

- preserve irradiance trimming
- do not replace it with blunt clock-based trimming
- treat continuity breaks as segmentation/windowing boundaries

### 8.3 Event Structure and Imbalance

Finding:

- labels `1`, `2`, and `3` are minority classes both by rows and by episodes
- label `4` is not a rare anomaly class; it is large by both rows and episode count
- row share and episode share tell different but complementary stories

Decision:

- future split and evaluation design must consider both row-level and episode-level imbalance
- classification metrics must emphasize macro/per-class behavior, not accuracy alone

### 8.4 Row-Wise vs Episode-Aware Significance

Finding:

- row-wise tests make all major sensor features appear strongly significant
- after episode aggregation, only a subset remains robustly discriminative

Decision:

- feature relevance should not be concluded from row-wise significance alone
- episode-aware evidence is preferred when making feature-importance claims in the thesis

### 8.5 Feature Redundancy

Finding:

- `pdc`, `pdc1`, and `pdc2` remain extremely correlated even after conditioning on normal operation and irradiance bins
- redundancy therefore appears structural, not merely an artifact of pooled trend correlation

Decision:

- these channels should not all be treated as independent evidence
- a strong default candidate is to drop `pdc` first and retain `pdc1` and `pdc2` when string asymmetry is useful

### 8.6 Targeted `vdc1` Concern

Finding:

- `vdc1` normal-period outliers are localized in specific days, hours, and short episodes rather than being uniformly random

Decision:

- `vdc1` requires dedicated inspection before final preprocessing policy is locked
- the correct response is targeted analysis, not blind global outlier clipping

### 8.7 Stationarity Caution

Finding:

- under the current protocol, both raw and first-round transformed Costa signals remain only partially stabilized across the tested normal episodes

Decision:

- pooled correlation must be interpreted with caution
- row-level significance claims should not rely on a stationarity assumption being fully satisfied
- stationarity is important for interpretation, but current split and continuity decisions do not need to wait for a final stationarity redesign

---

## 9. Costa Conclusions Already Strong Enough for Downstream Design

Even before finalizing the stationarity protocol, Costa EDA already supports the following design conclusions:

1. Costa is complete enough to model without missing-value imputation.
2. Costa requires continuity-aware handling because a small number of important intra-day breaks remain after trimming.
3. Episode-aware reasoning is necessary because dense row-wise analysis inflates statistical confidence.
4. Labels `1`, `2`, and `3` must be treated as minority classes in both row and episode terms.
5. Label `4` shadowing is a major class and should not be treated as a rare edge case.
6. `pdc`, `pdc1`, and `pdc2` are structurally redundant enough to justify pruning decisions.
7. `vdc1` deserves dedicated sensor/regime investigation before final preprocessing decisions.

---

## 10. Reusable Writing Pattern for Other Datasets

For La Reunion and Mendeley, reuse the same narrative chain:

test -> finding -> decision

Examples:

- gap analysis -> continuity breaks detected -> continuity-aware segmentation required
- missing-value analysis -> fault-overlapping null episodes found -> no imputation on fault rows
- grouped significance analysis -> row-wise inflation detected -> prefer grouped-level evidence for feature claims
- regime-binned correlation -> redundancy persists within regime -> prune structurally redundant channels

This keeps the thesis coherent across datasets while allowing each dataset to justify its own special decisions.

---

## 11. Current Status

- Costa EDA protocol: implemented and evidenced by generated reports/figures
- Costa stationarity interpretation: informative, multi-episode, and still treated as a supporting research conclusion rather than a hard modeling gate
- La Reunion and Mendeley dataset-specific EDA write-ups: to be completed using the same template
