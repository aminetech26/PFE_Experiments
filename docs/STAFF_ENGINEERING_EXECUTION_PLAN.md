# Staff Engineering Execution Plan — PV Fault Detection System (Research + Production)
## ESI × CDER · Ahmed Amine GUERRAICHE · March–April 2026

## 1) Executive Judgment on Existing Plans

### What is strong already
- `MASTER_PLAN.md` is execution-oriented, time-boxed, and aligned with internship deliverables.
- `TECHNICAL_DESIGN.md` is theoretically rigorous, correctly identifies leakage traps, and makes strong architecture decisions.
- Together, they already cover core ingredients for a high-quality PFE and paper: multi-task formulation, anti-leakage protocol, edge constraints, and statistical testing.

### Critical gaps to fix now
1. **Too many parallel tracks early**: scope threatens schedule reliability (core model, MTL, CEEMDAN, DANN, GUI, deployment, publication all at once).
2. **Missing hard go/no-go gates**: no explicit kill criteria for weak approaches.
3. **Insufficient production contracts**: no strict data schema/version contracts, model interface contracts, and runtime SLO contracts.
4. **Unclear definition of “done” per week**: many tasks listed, but acceptance criteria are not binary enough.
5. **Risk handling is implicit**: no quantified risk register with contingency actions.
6. **Compute budget not enforced operationally**: HPO budget exists conceptually but no experiment budget governance.

### Strategic correction
Use a **two-lane program**:
- **Lane A (Must Ship)**: leakage-safe baselines → one robust deployable model → ONNX/TensorRT → Jetson dashboard.
- **Lane B (Research Depth)**: MTL enhancements, CEEMDAN ablations, domain adaptation (DANN), publication-grade extras.

Rule: Lane B only consumes capacity if Lane A stays green on schedule and metrics.

---

## 2) Program Objectives and Non-Negotiables

## Objective hierarchy
1. **Primary (Must deliver by end-April)**
   - Reliable anomaly detection + fault classification + 5–10 min forecasting pipeline.
   - Reproducible experiments with leakage controls.
   - Edge deployment on Jetson Nano with measurable latency.
2. **Secondary (High value if schedule allows)**
   - MTL superiority demonstrated over separate models.
   - Domain adaptation evidence (Mendeley → La Réunion).
3. **Tertiary (Publication booster)**
   - Advanced ablations and cross-climate transfer analysis with strong statistical claims.

## Engineering non-negotiables
- No metric is accepted without confidence intervals and seed variability.
- No model is accepted without leakage report pass.
- No deployment claim is accepted without measured latency on target hardware.
- No architecture change without rollback path and measurable expected gain.

---

## 3) Target System Definition (v1.0)

## Inputs
- Time-windowed electrical + meteorological signals from Mendeley, La Réunion, Sonalgaz.

## Outputs
- Task A: anomaly score + binary alert
- Task B: fault class probabilities
- Task C: forecasted anomaly risk at +5 and +10 min

## v1.0 model family (production-first)
- Shared temporal encoder: **TCN (default)**
- Heads:
  - A: reconstruction/anomaly head
  - B: multiclass classification head
  - C: GRU-based forecasting head
- Optimizer: AdamW + cosine schedule
- Class imbalance: Focal loss + balanced batching

## Why this is the right v1.0
- Best tradeoff between latency, robustness, and deployment tractability.
- TensorRT-friendliness superior to recurrent-heavy end-to-end designs.
- Satisfies both scientific rigor and real-world constraints.

---

## 4) Architecture and Methodology Refinements

## 4.1 Data contracts (add this immediately)
Define and enforce schema contracts per dataset:
- Required columns, dtype, timestamp resolution, missing-value policy, unit normalization rules.
- Version tags: `dataset_name`, `snapshot_date`, `preprocess_version`, `feature_version`.
- Any schema drift fails pipeline stage before training.

## 4.2 Split contracts
- La Réunion/Sonalgaz: strict chronological split with purge gap.
- Mendeley: stratified-by-fault with temporal ordering preserved inside each fault file.
- All split artifacts persisted and versioned (never regenerated silently).

## 4.3 Feature policy
- Tier 1 (always-on): PR, dP/dt, dI/dt, dV/dt, thermal delta, robust window stats.
- Tier 2 (dataset-specific):
  - Mendeley: CEEMDAN IMF energies.
  - La Réunion: wavelet denoising + inverter differential signal.
- Tier 3 (experimental): GASF and heavy transforms behind explicit feature flags.

## 4.4 Model governance
- Baseline champion per task first (single-task).
- MTL only promoted if it beats champion on weighted utility score:
  - Utility = 0.5×(Task A primary metric) + 0.3×(Task B) + 0.2×(Task C) minus latency penalty.

## 4.5 Thresholding and operations
- Task A thresholds selected by validation PR curve under target false-alarm budget.
- Two operating points persisted:
  - `high_recall_mode` (safety-first)
  - `balanced_mode` (operations-first)

---

## 5) Delivery Gates (Go/No-Go)

## Gate G1 — Data readiness (end Week 1)
Pass criteria:
- All datasets ingested to Parquet with schema contracts passing.
- Missingness and outlier reports generated.
- Leakage pre-checks pass (duplicate/time boundary checks).

Fail action:
- Freeze feature work; fix ingestion/data quality first.

## Gate G2 — Baseline readiness (end Week 3)
Pass criteria:
- Per-task baseline leaderboard with mean ± 95% CI across 5 seeds.
- Label shuffle test collapses to near-chance.
- Experiment reproducibility validated from clean environment.

Fail action:
- Stop DL expansion; resolve split/leakage/feature bugs.

## Gate G3 — Deep model readiness (end Week 4)
Pass criteria:
- At least one DL candidate exceeds baseline by pre-defined margins:
  - Task A: +0.05 PR-AUC
  - Task B: +0.03 F1-macro
  - Task C: -0.01 normalized MAE
- Training stability acceptable (no divergence across seeds).

Fail action:
- Keep baseline as production candidate; postpone MTL.

## Gate G4 — Production readiness (end Week 7)
Pass criteria:
- ONNX parity test passes.
- TensorRT inference meets latency SLO on Jetson.
- Runtime memory stays within budget.

Fail action:
- Model compression/architecture reduction; fallback to smaller single-task runtime bundle.

## Gate G5 — Freeze (end Week 8)
Pass criteria:
- End-to-end demo stable.
- Final metrics package complete with statistical tests.
- Thesis-ready figures/tables auto-generated from run artifacts.

---

## 6) Revised 8-Week Plan (Staff-Engineer Version)

## Week 1 — Data Reliability Sprint
Deliverables:
- Data contracts + ingestion pipeline + quality report.
- Persistent split artifacts and leakage pre-check report.

Acceptance criteria:
- One command reproduces `raw -> interim -> processed`.
- Any schema mismatch fails fast.

## Week 2 — Feature and Evaluation Contract Sprint
Deliverables:
- Feature pipeline with tiered flags.
- Unified evaluation module per task with metric registry.

Acceptance criteria:
- Identical metrics from reruns (seed-controlled).
- Preprocessor fit scope validated in tests.

## Week 3 — Baseline Tournament Sprint
Deliverables:
- Baseline leaderboard for A/B/C tasks.
- Early HPO for top-2 models per task.

Acceptance criteria:
- Confidence intervals + leakage checks for every reported row.
- One baseline champion selected per task.

## Week 4 — DL Champion Sprint
Deliverables:
- TCN/GRU/LSTM candidates benchmarked against champions.
- Decision memo: promote or reject DL per task.

Acceptance criteria:
- Clear win or justified rejection with evidence.

## Week 5 — Signal Processing ROI Sprint
Deliverables:
- CEEMDAN and wavelet integrations as controlled ablations.

Acceptance criteria:
- Keep only components with statistically meaningful gains.

## Week 6 — MTL Qualification Sprint
Deliverables:
- MTL prototype with Kendall weighting.
- Compare against best separate-model bundle.

Acceptance criteria:
- Promote MTL only if utility score improves and latency penalty acceptable.

## Week 7 — Deployment Qualification Sprint
Deliverables:
- ONNX export, TensorRT engine, latency and memory profiles.
- Runtime monitoring hooks for drift and confidence.

Acceptance criteria:
- Meets target SLOs and produces stable inference in repeated runs.

## Week 8 — Integration and Freeze Sprint
Deliverables:
- GUI integration, alarm modes, final demo scenario.
- Final results pack (tables, plots, statistical appendix).

Acceptance criteria:
- Reproducible end-to-end demo from clean start.
- All artifacts traceable to versioned runs.

---

## 7) Metrics and SLO Framework

## Model quality metrics
- Task A: PR-AUC primary, recall@fixed-precision secondary.
- Task B: F1-macro primary, per-class recall and confusion matrix secondary.
- Task C: MAE primary, horizon-wise calibration error secondary.

## Production SLOs
- Inference latency: <= 50 ms/window on Jetson Nano.
- Runtime memory: <= 500 MB total app footprint.
- Alert freshness: <= 2 windows delay for anomaly status update.
- Stability: no crash in 4-hour continuous demo run.

## Statistical reporting
- 5 seeds minimum, bootstrap 95% CI, Wilcoxon paired tests, effect size.
- Promotion requires practical significance, not just p-value.

---

## 8) Experiment Budget Governance

## Compute budget rules
- HPO budget cap per task per week.
- Stop trials early with pruning when no progress after patience window.
- Every trial must log config hash, data hash, split id, feature version.

## Promotion policy
- Candidate promoted only when:
  - Metric gain exceeds minimal practical threshold.
  - Variance across seeds does not increase materially.
  - Latency/memory impact remains within budget.

---

## 9) Risk Register (Top 8)

1. **Leakage hidden in preprocessing**
   - Mitigation: unit tests + split artifact immutability + label shuffle checks.
2. **Overfitting to simulated Mendeley patterns**
   - Mitigation: cross-domain validation and domain-shift diagnostics.
3. **MTL instability from loss imbalance**
   - Mitigation: Kendall weighting + sigma monitoring + fallback to separate models.
4. **Jetson latency misses target**
   - Mitigation: architecture slimming + FP16/INT8 pipeline + head decoupling.
5. **Unlabeled Sonalgaz ambiguity**
   - Mitigation: weak-label extraction protocol and separate confidence tiers.
6. **Schedule collapse due to parallel scope**
   - Mitigation: two-lane strategy and strict gate reviews.
7. **Irreproducible results**
   - Mitigation: DVC+MLflow mandatory lineage and seed discipline.
8. **Publication over-optimization before core delivery**
   - Mitigation: lock Lane A first, then allocate spare capacity to Lane B.

---

## 10) Minimal Deliverable Set for End-April (Must Ship)

- Reproducible pipeline from ingestion to evaluation.
- One deployable model bundle satisfying quality and latency thresholds.
- Jetson demo with live dashboard and fault/anomaly outputs.
- Final benchmark report with statistical rigor and leakage evidence.

Anything beyond this is optional and publication-accelerating, not mandatory for project success.

---

## 11) Research Extensions (Only After Must-Ship Is Green)

Priority order:
1. DANN for Mendeley → La Réunion transfer.
2. CEEMDAN + GASF hybrid comparison.
3. Drift-adaptive thresholding and online recalibration.

Each extension must include a “worth it?” summary:
- gain, cost, deployment impact, confidence level.

---

## 12) Immediate Next 72 Hours

1. Implement schema/split contracts and freeze split artifacts.
2. Produce a single reproducible baseline run for each task with full lineage.
3. Add gate check script that outputs PASS/FAIL for G1 and G2 prerequisites.
4. Create a lightweight weekly review template: decision, evidence, next action.

This sequence maximizes delivery certainty while preserving scientific depth.

---

*End of Staff Engineering Execution Plan*
