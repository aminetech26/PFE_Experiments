# Thesis Positioning - Research-First PV Fault Detection and Diagnosis

**Author:** Ahmed Amine GUERRAICHE  
**Date:** April 2026  
**Status:** Canonical project positioning

---

## 1. Final Scope

This PFE is now framed as a **research-first PV fault detection and diagnosis (FDD)** project with a mandatory deployment deliverable.

The thesis has two primary scientific tasks:

1. **Fault detection** - detect whether the PV system is operating normally or under fault.
2. **Fault classification** - identify the fault class once abnormal operation is detected.

The project also has one mandatory engineering deliverable:

3. **Edge deployment with GUI** - deploy a selected model on edge hardware and expose results through a usable interface.

Forecasting is **not** a primary thesis task. It is retained only as an **optional residual-based anomaly analysis method**.

---

## 2. Why the Scope Changed

The project initially considered a third task: fault prediction / forecasting before fault onset.

That direction was reasonable at the time because a commercial partner indicated that long-term annotated operational data might become available. That dataset did not materialize. Once the available public datasets were examined carefully, a hard scientific constraint became clear:

- public PV fault datasets use **artificially induced faults**,
- those faults appear as **externally imposed step changes**,
- and they do **not** contain reliable pre-fault signatures from which true fault onset can be predicted.

As a result, direct fault prediction is not defended as a core contribution. This is a methodological correction, not a retreat. It improves the scientific honesty of the thesis.

---

## 3. Primary Dataset Strategy

The project now follows a **vertical-first, horizontal-later** strategy.

### 3.1 Vertical anchor: Costa

The **Costa** dataset is the primary benchmark because it offers the best current balance of:

- real fault data,
- meaningful sample size (~500k rows),
- four fault classes,
- comparability against published results,
- and direct relevance to both detection and classification.

Costa is therefore the main dataset for the first research phase and the benchmark that must be beaten or matched convincingly.

### 3.2 Horizontal expansion: La Reunion and Mendeley

After a strong Costa result is established, the project expands to broader research questions:

- **La Reunion** for real-world transfer and anomaly-focused evaluation under different sampling and climate conditions,
- **Mendeley / GPVS-Faults** for simulated-to-real transfer and domain adaptation studies.

These datasets support the generalization story, not the first proof-of-strength story.

---

## 4. Contribution Structure

### 4.1 Primary scientific contribution

Build a **leakage-safe, reproducible, research-grade PV FDD pipeline** centered on Costa, with rigorous evaluation and benchmark comparison.

### 4.2 Secondary scientific contribution

Quantify which choices matter most:

- feature families,
- window sizes,
- model families,
- calibration,
- and eventually transfer/adaptation strategies.

### 4.3 Mandatory engineering contribution

Select one model that offers a strong tradeoff between performance and deployability, then deploy it with a GUI on edge hardware.

### 4.4 Optional advanced contribution

Use forecasting only as a **normal-behavior modeling tool** whose residual can act as a complementary anomaly signal.

---

## 5. Research Philosophy

This thesis does not aim to impress by exploring the largest possible method zoo. It aims to make **defensible claims**.

The project priorities are therefore:

1. establish a trustworthy benchmark on Costa,
2. beat or strongly challenge the published baseline,
3. choose a deployable model,
4. then expand to transfer, adaptation, and broader generalization.

This ordering is deliberate. A weak vertical baseline cannot support strong horizontal claims.

---

## 6. What the Thesis Is Not Claiming

The thesis does **not** claim:

- true fault-onset prediction from current public datasets,
- universal cross-site generalization from the start,
- or edge deployment as the main scientific novelty.

Instead, it claims:

- rigorous problem formulation,
- strong benchmark execution,
- honest constraint handling,
- and research extensions built on a validated base.

---

## 7. Canonical One-Paragraph Thesis Framing

This thesis investigates photovoltaic fault detection and diagnosis through a research-grade, leakage-safe machine learning pipeline centered on the Costa real-world PV fault dataset. The primary objective is to build and evaluate models for fault detection and fault classification that match or exceed published benchmarks while remaining reproducible and deployable. Edge deployment with a GUI is retained as a mandatory engineering validation step. Additional datasets, namely La Reunion and Mendeley, are introduced only after the Costa benchmark is strong, in order to study transferability, domain shift, and sim-to-real generalization. Forecasting is no longer treated as a standalone thesis task; it is considered only as an optional residual-based anomaly analysis method because publicly available PV fault datasets do not contain scientifically valid pre-fault signatures.
