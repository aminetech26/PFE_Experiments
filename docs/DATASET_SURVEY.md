# Open Datasets for PV Fault Detection, Classification & Forecasting

**Survey Date:** April 2026
**Context:** Dataset landscape analysis for PFE multi-task system (anomaly detection, fault classification, fault forecasting). Constrained to electrical and meteorological time-series data .

**Primary reference:** Chen, X. et al. "Open data sets for assessing photovoltaic system reliability." *Applied Energy* 395, 126132 (2025). DOI: [10.1016/j.apenergy.2025.126132](https://doi.org/10.1016/j.apenergy.2025.126132)

---

## 1. Fault-Labeled Datasets (Electrical Time-Series)

These datasets contain explicit fault class annotations on electrical measurements. They are rare — the scarcity of labeled fault data in time-series form is a recognized gap in the field (Chen et al., 2025, Section 7.3).

### 1.1 Costa PV Fault Dataset

| Property | Detail |
|---|---|
| **Source** | Clayton H. Costa et al., Sensors MDPI (2020) |
| **Type** | Real (field measurements) |
| **PV System** | 2 strings x 8 modules (Canadian Solar C6SU-330P, 330W each), 5kW grid-tie inverter (NHS Solar 5K-GDM1) |
| **Location** | Brazil |
| **Duration** | 16-day acquisition campaign; paper describes daytime analysis window around ~07:30-17:00 |
| **Sampling rate** | 1 Hz |
| **Format** | MATLAB .mat (two files: `dataset_elec.mat`, `dataset_amb.mat`) |
| **License** | Public (citation required) |

**Features:**

| Column | Description |
|---|---|
| `vdc1` | DC voltage, string 1 |
| `vdc2` | DC voltage, string 2 |
| `idc1` | DC current, string 1 |
| `idc2` | DC current, string 2 |
| `irr` | Irradiance (W/m^2) |
| `pvt` | PV module temperature (C) |
| `f_nv` | Fault class label |

**Fault classes and sample counts:**

| Label | Fault Type | Samples | Notes |
|---|---|---|---|
| 0 | Normal operation | 309,253 | ~60% of total |
| 1 | Short-circuit between modules | 5,999 | Inter-module SC |
| 2 | Degradation / resistive fault | 10,371 | Simulates aging via added resistance |
| 3 | Open-circuit / disconnected string | 6,024 | String disconnection |
| 4 | Partial shadowing | 184,311 | Various shading patterns |

**Total:** ~515,958 labeled samples.

**Timestamp note:** The public MATLAB files expose ordered 1 Hz samples but do not include trustworthy absolute wall-clock timestamps. In this repository, Costa timestamps are reconstructed as a synthetic, solar-aligned clock so that diurnal features remain physically plausible; absolute clock-time interpretation remains approximate.

**Filtering note:** In this repository, Costa ingestion trims the raw continuous acquisition to `irr >= 100 W/m²`, which preserves physically meaningful operating periods and closely matches the widely cited `~515,958` sample daytime subset.

**Relevance:** Covers 4 of 5 target fault types (short-circuit, aging/degradation, open-circuit, shading). Only missing explicit overheating — though degradation/resistive faults cause thermal effects as a secondary signature. Includes meteorological data (irradiance + temperature). Real-world data at 1Hz provides a middle ground between La Reunion (0.14Hz) and Mendeley (10kHz).

**Download:** https://github.com/clayton-h-costa/pv_fault_dataset

**Associated paper:** Costa, C.H. et al. "A Monitoring System for Online Fault Detection and Classification in Photovoltaic Plants." *Sensors* 20(17), 4688 (2020). DOI: [10.3390/s20174688](https://doi.org/10.3390/s20174688)

**Associated simulation tool:** https://github.com/clayton-h-costa/pv_faultsim — MATLAB/PSim tool for generating additional fault scenarios. Could be used to produce synthetic overheating fault data.

---

### 1.2 GPVS-Faults (Grid-Connected PV System Faults)

| Property | Detail |
|---|---|
| **Source** | Mendeley Data |
| **Type** | Real (laboratory experimental) |
| **Sampling rate** | ~100 kHz (T_s = 9.9989 us) |
| **Format** | MATLAB .mat and CSV |
| **License** | CC BY 4.0 |

**Features (12 channels):**

| Column | Description |
|---|---|
| `Ipv` | PV array current |
| `Vpv` | PV array voltage |
| `Vdc` | DC bus voltage |
| `ia`, `ib`, `ic` | Three-phase AC currents |
| `va`, `vb`, `vc` | Three-phase AC voltages |
| `Iabc` | Current magnitude |
| `Vabc` | Voltage magnitude |
| `If`, `Vf` | Current and voltage frequency |

**Fault scenarios (16 files = 8 scenarios x 2 modes):**

| File prefix | Scenario | Description |
|---|---|---|
| F0 | Fault-free | Baseline normal operation |
| F1 | PV array fault (type 1) | Array-side electrical fault |
| F2 | PV array fault (type 2) | Array-side electrical fault |
| F3 | Inverter fault | Inverter malfunction |
| F4 | Grid anomaly | Grid-side disturbance |
| F5 | Feedback sensor fault | Measurement corruption |
| F6 | MPPT controller fault (type 1) | Control algorithm failure |
| F7 | MPPT controller fault (type 2) | Control algorithm failure |

Each scenario recorded under two operation modes:
- `M` = Maximum Power Point Tracking (MPPT)
- `L` = Limited Power (IPPT)

**Notes:** Faults were introduced manually halfway during experiments. High-frequency measurements are noisy with natural disturbances. After critical faults, the system may shut down. No meteorological data (irradiance/temperature) is directly included in the measurement channels.

**Relevance:** Covers inverter and controller faults not present in other datasets. Very high sampling rate enables signal processing techniques (spectral analysis, wavelet decomposition). Lab setting means controlled conditions but limited generalizability. Complements Costa (field data) and Mendeley (simulated data) for a three-domain comparison.

**Download:** https://data.mendeley.com/datasets/n76t439f65/1

**Associated paper:** Jovicic, A. et al. IEEE Access (2023). Available at: https://www.zemris.fer.hr/~ajovic/articles/Jovicic_et_al_IEEE_Access_2023_accepted.pdf

---

### 1.3 La Reunion (Already in Pipeline)

| Property | Detail |
|---|---|
| **Source** | University of La Reunion |
| **Type** | Real (field measurements) |
| **Location** | La Reunion island, France (tropical climate) |
| **Sampling rate** | ~7 seconds |
| **Duration** | Multi-year continuous monitoring |
| **Format** | CSV (ingested to Parquet in pipeline) |

**Datasets:**
- `dt1`: Solar and meteorological measurements (GTI, DTI, TA, TPV, wind, humidity)
- `dt2`: Electrical production from inverter 1 with fault labels (Ia, Ig, Eg, Fg, Pg, Va, Vg)
- `dt3`: Electrical production from inverter 2 (no faults)

**Fault classes:** Partial shading variants only (classes 0.0, 1.0, 2.1, 2.2, 2.3, 3.1, 3.2, 4.0). Limited fault diversity — all are shading-related.

**Class distribution:** ~97% normal (class 0), ~3% fault. Severe imbalance.

**Evaluable classes** (>= 3 segments for train/val/test): 3.1, 3.2, 4.0
**Train-only classes** (insufficient segments): 1.0, 2.1, 2.2, 2.3

**Relevance:** Primary dataset — rich meteorological context, continuous temporal coverage, real-world noise. Limitation is fault diversity (only shading). Best suited for anomaly detection (binary normal/fault) and forecasting (predicting fault onset). For multi-class classification, the limited evaluable classes constrain what can be claimed.

---

### 1.4 Mendeley PV Fault Dataset (Simulated)

| Property | Detail |
|---|---|
| **Source** | Commonly cited in PV fault detection literature |
| **Type** | Simulated (MATLAB/Simulink) |
| **Sampling rate** | 10 kHz |
| **Fault classes** | 8 classes (diverse fault types) |
| **Meteorological data** | None |

**Relevance:** Broad fault type coverage makes it useful for training classifiers. The simulation origin means no sensor noise, no meteorological variation, and no real-world non-stationarity. The absence of meteorological data is a significant limitation for transfer to real-world settings. Primary use case: sim-to-real transfer experiments paired with Costa or La Reunion as real-world validation targets.

---

### 1.5 Scientific Reports 2026 — 17-Class Fault Dataset

| Property | Detail |
|---|---|
| **Source** | Fault detection and diagnosis in PV systems using AI and time-frequency analysis |
| **Type** | Simulated |
| **Classes** | 17 (1 healthy + 16 fault types) |
| **Samples** | 12,835 (755 per class under clear/cloudy sky profiles) |
| **Features** | 5: solar irradiance, temperature, voltage at MPP, current at MPP, power at MPP |

**Fault types include:** Progressive short-circuit faults within a single string, pure partial-shading faults, combined inter-string short-circuit and asymmetric partial-shading patterns.

**Availability:** Check paper supplementary materials. Paper: https://www.nature.com/articles/s41598-026-39386-7

**Relevance:** Finest-grained fault taxonomy found (16 fault types). Small sample size (12,835 total) limits deep learning. Simulated data. Potentially useful as a fine-grained classification benchmark if the dataset is publicly released.

---

### 1.6 Kaggle — Fault Detection in Photovoltaic Farms

| Property | Detail |
|---|---|
| **Source** | Amr E. Rashed |
| **Type** | Simulated (MATLAB/Simulink, 250kW plant) |
| **Samples** | 600 training instances, 30 features each |
| **Fault classes** | 4: fault-free (100), string fault (153), string-to-ground (149), string-to-string (198) |
| **Format** | CSV |

**Relevance:** Very small dataset (600 samples). Limited utility for deep learning. May serve as a quick sanity-check benchmark for classical ML methods. Fault types are string-level electrical faults.

**Download:** https://www.kaggle.com/datasets/amrezzeldinrashed/fault-detection-dataset-in-photovoltaic-farms

**GitHub:** https://github.com/amrrashed/Fault-Detection-Dataset-in-Photovoltaic-Farms

---

### 1.7 IEEE DataPort — Partial Shading and Fault Simulation

| Property | Detail |
|---|---|
| **Source** | IEEE DataPort |
| **Type** | Simulated (LTSpice + Python, 2-diode model, gallium-based multijunction cells) |
| **Samples** | 6,965,234 scenarios |
| **Fault types** | Partial shading, open/short circuit, physical damage |
| **Format** | CSV (3 files, ~1.4 GB total) |
| **Access** | Requires IEEE DataPort subscription (paid) |

**Relevance:** Massive scale but paywalled and simulated. Gallium-based cells differ from silicon cells used in most real installations. Low priority unless subscription is available.

**URL:** https://ieee-dataport.org/documents/partial-shading-and-fault-simulation-dataset-photovoltaics-module

---

## 2. Unlabeled Performance & Meteorological Datasets

These datasets lack fault labels but contain continuous electrical performance and meteorological time-series. Useful for: unsupervised anomaly detection training (learn normal patterns), performance forecasting baselines, pretraining representations, and cross-site transfer experiments.

### 2.1 NIST PV Arrays and Weather Station

| Property | Detail |
|---|---|
| **Source** | National Institute of Standards and Technology |
| **Location** | Gaithersburg, Maryland, USA |
| **PV systems** | 3 grid-connected arrays (roof, parking lot, ground installations), monocrystalline silicon modules |
| **Duration** | 2015-2018 |
| **Resolution** | 1-second instantaneous + 1-minute averages (1-second data available on request) |
| **Measurements** | 360+ channels: irradiance (GHI, DHI, DNI), module temperature, ambient temperature, wind speed/direction, humidity, precipitation, air pressure, array voltage/current/power, IV curves (1-min) |
| **Format** | CSV |
| **Access** | Free (1-min averaged data downloadable; 1-second data via request to william.healy@nist.gov) |

**Relevance:** Highest-resolution publicly available PV performance dataset with comprehensive meteorological co-measurements. No fault labels, but performance deviations from expected output can serve as anomaly detection targets. 1-second resolution enables signal processing techniques not possible on La Reunion (7s) or Costa (1s averaged differently). Three different installation types (roof/parking/ground) within the same climate allow controlled comparison.

**Download:** https://catalog.data.gov/dataset/nist-campus-photovoltaic-pv-arrays-and-weather-station-data-sets-05b4d

**Paper:** Boyd, M.T. "Performance Data from the NIST Photovoltaic Arrays and Weather Station." *J. Res. NIST* 122, 40 (2017).

---

### 2.2 DKASC (Desert Knowledge Australia Solar Centre)

| Property | Detail |
|---|---|
| **Source** | Desert Knowledge Australia / ARENA |
| **Location** | Alice Springs, Central Australia (arid desert climate) |
| **PV systems** | 38 systems (2-10.5 kW), multiple technologies |
| **Duration** | 2009-2024 (15 years) |
| **Resolution** | Meteorological: 5-second (sampled at 1-second). Electrical: sub-hourly (higher resolution available on request) |
| **Measurements** | Active power (kW), temperature, relative humidity, global horizontal radiation (GHI), diffuse horizontal radiation (DHI) |
| **Access** | Free, open access via web portal |

**Relevance:** Longest temporal coverage of any publicly available PV dataset (15 years). Arid climate contrasts with La Reunion (tropical) and NIST (temperate), providing climate diversity for transfer learning experiments. Multiple PV technologies within a single site. Long-term degradation patterns visible across the 15-year span.

**Download:** https://dkasolarcentre.com.au/

---

### 2.3 PVDAQ (NREL Photovoltaic Data Acquisition)

| Property | Detail |
|---|---|
| **Source** | National Renewable Energy Laboratory |
| **Location** | 158 sites across the United States |
| **PV systems** | 158 systems (experimental and commercial) |
| **Duration** | 1-29 years depending on site |
| **Resolution** | Typically 15-minute averaged (varies by site) |
| **Measurements** | System performance (power, energy), irradiance, temperature, wind (varies by site) |
| **Access** | Free via OEDI portal and Python download scripts |

**Note:** The PVDAQ V3 API has been decommissioned. Current access is via the PVDAQ data map on OEDI and the GitHub-based download tools.

**Relevance:** Largest scale PV performance database. 158 systems across diverse US climates enable large-scale pretraining and cross-site generalization studies. Lower temporal resolution (15-min) limits signal processing applications but is adequate for power forecasting and coarse anomaly detection.

**Download:** https://data.openei.org/submissions/4568

**GitHub access tool:** https://github.com/openEDI/documentation/blob/main/pvdaq.md

---

### 2.4 Hong Kong HKUST Rooftop PV Dataset

| Property | Detail |
|---|---|
| **Source** | Hong Kong University of Science and Technology |
| **Location** | Sai Kung District, Hong Kong (22.3N, 114.3E) — subtropical coastal, urban |
| **PV systems** | 60 grid-connected rooftop stations (6,085 modules total, 2,230.8 kW combined capacity) |
| **Duration** | 2021-2023 (3 years) |
| **Resolution** | PV generation: 5-minute intervals (inverter-level). Weather: 1-minute intervals |
| **Measurements** | PV power generation per inverter; solar radiation, temperature, humidity, wind speed/direction, pressure (6 samplers on 10m weather tower) |
| **Stations** | 23 without optimizers, 37 with panel-level optimizers |
| **Data quality** | Grade A (missing rate < 10%), generation accuracy ~2.5%, weather uncertainty 1-10% |
| **Format** | CSV + Brick schema metadata (.ttl) with SPARQL query support |
| **Access** | Free, open source via Dryad repository |

**Relevance:** Large fleet (60 stations) in a single urban subtropical environment. 5-minute resolution with synchronized meteorological data. The multi-station setup enables inter-station comparison: if most stations produce normally but one drops, that's an anomaly signal without needing fault labels. Urban environment (shading from buildings, reflections) creates different challenge profile than rural/desert installations. Brick schema metadata enables smart querying.

**Download:** https://datadryad.org/dataset/doi:10.5061/dryad.m37pvmd99

**Paper:** "A high-resolution three-year dataset supporting rooftop photovoltaics (PV) generation analytics." *Scientific Data* (2025). DOI: [10.1038/s41597-025-04397-y](https://doi.org/10.1038/s41597-025-04397-y)

---

### 2.5 SKIPP'D (Stanford Sky Images and PV Power Generation Dataset)

| Property | Detail |
|---|---|
| **Source** | Stanford University, Environmental Assessment and Optimization Group |
| **Location** | Stanford Campus, California, USA |
| **PV system** | 30 kW rooftop array |
| **Duration** | 2017-2019 (3 years) |
| **Resolution** | 1-minute (power + sky images) |
| **Measurements** | PV power generation (kW). Sky images also available but not required for our use |
| **Format** | CSV (power), JPEG (images — optional) |
| **Access** | Free via Hugging Face, Stanford Digital Repository, GitHub |

**Relevance:** Established benchmark for short-term solar power forecasting. 1-minute PV power at 30kW scale. Well-documented with existing baselines in the literature, which enables direct comparison. The power-only time-series (ignoring images) can be used as a standard forecasting benchmark.

**Download:** https://github.com/yuhao-nie/Stanford-solar-forecasting-dataset

**Hugging Face:** https://huggingface.co/datasets/torchgeo/skippd

**Paper:** Nie, Y. et al. "SKIPP'D: a SKy Images and Photovoltaic Power Generation Dataset for Short-term Solar Forecasting." *Solar Energy* (2023). DOI: [10.1016/j.solener.2023.03.043](https://doi.org/10.1016/j.solener.2023.03.043)

---

### 2.6 DOE Regional Test Centers (DOE-RTC)

| Property | Detail |
|---|---|
| **Source** | Sandia National Laboratories / NREL (US Department of Energy) |
| **Locations** | Albuquerque, NM; Denver, CO; Las Vegas, NV; Orlando, FL (4 US climate zones) |
| **PV systems** | 8 identical c-Si systems across the 4 sites |
| **Duration** | 2014-2019 |
| **Resolution** | 1-minute performance data |
| **Measurements** | PV DC current/voltage, ground-based and satellite weather data |
| **Access** | Data available through DuraMAT Datahub (registration required) |

**Relevance:** Same hardware deployed across 4 different climates — the cleanest available setup for evaluating how climate affects PV performance and fault signatures. Ideal for transfer learning and domain adaptation experiments. 1-minute resolution is suitable for ML applications.

**Access:** https://datahub.duramat.org/ (search for "RTC" datasets)

**Note:** Data accessibility varies. Some RTC datasets require contacting Sandia directly. The DuraMAT Datahub is the primary portal.

---

### 2.7 Other Notable Datasets

| Dataset | Source | Location | Resolution | Duration | Access |
|---|---|---|---|---|---|
| **Elia Open Data** | Elia Group | Belgium, Germany | Varies | 2018-2024 | https://opendata.elia.be/ |
| **Chile GCPV** (BOKU) | Univ. of Natural Resources, Vienna | Chile | Hourly capacity factors | 2014-2018 | See paper ref [97] in Chen et al. |
| **INESC TEC** | INESC TEC, Portugal | Portugal | Hourly | 2011-2013 | See paper ref [99] in Chen et al. |
| **CWRU/UCF SunSmart** | Case Western / UCF | Florida, USA | Time series | 2012-2016 | See paper ref [102] in Chen et al. |
| **Sandia Spectral Irradiance** | Sandia National Labs | Albuquerque, NM | 5-minute | 2013-2021 | DuraMAT Datahub |

---

## 3. Dataset Comparison Matrix

### 3.1 By Task Suitability

| Dataset | Anomaly Detection | Classification | Forecasting | Transfer Learning |
|---|---|---|---|---|
| **La Reunion** | Primary (labeled) | Limited (shading only) | Primary (temporal) | Target domain |
| **Costa** | Secondary (labeled) | Primary (4 fault types) | Secondary (16 days) | Source/target |
| **Mendeley (sim)** | -- | Training source (8 classes) | -- | Source (sim-to-real) |
| **GPVS-Faults** | Tertiary (lab) | Supplementary (7 types) | -- | Source (lab-to-field) |
| **NIST** | Pretraining | -- | Enrichment | Source |
| **DKASC** | Pretraining | -- | Enrichment | Source (arid climate) |
| **PVDAQ** | Pretraining | -- | Enrichment | Source (scale) |
| **HK HKUST** | Enrichment (fleet) | -- | Secondary benchmark | Source (subtropical) |
| **SKIPP'D** | -- | -- | Benchmark | -- |

### 3.2 By Data Characteristics

| Dataset | Real/Sim | Resolution | Meteorological | Fault Labels | Samples | Climate |
|---|---|---|---|---|---|---|
| La Reunion | Real | ~7s | Full | Shading only | ~51M | Tropical |
| Costa | Real | 1 Hz | Irr + Temp | 4 types | ~516K | Tropical (Brazil) |
| GPVS-Faults | Real (lab) | ~100 kHz | No | 7 types | ~100K/file | Controlled |
| Mendeley (sim) | Simulated | 10 kHz | No | 8 classes | -- | N/A |
| NIST | Real | 1s / 1min | Full | No | 4 years | Temperate |
| DKASC | Real | 5s-hourly | Partial | No | 15 years | Arid |
| PVDAQ | Real | ~15 min | Varies | No | 158 sites | Mixed US |
| HK HKUST | Real | 5 min | Full | No | 3 years, 60 stations | Subtropical |
| SKIPP'D | Real | 1 min | No (images) | No | 3 years | Mediterranean |
| DOE-RTC | Real | 1 min | Full | No | 5 years, 4 climates | Multi-climate |

### 3.3 Sampling Rate Spectrum

```
100 kHz ──── GPVS-Faults (lab, grid-side measurements)
 10 kHz ──── Mendeley simulated
  1 Hz  ──── Costa (real field, labeled faults)
  ~0.14 Hz ── La Reunion (~7s, real field, labeled faults)
  1/60 Hz ─── NIST (1-min avg), SKIPP'D, DOE-RTC
  1/300 Hz ── HK HKUST (5-min), DKASC (sub-hourly)
  1/900 Hz ── PVDAQ (15-min avg)
```

This spectrum matters for feature engineering: signal processing techniques (wavelet decomposition, spectral analysis) become increasingly limited as sampling rate decreases. At 15-minute resolution (PVDAQ), only slow-dynamics features (daily patterns, degradation trends) are extractable.

---

## 4. Identified Gaps in the Dataset Landscape

1. **Fault-labeled time-series data is extremely scarce.** Only La Reunion and Costa provide real-world fault labels on electrical time-series with meteorological context. Most "fault detection" datasets are either simulated or image-based.

2. **No public dataset includes overheating fault labels** on electrical time-series. Overheating signatures must be either simulated or inferred from temperature-correlated performance drops.

3. **Cross-site labeled data does not exist.** No two labeled datasets share the same fault taxonomy, hardware configuration, or measurement protocol. Any cross-site transfer experiment must handle significant domain shift.

4. **Meteorological richness varies enormously.** La Reunion and NIST provide comprehensive weather context (irradiance components, temperature, humidity, wind, pressure). Costa provides only irradiance and module temperature. Mendeley and GPVS-Faults provide none.

5. **The sim-to-real gap is compounded by missing modalities.** Simulated datasets (Mendeley) lack meteorological channels entirely, making direct feature-level transfer to real datasets impossible without domain adaptation at the representation level.

---

## 5. Data Processing Tools

Relevant open-source tools for processing the datasets above (from Chen et al., 2025, Table 6):

| Function | Tool | Description |
|---|---|---|
| Time correction | pvlib | Solar time correction, timezone conversions, timestamp alignment |
| Time correction | solar-data-tools | Time shift correction, DST handling |
| Clear sky detection | pvlib | Compare measured vs modeled clear-sky irradiance |
| Clear sky detection | solar-data-tools | Irradiance pattern and variability analysis |
| Outlier detection | Rdtools | Multi-step clear sky workflow, remove outages/outliers |
| Outlier detection | PV-Pro | Detect outliers based on current vs irradiance, voltage vs temperature |
| Degradation analysis | Rdtools | Quantify I-V feature degradation including steps |
| Power forecasting | Analytics-Project | ANN model for power forecasting using NWP data |

**Data publication platforms (for contributing back):**

| Platform | URL | Owner |
|---|---|---|
| DuraMAT Datahub | https://datahub.duramat.org/ | DOE/NREL |
| Open Energy Data Initiative | https://data.openei.org/ | DOE |
| Zenodo | https://zenodo.org/ | European Commission |
| Dryad | https://datadryad.org/ | Independent |
| GitHub LFS | https://github.com/ (large file storage) | GitHub |

---

## 6. References

1. Chen, X. et al. "Open data sets for assessing photovoltaic system reliability." *Applied Energy* 395, 126132 (2025). [Preprint](https://engrxiv.org/preprint/view/4215)
2. Costa, C.H. et al. "A Monitoring System for Online Fault Detection and Classification in Photovoltaic Plants." *Sensors* 20(17), 4688 (2020). [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7506914/)
3. Boyd, M.T. "Performance Data from the NIST Photovoltaic Arrays and Weather Station." *J. Res. NIST* 122, 40 (2017). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7339723/)
4. Nie, Y. et al. "SKIPP'D: a SKy Images and Photovoltaic Power Generation Dataset for Short-term Solar Forecasting." *Solar Energy* (2023). [arXiv](https://arxiv.org/abs/2207.00913)
5. "A high-resolution three-year dataset supporting rooftop photovoltaics (PV) generation analytics." *Scientific Data* (2025). [Nature](https://www.nature.com/articles/s41597-025-04397-y)
6. "Fault detection and diagnosis in photovoltaic systems using artificial intelligence and time-frequency analysis." *Scientific Reports* (2026). [Nature](https://www.nature.com/articles/s41598-026-39386-7)
7. Jovicic, A. et al. "GPVS-Faults: Experimental Data for fault scenarios in grid-connected PV systems under MPPT and IPPT modes." Mendeley Data (2020). [Dataset](https://data.mendeley.com/datasets/n76t439f65/1)
