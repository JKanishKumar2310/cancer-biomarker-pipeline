# 🧬 Multi-Cancer Biomarker Discovery & Translational Exploration Platform

[![CI Pipeline & Smoke Test](https://github.com/JKanishKumar2310/cancer-biomarker-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/JKanishKumar2310/cancer-biomarker-pipeline/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: Plotly Dash](https://img.shields.io/badge/Dashboard-Plotly%20Dash-orange.svg)](https://dash.plotly.com/)

An open-source, end-to-end computational biology pipeline combining **patient-paired differential expression analysis**, a **leak-free multi-model machine learning ensemble** (Random Forest, Gradient Boosting, L1-penalized sparse regression with `StratifiedGroupKFold`), **autonomous NCBI GEO dataset curation**, **disease-matched zero-shot external validation**, and **prognostic survival modeling** (Kaplan-Meier, 95% CI Hazard Ratios, Cox Proportional Hazards regression, and Benjamini-Hochberg FDR control).

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [Key Features](#key-features)
- [Multi-Cancer Supported Cohorts](#multi-cancer-supported-cohorts)
- [Statistical & Machine Learning Architecture](#statistical--machine-learning-architecture)
- [AI Autonomous GEO Ingestion & Curation](#ai-autonomous-geo-ingestion--curation)
- [Project Architecture](#project-architecture)
- [Installation & Setup](#installation--setup)
- [Execution & Quickstart](#execution--quickstart)
- [Continuous Integration & Smoke Testing](#continuous-integration--smoke-testing)
- [In Silico Multi-Cohort Validation Results](#in-silico-multi-cohort-validation-results)
- [Targeted Drug Sensitivity Mapping](#targeted-drug-sensitivity-mapping)
- [Scientific Limitations & Intended Use](#scientific-limitations--intended-use)
- [References](#references)

---

## Abstract

Cancer biomarker identification requires rigorous statistical modeling to distinguish true biological signals from technical batch artifacts and patient-level confounding. This platform provides an exploratory computational framework that:

1. **Patient-Paired Discovery:** Analyzes intra-patient paired tumor/normal microarrays across major human malignancies (**Colorectal Adenoma/Carcinoma**, **Lung Adenocarcinoma**, and **Breast Carcinoma**) using paired Student's t-tests.
2. **Leak-Free ML Ensemble:** Deploys Random Forest (bagging), Gradient Boosting (boosting), and L1-penalized Logistic Regression (sparse Lasso feature selection) with fold-local scaling and patient-grouped cross-validation (`StratifiedGroupKFold`), ensuring held-out samples never influence transformations.
3. **Disease-Matched External Validation:** Evaluates locked consensus biomarker signatures on independent international cohorts without retraining, recording full data provenance.
4. **Prognostic Survival Modeling:** Associates biomarker expression with 10-year overall and relapse-free survival using non-parametric Kaplan-Meier analysis, Peto log-rank Hazard Ratios with 95% Confidence Intervals, continuous Cox Proportional Hazards regression (`statsmodels PHReg`), and Benjamini-Hochberg FDR correction.
5. **Autonomous Dataset Ingestion:** Combines regex header parsing and pathology ontology matching to ingest and classify public NCBI GEO series matrices on demand.
6. **Translational Drug Matching:** Maps consensus biomarker candidates to FDA-approved therapies, kinase inhibitors, and antibody-drug conjugates (ADCs).

---

## Key Features

| Feature | Biological & Technical Function |
|---|---|
| 🤖 **AI Oncologist Copilot** | Interactive real-time research assistant grounded in pipeline telemetry (biomarkers, pathways, survival, drugs) |
| ⏳ **Prognostic Survival Analysis** | 10-year overall & relapse-free survival with Log-Rank tests, 95% CI Hazard Ratios, Cox PH models, and BH-FDR |
| 🌐 **Zero-Shot External Validation** | Generalization testing on independent disease-matched international cohorts with full data provenance tracking |
| 🧬 **Patient-Aware Paired Design** | Intra-patient paired Student's t-test and `StratifiedGroupKFold` preventing intra-patient data leakage |
| 🌲 **Multi-Model ML Ensemble** | Random Forest + Gradient Boosting + L1-Logistic Regression feature consensus with fold-local scaling |
| 💊 **Targeted Drug Actionability** | Direct cross-referencing against FDA oncology approvals, OncoKB, and NCCN guidelines |
| 🎯 **Dual-Filter Consensus** | Requires convergence of both statistical FDR (< 0.05) and multi-model ML feature weights |
| 🌋 **Interactive Volcano Plot** | Real-time fold-change & p-value slider adjustment with clickable gene detail boxplots |
| 🔥 **Expression Heatmap** | Clustered Z-score expression view of top differentially expressed genes |
| 📊 **High-Dimensional PCA** | Variance-retained principal component clustering showing tumor vs. normal separation |
| 🧬 **Pathway Enrichment** | GO Biological Process and KEGG pathway enrichment via Enrichr |
| 🌙 **Glassmorphic Dark UI** | 7-tab Plotly Dash dashboard with responsive layout, animated stats, and dynamic re-indexing |

---

## Multi-Cancer Supported Cohorts

The pipeline includes built-in configurations and disease-matched validation cohorts across 3 major malignancies:

| Cancer Type | Discovery Cohort (Paired) | External Validation Cohort (Independent) | Survival Cohort (Clinical Outcomes) | Platform |
|---|---|---|---|---|
| **Colorectal Adenoma / Carcinoma** | **GSE8671** (n=64: 32 adenomas + 32 matched normal mucosa) | **GSE20916** (n=145: 101 adenomas/carcinomas + 44 normal mucosa) | **GSE39582** (French CIT Cohort, n=566, 10-yr OS & RFS) | Affymetrix GPL570 |
| **Lung Adenocarcinoma (LUAD)** | **GSE19804** (n=120: 60 LUAD tumors + 60 paired normals) | **GSE18842** (Europe, n=91: 46 tumors + 45 normals) | **GSE31210** (Japan, n=226, 10-yr follow-up) | Affymetrix GPL570 |
| **Breast Carcinoma (BRCA)** | **GSE15852** (n=86: 43 tumors + 43 paired normals) | **GSE42568** (Europe, n=121: 104 tumors + 17 normals) | **GSE1456** (Stockholm, n=159, 10-yr follow-up) | Affymetrix GPL96 |

---

## Statistical & Machine Learning Architecture

### 1. Patient-Aware Paired Experimental Design
Microarray discovery cohorts (`GSE8671`, `GSE19804`, `GSE15852`) comprise 1:1 matched tumor and adjacent normal tissues extracted from the same surgical patient. 
- **Paired Student's t-test (`scipy.stats.ttest_rel`):** Calculates differences $\Delta_i = X_{i, \text{tumor}} - X_{i, \text{normal}}$ per patient, eliminating inter-individual genetic background variance.
- Unpaired cohorts automatically fall back to Welch's t-test with unequal variance assumptions.
- P-values are adjusted across all genome features using the Benjamini-Hochberg False Discovery Rate (FDR).

### 2. Leak-Free Cross-Validation (`StratifiedGroupKFold`)
To prevent optimistic performance estimates and patient identity leakage:
- **Grouped Folds:** Normal and tumor samples from the same patient ID are strictly assigned to the same cross-validation fold via `StratifiedGroupKFold`.
- **Fold-Local Transformations:** Feature scaling (`StandardScaler`) is fitted strictly on the training partition of each fold using `sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('l1', ...)])`, ensuring test partitions remain unobserved.
- **Consensus Signature Evaluation:** In addition to component model evaluation, the final selected multi-gene consensus signature is evaluated out-of-fold under cross-validation.

### 3. Multi-Model ML Ensemble
Three complementary model architectures rank candidate genes:
1. **Random Forest (Bagging):** Captures non-linear gene interactions ($n=200$ trees, balanced class weights).
2. **Gradient Boosting (Boosting):** Sequential gradient loss minimization focusing on borderline samples.
3. **L1-Penalized Logistic Regression (Lasso / Sparse Selection):** Shrinks non-informative coefficients to zero:

$$\text{Ensemble Score} = 0.45 \cdot \text{Norm}(\text{RF}) + 0.35 \cdot \text{Norm}(\text{GB}) + 0.20 \cdot \text{Norm}(\text{L1})$$

### 4. Rigorous Clinical Survival Modeling
Biomarkers are evaluated against long-term patient outcomes (overall survival and relapse-free survival):
- **Kaplan-Meier Curves:** Non-parametric cumulative survival $S(t) = \prod_{t_i \le t} (1 - d_i / n_i)$.
- **Peto Log-Rank Hazard Ratio & 95% CI:** Standard error $\text{SE} = \sqrt{1 / V_{\text{total}}}$, yielding $\text{HR}_{95\% \text{ CI}} = \exp(\ln(\text{HR}) \pm 1.96 \cdot \text{SE})$.
- **Cox Proportional Hazards Regression:** Continuous model fitting via `statsmodels.duration.hazard_regression.PHReg`.
- **Multiple Testing Correction:** Benjamini-Hochberg FDR control applied across all candidate biomarker survival tests.

### 5. Provenance Tracking & Synthetic Isolation
- All cross-cohort validation outputs record source cohort accessions, sample counts, tested gene lists, and data provenance in `results/external_validation_metrics.csv`.
- Offline or CI smoke tests run with `--test-mode` are strictly segregated into `results/synthetic_test/` and never overwrite authentic experimental results.
- The pipeline fails explicitly with clear diagnostic errors if required validation data cannot be retrieved.

---

## AI Autonomous GEO Ingestion & Curation

The platform features an autonomous ingestion layer ([src/ai_geo_curator.py](src/ai_geo_curator.py)):
- Fetches NCBI GEO series matrix metadata headers on demand via HTTPS.
- Utilizes structured regex parsing combined with pathology ontology keyword mapping to distinguish comparative disease groups.
- Automatically resolves platform probe annotations (`GPL570`, `GPL96`).
- Accessible live directly through the **Dashboard Ingestion Bar**.

---

## 🤖 Interactive AI Oncologist Copilot Chatbot

Tab 7 provides an integrated translational AI research assistant ([src/ai_copilot.py](src/ai_copilot.py)) connected to pipeline telemetry:
- **Grounded Telemetry:** Feeds real-time consensus biomarkers, fold-changes, FDR values, survival hazard ratios, enriched pathways, and targeted therapeutics into the reasoning context.
- **Translational Reasoning:** Explains biological mechanisms of action, downstream pathways, and clinical indications.
- **Wet-Lab Validation Protocols:** Suggests experimental assays (RT-qPCR, Western blot, immunohistochemistry, CRISPR knockdown, cell viability assays) to guide laboratory verification.
- **One-Click Inquiry Chips:** Instant prompts for common queries (Summarize Findings, Drug Opportunities, Biomarker Significance, Survival Prognosis, Experimental Protocols).

---

## Project Architecture

```
cancer-biomarker-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml                   # Automated GitHub Actions CI workflow
├── config.py                        # Central configuration (parameters, thresholds, cohorts)
├── requirements.txt                 # Pinned dependencies
├── run_analysis.py                  # Full computational pipeline CLI orchestrator
├── run_dashboard.py                 # Interactive Plotly Dash dashboard server
├── data/                            # Cached GEO series matrices & annotations (git-ignored)
├── results/                         # Authentic experimental results, survival metrics, ROC curves
│   └── synthetic_test/              # Isolated outputs from CI / unit test runs
│
├── src/
│   ├── data_loader.py               # GEO series matrix parser with patient metadata extraction
│   ├── preprocessing.py             # Quantile normalization, log2 transform, variance filter
│   ├── differential_expression.py   # Patient-paired Student's t-test + BH-FDR correction
│   ├── ml_biomarkers.py             # Multi-model ensemble with StratifiedGroupKFold & fold-local scaling
│   ├── survival_analysis.py         # Kaplan-Meier, 95% CI Log-Rank, Cox PH regression, BH-FDR
│   ├── external_validation.py       # Disease-matched zero-shot external validation & provenance
│   ├── drug_mapping.py              # FDA oncology targeted therapy matching
│   ├── pathway_analysis.py          # GO & KEGG enrichment analysis via Enrichr
│   ├── ai_geo_curator.py            # AI NCBI metadata parser & sample classifier
│   ├── ai_annotator.py              # Biological context generator
│   ├── ai_copilot.py                # Interactive AI Oncologist reasoning engine
│   └── utils.py                     # Logging & helpers
│
└── dashboard/
    ├── app.py                       # Dash application entry point
    ├── layout.py                    # 7-Tab Glassmorphic UI layout & AI Copilot interface
    ├── callbacks.py                 # Interactive Plotly callbacks & live re-indexing
    └── assets/
        └── style.css                # Custom styling & CSS animations
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JKanishKumar2310/cancer-biomarker-pipeline.git
   cd cancer-biomarker-pipeline
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution & Quickstart

### Option A: Run Full Analysis Pipeline (CLI)
```bash
python run_analysis.py
```
This automatically executes all 8 stages:
1. Load & map discovery data with patient metadata
2. Preprocess with quantile normalization & variance filtering
3. Differential expression (patient-paired Student's t-test + FDR)
4. Multi-model ML ensemble ranking with `StratifiedGroupKFold`
5. Pathway enrichment analysis (GO + KEGG)
6. Clinical survival analysis (Kaplan-Meier, 95% CI HR, Cox PH, FDR)
7. Disease-matched cross-cohort external validation
8. Targeted drug actionability mapping

### Option B: Launch Interactive Web Dashboard
```bash
python run_dashboard.py
```
Open **`http://127.0.0.1:8050`** in your browser.

---

## Continuous Integration & Smoke Testing

This repository includes an automated GitHub Actions CI workflow ([.github/workflows/ci.yml](.github/workflows/ci.yml)) that verifies:
- Syntax compilation across all modules.
- End-to-end execution of all pipeline steps in test mode without requiring gigabytes of remote data download:
  ```bash
  python run_analysis.py --test-mode
  ```
- Strict output isolation: test runs write solely to `results/synthetic_test/`.
- Headless initialization of the Dash web server.

---

## In Silico Multi-Cohort Validation Results

### Benchmark Summary Across Three Independent Cohorts:

| Cohort | Cancer Type | Tested Signature | External Accuracy | External ROC-AUC | Top Prognostic Hits (OS $p < 0.05$) |
|---|---|---|---|---|---|
| **GSE8671** | Colorectal Adenoma | 20 genes | **77.9%** (GSE20916) | **0.9956** | `CDX2` ($p=7.6\times 10^{-4}$, FDR=0.041, HR=0.62 [95% CI 0.46-0.82]), `LGR5`, `MKI67` (GSE39582, n=566) |
| **GSE19804** | Lung Adenocarcinoma | 20 genes | **72.5%** (GSE18842) | **0.9406** | `ARRB1` (Protective, HR=0.26), `TOP2A`, `EPCAM` (GSE31210, n=226) |
| **GSE15852** | Breast Carcinoma | 20 genes | **95.9%** (GSE42568) | **0.9695** | `MELK` ($p=2.7\times 10^{-4}$, HR=3.48), `PTEN` (GSE1456, n=159) |

---

## Targeted Drug Sensitivity Mapping

Identified biomarkers are automatically matched to FDA-approved therapeutic agents:
- **`EGFR`:** Osimertinib, Gefitinib, Erlotinib (1st/2nd/3rd gen TKIs in NSCLC); Cetuximab, Panitumumab in Colorectal.
- **`KRAS`:** Sotorasib, Adagrasib (Covalent KRAS G12C inhibitors).
- **`BRAF`:** Encorafenib, Dabrafenib + Trametinib (V600E targeted combinations).
- **`VEGFA`:** Bevacizumab (Anti-angiogenic monoclonal antibody).
- **`TOP2A`:** Etoposide, Irinotecan (Topoisomerase cleavage complexes).
- **`TACSTD2` (TROP2):** Sacituzumab govitecan, Datopotamab deruxtecan (Antibody-Drug Conjugates).

---

## Scientific Limitations & Intended Use

1. **Retrospective In Silico Nature:** All analyses performed by this pipeline represent exploratory in silico bioinformatic modeling on retrospective public datasets.
2. **Pre-Clinical Status:** Candidate biomarkers and predicted therapeutic vulnerabilities have not been validated in prospective clinical trials and should not be used as clinical diagnostic or treatment decision tools.
3. **Array Platform Boundaries:** Microarray probe intensity hybridization dynamics, background cross-hybridization, and platform-specific probe sets can introduce variance between different array technologies.
4. **Experimental Requirement:** Prior to translational claims, candidate gene signatures require orthogonal bench validation (RT-qPCR, Western blotting, immunohistochemistry, and functional genetic perturbation models).

---

## Author & Citation

- **Author:** Kanish ([@JKanishKumar2310](https://github.com/JKanishKumar2310))
- **License:** MIT License
