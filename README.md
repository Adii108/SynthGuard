# 🛡️ SynthGuard: Generative AI for Synthetic Tabular Data Generation, Evaluation & Privacy Risk Analysis

> **"Generate realistic synthetic tabular data while evaluating its utility and privacy risks."**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 20 Passed](https://img.shields.io/badge/Tests-20%20Passed-brightgreen.svg)](#-how-to-run-the-project)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](app.py)

---

## 📌 1. Project Overview
**SynthGuard** is an enterprise-grade Generative AI and Data Science platform developed to generate high-fidelity synthetic tabular datasets while systematically measuring and balancing the trade-off between **Data Utility** and **Empirical Privacy Risk**.

Traditional synthetic data workflows either produce unrealistic data (destroying multi-variable correlations) or suffer from severe privacy leakage (memorization of sensitive training rows). SynthGuard solves this dilemma through **natural language intent understanding (LLMs)**, **probabilistic & deep generative modeling (CTGAN, TVAE, Gaussian Copula)**, and an **in-depth statistical, ML utility (TSTR), and geometric privacy risk evaluation suite**.

---

## 🎯 2. Problem Statement
1. **Privacy vs. Utility Dilemma**: Organizations need to share and analyze tabular data without leaking sensitive or personally identifiable information (PII).
2. **Complex Multi-Modal Distributions**: Tabular datasets contain mixed types (continuous, skewed, discrete, multi-class categorical, binary) and complex non-linear correlations that standard random samplers cannot preserve.
3. **Lack of Automated Evaluation**: Data teams struggle to verify if synthetic data preserves machine learning downstream performance (*Train on Synthetic, Test on Real*) and whether records are memorized.
4. **Accessible GenAI Interface**: Non-technical domain experts need natural language interfaces to specify generation goals, sensitive column priorities, and row counts.

---

## ⚙️ 3. How SynthGuard Works
SynthGuard orchestrates a modular pipeline:
1. **Validates & Profiles** raw CSV datasets, automatically inferring data types, cardinality, and distribution parameters.
2. **Translates Natural Language Requests** using LLM/semantic NLP into structured generation parameters (target column, priority features, record counts, privacy posture).
3. **Generates Synthetic Data** using modern tabular generative architectures (CTGAN with Mode-Specific Normalization, TVAE with Latent Manifold representations, or Gaussian Copula).
4. **Evaluates Distribution & Correlation Fidelity** via two-sample Kolmogorov-Smirnov tests, 1-Wasserstein distances, Total Variation Distance (TVD), and Pearson/Cramer's V correlation matrices.
5. **Benchmarks ML Utility** via the rigorous **TSTR (Train on Synthetic / Test on Real)** framework compared against empirical **TRTR (Train on Real / Test on Real)** baselines.
6. **Quantifies Empirical Privacy Risk** using Exact Duplicate Hash checks, Distance to Closest Record (DCR), Nearest Neighbor Distance Ratio (NNDR), and Memorization Outlier detection.
7. **Compiles an AI Audit Report** grounded strictly on computed metrics without hallucinations.

---

## 🔄 4. Complete Workflow

```
┌────────────────────────────────────────────────────────┐
│                   1. DATASET INGESTION                 │
│  - CSV Upload & Validation (Format, Headers, Bounds)   │
│  - Inferred Semantic Types & Missing Value Profiling   │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│                 2. DATASET PROFILING                   │
│  - Descriptive Stats (Mean, Std, Quantiles, Skewness)  │
│  - Categorical Distributions & Cardinality Checks      │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│             3. NATURAL LANGUAGE INTENT (LLM)           │
│  - Prompt Parsing (Target Columns, Sample Size, Goals) │
│  - Automated Generation Parameter Configuration        │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│           4. SYNTHETIC DATA GENERATION ENGINE          │
│  - CTGAN (Conditional Tabular GAN Architecture)        │
│  - TVAE (Tabular Variational Autoencoder)              │
│  - Gaussian Copula (Probabilistic Multivariate Base)   │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│                 5. COMPREHENSIVE EVALUATION            │
│  ├─ Statistical Similarity (KS-Test, Wasserstein, TVD) │
│  ├─ Correlation Preservation (Pearson/Spearman/Cramer) │
│  ├─ ML Utility Benchmark (TSTR - Train Synth/Test Real)│
│  └─ Empirical Privacy Risk (Exact Match, DCR, NNDR)    │
└──────────────────────────┬─────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────┐
│          6. AI REPORT & ARTIFACT EXPORT ENGINE         │
│  - Context-Aware Executive & Data Science Summary      │
│  - CSV Export & Downloadable Evaluation Report         │
└────────────────────────────────────────────────────────┘
```

---

## 🏗️ 5. Project Architecture & Code Structure

```
SynthGuard/
├── app.py                      # Main Streamlit Dashboard Application
├── README.md                   # Comprehensive documentation & Phase tracker
├── requirements.txt            # Python dependencies
├── generate_sample_data.py     # Benchmark dataset creator
├── synthguard/                 # Core modular package
│   ├── __init__.py
│   ├── profiling/              # Phase 1: Ingestion & Profiling
│   │   ├── validator.py        # Dataset validation & error handling
│   │   └── profiler.py         # Descriptive stats & column typing
│   ├── generators/             # Phase 2: Synthetic Data Generation
│   │   ├── base.py             # Abstract generator interface
│   │   ├── ctgan_generator.py  # CTGAN with Mode-Specific Normalization (VGM)
│   │   ├── tvae_generator.py   # Tabular Variational Autoencoder
│   │   └── copula_generator.py # Multivariate Gaussian Copula
│   ├── evaluation/             # Phase 3: Evaluation Suite
│   │   ├── statistical.py      # KS-Test, Wasserstein, Total Variation Distance
│   │   ├── correlation.py      # Pearson/Spearman, Cramer's V, Frobenius Norm
│   │   ├── ml_utility.py       # TSTR vs TRTR downstream ML benchmark
│   │   └── privacy.py          # Exact match, DCR, NNDR, memorization risk
│   ├── nlp/                    # Phase 4: Natural Language Interface
│   │   ├── intent_parser.py    # Rule-guided & semantic NLP intent parser
│   │   └── llm_service.py      # Gemini/OpenAI API + deterministic fallback
│   └── reporting/              # Phase 5: Automated Reporting
│       ├── pipeline.py         # End-to-end evaluation orchestrator
│       └── report_generator.py # Factual AI Report generator (MD / HTML)
├── data/                       # Built-in demo benchmark datasets
│   ├── employee_attrition.csv  # HR Analytics
│   ├── customer_churn.csv      # Telecom Customer Churn
│   └── heart_disease.csv       # Medical Risk
└── tests/                      # Automated test suite (20 tests)
    ├── test_profiler.py
    ├── test_generators.py
    ├── test_evaluators.py
    ├── test_nlp.py
    ├── test_pipeline.py
    └── test_end_to_end.py
```

---

## 🛠️ 6. Technologies Used
- **Core Language**: Python 3.10+
- **Frontend / Dashboard**: Streamlit, Plotly Express & Graph Objects
- **Data & Numerical Science**: Pandas, NumPy, SciPy, Statsmodels
- **Machine Learning & Modeling**: Scikit-Learn, Random Forests, Gradient Boosting, Gaussian Mixtures
- **LLM & Semantic Engine**: Google Gemini / OpenAI API + Built-in Deterministic Parser
- **Testing**: Pytest

---

## 📈 7. Phase-wise Development & Current Progress

### Phase 1 — Dataset Ingestion & Profiling
**Status: Completed ✅**
- [x] Basic modular project structure (`synthguard/`, `tests/`, `data/`)
- [x] Robust CSV dataset validation (`DatasetValidator`)
- [x] Semantic column typing & missing value/duplicate detection
- [x] Detailed numerical and categorical statistical profiling (`DatasetProfiler`)
- [x] Benchmark datasets (`employee_attrition.csv`, `customer_churn.csv`, `heart_disease.csv`)
- [x] Unit test suite for validation & profiling (`tests/test_profiler.py`)

### Phase 2 — Synthetic Data Generation
**Status: Completed ✅**
- [x] Base generator interface (`BaseTabularGenerator`)
- [x] CTGAN (Conditional Tabular GAN) with Mode-Specific Normalization (VGM)
- [x] TVAE (Tabular Variational Autoencoder) with latent Gaussian manifold
- [x] Gaussian Copula probabilistic baseline generator
- [x] Deterministic seed configuration and sampling verification
- [x] Unit tests for all generators (`tests/test_generators.py`)

### Phase 3 — Comprehensive Evaluation Suite
**Status: Completed ✅**
- [x] Statistical similarity metrics (Kolmogorov-Smirnov Test, 1-Wasserstein, Total Variation Distance)
- [x] Correlation preservation, Frobenius norm difference & Cramer's V association
- [x] ML Utility benchmark: TSTR (Train on Synthetic, Test on Real) vs TRTR (Train on Real, Test on Real) for Classification & Regression
- [x] Empirical Privacy Risk (Exact duplicates, Distance to Closest Record - DCR, Nearest Neighbor Distance Ratio - NNDR, Memorization score)
- [x] Unit tests for all evaluators (`tests/test_evaluators.py`)

### Phase 4 — Natural Language GenAI Interface
**Status: Completed ✅**
- [x] Natural language prompt parser (`LocalIntentParser`)
- [x] Automated generation parameter extraction (sample sizes, priority columns, target objectives, privacy constraints)
- [x] LLM integration (Gemini / OpenAI API + deterministic local fallback)
- [x] Interactive configuration preview & parameter validation
- [x] Unit tests for natural language parsing (`tests/test_nlp.py`)

### Phase 5 — Automated Evaluation & AI Reporting
**Status: Completed ✅**
- [x] End-to-end evaluation pipeline orchestrator (`EvaluationPipeline`)
- [x] Structured evaluation summary aggregation
- [x] AI-generated executive & data science report (grounded strictly in computed metrics, avoiding hallucinations)
- [x] Multi-format report export (Markdown, HTML, text summary)
- [x] Unit & integration tests for pipeline & reporting (`tests/test_pipeline.py`)

### Phase 6 — Streamlit UI / UX
**Status: Completed ✅**
- [x] Clean, professional multi-stage dashboard layout (`app.py`)
- [x] Ingestion & validation interface with CSV upload and built-in demo datasets
- [x] Natural language generation studio with real-time intent configuration & reasoning
- [x] Multi-tab evaluation suite (Distributions, Correlation heatmaps, TSTR ML benchmark, Privacy risk & DCR)
- [x] Live AI Report viewer with Markdown and HTML export options
- [x] Downloadable synthetic CSV and audit reports

### Phase 7 — Testing & Final Refinement
**Status: Completed ✅**
- [x] End-to-end multi-dataset integration test suite (`tests/test_end_to_end.py`)
- [x] Edge-case verification (pure numerical, pure categorical, binary targets, skewed continuous variables)
- [x] 100% test pass rate across 20 pytest suites
- [x] Comprehensive documentation, viva preparation guides & architecture walkthrough

---

## 💻 8. How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/SynthGuard.git
cd SynthGuard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run unit & integration tests
```bash
python -m pytest -v
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 💡 9. Example Usage

### Step 1: Upload / Load Dataset
Select the built-in **Employee Attrition** demo dataset from the Dashboard. The profiler automatically discovers 600 records across 7 features (`Age`, `JobRole`, `MonthlyIncome`, `YearsAtCompany`, `WorkLifeBalance`, `OverTime`, `Attrition`).

### Step 2: Natural Language Request
Enter your prompt:
> *"Generate 1000 synthetic employee records similar to this dataset while preserving the relationships between Age, MonthlyIncome, JobRole and Attrition."*

The AI parses this into:
- **Target Column**: `Attrition`
- **Focus Columns**: `Age, MonthlyIncome, JobRole, Attrition`
- **Record Count**: `1,000`
- **Recommended Model**: `CTGAN`

### Step 3: Generation & Evaluation
Click **"Train Model & Generate Synthetic Dataset"**. The system fits the model and executes the full evaluation pipeline, rendering:
- Marginal distribution overlays (Continuous histograms + Categorical bar proportions)
- Real vs Synthetic Correlation heatmaps
- TSTR vs TRTR ML predictive metrics (e.g. 84.5% Utility Retention)
- Privacy Risk metrics (0 exact matches, 5th percentile DCR = 0.281, Low Observed Risk)

### Step 4: Export Audit Report
Download the complete formatted **Markdown (.md)** or **HTML** audit report and the generated **CSV dataset**.

---

## 🔬 10. Evaluation Methods

1. **Marginal Distribution Fidelity**:
   - **Two-Sample Kolmogorov-Smirnov (KS) Test**: Quantifies the maximum vertical distance between empirical CDFs of real and synthetic continuous variables.
   - **1-Wasserstein Distance**: Measures the minimum "work" required to transform the synthetic distribution into the real distribution.
   - **Total Variation Distance (TVD)**: Computes the divergence between categorical probability mass functions:
     $$\text{TVD}(P, Q) = \frac{1}{2} \sum_{x} |P(x) - Q(x)|$$

2. **Correlation & Feature Interactions**:
   - **Pearson & Spearman Correlation Matrices**: Compares linear and monotonic feature relationships.
   - **Cramer's V Association**: Evaluates relationship strengths between categorical columns.
   - **Matrix Frobenius Norm Distance**: Quantifies overall structural drift across feature interaction space.

3. **Machine Learning Utility (TSTR vs TRTR)**:
   - **TSTR (Train on Synthetic, Test on Real)**: Trains downstream models (Random Forests, Gradient Boosting) exclusively on synthetic data and tests on hold-out real data.
   - **TRTR (Train on Real, Test on Real)**: Baseline model trained on real data.
   - Compares Accuracy, Precision, Recall, F1-Score (Classification) and $R^2$, RMSE, MAE (Regression).

---

## 🔒 11. Privacy Checks

SynthGuard avoids superficial binary "Safe/Unsafe" claims and provides nuanced empirical privacy diagnostics:
- **Exact Duplicate Check**: Hashes all records to identify identical rows copied from training data.
- **Distance to Closest Record (DCR)**: Measures Euclidean distance in normalized feature space from every synthetic record to its nearest real neighbor.
- **Nearest Neighbor Distance Ratio (NNDR)**: Computes $d_1 / d_2$ (ratio of distance to 1st nearest neighbor vs 2nd nearest neighbor). Values approaching zero indicate isolated training set memorization.
- **Memorization Outlier Detection**: Identifies sparse sub-population points that may leak sensitive attributes.

---

## ⚠️ 12. Limitations
1. **Empirical vs. Mathematical Privacy**: While low DCR and zero exact duplicates confirm lack of direct memorization, it does not replace formal $(\epsilon, \delta)$-Differential Privacy mathematical guarantees.
2. **Rare Sub-populations**: Extreme tail distributions with very low sample counts ($<5$ occurrences) may experience sampling variance.
3. **High-Order Multi-variable Interactions**: Beyond 3-way interactions, neural and copula samplers may introduce mild smoothing.

---

## 🔮 13. Future Improvements
- [ ] Implement $(\epsilon, \delta)$-Differential Privacy noise mechanisms (DP-SGD & Gaussian DP mechanisms).
- [ ] Support sequential and time-series tabular data generation.
- [ ] Multi-table relational database synthesis with foreign-key constraint preservation.
- [ ] Integration with cloud data warehouses (Snowflake, BigQuery).
