# 🛡️ SynthGuard: Generative AI for Synthetic Tabular Data Generation, Evaluation & Privacy Risk Analysis

> **"Generate realistic synthetic tabular data while evaluating its utility and privacy risks."**

---

## 📌 Project Overview
**SynthGuard** is an end-to-end Generative AI and Data Science platform designed to generate high-fidelity synthetic tabular datasets while systematically quantifying the trade-off between **Data Utility** and **Privacy Risk**.

Traditional synthetic data workflows either produce unrealistic data or suffer from dangerous privacy leakage (training data memorization). SynthGuard combines **natural language intent understanding (LLMs)**, **probabilistic & deep generative modeling (CTGAN, TVAE, Gaussian Copula)**, and an **in-depth statistical, ML utility, and empirical privacy risk evaluation suite**.

---

## 🎯 Problem Statement
1. **Privacy vs. Utility Dilemma**: Organizations need to share and analyze tabular data without leaking sensitive or personally identifiable information (PII).
2. **Complex Multi-Modal Distributions**: Tabular datasets contain mixed types (continuous, skewed, discrete, multi-class categorical, binary) and complex non-linear correlations that standard random samplers cannot preserve.
3. **Lack of Automated Evaluation**: Data teams struggle to verify if synthetic data preserves machine learning downstream performance (Train on Synthetic, Test on Real) and whether records are memorized.
4. **Accessible GenAI Interface**: Non-technical domain experts need natural language interfaces to specify generation goals, sensitive column priorities, and row counts.

---

## 🔄 Complete Workflow Architecture

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

## 🚀 Phase-wise Development Progress

### Phase 1 — Dataset Ingestion & Profiling
**Status: Completed ✅**
- [x] Basic modular project structure (`synthguard/`, `tests/`, `data/`)
- [x] Robust CSV dataset validation (`DatasetValidator`)
- [x] Semantic column typing & missing value/duplicate detection
- [x] Detailed numerical and categorical statistical profiling (`DatasetProfiler`)
- [x] Realistic sample benchmark datasets (`employee_attrition.csv`, `customer_churn.csv`, `heart_disease.csv`)
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
**Status: In Progress ⏳**
- [ ] End-to-end evaluation pipeline orchestrator (`EvaluationPipeline`)
- [ ] Structured evaluation summary aggregation
- [ ] AI-generated executive & data science report (grounded strictly in computed metrics, avoiding hallucinations)
- [ ] Multi-format report export (Markdown, HTML, text summary)
- [ ] Integration tests for pipeline & reporting

### Phase 6 — Streamlit UI / UX
**Status: Pending ⏹️**
- [ ] Multi-stage dashboard with progress indicators
- [ ] Interactive distribution charts & correlation heatmaps
- [ ] Real-time generation feedback
- [ ] One-click demo datasets

### Phase 7 — Testing & Final Refinement
**Status: Pending ⏹️**
- [ ] End-to-end integration tests
- [ ] Error handling & edge case validation
- [ ] Complete viva & demonstration readiness

---

## 🛠️ Technologies Used
- **Language**: Python 3.10+
- **Frontend / Dashboard**: Streamlit, Plotly, Altair
- **Data & Numerical Processing**: Pandas, NumPy, SciPy, Statsmodels
- **Machine Learning & Generation**: Scikit-Learn, XGBoost, Torch/NumPy Generative Architectures
- **LLM / GenAI**: Google Gemini / OpenAI API + Built-in Semantic Intent Engine
- **Testing**: Pytest

---

## 💻 How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/SynthGuard.git
   cd SynthGuard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run unit tests**:
   ```bash
   python -m pytest -v
   ```

4. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```
