# SynthGuard

Synthetic data generation, evaluation and privacy analysis for tabular datasets.

SynthGuard is a tool that takes a real tabular dataset (such as a CSV file), generates synthetic data from it based on what you ask for, and then evaluates how good and safe that synthetic data actually is.

Synthetic data is useful when you need realistic records for development, testing, or machine learning experiments, but you cannot share the original data because of privacy or compliance concerns.

However, simply generating synthetic rows is not enough. You need to know two things: does the synthetic data still preserve the useful patterns from the real dataset, and does it accidentally copy or expose sensitive real records? SynthGuard was built to answer both questions in one place.

---

## The Problem

Organizations often want to share data with developers, data scientists, or external teams to build models and test software. But real datasets frequently contain personal or sensitive information.

Anonymizing data by simply removing names or IDs often fails because individuals can still be re-identified through combinations of other attributes. On the other hand, generating purely random data destroys the statistical relationships between columns, making the data useless for analysis or machine learning.

Synthetic data tries to solve this by creating entirely new artificial records that look and behave like the real data. But without proper evaluation, synthetic data can either be inaccurate (failing to preserve patterns) or unsafe (memorizing real records).

---

## What SynthGuard Does

SynthGuard guides you through a complete workflow:

1. **Upload a dataset**: You provide a CSV file containing your original tabular data.
2. **Understand the dataset**: The system checks the file, finds column types, counts missing values, and calculates basic statistics.
3. **Describe what you want**: You write a plain English prompt describing how many records you want and which columns are important.
4. **Generate synthetic data**: A generative model creates new synthetic rows based on your request.
5. **Compare with the original**: The system checks if the numerical and categorical distributions match the real data.
6. **Check ML usefulness**: A model is trained on synthetic data and tested on real data to see if predictive accuracy holds up.
7. **Check privacy risks**: The system checks for exact duplicate rows, near-duplicates, and potential record memorization.
8. **Generate a report**: A clear summary report is created explaining the results with download options.

---

## A Quick Look at the Project

The application is organized into six clear sections:

- **Overview**: The starting page that introduces the project and lets you load a demo dataset or start with your own file.
- **Dataset Profiling**: Shows your uploaded data, detects continuous and categorical columns, flags missing values, and displays basic summaries.
- **Generate Data**: A workspace where you enter your natural language request, see the interpreted configuration, select a generation model, and generate records.
- **Evaluate Utility**: Compares distributions, displays correlation matrices for real and synthetic data, and benchmarks downstream machine learning performance.
- **Privacy Analysis**: Evaluates duplicate records, calculates distances between synthetic and real records (DCR), and checks for memorization.
- **Final Report**: Gathers all evaluation metrics into a human-readable summary document that you can download as Markdown or HTML.

---

## How the Generation Works

SynthGuard includes three generative methods implemented directly in the project:

### 1. CTGAN (Conditional Tabular GAN)
CTGAN uses a conditional generative adversarial network designed specifically for tabular data. It handles multi-modal continuous columns by breaking them down into Gaussian mixture components and uses one-hot encoding for categorical columns.

### 2. TVAE (Tabular Variational Autoencoder)
TVAE encodes mixed tabular rows into a continuous latent space and reconstructs them using a decoder network. It balances reconstruction accuracy with a regularization penalty to generate diverse samples.

### 3. Gaussian Copula
A statistical baseline model. It models the marginal distributions of individual columns and links them using a multivariate Gaussian covariance matrix to preserve linear correlations.

---

## How the Data is Evaluated

### Statistical Similarity
The system compares the shape of individual columns between real and synthetic data:
- For numerical columns, it uses the two-sample Kolmogorov-Smirnov (KS) test and 1-Wasserstein distance to check if the synthetic numbers follow the same range and spread.
- For categorical columns, it calculates Total Variation Distance (TVD) to see if category proportions match the original data.

### Correlation and Relationships
Generating correct individual columns is not enough; the relationships between columns must also be preserved. SynthGuard computes:
- Pearson correlation matrices for both real and synthetic data.
- The difference matrix and mean absolute error across all column pairs.
- A list of preserved relationships versus relationships that showed drift.

### Machine Learning Utility (TSTR)
To test whether synthetic data is actually useful for machine learning, SynthGuard uses the Train on Synthetic, Test on Real (TSTR) approach:
1. A machine learning model (such as Random Forest) is trained entirely on the synthetic data.
2. The model is tested on unseen real data.
3. The resulting accuracy and F1-score (or R-squared for regression) are compared against a baseline model trained directly on real data (TRTR).

If the synthetic model performs close to the real baseline, the synthetic data successfully preserved the underlying decision boundaries.

### Privacy Checks
SynthGuard checks for empirical privacy risks rather than assuming synthetic data is automatically safe:
- **Exact Matches**: Checks if any synthetic row is an exact identical copy of a real row.
- **Distance to Closest Record (DCR)**: Measures how close each synthetic row is to its nearest real neighbor in normalized space. If many synthetic rows are extremely close to real rows, there may be a risk of attribute disclosure.
- **Nearest Neighbor Distance Ratio (NNDR)**: Compares the distance to the first nearest neighbor against the second nearest neighbor to detect isolated memorized points.

---

## Natural Language Input

Instead of forcing users to manually configure dozens of settings, SynthGuard accepts requests in plain language.

For example, you can write:

```text
Generate 1,000 employee records while preserving the relationships between salary, age, job role and attrition.
```

SynthGuard interprets this request and extracts:
- Target row count: 1,000
- Recommended model: CTGAN
- Key focus columns: Age, Salary, Job Role, Attrition
- Downstream target column: Attrition
- Privacy priority: Standard

You can review and adjust these settings before running the generation.

---

## Project Workflow

```text
User
  |
  v
Dataset Upload
  |
  v
Dataset Profiling (Types, Missing values, Stats)
  |
  v
Natural Language Requirement ("Generate 1000 records...")
  |
  v
Generation Configuration (Model, Row count, Target)
  |
  v
Synthetic Data Generation (CTGAN / TVAE / Copula)
  |
  v
Evaluation Suite
  |-- Statistical Similarity (KS Test, TVD)
  |-- Correlation Preservation (Matrices, Drift)
  |-- ML Utility (Train on Synthetic, Test on Real)
  |-- Privacy Risk (Exact matches, DCR, NNDR)
  |
  v
Final Report (Markdown / HTML / CSV Download)
```

---

## Technologies Used

### Language
- Python 3.10+

### Data Processing and Math
- Pandas
- NumPy
- SciPy
- Statsmodels

### Machine Learning and Modeling
- Scikit-learn

### User Interface and Charts
- Streamlit
- Plotly

### Testing
- Pytest

---

## Project Structure

```text
SynthGuard/
|-- app.py                      # Main Streamlit application
|-- requirements.txt            # Python package dependencies
|-- generate_sample_data.py     # Script to generate sample demo datasets
|-- README.md                   # Project documentation
|
|-- synthguard/                 # Core package logic
|   |-- profiling/              # CSV validation and dataset profiling
|   |   |-- validator.py        # File validation rules
|   |   `-- profiler.py         # Statistical profiling and type inference
|   |-- generators/             # Synthetic data generators
|   |   |-- base.py             # Generator base class
|   |   |-- ctgan_generator.py  # CTGAN implementation
|   |   |-- tvae_generator.py   # TVAE implementation
|   |   `-- copula_generator.py # Gaussian Copula baseline
|   |-- evaluation/             # Evaluation metrics
|   |   |-- statistical.py      # KS-test, Wasserstein, TVD
|   |   |-- correlation.py      # Correlation matrices and error calculation
|   |   |-- ml_utility.py       # TSTR vs TRTR benchmark
|   |   `-- privacy.py          # Exact matches, DCR, NNDR checks
|   |-- nlp/                    # Natural language request parsing
|   |   |-- intent_parser.py    # Rule-based and semantic prompt parser
|   |   `-- llm_service.py      # LLM integration with local fallback
|   |-- reporting/              # Evaluation pipeline and reporting
|   |   |-- pipeline.py         # End-to-end evaluation runner
|   |   `-- report_generator.py # Formatted report generator
|   |-- ui/                     # Modular Streamlit views and styling
|   `-- utils/                  # Helper functions and compatibility patches
|
|-- data/                       # Sample datasets for quick testing
|   |-- employee_attrition.csv  # HR analytics dataset
|   |-- customer_churn.csv      # Customer churn dataset
|   `-- heart_disease.csv       # Medical risk dataset
|
`-- tests/                      # Automated test suite (20 tests)
    |-- test_profiler.py
    |-- test_generators.py
    |-- test_evaluators.py
    |-- test_nlp.py
    |-- test_pipeline.py
    `-- test_end_to_end.py
```

---

## How to Run It

### 1. Clone the repository
```bash
git clone https://github.com/Adii108/SynthGuard.git
cd SynthGuard
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the automated tests
```bash
python -m pytest -v
```

### 5. Start the application
```bash
streamlit run app.py
```

Open your browser and navigate to the local URL shown in your terminal (typically `http://localhost:8501`).

---

## Example Use

1. Open the application and select the pre-loaded **Employee Attrition** dataset.
2. The dataset profiling page displays 600 records across 7 columns (`Age`, `JobRole`, `MonthlyIncome`, `YearsAtCompany`, `WorkLifeBalance`, `OverTime`, `Attrition`).
3. In the Generate page, enter:
   `"Generate 1000 employee records while preserving the relationship between salary, age, job role and attrition."`
4. Click **Parse Intent**, review the extracted settings, and click **Train Model & Generate Synthetic Dataset**.
5. Move to the **Evaluate** tab to view:
   - Real vs synthetic distribution overlays for continuous and categorical columns.
   - Correlation matrices showing how well relationships were preserved.
   - Machine learning utility results comparing a model trained on synthetic data against real baseline performance.
6. Check the **Privacy** tab to review the exact match count (0 found) and the Distance to Closest Record distribution.
7. Download the generated synthetic CSV and the evaluation report.

---

## Current Development Status

- [x] **Phase 1 — Dataset Ingestion and Profiling**: CSV upload, schema validation, column type inference, missing value detection, and statistical summaries.
- [x] **Phase 2 — Synthetic Data Generation**: CTGAN, TVAE, and Gaussian Copula implementations with deterministic seed control and CSV export.
- [x] **Phase 3 — Evaluation Suite**: Kolmogorov-Smirnov similarity, Total Variation Distance, correlation matrix error, TSTR machine learning utility, and empirical privacy checks.
- [x] **Phase 4 — Natural Language Interface**: Natural language prompt interpretation, parameter extraction, and configuration preview.
- [x] **Phase 5 — Automated Reporting**: Evaluation pipeline orchestration, grounded report generation, and Markdown/HTML downloads.
- [x] **Phase 6 — Streamlit Interface**: Multi-step workflow navigation, interactive Plotly charts, and responsive layouts.
- [x] **Phase 7 — Testing and Refinement**: 20 automated unit and integration tests across multiple benchmark datasets and edge cases.

---

## Limitations

Being upfront about what the tool can and cannot do:

- **Empirical privacy vs mathematical guarantees**: Zero exact duplicates and positive DCR values show that records were not directly copied, but they do not provide formal differential privacy guarantees.
- **Dataset quality dependency**: If the original dataset is very small or has extreme class imbalance, the generated synthetic data will reflect those shortcomings.
- **Training time**: Complex deep generative models take longer to fit than simple statistical baselines on larger datasets.
- **Complex multi-column interactions**: Very subtle higher-order interactions across three or more columns may experience slight smoothing in generated samples.

---

## Why This Project

When exploring synthetic data tools, many existing solutions either stop at generating a CSV without verifying if the data is accurate, or require writing complex code with unclear privacy risks.

I wanted to build a practical project that connects the entire process: taking a dataset, specifying what you need in plain language, generating synthetic records, and giving you an honest evaluation of both its usefulness and its privacy trade-offs before you use it.

---

## Future Improvements

- Add differential privacy noise mechanisms with configurable epsilon parameters.
- Support time-series and sequential tabular datasets.
- Support multi-table relational datasets with foreign-key constraints.
- Add additional evaluation metrics for imbalanced datasets.
