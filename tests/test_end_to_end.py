"""
Phase 7: End-to-End Integration Tests & Edge Case Validation.
Tests full workflow across multiple distinct tabular datasets and edge scenarios.
"""
import pytest
import pandas as pd
import numpy as np
from synthguard.profiling import DatasetValidator, DatasetProfiler
from synthguard.generators import get_generator
from synthguard.nlp import LocalIntentParser
from synthguard.reporting import EvaluationPipeline, ReportGenerator

@pytest.mark.parametrize("dataset_path,target_col,model_name", [
    ("data/employee_attrition.csv", "Attrition", "CTGAN"),
    ("data/customer_churn.csv", "Churn", "TVAE"),
    ("data/heart_disease.csv", "target", "Gaussian Copula"),
])
def test_full_pipeline_multi_dataset(dataset_path, target_col, model_name):
    # 1. Load & Validate
    df = pd.read_csv(dataset_path)
    validator = DatasetValidator()
    val_res = validator.validate_dataframe(df)
    assert val_res.is_valid

    # 2. Profile
    profiler = DatasetProfiler()
    profile = profiler.profile(df)
    assert profile.row_count == len(df)

    # 3. NLP Prompt Intent
    parser = LocalIntentParser()
    config = parser.parse(f"Generate 200 records using {model_name} preserving {target_col}", list(df.columns), len(df))
    assert config.selected_model == model_name

    # 4. Generate
    gen = get_generator(model_name, random_state=42)
    gen.fit(df, categorical_columns=profile.categorical_columns, numerical_columns=profile.numerical_columns, epochs=20)
    df_synth = gen.sample(200)
    assert len(df_synth) == 200
    assert list(df_synth.columns) == list(df.columns)

    # 5. Evaluate Pipeline
    pipeline = EvaluationPipeline(random_state=42)
    summary = pipeline.run(df, df_synth, target_column=target_col, selected_model=model_name)
    assert summary.statistical.overall_similarity_score > 0.0
    assert summary.correlation.correlation_fidelity_score > 0.0
    assert summary.ml_utility.utility_retention_score >= 0.0
    assert summary.privacy.risk_level in ["Low Observed Risk", "Moderate Caution Required", "Elevated Privacy Risk"]

    # 6. Report Generation
    report_gen = ReportGenerator()
    md = report_gen.generate_markdown_report(summary, config, dataset_name=dataset_path)
    assert len(md) > 200
    assert "# 🛡️ SynthGuard: Synthetic Data Evaluation & Privacy Audit Report" in md

def test_edge_case_only_numerical_columns():
    df = pd.DataFrame({
        "feat1": np.random.normal(10, 2, 80),
        "feat2": np.random.normal(50, 10, 80),
        "target": np.random.normal(100, 15, 80)
    })
    validator = DatasetValidator()
    assert validator.validate_dataframe(df).is_valid
    profiler = DatasetProfiler()
    prof = profiler.profile(df)
    assert len(prof.categorical_columns) == 0

    gen = get_generator("Gaussian Copula", random_state=42)
    gen.fit(df, numerical_columns=list(df.columns), categorical_columns=[])
    synth = gen.sample(50)
    assert len(synth) == 50

    pipeline = EvaluationPipeline(random_state=42)
    summary = pipeline.run(df, synth, target_column="target", selected_model="Gaussian Copula")
    assert summary.ml_utility.task_type == "regression"

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_end_to_end.py"])
