"""
Tests for Phase 5: Automated Evaluation Pipeline & AI Report Generation.
"""
import pytest
import pandas as pd
from synthguard.generators import GaussianCopulaGenerator
from synthguard.nlp import LocalIntentParser
from synthguard.reporting import EvaluationPipeline, ReportGenerator

def test_pipeline_and_report_generation():
    df_real = pd.read_csv("data/customer_churn.csv")
    gen = GaussianCopulaGenerator(random_state=42)
    gen.fit(df_real)
    df_synth = gen.sample(250)

    parser = LocalIntentParser()
    config = parser.parse("Generate 250 records for customer churn analysis", list(df_real.columns), len(df_real))

    pipeline = EvaluationPipeline(random_state=42)
    summary = pipeline.run(df_real, df_synth, target_column=config.target_column, selected_model="Gaussian Copula")

    assert summary.real_rows == len(df_real)
    assert summary.synth_rows == 250
    assert summary.statistical.overall_similarity_score > 0
    assert summary.correlation.correlation_fidelity_score > 0
    assert summary.ml_utility.task_type == "classification"
    assert summary.privacy.risk_level in ["Low Observed Risk", "Moderate Caution Required", "Elevated Privacy Risk"]

    report_gen = ReportGenerator()
    md_report = report_gen.generate_markdown_report(summary, config, "Customer Churn Dataset")
    assert "# 🛡️ SynthGuard: Synthetic Data Evaluation & Privacy Audit Report" in md_report
    assert "Customer Churn Dataset" in md_report
    assert "Train on Synthetic / Test on Real" in md_report
    assert "Empirical Privacy Risk" in md_report

    html_report = report_gen.generate_html_report(summary, config)
    assert "<!DOCTYPE html>" in html_report

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_pipeline.py"])
