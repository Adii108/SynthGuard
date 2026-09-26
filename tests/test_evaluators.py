"""
Tests for Phase 3: Evaluation Suite (Statistical, Correlation, ML Utility, Privacy).
"""
import pytest
import pandas as pd
import numpy as np
from synthguard.generators import GaussianCopulaGenerator
from synthguard.evaluation.statistical import StatisticalEvaluator
from synthguard.evaluation.correlation import CorrelationEvaluator
from synthguard.evaluation.ml_utility import MLUtilityEvaluator
from synthguard.evaluation.privacy import PrivacyRiskEvaluator

@pytest.fixture
def real_and_synth_data():
    np.random.seed(42)
    df_real = pd.read_csv("data/employee_attrition.csv")
    gen = GaussianCopulaGenerator(random_state=42)
    gen.fit(df_real)
    df_synth = gen.sample(300)
    return df_real, df_synth

def test_statistical_evaluator(real_and_synth_data):
    df_real, df_synth = real_and_synth_data
    evaluator = StatisticalEvaluator()
    res = evaluator.evaluate(df_real, df_synth)

    assert 0.0 <= res.overall_similarity_score <= 1.0
    assert "Age" in res.column_scores
    assert res.column_scores["Age"].column_type == "numerical"
    assert "JobRole" in res.column_scores
    assert res.column_scores["JobRole"].column_type == "categorical"
    assert len(res.summary_insights) > 0

def test_correlation_evaluator(real_and_synth_data):
    df_real, df_synth = real_and_synth_data
    evaluator = CorrelationEvaluator()
    res = evaluator.evaluate(df_real, df_synth)

    assert 0.0 <= res.correlation_fidelity_score <= 1.0
    assert res.mean_absolute_error >= 0.0
    assert "Age" in res.real_corr_matrix
    assert len(res.preserved_pairs) >= 0

def test_ml_utility_evaluator(real_and_synth_data):
    df_real, df_synth = real_and_synth_data
    evaluator = MLUtilityEvaluator(random_state=42)
    res = evaluator.evaluate(df_real, df_synth, target_column="Attrition")

    assert res.task_type == "classification"
    assert "accuracy" in res.tstr_metrics
    assert "f1_score" in res.tstr_metrics
    assert 0.0 <= res.utility_retention_score <= 1.0

def test_privacy_risk_evaluator(real_and_synth_data):
    df_real, df_synth = real_and_synth_data
    evaluator = PrivacyRiskEvaluator()
    res = evaluator.evaluate(df_real, df_synth)

    assert res.exact_matches_count >= 0
    assert res.mean_dcr >= 0.0
    assert res.risk_level in ["Low Observed Risk", "Moderate Caution Required", "Elevated Privacy Risk"]
    assert len(res.recommendations) > 0

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_evaluators.py"])
