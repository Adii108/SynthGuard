"""
Tests for Phase 1: Dataset Ingestion, Validation, and Profiling.
"""
import pytest
import pandas as pd
import numpy as np
from synthguard.profiling.validator import DatasetValidator
from synthguard.profiling.profiler import DatasetProfiler

def test_validator_valid_dataset():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 28, 33, 42, 51],
        "salary": [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 55000, 65000, 75000, 85000],
        "department": ["IT", "HR", "Sales", "IT", "HR", "Sales", "IT", "HR", "Sales", "IT", "HR", "Sales"],
        "left": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    })
    validator = DatasetValidator(min_rows=10, min_cols=2)
    result = validator.validate_dataframe(df)
    assert result.is_valid
    assert len(result.errors) == 0
    assert result.row_count == 12
    assert result.column_count == 4

def test_validator_too_few_rows():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validator = DatasetValidator(min_rows=10)
    result = validator.validate_dataframe(df)
    assert not result.is_valid
    assert any("rows" in err for err in result.errors)

def test_validator_empty_dataset():
    df = pd.DataFrame()
    validator = DatasetValidator()
    result = validator.validate_dataframe(df)
    assert not result.is_valid
    assert any("empty" in err for err in result.errors)

def test_profiler_statistics():
    df = pd.read_csv("data/employee_attrition.csv")
    profiler = DatasetProfiler()
    profile = profiler.profile(df)

    assert profile.row_count == 600
    assert "Age" in profile.numerical_columns
    assert "JobRole" in profile.categorical_columns
    assert "OverTime" in profile.categorical_columns
    assert profile.column_profiles["Age"].inferred_type in ["numerical_continuous", "numerical_discrete"]
    assert profile.column_profiles["JobRole"].unique_count == 5
    assert profile.column_profiles["Age"].stats["mean"] > 30

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_profiler.py"])
