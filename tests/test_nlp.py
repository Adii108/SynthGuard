"""
Tests for Phase 4: Natural Language GenAI Interface.
"""
import pytest
from synthguard.nlp import LocalIntentParser, LLMService, GenerationConfig

def test_intent_parser_employee_prompt():
    parser = LocalIntentParser()
    columns = ["Age", "JobRole", "MonthlyIncome", "YearsAtCompany", "WorkLifeBalance", "OverTime", "Attrition"]
    prompt = "Generate 1000 synthetic employee data records similar to this dataset while preserving the relationships between Age, MonthlyIncome, JobRole and Attrition."

    config = parser.parse(prompt, columns, dataset_row_count=600)
    assert config.num_records == 1000
    assert "Age" in config.important_columns
    assert "MonthlyIncome" in config.important_columns
    assert "JobRole" in config.important_columns
    assert config.target_column in ["Attrition", "Age"]
    assert config.selected_model in ["CTGAN", "TVAE", "Gaussian Copula"]

def test_intent_parser_model_selection():
    parser = LocalIntentParser()
    columns = ["age", "chol", "trestbps", "target"]
    prompt = "Use TVAE to create 350 samples with high privacy protection and predict target"

    config = parser.parse(prompt, columns, dataset_row_count=400)
    assert config.num_records == 350
    assert config.selected_model == "TVAE"
    assert config.privacy_priority == "High (Anti-Memorization)"
    assert config.target_column == "target"

def test_llm_service_fallback():
    service = LLMService(api_key=None)
    columns = ["Tenure", "MonthlyCharges", "Churn"]
    config = service.interpret_request("Make 500 rows for Churn analysis", columns, 300)
    assert isinstance(config, GenerationConfig)
    assert config.num_records == 500
    assert "Churn" in config.important_columns or config.target_column == "Churn"

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_nlp.py"])
