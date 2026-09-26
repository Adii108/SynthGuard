"""
Tests for Phase 2: Synthetic Tabular Data Generators (CTGAN, TVAE, Gaussian Copula).
"""
import pytest
import pandas as pd
import numpy as np
from synthguard.generators import get_generator, GaussianCopulaGenerator, CTGANGenerator, TVAEGenerator

@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 150
    return pd.DataFrame({
        "Age": np.random.randint(22, 60, size=n),
        "Salary": np.random.uniform(30000, 120000, size=n).round(2),
        "Department": np.random.choice(["Sales", "Engineering", "Marketing"], size=n),
        "Promoted": np.random.choice(["Yes", "No"], size=n, p=[0.25, 0.75])
    })

def test_copula_generator(sample_df):
    gen = GaussianCopulaGenerator(random_state=42)
    gen.fit(sample_df, categorical_columns=["Department", "Promoted"], numerical_columns=["Age", "Salary"])
    synth = gen.sample(50)
    assert len(synth) == 50
    assert list(synth.columns) == list(sample_df.columns)
    assert set(synth["Department"].unique()).issubset(set(sample_df["Department"].unique()))
    assert synth["Salary"].min() >= 25000

def test_ctgan_generator(sample_df):
    gen = CTGANGenerator(random_state=42)
    gen.fit(sample_df, categorical_columns=["Department", "Promoted"], numerical_columns=["Age", "Salary"], epochs=20)
    synth = gen.sample(60)
    assert len(synth) == 60
    assert list(synth.columns) == list(sample_df.columns)
    assert set(synth["Promoted"].unique()).issubset(set(sample_df["Promoted"].unique()))

def test_tvae_generator(sample_df):
    gen = TVAEGenerator(random_state=42)
    gen.fit(sample_df, categorical_columns=["Department", "Promoted"], numerical_columns=["Age", "Salary"], epochs=25)
    synth = gen.sample(70)
    assert len(synth) == 70
    assert list(synth.columns) == list(sample_df.columns)

def test_generator_factory():
    g1 = get_generator("CTGAN")
    assert isinstance(g1, CTGANGenerator)
    g2 = get_generator("TVAE")
    assert isinstance(g2, TVAEGenerator)
    g3 = get_generator("Gaussian Copula")
    assert isinstance(g3, GaussianCopulaGenerator)

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_generators.py"])
