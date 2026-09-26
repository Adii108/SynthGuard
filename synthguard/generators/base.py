"""
Base class for synthetic tabular data generators.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

class BaseTabularGenerator(ABC):
    """Abstract base class for tabular generators."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.is_fitted = False
        self.columns: List[str] = []
        self.dtypes: Dict[str, str] = {}
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []

    @abstractmethod
    def fit(self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None, numerical_columns: Optional[List[str]] = None, epochs: int = 50, verbose: bool = False) -> 'BaseTabularGenerator':
        """Fit the generative model on the input DataFrame."""
        pass

    @abstractmethod
    def sample(self, n_samples: int) -> pd.DataFrame:
        """Generate n_samples of synthetic tabular data."""
        pass

    def _infer_column_types(self, df: pd.DataFrame, categorical_columns: Optional[List[str]], numerical_columns: Optional[List[str]]):
        """Helper to assign numerical and categorical columns."""
        self.columns = list(df.columns)
        self.dtypes = {col: str(df[col].dtype) for col in df.columns}

        if categorical_columns is not None:
            self.categorical_columns = list(categorical_columns)
        else:
            self.categorical_columns = [
                col for col in df.columns 
                if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 5
            ]

        if numerical_columns is not None:
            self.numerical_columns = list(numerical_columns)
        else:
            self.numerical_columns = [col for col in df.columns if col not in self.categorical_columns]
