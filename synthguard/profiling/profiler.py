"""
Dataset Profiler: Extracts comprehensive statistical profiles and schema metadata.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: str  # 'numerical_continuous', 'numerical_discrete', 'categorical', 'binary', 'id_text'
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    is_constant: bool
    is_unique_id: bool
    stats: Dict[str, Any] = field(default_factory=dict)
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    numerical_columns: List[str]
    categorical_columns: List[str]
    binary_columns: List[str]
    id_columns: List[str]
    missing_cell_count: int
    missing_cell_percentage: float
    duplicate_row_count: int
    duplicate_row_percentage: float
    memory_usage_kb: float
    column_profiles: Dict[str, ColumnProfile]
    numerical_summary: Dict[str, Dict[str, float]]
    categorical_summary: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetProfiler:
    """
    Computes statistical profiles and infers semantic types for tabular data.
    """

    def __init__(self, discrete_threshold: int = 15, id_uniqueness_threshold: float = 0.95):
        self.discrete_threshold = discrete_threshold
        self.id_uniqueness_threshold = id_uniqueness_threshold

    def profile(self, df: pd.DataFrame) -> DatasetProfile:
        if df.empty:
            raise ValueError("Cannot profile an empty DataFrame.")

        n_rows, n_cols = df.shape
        col_profiles: Dict[str, ColumnProfile] = {}
        num_cols: List[str] = []
        cat_cols: List[str] = []
        bin_cols: List[str] = []
        id_cols: List[str] = []

        total_cells = n_rows * n_cols
        total_missing = int(df.isnull().sum().sum())
        total_duplicates = int(df.duplicated().sum())

        for col in df.columns:
            series = df[col]
            missing_count = int(series.isnull().sum())
            missing_pct = round((missing_count / n_rows) * 100, 2)
            valid_series = series.dropna()
            unique_count = int(valid_series.nunique())
            unique_pct = round((unique_count / max(1, len(valid_series))) * 100, 2)
            is_constant = unique_count <= 1
            is_id = False

            # Infer Column Type
            is_numeric = pd.api.types.is_numeric_dtype(series)

            if is_constant:
                inferred_type = "constant"
            elif unique_count == 2:
                inferred_type = "binary"
                bin_cols.append(col)
                if is_numeric:
                    num_cols.append(col)
                else:
                    cat_cols.append(col)
            elif is_numeric:
                # Check if it looks like an ID column (e.g., sequentially increasing or 100% unique integers)
                if unique_count == n_rows and "id" in col.lower():
                    inferred_type = "id_text"
                    id_cols.append(col)
                    is_id = True
                elif unique_count <= self.discrete_threshold:
                    inferred_type = "numerical_discrete"
                    num_cols.append(col)
                else:
                    inferred_type = "numerical_continuous"
                    num_cols.append(col)
            else:
                # String / Object / Categorical
                if unique_pct > (self.id_uniqueness_threshold * 100) and n_rows > 30:
                    inferred_type = "id_text"
                    id_cols.append(col)
                    is_id = True
                else:
                    inferred_type = "categorical"
                    cat_cols.append(col)

            # Compute Column Specific Stats
            stats = {}
            if is_numeric and not valid_series.empty:
                v = valid_series.astype(float)
                stats = {
                    "mean": round(float(v.mean()), 3),
                    "std": round(float(v.std(ddof=1) if len(v) > 1 else 0.0), 3),
                    "min": round(float(v.min()), 3),
                    "q25": round(float(v.quantile(0.25)), 3),
                    "median": round(float(v.median()), 3),
                    "q75": round(float(v.quantile(0.75)), 3),
                    "max": round(float(v.max()), 3),
                    "skew": round(float(v.skew() if len(v) > 2 else 0.0), 3),
                }
            else:
                # Categorical / string stats
                top_counts = valid_series.value_counts().head(10).to_dict()
                stats = {
                    "top_categories": {str(k): int(v) for k, v in top_counts.items()},
                    "mode": str(valid_series.mode().iloc[0]) if not valid_series.empty else "N/A"
                }

            # Sample values
            sample_vals = [
                val if pd.notnull(val) else None 
                for val in valid_series.head(5).tolist()
            ]

            col_profiles[col] = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_percentage=missing_pct,
                unique_count=unique_count,
                unique_percentage=unique_pct,
                is_constant=is_constant,
                is_unique_id=is_id,
                stats=stats,
                sample_values=sample_vals
            )

        # Numerical summary table
        num_summary = {}
        for col in num_cols:
            if col in col_profiles and col_profiles[col].stats and "mean" in col_profiles[col].stats:
                num_summary[col] = col_profiles[col].stats

        # Categorical summary table
        cat_summary = {}
        for col in cat_cols:
            if col in col_profiles and col_profiles[col].stats:
                cat_summary[col] = col_profiles[col].stats

        memory_kb = round(df.memory_usage(deep=True).sum() / 1024.0, 2)

        return DatasetProfile(
            row_count=n_rows,
            column_count=n_cols,
            numerical_columns=list(dict.fromkeys(num_cols)),
            categorical_columns=list(dict.fromkeys(cat_cols)),
            binary_columns=bin_cols,
            id_columns=id_cols,
            missing_cell_count=total_missing,
            missing_cell_percentage=round((total_missing / total_cells) * 100, 2),
            duplicate_row_count=total_duplicates,
            duplicate_row_percentage=round((total_duplicates / n_rows) * 100, 2),
            memory_usage_kb=memory_kb,
            column_profiles=col_profiles,
            numerical_summary=num_summary,
            categorical_summary=cat_summary
        )
