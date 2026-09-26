"""
Dataset Validator: Ensures uploaded CSV files are valid and suitable for synthetic data processing.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import pandas as pd
import io

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    dataframe: Optional[pd.DataFrame] = None
    row_count: int = 0
    column_count: int = 0


class DatasetValidator:
    """
    Validates CSV files or DataFrames against standard requirements:
    - Non-empty
    - Valid delimiter & parsing
    - Minimum rows (default 10)
    - Minimum columns (default 2)
    - Not entirely null/constant columns
    """

    def __init__(self, min_rows: int = 10, min_cols: int = 2, max_missing_ratio: float = 0.8):
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.max_missing_ratio = max_missing_ratio

    def validate_file(self, file_buffer_or_path) -> ValidationResult:
        """Parse and validate a file buffer or file path."""
        try:
            if hasattr(file_buffer_or_path, "read"):
                # Handle StringIO/BytesIO or Streamlit UploadedFile
                content = file_buffer_or_path.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                df = pd.read_csv(io.StringIO(content))
            else:
                df = pd.read_csv(file_buffer_or_path)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to parse CSV file: {str(e)}. Please ensure it is a valid comma-separated CSV."]
            )

        return self.validate_dataframe(df)

    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """Validate a pandas DataFrame."""
        errors = []
        warnings = []

        if df is None or not isinstance(df, pd.DataFrame):
            return ValidationResult(is_valid=False, errors=["Input data is not a valid tabular DataFrame."])

        if df.empty:
            return ValidationResult(is_valid=False, errors=["Dataset is completely empty (0 rows or 0 columns)."])

        rows, cols = df.shape

        if rows < self.min_rows:
            errors.append(f"Dataset has only {rows} rows. A minimum of {self.min_rows} rows is required for synthetic learning.")

        if cols < self.min_cols:
            errors.append(f"Dataset has only {cols} column(s). At least {self.min_cols} columns are required.")

        # Check for unnamed/empty headers
        unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:") or not str(c).strip()]
        if unnamed_cols:
            warnings.append(f"Found {len(unnamed_cols)} unnamed column headers, which may be index columns.")

        # Check for completely null columns
        all_null_cols = [col for col in df.columns if df[col].isnull().all()]
        if all_null_cols:
            errors.append(f"Column(s) contain 100% missing values: {', '.join(all_null_cols)}")

        # Check for columns with excessive missing data
        high_missing_cols = [
            col for col in df.columns 
            if (df[col].isnull().mean() > self.max_missing_ratio and col not in all_null_cols)
        ]
        if high_missing_cols:
            warnings.append(
                f"Column(s) have >{int(self.max_missing_ratio * 100)}% missing values: {', '.join(high_missing_cols)}. Imputation recommended."
            )

        # Check constant columns (single unique value)
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
        if constant_cols:
            warnings.append(f"Constant column(s) detected (single value): {', '.join(constant_cols)}")

        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            dataframe=df if is_valid else None,
            row_count=rows,
            column_count=cols
        )
