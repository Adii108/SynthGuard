"""
Gaussian Copula Tabular Generator.
Models multivariate tabular dependencies via Gaussian Copula with empirical/parametric marginal distributions.
"""
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats
from synthguard.generators.base import BaseTabularGenerator

class GaussianCopulaGenerator(BaseTabularGenerator):
    """
    Gaussian Copula Generative Model:
    1. Transforms numerical columns to uniform marginals via empirical CDF (ranks) + jitter.
    2. Maps uniform marginals to standard Normal via probit (inverse standard normal CDF).
    3. Encodes categorical variables into continuous quantile latent space.
    4. Computes multivariate Gaussian correlation matrix with shrinkage.
    5. Samples from multivariate Gaussian and inverts marginal transformations back to original space.
    """

    def __init__(self, random_state: int = 42, jitter: float = 1e-4):
        super().__init__(random_state=random_state)
        self.jitter = jitter
        self.cov_matrix: Optional[np.ndarray] = None
        self.mean_vector: Optional[np.ndarray] = None
        self.marginal_distributions: Dict[str, Any] = {}
        self.cat_mappings: Dict[str, Dict[str, Any]] = {}

    def fit(self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None, numerical_columns: Optional[List[str]] = None, epochs: int = 1, verbose: bool = False) -> 'GaussianCopulaGenerator':
        np.random.seed(self.random_state)
        df_clean = df.copy()

        # Handle missing values temporarily during fit
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "missing")

        self._infer_column_types(df_clean, categorical_columns, numerical_columns)
        n_rows = len(df_clean)

        latent_data = []

        # Process Numerical Columns
        for col in self.numerical_columns:
            values = df_clean[col].values.astype(float)
            sorted_vals = np.sort(values)
            self.marginal_distributions[col] = {
                "sorted_vals": sorted_vals,
                "min": float(values.min()),
                "max": float(values.max()),
                "is_integer": bool(np.all(np.equal(np.mod(values, 1), 0)))
            }
            # Empirical CDF to uniform [0, 1]
            ranks = stats.rankdata(values, method="average") / (n_rows + 1.0)
            ranks = np.clip(ranks, 1e-6, 1.0 - 1e-6)
            # Uniform to standard normal
            norm_z = stats.norm.ppf(ranks)
            latent_data.append(norm_z)

        # Process Categorical Columns
        for col in self.categorical_columns:
            series = df_clean[col].astype(str)
            value_counts = series.value_counts(normalize=True)
            cats = value_counts.index.tolist()
            cum_probs = np.cumsum(value_counts.values)

            self.cat_mappings[col] = {
                "categories": cats,
                "probabilities": value_counts.values,
                "cum_probs": cum_probs
            }

            # Map categories to continuous normal intervals
            cat_to_interval = {}
            prev = 0.0
            for cat, cp in zip(cats, cum_probs):
                cat_to_interval[cat] = (prev, cp)
                prev = cp

            # Sample uniform within each category interval and map to standard normal
            u_vals = []
            for item in series:
                low, high = cat_to_interval.get(item, (0.0, 1.0))
                u = np.random.uniform(max(1e-6, low + 1e-6), min(1.0 - 1e-6, high - 1e-6))
                u_vals.append(u)

            norm_z = stats.norm.ppf(np.array(u_vals))
            latent_data.append(norm_z)

        if latent_data:
            Z = np.column_stack(latent_data)
            # Add subtle jitter to ensure positive definiteness
            cov = np.cov(Z, rowvar=False)
            cov = np.atleast_2d(cov)
            if cov.shape[0] == 1 and cov.shape[1] == 1:
                cov[0, 0] = max(cov[0, 0], 1e-4)
            else:
                cov = cov + np.eye(cov.shape[0]) * 1e-4
            self.cov_matrix = cov
            self.mean_vector = np.mean(Z, axis=0)

        self.is_fitted = True
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Generator must be fitted before sampling.")

        np.random.seed(self.random_state)
        synth_dict = {}

        if self.cov_matrix is not None and self.cov_matrix.size > 0:
            Z_synth = np.random.multivariate_normal(
                self.mean_vector, self.cov_matrix, size=n_samples
            )
            # Convert normal Z to Uniform U via CDF
            U_synth = stats.norm.cdf(Z_synth)
            U_synth = np.clip(U_synth, 1e-6, 1.0 - 1e-6)

            idx = 0
            for col in self.numerical_columns:
                u_col = U_synth[:, idx]
                meta = self.marginal_distributions[col]
                sorted_vals = meta["sorted_vals"]
                # Invert empirical CDF via quantile interpolation
                synth_vals = np.quantile(sorted_vals, u_col)
                # Add light Gaussian perturbation for continuous support
                std_pert = (meta["max"] - meta["min"]) * 0.005 if meta["max"] > meta["min"] else 0.0
                if std_pert > 0 and not meta["is_integer"]:
                    synth_vals += np.random.normal(0, std_pert, size=n_samples)
                    synth_vals = np.clip(synth_vals, meta["min"], meta["max"])
                if meta["is_integer"]:
                    synth_vals = np.round(synth_vals).astype(int)
                synth_dict[col] = synth_vals
                idx += 1

            for col in self.categorical_columns:
                u_col = U_synth[:, idx]
                meta = self.cat_mappings[col]
                cats = meta["categories"]
                cum_probs = meta["cum_probs"]

                synth_cats = []
                for u in u_col:
                    cat_idx = np.searchsorted(cum_probs, u)
                    cat_idx = min(cat_idx, len(cats) - 1)
                    synth_cats.append(cats[cat_idx])
                synth_dict[col] = synth_cats
                idx += 1
        else:
            # Degenerate case (e.g. empty column list)
            return pd.DataFrame(index=range(n_samples))

        df_synth = pd.DataFrame(synth_dict)[self.columns]
        return df_synth
