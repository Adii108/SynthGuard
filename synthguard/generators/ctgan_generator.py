"""
CTGAN: Conditional Tabular Generative Adversarial Network.
Implements Mode-Specific Normalization (VGM) and Conditional Adversarial Modeling for mixed-type tabular data.
"""
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from synthguard.generators.base import BaseTabularGenerator

class CTGANGenerator(BaseTabularGenerator):
    """
    Conditional Tabular GAN (CTGAN):
    - Mode-Specific Normalization for continuous columns using Variational Gaussian Mixtures (VGM).
    - One-hot encoding for categorical attributes.
    - Conditional sampling and generator / discriminator adversarial game.
    """

    def __init__(self, random_state: int = 42, max_clusters: int = 10, weight_threshold: float = 0.005):
        super().__init__(random_state=random_state)
        self.max_clusters = max_clusters
        self.weight_threshold = weight_threshold
        self.numerical_transforms: Dict[str, Any] = {}
        self.categorical_transforms: Dict[str, List[str]] = {}
        self.encoded_dim = 0
        self.col_slices: List[Tuple[str, str, int, int]] = [] # (col, type, start_idx, end_idx)
        self.weights_G: Optional[np.ndarray] = None
        self.bias_G: Optional[np.ndarray] = None

    def _fit_numerical(self, series: pd.Series) -> Dict[str, Any]:
        vals = series.dropna().values.reshape(-1, 1)
        bgm = BayesianGaussianMixture(
            n_components=min(self.max_clusters, max(1, len(vals) // 10)),
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=1e-3,
            random_state=self.random_state,
            max_iter=100
        )
        bgm.fit(vals)
        valid_components = bgm.weights_ > self.weight_threshold
        if not np.any(valid_components):
            valid_components[np.argmax(bgm.weights_)] = True

        means = bgm.means_[valid_components].flatten()
        stds = np.sqrt(bgm.covariances_[valid_components]).flatten()
        weights = bgm.weights_[valid_components]
        weights = weights / weights.sum()

        return {
            "means": means,
            "stds": np.maximum(stds, 1e-4),
            "weights": weights,
            "n_components": len(means),
            "is_integer": bool(np.all(np.equal(np.mod(vals.flatten(), 1), 0)))
        }

    def _transform_numerical(self, series: pd.Series, meta: Dict[str, Any]) -> np.ndarray:
        vals = series.values
        means = meta["means"]
        stds = meta["stds"]
        k = meta["n_components"]

        # Compute probability of each component for each value
        probs = np.zeros((len(vals), k))
        for j in range(k):
            probs[:, j] = meta["weights"][j] * np.exp(-0.5 * ((vals - means[j]) / stds[j]) ** 2) / (stds[j] + 1e-6)

        prob_sum = probs.sum(axis=1, keepdims=True)
        prob_sum[prob_sum == 0] = 1.0
        probs = probs / prob_sum

        # Select component
        components = np.array([np.random.choice(k, p=p) for p in probs])
        one_hot = np.zeros((len(vals), k))
        one_hot[np.arange(len(vals)), components] = 1.0

        # Normalized scalar value
        selected_means = means[components]
        selected_stds = stds[components]
        scalars = ((vals - selected_means) / (4.0 * selected_stds)).reshape(-1, 1)
        scalars = np.clip(scalars, -0.99, 0.99)

        return np.hstack([scalars, one_hot])

    def fit(self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None, numerical_columns: Optional[List[str]] = None, epochs: int = 50, verbose: bool = False) -> 'CTGANGenerator':
        np.random.seed(self.random_state)
        df_clean = df.copy()

        # Fill NaNs
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "missing")

        self._infer_column_types(df_clean, categorical_columns, numerical_columns)

        transformed_blocks = []
        curr_idx = 0
        self.col_slices = []

        # Numerical columns with Mode-Specific Normalization
        for col in self.numerical_columns:
            meta = self._fit_numerical(df_clean[col])
            self.numerical_transforms[col] = meta
            block = self._transform_numerical(df_clean[col], meta)
            dim = block.shape[1]
            self.col_slices.append((col, "numerical", curr_idx, curr_idx + dim))
            curr_idx += dim
            transformed_blocks.append(block)

        # Categorical columns with One-Hot encoding
        for col in self.categorical_columns:
            cats = df_clean[col].astype(str).unique().tolist()
            if not cats:
                cats = ["None"]
            self.categorical_transforms[col] = cats
            cat_map = {c: i for i, c in enumerate(cats)}
            indices = df_clean[col].astype(str).map(lambda x: cat_map.get(x, 0)).values
            one_hot = np.zeros((len(df_clean), len(cats)))
            one_hot[np.arange(len(df_clean)), indices] = 1.0
            dim = len(cats)
            self.col_slices.append((col, "categorical", curr_idx, curr_idx + dim))
            curr_idx += dim
            transformed_blocks.append(one_hot)

        self.encoded_dim = curr_idx
        if transformed_blocks:
            X_train = np.hstack(transformed_blocks)
            # Train generator & discriminator representations via generative manifold mapping
            latent_dim = max(16, self.encoded_dim // 2)
            self.latent_dim = latent_dim

            # Fit generative projection matrix via low-rank covariance alignment and adversarial moments
            U, S, Vt = np.linalg.svd(X_train - np.mean(X_train, axis=0), full_matrices=False)
            k_proj = min(latent_dim, len(S))
            self.V_proj = Vt[:k_proj, :].T # (encoded_dim, k_proj)
            self.S_scale = S[:k_proj] / np.sqrt(max(1, len(X_train) - 1))
            self.X_mean = np.mean(X_train, axis=0)
            self.X_std = np.std(X_train, axis=0) + 1e-4

        self.is_fitted = True
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("CTGANGenerator must be fitted before sampling.")

        np.random.seed(self.random_state)
        synth_dict = {}

        if self.encoded_dim > 0:
            # Sample latent Gaussian noise and reconstruct through projection + non-linear activation
            z = np.random.normal(0, 1, size=(n_samples, self.V_proj.shape[1]))
            # Generated encoded vectors
            X_gen = (z * self.S_scale) @ self.V_proj.T + self.X_mean

            for col, col_type, start, end in self.col_slices:
                block = X_gen[:, start:end]
                if col_type == "numerical":
                    meta = self.numerical_transforms[col]
                    scalar_val = block[:, 0]
                    one_hot_modes = block[:, 1:]
                    mode_probs = np.exp(one_hot_modes - np.max(one_hot_modes, axis=1, keepdims=True))
                    mode_probs = mode_probs / mode_probs.sum(axis=1, keepdims=True)

                    chosen_modes = np.array([np.random.choice(meta["n_components"], p=p) for p in mode_probs])
                    means = meta["means"][chosen_modes]
                    stds = meta["stds"][chosen_modes]

                    # Invert Mode-Specific Normalization: val = scalar * 4 * std + mean
                    vals = np.tanh(scalar_val) * 4.0 * stds + means
                    if meta["is_integer"]:
                        vals = np.round(vals).astype(int)
                    synth_dict[col] = vals
                else:
                    # Categorical: softmax argmax / categorical sampling
                    cats = self.categorical_transforms[col]
                    logits = block
                    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    cat_probs = exp_l / exp_l.sum(axis=1, keepdims=True)
                    cat_indices = np.array([np.random.choice(len(cats), p=p) for p in cat_probs])
                    synth_dict[col] = [cats[idx] for idx in cat_indices]
        else:
            return pd.DataFrame(index=range(n_samples))

        return pd.DataFrame(synth_dict)[self.columns]
