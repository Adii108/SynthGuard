"""
TVAE: Tabular Variational Autoencoder.
Generative model using continuous latent space Gaussian representations with probabilistic reconstruction.
"""
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from synthguard.generators.base import BaseTabularGenerator

class TVAEGenerator(BaseTabularGenerator):
    """
    Tabular Variational Autoencoder (TVAE):
    - Encodes tabular continuous and one-hot categorical features into a latent Gaussian distribution q(z|x).
    - Decodes latent vector z ~ N(0, I) back to tabular data space using mixed reconstruction objectives.
    - Preserves complex multi-variable interactions through non-linear latent manifold learning.
    """

    def __init__(self, random_state: int = 42, latent_dim: int = 16, hidden_dim: int = 64, lr: float = 0.005):
        super().__init__(random_state=random_state)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.col_slices: List[Tuple[str, str, int, int]] = []
        self.num_stats: Dict[str, Dict[str, float]] = {}
        self.cat_mappings: Dict[str, List[str]] = {}
        self.encoder_weights = {}
        self.decoder_weights = {}

    def _prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        curr_idx = 0
        self.col_slices = []

        for col in self.numerical_columns:
            vals = df[col].astype(float).values
            mean_val = float(vals.mean())
            std_val = float(vals.std()) if vals.std() > 0 else 1.0
            is_int = bool(np.all(np.equal(np.mod(vals, 1), 0)))
            self.num_stats[col] = {
                "mean": mean_val,
                "std": std_val,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "is_integer": is_int
            }
            norm_vals = ((vals - mean_val) / std_val).reshape(-1, 1)
            blocks.append(norm_vals)
            self.col_slices.append((col, "numerical", curr_idx, curr_idx + 1))
            curr_idx += 1

        for col in self.categorical_columns:
            cats = df[col].astype(str).unique().tolist()
            if not cats:
                cats = ["None"]
            self.cat_mappings[col] = cats
            cat_to_idx = {c: i for i, c in enumerate(cats)}
            one_hot = np.zeros((len(df), len(cats)))
            for row_idx, item in enumerate(df[col].astype(str)):
                one_hot[row_idx, cat_to_idx.get(item, 0)] = 1.0
            blocks.append(one_hot)
            dim = len(cats)
            self.col_slices.append((col, "categorical", curr_idx, curr_idx + dim))
            curr_idx += dim

        if blocks:
            return np.hstack(blocks)
        return np.zeros((len(df), 0))

    def fit(self, df: pd.DataFrame, categorical_columns: Optional[List[str]] = None, numerical_columns: Optional[List[str]] = None, epochs: int = 50, verbose: bool = False) -> 'TVAEGenerator':
        np.random.seed(self.random_state)
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "missing")

        self._infer_column_types(df_clean, categorical_columns, numerical_columns)
        X = self._prepare_data(df_clean)
        in_dim = X.shape[1]
        latent_dim = min(self.latent_dim, max(4, in_dim // 2))
        hidden_dim = max(self.hidden_dim, in_dim * 2)

        # Initialize Weights (He normal)
        W_enc = np.random.randn(in_dim, hidden_dim) * np.sqrt(2.0 / in_dim)
        b_enc = np.zeros(hidden_dim)
        W_mu = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        b_mu = np.zeros(latent_dim)
        W_logvar = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim) * 0.1
        b_logvar = np.zeros(latent_dim)

        W_dec = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        b_dec = np.zeros(hidden_dim)
        W_out = np.random.randn(hidden_dim, in_dim) * np.sqrt(2.0 / hidden_dim)
        b_out = np.zeros(in_dim)

        # Mini-batch gradient descent for VAE objective
        batch_size = min(64, len(X))
        n_batches = max(1, len(X) // batch_size)

        for ep in range(epochs):
            indices = np.random.permutation(len(X))
            for b in range(n_batches):
                batch_idx = indices[b * batch_size:(b + 1) * batch_size]
                x_b = X[batch_idx]

                # Forward pass: Encoder
                h_enc = np.maximum(0, x_b @ W_enc + b_enc) # ReLU
                mu = h_enc @ W_mu + b_mu
                logvar = np.clip(h_enc @ W_logvar + b_logvar, -4.0, 4.0)
                std = np.exp(0.5 * logvar)
                eps = np.random.randn(*mu.shape)
                z = mu + eps * std

                # Decoder
                h_dec = np.maximum(0, z @ W_dec + b_dec)
                x_recon = h_dec @ W_out + b_out

                # Loss: Reconstruction + KL divergence
                diff = x_recon - x_b
                d_out = diff / batch_size

                # Decoder backprop
                d_h_dec = d_out @ W_out.T * (h_dec > 0)
                d_W_out = h_dec.T @ d_out
                d_b_out = d_out.sum(axis=0)

                d_z = d_h_dec @ W_dec.T
                d_W_dec = z.T @ d_h_dec
                d_b_dec = d_h_dec.sum(axis=0)

                # KL divergence gradient on mu and logvar: KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
                d_mu = d_z + (mu / batch_size) * 0.05
                d_logvar = d_z * eps * 0.5 * std + 0.5 * (np.exp(logvar) - 1.0) * (0.05 / batch_size)

                # Encoder backprop
                d_h_enc = (d_mu @ W_mu.T + d_logvar @ W_logvar.T) * (h_enc > 0)
                d_W_mu = h_enc.T @ d_mu
                d_b_mu = d_mu.sum(axis=0)
                d_W_logvar = h_enc.T @ d_logvar
                d_b_logvar = d_logvar.sum(axis=0)
                d_W_enc = x_b.T @ d_h_enc
                d_b_enc = d_h_enc.sum(axis=0)

                # Adam/SGD parameter updates
                W_out -= self.lr * np.clip(d_W_out, -1.0, 1.0)
                b_out -= self.lr * np.clip(d_b_out, -1.0, 1.0)
                W_dec -= self.lr * np.clip(d_W_dec, -1.0, 1.0)
                b_dec -= self.lr * np.clip(d_b_dec, -1.0, 1.0)
                W_mu -= self.lr * np.clip(d_W_mu, -1.0, 1.0)
                b_mu -= self.lr * np.clip(d_b_mu, -1.0, 1.0)
                W_logvar -= self.lr * np.clip(d_W_logvar, -1.0, 1.0)
                b_logvar -= self.lr * np.clip(d_b_logvar, -1.0, 1.0)
                W_enc -= self.lr * np.clip(d_W_enc, -1.0, 1.0)
                b_enc -= self.lr * np.clip(d_b_enc, -1.0, 1.0)

        self.decoder_weights = {
            "W_dec": W_dec,
            "b_dec": b_dec,
            "W_out": W_out,
            "b_out": b_out,
            "latent_dim": latent_dim
        }
        self.is_fitted = True
        return self

    def sample(self, n_samples: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("TVAEGenerator must be fitted before sampling.")

        np.random.seed(self.random_state)
        latent_dim = self.decoder_weights["latent_dim"]
        W_dec = self.decoder_weights["W_dec"]
        b_dec = self.decoder_weights["b_dec"]
        W_out = self.decoder_weights["W_out"]
        b_out = self.decoder_weights["b_out"]

        # Sample prior z ~ N(0, I)
        z = np.random.randn(n_samples, latent_dim)
        h_dec = np.maximum(0, z @ W_dec + b_dec)
        x_recon = h_dec @ W_out + b_out

        synth_dict = {}
        for col, col_type, start, end in self.col_slices:
            block = x_recon[:, start:end]
            if col_type == "numerical":
                stats_meta = self.num_stats[col]
                # Inverse normalize: val = norm * std + mean
                raw_vals = block[:, 0] * stats_meta["std"] + stats_meta["mean"]
                clipped = np.clip(raw_vals, stats_meta["min"], stats_meta["max"])
                if stats_meta["is_integer"]:
                    clipped = np.round(clipped).astype(int)
                synth_dict[col] = clipped
            else:
                cats = self.cat_mappings[col]
                logits = block
                exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp_l / exp_l.sum(axis=1, keepdims=True)
                cat_indices = np.array([np.random.choice(len(cats), p=p) for p in probs])
                synth_dict[col] = [cats[i] for i in cat_indices]

        return pd.DataFrame(synth_dict)[self.columns]
