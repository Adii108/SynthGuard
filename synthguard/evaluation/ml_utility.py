"""
Machine Learning Utility Evaluator: Train on Synthetic, Test on Real (TSTR).
Compares downstream ML performance when trained on synthetic vs real data.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

@dataclass
class MLUtilityResult:
    task_type: str  # 'classification' or 'regression'
    target_column: str
    model_name: str
    tstr_metrics: Dict[str, float]  # Train on Synthetic, Test on Real
    trtr_metrics: Dict[str, float]  # Train on Real, Test on Real (Baseline)
    utility_gap: Dict[str, float]   # TRTR - TSTR difference
    utility_retention_score: float  # [0.0, 1.0]
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MLUtilityEvaluator:
    """
    Benchmarks synthetic data utility for downstream Machine Learning models (TSTR vs TRTR).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _detect_target(self, df: pd.DataFrame, user_target: Optional[str] = None) -> Tuple[str, str]:
        if user_target and user_target in df.columns:
            target = user_target
        else:
            # Look for common target names or last column
            candidates = ["target", "label", "churn", "attrition", "class", "outcome", "status", "y"]
            found = [c for c in df.columns if c.lower() in candidates]
            target = found[0] if found else df.columns[-1]

        # Determine task type
        series = df[target].dropna()
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
            task_type = "regression"
        else:
            task_type = "classification"

        return target, task_type

    def evaluate(self, df_real: pd.DataFrame, df_synth: pd.DataFrame, target_column: Optional[str] = None) -> MLUtilityResult:
        target, task_type = self._detect_target(df_real, target_column)

        feature_cols = [c for c in df_real.columns if c != target and c in df_synth.columns]
        if not feature_cols:
            raise ValueError(f"No feature columns found for target '{target}'.")

        # Split real data into Train (40%) and Test (60%) to ensure clean evaluation
        real_train, real_test = train_test_split(df_real, test_size=0.5, random_state=self.random_state)
        synth_train = df_synth.copy()

        cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df_real[c]) or df_real[c].nunique() <= 5]
        num_cols = [c for c in feature_cols if c not in cat_cols]

        def _clean_features(df_subset):
            df_out = pd.DataFrame(index=df_subset.index)
            for c in num_cols:
                df_out[c] = pd.to_numeric(df_subset[c], errors='coerce').fillna(0.0)
            for c in cat_cols:
                df_out[c] = df_subset[c].astype(str).fillna("missing")
            return df_out

        X_real_train_df = _clean_features(real_train[feature_cols])
        X_real_test_df = _clean_features(real_test[feature_cols])
        X_synth_train_df = _clean_features(synth_train[feature_cols])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ],
            remainder='drop'
        )

        # Fit preprocessor on combined data
        combined_features = pd.concat([X_real_train_df, X_synth_train_df], axis=0)
        preprocessor.fit(combined_features)

        X_real_train = preprocessor.transform(X_real_train_df)
        X_real_test = preprocessor.transform(X_real_test_df)
        X_synth_train = preprocessor.transform(X_synth_train_df)

        if task_type == "classification":
            # Map target labels to integers
            classes = sorted(list(set(df_real[target].dropna().astype(str).unique())))
            label_map = {c: i for i, c in enumerate(classes)}

            y_real_train = real_train[target].astype(str).map(lambda x: label_map.get(x, 0)).values
            y_real_test = real_test[target].astype(str).map(lambda x: label_map.get(x, 0)).values
            y_synth_train = synth_train[target].astype(str).map(lambda x: label_map.get(x, 0)).values

            # Model: Random Forest Classifier
            model_tstr = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state)
            model_tstr.fit(X_synth_train, y_synth_train)
            preds_tstr = model_tstr.predict(X_real_test)

            model_trtr = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state)
            model_trtr.fit(X_real_train, y_real_train)
            preds_trtr = model_trtr.predict(X_real_test)

            is_binary = len(classes) == 2
            avg_mode = 'binary' if is_binary else 'weighted'

            tstr_m = {
                "accuracy": round(float(accuracy_score(y_real_test, preds_tstr)), 4),
                "precision": round(float(precision_score(y_real_test, preds_tstr, average=avg_mode, zero_division=0)), 4),
                "recall": round(float(recall_score(y_real_test, preds_tstr, average=avg_mode, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_real_test, preds_tstr, average=avg_mode, zero_division=0)), 4),
            }

            trtr_m = {
                "accuracy": round(float(accuracy_score(y_real_test, preds_trtr)), 4),
                "precision": round(float(precision_score(y_real_test, preds_trtr, average=avg_mode, zero_division=0)), 4),
                "recall": round(float(recall_score(y_real_test, preds_trtr, average=avg_mode, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_real_test, preds_trtr, average=avg_mode, zero_division=0)), 4),
            }

            gap = {k: round(trtr_m[k] - tstr_m[k], 4) for k in tstr_m}
            # Retention score = TSTR F1 / TRTR F1
            retention = min(1.0, max(0.0, (tstr_m["f1_score"] / max(0.01, trtr_m["f1_score"]))))
            model_label = "Random Forest Classifier"

            summary = (
                f"TSTR Accuracy: {tstr_m['accuracy']:.1%} (TRTR baseline: {trtr_m['accuracy']:.1%}). "
                f"TSTR F1-Score: {tstr_m['f1_score']:.3f} vs baseline {trtr_m['f1_score']:.3f}. "
                f"Utility Retention: {retention:.1%} of real-data predictive power."
            )

        else:
            # Regression Task
            y_real_train = real_train[target].astype(float).fillna(0).values
            y_real_test = real_test[target].astype(float).fillna(0).values
            y_synth_train = synth_train[target].astype(float).fillna(0).values

            model_tstr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state)
            model_tstr.fit(X_synth_train, y_synth_train)
            preds_tstr = model_tstr.predict(X_real_test)

            model_trtr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state)
            model_trtr.fit(X_real_train, y_real_train)
            preds_trtr = model_trtr.predict(X_real_test)

            tstr_m = {
                "r2_score": round(float(r2_score(y_real_test, preds_tstr)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_real_test, preds_tstr))), 3),
                "mae": round(float(mean_absolute_error(y_real_test, preds_tstr)), 3),
            }

            trtr_m = {
                "r2_score": round(float(r2_score(y_real_test, preds_trtr)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_real_test, preds_trtr))), 3),
                "mae": round(float(mean_absolute_error(y_real_test, preds_trtr)), 3),
            }

            gap = {
                "r2_diff": round(trtr_m["r2_score"] - tstr_m["r2_score"], 4),
                "rmse_diff": round(tstr_m["rmse"] - trtr_m["rmse"], 3),
                "mae_diff": round(tstr_m["mae"] - trtr_m["mae"], 3),
            }
            retention = min(1.0, max(0.0, (tstr_m["r2_score"] / max(0.01, trtr_m["r2_score"])))) if trtr_m["r2_score"] > 0 else 0.5
            model_label = "Random Forest Regressor"

            summary = (
                f"TSTR R² Score: {tstr_m['r2_score']:.3f} (TRTR baseline: {trtr_m['r2_score']:.3f}). "
                f"TSTR RMSE: {tstr_m['rmse']:.2f} vs baseline {trtr_m['rmse']:.2f}. "
                f"Utility Retention: {retention:.1%} of baseline regression utility."
            )

        return MLUtilityResult(
            task_type=task_type,
            target_column=target,
            model_name=model_label,
            tstr_metrics=tstr_m,
            trtr_metrics=trtr_m,
            utility_gap=gap,
            utility_retention_score=round(retention, 3),
            summary_text=summary
        )
