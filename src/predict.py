"""
predict.py — Prediction utilities for MalwareShield AI
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASELINE_DIR, DEFENDED_DIR, SCALER_PATH, FEATURE_NAMES_PATH


def load_scaler():
    """Load the fitted StandardScaler."""
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


def preprocess_input(X: np.ndarray, scaler=None) -> np.ndarray:
    """Apply scaler if available, else return as-is."""
    if scaler is not None:
        return scaler.transform(X)
    return X


def predict_single(model, X: np.ndarray, scaler=None) -> dict:
    """
    Run a single prediction.
    
    Returns:
        {
            "label": 0 or 1,
            "label_name": "Benign" or "Malicious",
            "confidence": float,
            "probabilities": {"benign": float, "malicious": float}
        }
    """
    X_sc = preprocess_input(X.reshape(1, -1), scaler)
    label = int(model.predict(X_sc)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_sc)[0]
        confidence = float(probs[label])
        probabilities = {"benign": float(probs[0]), "malicious": float(probs[1])}
    else:
        confidence = 1.0
        probabilities = {"benign": 1.0 - label, "malicious": float(label)}

    return {
        "label": label,
        "label_name": "Malicious" if label == 1 else "Benign",
        "confidence": confidence,
        "probabilities": probabilities,
    }


def predict_batch(model, X: np.ndarray, scaler=None) -> dict:
    """
    Run batch predictions.
    
    Returns:
        {
            "labels": array,
            "probabilities": array (n, 2),
        }
    """
    X_sc = preprocess_input(X, scaler)
    labels = model.predict(X_sc)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_sc)
    else:
        probs = np.column_stack([1 - labels, labels]).astype(float)
    return {"labels": labels, "probabilities": probs}


def validate_feature_vector(X: np.ndarray, expected_features: int) -> tuple:
    """Validate that input has correct number of features."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != expected_features:
        return False, f"Expected {expected_features} features, got {X.shape[1]}"
    if np.any(np.isnan(X)):
        return False, "Input contains NaN values"
    if np.any(np.isinf(X)):
        return False, "Input contains infinite values"
    return True, "OK"


def prepare_feature_frame(df: pd.DataFrame, feature_names: list,
                          expected_features: int) -> pd.DataFrame:
    """Order uploaded features and convert feature/value CSVs to one row."""
    data = df.copy()
    data.columns = [str(column).strip() for column in data.columns]
    data = data.drop(columns=[
        column for column in data.columns
        if column.lower() in {"label", "class", "target"}
    ])

    if len(data.columns) == 2:
        normalized = {column.lower(): column for column in data.columns}
        feature_column = next(
            (normalized[name] for name in ("feature", "feature_name", "name", "attribute")
             if name in normalized),
            None,
        )
        value_column = next(
            (normalized[name] for name in ("value", "feature_value") if name in normalized),
            None,
        )
        if feature_column is not None and value_column is not None:
            values = data.set_index(feature_column)[value_column]
            if values.index.has_duplicates:
                raise ValueError("Feature/value CSV contains duplicate feature names.")
            data = values.to_frame().T

    if feature_names and all(name in data.columns for name in feature_names):
        data = data.loc[:, feature_names]
    elif data.shape[1] != expected_features:
        raise ValueError(
            f"The selected model requires {expected_features} features; "
            f"the uploaded data contains {data.shape[1]}. Provide all model features, "
            "either as columns or as feature/value rows."
        )

    if data.shape[1] != expected_features:
        raise ValueError(
            f"The selected model requires {expected_features} features; "
            f"the uploaded data contains {data.shape[1]}."
        )
    return data.apply(pd.to_numeric, errors="coerce")


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances where available."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    return df
