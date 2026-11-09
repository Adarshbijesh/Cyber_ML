"""
data_loader.py — Dataset loading utilities for MalwareShield AI
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, N_FEATURES, FEATURE_NAMES_PATH
)


def get_feature_names() -> list:
    """Return feature column names."""
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH) as f:
            return json.load(f)

    # Synthetic fallback names
    names = []
    names += [f"pe_header_{i}" for i in range(10)]
    names += [f"section_stat_{i}" for i in range(10)]
    names += [f"import_count_{i}" for i in range(5)]
    names += [f"export_count_{i}" for i in range(5)]
    names += [f"entropy_{i}" for i in range(8)]
    names += [f"string_feat_{i}" for i in range(6)]
    names += [f"misc_feat_{i}" for i in range(6)]
    return names[:N_FEATURES]


def load_dataset(path: str = None) -> pd.DataFrame:
    """Load the main feature dataset."""
    p = path or DATASET_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Dataset not found at {p}. Run `python src/preprocessing.py` first."
        )
    df = pd.read_csv(p)
    return df


def load_splits():
    """Load pre-computed train/val/test splits."""
    if not all(os.path.exists(p) for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]):
        raise FileNotFoundError("Splits not found. Run `python src/preprocessing.py` first.")

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, val, test


def split_features_labels(df: pd.DataFrame, label_col: str = "label"):
    """Separate feature matrix and label vector."""
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X.values, y.values, list(X.columns)


def get_class_distribution(df: pd.DataFrame, label_col: str = "label") -> dict:
    """Return class counts."""
    counts = df[label_col].value_counts().to_dict()
    return {
        "benign": int(counts.get(0, 0)),
        "malicious": int(counts.get(1, 0)),
        "total": len(df),
    }


def load_sample_for_demo(n: int = 5) -> pd.DataFrame:
    """Load a small sample for dashboard demo."""
    from config import SAMPLE_PATH
    if os.path.exists(SAMPLE_PATH):
        return pd.read_csv(SAMPLE_PATH).head(n)
    # Fall back to test set
    if os.path.exists(TEST_PATH):
        return pd.read_csv(TEST_PATH).head(n)
    raise FileNotFoundError("No sample file found.")
