"""
test_preprocessing.py — Tests for data loading and preprocessing
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import generate_synthetic_dataset, preprocess_and_split
from config import N_FEATURES, N_SAMPLES, MALICIOUS_RATIO


class TestSyntheticDataset:
    def test_shape(self):
        df = generate_synthetic_dataset(n_samples=100, n_features=N_FEATURES)
        assert len(df) == 100
        assert len(df.columns) == N_FEATURES + 1  # +1 for label

    def test_label_ratio(self):
        df = generate_synthetic_dataset(n_samples=1000)
        ratio = df["label"].mean()
        assert abs(ratio - MALICIOUS_RATIO) < 0.05, f"Expected ~{MALICIOUS_RATIO}, got {ratio}"

    def test_no_nulls(self):
        df = generate_synthetic_dataset(n_samples=200)
        assert df.isnull().sum().sum() == 0

    def test_binary_labels(self):
        df = generate_synthetic_dataset(n_samples=100)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_reproducibility(self):
        df1 = generate_synthetic_dataset(n_samples=50, seed=42)
        df2 = generate_synthetic_dataset(n_samples=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_data(self):
        df1 = generate_synthetic_dataset(n_samples=50, seed=1)
        df2 = generate_synthetic_dataset(n_samples=50, seed=2)
        assert not df1.equals(df2)


class TestPreprocessing:
    def setup_method(self):
        self.df = generate_synthetic_dataset(n_samples=500)

    def test_split_sizes(self):
        train, val, test, _ = preprocess_and_split(self.df)
        total = len(train) + len(val) + len(test)
        assert total == len(self.df)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_no_label_in_features(self):
        train, val, test, _ = preprocess_and_split(self.df)
        for split in [train, val, test]:
            assert "label" in split.columns

    def test_scaler_applied(self):
        train, val, test, scaler = preprocess_and_split(self.df)
        X_train = train.drop(columns=["label"]).values
        # After StandardScaler, mean should be ~0, std ~1
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0)
        assert np.allclose(means, 0, atol=0.1), "Train features should be centered"
        assert np.allclose(stds, 1, atol=0.2), "Train features should be unit variance"

    def test_stratified_split(self):
        train, val, test, _ = preprocess_and_split(self.df)
        orig_ratio = self.df["label"].mean()
        for split in [train, val, test]:
            split_ratio = split["label"].mean()
            assert abs(split_ratio - orig_ratio) < 0.1, \
                f"Split ratio {split_ratio} deviates from original {orig_ratio}"
