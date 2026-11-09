"""
test_models.py — Tests for model training and loading
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import generate_synthetic_dataset, preprocess_and_split
from src.data_loader import split_features_labels
from src.train import evaluate_model, load_model, list_available_models, MODELS
from src.predict import predict_single, predict_batch, validate_feature_vector
from config import BASELINE_DIR, N_FEATURES


@pytest.fixture(scope="module")
def data():
    df = generate_synthetic_dataset(n_samples=400, seed=42)
    train, val, test, scaler = preprocess_and_split(df, seed=42)
    X_train, y_train, feat_names = split_features_labels(train)
    X_test, y_test, _ = split_features_labels(test)
    return X_train, y_train, X_test, y_test, feat_names, scaler


@pytest.fixture(scope="module")
def trained_lr(data):
    from sklearn.linear_model import LogisticRegression
    X_train, y_train, X_test, y_test, _, _ = data
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


class TestModelEvaluation:
    def test_evaluate_returns_metrics(self, trained_lr, data):
        X_train, y_train, X_test, y_test, _, _ = data
        metrics = evaluate_model(trained_lr, X_test, y_test)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_metrics_in_range(self, trained_lr, data):
        X_train, y_train, X_test, y_test, _, _ = data
        metrics = evaluate_model(trained_lr, X_test, y_test)
        for key in ["accuracy", "precision", "recall", "f1"]:
            val = metrics[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_roc_auc_in_range(self, trained_lr, data):
        X_train, y_train, X_test, y_test, _, _ = data
        metrics = evaluate_model(trained_lr, X_test, y_test)
        if metrics["roc_auc"] is not None:
            assert 0.0 <= metrics["roc_auc"] <= 1.0


class TestPrediction:
    def test_predict_single_returns_dict(self, trained_lr, data):
        X_train, y_train, X_test, y_test, _, _ = data
        result = predict_single(trained_lr, X_test[0])
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["label"] in [0, 1]

    def test_predict_single_confidence_in_range(self, trained_lr, data):
        X_train, y_train, X_test, y_test, _, _ = data
        for i in range(5):
            result = predict_single(trained_lr, X_test[i])
            assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_batch(self, trained_lr, data):
        X_train, y_train, X_test, y_test, _, _ = data
        result = predict_batch(trained_lr, X_test[:10])
        assert len(result["labels"]) == 10
        assert result["probabilities"].shape == (10, 2)

    def test_validate_feature_vector_correct(self, data):
        X_train, y_train, X_test, y_test, _, _ = data
        valid, msg = validate_feature_vector(X_test[0], N_FEATURES)
        assert valid is True

    def test_validate_feature_vector_wrong_size(self):
        X_wrong = np.zeros(10)
        valid, msg = validate_feature_vector(X_wrong, N_FEATURES)
        assert valid is False

    def test_validate_feature_vector_nan(self):
        X_nan = np.full(N_FEATURES, np.nan)
        valid, msg = validate_feature_vector(X_nan, N_FEATURES)
        assert valid is False


class TestSavedModels:
    def test_models_saved(self):
        if not os.path.exists(BASELINE_DIR):
            pytest.skip("Baseline models not trained yet")
        models = list_available_models()
        assert len(models) > 0

    def test_load_model_rf(self):
        if not os.path.exists(BASELINE_DIR):
            pytest.skip("Baseline models not trained yet")
        models = list_available_models()
        if "random_forest" in models:
            model = load_model("random_forest")
            assert hasattr(model, "predict")

    def test_load_nonexistent_model(self):
        with pytest.raises(FileNotFoundError):
            load_model("this_model_does_not_exist")
