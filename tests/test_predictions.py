"""
test_predictions.py — Integration tests for end-to-end prediction pipeline
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import generate_synthetic_dataset, preprocess_and_split
from src.data_loader import split_features_labels, get_class_distribution
from src.predict import (
    predict_single, predict_batch, validate_feature_vector, prepare_feature_frame,
    get_feature_importance,
)
from src.evaluation import compute_full_metrics
from config import N_FEATURES, BASELINE_DIR, RESULTS_DIR
import joblib


@pytest.fixture(scope="module")
def full_pipeline():
    """Run full pipeline on a small dataset."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    df = generate_synthetic_dataset(n_samples=500, seed=42)
    train, val, test, scaler = preprocess_and_split(df, seed=42)
    X_train, y_train, feat_names = split_features_labels(train)
    X_test, y_test, _ = split_features_labels(test)

    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=30, random_state=42)
    rf.fit(X_train, y_train)

    return lr, rf, X_train, y_train, X_test, y_test, feat_names, scaler


class TestEndToEndPipeline:
    def test_full_pipeline_runs(self, full_pipeline):
        lr, rf, X_train, y_train, X_test, y_test, feat_names, scaler = full_pipeline
        assert lr is not None
        assert rf is not None

    def test_prediction_on_all_test(self, full_pipeline):
        lr, rf, X_train, y_train, X_test, y_test, feat_names, scaler = full_pipeline
        result = predict_batch(lr, X_test)
        assert len(result["labels"]) == len(X_test)

    def test_metrics_computation(self, full_pipeline):
        lr, rf, X_train, y_train, X_test, y_test, feat_names, scaler = full_pipeline
        preds = lr.predict(X_test)
        probs = lr.predict_proba(X_test)[:, 1]
        metrics = compute_full_metrics(y_test, preds, probs)
        assert metrics["accuracy"] > 0.5  # Should be significantly better than random
        assert "confusion_matrix" in metrics

    def test_feature_importance_rf(self, full_pipeline):
        lr, rf, X_train, y_train, X_test, y_test, feat_names, scaler = full_pipeline
        df_imp = get_feature_importance(rf, feat_names)
        assert len(df_imp) == N_FEATURES
        assert "feature" in df_imp.columns
        assert "importance" in df_imp.columns

    def test_feature_importance_lr(self, full_pipeline):
        lr, rf, X_train, y_train, X_test, y_test, feat_names, scaler = full_pipeline
        df_imp = get_feature_importance(lr, feat_names)
        assert len(df_imp) == N_FEATURES


class TestDataLoader:
    def test_class_distribution(self):
        df = generate_synthetic_dataset(n_samples=200)
        dist = get_class_distribution(df)
        assert dist["total"] == 200
        assert dist["benign"] + dist["malicious"] == 200
        assert dist["malicious"] > 0
        assert dist["benign"] > 0

    def test_feature_label_split(self):
        df = generate_synthetic_dataset(n_samples=100)
        train, _, _, _ = preprocess_and_split(df)
        X, y, feat_names = split_features_labels(train)
        assert X.shape[1] == N_FEATURES
        assert len(y) == len(X)
        assert len(feat_names) == N_FEATURES


class TestInputValidation:
    def test_valid_input_accepted(self):
        x = np.zeros(N_FEATURES)
        valid, msg = validate_feature_vector(x, N_FEATURES)
        assert valid is True

    def test_wrong_feature_count_rejected(self):
        x = np.zeros(N_FEATURES - 5)
        valid, msg = validate_feature_vector(x, N_FEATURES)
        assert valid is False

    def test_nan_rejected(self):
        x = np.ones(N_FEATURES)
        x[5] = np.nan
        valid, msg = validate_feature_vector(x, N_FEATURES)
        assert valid is False

    def test_inf_rejected(self):
        x = np.ones(N_FEATURES)
        x[10] = np.inf
        valid, msg = validate_feature_vector(x, N_FEATURES)
        assert valid is False

    def test_2d_input_handled(self):
        x = np.zeros((1, N_FEATURES))
        valid, msg = validate_feature_vector(x, N_FEATURES)
        assert valid is True

    def test_feature_value_csv_converted_to_one_ordered_sample(self):
        names = ["feature_a", "feature_b", "feature_c"]
        data = pd.DataFrame({
            "feature": ["feature_c", "feature_a", "feature_b"],
            "value": [3, 1, 2],
        })

        result = prepare_feature_frame(data, names, len(names))

        assert result.columns.tolist() == names
        assert result.iloc[0].tolist() == [1, 2, 3]

    def test_wide_csv_columns_reordered_to_model_schema(self):
        names = ["feature_a", "feature_b", "feature_c"]
        data = pd.DataFrame({"feature_c": [3], "feature_a": [1], "feature_b": [2]})

        result = prepare_feature_frame(data, names, len(names))

        assert result.columns.tolist() == names
        assert result.iloc[0].tolist() == [1, 2, 3]

    def test_unmatched_partial_csv_reports_required_model_width(self):
        data = pd.DataFrame({"first": [1], "second": [2]})

        with pytest.raises(ValueError, match="requires 3 features"):
            prepare_feature_frame(data, ["feature_a", "feature_b", "feature_c"], 3)


class TestCSVExport:
    def test_results_csv_exists_after_experiments(self):
        csv_path = os.path.join(RESULTS_DIR, "attack_results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            assert len(df) > 0
            assert "Attack Success Rate" in df.columns
        else:
            pytest.skip("Attack results CSV not yet generated")

    def test_metrics_json_loadable(self):
        json_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
        if os.path.exists(json_path):
            import json
            with open(json_path) as f:
                data = json.load(f)
            assert "best_model" in data
        else:
            pytest.skip("Baseline metrics JSON not yet generated")
