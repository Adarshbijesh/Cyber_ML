"""
train.py — Train baseline ML models for malware detection
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, f1_score,
    precision_score, recall_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASELINE_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH,
    RF_PARAMS, GB_PARAMS, LR_PARAMS, RANDOM_SEED, RESULTS_DIR
)
from src.data_loader import split_features_labels


MODELS = {
    "random_forest": RandomForestClassifier(**RF_PARAMS),
    "gradient_boosting": GradientBoostingClassifier(**GB_PARAMS),
    "logistic_regression": LogisticRegression(**LR_PARAMS),
}


def evaluate_model(model, X, y, model_name: str = "") -> dict:
    """Compute comprehensive metrics for a fitted model."""
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
    else:
        metrics["roc_auc"] = None

    report = classification_report(y, y_pred, target_names=["Benign", "Malicious"], output_dict=True)
    metrics["report"] = report
    return metrics


def train_all_models():
    """Train all baseline models and save them to disk."""
    os.makedirs(BASELINE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train, feat_names = split_features_labels(train_df)
    X_val, y_val, _ = split_features_labels(val_df)
    X_test, y_test, _ = split_features_labels(test_df)

    all_metrics = {}
    best_val_f1 = -1
    best_model_name = None

    for name, model in MODELS.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        val_metrics = evaluate_model(model, X_val, y_val, name)
        test_metrics = evaluate_model(model, X_test, y_test, name)

        print(f"  Val  Acc={val_metrics['accuracy']:.4f} F1={val_metrics['f1']:.4f} AUC={val_metrics.get('roc_auc','N/A')}")
        print(f"  Test Acc={test_metrics['accuracy']:.4f} F1={test_metrics['f1']:.4f} AUC={test_metrics.get('roc_auc','N/A')}")

        # Save model
        model_path = os.path.join(BASELINE_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)

        all_metrics[name] = {
            "val": val_metrics,
            "test": test_metrics,
            "train_time_s": elapsed,
            "model_path": model_path,
        }

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_name = name

    # Mark best model
    all_metrics["best_model"] = best_model_name
    print(f"\nBest model (by val F1): {best_model_name}")

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "baseline_metrics.json")
    # Remove report dicts for JSON serialization (keep flat metrics)
    flat = {}
    for k, v in all_metrics.items():
        if isinstance(v, dict):
            flat[k] = {
                "val": {kk: vv for kk, vv in v.get("val", {}).items() if kk != "report"},
                "test": {kk: vv for kk, vv in v.get("test", {}).items() if kk != "report"},
                "train_time_s": v.get("train_time_s"),
                "model_path": v.get("model_path"),
            }
        else:
            flat[k] = v
    with open(metrics_path, "w") as f:
        json.dump(flat, f, indent=2)

    print(f"Metrics saved -> {metrics_path}")
    return all_metrics, feat_names


def load_model(model_name: str, model_dir: str = None):
    """Load a saved model from disk."""
    d = model_dir or BASELINE_DIR
    path = os.path.join(d, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def list_available_models(model_dir: str = None) -> list:
    """Return names of all saved baseline models."""
    d = model_dir or BASELINE_DIR
    if not os.path.exists(d):
        return []
    return [f.replace(".joblib", "") for f in os.listdir(d) if f.endswith(".joblib")]


if __name__ == "__main__":
    train_all_models()
