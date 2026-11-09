"""
evaluation.py — Comprehensive evaluation metrics and robustness analysis
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, MUTABLE_FEATURES, PGD_EPSILON
from src.attacks import (
    fgsm_attack, pgd_attack, genetic_attack, random_attack, run_attack_experiment
)


def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          y_prob: np.ndarray = None, label: str = "") -> dict:
    """Compute a full set of classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics = {
        "label": label,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "confusion_matrix": cm.tolist(),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def evaluate_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    surrogate_model,
    epsilon: float = PGD_EPSILON,
    n_attack_samples: int = 200,
    seed: int = 42,
) -> dict:
    """
    Full robustness evaluation:
    - Clean accuracy
    - Adversarial accuracy under each attack
    - Attack success rates
    """
    # Clean metrics
    clean_preds = model.predict(X_test)
    clean_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    clean_metrics = compute_full_metrics(y_test, clean_preds, clean_probs, label="Clean")

    # Attacks
    attack_configs = {
        "FGSM": {
            "fn": fgsm_attack,
            "needs_surrogate": True,
            "kwargs": {"epsilon": epsilon},
        },
        "PGD": {
            "fn": pgd_attack,
            "needs_surrogate": True,
            "kwargs": {"epsilon": epsilon},
        },
        "GeneticAlgorithm": {
            "fn": genetic_attack,
            "needs_surrogate": False,
            "kwargs": {"epsilon": epsilon},
        },
        "Random": {
            "fn": random_attack,
            "needs_surrogate": False,
            "kwargs": {"epsilon": epsilon},
        },
    }

    attack_results = {}
    for name, cfg in attack_configs.items():
        print(f"  Running {name} attack on {n_attack_samples} samples...")
        surr = surrogate_model if cfg["needs_surrogate"] else None
        try:
            exp = run_attack_experiment(
                cfg["fn"], surr, model, X_test, y_test,
                n_samples=n_attack_samples, seed=seed, **cfg["kwargs"]
            )
            attack_results[name] = {
                "attack_success_rate": exp.get("attack_success_rate", 0),
                "n_samples": exp.get("n_samples", 0),
                "n_changed": exp.get("n_changed", 0),
                "avg_perturbation": exp.get("avg_perturbation_magnitude", 0),
            }
        except Exception as e:
            attack_results[name] = {"error": str(e), "attack_success_rate": 0}

    # Adversarial accuracy = 1 - attack_success_rate (approx)
    adv_accuracies = {}
    for name, res in attack_results.items():
        asr = res.get("attack_success_rate", 0)
        adv_accuracies[name] = max(0.0, clean_metrics["accuracy"] * (1 - asr))

    return {
        "clean_metrics": clean_metrics,
        "attack_results": attack_results,
        "adversarial_accuracies": adv_accuracies,
        "epsilon": epsilon,
    }


def compute_roc_data(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute ROC curve data points."""
    if not hasattr(model, "predict_proba"):
        return {}
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(auc),
    }


def build_comparison_table(results_dict: dict) -> pd.DataFrame:
    """Build a summary comparison DataFrame from multiple model results."""
    rows = []
    for model_name, metrics in results_dict.items():
        clean = metrics.get("clean_metrics", {})
        rows.append({
            "Model": model_name,
            "Clean Accuracy": clean.get("accuracy", 0),
            "Precision": clean.get("precision", 0),
            "Recall": clean.get("recall", 0),
            "F1-Score": clean.get("f1", 0),
            "ROC-AUC": clean.get("roc_auc", 0),
            "FPR": clean.get("false_positive_rate", 0),
            "FNR": clean.get("false_negative_rate", 0),
        })
    return pd.DataFrame(rows).set_index("Model")


def save_results(results: dict, filename: str):
    """Save results dict as JSON to experiments/results/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)

    def default_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=default_serializer)
    return path
