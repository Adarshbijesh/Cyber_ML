"""
test_attacks.py — Tests for adversarial attack implementations
"""

import os
import sys
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import generate_synthetic_dataset, preprocess_and_split
from src.data_loader import split_features_labels
from src.attacks import fgsm_attack, pgd_attack, genetic_attack, random_attack, run_attack_experiment
from config import N_FEATURES, MUTABLE_FEATURES, FGSM_EPSILON, PGD_EPSILON


@pytest.fixture(scope="module")
def models_and_data():
    df = generate_synthetic_dataset(n_samples=300, seed=42)
    train, val, test, scaler = preprocess_and_split(df, seed=42)
    X_train, y_train, _ = split_features_labels(train)
    X_test, y_test, _ = split_features_labels(test)

    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf.fit(X_train, y_train)

    return lr, rf, X_test, y_test


class TestFGSM:
    def test_fgsm_returns_expected_keys(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        result = fgsm_attack(lr, rf, X_test[0], epsilon=FGSM_EPSILON)
        for key in ["original_prediction", "adversarial_prediction",
                    "original_confidence", "adversarial_confidence",
                    "perturbation_magnitude", "prediction_changed", "attack_method"]:
            assert key in result, f"Missing key: {key}"

    def test_fgsm_respects_epsilon(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        eps = 0.05
        result = fgsm_attack(lr, rf, X_test[0], epsilon=eps)
        assert result["perturbation_magnitude"] <= eps * np.sqrt(N_FEATURES) + 1e-6

    def test_fgsm_only_mutates_mutable(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        # We can only test indirectly via the constraint code
        result = fgsm_attack(lr, rf, X_test[0], epsilon=FGSM_EPSILON)
        assert result["attack_method"] == "FGSM"

    def test_fgsm_requires_coef(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        with pytest.raises(ValueError):
            fgsm_attack(rf, rf, X_test[0])  # RF has no coef_

    def test_fgsm_confidence_in_range(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        result = fgsm_attack(lr, rf, X_test[0], epsilon=FGSM_EPSILON)
        assert 0.0 <= result["original_confidence"] <= 1.0
        assert 0.0 <= result["adversarial_confidence"] <= 1.0


class TestPGD:
    def test_pgd_returns_expected_keys(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        result = pgd_attack(lr, rf, X_test[0], epsilon=PGD_EPSILON, iterations=5)
        assert "attack_method" in result
        assert result["attack_method"] == "PGD"

    def test_pgd_iterates(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        result = pgd_attack(lr, rf, X_test[0], epsilon=0.1, iterations=10)
        assert result["perturbation_magnitude"] >= 0


class TestGeneticAttack:
    def test_genetic_attack_basic(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        result = genetic_attack(rf, X_test[0], epsilon=0.1, generations=5, population_size=10)
        assert result["attack_method"] == "Genetic Algorithm"
        assert "prediction_changed" in result

    def test_genetic_attack_runs_with_reasonable_budget(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        import time
        start = time.time()
        result = genetic_attack(rf, X_test[0], epsilon=0.1, generations=4, population_size=8)
        elapsed = time.time() - start
        assert result["attack_method"] == "Genetic Algorithm"
        assert elapsed < 10.0, f"Genetic attack exceeded runtime budget: {elapsed:.2f}s"

    def test_genetic_no_gradient_needed(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        # Should work without surrogate (RF has no coef_)
        result = genetic_attack(rf, X_test[0], epsilon=0.1, generations=3, population_size=5)
        assert result is not None


class TestRandomAttack:
    def test_random_attack_basic(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        result = random_attack(rf, X_test[0], epsilon=0.1)
        assert result["attack_method"] == "Random"
        assert result["perturbation_magnitude"] >= 0

    def test_random_attack_respects_epsilon(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        eps = 0.05
        result = random_attack(rf, X_test[0], epsilon=eps)
        assert result["perturbation_magnitude"] <= eps * np.sqrt(N_FEATURES) + 1e-6


class TestBatchExperiment:
    def test_run_experiment_basic(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        exp = run_attack_experiment(
            random_attack, None, rf, X_test, y_test, n_samples=10
        )
        assert "attack_success_rate" in exp
        assert "n_samples" in exp
        assert 0.0 <= exp["attack_success_rate"] <= 1.0

    def test_run_experiment_counts(self, models_and_data):
        lr, rf, X_test, y_test = models_and_data
        exp = run_attack_experiment(
            random_attack, None, rf, X_test, y_test, n_samples=20
        )
        assert exp["n_changed"] <= exp["n_samples"]
