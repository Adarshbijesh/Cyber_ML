"""
defense.py — Defense mechanisms for adversarial robustness

Implements:
1. Adversarial Training
2. Feature Normalization / Input Validation defense
3. Ensemble-Based Detection
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DEFENDED_DIR, BASELINE_DIR, RESULTS_DIR,
    RF_PARAMS, LR_PARAMS, GB_PARAMS,
    MUTABLE_FEATURES, PGD_EPSILON, PGD_ALPHA, PGD_ITERATIONS,
    ADV_TRAINING_RATIO, ENSEMBLE_N_ESTIMATORS, RANDOM_SEED, N_FEATURES,
)
from src.attacks import pgd_attack, random_attack


# ── 1. Adversarial Training ────────────────────────────────────────────────────

def generate_adversarial_examples(
    surrogate_model,
    target_model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float = PGD_EPSILON,
    n_adv: int = None,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Generate adversarial examples by applying PGD to a subset of training samples.
    Returns augmented (X_aug, y_aug).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    n_adv = n_adv or max(1, int(n * ADV_TRAINING_RATIO))
    n_adv = min(n_adv, n)

    idx = rng.choice(n, n_adv, replace=False)
    X_adv_list = []
    for i in idx:
        x = X[i]
        try:
            res = pgd_attack(surrogate_model, target_model, x, epsilon=epsilon)
            x_adv = x.copy()
            # Reconstruct adversarial sample from result (use perturbation direction)
            # We regenerate it via direct computation since pgd_attack returns metrics
            from config import MUTABLE_FEATURES, PGD_ALPHA, PGD_ITERATIONS
            grad_sign = np.sign(surrogate_model.coef_[0])
            x_adv2 = x.copy()
            for _ in range(PGD_ITERATIONS):
                for feat_i in MUTABLE_FEATURES:
                    if feat_i < len(x_adv2):
                        x_adv2[feat_i] = np.clip(
                            x_adv2[feat_i] + PGD_ALPHA * grad_sign[feat_i],
                            x[feat_i] - epsilon,
                            x[feat_i] + epsilon,
                        )
            X_adv_list.append(x_adv2)
        except Exception:
            X_adv_list.append(x)

    X_adv = np.array(X_adv_list)
    y_adv = y[idx]

    # Augment: original + adversarial
    X_aug = np.vstack([X, X_adv])
    y_aug = np.concatenate([y, y_adv])

    return X_aug, y_aug


def train_adversarially(
    X_train: np.ndarray,
    y_train: np.ndarray,
    surrogate_model,
    seed: int = RANDOM_SEED,
) -> RandomForestClassifier:
    """
    Train a Random Forest with adversarial data augmentation.
    """
    # Build a temporary RF to attack
    temp_rf = RandomForestClassifier(**RF_PARAMS)
    temp_rf.fit(X_train, y_train)

    print("  Generating adversarial training examples...")
    X_aug, y_aug = generate_adversarial_examples(
        surrogate_model, temp_rf, X_train, y_train, seed=seed
    )
    print(f"  Augmented training set: {len(X_aug)} samples (was {len(X_train)})")

    # Train robust model on augmented data
    adv_rf = RandomForestClassifier(**RF_PARAMS)
    adv_rf.fit(X_aug, y_aug)
    return adv_rf


# ── 2. Feature Normalization / Input Validation ────────────────────────────────

class InputValidationDefense:
    """
    Defense via feature clipping and anomaly detection.
    
    - Clips features to observed training distribution (mean ± k*std).
    - Flags inputs that deviate excessively from normal bounds as suspicious.
    - Applies clipping before prediction.
    """

    def __init__(self, k: float = 3.0):
        self.k = k
        self.lower_bounds = None
        self.upper_bounds = None
        self.base_model = None

    def fit(self, model, X_train: np.ndarray):
        """Fit bounds from training data."""
        self.base_model = model
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        self.lower_bounds = mean - self.k * std
        self.upper_bounds = mean + self.k * std
        return self

    def sanitize(self, X: np.ndarray) -> np.ndarray:
        """Clip features to observed bounds."""
        X_clean = X.copy()
        X_clean = np.clip(X_clean, self.lower_bounds, self.upper_bounds)
        return X_clean

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_san = self.sanitize(X)
        return self.base_model.predict(X_san)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_san = self.sanitize(X)
        if hasattr(self.base_model, "predict_proba"):
            return self.base_model.predict_proba(X_san)
        preds = self.predict(X_san)
        return np.column_stack([1 - preds, preds]).astype(float)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Fraction of features outside training bounds — higher = more suspicious."""
        below = (X < self.lower_bounds).mean(axis=1)
        above = (X > self.upper_bounds).mean(axis=1)
        return below + above


# ── 3. Ensemble-Based Detection ────────────────────────────────────────────────

def build_ensemble_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = RANDOM_SEED,
):
    """
    Build a soft-voting ensemble combining RandomForest, GradientBoosting, LogisticRegression.
    
    Ensemble reduces attack surface by requiring adversarial examples to fool
    multiple diverse classifiers simultaneously.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                    random_state=seed)
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                            random_state=seed)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    return ensemble


# ── Main defense training routine ──────────────────────────────────────────────

def train_all_defenses(X_train, y_train, X_val, y_val, X_test, y_test,
                        surrogate_model, baseline_rf):
    """Train and evaluate all three defense models."""
    os.makedirs(DEFENDED_DIR, exist_ok=True)

    defense_results = {}

    # 1. Adversarial Training
    print("\n[1/3] Adversarial Training...")
    adv_rf = train_adversarially(X_train, y_train, surrogate_model)
    joblib.dump(adv_rf, os.path.join(DEFENDED_DIR, "adversarial_training_rf.joblib"))
    adv_test_acc = accuracy_score(y_test, adv_rf.predict(X_test))
    adv_test_f1 = f1_score(y_test, adv_rf.predict(X_test))
    adv_test_auc = roc_auc_score(y_test, adv_rf.predict_proba(X_test)[:, 1])
    defense_results["adversarial_training"] = {
        "clean_accuracy": adv_test_acc,
        "clean_f1": adv_test_f1,
        "clean_auc": adv_test_auc,
    }
    print(f"  Test Acc={adv_test_acc:.4f} F1={adv_test_f1:.4f} AUC={adv_test_auc:.4f}")

    # 2. Input Validation Defense (wraps baseline RF)
    print("\n[2/3] Input Validation Defense...")
    iv_defense = InputValidationDefense(k=3.0)
    iv_defense.fit(baseline_rf, X_train)
    iv_preds = iv_defense.predict(X_test)
    iv_probs = iv_defense.predict_proba(X_test)
    iv_acc = accuracy_score(y_test, iv_preds)
    iv_f1 = f1_score(y_test, iv_preds)
    iv_auc = roc_auc_score(y_test, iv_probs[:, 1])
    joblib.dump(iv_defense, os.path.join(DEFENDED_DIR, "input_validation.joblib"))
    defense_results["input_validation"] = {
        "clean_accuracy": iv_acc,
        "clean_f1": iv_f1,
        "clean_auc": iv_auc,
    }
    print(f"  Test Acc={iv_acc:.4f} F1={iv_f1:.4f} AUC={iv_auc:.4f}")

    # 3. Ensemble Defense
    print("\n[3/3] Ensemble-Based Defense...")
    ensemble = build_ensemble_model(X_train, y_train)
    joblib.dump(ensemble, os.path.join(DEFENDED_DIR, "ensemble_defense.joblib"))
    ens_preds = ensemble.predict(X_test)
    ens_probs = ensemble.predict_proba(X_test)
    ens_acc = accuracy_score(y_test, ens_preds)
    ens_f1 = f1_score(y_test, ens_preds)
    ens_auc = roc_auc_score(y_test, ens_probs[:, 1])
    defense_results["ensemble"] = {
        "clean_accuracy": ens_acc,
        "clean_f1": ens_f1,
        "clean_auc": ens_auc,
    }
    print(f"  Test Acc={ens_acc:.4f} F1={ens_f1:.4f} AUC={ens_auc:.4f}")

    return defense_results, adv_rf, iv_defense, ensemble
