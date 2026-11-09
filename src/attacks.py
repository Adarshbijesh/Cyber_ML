"""
attacks.py — Adversarial attack implementations for MalwareShield AI

IMPORTANT RESEARCH NOTE:
------------------------
All attacks here operate EXCLUSIVELY in feature space. They produce
modified feature vectors for research/educational purposes only.
These are NOT functional PE file generators. Feature-space perturbations
do not translate to valid executable modifications.

Attack methods:
  1. FGSM   — Fast Gradient Sign Method (via surrogate logistic regression)
  2. PGD    — Projected Gradient Descent (constrained iterative)
  3. GA     — Genetic Algorithm feature perturbation
  4. Random — Random perturbation baseline (control experiment)
"""

import os
import sys
import numpy as np
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MUTABLE_FEATURES, FGSM_EPSILON, PGD_EPSILON, PGD_ALPHA, PGD_ITERATIONS,
    GA_POPULATION, GA_GENERATIONS, GA_MUTATION_RATE, RANDOM_ATTACK_EPSILON,
    RANDOM_SEED, N_FEATURES,
)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _clip_to_bounds(X_adv: np.ndarray, X_orig: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Clip adversarial sample so that ONLY mutable features are changed
    and perturbation magnitude stays within epsilon (L-inf norm).
    """
    X_clipped = X_orig.copy()
    for i in MUTABLE_FEATURES:
        if i < X_adv.shape[-1]:
            delta = np.clip(X_adv[..., i] - X_orig[..., i], -epsilon, epsilon)
            X_clipped[..., i] = X_orig[..., i] + delta
    return X_clipped


def _record_result(x_orig, x_adv, model, orig_pred, orig_conf) -> dict:
    """Compute attack result metrics."""
    x_adv_2d = x_adv.reshape(1, -1)
    adv_pred = int(model.predict(x_adv_2d)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_adv_2d)[0]
        adv_conf = float(probs[adv_pred])
    else:
        adv_conf = 1.0

    perturbation_mag = float(np.linalg.norm(x_adv - x_orig))
    prediction_changed = orig_pred != adv_pred

    return {
        "original_prediction": orig_pred,
        "adversarial_prediction": adv_pred,
        "original_confidence": orig_conf,
        "adversarial_confidence": adv_conf,
        "perturbation_magnitude": perturbation_mag,
        "prediction_changed": prediction_changed,
        "original_label_name": "Malicious" if orig_pred == 1 else "Benign",
        "adversarial_label_name": "Malicious" if adv_pred == 1 else "Benign",
    }


# ── FGSM ───────────────────────────────────────────────────────────────────────

def fgsm_attack(
    surrogate_model,       # Must have coef_ (Logistic Regression)
    target_model,          # The model being attacked (can be any classifier)
    X: np.ndarray,         # Original features (1D or 2D)
    epsilon: float = FGSM_EPSILON,
) -> dict:
    """
    Fast Gradient Sign Method attack via a differentiable surrogate model.
    
    Methodology:
    - We fit a LogisticRegression surrogate (already trained).
    - We compute gradients of the surrogate's loss w.r.t. X.
    - We perturb only MUTABLE_FEATURES in the direction that increases
      the surrogate's probability of predicting class 1 (malicious).
    - Perturbation is clipped to [-epsilon, +epsilon] per feature.
    - Attack success is measured on the TARGET model.
    
    NOTE: Gradient transfer from surrogate to tree model is an approximation;
    real-world transfer attack success rates vary.
    """
    if not hasattr(surrogate_model, "coef_"):
        raise ValueError("FGSM requires a model with coef_ attribute (LogisticRegression).")

    x = np.array(X, dtype=np.float64).flatten()
    x_2d = x.reshape(1, -1)

    # Original prediction on target model
    orig_pred = int(target_model.predict(x_2d)[0])
    if hasattr(target_model, "predict_proba"):
        orig_probs = target_model.predict_proba(x_2d)[0]
        orig_conf = float(orig_probs[orig_pred])
    else:
        orig_conf = 1.0

    # Compute gradient sign from surrogate
    # gradient of cross-entropy loss w.r.t. input = coef_ * (pred - true_label)
    # For attacking toward malicious (class=1): sign(coef_[0]) increases P(malicious)
    grad_sign = np.sign(surrogate_model.coef_[0])

    # Apply perturbation only to mutable features
    x_adv = x.copy()
    for i in MUTABLE_FEATURES:
        if i < len(x_adv):
            x_adv[i] = x[i] + epsilon * grad_sign[i]

    # Clip back to epsilon-ball
    x_adv = _clip_to_bounds(x_adv, x, epsilon)

    result = _record_result(x, x_adv, target_model, orig_pred, orig_conf)
    result["attack_method"] = "FGSM"
    result["epsilon"] = epsilon
    result["perturbed_features"] = MUTABLE_FEATURES[:10]  # list first 10 for display
    return result


# ── PGD ────────────────────────────────────────────────────────────────────────

def pgd_attack(
    surrogate_model,
    target_model,
    X: np.ndarray,
    epsilon: float = PGD_EPSILON,
    alpha: float = PGD_ALPHA,
    iterations: int = PGD_ITERATIONS,
) -> dict:
    """
    Projected Gradient Descent attack (iterative FGSM).
    
    PGD iteratively applies small FGSM steps and projects back
    into the epsilon-ball after each step.
    """
    if not hasattr(surrogate_model, "coef_"):
        raise ValueError("PGD requires a model with coef_ attribute.")

    x = np.array(X, dtype=np.float64).flatten()
    x_2d = x.reshape(1, -1)

    orig_pred = int(target_model.predict(x_2d)[0])
    if hasattr(target_model, "predict_proba"):
        orig_probs = target_model.predict_proba(x_2d)[0]
        orig_conf = float(orig_probs[orig_pred])
    else:
        orig_conf = 1.0

    grad_sign = np.sign(surrogate_model.coef_[0])

    # Start from random point in epsilon-ball
    rng = np.random.default_rng(RANDOM_SEED)
    x_adv = x.copy()
    noise = rng.uniform(-epsilon, epsilon, size=x.shape)
    for i in MUTABLE_FEATURES:
        if i < len(x_adv):
            x_adv[i] = x[i] + noise[i]

    for _ in range(iterations):
        # Gradient step
        x_adv_new = x_adv.copy()
        for i in MUTABLE_FEATURES:
            if i < len(x_adv_new):
                x_adv_new[i] = x_adv[i] + alpha * grad_sign[i]
        # Project back to epsilon-ball
        x_adv = _clip_to_bounds(x_adv_new, x, epsilon)

    result = _record_result(x, x_adv, target_model, orig_pred, orig_conf)
    result["attack_method"] = "PGD"
    result["epsilon"] = epsilon
    result["iterations"] = iterations
    return result


# ── Genetic Algorithm ──────────────────────────────────────────────────────────

def genetic_attack(
    target_model,
    X: np.ndarray,
    epsilon: float = PGD_EPSILON,
    population_size: int = GA_POPULATION,
    generations: int = GA_GENERATIONS,
    mutation_rate: float = GA_MUTATION_RATE,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Genetic algorithm-based feature-space adversarial attack.
    
    Operates in feature space only. Does not require gradient information.
    Suitable for attacking black-box models (tree ensembles, etc.).
    
    Fitness function: probability of being classified as malicious (class=1).
    """
    rng = np.random.default_rng(seed)
    x = np.array(X, dtype=np.float64).flatten()
    x_2d = x.reshape(1, -1)
    n_features = len(x)

    orig_pred = int(target_model.predict(x_2d)[0])
    if hasattr(target_model, "predict_proba"):
        orig_probs = target_model.predict_proba(x_2d)[0]
        orig_conf = float(orig_probs[orig_pred])
    else:
        orig_conf = float(orig_pred)

    mutable = [i for i in MUTABLE_FEATURES if i < n_features]

    def fitness(individual):
        """Maximize probability of class 1 (malicious)."""
        ind_2d = individual.reshape(1, -1)
        if hasattr(target_model, "predict_proba"):
            return target_model.predict_proba(ind_2d)[0][1]
        return float(target_model.predict(ind_2d)[0])

    def mutate(individual):
        """Randomly perturb mutable features."""
        ind = individual.copy()
        for i in mutable:
            if rng.random() < mutation_rate:
                delta = rng.uniform(-epsilon, epsilon)
                ind[i] = np.clip(x[i] + np.clip(ind[i] - x[i] + delta, -epsilon, epsilon), -1e9, 1e9)
        return ind

    def crossover(p1, p2):
        """Single-point crossover on mutable features."""
        child = p1.copy()
        pt = rng.integers(0, len(mutable))
        for i in mutable[pt:]:
            child[i] = p2[i]
        return child

    # Initialize population
    population = []
    for _ in range(population_size):
        ind = x.copy()
        for i in mutable:
            ind[i] = x[i] + rng.uniform(-epsilon * 0.5, epsilon * 0.5)
        population.append(ind)

    best_individual = x.copy()
    best_fitness = fitness(x)

    for gen in range(generations):
        scores = [fitness(ind) for ind in population]
        sorted_pop = [ind for _, ind in sorted(zip(scores, population), key=lambda t: -t[0])]

        # Track best
        if scores[0] > best_fitness or gen == 0:
            best_fitness = max(scores)
            best_individual = sorted_pop[0].copy()

        # Very strong adversarial candidates can be accepted early without
        # spending the full search budget.
        if best_fitness >= 0.99:
            break

        # Early stop if attack succeeds
        if int(target_model.predict(best_individual.reshape(1, -1))[0]) != orig_pred:
            break

        # Selection + reproduction
        elite = sorted_pop[: max(2, population_size // 5)]
        new_pop = elite[:]
        while len(new_pop) < population_size:
            p1, p2 = rng.choice(len(elite), 2, replace=False)
            child = crossover(elite[p1], elite[p2])
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

    # Clip final individual
    x_adv = _clip_to_bounds(best_individual, x, epsilon)

    result = _record_result(x, x_adv, target_model, orig_pred, orig_conf)
    result["attack_method"] = "Genetic Algorithm"
    result["epsilon"] = epsilon
    result["generations"] = generations
    return result


# ── Random Perturbation (baseline) ────────────────────────────────────────────

def random_attack(
    target_model,
    X: np.ndarray,
    epsilon: float = RANDOM_ATTACK_EPSILON,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Random perturbation baseline (control experiment).
    
    Adds uniform random noise to mutable features within epsilon.
    Not directed — serves as a lower bound for attack success rates.
    """
    rng = np.random.default_rng(seed)
    x = np.array(X, dtype=np.float64).flatten()
    x_2d = x.reshape(1, -1)

    orig_pred = int(target_model.predict(x_2d)[0])
    if hasattr(target_model, "predict_proba"):
        orig_probs = target_model.predict_proba(x_2d)[0]
        orig_conf = float(orig_probs[orig_pred])
    else:
        orig_conf = float(orig_pred)

    x_adv = x.copy()
    for i in MUTABLE_FEATURES:
        if i < len(x_adv):
            x_adv[i] = x[i] + rng.uniform(-epsilon, epsilon)

    result = _record_result(x, x_adv, target_model, orig_pred, orig_conf)
    result["attack_method"] = "Random"
    result["epsilon"] = epsilon
    return result


# ── Batch experiment runner ────────────────────────────────────────────────────

def run_attack_experiment(
    attack_fn,
    surrogate_model,
    target_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int = 200,
    seed: int = RANDOM_SEED,
    **attack_kwargs,
) -> dict:
    """
    Run an attack on `n_samples` test samples and collect aggregate statistics.
    
    Only attacks samples where the original prediction matches the true label
    (correctly classified samples), to get meaningful attack success rate.
    """
    rng = np.random.default_rng(seed)

    # Find correctly classified malicious samples (most interesting targets)
    preds = target_model.predict(X_test)
    correct_mal_idx = np.where((preds == 1) & (y_test == 1))[0]
    correct_ben_idx = np.where((preds == 0) & (y_test == 0))[0]

    # Mix malicious and benign
    n_mal = min(n_samples // 2, len(correct_mal_idx))
    n_ben = min(n_samples - n_mal, len(correct_ben_idx))
    selected = np.concatenate([
        rng.choice(correct_mal_idx, n_mal, replace=False),
        rng.choice(correct_ben_idx, n_ben, replace=False),
    ])

    results = []
    for idx in selected:
        x = X_test[idx]
        try:
            if surrogate_model is not None:
                res = attack_fn(surrogate_model, target_model, x, **attack_kwargs)
            else:
                res = attack_fn(target_model, x, **attack_kwargs)
            results.append(res)
        except Exception as e:
            warnings.warn(f"Attack failed on sample {idx}: {e}")

    if not results:
        return {"error": "No results collected"}

    n_changed = sum(r["prediction_changed"] for r in results)
    attack_success_rate = n_changed / len(results)
    avg_perturbation = float(np.mean([r["perturbation_magnitude"] for r in results]))
    avg_orig_conf = float(np.mean([r["original_confidence"] for r in results]))
    avg_adv_conf = float(np.mean([r["adversarial_confidence"] for r in results]))

    return {
        "n_samples": len(results),
        "n_changed": int(n_changed),
        "attack_success_rate": float(attack_success_rate),
        "avg_perturbation_magnitude": avg_perturbation,
        "avg_original_confidence": avg_orig_conf,
        "avg_adversarial_confidence": avg_adv_conf,
        "results": results,
        "attack_method": results[0].get("attack_method", "Unknown"),
    }
