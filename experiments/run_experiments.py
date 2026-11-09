"""
run_experiments.py — End-to-end pipeline: preprocess → train → attack → defend → evaluate
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BASELINE_DIR, DEFENDED_DIR, RESULTS_DIR, TRAIN_PATH, VAL_PATH, TEST_PATH,
    SCALER_PATH, RANDOM_SEED, PGD_EPSILON,
)
from src.data_loader import split_features_labels
from src.preprocessing import run_preprocessing
from src.train import train_all_models, load_model
from src.evaluation import (
    evaluate_robustness, compute_roc_data, build_comparison_table,
    compute_full_metrics, save_results,
)
from src.defense import train_all_defenses
from src.attacks import run_attack_experiment, pgd_attack, fgsm_attack, genetic_attack, random_attack


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Phase 1: Preprocess ────────────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1: Preprocessing")
    print("=" * 60)
    if not os.path.exists(TRAIN_PATH):
        run_preprocessing()
    else:
        print("  Splits already exist, skipping generation.")

    # ── Phase 2: Train Baseline Models ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Training Baseline Models")
    print("=" * 60)
    models_exist = all(
        os.path.exists(os.path.join(BASELINE_DIR, f"{n}.joblib"))
        for n in ["random_forest", "gradient_boosting", "logistic_regression"]
    )
    if not models_exist:
        all_metrics, feat_names = train_all_models()
    else:
        print("  Baseline models already exist, loading...")

    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train, feat_names = split_features_labels(train_df)
    X_val, y_val, _ = split_features_labels(val_df)
    X_test, y_test, _ = split_features_labels(test_df)

    # Load models
    rf = load_model("random_forest")
    gb = load_model("gradient_boosting")
    lr = load_model("logistic_regression")
    models = {"Random Forest": rf, "Gradient Boosting": gb, "Logistic Regression": lr}
    surrogate = lr  # LR is differentiable, used as surrogate for gradient attacks

    # ── Phase 3: ROC data ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Computing ROC Curves")
    print("=" * 60)
    roc_data = {}
    for name, model in models.items():
        roc_data[name] = compute_roc_data(model, X_test, y_test)
        print(f"  {name}: AUC = {roc_data[name].get('auc', 'N/A'):.4f}")
    save_results(roc_data, "roc_data.json")

    # ── Phase 4: Adversarial Attack Experiments ────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4: Adversarial Attack Experiments")
    print("=" * 60)
    n_attack_samples = 40
    ga_attack_samples = min(12, n_attack_samples)

    attack_experiment_results = {}
    for model_name, model in [("Random Forest", rf), ("Logistic Regression", lr)]:
        print(f"\n  Attacking {model_name}...")
        attack_experiment_results[model_name] = {}

        # FGSM
        print("    FGSM...")
        try:
            res = run_attack_experiment(fgsm_attack, surrogate, model, X_test, y_test,
                                        n_samples=n_attack_samples, epsilon=PGD_EPSILON)
            attack_experiment_results[model_name]["FGSM"] = {
                "attack_success_rate": res.get("attack_success_rate"),
                "n_samples": res.get("n_samples"),
                "avg_perturbation": res.get("avg_perturbation_magnitude"),
            }
            print(f"    ASR={res.get('attack_success_rate', 0):.3f}")
        except Exception as e:
            print(f"    FGSM error: {e}")

        # PGD
        print("    PGD...")
        try:
            res = run_attack_experiment(pgd_attack, surrogate, model, X_test, y_test,
                                        n_samples=n_attack_samples, epsilon=PGD_EPSILON)
            attack_experiment_results[model_name]["PGD"] = {
                "attack_success_rate": res.get("attack_success_rate"),
                "n_samples": res.get("n_samples"),
                "avg_perturbation": res.get("avg_perturbation_magnitude"),
            }
            print(f"    ASR={res.get('attack_success_rate', 0):.3f}")
        except Exception as e:
            print(f"    PGD error: {e}")

        # Genetic
        print("    Genetic Algorithm...")
        try:
            res = run_attack_experiment(genetic_attack, None, model, X_test, y_test,
                                        n_samples=ga_attack_samples, epsilon=PGD_EPSILON)
            attack_experiment_results[model_name]["GeneticAlgorithm"] = {
                "attack_success_rate": res.get("attack_success_rate"),
                "n_samples": res.get("n_samples"),
                "avg_perturbation": res.get("avg_perturbation_magnitude"),
            }
            print(f"    ASR={res.get('attack_success_rate', 0):.3f}")
        except Exception as e:
            print(f"    GA error: {e}")

        # Random
        print("    Random...")
        try:
            res = run_attack_experiment(random_attack, None, model, X_test, y_test,
                                        n_samples=n_attack_samples, epsilon=PGD_EPSILON)
            attack_experiment_results[model_name]["Random"] = {
                "attack_success_rate": res.get("attack_success_rate"),
                "n_samples": res.get("n_samples"),
                "avg_perturbation": res.get("avg_perturbation_magnitude"),
            }
            print(f"    ASR={res.get('attack_success_rate', 0):.3f}")
        except Exception as e:
            print(f"    Random error: {e}")

    save_results(attack_experiment_results, "attack_results.json")

    # ── Phase 5: Train Defenses ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 5: Training Defense Models")
    print("=" * 60)
    defended_exists = os.path.exists(os.path.join(DEFENDED_DIR, "adversarial_training_rf.joblib"))
    if not defended_exists:
        defense_metrics, adv_rf, iv_defense, ensemble = train_all_defenses(
            X_train, y_train, X_val, y_val, X_test, y_test, surrogate, rf
        )
        save_results(defense_metrics, "defense_metrics.json")
    else:
        print("  Defense models already exist.")
        adv_rf = joblib.load(os.path.join(DEFENDED_DIR, "adversarial_training_rf.joblib"))
        iv_defense = joblib.load(os.path.join(DEFENDED_DIR, "input_validation.joblib"))
        ensemble = joblib.load(os.path.join(DEFENDED_DIR, "ensemble_defense.joblib"))

    # ── Phase 6: Robustness Evaluation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 6: Robustness Evaluation")
    print("=" * 60)

    all_model_results = {}
    for model_name, model in [("Random Forest (Baseline)", rf),
                               ("Adversarial Training RF", adv_rf),
                               ("Ensemble Defense", ensemble)]:
        print(f"\n  Evaluating {model_name}...")
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        clean_m = compute_full_metrics(y_test, preds, probs, label=model_name)
        all_model_results[model_name] = {"clean_metrics": clean_m}

    save_results(all_model_results, "robustness_evaluation.json")

    # ── Phase 7: Save combined summary ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 7: Saving Experiment Summary")
    print("=" * 60)

    summary = {
        "dataset": {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
        "baseline_models": {
            name: {
                "test_accuracy": float(np.mean(model.predict(X_test) == y_test)),
            }
            for name, model in models.items()
        },
        "attack_results": attack_experiment_results,
        "defense_clean_accuracy": {
            name: float(np.mean(m.predict(X_test) == y_test))
            for name, m in [("Adversarial Training RF", adv_rf),
                             ("Ensemble Defense", ensemble)]
        },
    }
    save_results(summary, "experiment_summary.json")

    # Also save as CSV
    csv_rows = []
    for model_name, attacks in attack_experiment_results.items():
        for atk_name, res in attacks.items():
            csv_rows.append({
                "Model": model_name,
                "Attack": atk_name,
                "Attack Success Rate": res.get("attack_success_rate", 0),
                "N Samples": res.get("n_samples", 0),
                "Avg Perturbation": res.get("avg_perturbation", 0),
            })
    pd.DataFrame(csv_rows).to_csv(os.path.join(RESULTS_DIR, "attack_results.csv"), index=False)

    print("\n✅ All experiments complete!")
    print(f"   Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
