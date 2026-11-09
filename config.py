"""
MalwareShield AI — Global Configuration
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BASELINE_DIR = os.path.join(MODELS_DIR, "baseline")
DEFENDED_DIR = os.path.join(MODELS_DIR, "defended")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")

# ── Dataset ────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(PROCESSED_DIR, "demo_dataset.csv")
SAMPLE_PATH = os.path.join(SAMPLE_DIR, "sample_features.csv")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(PROCESSED_DIR, "feature_names.json")

# ── Data Generation ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_SAMPLES = 10_000
N_FEATURES = 50
MALICIOUS_RATIO = 0.40  # 40% malicious
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Feature Groups ─────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "PE Header": list(range(0, 10)),
    "Section Stats": list(range(10, 20)),
    "Import/Export": list(range(20, 30)),
    "Entropy": list(range(30, 38)),
    "String Features": list(range(38, 44)),
    "Misc": list(range(44, 50)),
}

MUTABLE_FEATURES = list(range(20, 50))   # imports, entropy, strings, misc — safe to perturb in feature space
IMMUTABLE_FEATURES = list(range(0, 20))  # PE header / section structure — not perturbed

# ── Model Hyperparameters ──────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

GB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": RANDOM_SEED,
}

LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

# ── Attack Parameters ──────────────────────────────────────────────────────────
FGSM_EPSILON = 0.1
PGD_EPSILON = 0.1
PGD_ALPHA = 0.02
PGD_ITERATIONS = 20
GA_POPULATION = 12
GA_GENERATIONS = 8
GA_MUTATION_RATE = 0.15
RANDOM_ATTACK_EPSILON = 0.1

# ── Defense Parameters ─────────────────────────────────────────────────────────
ADV_TRAINING_RATIO = 0.08   # modest adversarial augmentation for a fast, stable demo
ENSEMBLE_N_ESTIMATORS = 5

# ── Streamlit ──────────────────────────────────────────────────────────────────
APP_TITLE = "MalwareShield AI"
APP_SUBTITLE = "Adversarial Robustness Laboratory"
APP_ICON = "🛡️"
