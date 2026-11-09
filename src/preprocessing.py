"""
preprocessing.py — Synthetic dataset generation and feature preprocessing
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATASET_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH, SCALER_PATH,
    FEATURE_NAMES_PATH, SAMPLE_PATH, PROCESSED_DIR, SAMPLE_DIR,
    RANDOM_SEED, N_SAMPLES, N_FEATURES, MALICIOUS_RATIO,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)


def generate_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    malicious_ratio: float = MALICIOUS_RATIO,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic PE-feature dataset with realistic distributions.
    
    Feature groups (all float32):
      0-9  : PE Header features (e.g. image size, num sections, entry RVA)
      10-19: Section statistics (mean/std entropy per section)
      20-24: Import counts (DLL counts, function counts)
      25-29: Export counts
      30-37: Entropy values (overall, per-section)
      38-43: String-based features (URL count, registry key count, etc.)
      44-49: Misc (resource count, overlay presence, TLS, etc.)
    
    NOTE: This is a SYNTHETIC DEMONSTRATION dataset. It is NOT real malware data.
    Feature distributions are designed to resemble PE analysis outputs for
    educational purposes only.
    """
    rng = np.random.default_rng(seed)
    n_mal = int(n_samples * malicious_ratio)
    n_ben = n_samples - n_mal

    def benign_features():
        x = np.zeros((n_ben, n_features), dtype=np.float32)
        # PE Header
        x[:, 0] = rng.integers(1, 10, n_ben)           # num sections
        x[:, 1] = rng.uniform(0, 1e7, n_ben)            # image size
        x[:, 2] = rng.uniform(0, 2**32, n_ben)          # entry RVA
        x[:, 3] = rng.uniform(512, 65536, n_ben)        # section alignment
        x[:, 4] = rng.choice([0, 1], n_ben, p=[0.8, 0.2])  # debug stripped
        x[:, 5] = rng.uniform(0, 100, n_ben)             # compile time (days ago proxy)
        x[:, 6] = rng.uniform(0, 5, n_ben)               # subsystem
        x[:, 7] = rng.uniform(0, 1e6, n_ben)             # size of code
        x[:, 8] = rng.uniform(0, 5e5, n_ben)             # size of init data
        x[:, 9] = rng.uniform(0, 5e5, n_ben)             # size of uninit data
        # Section stats
        x[:, 10:20] = rng.uniform(3.5, 6.5, (n_ben, 10))  # section entropy (benign: moderate)
        # Imports/Exports
        x[:, 20] = rng.integers(2, 30, n_ben)            # num imported DLLs
        x[:, 21] = rng.integers(5, 200, n_ben)           # num imported functions
        x[:, 22] = rng.integers(0, 20, n_ben)            # num exported functions
        x[:, 23] = rng.uniform(0, 10, n_ben)             # import entropy
        x[:, 24] = rng.uniform(0, 5, n_ben)              # export entropy
        # More import/export features
        x[:, 25:30] = rng.uniform(0, 50, (n_ben, 5))
        # Entropy
        x[:, 30:38] = rng.uniform(4.0, 7.0, (n_ben, 8))  # benign: moderate entropy
        # String features
        x[:, 38] = rng.integers(0, 5, n_ben)             # suspicious URL count
        x[:, 39] = rng.integers(0, 10, n_ben)            # registry key refs
        x[:, 40] = rng.integers(0, 5, n_ben)             # crypto API refs
        x[:, 41] = rng.integers(0, 3, n_ben)             # packing indicators
        x[:, 42] = rng.uniform(0, 1, n_ben)              # printable string ratio
        x[:, 43] = rng.uniform(0, 500, n_ben)            # avg string length
        # Misc
        x[:, 44] = rng.integers(0, 20, n_ben)            # resource count
        x[:, 45] = rng.choice([0, 1], n_ben, p=[0.9, 0.1])  # overlay present
        x[:, 46] = rng.choice([0, 1], n_ben, p=[0.95, 0.05]) # TLS callbacks
        x[:, 47] = rng.uniform(0, 1, n_ben)              # signature score
        x[:, 48] = rng.integers(0, 100, n_ben)           # API call depth proxy
        x[:, 49] = rng.uniform(0, 1, n_ben)              # behavioral similarity score
        return x

    def malicious_features():
        x = np.zeros((n_mal, n_features), dtype=np.float32)
        # PE Header (similar structure, slight differences)
        x[:, 0] = rng.integers(1, 15, n_mal)
        x[:, 1] = rng.uniform(0, 2e7, n_mal)
        x[:, 2] = rng.uniform(0, 2**32, n_mal)
        x[:, 3] = rng.uniform(512, 65536, n_mal)
        x[:, 4] = rng.choice([0, 1], n_mal, p=[0.4, 0.6])  # more often stripped
        x[:, 5] = rng.uniform(0, 500, n_mal)               # older compile times
        x[:, 6] = rng.uniform(0, 5, n_mal)
        x[:, 7] = rng.uniform(0, 2e6, n_mal)
        x[:, 8] = rng.uniform(0, 1e6, n_mal)
        x[:, 9] = rng.uniform(0, 1e6, n_mal)
        # Section stats — malicious often have high entropy (packed)
        x[:, 10:20] = rng.uniform(6.0, 8.0, (n_mal, 10))
        # Imports/Exports
        x[:, 20] = rng.integers(1, 10, n_mal)             # fewer imported DLLs (packed)
        x[:, 21] = rng.integers(1, 50, n_mal)
        x[:, 22] = rng.integers(0, 5, n_mal)
        x[:, 23] = rng.uniform(0, 5, n_mal)
        x[:, 24] = rng.uniform(0, 2, n_mal)
        x[:, 25:30] = rng.uniform(0, 20, (n_mal, 5))
        # Entropy — typically higher for malicious
        x[:, 30:38] = rng.uniform(6.5, 8.0, (n_mal, 8))
        # String features
        x[:, 38] = rng.integers(5, 30, n_mal)             # more suspicious URLs
        x[:, 39] = rng.integers(10, 50, n_mal)            # more registry refs
        x[:, 40] = rng.integers(3, 15, n_mal)             # more crypto API
        x[:, 41] = rng.integers(1, 5, n_mal)              # packing indicators
        x[:, 42] = rng.uniform(0, 0.5, n_mal)             # lower printable ratio
        x[:, 43] = rng.uniform(0, 200, n_mal)             # shorter avg strings
        # Misc
        x[:, 44] = rng.integers(0, 5, n_mal)
        x[:, 45] = rng.choice([0, 1], n_mal, p=[0.5, 0.5])  # overlay often present
        x[:, 46] = rng.choice([0, 1], n_mal, p=[0.6, 0.4])  # TLS callbacks
        x[:, 47] = rng.uniform(0, 0.3, n_mal)             # low signature score
        x[:, 48] = rng.integers(50, 200, n_mal)           # higher API depth
        x[:, 49] = rng.uniform(0.5, 1, n_mal)             # higher behavioral similarity
        return x

    X_ben = benign_features()
    X_mal = malicious_features()
    X = np.vstack([X_ben, X_mal])
    y = np.array([0] * n_ben + [1] * n_mal, dtype=np.int32)

    # Shuffle
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    # Build feature names
    feat_names = []
    feat_names += [f"pe_num_sections", "pe_image_size", "pe_entry_rva",
                   "pe_section_align", "pe_debug_stripped", "pe_compile_age",
                   "pe_subsystem", "pe_code_size", "pe_init_data_size", "pe_uninit_data_size"]
    feat_names += [f"sec_entropy_{i}" for i in range(10)]
    feat_names += ["import_dll_count", "import_func_count", "export_func_count",
                   "import_entropy", "export_entropy"]
    feat_names += [f"import_misc_{i}" for i in range(5)]
    feat_names += [f"file_entropy_{i}" for i in range(8)]
    feat_names += ["str_suspicious_urls", "str_registry_refs", "str_crypto_api",
                   "str_packing_indicators", "str_printable_ratio", "str_avg_len"]
    feat_names += ["misc_resource_count", "misc_overlay_present", "misc_tls_callbacks",
                   "misc_signature_score", "misc_api_depth", "misc_behavioral_sim"]

    df = pd.DataFrame(X, columns=feat_names)
    df["label"] = y
    return df


def preprocess_and_split(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """
    Split into train/val/test and fit a StandardScaler on training data.
    Returns (train_df, val_df, test_df, scaler).
    """
    label_col = "label"
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    feat_cols = [c for c in df.columns if c != label_col]

    # 70/15/15 split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(1 - TRAIN_RATIO), random_state=seed, stratify=y
    )
    relative_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1 - relative_val), random_state=seed, stratify=y_tmp
    )

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_sc, columns=feat_cols)
    train_df["label"] = y_train

    val_df = pd.DataFrame(X_val_sc, columns=feat_cols)
    val_df["label"] = y_val

    test_df = pd.DataFrame(X_test_sc, columns=feat_cols)
    test_df["label"] = y_test

    return train_df, val_df, test_df, scaler


def run_preprocessing():
    """End-to-end: generate data, preprocess, save splits."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"  Saved {len(df)} samples -> {DATASET_PATH}")

    print("Splitting and scaling...")
    train_df, val_df, test_df, scaler = preprocess_and_split(df)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)

    # Save sample
    sample = test_df.head(20)
    sample.to_csv(SAMPLE_PATH, index=False)

    # Save feature names
    feat_names = [c for c in df.columns if c != "label"]
    with open(FEATURE_NAMES_PATH, "w") as f:
        import json
        json.dump(feat_names, f)

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"  Scaler saved -> {SCALER_PATH}")
    print("Preprocessing complete.")
    return train_df, val_df, test_df, scaler


if __name__ == "__main__":
    run_preprocessing()
