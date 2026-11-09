"""Utilities for preparing the extracted EMBER 2017 JSONL features."""

import json
import os
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher


EMBER_FEATURE_COUNT = 2351
EMBER_FEATURE_NAMES = [f"ember_{index:04d}" for index in range(EMBER_FEATURE_COUNT)]


def _hashed(values, n_features: int, input_type: str = "string") -> np.ndarray:
    if not values:
        return np.zeros(n_features, dtype=np.float32)
    return FeatureHasher(n_features=n_features, input_type=input_type).transform(
        [values]
    ).toarray()[0].astype(np.float32)


def ember_feature_vector(record: dict) -> np.ndarray:
    """Convert one raw EMBER 2017 JSON record to its 2351 numeric features."""
    histogram = np.asarray(record["histogram"], dtype=np.float32)
    byteentropy = np.asarray(record["byteentropy"], dtype=np.float32)
    if histogram.size != 256 or byteentropy.size != 256:
        raise ValueError("EMBER histogram groups must each contain 256 values")
    histogram_sum = float(histogram.sum())
    byteentropy_sum = float(byteentropy.sum())
    if histogram_sum:
        histogram /= histogram_sum
    if byteentropy_sum:
        byteentropy /= byteentropy_sum

    strings = record["strings"]
    printable_count = float(strings["printables"])
    printable_histogram = np.asarray(strings["printabledist"], dtype=np.float32)
    if printable_histogram.size != 96:
        raise ValueError("EMBER printable-character histogram must contain 96 values")
    if printable_count:
        printable_histogram /= printable_count
    string_features = np.concatenate((
        np.asarray([
            strings["numstrings"], strings["avlength"], strings["printables"],
        ], dtype=np.float32),
        printable_histogram,
        np.asarray([
            strings["entropy"], strings["paths"], strings["urls"],
            strings["registry"], strings["MZ"],
        ], dtype=np.float32),
    ))

    general = record["general"]
    general_features = np.asarray([
        general[key] for key in (
            "size", "vsize", "has_debug", "exports", "imports",
            "has_relocations", "has_resources", "has_signature", "has_tls",
            "symbols",
        )
    ], dtype=np.float32)

    header = record["header"]
    coff = header["coff"]
    optional = header["optional"]
    header_features = np.concatenate((
        np.asarray([coff["timestamp"]], dtype=np.float32),
        _hashed([coff["machine"]], 10),
        _hashed(coff["characteristics"], 10),
        _hashed([optional["subsystem"]], 10),
        _hashed(optional["dll_characteristics"], 10),
        _hashed([optional["magic"]], 10),
        np.asarray([
            optional[key] for key in (
                "major_image_version", "minor_image_version",
                "major_linker_version", "minor_linker_version",
                "major_operating_system_version", "minor_operating_system_version",
                "major_subsystem_version", "minor_subsystem_version",
                "sizeof_code", "sizeof_headers", "sizeof_heap_commit",
            )
        ], dtype=np.float32),
    ))

    section = record["section"]
    sections = section["sections"]
    section_summary = np.asarray([
        len(sections),
        sum(item["size"] == 0 for item in sections),
        sum(item["name"] == "" for item in sections),
        sum("MEM_READ" in item["props"] and "MEM_EXECUTE" in item["props"] for item in sections),
        sum("MEM_WRITE" in item["props"] for item in sections),
    ], dtype=np.float32)
    section_features = np.concatenate((
        section_summary,
        _hashed([(item["name"], item["size"]) for item in sections], 50, "pair"),
        _hashed([(item["name"], item["entropy"]) for item in sections], 50, "pair"),
        _hashed([(item["name"], item["vsize"]) for item in sections], 50, "pair"),
        _hashed([section["entry"]], 50),
        _hashed([
            prop for item in sections if item["name"] == section["entry"]
            for prop in item["props"]
        ], 50),
    ))

    imports = record["imports"]
    import_libraries = list({library.lower() for library in imports})
    import_functions = [
        f"{library.lower()}:{function}"
        for library, functions in imports.items()
        for function in functions
    ]
    import_features = np.concatenate((
        _hashed(import_libraries, 256),
        _hashed(import_functions, 1024),
    ))
    export_features = _hashed(record["exports"], 128)

    vector = np.concatenate((
        histogram, byteentropy, string_features, general_features,
        header_features, section_features, import_features, export_features,
    )).astype(np.float32)
    if vector.size != EMBER_FEATURE_COUNT:
        raise ValueError(f"Expected {EMBER_FEATURE_COUNT} features, got {vector.size}")
    if not np.isfinite(vector).all():
        raise ValueError("EMBER feature vector contains a non-finite value")
    return vector


def collect_labeled_records(
    paths: Iterable[str],
    samples_per_class: int = 100,
    max_records: int = 250_000,
) -> pd.DataFrame:
    """Stream JSONL files until a balanced count of labels 0 and 1 is collected."""
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    if max_records < 1:
        raise ValueError("max_records must be positive")

    rows = []
    counts = Counter()
    seen = 0
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"EMBER JSONL file not found: {path}")
        with open(path, "r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                seen += 1
                if seen > max_records:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error

                label = record.get("label")
                if label not in (0, 1) or counts[label] >= samples_per_class:
                    continue
                try:
                    features = ember_feature_vector(record)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid EMBER record at {path}:{line_number}: {error}") from error
                rows.append(np.append(features, np.float32(label)))
                counts[label] += 1
                if counts[0] == samples_per_class and counts[1] == samples_per_class:
                    break
            if seen >= max_records or (
                counts[0] == samples_per_class and counts[1] == samples_per_class
            ):
                break

    if counts[0] != samples_per_class or counts[1] != samples_per_class:
        raise ValueError(
            f"Collected {counts[0]} benign and {counts[1]} malicious rows; "
            f"requested {samples_per_class} of each from {seen} records. "
            "Try more input files or increase --max-records."
        )
    return pd.DataFrame(rows, columns=EMBER_FEATURE_NAMES + ["label"])


def prepare_ember_csvs(
    ember_dir: str,
    output_dir: str,
    samples_per_class: int = 100,
    max_records: int = 250_000,
) -> tuple[str, str]:
    """Create separate balanced train/test CSVs from the EMBER train and test files."""
    train_shards = [
        os.path.join(ember_dir, f"train_features_{index}.jsonl")
        for index in range(6)
    ]
    test_file = os.path.join(ember_dir, "test_features.jsonl")
    train = collect_labeled_records(train_shards, samples_per_class, max_records)
    test = collect_labeled_records([test_file], samples_per_class, max_records)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path