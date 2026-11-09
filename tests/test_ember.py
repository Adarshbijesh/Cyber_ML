import json

import numpy as np
import pytest

from src.ember import (
    EMBER_FEATURE_COUNT,
    collect_labeled_records,
    ember_feature_vector,
)


def make_record(label):
    return {
        "label": label,
        "histogram": [1] + [0] * 255,
        "byteentropy": [1] + [0] * 255,
        "strings": {
            "numstrings": 1,
            "avlength": 5.0,
            "printables": 5,
            "printabledist": [5] + [0] * 95,
            "entropy": 0.0,
            "paths": 0,
            "urls": 0,
            "registry": 0,
            "MZ": 1,
        },
        "general": {
            "size": 1,
            "vsize": 1,
            "has_debug": 0,
            "exports": 0,
            "imports": 1,
            "has_relocations": 0,
            "has_resources": 0,
            "has_signature": 0,
            "has_tls": 0,
            "symbols": 0,
        },
        "header": {
            "coff": {"timestamp": 0, "machine": "I386", "characteristics": []},
            "optional": {
                "subsystem": "WINDOWS_GUI",
                "dll_characteristics": [],
                "magic": "PE32",
                "major_image_version": 0,
                "minor_image_version": 0,
                "major_linker_version": 0,
                "minor_linker_version": 0,
                "major_operating_system_version": 0,
                "minor_operating_system_version": 0,
                "major_subsystem_version": 0,
                "minor_subsystem_version": 0,
                "sizeof_code": 0,
                "sizeof_headers": 0,
                "sizeof_heap_commit": 0,
            },
        },
        "section": {
            "entry": ".text",
            "sections": [{
                "name": ".text",
                "size": 1,
                "entropy": 1.0,
                "vsize": 1,
                "props": ["MEM_READ", "MEM_EXECUTE"],
            }],
        },
        "imports": {"KERNEL32.dll": ["ExitProcess"]},
        "exports": [],
    }


def test_feature_vector_has_expected_size_and_finite_values():
    vector = ember_feature_vector(make_record(0))

    assert vector.shape == (EMBER_FEATURE_COUNT,)
    assert np.isfinite(vector).all()
    assert vector[0] == 1.0


def test_collect_labeled_records_balances_classes_and_skips_unlabeled(tmp_path):
    source = tmp_path / "records.jsonl"
    records = [make_record(-1), make_record(0), make_record(1), make_record(1)]
    source.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    frame = collect_labeled_records([str(source)], samples_per_class=1)

    assert frame.shape == (2, EMBER_FEATURE_COUNT + 1)
    assert frame["label"].value_counts().to_dict() == {0: 1, 1: 1}


def test_collect_labeled_records_reports_missing_class(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps(make_record(0)), encoding="utf-8")

    with pytest.raises(ValueError, match="Collected 1 benign and 0 malicious"):
        collect_labeled_records([str(source)], samples_per_class=1)