from __future__ import annotations

import csv

from pet_ai.data.manifest import validate_manifest


def write_manifest(path, rows):
    fields = ["case_id", "patient_key", "ct_available", "pet_available", "seg_available", "label_status", "split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_duplicate_case_id_detected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"case_id": "CASE001", "patient_key": "P1", "ct_available": "YES", "pet_available": "YES", "seg_available": "YES", "label_status": "POSITIVE", "split": "train"},
        {"case_id": "CASE001", "patient_key": "P2", "ct_available": "YES", "pet_available": "YES", "seg_available": "NO", "label_status": "NEGATIVE", "split": "validation"},
    ]
    write_manifest(manifest, rows)

    result = validate_manifest(manifest)

    assert not result.ok
    assert any("duplicate case_id" in error for error in result.errors)


def test_missing_modality_and_invalid_label_detected(tmp_path):
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [{"case_id": "CASE002", "patient_key": "P2", "ct_available": "NO", "pet_available": "YES", "seg_available": "YES", "label_status": "MAYBE", "split": "train"}],
    )

    result = validate_manifest(manifest)

    assert not result.ok
    assert any("missing CT" in error for error in result.errors)
    assert any("invalid label_status" in error for error in result.errors)
