from __future__ import annotations

from pet_ai.data.split_validation import validate_splits


def test_patient_leakage_across_train_validation_detected():
    rows = [
        {"case_id": "CASE001", "patient_key": "PATIENT_A", "label_status": "POSITIVE", "split": "train"},
        {"case_id": "CASE002", "patient_key": "PATIENT_A", "label_status": "NEGATIVE", "split": "validation"},
    ]

    result = validate_splits(rows)

    assert not result.ok
    assert any("Patient-level leakage" in error for error in result.errors)


def test_split_counts_by_label():
    rows = [
        {"case_id": "CASE001", "patient_key": "P1", "label_status": "POSITIVE", "split": "train"},
        {"case_id": "CASE002", "patient_key": "P2", "label_status": "NEGATIVE", "split": "train"},
        {"case_id": "CASE003", "patient_key": "P3", "label_status": "NEGATIVE", "split": "test"},
    ]

    result = validate_splits(rows)

    assert result.ok
    assert result.counts_by_split["train"]["POSITIVE"] == 1
    assert result.counts_by_split["train"]["NEGATIVE"] == 1
    assert result.counts_by_split["test"]["NEGATIVE"] == 1
