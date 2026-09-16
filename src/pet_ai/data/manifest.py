"""Public-safe dataset manifest validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = (
    "case_id",
    "patient_key",
    "ct_available",
    "pet_available",
    "seg_available",
    "label_status",
    "split",
)
VALID_BOOLEAN = {"YES", "NO", "TRUE", "FALSE", "1", "0", "Y", "N"}
VALID_LABELS = {"POSITIVE", "NEGATIVE", "UNKNOWN", "NOT_AVAILABLE"}
VALID_SPLITS = {"train", "validation", "val", "test", "development", "holdout", "none", ""}
FORBIDDEN_FIELDS = {
    "name",
    "patientname",
    "patient_name",
    "mrn",
    "dob",
    "birthdate",
    "patientbirthdate",
    "accession",
    "accessionnumber",
    "local_path",
    "path",
}


@dataclass(frozen=True)
class ManifestValidationResult:
    """Structured validation result for a CSV manifest."""

    rows: list[dict[str, str]]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read a CSV manifest into normalized string dictionaries."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def validate_manifest_rows(rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> ManifestValidationResult:
    """Validate manifest rows without touching image data."""

    errors: list[str] = []
    warnings: list[str] = []
    names = fieldnames or (list(rows[0].keys()) if rows else [])
    normalized_names = {name.lower().strip() for name in names}

    missing = [field for field in REQUIRED_FIELDS if field not in names]
    if missing:
        errors.append(f"Missing required field(s): {', '.join(missing)}")

    forbidden = sorted(FORBIDDEN_FIELDS & normalized_names)
    if forbidden:
        errors.append(f"Forbidden patient/private field(s) present: {', '.join(forbidden)}")

    seen_case_ids: dict[str, int] = {}
    for index, row in enumerate(rows, start=2):
        case_id = row.get("case_id", "")
        patient_key = row.get("patient_key", "")
        if not case_id:
            errors.append(f"Row {index}: case_id is empty")
        elif case_id in seen_case_ids:
            errors.append(f"Row {index}: duplicate case_id '{case_id}' first seen on row {seen_case_ids[case_id]}")
        else:
            seen_case_ids[case_id] = index

        if not patient_key:
            errors.append(f"Row {index}: patient_key is empty")

        for field in ("ct_available", "pet_available", "seg_available"):
            value = row.get(field, "").upper()
            if value not in VALID_BOOLEAN:
                errors.append(f"Row {index}: {field} must be one of {sorted(VALID_BOOLEAN)}, got '{row.get(field, '')}'")

        ct_available = row.get("ct_available", "").upper() in {"YES", "TRUE", "1", "Y"}
        pet_available = row.get("pet_available", "").upper() in {"YES", "TRUE", "1", "Y"}
        seg_available = row.get("seg_available", "").upper() in {"YES", "TRUE", "1", "Y"}
        if not ct_available:
            errors.append(f"Row {index}: missing CT modality reference for case_id '{case_id}'")
        if not pet_available:
            errors.append(f"Row {index}: missing PET modality reference for case_id '{case_id}'")
        if row.get("label_status", "").upper() == "POSITIVE" and not seg_available:
            errors.append(f"Row {index}: positive case_id '{case_id}' has no segmentation reference")

        label = row.get("label_status", "").upper()
        if label not in VALID_LABELS:
            errors.append(f"Row {index}: invalid label_status '{row.get('label_status', '')}'")

        split = row.get("split", "").lower()
        if split not in VALID_SPLITS:
            errors.append(f"Row {index}: invalid split '{row.get('split', '')}'")

    if not rows:
        warnings.append("Manifest contains no rows")

    return ManifestValidationResult(rows=rows, errors=errors, warnings=warnings)


def validate_manifest(path: Path) -> ManifestValidationResult:
    """Read and validate a manifest file."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
        fieldnames = reader.fieldnames or []
    return validate_manifest_rows(rows, fieldnames)
