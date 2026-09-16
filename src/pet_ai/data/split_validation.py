"""Patient-level split validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

VALID_SPLITS = {"train", "validation", "val", "test"}
VALID_LABELS = {"POSITIVE", "NEGATIVE", "UNKNOWN", "NOT_AVAILABLE"}


@dataclass(frozen=True)
class SplitValidationResult:
    errors: list[str]
    counts_by_split: dict[str, dict[str, int]]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_splits(rows: list[dict[str, str]]) -> SplitValidationResult:
    """Detect case duplication, row duplication, invalid labels, and patient leakage."""

    errors: list[str] = []
    case_ids: Counter[str] = Counter()
    row_keys: Counter[tuple[tuple[str, str], ...]] = Counter()
    patient_splits: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"POSITIVE": 0, "NEGATIVE": 0, "UNKNOWN": 0, "NOT_AVAILABLE": 0})

    for index, row in enumerate(rows, start=2):
        normalized = {key: (value or "").strip() for key, value in row.items()}
        row_keys[tuple(sorted(normalized.items()))] += 1
        case_id = normalized.get("case_id", "")
        patient_key = normalized.get("patient_key", "")
        split = normalized.get("split", "").lower()
        label = normalized.get("label_status", "").upper()

        if case_id:
            case_ids[case_id] += 1
        else:
            errors.append(f"Row {index}: case_id is empty")

        if split == "val":
            split = "validation"
        if split not in VALID_SPLITS:
            errors.append(f"Row {index}: split must be train/validation/test, got '{normalized.get('split', '')}'")

        if label not in VALID_LABELS:
            errors.append(f"Row {index}: invalid label_status '{normalized.get('label_status', '')}'")
        elif split in VALID_SPLITS:
            counts[split][label] += 1

        if patient_key and split in VALID_SPLITS:
            patient_splits[patient_key].add(split)
        elif not patient_key:
            errors.append(f"Row {index}: patient_key is empty")

    for case_id, count in sorted(case_ids.items()):
        if count > 1:
            errors.append(f"Duplicate case_id '{case_id}' appears {count} times")

    duplicate_rows = sum(count - 1 for count in row_keys.values() if count > 1)
    if duplicate_rows:
        errors.append(f"Duplicate manifest rows detected: {duplicate_rows}")

    for patient_key, splits in sorted(patient_splits.items()):
        if len(splits) > 1:
            errors.append(f"Patient-level leakage: patient_key '{patient_key}' appears in splits {sorted(splits)}")

    return SplitValidationResult(errors=errors, counts_by_split=dict(counts))
