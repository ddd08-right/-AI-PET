from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pet_ai.data.manifest import validate_manifest
from pet_ai.data.split_validation import validate_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a public-safe PET/CT dataset manifest CSV.")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    result = validate_manifest(args.manifest)
    split_result = validate_splits(result.rows) if not result.errors else None
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if split_result is not None:
        for split, counts in sorted(split_result.counts_by_split.items()):
            print(f"{split}: POSITIVE={counts['POSITIVE']} NEGATIVE={counts['NEGATIVE']} UNKNOWN={counts['UNKNOWN']} NOT_AVAILABLE={counts['NOT_AVAILABLE']}")
        for error in split_result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
    if result.ok and (split_result is None or split_result.ok):
        print(f"PASS: {len(result.rows)} manifest row(s) validated")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
