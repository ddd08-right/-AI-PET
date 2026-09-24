"""Aggregate read-only QC for externally stored PET/CT examinations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pet_ai.real_data.audit import audit_examination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit external CT/PET/reference triplets without emitting their paths."
    )
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        metavar=("CT", "PET", "REFERENCE_SEG"),
        help="external paths for one examination; repeat for additional examinations",
    )
    parser.add_argument("--manifest", type=Path, help="CSV containing external triplet paths")
    parser.add_argument("--ct-column", default="ct_path")
    parser.add_argument("--pet-column", default="pet_suv_path")
    parser.add_argument("--reference-column", default="seg_path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check manifest/path availability without opening image payloads",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new directory for audit_summary.json; existing directories are refused",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional aggregate JSON output path outside the public repository",
    )
    return parser


def cases_from_manifest(
    manifest: Path, *, ct_column: str, pet_column: str, reference_column: str
) -> list[list[str]]:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {ct_column, pet_column, reference_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest lacks required path columns: {sorted(missing)}")
        return [
            [row[ct_column], row[pet_column], row[reference_column]]
            for row in reader
        ]


def dry_run_aggregate(cases: list[list[str]]) -> dict[str, Any]:
    available = sum(all(Path(value).is_file() for value in case) for case in cases)
    return {
        "candidate_examinations": len(cases),
        "triplets_with_all_files_present": available,
        "triplets_with_missing_files": len(cases) - available,
        "image_payloads_opened": "NO",
        "source_paths_in_report": "NO",
        "status": "PASS" if available == len(cases) else "FAIL",
    }


def aggregate(cases: list[list[str]]) -> dict[str, Any]:
    audits = [audit_examination(*(Path(value) for value in case)) for case in cases]
    failure_counts: Counter[str] = Counter()
    for audit in audits:
        failure_counts.update(audit.failure_reasons)
    return {
        "candidate_examinations": len(audits),
        "readable": sum(audit.readable for audit in audits),
        "qc_pass": sum(audit.ok for audit in audits),
        "qc_fail": sum(not audit.ok for audit in audits),
        "reference_nonempty": sum(audit.reference_empty is False for audit in audits),
        "reference_empty": sum(audit.reference_empty is True for audit in audits),
        "failure_reasons": dict(sorted(failure_counts.items())),
        "source_paths_in_report": "NO",
        "silent_exclusions": "NO",
    }


def main() -> int:
    args = build_parser().parse_args()
    if bool(args.case) == bool(args.manifest):
        raise SystemExit("provide exactly one of --case or --manifest")
    try:
        cases = args.case or cases_from_manifest(
            args.manifest,
            ct_column=args.ct_column,
            pet_column=args.pet_column,
            reference_column=args.reference_column,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"preflight failed: {error}") from error
    report = dry_run_aggregate(cases) if args.dry_run else aggregate(cases)
    rendered = json.dumps(report, indent=2) + "\n"
    output = args.output
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        output = args.output_dir / "audit_summary.json"
    if output is not None:
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output.name}")
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    failed = report.get("qc_fail", report.get("triplets_with_missing_files", 0))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
