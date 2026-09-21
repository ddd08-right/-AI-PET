"""Aggregate read-only QC for externally stored PET/CT examinations."""

from __future__ import annotations

import argparse
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
        required=True,
        metavar=("CT", "PET", "REFERENCE_SEG"),
        help="external paths for one examination; repeat for additional examinations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional aggregate JSON output path outside the public repository",
    )
    return parser


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
    report = aggregate(args.case)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["qc_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
