from __future__ import annotations

import argparse
import csv
from pathlib import Path

from pet_ai.data.manifest import REQUIRED_FIELDS


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an empty dataset manifest template.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
    print(f"Wrote manifest template: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
