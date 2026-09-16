from __future__ import annotations

import argparse
import json
from pathlib import Path

from pet_ai.evaluation.segmentation import evaluate_nifti_segmentation


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate binary segmentation against ground truth.")
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    args = parser.parse_args()

    metrics = evaluate_nifti_segmentation(args.prediction, args.ground_truth)
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
