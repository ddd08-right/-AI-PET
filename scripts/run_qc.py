from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pet_ai.qc.geometry import compare_nifti_geometry
from pet_ai.qc.labels import validate_segmentation_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PET/CT/SEG geometry and label QC on NIfTI files.")
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--pet", required=True, type=Path)
    parser.add_argument("--seg", type=Path)
    parser.add_argument("--require-non-empty-seg", action="store_true")
    args = parser.parse_args()

    results: dict[str, object] = {"ct_pet_geometry": compare_nifti_geometry(args.ct, args.pet).to_dict()}
    ok = bool(results["ct_pet_geometry"]["ok"])  # type: ignore[index]

    if args.seg is not None:
        seg_result = validate_segmentation_labels(args.seg, reference_path=args.ct, require_non_empty=args.require_non_empty_seg)
        results["segmentation_labels"] = {
            "ok": seg_result.ok,
            "unique_labels": seg_result.unique_labels,
            "nonzero_voxels": seg_result.nonzero_voxels,
            "messages": seg_result.messages,
        }
        ok = ok and seg_result.ok

    print(json.dumps(results, indent=2, sort_keys=True))
    if not ok:
        print("QC failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
