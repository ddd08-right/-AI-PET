
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from pet_ai.data.manifest import REQUIRED_FIELDS, validate_manifest
from pet_ai.data.split_validation import validate_splits
from pet_ai.evaluation.lesion_metrics import evaluate_lesions
from pet_ai.evaluation.segmentation import evaluate_nifti_segmentation
from pet_ai.qc.geometry import compare_nifti_geometry
from pet_ai.qc.labels import validate_segmentation_labels
from pet_ai.reproducibility.run_manifest import create_run_manifest, write_run_manifest


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    image = nib.Nifti1Image(data, affine)
    image.header.set_zooms((2.0, 2.0, 2.0))
    image.header.set_xyzt_units("mm")
    nib.save(image, str(path))


def _write_manifest(path: Path) -> None:
    rows = [
        {
            "case_id": "SYNTH_CASE_001",
            "patient_key": "SYNTH_PATIENT_001",
            "ct_available": "YES",
            "pet_available": "YES",
            "seg_available": "YES",
            "label_status": "POSITIVE",
            "split": "train",
        },
        {
            "case_id": "SYNTH_CASE_002",
            "patient_key": "SYNTH_PATIENT_002",
            "ct_available": "YES",
            "pet_available": "YES",
            "seg_available": "NO",
            "label_status": "NEGATIVE",
            "split": "validation",
        },
        {
            "case_id": "SYNTH_CASE_003",
            "patient_key": "SYNTH_PATIENT_003",
            "ct_available": "YES",
            "pet_available": "YES",
            "seg_available": "YES",
            "label_status": "POSITIVE",
            "split": "test",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_pass(label: str) -> None:
    print(f"[PASS] {label}")


def main() -> int:
    print("PET/CT AI Engineering Demo")
    print()

    rng = np.random.default_rng(20260915)
    with tempfile.TemporaryDirectory(prefix="pet_ai_demo_") as temp_name:
        temp_dir = Path(temp_name)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        shape = (24, 24, 24)
        ct = rng.normal(loc=0.0, scale=100.0, size=shape).astype(np.float32)
        pet = rng.gamma(shape=2.0, scale=1.0, size=shape).astype(np.float32)
        gt = np.zeros(shape, dtype=np.uint8)
        pred = np.zeros(shape, dtype=np.uint8)
        gt[6:10, 6:10, 6:10] = 1
        gt[15:17, 15:17, 15:17] = 1
        pred[6:10, 6:10, 6:10] = 1
        pred[19:21, 19:21, 19:21] = 1

        ct_path = temp_dir / "synthetic_ct.nii.gz"
        pet_path = temp_dir / "synthetic_pet.nii.gz"
        gt_path = temp_dir / "synthetic_gt.nii.gz"
        pred_path = temp_dir / "synthetic_prediction.nii.gz"
        manifest_path = temp_dir / "synthetic_manifest.csv"
        run_manifest_path = temp_dir / "run_manifest.json"

        _save_nifti(ct_path, ct, affine)
        _save_nifti(pet_path, pet, affine)
        _save_nifti(gt_path, gt, affine)
        _save_nifti(pred_path, pred, affine)
        _write_manifest(manifest_path)
        _print_pass("synthetic data generation")

        manifest_result = validate_manifest(manifest_path)
        if not manifest_result.ok:
            raise RuntimeError(f"manifest validation failed: {manifest_result.errors}")
        _print_pass("manifest validation")

        split_result = validate_splits(manifest_result.rows)
        if not split_result.ok:
            raise RuntimeError(f"split validation failed: {split_result.errors}")
        _print_pass("patient-level split validation")

        geometry_result = compare_nifti_geometry(ct_path, pet_path)
        if not geometry_result.ok:
            raise RuntimeError(f"geometry QC failed: {geometry_result.messages}")
        _print_pass("PET/CT geometry QC")

        label_result = validate_segmentation_labels(gt_path, reference_path=ct_path, require_non_empty=True)
        if not label_result.ok:
            raise RuntimeError(f"label QC failed: {label_result.messages}")
        _print_pass("segmentation label QC")

        voxel_metrics = evaluate_nifti_segmentation(pred_path, gt_path)
        if (
            voxel_metrics.counts.true_positive_voxels <= 0
            or voxel_metrics.counts.false_negative_voxels <= 0
        ):
            raise RuntimeError("voxel metrics did not produce the expected synthetic TP/FN pattern")
        _print_pass("voxel metrics")

        lesion_metrics = evaluate_lesions(pred, gt, voxel_volume_mm3=8.0, connectivity=26)
        if (
            lesion_metrics.true_positive_lesions != 1
            or lesion_metrics.false_negative_lesion_count != 1
            or lesion_metrics.false_positive_lesion_count != 1
        ):
            raise RuntimeError("lesion metrics did not produce the expected synthetic TP/FN/FP pattern")
        _print_pass("lesion metrics")

        run_manifest = create_run_manifest(
            repo_root=Path.cwd(),
            dataset_manifest=manifest_path,
            split_manifest=None,
            config=None,
            checkpoint=None,
            seed=20260915,
            command=["python", "scripts/demo_pipeline.py"],
            exit_status=0,
            notes="Synthetic public-safe demo run",
            gpu_name="NOT_VERIFIED",
        )
        write_run_manifest(run_manifest, run_manifest_path)
        if not run_manifest_path.exists():
            raise RuntimeError("run manifest was not written")
        _print_pass("run provenance")

    print()
    print("Demo completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
