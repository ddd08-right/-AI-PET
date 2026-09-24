from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    script = Path(__file__).parents[1] / "scripts" / "audit_public_pet_cohort.py"
    spec = importlib.util.spec_from_file_location("audit_public_pet_cohort", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_dry_run_reports_presence_without_opening_payloads(tmp_path: Path) -> None:
    for name in ("ct.nii.gz", "pet.nii.gz", "seg.nii.gz"):
        (tmp_path / name).write_bytes(b"SYNTHETIC_NOT_A_NIFTI")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "ct_path,pet_suv_path,seg_path\n"
        f"{tmp_path / 'ct.nii.gz'},{tmp_path / 'pet.nii.gz'},{tmp_path / 'seg.nii.gz'}\n",
        encoding="utf-8",
    )

    module = _module()
    cases = module.cases_from_manifest(
        manifest,
        ct_column="ct_path",
        pet_column="pet_suv_path",
        reference_column="seg_path",
    )
    report = module.dry_run_aggregate(cases)

    assert report["candidate_examinations"] == 1
    assert report["triplets_with_all_files_present"] == 1
    assert report["image_payloads_opened"] == "NO"
