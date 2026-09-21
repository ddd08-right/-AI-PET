from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pet_ai.real_data.audit import audit_examination


def _write(path: Path, data: np.ndarray, affine: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def _triplet(
    tmp_path: Path,
    *,
    ct: np.ndarray | None = None,
    pet: np.ndarray | None = None,
    segmentation: np.ndarray | None = None,
    pet_affine: np.ndarray | None = None,
    segmentation_affine: np.ndarray | None = None,
) -> tuple[Path, Path, Path]:
    shape = (3, 4, 5)
    affine = np.diag([2.0, 2.5, 4.0, 1.0])
    ct_data = np.zeros(shape, dtype=np.float32) if ct is None else ct
    pet_data = np.ones(shape, dtype=np.float32) if pet is None else pet
    seg_data = np.zeros(shape, dtype=np.uint8) if segmentation is None else segmentation
    return (
        _write(tmp_path / "ct.nii.gz", ct_data, affine),
        _write(tmp_path / "pet.nii.gz", pet_data, affine if pet_affine is None else pet_affine),
        _write(
            tmp_path / "seg.nii.gz",
            seg_data,
            affine if segmentation_affine is None else segmentation_affine,
        ),
    )


def test_valid_aligned_triplet_and_physical_volume(tmp_path: Path) -> None:
    segmentation = np.zeros((3, 4, 5), dtype=np.uint8)
    segmentation[0, 0, :2] = 1
    result = audit_examination(*_triplet(tmp_path, segmentation=segmentation))

    assert result.ok
    assert result.spacing_mm == pytest.approx((2.0, 2.5, 4.0))
    assert result.voxel_volume_ml == pytest.approx(0.02)
    assert result.reference_foreground_voxels == 2
    assert result.reference_volume_ml == pytest.approx(0.04)
    assert result.reference_empty is False


def test_reference_geometry_mismatch_has_no_physical_volume(tmp_path: Path) -> None:
    segmentation = np.zeros((3, 4, 5), dtype=np.uint8)
    segmentation[0, 0, :2] = 1
    segmentation_affine = np.diag([2.0, 2.5, 5.0, 1.0])
    result = audit_examination(
        *_triplet(tmp_path, segmentation=segmentation, segmentation_affine=segmentation_affine)
    )

    assert not result.ok
    assert result.reference_volume_ml is None
    assert result.reference_empty is False


def test_geometry_mismatch_is_rejected_without_resampling(tmp_path: Path) -> None:
    pet_affine = np.diag([2.0, 2.5, 5.0, 1.0])
    ct_path, pet_path, seg_path = _triplet(tmp_path, pet_affine=pet_affine)
    original_pet_affine = nib.load(pet_path).affine.copy()

    result = audit_examination(ct_path, pet_path, seg_path)

    assert not result.ok
    assert any("PET geometry" in reason for reason in result.failure_reasons)
    np.testing.assert_array_equal(nib.load(pet_path).affine, original_pet_affine)


def test_nonbinary_reference_is_rejected(tmp_path: Path) -> None:
    segmentation = np.zeros((3, 4, 5), dtype=np.uint8)
    segmentation[0, 0, 0] = 2
    result = audit_examination(*_triplet(tmp_path, segmentation=segmentation))

    assert not result.ok
    assert result.reference_volume_ml is None
    assert result.reference_empty is None
    assert any("not binary" in reason for reason in result.failure_reasons)


@pytest.mark.parametrize("role", ["ct", "pet", "segmentation"])
def test_nonfinite_values_are_rejected(tmp_path: Path, role: str) -> None:
    arrays = {
        "ct": np.zeros((3, 4, 5), dtype=np.float32),
        "pet": np.ones((3, 4, 5), dtype=np.float32),
        "segmentation": np.zeros((3, 4, 5), dtype=np.float32),
    }
    arrays[role][0, 0, 0] = np.nan if role != "pet" else np.inf
    result = audit_examination(
        *_triplet(
            tmp_path,
            ct=arrays["ct"],
            pet=arrays["pet"],
            segmentation=arrays["segmentation"],
        )
    )

    assert not result.ok
    assert result.readable
    assert any("NaN or Inf" in reason for reason in result.failure_reasons)


def test_empty_binary_reference_is_valid(tmp_path: Path) -> None:
    result = audit_examination(*_triplet(tmp_path))

    assert result.ok
    assert result.reference_empty is True
    assert result.reference_foreground_voxels == 0
    assert result.reference_volume_ml == 0.0


def test_missing_file_is_not_readable(tmp_path: Path) -> None:
    ct_path, pet_path, seg_path = _triplet(tmp_path)
    seg_path.unlink()

    result = audit_examination(ct_path, pet_path, seg_path)

    assert not result.readable
    assert not result.ok
    assert any("missing reference segmentation file" in reason for reason in result.failure_reasons)


def test_unreadable_file_is_not_readable(tmp_path: Path) -> None:
    ct_path, pet_path, seg_path = _triplet(tmp_path)
    pet_path.write_bytes(b"not a NIfTI image")

    result = audit_examination(ct_path, pet_path, seg_path)

    assert not result.readable
    assert not result.ok
    assert any("unreadable PET file" in reason for reason in result.failure_reasons)
