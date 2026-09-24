from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest

from pet_ai.qc.geometry import compare_nifti_geometry, load_nifti_geometry


def save_image(path, shape=(4, 4, 4), affine=None, unit="mm"):
    affine = np.eye(4) if affine is None else affine
    image = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine)
    image.header.set_xyzt_units(unit)
    nib.save(image, str(path))


def test_pet_ct_different_geometry_detected(tmp_path):
    ct = tmp_path / "ct.nii.gz"
    pet = tmp_path / "pet.nii.gz"
    save_image(ct, shape=(4, 4, 4), affine=np.diag([2.0, 2.0, 3.0, 1.0]))
    save_image(pet, shape=(5, 4, 4), affine=np.diag([2.0, 2.0, 3.0, 1.0]))

    result = compare_nifti_geometry(ct, pet)

    assert not result.ok
    assert not result.shape_match
    assert any("shape mismatch" in message for message in result.messages)


def test_equivalent_meter_and_millimetre_grids_match(tmp_path):
    millimetres = tmp_path / "millimetres.nii.gz"
    metres = tmp_path / "metres.nii.gz"
    affine_mm = np.diag([2.0, 3.0, 4.0, 1.0])
    affine_mm[:3, 3] = [10.0, -20.0, 30.0]
    affine_m = np.diag([0.002, 0.003, 0.004, 1.0])
    affine_m[:3, 3] = [0.01, -0.02, 0.03]
    save_image(millimetres, affine=affine_mm, unit="mm")
    save_image(metres, affine=affine_m, unit="meter")

    assert compare_nifti_geometry(millimetres, metres).ok


def test_same_numeric_grid_with_different_units_fails(tmp_path):
    millimetres = tmp_path / "millimetres.nii.gz"
    metres = tmp_path / "metres.nii.gz"
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    save_image(millimetres, affine=affine, unit="mm")
    save_image(metres, affine=affine, unit="meter")

    assert not compare_nifti_geometry(millimetres, metres).ok


def test_unknown_spatial_unit_is_rejected(tmp_path):
    image = tmp_path / "unknown.nii.gz"
    save_image(image, unit="unknown")

    with pytest.raises(ValueError, match="spatial unit"):
        compare_nifti_geometry(image, image)


def test_conflicting_valid_qform_and_sform_are_rejected(tmp_path):
    path = tmp_path / "conflict.nii.gz"
    image = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm")
    image.set_qform(np.eye(4), code=1)
    shifted = np.eye(4)
    shifted[0, 3] = 5.0
    image.set_sform(shifted, code=1)
    nib.save(image, path)

    with pytest.raises(ValueError, match="qform and sform conflict"):
        load_nifti_geometry(path)


def test_only_one_valid_coded_transform_is_accepted(tmp_path):
    path = tmp_path / "sform_only.nii.gz"
    image = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
    image.header.set_xyzt_units("mm")
    image.set_qform(np.eye(4), code=0)
    image.set_sform(np.eye(4), code=1)
    nib.save(image, path)

    assert load_nifti_geometry(path).spacing == pytest.approx((1.0, 1.0, 1.0))
