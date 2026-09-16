from __future__ import annotations

import nibabel as nib
import numpy as np

from pet_ai.qc.geometry import compare_nifti_geometry


def save_image(path, shape=(4, 4, 4), affine=None):
    affine = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), str(path))


def test_pet_ct_different_geometry_detected(tmp_path):
    ct = tmp_path / "ct.nii.gz"
    pet = tmp_path / "pet.nii.gz"
    save_image(ct, shape=(4, 4, 4), affine=np.diag([2.0, 2.0, 3.0, 1.0]))
    save_image(pet, shape=(5, 4, 4), affine=np.diag([2.0, 2.0, 3.0, 1.0]))

    result = compare_nifti_geometry(ct, pet)

    assert not result.ok
    assert not result.shape_match
    assert any("shape mismatch" in message for message in result.messages)
