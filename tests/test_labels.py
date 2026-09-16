from __future__ import annotations

import nibabel as nib
import numpy as np

from pet_ai.qc.labels import validate_segmentation_labels


def test_invalid_segmentation_label_detected(tmp_path):
    seg = tmp_path / "seg.nii.gz"
    data = np.zeros((3, 3, 3), dtype=np.int16)
    data[1, 1, 1] = 2
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(seg))

    result = validate_segmentation_labels(seg)

    assert not result.ok
    assert any("invalid labels" in message for message in result.messages)
