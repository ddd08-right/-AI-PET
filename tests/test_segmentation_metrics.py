from __future__ import annotations

import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pet_ai.evaluation.segmentation import evaluate_binary_segmentation, evaluate_nifti_segmentation


def test_known_dice_example_produces_expected_dice():
    gt = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8).reshape(1, 2, 3)
    pred = np.array([1, 1, 0, 1, 0, 0], dtype=np.uint8).reshape(1, 2, 3)

    metrics = evaluate_binary_segmentation(pred, gt, voxel_volume_mm3=2.0)

    assert metrics.counts.true_positive_voxels == 2
    assert metrics.counts.false_positive_voxels == 1
    assert metrics.counts.false_negative_voxels == 1
    assert metrics.dice == 4 / 6


def test_empty_gt_produces_nan_dice_and_correct_fpv():
    gt = np.zeros((2, 2, 2), dtype=np.uint8)
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    pred[0, 0, 0] = 1
    pred[1, 1, 1] = 1

    metrics = evaluate_binary_segmentation(pred, gt, voxel_volume_mm3=6.0)

    assert math.isnan(metrics.dice)
    assert metrics.fpv_ml == 0.012
    assert metrics.fnv_ml == 0.0


def test_fpv_fnv_volume_unit_conversion_is_correct():
    gt = np.array([1, 1, 0, 0], dtype=np.uint8).reshape(1, 2, 2)
    pred = np.array([0, 1, 1, 1], dtype=np.uint8).reshape(1, 2, 2)

    metrics = evaluate_binary_segmentation(pred, gt, voxel_volume_mm3=5.0)

    assert metrics.counts.false_positive_voxels == 2
    assert metrics.counts.false_negative_voxels == 1
    assert metrics.fpv_ml == 0.01
    assert metrics.fnv_ml == 0.005


@pytest.mark.parametrize(
    "bad_prediction",
    [
        np.full((2, 2, 2), np.nan),
        np.full((2, 2, 2), 0.5),
        np.full((2, 2, 2), -1.0),
        np.zeros((2, 2)),
    ],
)
def test_invalid_prediction_arrays_are_rejected(bad_prediction):
    with pytest.raises(ValueError):
        evaluate_binary_segmentation(bad_prediction, np.ones((2, 2, 2)), voxel_volume_mm3=1.0)


@pytest.mark.parametrize("voxel_volume", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_voxel_volume_is_rejected(voxel_volume):
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_binary_segmentation(
            np.zeros((2, 2, 2), dtype=bool),
            np.zeros((2, 2, 2), dtype=bool),
            voxel_volume_mm3=voxel_volume,
        )


def _save_mask(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    image = nib.Nifti1Image(data, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image, path)


def test_nifti_affine_mismatch_is_rejected(tmp_path):
    gt_path = tmp_path / "gt.nii.gz"
    pred_path = tmp_path / "pred.nii.gz"
    data = np.ones((2, 2, 2), dtype=np.uint8)
    _save_mask(gt_path, data, np.eye(4))
    shifted = np.eye(4)
    shifted[0, 3] = 100.0
    _save_mask(pred_path, data, shifted)

    with pytest.raises(ValueError, match="physical grids differ"):
        evaluate_nifti_segmentation(pred_path, gt_path)


def test_valid_perfect_and_empty_gt_cases_remain_supported(tmp_path):
    gt_path = tmp_path / "gt.nii.gz"
    pred_path = tmp_path / "pred.nii.gz"
    data = np.ones((2, 2, 2), dtype=np.uint8)
    _save_mask(gt_path, data, np.eye(4))
    _save_mask(pred_path, data, np.eye(4))
    assert evaluate_nifti_segmentation(pred_path, gt_path).dice == 1.0

    empty = np.zeros_like(data)
    _save_mask(gt_path, empty, np.eye(4))
    _save_mask(pred_path, empty, np.eye(4))
    assert math.isnan(evaluate_nifti_segmentation(pred_path, gt_path).dice)


def test_reflected_sheared_grid_uses_absolute_affine_determinant(tmp_path):
    gt_path = tmp_path / "gt_shear.nii.gz"
    pred_path = tmp_path / "pred_shear.nii.gz"
    data = np.ones((2, 2, 2), dtype=np.uint8)
    affine = np.array(
        [
            [2.0, 1.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, -20.0],
            [0.0, 0.0, -4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    _save_mask(gt_path, data, affine)
    _save_mask(pred_path, data, affine)

    metrics = evaluate_nifti_segmentation(pred_path, gt_path)

    assert metrics.voxel_volume_mm3 == pytest.approx(24.0)
    assert metrics.dice == 1.0
