from __future__ import annotations

import math

import numpy as np

from pet_ai.evaluation.segmentation import evaluate_binary_segmentation


def test_known_dice_example_produces_expected_dice():
    gt = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    pred = np.array([1, 1, 0, 1, 0, 0], dtype=np.uint8)

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
    gt = np.array([1, 1, 0, 0], dtype=np.uint8)
    pred = np.array([0, 1, 1, 1], dtype=np.uint8)

    metrics = evaluate_binary_segmentation(pred, gt, voxel_volume_mm3=5.0)

    assert metrics.counts.false_positive_voxels == 2
    assert metrics.counts.false_negative_voxels == 1
    assert metrics.fpv_ml == 0.01
    assert metrics.fnv_ml == 0.005
