
from __future__ import annotations

import math

import numpy as np

from pet_ai.evaluation.lesion_metrics import (
    connected_component_labels,
    evaluate_lesions,
    match_lesions,
)


def test_one_gt_lesion_perfectly_detected() -> None:
    gt = np.zeros((5, 5, 5), dtype=np.uint8)
    gt[1:3, 1:3, 1:3] = 1
    pred = gt.copy()

    result = evaluate_lesions(pred, gt, voxel_volume_mm3=8.0)

    assert result.gt_lesion_count == 1
    assert result.prediction_lesion_count == 1
    assert result.true_positive_lesions == 1
    assert result.false_positive_lesion_count == 0
    assert result.false_negative_lesion_count == 0
    assert result.lesion_sensitivity == 1.0
    assert result.gt_lesions[0].volume_ml == 8 * 8.0 / 1000.0


def test_two_gt_lesions_one_detected_one_missed() -> None:
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    gt[1:3, 1:3, 1:3] = 1
    gt[5:7, 5:7, 5:7] = 1
    pred = np.zeros_like(gt)
    pred[1:3, 1:3, 1:3] = 1

    result = evaluate_lesions(pred, gt, voxel_volume_mm3=1.0)

    assert result.gt_lesion_count == 2
    assert result.true_positive_lesions == 1
    assert result.false_negative_lesion_count == 1
    assert result.false_positive_lesion_count == 0
    assert result.lesion_sensitivity == 0.5


def test_extra_predicted_lesion_counts_as_false_positive() -> None:
    gt = np.zeros((8, 8, 8), dtype=np.uint8)
    gt[1:3, 1:3, 1:3] = 1
    pred = gt.copy()
    pred[5:7, 5:7, 5:7] = 1

    result = evaluate_lesions(pred, gt, voxel_volume_mm3=1.0)

    assert result.true_positive_lesions == 1
    assert result.false_positive_lesion_count == 1
    assert result.false_negative_lesion_count == 0


def test_empty_gt_with_false_positive_prediction_has_undefined_sensitivity() -> None:
    gt = np.zeros((5, 5, 5), dtype=np.uint8)
    pred = np.zeros_like(gt)
    pred[2, 2, 2] = 1

    result = evaluate_lesions(pred, gt, voxel_volume_mm3=4.0)

    assert result.gt_lesion_count == 0
    assert result.prediction_lesion_count == 1
    assert result.false_positive_lesion_count == 1
    assert math.isnan(result.lesion_sensitivity)
    assert result.to_dict()["lesion_sensitivity"] is None
    assert result.to_dict()["lesion_sensitivity_status"] == "UNDEFINED_EMPTY_GT"


def test_small_lesion_volume_calculation_and_failure_reporting() -> None:
    gt = np.zeros((6, 6, 6), dtype=np.uint8)
    gt[1, 1, 1] = 1
    gt[4:6, 4:6, 4:6] = 1
    pred = np.zeros_like(gt)
    pred[4:6, 4:6, 4:6] = 1

    result = evaluate_lesions(pred, gt, voxel_volume_mm3=125.0, small_lesion_threshold_ml=0.2)

    assert sorted(component.volume_ml for component in result.gt_lesions) == [0.125, 1.0]
    assert result.false_negative_lesion_count == 1
    assert len(result.small_lesion_failures) == 1
    assert result.small_lesion_failures[0].volume_ml == 0.125


def test_split_merge_ambiguity_is_reported_deterministically() -> None:
    gt = np.zeros((7, 7, 7), dtype=np.uint8)
    gt[2:5, 2:5, 2:5] = 1
    pred = np.zeros_like(gt)
    pred[2:5, 2:5, 2] = 1
    pred[2:5, 2:5, 4] = 1

    result = evaluate_lesions(pred, gt, voxel_volume_mm3=1.0, connectivity=6)

    assert result.gt_lesion_count == 1
    assert result.prediction_lesion_count == 2
    assert result.true_positive_lesions == 1
    assert result.false_positive_lesion_count == 1
    assert result.false_negative_lesion_count == 0
    assert result.matches[0].gt_component_id == 1
    assert result.matches[0].prediction_component_id == 1
    assert result.ambiguous_matches == [
        "GT component 1 overlaps multiple prediction components [1, 2]"
    ]


def test_connectivity_changes_component_count() -> None:
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[1, 1, 1] = 1

    _, components_6 = connected_component_labels(mask, connectivity=6)
    _, components_26 = connected_component_labels(mask, connectivity=26)

    assert len(components_6) == 2
    assert len(components_26) == 1


def test_greedy_matching_does_not_guarantee_maximum_cardinality() -> None:
    gt_labels = np.zeros((1, 1, 25), dtype=np.int32)
    pred_labels = np.zeros_like(gt_labels)
    gt_labels[..., :17] = 1
    gt_labels[..., 17:] = 2
    pred_labels[..., :9] = 1
    pred_labels[..., 9:17] = 2
    pred_labels[..., 17:] = 1

    matches, _ = match_lesions(gt_labels, pred_labels)

    observed = [
        (item.gt_component_id, item.prediction_component_id, item.overlap_voxels)
        for item in matches
    ]
    assert observed == [(1, 1, 9)]
