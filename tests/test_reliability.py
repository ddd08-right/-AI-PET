from __future__ import annotations

import numpy as np
import pytest

from pet_ai.reliability import (
    mean_pairwise_dice,
    pairwise_dice_scores,
    predictive_entropy,
    segmentation_disagreement,
    volume_coefficient_of_variation,
)


def test_identical_non_empty_masks_have_dice_one():
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = 1

    assert pairwise_dice_scores([mask, mask]).tolist() == [1.0]
    assert mean_pairwise_dice([mask, mask]) == 1.0
    assert segmentation_disagreement([mask, mask]) == 0.0


def test_disjoint_non_empty_masks_have_dice_zero():
    first = np.zeros((2, 2, 2), dtype=np.uint8)
    second = np.zeros_like(first)
    first[0, 0, 0] = 1
    second[1, 1, 1] = 1

    assert pairwise_dice_scores([first, second]).tolist() == [0.0]


def test_both_empty_masks_have_dice_one():
    empty = np.zeros((2, 2, 2), dtype=np.uint8)

    assert pairwise_dice_scores([empty, empty]).tolist() == [1.0]


def test_one_empty_mask_has_dice_zero():
    empty = np.zeros((2, 2, 2), dtype=np.uint8)
    non_empty = empty.copy()
    non_empty[0, 0, 0] = 1

    assert pairwise_dice_scores([empty, non_empty]).tolist() == [0.0]


@pytest.mark.parametrize("volumes", [[10, 10, 10, 10], [0, 0, 0]])
def test_stable_volumes_have_zero_coefficient_of_variation(volumes):
    assert volume_coefficient_of_variation(volumes) == 0.0


def test_negative_physical_volume_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        volume_coefficient_of_variation([1, -1, 2])


def test_entropy_is_maximal_near_half_probability():
    midpoint = predictive_entropy([np.full((2, 2, 2), 0.5)])
    near_zero = predictive_entropy([np.full((2, 2, 2), 0.001)])
    near_one = predictive_entropy([np.full((2, 2, 2), 0.999)])

    assert np.all(midpoint > near_zero)
    assert np.all(midpoint > near_one)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_probability_outside_unit_interval_is_rejected(probability):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        predictive_entropy([np.full((2, 2, 2), probability)])
