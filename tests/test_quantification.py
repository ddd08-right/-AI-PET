from __future__ import annotations

import numpy as np
import pytest

from pet_ai.quantification import (
    mask_volume_ml,
    masked_max,
    masked_mean,
    uptake_volume_product,
    voxel_volume_ml,
)


def test_one_thousand_unit_voxels_equal_one_ml():
    mask = np.ones((10, 10, 10), dtype=np.uint8)

    assert mask_volume_ml(mask, [1, 1, 1]) == 1.0


def test_doubling_one_spacing_dimension_doubles_volume():
    mask = np.ones((2, 3, 4), dtype=np.uint8)

    baseline = mask_volume_ml(mask, [1, 1, 1])
    doubled = mask_volume_ml(mask, [2, 1, 1])

    assert doubled == 2 * baseline


def test_empty_mask_has_zero_volume():
    assert mask_volume_ml(np.zeros((2, 2, 2)), [1, 1, 1]) == 0.0


@pytest.mark.parametrize(
    "spacing",
    ([1, 1], [1, 1, 1, 1], [0, 1, 1], [-1, 1, 1], [np.nan, 1, 1], [np.inf, 1, 1]),
)
def test_invalid_spacing_is_rejected(spacing):
    with pytest.raises(ValueError):
        voxel_volume_ml(spacing)


def test_image_mask_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shapes differ"):
        masked_mean(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)))


def test_masked_statistics_and_product_follow_generic_definitions():
    image = np.arange(8, dtype=float).reshape(2, 2, 2)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[0, 0, 1] = 1
    mask[1, 1, 1] = 1

    assert masked_mean(image, mask) == 4.0
    assert masked_max(image, mask) == 7.0
    assert uptake_volume_product(image, mask, [1, 1, 1]) == 0.008


@pytest.mark.parametrize("function", [masked_mean, masked_max])
def test_empty_masked_statistic_is_explicitly_undefined(function):
    with pytest.raises(ValueError, match="empty mask"):
        function(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)))


def test_non_finite_foreground_image_value_is_rejected():
    image = np.ones((2, 2, 2))
    image[0, 0, 0] = np.nan
    mask = np.zeros_like(image)
    mask[0, 0, 0] = 1

    with pytest.raises(ValueError, match="finite"):
        masked_mean(image, mask)
