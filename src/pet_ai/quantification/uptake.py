"""Generic masked image-value summaries.

These functions do not assume that image values are SUV. SUVmean/SUVmax
terminology is appropriate only after PET values have been correctly converted
and independently validated as SUV-scaled data.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pet_ai.quantification.volume import _validated_binary_mask, mask_volume_ml


def _masked_values(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim != 3:
        raise ValueError("image and mask must both be 3D arrays")
    mask_array = _validated_binary_mask(mask)
    if image_array.shape != mask_array.shape:
        raise ValueError(
            f"image and mask shapes differ: {image_array.shape} != {mask_array.shape}"
        )
    values = image_array[mask_array]
    if values.size == 0:
        raise ValueError("masked statistic is undefined for an empty mask")
    if not np.all(np.isfinite(values)):
        raise ValueError("foreground image values must be finite")
    return values


def masked_mean(image: np.ndarray, mask: np.ndarray) -> float:
    """Return the mean finite image value in the binary mask.

    Raises ValueError when the mask is empty.
    """
    return float(np.mean(_masked_values(image, mask)))


def masked_max(image: np.ndarray, mask: np.ndarray) -> float:
    """Return the maximum finite image value in the binary mask.

    Raises ValueError when the mask is empty.
    """
    return float(np.max(_masked_values(image, mask)))


def uptake_volume_product(
    image: np.ndarray, mask: np.ndarray, spacing_mm: Sequence[float]
) -> float:
    """Return masked mean image value multiplied by generic mask volume in mL.

    The result has units of image-value times mL. It is undefined for an empty
    mask and is not automatically an SUV-derived or clinical quantity.
    """
    mean_value = masked_mean(image, mask)
    return mean_value * mask_volume_ml(mask, spacing_mm)
