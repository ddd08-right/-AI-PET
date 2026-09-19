"""Simple ensemble-disagreement candidates for segmentation reliability."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validated_binary_masks(binary_masks: Sequence[np.ndarray]) -> list[np.ndarray]:
    masks = [np.asarray(mask) for mask in binary_masks]
    if len(masks) < 2:
        raise ValueError("at least two binary masks are required")
    expected_shape = masks[0].shape
    if len(expected_shape) != 3:
        raise ValueError("binary masks must be 3D")
    for mask in masks:
        if mask.ndim != 3:
            raise ValueError("binary masks must be 3D")
        if mask.shape != expected_shape:
            raise ValueError("all binary masks must have the same shape")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("binary masks may contain only 0 and 1")
    return [mask.astype(bool, copy=False) for mask in masks]


def pairwise_dice_scores(binary_masks: Sequence[np.ndarray]) -> np.ndarray:
    """Return Dice for every unique mask pair in deterministic input order.

    Two empty masks have Dice 1.0; one empty and one non-empty mask have Dice
    0.0. Otherwise Dice is twice the intersection divided by the foreground
    voxel-count sum.
    """
    masks = _validated_binary_masks(binary_masks)
    scores: list[float] = []
    for first_index, first in enumerate(masks[:-1]):
        for second in masks[first_index + 1 :]:
            first_count = int(np.count_nonzero(first))
            second_count = int(np.count_nonzero(second))
            denominator = first_count + second_count
            if denominator == 0:
                scores.append(1.0)
            else:
                intersection = int(np.count_nonzero(first & second))
                scores.append(2.0 * intersection / denominator)
    return np.asarray(scores, dtype=float)


def mean_pairwise_dice(binary_masks: Sequence[np.ndarray]) -> float:
    """Return mean pairwise Dice; higher values indicate greater agreement."""
    return float(np.mean(pairwise_dice_scores(binary_masks)))


def segmentation_disagreement(binary_masks: Sequence[np.ndarray]) -> float:
    """Return 1 - mean pairwise Dice; higher values indicate predicted risk."""
    return 1.0 - mean_pairwise_dice(binary_masks)


def volume_coefficient_of_variation(volumes: Sequence[float]) -> float:
    """Return population SD divided by mean for non-negative physical volumes.

    Exactly all-zero input returns 0.0 because its predictions are stable at
    zero volume.
    """
    values = np.asarray(volumes, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("volumes must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)):
        raise ValueError("volumes must be finite")
    if np.any(values < 0):
        raise ValueError("physical volumes must be non-negative")
    if np.all(values == 0):
        return 0.0
    return float(np.std(values, ddof=0) / np.mean(values))


def predictive_entropy(probability_maps: Sequence[np.ndarray]) -> np.ndarray:
    """Return binary entropy of the ensemble-mean probability map.

    Inputs are same-shaped 3D probability maps in [0, 1], not logits. For log
    evaluation only, probabilities are clipped to [eps, 1-eps], where eps is
    machine epsilon for float64. The conceptual probabilities are unchanged.
    """
    maps = [np.asarray(probability_map, dtype=float) for probability_map in probability_maps]
    if not maps:
        raise ValueError("at least one probability map is required")
    expected_shape = maps[0].shape
    if len(expected_shape) != 3:
        raise ValueError("probability maps must be 3D")
    for probability_map in maps:
        if probability_map.ndim != 3 or probability_map.shape != expected_shape:
            raise ValueError("probability maps must be same-shaped 3D arrays")
        if not np.all(np.isfinite(probability_map)):
            raise ValueError("probability maps must be finite")
        if np.any((probability_map < 0) | (probability_map > 1)):
            raise ValueError("probability maps must contain values in [0, 1], not logits")
    mean_probability = np.mean(np.stack(maps), axis=0)
    epsilon = np.finfo(np.float64).eps
    safe_probability = np.clip(mean_probability, epsilon, 1.0 - epsilon)
    return -safe_probability * np.log(safe_probability) - (
        1.0 - safe_probability
    ) * np.log(1.0 - safe_probability)
