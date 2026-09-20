"""Physical volume calculations for three-dimensional masks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validated_spacing(spacing_mm: Sequence[float]) -> np.ndarray:
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.shape != (3,):
        raise ValueError("spacing_mm must contain exactly three values in D, H, W order")
    if not np.all(np.isfinite(spacing)):
        raise ValueError("spacing_mm values must be finite")
    if np.any(spacing <= 0):
        raise ValueError("spacing_mm values must be strictly positive")
    return spacing


def _validated_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return a 3D boolean mask after rejecting ambiguous non-binary values."""
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError("mask must be a 3D array")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("mask must be boolean or contain only binary values 0 and 1")
    return array.astype(bool, copy=False)


def voxel_volume_ml(spacing_mm: Sequence[float]) -> float:
    """Return one voxel's volume in mL for D/H/W spacing given in millimetres."""
    spacing = _validated_spacing(spacing_mm)
    return float(np.prod(spacing) / 1000.0)


def mask_volume_ml(mask: np.ndarray, spacing_mm: Sequence[float]) -> float:
    """Return generic binary-mask volume in mL; an empty mask has volume 0.0 mL."""
    array = _validated_binary_mask(mask)
    return float(np.count_nonzero(array) * voxel_volume_ml(spacing_mm))
