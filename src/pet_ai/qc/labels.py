"""Segmentation label quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from pet_ai.qc.geometry import compare_nifti_geometry


@dataclass(frozen=True)
class LabelQCResult:
    ok: bool
    unique_labels: list[int | float]
    nonzero_voxels: int
    messages: list[str]


def validate_segmentation_labels(
    segmentation_path: Path,
    *,
    reference_path: Path | None = None,
    allowed_labels: set[int] | None = None,
    require_binary: bool = True,
    require_non_empty: bool = False,
) -> LabelQCResult:
    image = nib.load(str(segmentation_path))
    data = np.asanyarray(image.dataobj)
    messages: list[str] = []

    if np.isnan(data).any():
        messages.append("segmentation contains NaN")
    if np.isinf(data).any():
        messages.append("segmentation contains Inf")

    finite = data[np.isfinite(data)]
    unique = np.unique(finite) if finite.size else np.array([])
    integer_like = np.all(np.equal(unique, np.round(unique))) if unique.size else True
    if not integer_like:
        messages.append("segmentation contains non-integer labels")

    labels = {int(value) for value in unique if float(value).is_integer()}
    expected = allowed_labels if allowed_labels is not None else ({0, 1} if require_binary else labels)
    invalid = sorted(labels - expected)
    if invalid:
        messages.append(f"invalid labels: {invalid}; allowed labels: {sorted(expected)}")

    if require_binary and not labels.issubset({0, 1}):
        messages.append(f"segmentation is not binary: labels={sorted(labels)}")

    nonzero_voxels = int(np.count_nonzero(data))
    if require_non_empty and nonzero_voxels == 0:
        messages.append("segmentation is empty")

    if reference_path is not None:
        geometry = compare_nifti_geometry(reference_path, segmentation_path)
        if not geometry.ok:
            messages.extend(f"geometry: {message}" for message in geometry.messages)

    return LabelQCResult(
        ok=not messages,
        unique_labels=sorted(labels),
        nonzero_voxels=nonzero_voxels,
        messages=messages,
    )
