"""NIfTI geometry comparison helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class ImageGeometry:
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    orientation: tuple[str, ...]
    affine: list[list[float]]


@dataclass(frozen=True)
class GeometryQCResult:
    reference: ImageGeometry
    moving: ImageGeometry
    shape_match: bool
    spacing_match: bool
    orientation_match: bool
    affine_match: bool
    ok: bool
    messages: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reference"] = asdict(self.reference)
        data["moving"] = asdict(self.moving)
        return data


def load_nifti_geometry(path: Path) -> ImageGeometry:
    image = nib.load(str(path))
    orientation = tuple(nib.aff2axcodes(image.affine))
    spacing = tuple(float(value) for value in image.header.get_zooms()[: len(image.shape)])
    return ImageGeometry(
        shape=tuple(int(value) for value in image.shape),
        spacing=spacing,
        orientation=orientation,
        affine=np.asarray(image.affine, dtype=float).round(8).tolist(),
    )


def compare_nifti_geometry(reference_path: Path, moving_path: Path, *, atol: float = 1e-5) -> GeometryQCResult:
    reference = load_nifti_geometry(reference_path)
    moving = load_nifti_geometry(moving_path)
    messages: list[str] = []

    shape_match = reference.shape == moving.shape
    if not shape_match:
        messages.append(f"shape mismatch: reference={reference.shape}, moving={moving.shape}")

    spacing_match = len(reference.spacing) == len(moving.spacing) and np.allclose(reference.spacing, moving.spacing, atol=atol)
    if not spacing_match:
        messages.append(f"spacing mismatch: reference={reference.spacing}, moving={moving.spacing}")

    orientation_match = reference.orientation == moving.orientation
    if not orientation_match:
        messages.append(f"orientation mismatch: reference={reference.orientation}, moving={moving.orientation}")

    affine_match = np.allclose(np.asarray(reference.affine), np.asarray(moving.affine), atol=atol)
    if not affine_match:
        messages.append("affine mismatch")

    ok = shape_match and spacing_match and orientation_match and affine_match
    return GeometryQCResult(
        reference=reference,
        moving=moving,
        shape_match=shape_match,
        spacing_match=spacing_match,
        orientation_match=orientation_match,
        affine_match=affine_match,
        ok=ok,
        messages=messages,
    )
