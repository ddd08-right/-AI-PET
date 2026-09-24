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
    spatial_unit: str
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


_UNIT_TO_MM = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}


def _validated_spatial_geometry(image: nib.spatialimages.SpatialImage) -> tuple[np.ndarray, tuple[float, ...], str]:
    """Return affine and spacing in millimetres after strict NIfTI validation."""
    if len(image.shape) != 3:
        raise ValueError(f"NIfTI image must be 3D; observed shape={image.shape}")
    spatial_unit = image.header.get_xyzt_units()[0]
    if spatial_unit not in _UNIT_TO_MM:
        raise ValueError("NIfTI spatial unit must be explicitly set to mm, meter, or micron")
    scale = _UNIT_TO_MM[spatial_unit]
    affine = np.asarray(image.affine, dtype=float)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("NIfTI affine must be a finite 4x4 matrix")
    if not np.allclose(affine[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0):
        raise ValueError("NIfTI affine must have a valid homogeneous final row")
    affine_mm = affine.copy()
    affine_mm[:3, :] *= scale
    if abs(float(np.linalg.det(affine_mm[:3, :3]))) <= np.finfo(float).eps:
        raise ValueError("NIfTI affine is degenerate")

    spacing_native = np.asarray(image.header.get_zooms()[:3], dtype=float)
    spacing_mm = spacing_native * scale
    if spacing_mm.shape != (3,) or not np.all(np.isfinite(spacing_mm)) or np.any(spacing_mm <= 0):
        raise ValueError("NIfTI spacing must contain three finite positive values")
    affine_spacing_mm = np.linalg.norm(affine_mm[:3, :3], axis=0)
    if not np.allclose(spacing_mm, affine_spacing_mm, atol=1e-5, rtol=0):
        raise ValueError("NIfTI header spacing conflicts with the selected affine")

    coded_affines: list[tuple[str, np.ndarray]] = []
    for name, getter in (("qform", image.get_qform), ("sform", image.get_sform)):
        form, code = getter(coded=True)
        if int(code) == 0:
            continue
        form = np.asarray(form, dtype=float)
        if form.shape != (4, 4) or not np.all(np.isfinite(form)):
            raise ValueError(f"coded {name} must be a finite 4x4 matrix")
        form_mm = form.copy()
        form_mm[:3, :] *= scale
        if abs(float(np.linalg.det(form_mm[:3, :3]))) <= np.finfo(float).eps:
            raise ValueError(f"coded {name} is degenerate")
        coded_affines.append((name, form_mm))
    if len(coded_affines) == 2 and not np.allclose(
        coded_affines[0][1], coded_affines[1][1], atol=1e-5, rtol=0
    ):
        raise ValueError("coded qform and sform conflict")
    return affine_mm, tuple(float(value) for value in spacing_mm), spatial_unit


def load_nifti_geometry(path: Path) -> ImageGeometry:
    image = nib.load(str(path))
    affine_mm, spacing_mm, spatial_unit = _validated_spatial_geometry(image)
    orientation = tuple(nib.aff2axcodes(affine_mm))
    return ImageGeometry(
        shape=tuple(int(value) for value in image.shape),
        spacing=spacing_mm,
        spatial_unit=spatial_unit,
        orientation=orientation,
        affine=affine_mm.round(8).tolist(),
    )


def compare_nifti_geometry(reference_path: Path, moving_path: Path, *, atol: float = 1e-5) -> GeometryQCResult:
    reference = load_nifti_geometry(reference_path)
    moving = load_nifti_geometry(moving_path)
    messages: list[str] = []

    shape_match = reference.shape == moving.shape
    if not shape_match:
        messages.append(f"shape mismatch: reference={reference.shape}, moving={moving.shape}")

    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    spacing_match = len(reference.spacing) == len(moving.spacing) and np.allclose(
        reference.spacing, moving.spacing, atol=atol, rtol=0
    )
    if not spacing_match:
        messages.append(f"spacing mismatch: reference={reference.spacing}, moving={moving.spacing}")

    orientation_match = reference.orientation == moving.orientation
    if not orientation_match:
        messages.append(f"orientation mismatch: reference={reference.orientation}, moving={moving.orientation}")

    affine_match = np.allclose(
        np.asarray(reference.affine), np.asarray(moving.affine), atol=atol, rtol=0
    )
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
