"""Read-only QC for an external CT, PET, and binary reference-mask triplet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from pet_ai.qc.geometry import compare_nifti_geometry, load_nifti_geometry
from pet_ai.qc.labels import validate_segmentation_labels
from pet_ai.quantification.volume import mask_volume_ml, voxel_volume_ml


@dataclass(frozen=True)
class ExaminationAudit:
    """Public-safe QC result with no source paths or examination identifiers."""

    ok: bool
    readable: bool
    shape: tuple[int, ...] | None
    spacing_mm: tuple[float, ...] | None
    voxel_volume_ml: float | None
    reference_foreground_voxels: int | None
    reference_volume_ml: float | None
    reference_empty: bool | None
    reference_labels: list[int | float] | None
    failure_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _missing_reasons(paths: dict[str, Path]) -> list[str]:
    return [f"missing {role} file" for role, path in paths.items() if not path.is_file()]


def _load_image(path: Path, role: str) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    try:
        image = nib.load(str(path))
        data = np.asanyarray(image.dataobj)
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
        raise ValueError(f"unreadable {role} file: {type(error).__name__}") from error
    return image, data


def audit_examination(
    ct_path: Path,
    pet_path: Path,
    reference_segmentation_path: Path,
    *,
    geometry_atol: float = 1e-5,
) -> ExaminationAudit:
    """Audit one external examination without changing, repairing, or resampling any file."""
    paths = {
        "CT": Path(ct_path),
        "PET": Path(pet_path),
        "reference segmentation": Path(reference_segmentation_path),
    }
    failures = _missing_reasons(paths)
    if failures:
        return ExaminationAudit(False, False, None, None, None, None, None, None, None, failures)

    loaded: dict[str, tuple[nib.spatialimages.SpatialImage, np.ndarray]] = {}
    for role, path in paths.items():
        try:
            loaded[role] = _load_image(path, role)
        except ValueError as error:
            failures.append(str(error))
    if failures:
        return ExaminationAudit(False, False, None, None, None, None, None, None, None, failures)

    for role, (_, data) in loaded.items():
        if not np.all(np.isfinite(data)):
            failures.append(f"{role} contains NaN or Inf")

    try:
        ct_geometry = load_nifti_geometry(paths["CT"])
    except ValueError as error:
        ct_geometry = None
        failures.append(f"CT geometry: {error}")
    if ct_geometry is not None:
        if len(ct_geometry.shape) != 3:
            failures.append(f"CT must be 3D; observed shape={ct_geometry.shape}")
        if len(ct_geometry.spacing) != 3:
            failures.append(f"CT spacing must have three values; observed={ct_geometry.spacing}")

    reference_geometry = None
    if ct_geometry is not None:
        try:
            pet_geometry = compare_nifti_geometry(paths["CT"], paths["PET"], atol=geometry_atol)
            failures.extend(f"PET geometry: {message}" for message in pet_geometry.messages)
        except ValueError as error:
            failures.append(f"PET geometry: {error}")
        try:
            reference_geometry = compare_nifti_geometry(
                paths["CT"], paths["reference segmentation"], atol=geometry_atol
            )
            failures.extend(
                f"reference segmentation geometry: {message}"
                for message in reference_geometry.messages
            )
        except ValueError as error:
            failures.append(f"reference segmentation geometry: {error}")

    label_result = validate_segmentation_labels(
        paths["reference segmentation"],
        allowed_labels={0, 1},
        require_binary=True,
        require_non_empty=False,
    )
    failures.extend(f"reference segmentation: {message}" for message in label_result.messages)

    spacing: tuple[float, ...] | None = None
    per_voxel_ml: float | None = None
    reference_volume: float | None = None
    ct_geometry_valid_for_volume = (
        ct_geometry is not None
        and len(ct_geometry.shape) == 3
        and len(ct_geometry.spacing) == 3
    )
    if ct_geometry is not None and len(ct_geometry.spacing) == 3:
        spacing = ct_geometry.spacing
    if ct_geometry_valid_for_volume:
        try:
            per_voxel_ml = voxel_volume_ml(spacing)
            if label_result.ok and reference_geometry is not None and reference_geometry.ok:
                reference_volume = mask_volume_ml(
                    loaded["reference segmentation"][1], spacing
                )
        except ValueError as error:
            failures.append(f"physical volume: {error}")

    return ExaminationAudit(
        ok=not failures,
        readable=True,
        shape=ct_geometry.shape if ct_geometry is not None else tuple(loaded["CT"][1].shape),
        spacing_mm=spacing,
        voxel_volume_ml=per_voxel_ml,
        reference_foreground_voxels=label_result.nonzero_voxels,
        reference_volume_ml=reference_volume,
        reference_empty=(label_result.nonzero_voxels == 0) if label_result.ok else None,
        reference_labels=label_result.unique_labels,
        failure_reasons=failures,
    )
