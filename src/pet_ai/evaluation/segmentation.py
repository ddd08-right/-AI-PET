"""Binary segmentation metrics for PET/CT lesion masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from pet_ai.qc.geometry import compare_nifti_geometry, load_nifti_geometry


@dataclass(frozen=True)
class SegmentationCounts:
    true_positive_voxels: int
    false_positive_voxels: int
    false_negative_voxels: int
    gt_positive_voxels: int
    prediction_positive_voxels: int


@dataclass(frozen=True)
class SegmentationMetrics:
    counts: SegmentationCounts
    dice: float
    fpv_ml: float
    fnv_ml: float
    voxel_volume_mm3: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if np.isnan(self.dice):
            data["dice"] = None
            data["dice_status"] = "UNDEFINED_EMPTY_GT"
        else:
            data["dice_status"] = "DEFINED"
        return data


def voxel_counts(prediction: np.ndarray, ground_truth: np.ndarray) -> SegmentationCounts:
    prediction = _validated_binary_array(prediction, name="prediction")
    ground_truth = _validated_binary_array(ground_truth, name="ground_truth")
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"prediction and ground_truth shapes differ: {prediction.shape} != {ground_truth.shape}")

    pred = prediction.astype(bool, copy=False)
    gt = ground_truth.astype(bool, copy=False)
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    return SegmentationCounts(
        true_positive_voxels=tp,
        false_positive_voxels=fp,
        false_negative_voxels=fn,
        gt_positive_voxels=int(np.count_nonzero(gt)),
        prediction_positive_voxels=int(np.count_nonzero(pred)),
    )


def _validated_binary_array(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 3:
        raise ValueError(f"{name} must be a 3D array; observed shape={value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError(f"{name} must be boolean or contain only binary values 0 and 1")
    return value


def dice_from_counts(counts: SegmentationCounts) -> float:
    if counts.gt_positive_voxels == 0:
        return float("nan")
    denominator = 2 * counts.true_positive_voxels + counts.false_positive_voxels + counts.false_negative_voxels
    if denominator == 0:
        return float("nan")
    return 2 * counts.true_positive_voxels / denominator


def evaluate_binary_segmentation(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    voxel_volume_mm3: float,
) -> SegmentationMetrics:
    if not np.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
        raise ValueError("voxel_volume_mm3 must be finite and positive")
    counts = voxel_counts(prediction, ground_truth)
    return SegmentationMetrics(
        counts=counts,
        dice=dice_from_counts(counts),
        fpv_ml=counts.false_positive_voxels * voxel_volume_mm3 / 1000.0,
        fnv_ml=counts.false_negative_voxels * voxel_volume_mm3 / 1000.0,
        voxel_volume_mm3=float(voxel_volume_mm3),
    )


def evaluate_nifti_segmentation(prediction_path: Path, ground_truth_path: Path) -> SegmentationMetrics:
    geometry = compare_nifti_geometry(ground_truth_path, prediction_path)
    if not geometry.ok:
        raise ValueError("prediction and ground_truth physical grids differ: " + "; ".join(geometry.messages))
    prediction_image = nib.load(str(prediction_path))
    ground_truth_image = nib.load(str(ground_truth_path))
    prediction = np.asanyarray(prediction_image.dataobj)
    ground_truth = np.asanyarray(ground_truth_image.dataobj)
    validated = load_nifti_geometry(ground_truth_path)
    voxel_volume_mm3 = abs(float(np.linalg.det(np.asarray(validated.affine)[:3, :3])))
    return evaluate_binary_segmentation(prediction, ground_truth, voxel_volume_mm3=voxel_volume_mm3)
