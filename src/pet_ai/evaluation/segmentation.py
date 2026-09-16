"""Binary segmentation metrics for PET/CT lesion masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


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
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"prediction and ground_truth shapes differ: {prediction.shape} != {ground_truth.shape}")

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)
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
    if voxel_volume_mm3 <= 0:
        raise ValueError("voxel_volume_mm3 must be positive")
    counts = voxel_counts(prediction, ground_truth)
    return SegmentationMetrics(
        counts=counts,
        dice=dice_from_counts(counts),
        fpv_ml=counts.false_positive_voxels * voxel_volume_mm3 / 1000.0,
        fnv_ml=counts.false_negative_voxels * voxel_volume_mm3 / 1000.0,
        voxel_volume_mm3=float(voxel_volume_mm3),
    )


def evaluate_nifti_segmentation(prediction_path: Path, ground_truth_path: Path) -> SegmentationMetrics:
    prediction_image = nib.load(str(prediction_path))
    ground_truth_image = nib.load(str(ground_truth_path))
    prediction = np.asanyarray(prediction_image.dataobj)
    ground_truth = np.asanyarray(ground_truth_image.dataobj)
    zooms = ground_truth_image.header.get_zooms()[:3]
    voxel_volume_mm3 = float(np.prod(zooms))
    return evaluate_binary_segmentation(prediction, ground_truth, voxel_volume_mm3=voxel_volume_mm3)
