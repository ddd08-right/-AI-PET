
"""Lesion-level evaluation for binary 3D segmentation masks.

Matching policy
---------------
Ground-truth and predicted lesions are first identified as connected components.
A predicted lesion can match a ground-truth lesion only when they overlap by at
least one voxel. Matches are one-to-one and deterministic: candidate pairs are
sorted by descending overlap voxel count, then ascending ground-truth component
ID, then ascending prediction component ID. Greedy assignment is then applied.

This policy is simple and auditable, but overlap matching cannot fully resolve
split/merge ambiguity. If one ground-truth component overlaps multiple predicted
components, or one predicted component overlaps multiple ground-truth components,
the ambiguity is reported in ``ambiguous_matches`` instead of being hidden.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LesionComponent:
    """A connected component measured in voxels and milliliters."""

    component_id: int
    voxel_count: int
    volume_ml: float


@dataclass(frozen=True)
class LesionMatch:
    """One deterministic ground-truth to prediction component match."""

    gt_component_id: int
    prediction_component_id: int
    overlap_voxels: int


@dataclass(frozen=True)
class LesionEvaluation:
    """Lesion-level segmentation summary."""

    gt_lesions: list[LesionComponent]
    prediction_lesions: list[LesionComponent]
    matches: list[LesionMatch]
    false_negative_lesions: list[LesionComponent]
    false_positive_lesions: list[LesionComponent]
    small_lesion_failures: list[LesionComponent]
    ambiguous_matches: list[str]

    @property
    def gt_lesion_count(self) -> int:
        return len(self.gt_lesions)

    @property
    def prediction_lesion_count(self) -> int:
        return len(self.prediction_lesions)

    @property
    def true_positive_lesions(self) -> int:
        return len(self.matches)

    @property
    def false_positive_lesion_count(self) -> int:
        return len(self.false_positive_lesions)

    @property
    def false_negative_lesion_count(self) -> int:
        return len(self.false_negative_lesions)

    @property
    def lesion_sensitivity(self) -> float:
        if self.gt_lesion_count == 0:
            return float("nan")
        return self.true_positive_lesions / self.gt_lesion_count

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        sensitivity = self.lesion_sensitivity
        data.update(
            {
                "gt_lesion_count": self.gt_lesion_count,
                "prediction_lesion_count": self.prediction_lesion_count,
                "true_positive_lesions": self.true_positive_lesions,
                "false_positive_lesion_count": self.false_positive_lesion_count,
                "false_negative_lesion_count": self.false_negative_lesion_count,
                "lesion_sensitivity": None if np.isnan(sensitivity) else sensitivity,
                "lesion_sensitivity_status": "UNDEFINED_EMPTY_GT"
                if np.isnan(sensitivity)
                else "DEFINED",
            }
        )
        return data


def neighbor_offsets(connectivity: int) -> list[tuple[int, int, int]]:
    """Return 3D neighbor offsets for 6, 18, or 26 connectivity."""

    if connectivity not in {6, 18, 26}:
        raise ValueError("connectivity must be one of 6, 18, or 26")
    offsets: list[tuple[int, int, int]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                distance = abs(dz) + abs(dy) + abs(dx)
                if connectivity == 6 and distance == 1:
                    offsets.append((dz, dy, dx))
                elif connectivity == 18 and distance <= 2:
                    offsets.append((dz, dy, dx))
                elif connectivity == 26:
                    offsets.append((dz, dy, dx))
    return offsets


def connected_component_labels(
    mask: np.ndarray, *, connectivity: int = 26
) -> tuple[np.ndarray, list[LesionComponent]]:
    """Label connected foreground components in a 3D binary mask."""

    if mask.ndim != 3:
        raise ValueError(f"lesion metrics require 3D arrays, got shape {mask.shape}")
    foreground = mask.astype(bool)
    labels = np.zeros(foreground.shape, dtype=np.int32)
    components: list[LesionComponent] = []
    offsets = neighbor_offsets(connectivity)
    next_id = 1

    for start in np.argwhere(foreground):
        z, y, x = (int(value) for value in start)
        if labels[z, y, x] != 0:
            continue
        queue: deque[tuple[int, int, int]] = deque([(z, y, x)])
        labels[z, y, x] = next_id
        count = 0
        while queue:
            cz, cy, cx = queue.popleft()
            count += 1
            for dz, dy, dx in offsets:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                in_bounds = (
                    0 <= nz < foreground.shape[0]
                    and 0 <= ny < foreground.shape[1]
                    and 0 <= nx < foreground.shape[2]
                )
                if not in_bounds:
                    continue
                if foreground[nz, ny, nx] and labels[nz, ny, nx] == 0:
                    labels[nz, ny, nx] = next_id
                    queue.append((nz, ny, nx))
        components.append(LesionComponent(component_id=next_id, voxel_count=count, volume_ml=0.0))
        next_id += 1
    return labels, components


def _with_volumes(
    components: list[LesionComponent], *, voxel_volume_ml: float
) -> list[LesionComponent]:
    return [
        LesionComponent(item.component_id, item.voxel_count, item.voxel_count * voxel_volume_ml)
        for item in components
    ]


def _overlap_candidates(gt_labels: np.ndarray, pred_labels: np.ndarray) -> dict[tuple[int, int], int]:
    pairs: dict[tuple[int, int], int] = {}
    overlap = (gt_labels > 0) & (pred_labels > 0)
    for gt_id, pred_id in zip(gt_labels[overlap], pred_labels[overlap], strict=True):
        key = (int(gt_id), int(pred_id))
        pairs[key] = pairs.get(key, 0) + 1
    return pairs


def _ambiguous_overlap_messages(candidates: dict[tuple[int, int], int]) -> list[str]:
    gt_to_pred: dict[int, set[int]] = {}
    pred_to_gt: dict[int, set[int]] = {}
    for gt_id, pred_id in candidates:
        gt_to_pred.setdefault(gt_id, set()).add(pred_id)
        pred_to_gt.setdefault(pred_id, set()).add(gt_id)

    messages: list[str] = []
    for gt_id, pred_ids in sorted(gt_to_pred.items()):
        if len(pred_ids) > 1:
            messages.append(
                f"GT component {gt_id} overlaps multiple prediction components {sorted(pred_ids)}"
            )
    for pred_id, gt_ids in sorted(pred_to_gt.items()):
        if len(gt_ids) > 1:
            messages.append(
                f"Prediction component {pred_id} overlaps multiple GT components {sorted(gt_ids)}"
            )
    return messages


def match_lesions(gt_labels: np.ndarray, pred_labels: np.ndarray) -> tuple[list[LesionMatch], list[str]]:
    """Create deterministic one-to-one overlap matches between labeled lesions."""

    candidates = _overlap_candidates(gt_labels, pred_labels)
    ordered = sorted(candidates.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[LesionMatch] = []
    for (gt_id, pred_id), overlap_voxels in ordered:
        if gt_id in used_gt or pred_id in used_pred:
            continue
        matches.append(
            LesionMatch(
                gt_component_id=gt_id,
                prediction_component_id=pred_id,
                overlap_voxels=overlap_voxels,
            )
        )
        used_gt.add(gt_id)
        used_pred.add(pred_id)
    return matches, _ambiguous_overlap_messages(candidates)


def evaluate_lesions(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    voxel_volume_mm3: float,
    connectivity: int = 26,
    small_lesion_threshold_ml: float = 1.0,
) -> LesionEvaluation:
    """Evaluate lesion detection using connected components and overlap matching."""

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"prediction and ground_truth shapes differ: {prediction.shape} != {ground_truth.shape}"
        )
    if voxel_volume_mm3 <= 0:
        raise ValueError("voxel_volume_mm3 must be positive")
    if small_lesion_threshold_ml < 0:
        raise ValueError("small_lesion_threshold_ml must be non-negative")

    voxel_volume_ml = voxel_volume_mm3 / 1000.0
    gt_labels, gt_components_raw = connected_component_labels(ground_truth, connectivity=connectivity)
    pred_labels, pred_components_raw = connected_component_labels(prediction, connectivity=connectivity)
    gt_components = _with_volumes(gt_components_raw, voxel_volume_ml=voxel_volume_ml)
    pred_components = _with_volumes(pred_components_raw, voxel_volume_ml=voxel_volume_ml)
    matches, ambiguity = match_lesions(gt_labels, pred_labels)

    matched_gt = {match.gt_component_id for match in matches}
    matched_pred = {match.prediction_component_id for match in matches}
    false_negative = [component for component in gt_components if component.component_id not in matched_gt]
    false_positive = [
        component for component in pred_components if component.component_id not in matched_pred
    ]
    small_failures = [
        component for component in false_negative if component.volume_ml <= small_lesion_threshold_ml
    ]

    return LesionEvaluation(
        gt_lesions=gt_components,
        prediction_lesions=pred_components,
        matches=matches,
        false_negative_lesions=false_negative,
        false_positive_lesions=false_positive,
        small_lesion_failures=small_failures,
        ambiguous_matches=ambiguity,
    )
