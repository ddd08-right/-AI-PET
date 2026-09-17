"""Native PyTorch segmentation losses computed from logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _check_matching_shape(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(f"logits and targets must have the same shape: {logits.shape} != {targets.shape}")


def soft_dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Compute soft Dice loss from raw logits.

    Sigmoid is applied inside this function. Empty-target batches are not hidden:
    when both prediction probabilities and targets are near empty, smoothing
    keeps the loss finite and close to zero rather than reporting a clinical
    Dice score.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    _check_matching_shape(logits, targets)

    probabilities = torch.sigmoid(logits)
    reduce_dims = tuple(range(1, logits.ndim))
    intersection = torch.sum(probabilities * targets, dim=reduce_dims)
    denominator = torch.sum(probabilities, dim=reduce_dims) + torch.sum(targets, dim=reduce_dims)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Combine BCEWithLogitsLoss and soft Dice loss.

    BCEWithLogitsLoss receives raw logits directly. Targets are expected to be
    float tensors with binary values in [0, 1].
    """

    if dice_weight < 0 or bce_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if dice_weight == 0 and bce_weight == 0:
        raise ValueError("at least one loss weight must be positive")
    _check_matching_shape(logits, targets)

    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss_from_logits(logits, targets, epsilon=epsilon)
    return bce_weight * bce + dice_weight * dice
