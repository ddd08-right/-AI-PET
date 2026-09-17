from __future__ import annotations

import torch

from pet_ai.losses.segmentation import bce_dice_loss, soft_dice_loss_from_logits


def test_soft_dice_loss_from_logits_is_finite() -> None:
    logits = torch.zeros(2, 1, 8, 8, 8)
    targets = torch.zeros(2, 1, 8, 8, 8)
    targets[:, :, 2:5, 2:5, 2:5] = 1.0

    loss = soft_dice_loss_from_logits(logits, targets)

    assert torch.isfinite(loss)


def test_bce_dice_loss_from_logits_is_finite() -> None:
    logits = torch.randn(2, 1, 8, 8, 8)
    targets = torch.zeros(2, 1, 8, 8, 8)
    targets[:, :, 1:4, 1:4, 1:4] = 1.0

    loss = bce_dice_loss(logits, targets)

    assert torch.isfinite(loss)
