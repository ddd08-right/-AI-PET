"""Segmentation loss functions."""

from pet_ai.losses.segmentation import bce_dice_loss, soft_dice_loss_from_logits

__all__ = ["bce_dice_loss", "soft_dice_loss_from_logits"]
