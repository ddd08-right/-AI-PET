"""Patient-level reliability candidates and risk--coverage evaluation."""

from pet_ai.reliability.disagreement import (
    mean_pairwise_dice,
    pairwise_dice_scores,
    predictive_entropy,
    segmentation_disagreement,
    volume_coefficient_of_variation,
)
from pet_ai.reliability.risk_coverage import (
    absolute_error,
    aurc,
    relative_error,
    risk_coverage_curve,
)

__all__ = [
    "absolute_error",
    "aurc",
    "mean_pairwise_dice",
    "pairwise_dice_scores",
    "predictive_entropy",
    "relative_error",
    "risk_coverage_curve",
    "segmentation_disagreement",
    "volume_coefficient_of_variation",
]
