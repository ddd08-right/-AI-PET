"""Deterministic synthetic demonstration of quantitative reliability methods."""

from __future__ import annotations

import numpy as np

from pet_ai.quantification import mask_volume_ml
from pet_ai.reliability import (
    absolute_error,
    aurc,
    predictive_entropy,
    risk_coverage_curve,
    segmentation_disagreement,
    volume_coefficient_of_variation,
)

SPACING_MM = (2.0, 2.0, 2.0)
RANDOM_SEED = 17


def cuboid(start: tuple[int, int, int], stop: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros((12, 12, 12), dtype=np.uint8)
    mask[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = 1
    return mask


def synthetic_inputs() -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Return controlled mathematical masks with increasing perturbations."""
    reference = cuboid((3, 3, 3), (8, 8, 8))
    shifted = cuboid((4, 3, 3), (9, 8, 8))
    smaller = cuboid((3, 3, 3), (7, 8, 8))
    larger = cuboid((2, 3, 3), (9, 8, 8))
    far_shift = cuboid((6, 3, 3), (11, 8, 8))
    empty = np.zeros_like(reference)
    return [
        (reference, [reference.copy(), reference.copy(), reference.copy()]),
        (reference, [reference.copy(), shifted, reference.copy()]),
        (reference, [smaller, reference.copy(), larger]),
        (reference, [smaller, shifted, far_shift]),
        (reference, [empty, smaller, far_shift]),
    ]


def probability_map(mask: np.ndarray) -> np.ndarray:
    """Map a binary synthetic mask to deliberately generic probabilities."""
    return np.where(mask != 0, 0.9, 0.1)


def main() -> None:
    print("SYNTHETIC DATA ONLY")
    print("ENGINEERING / METHODOLOGY DEMO")
    print("NO CLINICAL PERFORMANCE CLAIM")
    print()

    errors: list[float] = []
    candidate_scores: list[float] = []
    for case_index, (reference, predictions) in enumerate(synthetic_inputs(), start=1):
        reference_volume = mask_volume_ml(reference, SPACING_MM)
        predicted_volumes = [mask_volume_ml(mask, SPACING_MM) for mask in predictions]
        predicted_volume = float(np.mean(predicted_volumes))
        error = absolute_error(predicted_volume, reference_volume)
        disagreement = segmentation_disagreement(predictions)
        volume_cv = volume_coefficient_of_variation(predicted_volumes)
        entropy = float(np.mean(predictive_entropy([probability_map(mask) for mask in predictions])))
        candidate_score = disagreement + volume_cv + entropy
        errors.append(error)
        candidate_scores.append(candidate_score)
        print(
            f"case_index={case_index} reference_volume_ml={reference_volume:.6f} "
            f"predicted_volume_ml={predicted_volume:.6f} absolute_error_ml={error:.6f} "
            f"disagreement={disagreement:.6f} volume_cv={volume_cv:.6f} "
            f"mean_predictive_entropy={entropy:.6f}"
        )

    error_array = np.asarray(errors)
    rng = np.random.default_rng(RANDOM_SEED)
    rankings = {
        "candidate": np.asarray(candidate_scores),
        "random_fixed_seed": rng.random(error_array.size),
        "best_case_reference": error_array,
        "reverse_negative_control": -error_array,
    }
    print()
    print("Ranking comparison (best-case reference is not deployable):")
    for name, scores in rankings.items():
        coverage, risk = risk_coverage_curve(error_array, scores)
        print(
            f"{name}: aurc={aurc(error_array, scores):.6f}, "
            f"full_coverage={coverage[-1]:.6f}, full_risk_ml={risk[-1]:.6f}"
        )


if __name__ == "__main__":
    main()
