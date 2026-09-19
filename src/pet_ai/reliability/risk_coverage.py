"""Patient-level quantitative errors and deterministic risk--coverage curves."""

from __future__ import annotations

import numpy as np


def absolute_error(predicted: float, reference: float) -> float:
    """Return absolute patient-level error ``|predicted - reference|``."""
    if not np.isfinite(predicted) or not np.isfinite(reference):
        raise ValueError("predicted and reference must be finite")
    return float(abs(predicted - reference))


def relative_error(predicted: float, reference: float) -> float:
    """Return absolute error divided by ``|reference|``.

    When reference is exactly zero, relative error is mathematically undefined
    and this function returns NaN; it does not apply an arbitrary epsilon.
    """
    if not np.isfinite(predicted) or not np.isfinite(reference):
        raise ValueError("predicted and reference must be finite")
    if reference == 0:
        return float("nan")
    return absolute_error(predicted, reference) / abs(reference)


def _validated_observations(
    errors: np.ndarray | list[float], risk_scores: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    error_values = np.asarray(errors, dtype=float)
    score_values = np.asarray(risk_scores, dtype=float)
    if error_values.ndim != 1 or score_values.ndim != 1:
        raise ValueError("errors and risk_scores must be one-dimensional")
    if error_values.size == 0:
        raise ValueError("errors and risk_scores must not be empty")
    if error_values.size != score_values.size:
        raise ValueError("errors and risk_scores must have equal lengths")
    if not np.all(np.isfinite(error_values)):
        raise ValueError("errors contain NaN or infinite values; no observations were dropped")
    if np.any(error_values < 0):
        raise ValueError("errors must be non-negative")
    if not np.all(np.isfinite(score_values)):
        raise ValueError("risk_scores contain NaN or infinite values; no observations were dropped")
    return error_values, score_values


def risk_coverage_curve(
    errors: np.ndarray | list[float], risk_scores: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage and mean true error for lowest-risk covered cases.

    Larger risk scores mean less reliable cases. Stable ascending sorting keeps
    input order for tied scores. The grid is ``[1/n, 2/n, ..., 1]``; zero
    coverage is excluded because mean error for an empty covered set is
    undefined. NaN and infinite observations raise ValueError and none are
    silently excluded.
    """
    error_values, score_values = _validated_observations(errors, risk_scores)
    order = np.argsort(score_values, kind="stable")
    sorted_errors = error_values[order]
    counts = np.arange(1, sorted_errors.size + 1, dtype=float)
    coverage = counts / sorted_errors.size
    risk = np.cumsum(sorted_errors) / counts
    return coverage, risk


def aurc(errors: np.ndarray | list[float], risk_scores: np.ndarray | list[float]) -> float:
    """Return area under the risk--coverage curve; lower is better.

    Numerical integration uses a right-endpoint Riemann sum over the equally
    spaced coverage intervals of width ``1/n``. The zero-coverage point remains
    excluded because its risk is undefined.
    """
    coverage, risk = risk_coverage_curve(errors, risk_scores)
    interval_width = 1.0 / coverage.size
    return float(np.sum(risk) * interval_width)
