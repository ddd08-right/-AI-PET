from __future__ import annotations

import math

import numpy as np
import pytest

from pet_ai.reliability import absolute_error, aurc, relative_error, risk_coverage_curve


def test_absolute_and_relative_error_follow_definitions():
    assert absolute_error(8.0, 10.0) == 2.0
    assert relative_error(8.0, 10.0) == 0.2


def test_zero_reference_relative_error_remains_undefined():
    assert math.isnan(relative_error(1.0, 0.0))

    with pytest.raises(ValueError, match="no observations were dropped"):
        risk_coverage_curve([relative_error(1.0, 0.0)], [0.5])


def test_full_coverage_risk_equals_complete_cohort_mean():
    errors = np.array([0.4, 0.1, 0.8, 0.2])
    risk_scores = np.array([0.3, 0.1, 0.4, 0.2])

    coverage, risk = risk_coverage_curve(errors, risk_scores)

    assert coverage[-1] == 1.0
    assert risk[-1] == pytest.approx(np.mean(errors))


def test_best_case_reference_ranking_beats_reverse_ranking():
    errors = np.array([0.1, 0.4, 0.2, 0.9, 0.6])

    assert aurc(errors, errors) < aurc(errors, -errors)


def test_stable_ties_preserve_input_order_and_repeat_deterministically():
    errors = np.array([0.4, 0.1, 0.8, 0.2])
    tied_scores = np.ones(4)

    first_coverage, first_risk = risk_coverage_curve(errors, tied_scores)
    second_coverage, second_risk = risk_coverage_curve(errors, tied_scores)

    assert first_risk[0] == errors[0]
    np.testing.assert_array_equal(first_coverage, second_coverage)
    np.testing.assert_array_equal(first_risk, second_risk)
    assert aurc(errors, tied_scores) == aurc(errors, tied_scores)


def test_mismatched_lengths_fail_clearly():
    with pytest.raises(ValueError, match="equal lengths"):
        risk_coverage_curve([0.1, 0.2], [0.3])
