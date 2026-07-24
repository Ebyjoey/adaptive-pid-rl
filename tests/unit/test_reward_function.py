from __future__ import annotations

import pytest

from adaptive_pid.rewards.reward_function import (
    compute_reward,
    energy_term,
    gain_smoothness_term,
    oscillation_term,
    overshoot_term,
    saturation_term,
    settling_deficit_term,
    tracking_term,
)
from adaptive_pid.utils.types import PIDGains, RewardWeights


class TestTrackingTerm:
    def test_zero_error_zero_cost(self):
        assert tracking_term(0.0) == pytest.approx(0.0)

    def test_cost_is_quadratic_in_error(self):
        assert tracking_term(2.0) == pytest.approx(4.0)
        assert tracking_term(-3.0) == pytest.approx(9.0)


class TestOvershootTerm:
    def test_no_penalty_when_not_tracking_a_step(self):
        assert overshoot_term(theta=5.0, reference=1.0, reference_is_step=False) == 0.0

    def test_no_penalty_when_approaching_from_below(self):
        # reference=1.0 (positive), theta=0.5 has not yet passed the setpoint
        assert overshoot_term(theta=0.5, reference=1.0, reference_is_step=True) == pytest.approx(0.0)

    def test_penalty_when_overshooting_positive_reference(self):
        assert overshoot_term(theta=1.5, reference=1.0, reference_is_step=True) == pytest.approx(0.25)

    def test_penalty_when_overshooting_negative_reference(self):
        # reference = -1.0, theta = -1.5 means it went further negative than target
        assert overshoot_term(theta=-1.5, reference=-1.0, reference_is_step=True) == pytest.approx(0.25)


class TestSettlingDeficitTerm:
    def test_no_deficit_before_expected_settle_time(self):
        assert (
            settling_deficit_term(
                error=1.0, epsilon_settle=0.05, time_since_reference_change=0.5, expected_settle_time=2.0
            )
            == 0.0
        )

    def test_deficit_after_expected_time_if_still_outside_epsilon(self):
        assert (
            settling_deficit_term(
                error=1.0, epsilon_settle=0.05, time_since_reference_change=3.0, expected_settle_time=2.0
            )
            == 1.0
        )

    def test_no_deficit_after_expected_time_if_within_epsilon(self):
        assert (
            settling_deficit_term(
                error=0.01, epsilon_settle=0.05, time_since_reference_change=3.0, expected_settle_time=2.0
            )
            == 0.0
        )


class TestOscillationTerm:
    def test_zero_on_first_step(self):
        assert oscillation_term(derivative_error=5.0, prev_derivative_error=None) == 0.0

    def test_penalizes_change_in_derivative(self):
        assert oscillation_term(derivative_error=5.0, prev_derivative_error=2.0) == pytest.approx(9.0)


class TestEnergyTerm:
    def test_quadratic_in_control_effort(self):
        assert energy_term(3.0) == pytest.approx(9.0)
        assert energy_term(-3.0) == pytest.approx(9.0)


class TestGainSmoothnessTerm:
    def test_zero_when_gains_unchanged(self):
        gains = PIDGains(1.0, 2.0, 3.0)
        assert gain_smoothness_term(gains, gains) == pytest.approx(0.0)

    def test_penalizes_gain_change_magnitude(self):
        current = PIDGains(1.0, 0.0, 0.0)
        previous = PIDGains(0.0, 0.0, 0.0)
        assert gain_smoothness_term(current, previous) == pytest.approx(1.0)


class TestSaturationTerm:
    def test_no_penalty_below_threshold(self):
        assert saturation_term(control_effort=5.0, u_max=10.0, margin=0.1) == 0.0

    def test_penalty_at_or_above_threshold(self):
        assert saturation_term(control_effort=9.5, u_max=10.0, margin=0.1) == 1.0


class TestComputeReward:
    def _base_kwargs(self, **overrides):
        base = {
            "error": 0.0,
            "theta": 0.0,
            "reference": 0.0,
            "reference_is_step": True,
            "derivative_error": 0.0,
            "prev_derivative_error": 0.0,
            "control_effort": 0.0,
            "current_gains": PIDGains(1.0, 0.0, 0.0),
            "previous_gains": PIDGains(1.0, 0.0, 0.0),
            "u_max": 10.0,
            "epsilon_settle": 0.05,
            "time_since_reference_change": 0.1,
            "expected_settle_time": 2.0,
            "fell": False,
            "weights": RewardWeights(),
        }
        base.update(overrides)
        return base

    def test_perfect_tracking_gives_near_zero_negative_reward(self):
        reward, terms = compute_reward(**self._base_kwargs())
        assert reward == pytest.approx(0.0)
        assert terms.tracking == 0.0

    def test_fall_gives_exactly_negative_fall_penalty_and_ignores_other_terms(self):
        weights = RewardWeights(fall_penalty=42.0)
        reward, _terms = compute_reward(
            **self._base_kwargs(error=100.0, control_effort=100.0, fell=True, weights=weights)
        )
        assert reward == pytest.approx(-42.0)

    def test_reward_is_more_negative_with_larger_error(self):
        r_small, _ = compute_reward(**self._base_kwargs(error=0.1))
        r_large, _ = compute_reward(**self._base_kwargs(error=5.0))
        assert r_large < r_small

    def test_weights_scale_contribution(self):
        kwargs = self._base_kwargs(error=2.0)
        low_weight = RewardWeights(w_tracking=0.1)
        high_weight = RewardWeights(w_tracking=10.0)
        r_low, _ = compute_reward(**{**kwargs, "weights": low_weight})
        r_high, _ = compute_reward(**{**kwargs, "weights": high_weight})
        assert r_high < r_low  # more negative reward under a higher tracking weight
