from __future__ import annotations

import pytest

from adaptive_pid.control.gain_scheduler import GainScheduler
from adaptive_pid.utils.types import ControlLimits, PIDGains


def make_limits(**overrides) -> ControlLimits:
    defaults = {
        "kp_min": 0.0,
        "kp_max": 10.0,
        "ki_min": 0.0,
        "ki_max": 5.0,
        "kd_min": 0.0,
        "kd_max": 2.0,
        "kp_rate_max": 1.0,
        "ki_rate_max": 0.5,
        "kd_rate_max": 0.2,
        "u_max": 100.0,
        "integral_max": 50.0,
    }
    defaults.update(overrides)
    return ControlLimits(**defaults)


class TestInitialGains:
    def test_clamps_out_of_range_initial_gains(self):
        scheduler = GainScheduler(make_limits())
        gains = scheduler.initial_gains(kp=999.0, ki=-5.0, kd=1.0)
        assert gains.kp == pytest.approx(10.0)
        assert gains.ki == pytest.approx(0.0)
        assert gains.kd == pytest.approx(1.0)


class TestApplyAction:
    def test_zero_action_does_not_change_gains(self):
        scheduler = GainScheduler(make_limits())
        current = PIDGains(kp=5.0, ki=1.0, kd=0.5)
        new = scheduler.apply_action(current, (0.0, 0.0, 0.0), dt_outer=0.1)
        assert new.kp == pytest.approx(5.0)
        assert new.ki == pytest.approx(1.0)
        assert new.kd == pytest.approx(0.5)

    def test_positive_action_increases_gains_by_rate_times_dt(self):
        scheduler = GainScheduler(make_limits())
        current = PIDGains(kp=5.0, ki=1.0, kd=0.5)
        new = scheduler.apply_action(current, (1.0, 1.0, 1.0), dt_outer=0.1)
        assert new.kp == pytest.approx(5.0 + 1.0 * 1.0 * 0.1)
        assert new.ki == pytest.approx(1.0 + 0.5 * 1.0 * 0.1)
        assert new.kd == pytest.approx(0.5 + 0.2 * 1.0 * 0.1)

    def test_action_is_defensively_clipped_beyond_unit_range(self):
        """Even if a caller passes an out-of-[-1,1] action, the rate of
        change must not exceed rate_max * dt_outer."""
        scheduler = GainScheduler(make_limits())
        current = PIDGains(kp=5.0, ki=1.0, kd=0.5)
        new = scheduler.apply_action(current, (100.0, 100.0, 100.0), dt_outer=1.0)
        assert new.kp == pytest.approx(5.0 + 1.0)  # kp_rate_max=1.0, not 100.0
        assert new.ki == pytest.approx(1.0 + 0.5)

    def test_result_is_clamped_to_absolute_limits(self):
        scheduler = GainScheduler(make_limits(kp_max=5.5))
        current = PIDGains(kp=5.0, ki=1.0, kd=0.5)
        new = scheduler.apply_action(current, (1.0, 0.0, 0.0), dt_outer=1.0)
        assert new.kp == pytest.approx(5.5)  # would be 6.0 unclamped

    def test_negative_action_decreases_gains_and_floors_at_min(self):
        scheduler = GainScheduler(make_limits(kp_min=4.8))
        current = PIDGains(kp=5.0, ki=1.0, kd=0.5)
        new = scheduler.apply_action(current, (-1.0, 0.0, 0.0), dt_outer=1.0)
        assert new.kp == pytest.approx(4.8)  # would be 4.0 unclamped
