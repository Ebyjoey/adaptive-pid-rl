"""Unit tests for adaptive_pid.control.pid.

These tests instantiate no MuJoCo, no Gymnasium, no ROS2 -- purely
synthetic error sequences -- by design (see docs/architecture.md Section 2).
"""

from __future__ import annotations

import pytest

from adaptive_pid.control.pid import PIDController
from adaptive_pid.utils.types import PIDGains


def make_pid(kp=1.0, ki=0.0, kd=0.0, dt=0.01, integral_max=10.0, output_max=100.0) -> PIDController:
    return PIDController(PIDGains(kp, ki, kd), dt=dt, integral_max=integral_max, output_max=output_max)


class TestProportionalOnly:
    def test_zero_error_gives_zero_output(self):
        pid = make_pid(kp=2.0)
        u, ie, de = pid.step(0.0)
        assert u == pytest.approx(0.0)
        assert ie == pytest.approx(0.0)
        assert de == pytest.approx(0.0)

    def test_output_scales_linearly_with_kp(self):
        pid = make_pid(kp=3.0)
        u, _, _ = pid.step(2.0)
        assert u == pytest.approx(6.0)

    def test_first_step_derivative_is_zero_not_spiked(self):
        """First-call derivative must be 0, not (error - 0)/dt, to avoid a
        spurious derivative-kick on controller startup."""
        pid = make_pid(kp=0.0, ki=0.0, kd=5.0)
        u, _, de = pid.step(10.0)
        assert de == pytest.approx(0.0)
        assert u == pytest.approx(0.0)


class TestIntegralTerm:
    def test_integral_accumulates_over_steps(self):
        pid = make_pid(kp=0.0, ki=1.0, kd=0.0, dt=0.1)
        pid.step(1.0)  # integral = 0.1
        u, ie, _ = pid.step(1.0)  # integral = 0.2
        assert ie == pytest.approx(0.2)
        assert u == pytest.approx(0.2)

    def test_anti_windup_clamps_integral(self):
        pid = make_pid(kp=0.0, ki=1.0, kd=0.0, dt=1.0, integral_max=5.0)
        for _ in range(20):
            _, ie, _ = pid.step(10.0)  # would accumulate to 200 without clamping
        assert ie == pytest.approx(5.0)
        assert ie <= 5.0 + 1e-9

    def test_integral_clamps_to_boundary_not_stuck_below_it(self):
        """A single step whose candidate integral overshoots the limit must
        clamp exactly to the boundary (not get stuck at its pre-step value,
        which would be a worse anti-windup bug than no clamping at all)."""
        pid = make_pid(kp=0.0, ki=1.0, kd=0.0, dt=1.0, integral_max=2.0)
        _, ie, _ = pid.step(5.0)  # candidate = 0 + 5 = 5 -> clamps to 2.0
        assert ie == pytest.approx(2.0)

    def test_integral_recovers_when_error_reverses_sign(self):
        """After saturating positive, a negative error must immediately start
        pulling the integral back down (proving it clamped, not latched)."""
        pid = make_pid(kp=0.0, ki=1.0, kd=0.0, dt=1.0, integral_max=2.0)
        pid.step(5.0)  # clamps to 2.0
        _, ie, _ = pid.step(-1.0)  # candidate = 2.0 - 1.0 = 1.0
        assert ie == pytest.approx(1.0)


class TestDerivativeTerm:
    def test_derivative_responds_to_error_change(self):
        pid = make_pid(kp=0.0, ki=0.0, kd=1.0, dt=1.0)
        pid.step(0.0)
        u, _, de = pid.step(5.0)
        assert de == pytest.approx(5.0)
        assert u == pytest.approx(5.0)


class TestSaturation:
    def test_output_is_clamped_to_output_max(self):
        pid = make_pid(kp=1000.0, output_max=10.0)
        u, _, _ = pid.step(5.0)
        assert u == pytest.approx(10.0)

    def test_negative_saturation(self):
        pid = make_pid(kp=1000.0, output_max=10.0)
        u, _, _ = pid.step(-5.0)
        assert u == pytest.approx(-10.0)


class TestGainUpdates:
    def test_set_gains_does_not_reset_integral(self):
        pid = make_pid(kp=0.0, ki=1.0, kd=0.0, dt=1.0)
        pid.step(3.0)
        pid.set_gains(PIDGains(kp=1.0, ki=2.0, kd=0.0))
        u, ie, _ = pid.step(0.0)
        assert ie == pytest.approx(3.0)  # integral preserved across gain change
        assert u == pytest.approx(6.0)  # 2.0 * 3.0

    def test_reset_clears_integral_and_derivative_history(self):
        pid = make_pid(kp=0.0, ki=1.0, kd=1.0, dt=1.0)
        pid.step(5.0)
        pid.reset()
        _u, ie, de = pid.step(5.0)
        assert ie == pytest.approx(5.0)
        assert de == pytest.approx(0.0)  # no history after reset -> no derivative spike


class TestConstructorValidation:
    @pytest.mark.parametrize("bad_dt", [0.0, -0.1])
    def test_rejects_non_positive_dt(self, bad_dt):
        with pytest.raises(ValueError):
            PIDController(PIDGains(1, 0, 0), dt=bad_dt, integral_max=1.0, output_max=1.0)

    def test_rejects_non_positive_integral_max(self):
        with pytest.raises(ValueError):
            PIDController(PIDGains(1, 0, 0), dt=0.1, integral_max=0.0, output_max=1.0)

    def test_rejects_non_positive_output_max(self):
        with pytest.raises(ValueError):
            PIDController(PIDGains(1, 0, 0), dt=0.1, integral_max=1.0, output_max=-1.0)
