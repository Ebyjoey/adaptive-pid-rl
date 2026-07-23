from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    compute_control_effort_rms,
    compute_energy,
    compute_overshoot_pct,
    compute_rise_time,
    compute_rmse,
    compute_rollout_metrics,
    compute_settling_time,
    compute_steady_state_error,
)


class TestRMSE:
    def test_zero_error_gives_zero_rmse(self):
        assert compute_rmse(np.zeros(10)) == pytest.approx(0.0)

    def test_constant_error_gives_that_value(self):
        assert compute_rmse(np.full(10, 2.0)) == pytest.approx(2.0)

    def test_empty_array_gives_zero(self):
        assert compute_rmse(np.array([])) == pytest.approx(0.0)


class TestControlEffortAndEnergy:
    def test_control_effort_rms_of_constant_signal(self):
        assert compute_control_effort_rms(np.full(5, 3.0)) == pytest.approx(3.0)

    def test_energy_matches_analytical_integral(self):
        u = np.full(100, 2.0)
        dt = 0.01
        # energy = sum(u^2 * dt) = 100 * 4 * 0.01 = 4.0
        assert compute_energy(u, dt) == pytest.approx(4.0)


class TestRiseTime:
    def test_linear_ramp_gives_known_rise_time(self):
        # theta ramps linearly from 0 to 1 over [0, 1]s at dt=0.01
        times = np.arange(0, 1.0, 0.01)
        theta = times.copy()  # theta(t) = t, reaches 1.0 at t=1.0
        rise_time = compute_rise_time(theta, times, reference_amplitude=1.0, step_time=0.0)
        # 10% at t=0.1, 90% at t=0.9 -> rise time ~ 0.8s
        assert rise_time == pytest.approx(0.8, abs=0.02)

    def test_returns_none_if_never_reaches_90_percent(self):
        times = np.arange(0, 1.0, 0.01)
        theta = np.full_like(times, 0.5)  # never gets close to 1.0
        rise_time = compute_rise_time(theta, times, reference_amplitude=1.0, step_time=0.0)
        assert rise_time is None

    def test_returns_none_for_zero_amplitude(self):
        times = np.arange(0, 1.0, 0.01)
        theta = times.copy()
        assert compute_rise_time(theta, times, reference_amplitude=0.0, step_time=0.0) is None


class TestSettlingTime:
    def test_immediately_within_band_settles_at_first_time(self):
        times = np.arange(0, 1.0, 0.1)
        errors = np.full_like(times, 0.01)
        settling = compute_settling_time(errors, times, settle_epsilon=0.05)
        assert settling == pytest.approx(0.0)

    def test_settles_after_leaving_and_reentering_band(self):
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        errors = np.array([0.5, 0.5, 0.02, 0.5, 0.02])  # leaves band at t=0.3, re-enters at t=0.4
        settling = compute_settling_time(errors, times, settle_epsilon=0.05)
        assert settling == pytest.approx(0.4)

    def test_never_settling_returns_none(self):
        times = np.array([0.0, 0.1, 0.2])
        errors = np.array([0.5, 0.02, 0.5])  # last sample is outside the band
        settling = compute_settling_time(errors, times, settle_epsilon=0.05)
        assert settling is None


class TestOvershoot:
    def test_no_overshoot_gives_zero(self):
        times = np.arange(0, 1.0, 0.1)
        theta = np.full_like(times, 0.9)  # approaches but never exceeds reference=1.0
        overshoot = compute_overshoot_pct(theta, reference_amplitude=1.0, step_time=0.0, times=times)
        assert overshoot == pytest.approx(0.0)

    def test_ten_percent_overshoot(self):
        times = np.arange(0, 1.0, 0.1)
        theta = np.full_like(times, 1.1)  # 10% past reference=1.0
        overshoot = compute_overshoot_pct(theta, reference_amplitude=1.0, step_time=0.0, times=times)
        assert overshoot == pytest.approx(10.0)

    def test_negative_reference_overshoot(self):
        times = np.arange(0, 1.0, 0.1)
        theta = np.full_like(times, -1.2)  # 20% past reference=-1.0 in the negative direction
        overshoot = compute_overshoot_pct(theta, reference_amplitude=-1.0, step_time=0.0, times=times)
        assert overshoot == pytest.approx(20.0)


class TestSteadyStateError:
    def test_uses_final_fraction_of_trajectory(self):
        errors = np.concatenate([np.full(90, 1.0), np.full(10, 0.1)])
        sse = compute_steady_state_error(errors, fraction=0.1)
        assert sse == pytest.approx(0.1)

    def test_empty_gives_zero(self):
        assert compute_steady_state_error(np.array([])) == pytest.approx(0.0)


class TestComputeRolloutMetrics:
    def test_produces_all_fields_without_error(self):
        times = np.arange(0, 2.0, 0.01)
        theta = np.minimum(times, 1.0)  # ramps to 1.0 then holds
        errors = 1.0 - theta
        control = np.ones_like(times) * 0.5

        metrics = compute_rollout_metrics(
            times=times,
            errors=errors,
            theta=theta,
            control_signal=control,
            dt=0.01,
            settle_epsilon=0.05,
            reference_amplitude=1.0,
            step_time=0.0,
            fell=False,
        )
        assert metrics.rmse >= 0
        assert metrics.fell is False
        assert metrics.energy > 0
