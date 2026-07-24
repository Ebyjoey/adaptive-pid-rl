from __future__ import annotations

import math

import pytest

from adaptive_pid.estimation.disturbance_observer import DisturbanceObserver, DisturbanceObserverConfig


def make_observer(smoothing_alpha=1.0) -> DisturbanceObserver:
    # smoothing_alpha=1.0 (no smoothing) makes single-step assertions exact,
    # which is what most of these tests want; smoothing behavior itself is
    # tested separately below.
    config = DisturbanceObserverConfig(
        nominal_mass=0.3,
        nominal_length=0.5,
        nominal_damping=0.05,
        gravity=9.81,
        smoothing_alpha=smoothing_alpha,
    )
    return DisturbanceObserver(config)


class TestFirstStep:
    def test_first_update_returns_zero_no_history(self):
        obs = make_observer()
        estimate = obs.update(theta=0.0, theta_dot=0.0, control_torque=0.0, dt=0.01)
        assert estimate == pytest.approx(0.0)


class TestNominalDynamicsProduceZeroResidual:
    def test_zero_residual_when_observed_matches_nominal_prediction(self):
        """If the plant behaves exactly as the nominal model predicts (no
        disturbance, no model mismatch), the residual estimate should be
        ~zero."""
        obs = make_observer()
        dt = 0.01
        m, l, b, g = 0.3, 0.5, 0.05, 9.81
        theta = 0.1
        theta_dot = 0.0
        control_torque = 1.0

        obs.update(theta=theta, theta_dot=theta_dot, control_torque=control_torque, dt=dt)

        inertia = m * l**2
        predicted_alpha = (control_torque - b * theta_dot - m * g * l * math.sin(theta)) / inertia
        next_theta_dot = theta_dot + predicted_alpha * dt

        estimate = obs.update(theta=theta, theta_dot=next_theta_dot, control_torque=control_torque, dt=dt)
        assert estimate == pytest.approx(0.0, abs=1e-6)


class TestDisturbanceDetection:
    def test_unexpected_acceleration_produces_nonzero_residual(self):
        """If the observed angular velocity changes far more than the
        nominal model predicts (as would happen under an external
        disturbance torque), the residual estimate should be clearly
        nonzero and have the same sign as the unexplained acceleration."""
        obs = make_observer()
        dt = 0.01
        obs.update(theta=0.0, theta_dot=0.0, control_torque=0.0, dt=dt)
        # A big, otherwise-unexplained jump in theta_dot (as if a disturbance
        # torque had acted): nominal model predicts ~0 acceleration here.
        estimate = obs.update(theta=0.0, theta_dot=5.0, control_torque=0.0, dt=dt)
        assert estimate > 0


class TestSmoothing:
    def test_full_smoothing_alpha_one_tracks_raw_residual_immediately(self):
        obs_no_smooth = make_observer(smoothing_alpha=1.0)
        obs_no_smooth.update(theta=0.0, theta_dot=0.0, control_torque=0.0, dt=0.01)
        raw_estimate = obs_no_smooth.update(theta=0.0, theta_dot=3.0, control_torque=0.0, dt=0.01)

        obs_smoothed = make_observer(smoothing_alpha=0.1)
        obs_smoothed.update(theta=0.0, theta_dot=0.0, control_torque=0.0, dt=0.01)
        smoothed_estimate = obs_smoothed.update(theta=0.0, theta_dot=3.0, control_torque=0.0, dt=0.01)

        # With heavy smoothing (alpha=0.1), the first post-jump estimate
        # should be much smaller in magnitude than the unsmoothed version.
        assert abs(smoothed_estimate) < abs(raw_estimate)


class TestReset:
    def test_reset_clears_history_and_estimate(self):
        obs = make_observer()
        obs.update(theta=0.0, theta_dot=0.0, control_torque=0.0, dt=0.01)
        obs.update(theta=0.0, theta_dot=5.0, control_torque=0.0, dt=0.01)
        assert obs.estimate != 0.0
        obs.reset()
        assert obs.estimate == pytest.approx(0.0)
        # after reset, next update should behave like a fresh first-step (zero)
        estimate = obs.update(theta=0.0, theta_dot=0.0, control_torque=0.0, dt=0.01)
        assert estimate == pytest.approx(0.0)
