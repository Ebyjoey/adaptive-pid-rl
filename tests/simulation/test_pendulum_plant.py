"""Simulation-level tests requiring an actual MuJoCo instantiation.

Marked with @pytest.mark.simulation so CI can run the fast unit suite
separately from the (slightly) heavier simulation suite if needed
(``pytest -m "not simulation"`` for the fast path).
"""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_pid.envs.pendulum_plant import InvertedPendulumPlant
from adaptive_pid.utils.types import PlantParams

pytestmark = pytest.mark.simulation


def make_plant(dt=0.01) -> InvertedPendulumPlant:
    plant = InvertedPendulumPlant(dt=dt)
    plant.apply_params(PlantParams(mass=0.3, length=0.5, damping=0.05))
    return plant


class TestResetAndBasicStep:
    def test_reset_sets_initial_state(self):
        plant = make_plant()
        state = plant.reset(initial_theta=0.3, initial_theta_dot=-0.1)
        assert state.theta == pytest.approx(0.3)
        assert state.theta_dot == pytest.approx(-0.1)
        assert state.time == pytest.approx(0.0)

    def test_upright_equilibrium_is_stationary_with_zero_perturbation(self):
        """theta=0, theta_dot=0, zero control/disturbance is an (unstable)
        equilibrium: it should not spontaneously move."""
        plant = make_plant()
        plant.reset(initial_theta=0.0, initial_theta_dot=0.0)
        state = plant.step(control_torque=0.0)
        assert state.theta == pytest.approx(0.0, abs=1e-9)
        assert state.theta_dot == pytest.approx(0.0, abs=1e-9)

    def test_time_advances_by_dt_each_step(self):
        dt = 0.02
        plant = make_plant(dt=dt)
        plant.reset()
        state = plant.step(0.0)
        assert state.time == pytest.approx(dt)
        state = plant.step(0.0)
        assert state.time == pytest.approx(2 * dt)


class TestPhysicalBehavior:
    def test_perturbed_pendulum_falls_away_from_upright_with_no_control(self):
        plant = make_plant()
        plant.reset(initial_theta=0.2, initial_theta_dot=0.0)
        for _ in range(50):
            state = plant.step(control_torque=0.0)
        assert abs(state.theta) > 0.2  # gravity has pulled it further from upright

    def test_control_torque_influences_angular_acceleration(self):
        """Applying opposing torque to a perturbed pendulum should reduce (not
        necessarily eliminate, in one step) the rate at which it falls,
        relative to applying no torque, at otherwise identical conditions."""
        plant_uncontrolled = make_plant()
        plant_uncontrolled.reset(initial_theta=0.3, initial_theta_dot=0.0)
        for _ in range(20):
            state_uncontrolled = plant_uncontrolled.step(control_torque=0.0)

        plant_controlled = make_plant()
        plant_controlled.reset(initial_theta=0.3, initial_theta_dot=0.0)
        for _ in range(20):
            # restoring torque proportional to angle, opposing the fall
            state_controlled = plant_controlled.step(control_torque=-5.0 * plant_controlled.get_state().theta)

        assert abs(state_controlled.theta) < abs(state_uncontrolled.theta)

    def test_disturbance_torque_perturbs_stationary_pendulum(self):
        plant = make_plant()
        plant.reset(initial_theta=0.0, initial_theta_dot=0.0)
        state = plant.step(control_torque=0.0, disturbance_torque=5.0)
        assert state.theta_dot != 0.0


class TestParameterApplication:
    def test_heavier_mass_has_smaller_angular_response_to_same_external_torque(self):
        """Note: under gravity alone, a pendulum's angular acceleration is
        mass-independent (gravitational torque and inertia both scale with
        mass and cancel: alpha = g*sin(theta)/l) -- this is standard
        pendulum physics, not a bug. To actually isolate mass's effect on
        inertia, apply a pure external torque from theta=0 (zero gravity
        torque there) with zero damping: alpha = torque / inertia, so a
        heavier pendulum must show a smaller angular velocity response.
        """
        light = InvertedPendulumPlant(dt=0.01)
        light.apply_params(PlantParams(mass=0.1, length=0.5, damping=0.0))
        light.reset(initial_theta=0.0, initial_theta_dot=0.0)
        light_state = light.step(control_torque=1.0)

        heavy = InvertedPendulumPlant(dt=0.01)
        heavy.apply_params(PlantParams(mass=2.0, length=0.5, damping=0.0))
        heavy.reset(initial_theta=0.0, initial_theta_dot=0.0)
        heavy_state = heavy.step(control_torque=1.0)

        assert abs(heavy_state.theta_dot) < abs(light_state.theta_dot)

    def test_actuator_gain_scales_effective_torque(self):
        """Setting actuator_gain=0.5 should produce exactly half the
        angular-velocity response of actuator_gain=1.0 for the same
        commanded torque, from rest, over a single step (before nonlinear
        gravity/dynamics effects accumulate)."""
        full_gain = InvertedPendulumPlant(dt=0.01)
        full_gain.apply_params(PlantParams(mass=0.3, length=0.5, damping=0.0, actuator_gain=1.0))
        full_gain.reset(initial_theta=0.0, initial_theta_dot=0.0)
        state_full = full_gain.step(control_torque=2.0)

        half_gain = InvertedPendulumPlant(dt=0.01)
        half_gain.apply_params(PlantParams(mass=0.3, length=0.5, damping=0.0, actuator_gain=0.5))
        half_gain.reset(initial_theta=0.0, initial_theta_dot=0.0)
        state_half = half_gain.step(control_torque=2.0)

        assert state_half.theta_dot == pytest.approx(state_full.theta_dot * 0.5, rel=1e-3)


class TestNoisyObservation:
    def test_noisy_observation_differs_from_true_state_but_is_close(self):
        plant = make_plant()
        plant.reset(initial_theta=0.1, initial_theta_dot=0.0)
        rng = np.random.default_rng(42)
        noisy = plant.get_noisy_observation(theta_noise_std=0.01, theta_dot_noise_std=0.01, rng=rng)
        true_state = plant.get_state()
        assert noisy.theta != true_state.theta
        assert abs(noisy.theta - true_state.theta) < 0.1  # noise shouldn't be wildly larger than std

    def test_zero_noise_std_returns_true_state(self):
        plant = make_plant()
        plant.reset(initial_theta=0.1, initial_theta_dot=0.05)
        rng = np.random.default_rng(0)
        noisy = plant.get_noisy_observation(theta_noise_std=0.0, theta_dot_noise_std=0.0, rng=rng)
        true_state = plant.get_state()
        assert noisy.theta == pytest.approx(true_state.theta)
        assert noisy.theta_dot == pytest.approx(true_state.theta_dot)


class TestConstructorValidation:
    def test_missing_model_path_raises(self):
        with pytest.raises(FileNotFoundError):
            InvertedPendulumPlant(model_path="does/not/exist.xml")
