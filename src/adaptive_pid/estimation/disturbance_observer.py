"""Lightweight disturbance estimator.

Provides the ``d̂`` (disturbance estimate) observation feature described in
docs/mdp_design.md Section 2. A full nonlinear observer (e.g. an extended
Kalman filter over the true pendulum dynamics) is more than this feature
needs to be useful to the policy -- what matters is giving the agent a
*leading indicator* of "something is pushing on the plant that the model
doesn't expect," not a precisely calibrated torque estimate.

We use a residual estimator: given the commanded control torque and a
nominal (un-randomized) dynamics model, predict the expected angular
acceleration, compare to the observed acceleration, and attribute the
discrepancy to an unmodeled disturbance/parameter-mismatch torque. This is
a standard "disturbance observer" structure (residual-based estimation),
kept intentionally simple and dependency-free so it is unit-testable
without MuJoCo.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DisturbanceObserverConfig:
    nominal_mass: float
    nominal_length: float
    nominal_damping: float
    gravity: float = 9.81
    smoothing_alpha: float = 0.3  # exponential smoothing factor on the raw residual, in [0, 1]


class DisturbanceObserver:
    """Residual-based disturbance torque estimator.

    Nominal dynamics (undamped-gravity + linear damping) predict:

        alpha_hat = (u - damping*theta_dot - m*g*l*sin(theta)) / (m*l^2)

    The residual between the *actually observed* angular acceleration and
    this nominal prediction is attributed to disturbance/model-mismatch
    torque, then exponentially smoothed to suppress sensor-noise-driven
    jitter (a raw one-step finite-difference acceleration estimate is quite
    noisy, so smoothing is essential for this to be a useful, stable
    observation feature rather than a noise amplifier).
    """

    def __init__(self, config: DisturbanceObserverConfig) -> None:
        self._config = config
        self._smoothed_estimate: float = 0.0
        self._prev_theta_dot: float | None = None

    def reset(self) -> None:
        self._smoothed_estimate = 0.0
        self._prev_theta_dot = None

    def update(self, theta: float, theta_dot: float, control_torque: float, dt: float) -> float:
        """Update and return the current smoothed disturbance torque estimate.

        Must be called exactly once per inner-loop physics step, in step
        order, since it maintains internal derivative-estimation state.
        """
        c = self._config

        if self._prev_theta_dot is None:
            observed_alpha = 0.0  # no history yet; assume no disturbance on the first sample
            velocity_for_prediction = theta_dot
        else:
            observed_alpha = (theta_dot - self._prev_theta_dot) / dt
            # Use the *pre-step* velocity for the damping/gravity torque
            # prediction, since those forces acted on the state at the start
            # of the interval (matching a forward-Euler physics step) --
            # using the post-step velocity here would itself look like a
            # "residual," even under perfectly nominal (undisturbed)
            # dynamics.
            velocity_for_prediction = self._prev_theta_dot
        self._prev_theta_dot = theta_dot

        import math

        inertia = c.nominal_mass * c.nominal_length**2
        gravity_torque = c.nominal_mass * c.gravity * c.nominal_length * math.sin(theta)
        damping_torque = c.nominal_damping * velocity_for_prediction

        predicted_alpha = (control_torque - damping_torque - gravity_torque) / inertia
        raw_residual_torque = (observed_alpha - predicted_alpha) * inertia

        self._smoothed_estimate = (
            c.smoothing_alpha * raw_residual_torque + (1.0 - c.smoothing_alpha) * self._smoothed_estimate
        )
        return self._smoothed_estimate

    @property
    def estimate(self) -> float:
        return self._smoothed_estimate
