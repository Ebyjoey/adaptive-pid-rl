"""Discrete-time PID controller core.

This module has zero dependencies on MuJoCo, Gymnasium, or ROS2 by design:
it is the one piece of logic that must behave *identically* whether it is
driven from the training environment, the evaluation scripts, or the ROS2
``pid_controller_node``. Any divergence between those three would silently
invalidate sim-to-real comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_pid.utils.types import PIDGains


@dataclass
class PIDState:
    """Internal, mutable controller state (kept separate from the immutable
    ``PIDGains`` so that resetting a controller's error history never
    accidentally resets its tuned gains, and vice versa)."""

    integral: float = 0.0
    prev_error: float | None = None


class PIDController:
    """A standard discrete PID controller with anti-windup clamping and
    output saturation, in "position form":

        u_t = Kp*e_t + Ki*integral_t + Kd*derivative_t

    Anti-windup is implemented via clamped integration (the integral term
    stops accumulating further once it would exceed ``integral_max``) rather
    than back-calculation, since clamped integration is simpler to reason
    about and verify in unit tests, and is standard practice for
    industrial PID implementations where the actuator saturation limit is
    known and fixed.
    """

    def __init__(self, gains: PIDGains, dt: float, integral_max: float, output_max: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if integral_max <= 0:
            raise ValueError(f"integral_max must be positive, got {integral_max}")
        if output_max <= 0:
            raise ValueError(f"output_max must be positive, got {output_max}")

        self._gains = gains
        self._dt = dt
        self._integral_max = integral_max
        self._output_max = output_max
        self._state = PIDState()

    @property
    def gains(self) -> PIDGains:
        return self._gains

    def set_gains(self, gains: PIDGains) -> None:
        """Update the active gains without resetting error history.

        This is the entry point the RL agent's gain updates flow through:
        gain scheduling should not zero out accumulated integral error,
        since that would itself create a control transient.
        """
        self._gains = gains

    def reset(self) -> None:
        """Clear error history (integral + derivative memory). Call this at
        the start of a new episode/trajectory, not on every gain update."""
        self._state = PIDState()

    def step(self, error: float) -> tuple[float, float, float]:
        """Compute one control update.

        Returns
        -------
        (u, integral_error, derivative_error)
            ``u`` is the saturated control output; ``integral_error`` and
            ``derivative_error`` are exposed because the RL observation
            vector needs them directly (see ``envs.gym_env``).
        """
        # Clamped-integration anti-windup: integrate, then clamp the result to
        # +/- integral_max. Clamping the *result* (rather than refusing to
        # integrate at all once a step would overshoot the limit) ensures the
        # integral term correctly saturates at the boundary instead of
        # getting stuck below it when a single large-error step would
        # otherwise jump past the limit in one shot.
        candidate_integral = self._state.integral + error * self._dt
        self._state.integral = max(-self._integral_max, min(self._integral_max, candidate_integral))

        if self._state.prev_error is None:
            derivative = 0.0  # no history yet; avoid a derivative spike on the first step
        else:
            derivative = (error - self._state.prev_error) / self._dt

        self._state.prev_error = error

        u_unsaturated = (
            self._gains.kp * error
            + self._gains.ki * self._state.integral
            + self._gains.kd * derivative
        )
        u = max(-self._output_max, min(self._output_max, u_unsaturated))

        return u, self._state.integral, derivative
