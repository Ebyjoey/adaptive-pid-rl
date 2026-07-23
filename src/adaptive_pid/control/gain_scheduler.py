"""Gain scheduling: the single point where RL actions become safe PID gains.

Per docs/architecture.md Section 6, this is deliberately the *only* place
gain clamping/rate-limiting happens, so the Gymnasium env, evaluation
scripts, and ROS2 node cannot each implement (and potentially
mis-implement) their own version of the safety rails.
"""

from __future__ import annotations

from adaptive_pid.utils.types import ControlLimits, PIDGains


class GainScheduler:
    """Applies a bounded, rate-limited update to PID gains given a raw
    (typically ``tanh``-squashed, in ``[-1, 1]``) action from an RL policy.
    """

    def __init__(self, limits: ControlLimits) -> None:
        self._limits = limits

    def initial_gains(self, kp: float, ki: float, kd: float) -> PIDGains:
        """Clamp a requested initial gain triple (e.g. from Ziegler-Nichols
        or manual tuning) into the configured safety bounds."""
        return PIDGains(
            kp=_clip(kp, self._limits.kp_min, self._limits.kp_max),
            ki=_clip(ki, self._limits.ki_min, self._limits.ki_max),
            kd=_clip(kd, self._limits.kd_min, self._limits.kd_max),
        )

    def apply_action(self, current: PIDGains, action: tuple[float, float, float], dt_outer: float) -> PIDGains:
        """Integrate a raw ``[-1, 1]``-scaled action onto the current gains.

        ``action`` is expected pre-squashed (e.g. via ``tanh`` in the policy
        network / SB3's default continuous action handling); this function
        clips defensively regardless, since a misbehaving policy or an
        un-squashed exploration action must never be able to command an
        out-of-range or infinite gain.
        """
        a_kp, a_ki, a_kd = action
        a_kp = _clip(a_kp, -1.0, 1.0)
        a_ki = _clip(a_ki, -1.0, 1.0)
        a_kd = _clip(a_kd, -1.0, 1.0)

        new_kp = current.kp + a_kp * self._limits.kp_rate_max * dt_outer
        new_ki = current.ki + a_ki * self._limits.ki_rate_max * dt_outer
        new_kd = current.kd + a_kd * self._limits.kd_rate_max * dt_outer

        return PIDGains(
            kp=_clip(new_kp, self._limits.kp_min, self._limits.kp_max),
            ki=_clip(new_ki, self._limits.ki_min, self._limits.ki_max),
            kd=_clip(new_kd, self._limits.kd_min, self._limits.kd_max),
        )

    @property
    def limits(self) -> ControlLimits:
        return self._limits


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
