"""Reference trajectory generation.

Per docs/mdp_design.md Section 5, the reference is randomly sampled per
episode from several shapes so the policy does not overfit to one
trajectory type. Kept in ``envs`` (not ``control``) since it is specific to
training-environment episode structure, not a general control primitive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from adaptive_pid.utils.types import ReferenceType


@dataclass
class ReferenceTrajectory:
    """Callable-like object producing ``(reference, reference_rate,
    is_step_change_recent)`` for a given episode time."""

    kind: ReferenceType
    amplitude: float
    period: float  # seconds, used by doublet/sine-sweep/random-walk
    step_change_times: tuple[float, ...]  # times (s) at which a step-like discontinuity occurs

    def value_and_rate(self, t: float, dt: float) -> tuple[float, float]:
        """Return ``(reference(t), d(reference)/dt approximated at t)``."""
        r_now = self._value(t)
        r_prev = self._value(max(0.0, t - dt))
        rate = (r_now - r_prev) / dt if dt > 0 else 0.0
        return r_now, rate

    def time_since_last_step_change(self, t: float) -> float:
        past_changes = [ct for ct in self.step_change_times if ct <= t]
        if not past_changes:
            return t
        return t - max(past_changes)

    def is_step_reference(self) -> bool:
        """Whether overshoot penalization is meaningful for this trajectory
        shape (see reward_function.overshoot_term)."""
        return self.kind in (ReferenceType.STEP, ReferenceType.DOUBLET)

    def _value(self, t: float) -> float:
        if self.kind == ReferenceType.STEP:
            return self.amplitude if t >= self.step_change_times[0] else 0.0
        if self.kind == ReferenceType.DOUBLET:
            # Step up, then step back down halfway through.
            if len(self.step_change_times) < 2:
                raise ValueError("Doublet reference requires two step_change_times")
            t_up, t_down = self.step_change_times[0], self.step_change_times[1]
            if t < t_up:
                return 0.0
            if t < t_down:
                return self.amplitude
            return 0.0
        if self.kind == ReferenceType.SINE_SWEEP:
            freq = 1.0 / self.period
            return self.amplitude * math.sin(2 * math.pi * freq * t)
        if self.kind == ReferenceType.RANDOM_WALK:
            # Deterministic given (t, period, amplitude): a smoothly varying
            # signal built from a small fixed sum of sinusoids at
            # incommensurate frequencies, which behaves like a bounded random
            # walk without requiring per-step RNG state to be threaded
            # through this otherwise-stateless value function.
            return self.amplitude * (
                0.6 * math.sin(2 * math.pi * t / self.period)
                + 0.3 * math.sin(2 * math.pi * t / (self.period * 0.37) + 1.3)
                + 0.1 * math.sin(2 * math.pi * t / (self.period * 0.13) + 0.7)
            )
        raise ValueError(f"Unhandled reference kind: {self.kind}")


def sample_reference_trajectory(
    rng: np.random.Generator,
    episode_seconds: float,
    max_amplitude: float,
) -> ReferenceTrajectory:
    """Randomly sample a reference trajectory shape and parameters for one episode."""
    kind_options = [
        ReferenceType.STEP,
        ReferenceType.DOUBLET,
        ReferenceType.SINE_SWEEP,
        ReferenceType.RANDOM_WALK,
    ]
    # Select by integer index into a plain Python list, rather than
    # rng.choice(list_of_enum_members) directly: numpy silently coerces a
    # list of str-Enum members into a fixed-width string ndarray, which
    # truncates the enum's repr and corrupts the returned value.
    kind = kind_options[int(rng.integers(0, len(kind_options)))]
    amplitude = float(rng.uniform(0.3 * max_amplitude, max_amplitude))

    if kind == ReferenceType.STEP:
        step_time = float(rng.uniform(0.05, 0.2) * episode_seconds)
        return ReferenceTrajectory(kind, amplitude, period=episode_seconds, step_change_times=(step_time,))

    if kind == ReferenceType.DOUBLET:
        t_up = float(rng.uniform(0.05, 0.15) * episode_seconds)
        t_down = float(rng.uniform(0.5, 0.7) * episode_seconds)
        return ReferenceTrajectory(kind, amplitude, period=episode_seconds, step_change_times=(t_up, t_down))

    if kind == ReferenceType.SINE_SWEEP:
        period = float(rng.uniform(0.2, 0.5) * episode_seconds)
        return ReferenceTrajectory(kind, amplitude, period=period, step_change_times=())

    # RANDOM_WALK
    period = float(rng.uniform(0.3, 0.6) * episode_seconds)
    return ReferenceTrajectory(kind, amplitude, period=period, step_change_times=())
