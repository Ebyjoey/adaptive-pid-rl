"""Domain randomization for sim-to-real robustness.

Two distinct timescales of variation are modeled, matching the task's
required "dynamics change due to payload variation, friction, disturbances,
actuator degradation, sensor noise, and battery voltage changes":

1. **Episode-level** (sampled once at ``reset()``): payload mass, pendulum
   length/inertia, pivot friction/damping. These represent a *fixed physical
   configuration* for the duration of one episode -- e.g. "today the robot
   is carrying a 300g payload" -- matching how domain randomization is
   conventionally used for sim-to-real transfer.
2. **Step-level / time-varying** (resampled during the episode):
   disturbance torques, sensor noise, actuator degradation drift, and
   battery-voltage droop. These represent phenomena that genuinely change
   *while the system is running* -- a disturbance gust, a noisy encoder
   reading, a battery voltage sagging under load -- which is exactly the
   scenario the RL agent must learn to detect (via the disturbance-estimate
   observation feature) and adapt its gains to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from adaptive_pid.utils.types import PlantParams


@dataclass
class RandomizationRanges:
    """Uniform sampling ranges for every randomized quantity.

    Loaded from ``configs/env/pendulum.yaml``'s ``randomization`` section
    (a loosely-typed dict, since these ranges change frequently during
    experimentation and don't warrant the strict schema used for
    ``PlantParams``/``ControlLimits``).
    """

    mass_range: tuple[float, float] = (0.15, 0.6)
    length_range: tuple[float, float] = (0.35, 0.65)
    damping_range: tuple[float, float] = (0.01, 0.15)
    actuator_gain_range: tuple[float, float] = (0.7, 1.0)
    inertia_extra_range: tuple[float, float] = (0.0, 0.02)

    # Step-level (time-varying) ranges
    disturbance_prob_per_step: float = 0.02  # probability of a disturbance impulse each inner step
    disturbance_torque_range: tuple[float, float] = (-3.0, 3.0)
    theta_noise_std_range: tuple[float, float] = (0.0, 0.01)
    theta_dot_noise_std_range: tuple[float, float] = (0.0, 0.02)
    battery_voltage_droop_range: tuple[float, float] = (
        0.85,
        1.0,
    )  # multiplies actuator_gain further, per-episode drift

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RandomizationRanges:
        """Build from a raw YAML dict, tolerating missing keys (falls back
        to dataclass defaults) but rejecting unknown keys (typo protection).
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data.keys()) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown randomization key(s): {sorted(unknown)}. Valid keys: {sorted(valid_fields)}"
            )
        converted: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, list):
                converted[k] = tuple(v)
            elif k.endswith("_range") and isinstance(v, tuple) and len(v) == 2:
                converted[k] = (float(v[0]), float(v[1]))
            else:
                converted[k] = v
        return cls(**converted)


@dataclass
class DisturbanceEvent:
    """A single sampled disturbance impulse, for logging/plotting."""

    time: float
    torque: float


class DomainRandomizer:
    """Samples plant parameters and runtime disturbances/noise according to
    a ``RandomizationRanges`` specification.
    """

    def __init__(self, ranges: RandomizationRanges, seed: int | None = None) -> None:
        self._ranges = ranges
        self._rng = np.random.default_rng(seed)
        self._battery_droop_this_episode: float = 1.0

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def sample_episode_params(self) -> PlantParams:
        """Sample a fixed-for-the-episode plant configuration."""
        r = self._ranges
        self._battery_droop_this_episode = self._rng.uniform(*r.battery_voltage_droop_range)
        base_actuator_gain = self._rng.uniform(*r.actuator_gain_range)
        return PlantParams(
            mass=float(self._rng.uniform(*r.mass_range)),
            length=float(self._rng.uniform(*r.length_range)),
            damping=float(self._rng.uniform(*r.damping_range)),
            actuator_gain=float(base_actuator_gain * self._battery_droop_this_episode),
            inertia_extra=float(self._rng.uniform(*r.inertia_extra_range)),
        )

    def maybe_sample_disturbance(self, current_time: float) -> DisturbanceEvent | None:
        """Stochastically sample a disturbance torque impulse for the
        current inner-loop step. Called every physics step, not every
        outer-loop (gain-scheduling) step, since disturbances are a
        plant-level, not controller-level, phenomenon.
        """
        if self._rng.uniform() < self._ranges.disturbance_prob_per_step:
            torque = float(self._rng.uniform(*self._ranges.disturbance_torque_range))
            return DisturbanceEvent(time=current_time, torque=torque)
        return None

    def sample_sensor_noise_std(self) -> tuple[float, float]:
        """Sample (theta_noise_std, theta_dot_noise_std) for the current
        episode. Sampled once per episode (rather than per-step) since a
        given sensor's noise floor is a property of that sensor/episode
        instance, not something that changes every timestep."""
        r = self._ranges
        return (
            float(self._rng.uniform(*r.theta_noise_std_range)),
            float(self._rng.uniform(*r.theta_dot_noise_std_range)),
        )

    @property
    def rng(self) -> np.random.Generator:
        """Expose the underlying RNG so the owning environment can use the
        same, seed-controlled stream for anything else that needs
        randomness (e.g. reference-trajectory sampling), keeping a whole
        episode reproducible from one seed."""
        return self._rng
