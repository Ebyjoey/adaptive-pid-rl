"""Strongly-typed dataclasses shared across the adaptive_pid package.

Keeping these in one module (rather than letting every subpackage define its
own ad-hoc dict/tuple shapes) is what lets ``control``, ``rewards`` and
``envs`` interoperate without importing each other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ReferenceType(str, Enum):
    """Supported reference-trajectory shapes for training/evaluation."""

    STEP = "step"
    DOUBLET = "doublet"
    SINE_SWEEP = "sine_sweep"
    RANDOM_WALK = "random_walk"


@dataclass(frozen=True)
class PIDGains:
    """Immutable snapshot of a (Kp, Ki, Kd) triple.

    Frozen so that passing a ``PIDGains`` instance around (e.g. into a reward
    function, or across a ROS2 message boundary) can never result in a
    module silently mutating another module's state.
    """

    kp: float
    ki: float
    kd: float

    def as_array(self) -> tuple[float, float, float]:
        return (self.kp, self.ki, self.kd)

    def __sub__(self, other: PIDGains) -> PIDGains:
        return PIDGains(self.kp - other.kp, self.ki - other.ki, self.kd - other.kd)

    def norm(self) -> float:
        return (self.kp**2 + self.ki**2 + self.kd**2) ** 0.5


@dataclass
class PlantState:
    """Physical state of the controlled plant.

    Deliberately separate from the RL *observation* (see
    ``envs.gym_env.build_observation``) -- the plant does not know it is
    being controlled by an RL agent, and the RL observation includes
    controller-internal quantities (integral error, gains, etc.) that are
    not physical plant state.
    """

    theta: float          # rad, 0 = upright equilibrium
    theta_dot: float       # rad/s
    time: float = 0.0      # s, sim time since episode start


@dataclass
class PlantParams:
    """Physical parameters of the inverted pendulum plant.

    All fields are randomized within configured ranges by
    ``envs.domain_randomization.DomainRandomizer`` at each episode reset,
    and additionally perturbed within-episode for the fields marked
    "time-varying" in the docstring of that class.
    """

    mass: float             # kg, pole mass (payload variation)
    length: float           # m, pole length (center of mass distance)
    damping: float          # N*m*s/rad, pivot viscous friction
    gravity: float = 9.81   # m/s^2
    actuator_gain: float = 1.0  # actuator degradation multiplier on torque, 1.0 = nominal
    inertia_extra: float = 0.0  # kg*m^2, added rotor/inertia term


@dataclass
class ControlLimits:
    """Safety rails enforced outside the RL policy's control."""

    kp_min: float
    kp_max: float
    ki_min: float
    ki_max: float
    kd_min: float
    kd_max: float
    kp_rate_max: float   # max |dKp| per outer-loop step
    ki_rate_max: float
    kd_rate_max: float
    u_max: float          # N*m, actuator torque saturation
    integral_max: float   # anti-windup clamp on integral error


@dataclass
class RewardWeights:
    """Weights for each term of the shaped reward. See docs/mdp_design.md."""

    w_tracking: float = 1.0
    w_overshoot: float = 0.5
    w_settling: float = 0.3
    w_oscillation: float = 0.2
    w_energy: float = 0.05
    w_gain_smoothness: float = 0.1
    w_saturation: float = 0.2
    fall_penalty: float = 50.0


@dataclass
class EpisodeStats:
    """Accumulated per-episode metrics used by evaluation/benchmarking."""

    rmse: float = 0.0
    rise_time: float | None = None
    settling_time: float | None = None
    overshoot_pct: float = 0.0
    steady_state_error: float = 0.0
    control_effort: float = 0.0
    energy: float = 0.0
    total_reward: float = 0.0
    fell: bool = False
    gain_history: list[PIDGains] = field(default_factory=list)
