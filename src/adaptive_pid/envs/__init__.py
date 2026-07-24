"""Environment subpackage: MuJoCo plant, domain randomization, reference
trajectories, and the composed Gymnasium environment.

Importing this subpackage registers ``AdaptivePIDPendulum-v0`` with
Gymnasium's global registry, so the environment can be created either
directly (``GymPIDGainEnv(env_config, ...)``, giving full control over
config injection) or via ``gymnasium.make("AdaptivePIDPendulum-v0")`` for
tooling that expects the standard registry-based interface.
"""

from __future__ import annotations

from gymnasium.envs.registration import register

from adaptive_pid.envs.domain_randomization import DomainRandomizer, RandomizationRanges
from adaptive_pid.envs.gym_env import GymPIDGainEnv
from adaptive_pid.envs.pendulum_plant import InvertedPendulumPlant
from adaptive_pid.envs.reference_trajectory import ReferenceTrajectory, sample_reference_trajectory

__all__ = [
    "DomainRandomizer",
    "GymPIDGainEnv",
    "InvertedPendulumPlant",
    "RandomizationRanges",
    "ReferenceTrajectory",
    "sample_reference_trajectory",
]

register(
    id="AdaptivePIDPendulum-v0",
    entry_point="adaptive_pid.envs.gym_env:GymPIDGainEnv",
)