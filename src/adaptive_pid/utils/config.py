"""Typed configuration loading.

Design rationale: raw ``dict`` configs (the common pattern in RL repos) push
key-typo errors to the point of *use*, often deep inside a training loop
after minutes of compute. Loading into validated dataclasses up front makes
every error surface at *startup*, with a clear message naming the bad key.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, TypeVar

import yaml

from adaptive_pid.utils.types import ControlLimits, PlantParams, RewardWeights

T = TypeVar("T")


class ConfigError(Exception):
    """Raised when a YAML config does not match the expected schema."""


def _build_dataclass(cls: type[T], data: dict[str, Any], *, path: str) -> T:
    # Use a type guard to verify cls is a dataclass type
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"Expected a dataclass type for section '{path}', got {cls}")
    # mypy doesn't narrow the type after is_dataclass, use ignore
    field_names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data.keys()) - field_names
    if unknown:
        raise ConfigError(
            f"Unknown key(s) {sorted(unknown)} in section '{path}' for {cls.__name__}. "
            f"Valid keys: {sorted(field_names)}"
        )
    missing = {
        f.name
        for f in dataclasses.fields(cls)  # type: ignore[arg-type]
        if f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
        and f.name not in data
    }
    if missing:
        raise ConfigError(f"Missing required key(s) {sorted(missing)} in section '{path}' for {cls.__name__}")
    try:
        return cls(**data)  # type: ignore[return-value]
    except TypeError as exc:
        raise ConfigError(f"Failed to build {cls.__name__} from section '{path}': {exc}") from exc


@dataclasses.dataclass
class EnvConfig:
    """Top-level environment configuration (physics + episode + randomization)."""

    dt_inner: float
    outer_loop_ratio: int
    episode_seconds: float
    theta_fail: float
    settle_epsilon: float
    initial_gains: dict[str, float]
    nominal_plant: PlantParams
    limits: ControlLimits
    reward_weights: RewardWeights
    randomization: dict[str, Any]


def load_env_config(path: str | Path) -> EnvConfig:
    """Load and validate an environment YAML config into an ``EnvConfig``."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ConfigError(f"Config file is empty: {path}")

    try:
        nominal_plant = _build_dataclass(PlantParams, raw["nominal_plant"], path="nominal_plant")
        limits = _build_dataclass(ControlLimits, raw["limits"], path="limits")
        reward_weights = _build_dataclass(RewardWeights, raw.get("reward_weights", {}), path="reward_weights")
    except KeyError as exc:
        raise ConfigError(f"Missing required top-level section {exc} in {path}") from exc

    known_top_level = {
        "dt_inner",
        "outer_loop_ratio",
        "episode_seconds",
        "theta_fail",
        "settle_epsilon",
        "initial_gains",
        "nominal_plant",
        "limits",
        "reward_weights",
        "randomization",
    }
    unknown_top = set(raw.keys()) - known_top_level
    if unknown_top:
        raise ConfigError(f"Unknown top-level key(s) {sorted(unknown_top)} in {path}")

    return EnvConfig(
        dt_inner=raw["dt_inner"],
        outer_loop_ratio=raw["outer_loop_ratio"],
        episode_seconds=raw["episode_seconds"],
        theta_fail=raw["theta_fail"],
        settle_epsilon=raw["settle_epsilon"],
        initial_gains=raw["initial_gains"],
        nominal_plant=nominal_plant,
        limits=limits,
        reward_weights=reward_weights,
        randomization=raw.get("randomization", {}),
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Generic YAML loader for training/eval configs that don't need a strict schema
    (e.g. SB3 hyperparameter dicts, which are inherently open-ended)."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ConfigError(f"Config file is empty: {path}")
    return data
