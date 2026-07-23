"""Controller abstraction for evaluation.

Evaluation needs to run the *same* rollout loop against five different
control strategies (fixed PID, manual tuning, Ziegler-Nichols, PPO, SAC).
Rather than special-casing each in ``evaluate.py``, every strategy is
wrapped behind one ``GainPolicy`` interface: "given the current
observation, return a delta-gain action in [-1, 1]^3." This is a small
but important SOLID (dependency-inversion) decision -- ``evaluate.py``
depends only on this abstraction, never on SB3 or a specific baseline's
internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GainPolicy(ABC):
    """Common interface for anything that can be evaluated in `evaluate.py`."""

    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return a delta-gain action in ``[-1, 1]^3`` given the current
        12-dim observation (see docs/mdp_design.md Section 2 for layout)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class FixedGainPolicy(GainPolicy):
    """A policy that never adjusts gains: always emits the zero action.

    Used for the fixed-PID, manual-tuning, and Ziegler-Nichols baselines --
    the only difference between those three baselines is which gains the
    environment is *seeded* with at reset (see evaluation/benchmark.py),
    not any behavioral difference in this policy class.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    @property
    def name(self) -> str:
        return self._name


class SB3GainPolicy(GainPolicy):
    """Wraps a trained Stable-Baselines3 model (PPO or SAC) behind the
    ``GainPolicy`` interface, handling the observation normalization the
    model expects.
    """

    def __init__(self, model, name: str, obs_rms=None) -> None:
        self._model = model
        self._name = name
        self._obs_rms = obs_rms  # optional VecNormalize running stats, for consistent obs scaling

    def act(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs = observation
        if self._obs_rms is not None:
            obs = (observation - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + 1e-8)
            obs = np.clip(obs, -10.0, 10.0)
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    @property
    def name(self) -> str:
        return self._name
