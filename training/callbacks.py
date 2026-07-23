"""Custom SB3 training callbacks.

SB3's built-in logging covers episode reward/length, but this project's
evaluation criteria (docs/mdp_design.md) explicitly require visibility into
*why* the reward is what it is (per-term breakdown) and *how* gains evolve
during training -- both of which need a custom callback since they aren't
part of SB3's default Monitor/logger output.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GainAndRewardTermLoggingCallback(BaseCallback):
    """Logs mean PID gains and mean per-term reward contributions to
    TensorBoard every ``log_freq`` training steps, aggregated across all
    parallel envs since the last log point.

    This is what lets ``docs/`` plots and the eventual results dashboard
    show "how did Kp/Ki/Kd evolve over training" and "which reward term is
    dominating the policy's behavior at each training stage" -- both
    explicitly required by the project's evaluation criteria.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._log_freq = log_freq
        self._gain_accum: dict[str, list[float]] = defaultdict(list)
        self._term_accum: dict[str, list[float]] = defaultdict(list)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            gains = info.get("gains")
            if gains is not None:
                self._gain_accum["kp"].append(gains.kp)
                self._gain_accum["ki"].append(gains.ki)
                self._gain_accum["kd"].append(gains.kd)
            reward_terms = info.get("reward_terms")
            if reward_terms is not None:
                for term_name, value in reward_terms.items():
                    self._term_accum[term_name].append(value)

        if self.n_calls % self._log_freq == 0:
            for gain_name, values in self._gain_accum.items():
                if values:
                    self.logger.record(f"gains/{gain_name}_mean", float(np.mean(values)))
                    self.logger.record(f"gains/{gain_name}_std", float(np.std(values)))
            for term_name, values in self._term_accum.items():
                if values:
                    self.logger.record(f"reward_terms/{term_name}_mean", float(np.mean(values)))
            self._gain_accum.clear()
            self._term_accum.clear()

        return True
