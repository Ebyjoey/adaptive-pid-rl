"""Gymnasium environment for adaptive PID gain scheduling on an inverted pendulum.

This is the integration point: it composes ``InvertedPendulumPlant`` (MuJoCo
physics), ``PIDController`` + ``GainScheduler`` (control), ``DomainRandomizer``
(episode/step-level randomization), ``DisturbanceObserver`` (estimation), and
``compute_reward`` (rewards) -- all of which are independently unit-tested in
isolation. This module's own responsibility is purely the *wiring* and the
Gymnasium-required ``reset``/``step`` contract, per the one-way dependency
rule in docs/architecture.md.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from adaptive_pid.control.gain_scheduler import GainScheduler
from adaptive_pid.control.pid import PIDController
from adaptive_pid.envs.domain_randomization import DomainRandomizer, RandomizationRanges
from adaptive_pid.envs.pendulum_plant import InvertedPendulumPlant
from adaptive_pid.envs.reference_trajectory import ReferenceTrajectory, sample_reference_trajectory
from adaptive_pid.estimation.disturbance_observer import DisturbanceObserver, DisturbanceObserverConfig
from adaptive_pid.rewards.reward_function import compute_reward
from adaptive_pid.utils.config import EnvConfig
from adaptive_pid.utils.types import PIDGains

# Observation vector layout (see docs/mdp_design.md Section 2), fixed order:
OBS_DIM = 12
ACTION_DIM = 3


class GymPIDGainEnv(gym.Env):
    """A Gymnasium environment where the action is a delta-gain update for an
    inner-loop PID controller regulating a randomized inverted pendulum.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_config: EnvConfig,
        randomization_ranges: RandomizationRanges | None = None,
        model_path: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self._config = env_config
        self._ranges = randomization_ranges or RandomizationRanges.from_dict(env_config.randomization)

        self._plant = InvertedPendulumPlant(model_path=model_path, dt=env_config.dt_inner)
        self._randomizer = DomainRandomizer(self._ranges, seed=seed)
        self._gain_scheduler = GainScheduler(env_config.limits)

        initial_gains = self._gain_scheduler.initial_gains(**env_config.initial_gains)
        self._pid = PIDController(
            gains=initial_gains,
            dt=env_config.dt_inner,
            integral_max=env_config.limits.integral_max,
            output_max=env_config.limits.u_max,
        )
        self._initial_gains = initial_gains

        self._observer = DisturbanceObserver(
            DisturbanceObserverConfig(
                nominal_mass=env_config.nominal_plant.mass,
                nominal_length=env_config.nominal_plant.length,
                nominal_damping=env_config.nominal_plant.damping,
                gravity=env_config.nominal_plant.gravity,
            )
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self._max_inner_steps = int(round(env_config.episode_seconds / env_config.dt_inner))
        self._theta_noise_std = 0.0
        self._theta_dot_noise_std = 0.0
        self._inner_step_count = 0
        self._prev_derivative_error: float | None = None
        self._prev_u = 0.0
        self._reference: ReferenceTrajectory | None = None
        self._episode_reward_terms_sum: dict[str, float] = {}
        self._last_info: dict[str, Any] = {}

    # -- Gymnasium API -----------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._randomizer.seed(seed)

        plant_params = self._randomizer.sample_episode_params()
        self._plant.apply_params(plant_params)
        self._theta_noise_std, self._theta_dot_noise_std = self._randomizer.sample_sensor_noise_std()

        self._plant.reset(initial_theta=0.0, initial_theta_dot=0.0)
        self._pid.set_gains(self._initial_gains)
        self._pid.reset()
        self._observer.reset()

        self._reference = sample_reference_trajectory(
            self._randomizer.rng, self._config.episode_seconds, max_amplitude=0.5
        )
        self._inner_step_count = 0
        self._prev_derivative_error = None
        self._prev_u = 0.0
        self._episode_reward_terms_sum = {}

        obs = self._build_observation(gains=self._initial_gains, prev_u=0.0)
        info = {"plant_params": plant_params}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._reference is not None, "call reset() before step()"

        dt_outer = self._config.dt_inner * self._config.outer_loop_ratio
        previous_gains = self._pid.gains
        new_gains = self._gain_scheduler.apply_action(previous_gains, tuple(float(a) for a in action), dt_outer)
        self._pid.set_gains(new_gains)

        total_reward = 0.0
        terms_accum: dict[str, float] = {}
        terminated = False
        fell = False

        for _ in range(self._config.outer_loop_ratio):
            t = self._plant.get_state().time
            reference, _reference_rate = self._reference.value_and_rate(t, self._config.dt_inner)

            noisy_state = self._plant.get_noisy_observation(
                self._theta_noise_std, self._theta_dot_noise_std, self._randomizer.rng
            )
            error = reference - noisy_state.theta
            u, _integral_error, derivative_error = self._pid.step(error)

            disturbance_event = self._randomizer.maybe_sample_disturbance(t)
            disturbance_torque = disturbance_event.torque if disturbance_event is not None else 0.0

            new_state = self._plant.step(control_torque=u, disturbance_torque=disturbance_torque)
            self._observer.update(new_state.theta, new_state.theta_dot, u, self._config.dt_inner)

            fell = abs(new_state.theta) > self._config.theta_fail
            time_since_ref_change = self._reference.time_since_last_step_change(t)

            step_reward, reward_terms = compute_reward(
                error=error,
                theta=new_state.theta,
                reference=reference,
                reference_is_step=self._reference.is_step_reference(),
                derivative_error=derivative_error,
                prev_derivative_error=self._prev_derivative_error,
                control_effort=u,
                current_gains=new_gains,
                previous_gains=previous_gains,
                u_max=self._config.limits.u_max,
                epsilon_settle=self._config.settle_epsilon,
                time_since_reference_change=time_since_ref_change,
                expected_settle_time=0.5 * self._config.episode_seconds,
                fell=fell,
                weights=self._config.reward_weights,
            )
            self._prev_derivative_error = derivative_error
            total_reward += step_reward
            for field_name, value in reward_terms.__dict__.items():
                terms_accum[field_name] = terms_accum.get(field_name, 0.0) + value

            self._inner_step_count += 1
            self._prev_u = u

            if fell:
                terminated = True
                break
            if self._inner_step_count >= self._max_inner_steps:
                break

        truncated = (not terminated) and self._inner_step_count >= self._max_inner_steps
        obs = self._build_observation(gains=new_gains, prev_u=self._prev_u)

        for k, v in terms_accum.items():
            self._episode_reward_terms_sum[k] = self._episode_reward_terms_sum.get(k, 0.0) + v

        info = {
            "reward_terms": terms_accum,
            "gains": new_gains,
            "fell": fell,
            "sim_time": self._plant.get_state().time,
        }
        self._last_info = info
        return obs, total_reward, terminated, truncated, info

    @property
    def reference_trajectory(self) -> ReferenceTrajectory:
        """Public accessor for the current episode's reference trajectory,
        needed by evaluation/logging code (e.g. ``evaluation.rollout``) to
        recover the true reference value at a given time without reaching
        into private state."""
        assert self._reference is not None, "call reset() before accessing reference_trajectory"
        return self._reference

    @property
    def dt_inner(self) -> float:
        return self._config.dt_inner

    @property
    def last_control_effort(self) -> float:
        """The most recent (saturated) control torque commanded by the PID
        controller, exposed for evaluation-time logging."""
        return self._prev_u

    def get_plant_time(self) -> float:
        return self._plant.get_state().time

    # -- Internal helpers ----------------------------------------------------

    def _build_observation(self, gains: PIDGains, prev_u: float) -> np.ndarray:
        assert self._reference is not None
        t = self._plant.get_state().time
        noisy_state = self._plant.get_noisy_observation(
            self._theta_noise_std, self._theta_dot_noise_std, self._randomizer.rng
        )
        reference, reference_rate = self._reference.value_and_rate(t, self._config.dt_inner)
        error = reference - noisy_state.theta

        obs = np.array(
            [
                error,
                self._pid.integral_error,
                0.0 if self._prev_derivative_error is None else self._prev_derivative_error,
                noisy_state.theta_dot,
                prev_u,
                self._observer.estimate,
                gains.kp,
                gains.ki,
                gains.kd,
                reference,
                reference_rate,
                min(1.0, self._inner_step_count / max(1, self._max_inner_steps)),
            ],
            dtype=np.float32,
        )
        return obs
