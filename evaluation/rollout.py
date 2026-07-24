"""Rollout execution: runs a ``GainPolicy`` against a ``GymPIDGainEnv``
instance for one episode, recording the outer-loop-resolution time series
needed by ``evaluation.metrics``.

Uses only ``GymPIDGainEnv``'s public API (``reset``, ``step``,
``reference_trajectory``, ``last_control_effort``, ``get_plant_time``) --
never private attributes -- so this module stays decoupled from the
environment's internal composition, per the project's dependency-inversion
principles (docs/architecture.md Section 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adaptive_pid.envs.gym_env import GymPIDGainEnv
from adaptive_pid.utils.types import PIDGains
from evaluation.policies import GainPolicy


@dataclass
class RolloutRecording:
    times: np.ndarray
    theta: np.ndarray
    errors: np.ndarray
    control_signal: np.ndarray
    kp_history: np.ndarray
    ki_history: np.ndarray
    kd_history: np.ndarray
    reference_amplitude: float
    step_time: float
    fell: bool
    total_reward: float


def run_episode(
    env: GymPIDGainEnv, policy: GainPolicy, seed: int, deterministic: bool = True
) -> RolloutRecording:
    """Run one full episode of ``policy`` against ``env``, recording one
    sample per outer-loop (RL decision) step -- coarser than the 10ms inner
    physics loop, but sufficient resolution for RMSE, rise time, settling
    time, and overshoot, all of which operate on timescales well above a
    single physics tick.
    """
    obs, _reset_info = env.reset(seed=seed)
    reference_traj = env.reference_trajectory
    reference_amplitude = float(reference_traj.amplitude)
    step_time = (
        float(reference_traj.step_change_times[0]) if len(reference_traj.step_change_times) > 0 else 0.0
    )

    times: list[float] = []
    theta_list: list[float] = []
    errors: list[float] = []
    control_signal: list[float] = []
    kp_hist: list[float] = []
    ki_hist: list[float] = []
    kd_hist: list[float] = []

    fell = False
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = policy.act(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        fell = info["fell"]

        t = env.get_plant_time()
        reference, _rate = reference_traj.value_and_rate(t, env.dt_inner)
        # obs[0] is the (noisy) tracking error at the post-step observation;
        # theta is recovered from reference - error for consistent bookkeeping.
        error = float(obs[0])
        theta = reference - error

        times.append(t)
        theta_list.append(theta)
        errors.append(error)
        control_signal.append(env.last_control_effort)
        gains: PIDGains = info["gains"]
        kp_hist.append(gains.kp)
        ki_hist.append(gains.ki)
        kd_hist.append(gains.kd)

    return RolloutRecording(
        times=np.array(times),
        theta=np.array(theta_list),
        errors=np.array(errors),
        control_signal=np.array(control_signal),
        kp_history=np.array(kp_hist),
        ki_history=np.array(ki_hist),
        kd_history=np.array(kd_hist),
        reference_amplitude=reference_amplitude,
        step_time=step_time,
        fell=fell,
        total_reward=total_reward,
    )
