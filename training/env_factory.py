"""Shared environment construction for training scripts.

Both ``train_ppo.py`` and ``train_sac.py`` need an *identical* environment
setup (same config, same wrappers, same normalization) so that any
performance difference measured between PPO and SAC in evaluation reflects
the algorithms, not incidental differences in how each was trained. This
module is the single place that wiring is defined.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from adaptive_pid.envs.gym_env import GymPIDGainEnv
from adaptive_pid.utils.config import EnvConfig, load_env_config


def _make_single_env(env_config: EnvConfig, seed: int, monitor_dir: str | None):
    def _init():
        env = GymPIDGainEnv(env_config, seed=seed)
        log_file = str(Path(monitor_dir) / f"env_{seed}") if monitor_dir else None
        return Monitor(env, filename=log_file)

    return _init


def build_training_env(
    env_config_path: str,
    n_envs: int = 4,
    seed: int = 0,
    monitor_dir: str | None = None,
    norm_reward: bool = True,
) -> VecNormalize:
    """Build an ``n_envs``-way vectorized, ``VecNormalize``-wrapped training
    environment from a validated env config file.

    ``VecNormalize`` is used (rather than normalizing manually inside
    ``GymPIDGainEnv``) because SB3's implementation correctly maintains
    running statistics *only from training-time experience* and freezes
    them at evaluation time (via ``training=False``), which is the standard,
    correct way to avoid evaluation-time statistics leaking into a
    supposedly-held-out test distribution.
    """
    env_config = load_env_config(env_config_path)
    if monitor_dir:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)

    env_fns = [_make_single_env(env_config, seed=seed + i, monitor_dir=monitor_dir) for i in range(n_envs)]
    # Each of the n_envs needs a *different* seed, so we construct the
    # VecEnv directly from per-index env_fns rather than via
    # make_vec_env(single_fn, n_envs=n), which would give every sub-env the
    # same seed.
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)
    return vec_env


def build_eval_env(
    env_config_path: str, seed: int, vecnormalize_stats_path: str | None = None
) -> VecNormalize:
    """Build a single-env evaluation environment, optionally loading frozen
    ``VecNormalize`` statistics from a completed training run so evaluation
    observations are normalized identically to how the policy was trained
    (loading fresh/unfitted statistics here would silently corrupt policy
    inputs at evaluation time).
    """
    env_config = load_env_config(env_config_path)
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([_make_single_env(env_config, seed=seed, monitor_dir=None)])
    if vecnormalize_stats_path is not None:
        vec_env = VecNormalize.load(vecnormalize_stats_path, vec_env)
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env
