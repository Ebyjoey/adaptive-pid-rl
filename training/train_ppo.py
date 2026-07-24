#!/usr/bin/env python3
"""Train a PPO agent to perform adaptive PID gain scheduling.

Usage
-----
    python -m training.train_ppo --config configs/training/ppo.yaml
    python -m training.train_ppo --config configs/training/ppo.yaml --total-timesteps 5000 --n-envs 2  # smoke test

Resuming a long run across multiple sessions (e.g. time-limited or
preemptible compute):
    python -m training.train_ppo --config configs/training/ppo.yaml --total-timesteps 150000
    python -m training.train_ppo --config configs/training/ppo.yaml --total-timesteps 150000 --resume
    python -m training.train_ppo --config configs/training/ppo.yaml --total-timesteps 150000 --resume
    # each --resume call trains `total_timesteps` *additional* steps on top
    # of whatever was already saved to log_dir/final_model.zip, rather than
    # overwriting from scratch -- this is not a demo convenience, it is the
    # standard way to run RL training on infrastructure that cannot
    # guarantee an uninterrupted multi-hour session.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from adaptive_pid.utils.config import load_yaml
from adaptive_pid.utils.logging import get_logger
from training.callbacks import GainAndRewardTermLoggingCallback
from training.env_factory import build_training_env

logger = get_logger(__name__)

_ACTIVATION_FNS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}


def _resolve_policy_kwargs(raw_policy_kwargs: dict) -> dict:
    """SB3 policy_kwargs expects an actual nn.Module class for
    activation_fn, but YAML can only store the string name -- resolve it
    here rather than embedding non-serializable Python objects in config."""
    resolved = dict(raw_policy_kwargs)
    if "activation_fn" in resolved:
        name = resolved["activation_fn"]
        if name not in _ACTIVATION_FNS:
            raise ValueError(f"Unknown activation_fn '{name}'. Valid options: {sorted(_ACTIVATION_FNS)}")
        resolved["activation_fn"] = _ACTIVATION_FNS[name]
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/training/ppo.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None, help="override config's total_timesteps")
    parser.add_argument("--n-envs", type=int, default=None, help="override config's n_envs")
    parser.add_argument(
        "--resume", action="store_true",
        help="Load log_dir/final_model.zip + vecnormalize.pkl and train --total-timesteps additional "
             "steps on top, instead of starting from scratch."
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    total_timesteps = args.total_timesteps or cfg["total_timesteps"]
    n_envs = args.n_envs or cfg["n_envs"]

    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building {n_envs} parallel training envs from {cfg['env_config']}")
    train_env = build_training_env(
        cfg["env_config"], n_envs=n_envs, seed=cfg["seed"], monitor_dir=str(log_dir / "monitor")
    )
    eval_env = build_training_env(cfg["env_config"], n_envs=1, seed=cfg["seed"] + 1000, norm_reward=False)

    resume_model_path = log_dir / "final_model.zip"
    resume_vecnorm_path = log_dir / "vecnormalize.pkl"
    if args.resume:
        if not resume_model_path.exists():
            raise FileNotFoundError(
                f"--resume was given but no checkpoint exists at {resume_model_path}; run without --resume first."
            )
        logger.info(f"Resuming from {resume_model_path}")
        train_env = VecNormalize.load(str(resume_vecnorm_path), train_env.venv)
        model = PPO.load(str(resume_model_path), env=train_env)
    else:
        model = PPO(
            cfg["policy"],
            train_env,
            learning_rate=cfg["learning_rate"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_range=cfg["clip_range"],
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            policy_kwargs=_resolve_policy_kwargs(cfg.get("policy_kwargs", {})),
            tensorboard_log=cfg["tensorboard_log"],
            seed=cfg["seed"],
            verbose=1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg["checkpoint_freq"] // n_envs, 1),
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_pid_gain",
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(cfg["eval_freq"] // n_envs, 1),
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
    )
    # SB3's EvalCallback automatically calls sync_envs_normalization(model's
    # training env, eval_env) before each evaluation rollout as long as
    # eval_env is a VecNormalize instance, which it is here -- this keeps
    # evaluation-time observation normalization consistent with whatever
    # running statistics the policy is currently training under, without
    # normalizing evaluation *rewards* (norm_reward=False above), since we
    # want true, comparable returns for benchmark reporting.
    gain_logging_callback = GainAndRewardTermLoggingCallback(log_freq=1000)

    logger.info(f"{'Resuming' if args.resume else 'Starting'} PPO training for {total_timesteps} "
                f"{'additional ' if args.resume else ''}timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([checkpoint_callback, eval_callback, gain_logging_callback]),
        tb_log_name="ppo",
        reset_num_timesteps=not args.resume,
    )

    final_model_path = log_dir / "final_model.zip"
    model.save(str(final_model_path))
    train_env.save(str(log_dir / "vecnormalize.pkl"))
    logger.info(f"Saved final model to {final_model_path} and VecNormalize stats to {log_dir / 'vecnormalize.pkl'}")


if __name__ == "__main__":
    main()
