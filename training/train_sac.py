#!/usr/bin/env python3
"""Train a SAC agent to perform adaptive PID gain scheduling.

Usage
-----
    python -m training.train_sac --config configs/training/sac.yaml
    python -m training.train_sac --config configs/training/sac.yaml --total-timesteps 4000 --n-envs 1  # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from adaptive_pid.utils.config import load_yaml
from adaptive_pid.utils.logging import get_logger
from training.callbacks import GainAndRewardTermLoggingCallback
from training.env_factory import build_training_env

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/training/sac.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
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

    # ent_coef in SB3's SAC accepts the literal string "auto" directly (it is
    # not a Python-object placeholder like PPO's activation_fn), so no
    # resolution step is needed here -- passed straight through from YAML.
    model = SAC(
        cfg["policy"],
        train_env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        learning_starts=cfg["learning_starts"],
        batch_size=cfg["batch_size"],
        tau=cfg["tau"],
        gamma=cfg["gamma"],
        train_freq=cfg["train_freq"],
        gradient_steps=cfg["gradient_steps"],
        ent_coef=cfg["ent_coef"],
        policy_kwargs=cfg.get("policy_kwargs", {}),
        tensorboard_log=cfg["tensorboard_log"],
        seed=cfg["seed"],
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg["checkpoint_freq"] // n_envs, 1),
        save_path=str(log_dir / "checkpoints"),
        name_prefix="sac_pid_gain",
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
    gain_logging_callback = GainAndRewardTermLoggingCallback(log_freq=1000)

    logger.info(f"Starting SAC training for {total_timesteps} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([checkpoint_callback, eval_callback, gain_logging_callback]),
        tb_log_name="sac",
    )

    final_model_path = log_dir / "final_model.zip"
    model.save(str(final_model_path))
    train_env.save(str(log_dir / "vecnormalize.pkl"))
    logger.info(f"Saved final model to {final_model_path} and VecNormalize stats to {log_dir / 'vecnormalize.pkl'}")


if __name__ == "__main__":
    main()
