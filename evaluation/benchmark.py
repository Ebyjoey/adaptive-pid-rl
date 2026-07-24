#!/usr/bin/env python3
"""Benchmark all five control strategies (fixed PID, manual tuning,
Ziegler-Nichols, PPO, SAC) against an identical held-out domain-randomization
seed range, and write a summary CSV + markdown table.

Usage
-----
    python -m evaluation.benchmark --n-episodes 30
    python -m evaluation.benchmark --n-episodes 5 --skip-rl   # baselines only, no trained models required
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_pid.envs.gym_env import GymPIDGainEnv
from adaptive_pid.utils.config import load_env_config, load_yaml
from adaptive_pid.utils.logging import get_logger
from evaluation.metrics import compute_rollout_metrics
from evaluation.policies import FixedGainPolicy, GainPolicy, SB3GainPolicy
from evaluation.rollout import run_episode

logger = get_logger(__name__)

# Held-out seed range: disjoint from any seed used during training (which
# uses seeds starting at 0, per configs/training/*.yaml). This is what
# makes "held-out domain randomization" in docs/mdp_design.md Section 6 an
# actual guarantee rather than an assumption.
EVAL_SEED_START = 10_000


def build_baseline_policies(baselines_path: str) -> dict[str, tuple[GainPolicy, dict[str, float]]]:
    """Returns ``{policy_name: (policy, gains_dict)}`` for the three
    non-adaptive baselines. The policy itself never changes gains (see
    ``FixedGainPolicy``); what differs between baselines is purely which
    gains the environment is seeded with, which is applied by overriding
    ``env_config.initial_gains`` per baseline in ``run_benchmark``.
    """
    baselines = load_yaml(baselines_path)
    return {
        "Fixed PID": (FixedGainPolicy("Fixed PID"), _extract_gains(baselines["fixed_pid"])),
        "Manual Tuning": (FixedGainPolicy("Manual Tuning"), _extract_gains(baselines["manual_tuning"])),
        "Ziegler-Nichols": (FixedGainPolicy("Ziegler-Nichols"), _extract_gains(baselines["ziegler_nichols"])),
    }


def _extract_gains(section: dict[str, float]) -> dict[str, float]:
    """Extract just the {kp, ki, kd} keys a baseline config section needs as
    ``initial_gains``, ignoring any extra diagnostic fields (e.g. the
    Ziegler-Nichols section also stores ``ku``/``tu`` for reference, which
    must not be passed through to ``GainScheduler.initial_gains``)."""
    return {"kp": section["kp"], "ki": section["ki"], "kd": section["kd"]}


def try_load_sb3_policy(algo: str, log_dir: str) -> GainPolicy | None:
    """Attempt to load a trained SB3 model + its VecNormalize obs stats.
    Returns ``None`` (rather than raising) if no trained model exists yet,
    so the benchmark script can still run against just the baselines before
    training has been run -- this is deliberate, since training is compute-
    intensive and a reviewer may want to see the benchmark harness work
    before committing to a multi-hour training run.
    """
    model_path = Path(log_dir) / "final_model.zip"
    vecnorm_path = Path(log_dir) / "vecnormalize.pkl"
    if not model_path.exists():
        logger.warning(f"No trained {algo.upper()} model found at {model_path}; skipping {algo.upper()} in benchmark.")
        return None

    from stable_baselines3 import PPO, SAC

    model_cls = {"ppo": PPO, "sac": SAC}[algo]
    model = model_cls.load(str(model_path))

    obs_rms = None
    if vecnorm_path.exists():
        import pickle

        with open(vecnorm_path, "rb") as f:
            vecnorm = pickle.load(f)
        obs_rms = vecnorm.obs_rms

    return SB3GainPolicy(model, name=algo.upper(), obs_rms=obs_rms)


def run_benchmark(
    env_config_path: str,
    baselines_path: str,
    n_episodes: int,
    ppo_log_dir: str,
    sac_log_dir: str,
    skip_rl: bool,
) -> pd.DataFrame:
    base_cfg = load_env_config(env_config_path)
    baseline_policies = build_baseline_policies(baselines_path)

    all_policies: list[tuple[str, GainPolicy, dict[str, float] | None]] = []
    for name, (policy, gains) in baseline_policies.items():
        all_policies.append((name, policy, gains))

    if not skip_rl:
        ppo_policy = try_load_sb3_policy("ppo", ppo_log_dir)
        if ppo_policy is not None:
            all_policies.append(("PPO", ppo_policy, None))
        sac_policy = try_load_sb3_policy("sac", sac_log_dir)
        if sac_policy is not None:
            all_policies.append(("SAC", sac_policy, None))

    rows = []
    for policy_name, policy, fixed_gains in all_policies:
        logger.info(f"Evaluating {policy_name} over {n_episodes} held-out episodes...")
        for i in range(n_episodes):
            seed = EVAL_SEED_START + i

            # For non-adaptive baselines, override the seeded initial gains
            # to this baseline's specific tuned values; RL policies use
            # whatever initial_gains the config specifies (the shared ZN
            # seed point every policy starts exploration/adaptation from).
            cfg = base_cfg
            if fixed_gains is not None:
                import dataclasses

                cfg = dataclasses.replace(base_cfg, initial_gains=fixed_gains)

            env = GymPIDGainEnv(cfg, seed=seed)
            recording = run_episode(env, policy, seed=seed, deterministic=True)

            metrics = compute_rollout_metrics(
                times=recording.times,
                errors=recording.errors,
                theta=recording.theta,
                control_signal=recording.control_signal,
                dt=cfg.dt_inner * cfg.outer_loop_ratio,
                settle_epsilon=cfg.settle_epsilon,
                reference_amplitude=recording.reference_amplitude,
                step_time=recording.step_time,
                fell=recording.fell,
            )

            rows.append(
                {
                    "policy": policy_name,
                    "seed": seed,
                    "rmse": metrics.rmse,
                    "rise_time_s": metrics.rise_time,
                    "settling_time_s": metrics.settling_time,
                    "overshoot_pct": metrics.overshoot_pct,
                    "steady_state_error": metrics.steady_state_error,
                    "control_effort_rms": metrics.control_effort_rms,
                    "energy": metrics.energy,
                    "fell": metrics.fell,
                    "total_reward": recording.total_reward,
                    "kp_final": float(recording.kp_history[-1]) if len(recording.kp_history) > 0 else np.nan,
                    "ki_final": float(recording.ki_history[-1]) if len(recording.ki_history) > 0 else np.nan,
                    "kd_final": float(recording.kd_history[-1]) if len(recording.kd_history) > 0 else np.nan,
                }
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-episode rows into a mean/std summary table per policy,
    with clean flat column names (e.g. ``rmse_mean``, ``rmse_std``) rather
    than a MultiIndex or tuple-labeled columns -- those serialize to
    unreadable strings like ``"('rmse', 'mean')"`` when written to CSV,
    which is not an acceptable deliverable format for a benchmark table."""
    numeric_cols = ["rmse", "rise_time_s", "settling_time_s", "overshoot_pct", "steady_state_error",
                     "control_effort_rms", "energy", "total_reward"]
    grouped = df.groupby("policy")[numeric_cols].agg(["mean", "std"])
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    fall_rate = df.groupby("policy")["fell"].mean().rename("fall_rate")
    summary = pd.concat([grouped, fall_rate], axis=1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=str, default="configs/env/pendulum.yaml")
    parser.add_argument("--baselines", type=str, default="configs/training/baselines.yaml")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--ppo-log-dir", type=str, default="runs/ppo")
    parser.add_argument("--sac-log-dir", type=str, default="runs/sac")
    parser.add_argument("--skip-rl", action="store_true", help="only evaluate the three non-adaptive baselines")
    parser.add_argument("--output-dir", type=str, default="assets")
    args = parser.parse_args()

    df = run_benchmark(
        args.env_config, args.baselines, args.n_episodes, args.ppo_log_dir, args.sac_log_dir, args.skip_rl
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "benchmark_raw.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Wrote per-episode raw results to {raw_path}")

    summary = summarize(df)
    summary_path = output_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path)
    logger.info(f"Wrote summary table to {summary_path}")

    print("\n=== Benchmark Summary (per policy) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
