#!/usr/bin/env python3
"""Publication-quality plotting utilities.

Produces the figures required by the project's evaluation criteria: PPO/SAC
learning curves, reward curves, PID gain evolution over an episode/training
run, and benchmark comparison bar charts across all five control strategies.

Each function here takes already-computed data (arrays / DataFrames) rather
than reaching into SB3 logs or environment internals itself, keeping the
plotting code decoupled from how the data was produced (trained model
rollout vs. TensorBoard log parsing vs. benchmark CSV).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.figsize": (8, 5),
    }
)

_COLORS = {
    "Fixed PID": "#8c8c8c",
    "Manual Tuning": "#4c72b0",
    "Ziegler-Nichols": "#dd8452",
    "PPO": "#55a868",
    "SAC": "#c44e52",
}


def plot_learning_curve(
    timesteps: np.ndarray, episode_rewards: np.ndarray, title: str, output_path: str
) -> None:
    """Reward vs. training timesteps, with a rolling mean overlay to make
    the underlying trend visible through per-episode noise."""
    fig, ax = plt.subplots()
    ax.plot(timesteps, episode_rewards, alpha=0.25, color="tab:blue", label="Episode reward")
    if len(episode_rewards) >= 10:
        window = max(1, len(episode_rewards) // 20)
        rolling = pd.Series(episode_rewards).rolling(window, min_periods=1).mean()
        ax.plot(timesteps, rolling, color="tab:blue", linewidth=2, label=f"Rolling mean (window={window})")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode reward")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path)


def plot_reward_term_breakdown(
    timesteps: np.ndarray, terms: dict[str, np.ndarray], title: str, output_path: str
) -> None:
    """Stacked/overlaid view of each (negated, weighted) reward term's
    contribution over training, to diagnose which objective is dominating
    policy behavior at each training stage."""
    fig, ax = plt.subplots()
    for term_name, values in terms.items():
        ax.plot(timesteps, values, label=term_name, linewidth=1.5)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean per-step term value (unweighted)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, output_path)


def plot_gain_evolution(
    times: np.ndarray, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray, title: str, output_path: str
) -> None:
    """PID gain evolution over a single episode -- the core qualitative
    result this project needs to demonstrate: the agent adapting Kp/Ki/Kd
    online as plant conditions or tracking demands change."""
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    for ax, values, label, color in zip(
        axes, [kp, ki, kd], ["Kp", "Ki", "Kd"], ["#4c72b0", "#dd8452", "#55a868"]
    ):
        ax.plot(times, values, color=color, linewidth=1.8)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, output_path)


def plot_tracking_response(
    times: np.ndarray, theta: np.ndarray, reference: np.ndarray, title: str, output_path: str
) -> None:
    """Reference vs. actual angle over one episode -- the standard control
    engineering step/tracking-response plot."""
    fig, ax = plt.subplots()
    ax.plot(times, reference, "--", color="black", label="Reference", linewidth=1.5)
    ax.plot(times, theta, color="#4c72b0", label="Plant response (theta)", linewidth=1.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (rad)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_path)


def plot_benchmark_comparison(
    summary_df: pd.DataFrame, metric_col: str, ylabel: str, title: str, output_path: str
) -> None:
    """Bar chart comparing one metric's mean +/- std across all evaluated
    policies -- used for RMSE, overshoot, settling time, energy, etc.
    ``summary_df`` is expected in the flat ``{metric}_mean``/``{metric}_std``
    column format produced by ``evaluation.benchmark.summarize``.
    """
    policies = list(summary_df.index)
    means = [summary_df.loc[p, f"{metric_col}_mean"] for p in policies]
    stds = [summary_df.loc[p, f"{metric_col}_std"] for p in policies]
    colors = [_COLORS.get(p, "tab:gray") for p in policies]

    fig, ax = plt.subplots()
    ax.bar(policies, means, yerr=stds, capsize=4, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, output_path)


def plot_fall_rate_comparison(summary_df: pd.DataFrame, output_path: str) -> None:
    """Bar chart of episode failure (fall) rate per policy -- the single
    most important robustness metric under domain randomization."""
    policies = list(summary_df.index)
    fall_rates = [summary_df.loc[p, "fall_rate"] * 100 for p in policies]
    colors = [_COLORS.get(p, "tab:gray") for p in policies]

    fig, ax = plt.subplots()
    ax.bar(policies, fall_rates, color=colors)
    ax.set_ylabel("Episode fall rate (%)")
    ax.set_title("Robustness Under Domain Randomization")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, output_path)


def _save(fig: plt.Figure, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
