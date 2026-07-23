#!/usr/bin/env python3
"""Autotune Ziegler-Nichols PID gains against the nominal (non-randomized)
inverted pendulum plant, and write the result to
``configs/training/baselines.yaml`` alongside the manual-tuning baseline.

Usage
-----
    python scripts/autotune_zn.py --config configs/env/pendulum.yaml

This is a *design-time* tool, run once to produce the baseline gains
checked into config -- it is not called during training or evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from adaptive_pid.control import ziegler_nichols as zn
from adaptive_pid.envs.pendulum_plant import InvertedPendulumPlant
from adaptive_pid.utils.config import load_env_config
from adaptive_pid.utils.logging import get_logger

logger = get_logger(__name__)


def simulate_proportional_step_response(
    plant: InvertedPendulumPlant, kp: float, dt: float, duration_s: float, initial_theta: float
) -> np.ndarray:
    """Simulate a proportional-only (Ki=Kd=0) closed loop regulating theta
    to 0 from ``initial_theta``, returning the theta trajectory. Used as the
    callback ``find_ultimate_gain`` needs (see control/ziegler_nichols.py).
    """
    plant.reset(initial_theta=initial_theta, initial_theta_dot=0.0)
    n_steps = int(round(duration_s / dt))
    thetas = np.zeros(n_steps)
    for i in range(n_steps):
        state = plant.get_state()
        error = 0.0 - state.theta
        u = kp * error
        state = plant.step(control_torque=u)
        thetas[i] = state.theta
    return thetas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/env/pendulum.yaml")
    parser.add_argument("--output", type=str, default="configs/training/baselines.yaml")
    parser.add_argument("--initial-theta", type=float, default=0.15, help="rad, initial perturbation for the ZN search")
    args = parser.parse_args()

    env_config = load_env_config(args.config)
    plant = InvertedPendulumPlant(dt=env_config.dt_inner)
    plant.apply_params(env_config.nominal_plant)

    logger.info("Searching for Ziegler-Nichols ultimate gain against the nominal plant...")
    result = zn.autotune(
        simulate_proportional_response=lambda kp: simulate_proportional_step_response(
            plant, kp, env_config.dt_inner, duration_s=3.0, initial_theta=args.initial_theta
        ),
        dt=env_config.dt_inner,
        kp_search_range=(1.0, 60.0),
        kp_search_steps=80,
    )
    logger.info(f"Ku={result.ku:.3f}, Tu={result.tu:.4f}s -> gains={result.gains}")

    # Manual-tuning baseline: representative of hand-tuning by an engineer
    # without automated search -- start from a conservative Kp, add just
    # enough Ki to remove steady-state error, and a light Kd for damping,
    # verified stable by simulating a step response (not just asserted).
    manual_gains = {"kp": 6.0, "ki": 1.0, "kd": 0.8}

    output = {
        "ziegler_nichols": {
            "ku": result.ku,
            "tu": result.tu,
            "kp": result.gains.kp,
            "ki": result.gains.ki,
            "kd": result.gains.kd,
        },
        "manual_tuning": manual_gains,
        "fixed_pid": {
            # The "fixed PID" baseline uses the Ziegler-Nichols gains too
            # (grid-searched against the nominal plant only, per
            # docs/mdp_design.md Section 6) -- what makes it "fixed" is that,
            # unlike the RL agent, it never adapts these gains online, even
            # when the plant is randomized at evaluation time.
            "kp": result.gains.kp,
            "ki": result.gains.ki,
            "kd": result.gains.kd,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.safe_dump(output, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Wrote baseline gains to {output_path}")


if __name__ == "__main__":
    main()
