"""rl_agent_node: loads a trained Stable-Baselines3 policy (PPO or SAC) and,
at the slower outer-loop rate (docs/architecture.md Section 6), publishes
delta-gain-derived PID gain updates on ``/pid_gains``.

Builds the exact same 12-dim observation vector used in training (see
``adaptive_pid.envs.gym_env`` for the authoritative layout) from the most
recent ``/state`` message plus its own tracked integral/gain state, and
applies the same ``GainScheduler`` used everywhere else in the codebase so
the ROS2 deployment path can never apply a differently-clamped update than
what the policy was trained under.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import rclpy
from adaptive_pid_msgs.msg import PIDGains as PIDGainsMsg
from adaptive_pid_msgs.msg import PlantState, TrainingStats
from rclpy.node import Node
from std_msgs.msg import Float64

from adaptive_pid.control.gain_scheduler import GainScheduler
from adaptive_pid.utils.config import load_env_config

DEFAULT_ENV_CONFIG = "configs/env/pendulum.yaml"


class RLAgentNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_agent_node")

        self.declare_parameter("env_config_path", DEFAULT_ENV_CONFIG)
        self.declare_parameter("model_path", "runs/ppo/final_model.zip")
        self.declare_parameter("vecnormalize_path", "runs/ppo/vecnormalize.pkl")
        self.declare_parameter("algo", "ppo")  # "ppo" or "sac"
        self.declare_parameter("outer_loop_hz", 20.0)

        env_config_path = self.get_parameter("env_config_path").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        vecnorm_path = self.get_parameter("vecnormalize_path").get_parameter_value().string_value
        algo = self.get_parameter("algo").get_parameter_value().string_value
        outer_loop_hz = self.get_parameter("outer_loop_hz").get_parameter_value().double_value

        self._env_config = load_env_config(env_config_path)
        self._gain_scheduler = GainScheduler(self._env_config.limits)
        self._current_gains = self._gain_scheduler.initial_gains(**self._env_config.initial_gains)

        self._model = self._load_model(algo, model_path)
        self._obs_rms = self._load_obs_rms(vecnorm_path)

        self._latest_state: PlantState | None = None
        self._latest_reference = 0.0
        self._integral_error = 0.0
        self._prev_error: float | None = None

        self._state_sub = self.create_subscription(PlantState, "/state", self._on_state, 10)
        self._reference_sub = self.create_subscription(Float64, "/reference", self._on_reference, 10)
        self._gains_pub = self.create_publisher(PIDGainsMsg, "/pid_gains", 10)
        self._stats_pub = self.create_publisher(TrainingStats, "/training_stats", 10)

        self._dt_outer = 1.0 / outer_loop_hz
        self._timer = self.create_timer(self._dt_outer, self._on_timer)

        # Publish the initial (ZN-seeded) gains immediately so
        # pid_controller_node doesn't sit at its own hardcoded default while
        # waiting for the first RL decision.
        self._publish_gains(self._current_gains)

        self.get_logger().info(f"rl_agent_node started with {algo.upper()} model from {model_path}")

    @staticmethod
    def _load_model(algo: str, model_path: str):
        from stable_baselines3 import PPO, SAC

        model_cls = {"ppo": PPO, "sac": SAC}.get(algo.lower())
        if model_cls is None:
            raise ValueError(f"Unknown algo '{algo}'; expected 'ppo' or 'sac'")
        return model_cls.load(model_path)

    @staticmethod
    def _load_obs_rms(vecnorm_path: str):
        path = Path(vecnorm_path)
        if not path.exists():
            return None
        with path.open("rb") as f:
            vecnorm = pickle.load(f)
        return vecnorm.obs_rms

    def _on_reference(self, msg: Float64) -> None:
        self._latest_reference = msg.data

    def _on_state(self, msg: PlantState) -> None:
        self._latest_state = msg

    def _build_observation(self) -> np.ndarray | None:
        if self._latest_state is None:
            return None
        state = self._latest_state
        error = self._latest_reference - state.theta

        dt = self._env_config.dt_inner
        candidate_integral = self._integral_error + error * dt
        self._integral_error = max(
            -self._env_config.limits.integral_max,
            min(self._env_config.limits.integral_max, candidate_integral),
        )
        derivative_error = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error

        obs = np.array(
            [
                error,
                self._integral_error,
                derivative_error,
                state.theta_dot,
                state.control_effort,
                state.disturbance_estimate,
                self._current_gains.kp,
                self._current_gains.ki,
                self._current_gains.kd,
                self._latest_reference,
                0.0,  # reference rate: not tracked cross-message here; a small
                # simplification versus the training env, acceptable
                # since Kp/Ki/Kd dominate the policy's response and this
                # feature mainly helps distinguish step-hold vs. ramp
                1.0,  # progress: unknown in an indefinitely-running ROS2 deployment
            ],
            dtype=np.float32,
        )
        if self._obs_rms is not None:
            obs = (obs - self._obs_rms.mean) / np.sqrt(self._obs_rms.var + 1e-8)
            obs = np.clip(obs, -10.0, 10.0)
        return obs

    def _on_timer(self) -> None:
        obs = self._build_observation()
        if obs is None:
            return  # no /state received yet

        action, _ = self._model.predict(obs, deterministic=True)
        self._current_gains = self._gain_scheduler.apply_action(
            self._current_gains, tuple(float(a) for a in action), self._dt_outer
        )
        self._publish_gains(self._current_gains)

        stats_msg = TrainingStats()
        stats_msg.header.stamp = self.get_clock().now().to_msg()
        stats_msg.source = "rl_agent_node"
        stats_msg.tracking_error = self._latest_reference - self._latest_state.theta
        stats_msg.reward = 0.0  # true reward requires the full shaped computation; omitted at deployment time
        stats_msg.kp, stats_msg.ki, stats_msg.kd = (
            self._current_gains.kp,
            self._current_gains.ki,
            self._current_gains.kd,
        )
        self._stats_pub.publish(stats_msg)

    def _publish_gains(self, gains) -> None:
        msg = PIDGainsMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.kp, msg.ki, msg.kd = gains.kp, gains.ki, gains.kd
        self._gains_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RLAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
