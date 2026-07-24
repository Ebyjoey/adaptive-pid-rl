"""reference_node: publishes the reference (setpoint) trajectory on
``/reference``, reusing ``adaptive_pid.envs.reference_trajectory`` so the
ROS2 demo's reference shapes are identical to those used in training/eval.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from adaptive_pid.envs.reference_trajectory import sample_reference_trajectory
from adaptive_pid.utils.config import load_env_config

DEFAULT_ENV_CONFIG = "configs/env/pendulum.yaml"


class ReferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("reference_node")

        self.declare_parameter("env_config_path", DEFAULT_ENV_CONFIG)
        self.declare_parameter("seed", 0)
        env_config_path = self.get_parameter("env_config_path").get_parameter_value().string_value
        seed = self.get_parameter("seed").get_parameter_value().integer_value

        self._env_config = load_env_config(env_config_path)
        rng = np.random.default_rng(seed)
        self._trajectory = sample_reference_trajectory(rng, self._env_config.episode_seconds, max_amplitude=0.5)

        self._start_time = self.get_clock().now()
        self._reference_pub = self.create_publisher(Float64, "/reference", 10)
        self._timer = self.create_timer(self._env_config.dt_inner, self._on_timer)
        self.get_logger().info(f"reference_node started with trajectory kind={self._trajectory.kind.value}")

    def _on_timer(self) -> None:
        elapsed_s = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        reference, _rate = self._trajectory.value_and_rate(elapsed_s, self._env_config.dt_inner)
        msg = Float64()
        msg.data = reference
        self._reference_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ReferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
