"""disturbance_node: injects disturbance torque impulses onto ``/disturbance``
for HIL robustness testing and live demos, reusing
``adaptive_pid.envs.domain_randomization.DomainRandomizer`` so the
disturbance statistics match exactly what the RL agent was trained against.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from adaptive_pid.envs.domain_randomization import DomainRandomizer, RandomizationRanges
from adaptive_pid.utils.config import load_env_config

DEFAULT_ENV_CONFIG = "configs/env/pendulum.yaml"


class DisturbanceNode(Node):
    def __init__(self) -> None:
        super().__init__("disturbance_node")

        self.declare_parameter("env_config_path", DEFAULT_ENV_CONFIG)
        self.declare_parameter("seed", 0)
        self.declare_parameter("enabled", True)
        env_config_path = self.get_parameter("env_config_path").get_parameter_value().string_value
        seed = self.get_parameter("seed").get_parameter_value().integer_value
        self._enabled = self.get_parameter("enabled").get_parameter_value().bool_value

        env_config = load_env_config(env_config_path)
        ranges = RandomizationRanges.from_dict(env_config.randomization)
        self._randomizer = DomainRandomizer(ranges, seed=seed)
        self._dt = env_config.dt_inner

        self._start_time = self.get_clock().now()
        self._disturbance_pub = self.create_publisher(Float64, "/disturbance", 10)
        self._timer = self.create_timer(self._dt, self._on_timer)
        self.get_logger().info(f"disturbance_node started (enabled={self._enabled})")

    def _on_timer(self) -> None:
        if not self._enabled:
            return
        elapsed_s = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        event = self._randomizer.maybe_sample_disturbance(elapsed_s)
        if event is not None:
            msg = Float64()
            msg.data = event.torque
            self._disturbance_pub.publish(msg)
            self.get_logger().debug(f"Injected disturbance torque {event.torque:.3f} N*m at t={elapsed_s:.2f}s")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DisturbanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
