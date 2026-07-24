"""pid_controller_node: computes the control torque from the tracking error
between ``/reference`` and ``/state``, using the same ``PIDController`` core
(with clamped-integration anti-windup and output saturation) used in
training. Gains are updated live from ``/pid_gains`` whenever the
``rl_agent_node`` publishes an update, without resetting the controller's
integral/derivative history (see ``PIDController.set_gains``).
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from adaptive_pid.control.gain_scheduler import GainScheduler
from adaptive_pid.control.pid import PIDController
from adaptive_pid.utils.config import load_env_config
from adaptive_pid.utils.types import PIDGains
from adaptive_pid_msgs.msg import PIDGains as PIDGainsMsg
from adaptive_pid_msgs.msg import PlantState, TrainingStats

DEFAULT_ENV_CONFIG = "configs/env/pendulum.yaml"


class PIDControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("pid_controller_node")

        self.declare_parameter("env_config_path", DEFAULT_ENV_CONFIG)
        env_config_path = self.get_parameter("env_config_path").get_parameter_value().string_value
        self._env_config = load_env_config(env_config_path)

        gain_scheduler = GainScheduler(self._env_config.limits)
        initial_gains = gain_scheduler.initial_gains(**self._env_config.initial_gains)
        self._pid = PIDController(
            gains=initial_gains,
            dt=self._env_config.dt_inner,
            integral_max=self._env_config.limits.integral_max,
            output_max=self._env_config.limits.u_max,
        )

        self._latest_reference = 0.0
        self._latest_theta: float | None = None

        self._reference_sub = self.create_subscription(Float64, "/reference", self._on_reference, 10)
        self._state_sub = self.create_subscription(PlantState, "/state", self._on_state, 10)
        self._gains_sub = self.create_subscription(PIDGainsMsg, "/pid_gains", self._on_gains, 10)

        self._control_pub = self.create_publisher(Float64, "/control_input", 10)
        self._stats_pub = self.create_publisher(TrainingStats, "/training_stats", 10)

        self.get_logger().info(
            f"pid_controller_node started with initial gains {initial_gains.as_array()}"
        )

    def _on_reference(self, msg: Float64) -> None:
        self._latest_reference = msg.data

    def _on_gains(self, msg: PIDGainsMsg) -> None:
        self._pid.set_gains(PIDGains(kp=msg.kp, ki=msg.ki, kd=msg.kd))
        self.get_logger().debug(f"Gains updated: Kp={msg.kp:.3f} Ki={msg.ki:.3f} Kd={msg.kd:.3f}")

    def _on_state(self, msg: PlantState) -> None:
        self._latest_theta = msg.theta
        error = self._latest_reference - msg.theta
        u, _integral_error, _derivative_error = self._pid.step(error)

        control_msg = Float64()
        control_msg.data = u
        self._control_pub.publish(control_msg)

        stats_msg = TrainingStats()
        stats_msg.header.stamp = self.get_clock().now().to_msg()
        stats_msg.source = "pid_controller_node"
        stats_msg.tracking_error = error
        stats_msg.reward = 0.0  # reward is only meaningful from rl_agent_node
        gains = self._pid.gains
        stats_msg.kp, stats_msg.ki, stats_msg.kd = gains.kp, gains.ki, gains.kd
        self._stats_pub.publish(stats_msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PIDControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
