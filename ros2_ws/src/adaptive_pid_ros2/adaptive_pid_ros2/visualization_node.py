"""visualization_node: publishes an RViz ``Marker`` representing the
pendulum's current pose (for 3D visualization) and logs a compact live
summary line to the ROS2 logger (stdout) for terminal-only demos where
RViz isn't running.

A full plotting dashboard (matplotlib/plotly) is intentionally kept out of
this node -- GUI plotting libraries and rclpy's spin loop don't mix well in
a single process, so live plotting is better done by pointing
``evaluation/plots.py``-style tooling at the CSV ``logging_node`` writes, or
via ``rqt_plot`` subscribing directly to ``/state`` and ``/pid_gains``.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker

from adaptive_pid_msgs.msg import PIDGains as PIDGainsMsg
from adaptive_pid_msgs.msg import PlantState


class VisualizationNode(Node):
    def __init__(self) -> None:
        super().__init__("visualization_node")

        self.declare_parameter("pendulum_length", 0.5)
        self._length = self.get_parameter("pendulum_length").get_parameter_value().double_value

        self._latest_gains: tuple[float, float, float] | None = None
        self._latest_reference = 0.0

        self._state_sub = self.create_subscription(PlantState, "/state", self._on_state, 10)
        self._gains_sub = self.create_subscription(PIDGainsMsg, "/pid_gains", self._on_gains, 10)
        self._reference_sub = self.create_subscription(Float64, "/reference", self._on_reference, 10)

        self._marker_pub = self.create_publisher(Marker, "/pendulum_marker", 10)
        self._log_timer = self.create_timer(0.5, self._on_log_timer)

        self._latest_theta = 0.0
        self.get_logger().info("visualization_node started")

    def _on_reference(self, msg: Float64) -> None:
        self._latest_reference = msg.data

    def _on_gains(self, msg: PIDGainsMsg) -> None:
        self._latest_gains = (msg.kp, msg.ki, msg.kd)

    def _on_state(self, msg: PlantState) -> None:
        self._latest_theta = msg.theta
        self._publish_marker(msg.theta)

    def _publish_marker(self, theta: float) -> None:
        marker = Marker()
        marker.header.frame_id = "pendulum_base"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pendulum"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # theta=0 is upright (+Z); place the cylinder's midpoint at half the
        # pendulum length along the direction (sin(theta), 0, cos(theta)).
        half_length = self._length / 2.0
        marker.pose.position.x = half_length * math.sin(theta)
        marker.pose.position.y = 0.0
        marker.pose.position.z = half_length * math.cos(theta)

        # Orient the cylinder (whose local axis is +Z by default) to point
        # along the pendulum direction via a rotation about the Y axis by theta.
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = math.sin(theta / 2.0)
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = math.cos(theta / 2.0)

        marker.scale.x = 0.04
        marker.scale.y = 0.04
        marker.scale.z = self._length
        marker.color.r = 0.2
        marker.color.g = 0.5
        marker.color.b = 0.8
        marker.color.a = 1.0

        self._marker_pub.publish(marker)

    def _on_log_timer(self) -> None:
        gains_str = "Kp=?, Ki=?, Kd=?" if self._latest_gains is None else (
            f"Kp={self._latest_gains[0]:.2f}, Ki={self._latest_gains[1]:.2f}, Kd={self._latest_gains[2]:.2f}"
        )
        self.get_logger().info(
            f"theta={self._latest_theta:.3f} rad, reference={self._latest_reference:.3f} rad, {gains_str}"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
