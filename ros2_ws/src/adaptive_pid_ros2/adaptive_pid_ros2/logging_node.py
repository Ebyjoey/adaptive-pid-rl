"""logging_node: subscribes to every topic in the system and writes a
synchronized CSV log to disk, for offline analysis with
``evaluation/plots.py`` or any external tool. A full rosbag2 recording
should be taken alongside this node via ``ros2 bag record -a`` (see
``ros2_ws/README.md``); this node's CSV output is a convenience layer for
quick plotting without needing to parse bag files.
"""

from __future__ import annotations

import csv
from pathlib import Path

import rclpy
from adaptive_pid_msgs.msg import PIDGains as PIDGainsMsg
from adaptive_pid_msgs.msg import PlantState, TrainingStats
from rclpy.node import Node
from std_msgs.msg import Float64


class LoggingNode(Node):
    def __init__(self) -> None:
        super().__init__("logging_node")

        self.declare_parameter("output_path", "logs/ros2_run.csv")
        output_path = self.get_parameter("output_path").get_parameter_value().string_value
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = self._output_path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "stamp_ns",
                "topic",
                "theta",
                "theta_dot",
                "control_effort",
                "disturbance_estimate",
                "reference",
                "kp",
                "ki",
                "kd",
                "tracking_error",
                "reward",
                "disturbance_torque",
            ]
        )

        self._state_sub = self.create_subscription(PlantState, "/state", self._on_state, 10)
        self._reference_sub = self.create_subscription(Float64, "/reference", self._on_reference, 10)
        self._gains_sub = self.create_subscription(PIDGainsMsg, "/pid_gains", self._on_gains, 10)
        self._stats_sub = self.create_subscription(TrainingStats, "/training_stats", self._on_stats, 10)
        self._disturbance_sub = self.create_subscription(Float64, "/disturbance", self._on_disturbance, 10)

        self.get_logger().info(f"logging_node started, writing to {self._output_path}")

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_state(self, msg: PlantState) -> None:
        self._writer.writerow(
            [
                self._now_ns(),
                "state",
                msg.theta,
                msg.theta_dot,
                msg.control_effort,
                msg.disturbance_estimate,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        self._file.flush()

    def _on_reference(self, msg: Float64) -> None:
        self._writer.writerow([self._now_ns(), "reference", "", "", "", "", msg.data, "", "", "", "", "", ""])
        self._file.flush()

    def _on_gains(self, msg: PIDGainsMsg) -> None:
        self._writer.writerow(
            [self._now_ns(), "pid_gains", "", "", "", "", "", msg.kp, msg.ki, msg.kd, "", "", ""]
        )
        self._file.flush()

    def _on_stats(self, msg: TrainingStats) -> None:
        self._writer.writerow(
            [
                self._now_ns(),
                f"training_stats/{msg.source}",
                "",
                "",
                "",
                "",
                "",
                msg.kp,
                msg.ki,
                msg.kd,
                msg.tracking_error,
                msg.reward,
                "",
            ]
        )
        self._file.flush()

    def _on_disturbance(self, msg: Float64) -> None:
        self._writer.writerow(
            [self._now_ns(), "disturbance", "", "", "", "", "", "", "", "", "", "", msg.data]
        )
        self._file.flush()

    def destroy_node(self) -> bool:
        self._file.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LoggingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
