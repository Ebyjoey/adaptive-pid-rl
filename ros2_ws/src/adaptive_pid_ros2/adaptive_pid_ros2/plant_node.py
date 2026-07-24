"""plant_node: integrates the pendulum's physics at a fixed rate and
publishes its state on ``/state``.

This node reuses ``adaptive_pid.envs.pendulum_plant.InvertedPendulumPlant``
and ``adaptive_pid.estimation.disturbance_observer.DisturbanceObserver`` --
the exact same classes used by ``GymPIDGainEnv`` in training -- so the
physics driving a ROS2-based demo or HIL test can never silently diverge
from what the RL policy was trained against (docs/architecture.md Section 1).
"""

from __future__ import annotations

import rclpy
from adaptive_pid_msgs.msg import PlantState
from rclpy.node import Node
from std_msgs.msg import Float64

from adaptive_pid.envs.pendulum_plant import InvertedPendulumPlant
from adaptive_pid.estimation.disturbance_observer import DisturbanceObserver, DisturbanceObserverConfig
from adaptive_pid.utils.config import load_env_config

DEFAULT_ENV_CONFIG = "configs/env/pendulum.yaml"


class PlantNode(Node):
    def __init__(self) -> None:
        super().__init__("plant_node")

        self.declare_parameter("env_config_path", DEFAULT_ENV_CONFIG)
        self.declare_parameter("initial_theta", 0.15)
        env_config_path = self.get_parameter("env_config_path").get_parameter_value().string_value
        initial_theta = self.get_parameter("initial_theta").get_parameter_value().double_value

        self._env_config = load_env_config(env_config_path)
        self._plant = InvertedPendulumPlant(dt=self._env_config.dt_inner)
        self._plant.apply_params(self._env_config.nominal_plant)
        self._plant.reset(initial_theta=initial_theta, initial_theta_dot=0.0)

        self._observer = DisturbanceObserver(
            DisturbanceObserverConfig(
                nominal_mass=self._env_config.nominal_plant.mass,
                nominal_length=self._env_config.nominal_plant.length,
                nominal_damping=self._env_config.nominal_plant.damping,
                gravity=self._env_config.nominal_plant.gravity,
            )
        )

        self._latest_control_torque = 0.0
        self._latest_disturbance_torque = 0.0

        self._control_sub = self.create_subscription(Float64, "/control_input", self._on_control_input, 10)
        self._disturbance_sub = self.create_subscription(Float64, "/disturbance", self._on_disturbance, 10)
        self._state_pub = self.create_publisher(PlantState, "/state", 10)

        self._timer = self.create_timer(self._env_config.dt_inner, self._on_timer)
        self.get_logger().info(
            f"plant_node started (dt={self._env_config.dt_inner}s, initial_theta={initial_theta} rad)"
        )

    def _on_control_input(self, msg: Float64) -> None:
        self._latest_control_torque = msg.data

    def _on_disturbance(self, msg: Float64) -> None:
        self._latest_disturbance_torque = msg.data

    def _on_timer(self) -> None:
        state = self._plant.step(
            control_torque=self._latest_control_torque, disturbance_torque=self._latest_disturbance_torque
        )
        self._observer.update(
            state.theta, state.theta_dot, self._latest_control_torque, self._env_config.dt_inner
        )

        msg = PlantState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "pendulum_base"
        msg.theta = state.theta
        msg.theta_dot = state.theta_dot
        msg.control_effort = self._latest_control_torque
        msg.disturbance_estimate = self._observer.estimate
        self._state_pub.publish(msg)

        # Disturbance is a one-shot impulse per message received, not a
        # continuously-applied force, so it is cleared after being applied
        # for exactly one physics step -- matching how disturbance_node
        # publishes discrete events, not a continuous signal.
        self._latest_disturbance_torque = 0.0


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PlantNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
