"""Launch the full adaptive PID gain scheduling node graph:
reference_node, plant_node, disturbance_node, pid_controller_node,
rl_agent_node, logging_node, visualization_node.

Usage
-----
    ros2 launch adaptive_pid_ros2 adaptive_pid_system.launch.py
    ros2 launch adaptive_pid_ros2 adaptive_pid_system.launch.py algo:=sac model_path:=runs/sac/final_model.zip
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    env_config_arg = DeclareLaunchArgument(
        "env_config_path",
        default_value="configs/env/pendulum.yaml",
        description="Path to the validated environment YAML config (shared across all nodes).",
    )
    algo_arg = DeclareLaunchArgument(
        "algo", default_value="ppo", description="Which trained RL algorithm to load: 'ppo' or 'sac'."
    )
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="runs/ppo/final_model.zip",
        description="Path to the trained SB3 model .zip",
    )
    vecnormalize_path_arg = DeclareLaunchArgument(
        "vecnormalize_path",
        default_value="runs/ppo/vecnormalize.pkl",
        description="Path to the matching VecNormalize stats .pkl",
    )
    seed_arg = DeclareLaunchArgument(
        "seed", default_value="0", description="Random seed for reference/disturbance nodes."
    )

    env_config_path = LaunchConfiguration("env_config_path")
    algo = LaunchConfiguration("algo")
    model_path = LaunchConfiguration("model_path")
    vecnormalize_path = LaunchConfiguration("vecnormalize_path")
    seed = LaunchConfiguration("seed")

    reference_node = Node(
        package="adaptive_pid_ros2",
        executable="reference_node",
        name="reference_node",
        output="screen",
        parameters=[{"env_config_path": env_config_path, "seed": seed}],
    )
    plant_node = Node(
        package="adaptive_pid_ros2",
        executable="plant_node",
        name="plant_node",
        output="screen",
        parameters=[{"env_config_path": env_config_path}],
    )
    disturbance_node = Node(
        package="adaptive_pid_ros2",
        executable="disturbance_node",
        name="disturbance_node",
        output="screen",
        parameters=[{"env_config_path": env_config_path, "seed": seed}],
    )
    pid_controller_node = Node(
        package="adaptive_pid_ros2",
        executable="pid_controller_node",
        name="pid_controller_node",
        output="screen",
        parameters=[{"env_config_path": env_config_path}],
    )
    rl_agent_node = Node(
        package="adaptive_pid_ros2",
        executable="rl_agent_node",
        name="rl_agent_node",
        output="screen",
        parameters=[
            {
                "env_config_path": env_config_path,
                "algo": algo,
                "model_path": model_path,
                "vecnormalize_path": vecnormalize_path,
            }
        ],
    )
    logging_node = Node(
        package="adaptive_pid_ros2",
        executable="logging_node",
        name="logging_node",
        output="screen",
        parameters=[{"output_path": "logs/ros2_run.csv"}],
    )
    visualization_node = Node(
        package="adaptive_pid_ros2",
        executable="visualization_node",
        name="visualization_node",
        output="screen",
    )

    return LaunchDescription(
        [
            env_config_arg,
            algo_arg,
            model_path_arg,
            vecnormalize_path_arg,
            seed_arg,
            reference_node,
            plant_node,
            disturbance_node,
            pid_controller_node,
            rl_agent_node,
            logging_node,
            visualization_node,
        ]
    )
