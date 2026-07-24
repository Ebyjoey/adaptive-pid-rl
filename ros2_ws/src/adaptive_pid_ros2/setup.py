from setuptools import find_packages, setup

package_name = "adaptive_pid_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/adaptive_pid_system.launch.py"]),
    ],
    install_requires=["setuptools", "adaptive-pid-rl"],
    zip_safe=True,
    maintainer="Robotics Software Intern Candidate",
    maintainer_email="engineer@example.com",
    description=(
        "ROS2 Humble node graph for online adaptive PID gain scheduling "
        "(plant, reference, PID controller, RL agent, disturbance, "
        "logging, visualization nodes)."
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "plant_node = adaptive_pid_ros2.plant_node:main",
            "reference_node = adaptive_pid_ros2.reference_node:main",
            "pid_controller_node = adaptive_pid_ros2.pid_controller_node:main",
            "rl_agent_node = adaptive_pid_ros2.rl_agent_node:main",
            "disturbance_node = adaptive_pid_ros2.disturbance_node:main",
            "logging_node = adaptive_pid_ros2.logging_node:main",
            "visualization_node = adaptive_pid_ros2.visualization_node:main",
        ],
    },
)
