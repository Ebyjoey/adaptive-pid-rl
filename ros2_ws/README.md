# ROS2 Humble Workspace

This workspace contains two packages:

- **`adaptive_pid_msgs`** (`ament_cmake`) — custom message definitions (`PIDGains`, `PlantState`, `TrainingStats`).
- **`adaptive_pid_ros2`** (`ament_python`) — the seven-node system described in `docs/architecture.md`: `reference_node`, `plant_node`, `disturbance_node`, `pid_controller_node`, `rl_agent_node`, `logging_node`, `visualization_node`.

The node implementations import the `adaptive_pid` Python package (this repo's `src/`) directly — the same `InvertedPendulumPlant`, `PIDController`, `GainScheduler`, `DomainRandomizer`, and `DisturbanceObserver` classes used in training are reused here, so the ROS2 deployment's physics and control logic can never silently diverge from what an RL policy was trained against.

## Prerequisites

- ROS2 Humble (see `docker/Dockerfile` for a fully configured environment — this is the recommended way to run this workspace, since ROS2 Humble targets Ubuntu 22.04 specifically)
- The `adaptive_pid` Python package installed in the same Python environment `rclpy` uses: `pip install -e /path/to/repo` (done automatically in the provided Docker image)

## Build

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Run

Bring up the full system (defaults to a trained PPO policy):

```bash
ros2 launch adaptive_pid_ros2 adaptive_pid_system.launch.py \
    env_config_path:=configs/env/pendulum.yaml \
    algo:=ppo \
    model_path:=runs/ppo/final_model.zip \
    vecnormalize_path:=runs/ppo/vecnormalize.pkl
```

To run against SAC instead:

```bash
ros2 launch adaptive_pid_ros2 adaptive_pid_system.launch.py algo:=sac model_path:=runs/sac/final_model.zip vecnormalize_path:=runs/sac/vecnormalize.pkl
```

**Note:** `model_path`/`vecnormalize_path` must point at an actual trained model produced by `training/train_ppo.py` or `training/train_sac.py` (see the top-level README) — they are not included in this repository since trained artifacts are large binary files that belong in release assets or a model registry, not version control.

## Inspecting the running system

```bash
ros2 topic list
ros2 topic echo /state
ros2 topic echo /pid_gains
ros2 topic echo /training_stats
rqt_graph                      # visualize the live node graph
ros2 bag record -a -o my_run   # full rosbag2 recording, alongside logging_node's CSV output
```

## Running individual nodes (useful for debugging one piece at a time)

```bash
ros2 run adaptive_pid_ros2 plant_node --ros-args -p initial_theta:=0.2
ros2 run adaptive_pid_ros2 pid_controller_node
ros2 run adaptive_pid_ros2 rl_agent_node --ros-args -p algo:=ppo -p model_path:=runs/ppo/final_model.zip
```
