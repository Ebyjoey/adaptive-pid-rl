# Adaptive PID Gain Scheduling -- development/runtime container.
#
# Based on the official ROS2 Humble image (Ubuntu 22.04), with the Python RL
# stack (Gymnasium, MuJoCo, Stable-Baselines3, PyTorch) layered on top, so a
# single image can both run the training/evaluation pipeline AND build/run
# the ros2_ws package -- matching this project's architectural requirement
# that the same physics/control code runs in both contexts.

FROM osrf/ros:humble-desktop AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# MuJoCo needs a handful of rendering/GL libraries even in headless
# (EGL/osmesa) rendering mode; colcon/build tools for ros2_ws.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-colcon-common-extensions \
        libgl1-mesa-glx \
        libosmesa6 \
        libglfw3 \
        git \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install the Python package (and its pinned dependencies) first, in its own
# layer, so that iterating on application code (copied below) doesn't bust
# the dependency-install Docker layer cache on every rebuild.
COPY pyproject.toml /workspace/
RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    pip3 install --no-cache-dir -e /workspace

COPY . /workspace/
RUN pip3 install --no-cache-dir -e /workspace

# Build the ROS2 workspace (adaptive_pid_msgs + adaptive_pid_ros2), sourcing
# the base ROS2 environment first as colcon build requires it.
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd /workspace/ros2_ws && \
    colcon build --symlink-install"

# Source both the ROS2 base setup and this workspace's overlay on every
# interactive shell / `docker run`, so `ros2 launch ...` works out of the box.
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["/bin/bash"]
