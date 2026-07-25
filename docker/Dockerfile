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
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/workspace/.venv

# MuJoCo needs a handful of rendering/GL libraries even in headless
# (EGL/osmesa) rendering mode; colcon/build tools for ros2_ws.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-venv \
        python3-colcon-common-extensions \
        libgl1-mesa-glx \
        libosmesa6 \
        libglfw3 \
        git \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Create a virtual environment to avoid conflicts with system packages
# (the base ROS image has sympy/mpmath installed via distutils which pip
# cannot uninstall).
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Install the Python package (and its pinned dependencies) first, in its own
# layer, so that iterating on application code (copied below) doesn't bust
# the dependency-install Docker layer cache on every rebuild.
# Copy pyproject.toml AND src/ so setuptools can find the package during editable install.
COPY pyproject.toml /workspace/
COPY src/ /workspace/src/
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    # CPU-only torch: an unconstrained `pip install torch` on Linux resolves
    # the CUDA-enabled build, pulling several GB of NVIDIA wheels
    # (nvidia-cublas, nvidia-cusparse, triton, ...) that this project never
    # uses -- MuJoCo and Stable-Baselines3 both run CPU-only here. Installing
    # the CPU wheel explicitly first means the later `pip install -e` (which
    # also depends on torch>=2.1) is satisfied without pip re-resolving to
    # the CUDA build, saving multiple GB of download and build time and
    # avoiding a real risk of exhausting CI runner disk space.
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e /workspace

COPY . /workspace/
RUN pip install --no-cache-dir -e /workspace

# Build the ROS2 workspace (adaptive_pid_msgs + adaptive_pid_ros2), sourcing
# the base ROS2 environment first as colcon build requires it.
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd /workspace/ros2_ws && \
    colcon build --symlink-install"

# Source both the ROS2 base setup and this workspace's overlay on every
# interactive shell / `docker run`, so `ros2 launch ...` works out of the box.
# Also activate the Python venv.
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "source $VENV_PATH/bin/activate" >> /root/.bashrc

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash && source /workspace/.venv/bin/activate && exec \"$@\"", "--"]
CMD ["/bin/bash"]