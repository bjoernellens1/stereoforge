#!/usr/bin/env bash
# Run the temporal depth recorder with the Python interpreter from RoboStack/Pixi.
# Avoids accidentally using system or Miniforge Python under fish/bash.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
command -v pixi >/dev/null || { echo 'pixi is unavailable on PATH' >&2; exit 2; }
if [[ ! -d "$repo_root/ros_ws/.pixi/envs/default" ]]; then
    echo 'ROS Pixi environment missing. Run: pixi install -m ros_ws' >&2
    exit 2
fi
ros_prefix="$(realpath -e "$repo_root/ros_ws/.pixi/envs/default")"
ros_py="$ros_prefix/bin/python"
if [[ ! -x "$ros_py" ]]; then
    echo "ROS Pixi Python missing: $ros_py" >&2
    echo 'Run: pixi install -m ros_ws' >&2
    exit 2
fi
ros_lib="$ros_prefix/lib"
export LD_LIBRARY_PATH="$ros_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Do not inherit an unrelated Conda/Miniforge or sourced system-ROS Python.
unset PYTHONPATH PYTHONHOME
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}"
export ZENOH_SESSION_CONFIG_URI="${ZENOH_SESSION_CONFIG_URI:-$repo_root/ros_ws/zenoh_shm_session.json5}"
# Test import using the exact interpreter to be used for recording, without
# interfering with the user's selected interpreter for offline analysis.
pixi run -m ros_ws "$ros_py" -c 'import sys; import rclpy; from sensor_msgs.msg import Image; print("ROS recorder Python:", sys.executable)' || {
    echo 'ROS Python imports failed. Check: pixi run -m ros_ws python -c "import sys; print(sys.executable)"' >&2
    echo 'If necessary, add the explicit packages: pixi add -m ros_ws ros-jazzy-rclpy ros-jazzy-sensor-msgs' >&2
    exit 2
}
exec pixi run -m ros_ws "$ros_py" benchmarks/ros_depth_video_capture.py "$@"
