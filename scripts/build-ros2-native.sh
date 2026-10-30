#!/usr/bin/env bash
# Reproducible native GCC + Pixi ROS 2 build on Fedora/Linux.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
command -v pixi >/dev/null || { echo "pixi is required" >&2; exit 2; }
command -v /usr/bin/g++ >/dev/null || { echo "native /usr/bin/g++ is required" >&2; exit 2; }
pixi install -m ros_ws
pixi_lib="$(realpath ros_ws/.pixi/envs/default/lib)"
test -f "$pixi_lib/liblttng-ust.so" || { echo "Missing Pixi LTTng library: $pixi_lib/liblttng-ust.so" >&2; exit 2; }
test -f "$pixi_lib/liblttng-ust-common.so" || { echo "Missing Pixi LTTng common library" >&2; exit 2; }
build="${STEREOFORGE_ROS_BUILD_DIR:-build/ros-daxa-native}"
pixi run -m ros_ws env CC=/usr/bin/gcc CXX=/usr/bin/g++ \
 cmake -S . -B "$build" -G Ninja \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
 "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=bfd -L$pixi_lib" \
 -DSTEREOFORGE_DAXA=ON -DSTEREOFORGE_DAXA_STEREO=ON \
 -DSTEREOFORGE_DAXA_PREPROCESS=ON -DSTEREOFORGE_DAXA_DEPTH=ON \
 -DSTEREOFORGE_DAXA_REFINEMENT=ON -DSTEREOFORGE_NATIVE_CAPTURE=ON \
 -DSTEREOFORGE_NATIVE_CAPTURE_ZED=ON -DSTEREOFORGE_ROS2=ON \
 -DSTEREOFORGE_ROS2_PUBLISHER=ON
pixi run -m ros_ws cmake --build "$build" --target stereoforge_ros2_depth_publisher -j "${STEREOFORGE_BUILD_JOBS:-8}"
pub="$build/stereoforge_ros2_depth_publisher"
test -x "$pub" || { echo "Publisher was not generated: $pub" >&2; exit 1; }
# ROS/Conda dependency graph must be checked in its installed runtime.
# The ELF RUNPATH alone does not propagate to dependencies of dependencies.
if ! pixi run -m ros_ws env LD_LIBRARY_PATH="$pixi_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$pub" > "$build/ros-publisher-ldd.txt"; then
  echo "Runtime dependency inspection failed" >&2
  exit 1
fi
if grep -q 'not found' "$build/ros-publisher-ldd.txt"; then
  echo "Unresolved runtime libraries:" >&2
  grep 'not found' "$build/ros-publisher-ldd.txt" >&2
  exit 1
fi
"$pub" --help >/dev/null
echo "Publisher ready: $(realpath "$pub")"
echo "ROS 2 launch: pixi run -m ros_ws ros2 launch ros_ws/launch/stereoforge_publisher.launch.py ..."
