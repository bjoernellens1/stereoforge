# StereoForge ROS 2 workspace

The Pixi environment provides ROS 2 Jazzy, RViz, and the Linux OpenGL/Qt/X11
runtime libraries. Source it with `pixi shell -m ros_ws` or prefix commands
with `pixi run -m ros_ws`.

Install or refresh the locked environment from the repository root before
launching:

```bash
pixi install -m ros_ws
```

The launch files use the native publisher's supported command-line options and
default to the installed `rmw_zenoh_cpp` implementation. The publisher
requires a calibration manifest, clock map, frame ID, run ID, and new trace
path; launch does not invent defaults for those values.

RoboStack supplies the ROS Jazzy Zenoh RMW and its `rmw_zenohd` router through
`ros-jazzy-rmw-zenoh-cpp`. Start one router independently, then start the
publisher and RViz. Router and session configuration paths are optional launch
arguments; when supplied they are passed through as
`ZENOH_ROUTER_CONFIG_URI` and `ZENOH_SESSION_CONFIG_URI` without modifying the
existing JSON5 files.

The standard workflow starts the repository router helper, the built GPU
publisher, and RViz together. It uses `rmw_zenoh_cpp` and publishes the fixed
topics `/stereoforge/depth` and `/stereoforge/rectified_rgb`:

```bash
pixi run -m ros_ws ros2 launch \
  ros_ws/launch/stereoforge_standard.launch.py \
  manifest:=/absolute/path/to/calibration.json \
  clock_map:=/absolute/path/to/clock-map.json \
  frame_id:=stereoforge_camera \
  run_id:=local-camera \
  trace:=/tmp/stereoforge-trace.json
```

For the connected ZED2, the tracked official factory calibration source is
`configs/cameras/zed2/SN26041639.conf`. It must first be converted and paired
with a packed native capture using the workflow in
[`docs/camera-calibration.md`](../docs/camera-calibration.md); the `.conf` file
is not itself the `manifest:=` input. The generated Daxa fixture and measured
`clock-map.json` stay outside the repository.
`--device-id` only selects a video node and `--calibration-id` only supplies
an expected calibration identity. For this tracked ZED2 only, pass
`device_serial:=26041639` together with the matching calibration manifest; the
native publisher records the explicit serial override in its diagnostics and
rejects other serial/calibration combinations.

Use `source:=replay replay:=/absolute/path/to/replay` for a replay run.
Override `mode`, `stereo_variant`, `pipeline`, `coarse_manifest`,
`reliability`, or `rviz_config` as needed. The standard launch expects the
publisher at `/home/bjoern/git/stereoforge/build/ros-daxa/stereoforge_ros2_depth_publisher`;
rebuild that target if it is absent.

To use existing tuned Zenoh JSON5 files, add these launch arguments to the
standard command:

```text
zenoh_router_config:=/absolute/path/to/existing/router.json5
zenoh_session_config:=/absolute/path/to/existing/session.json5
```

Those paths are exported as `ZENOH_ROUTER_CONFIG_URI` and
`ZENOH_SESSION_CONFIG_URI`; the launch files do not create or modify config
files. The standard launch starts `scripts/stereoforge-ros router`, which uses
the installed `rmw_zenoh_cpp` package and `rmw_zenohd`.

```bash
pixi run -m ros_ws ros2 launch \
  /home/bjoern/git/stereoforge/ros_ws/launch/stereoforge_zenoh_router.launch.py \
  zenoh_router_config:=/absolute/path/to/existing/router.json5
```

```bash
pixi run -m ros_ws ros2 launch \
  /home/bjoern/git/stereoforge/ros_ws/launch/stereoforge_publisher.launch.py \
  manifest:=/absolute/path/to/calibration.json \
  clock_map:=/absolute/path/to/clock-map.json \
  frame_id:=stereoforge_camera \
  run_id:=local-camera \
  trace:=/tmp/stereoforge-trace.json \
  zenoh_router_config:=/absolute/path/to/existing/router.json5 \
  zenoh_session_config:=/absolute/path/to/existing/session.json5

pixi run -m ros_ws ros2 launch \
  /home/bjoern/git/stereoforge/ros_ws/launch/stereoforge_rviz.launch.py

pixi run -m ros_ws ros2 launch \
  /home/bjoern/git/stereoforge/ros_ws/launch/stereoforge_camera.launch.py \
  manifest:=/absolute/path/to/calibration.json \
  clock_map:=/absolute/path/to/clock-map.json \
  frame_id:=stereoforge_camera \
  run_id:=local-camera \
  trace:=/tmp/stereoforge-trace.json \
  zenoh_router_config:=/absolute/path/to/existing/router.json5 \
  zenoh_session_config:=/absolute/path/to/existing/session.json5
```

The launch-file and RViz paths above are real paths in this checkout. Replace
only the two `/absolute/path/to/...` values with paths to the actual calibration
manifest and clock-map files from your capture run; those files are not present
in this repository. The native ROS publisher is built at
`/home/bjoern/git/stereoforge/build/ros-daxa/stereoforge_ros2_depth_publisher`.
Live camera launch still requires the actual calibration manifest and clock-map
from the capture run; this repository does not fabricate those artifacts.

Set `rmw_implementation=rmw_fastrtps_cpp` explicitly for a DDS fallback; Zenoh
is intentionally the default. All communicating ROS processes must use
compatible middleware and config settings.
