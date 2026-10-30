# StereoForge ROS 2 workspace

The Pixi environment provides ROS 2 Jazzy, RViz, and the Linux OpenGL/Qt/X11
runtime libraries. Source it with `pixi shell -m ros_ws` or prefix commands
with `pixi run -m ros_ws`.

Build the native publisher on Fedora using the repository entry point. This
intentionally uses system GCC/G++ while resolving ROS 2 packages from Pixi.
The command also adds Pixi's LTTng libraries to the linker search path, which
is required when the native linker is used. It verifies the executable and
checks that all runtime shared libraries resolve under the Pixi runtime loader
path. Running `ldd` without Pixi's library path can misleadingly report
transitive ROS dependencies as `not found` even after linking succeeds.

```bash
bash scripts/build-ros2-native.sh
```

The default output is `build/ros-daxa-native/stereoforge_ros2_depth_publisher`.
To override the build directory, set `STEREOFORGE_ROS_BUILD_DIR=...` and
pass the corresponding `publisher_executable` to the publisher-only launch.
Do not configure this target directly with `pixi run cmake` and its implicit
Conda compiler: that can link Fedora's Vulkan/HIDAPI/X11 libraries against an
incompatible Conda sysroot. The helper uses `/usr/bin/gcc`,
`/usr/bin/g++`, and `-fuse-ld=bfd -L<resolved-pixi-lib>`. Changing a
compiler for an existing CMake cache requires a fresh build directory.

On other Linux distributions, check compatibility before reusing this
Fedora-specific native-toolchain build entry point. This is a validated
configuration from the local Fedora 44 linking failure, not a promise that
all systems supply compatible host/Conda system libraries.

For a quick standalone filter regression without ROS or Vulkan:
```bash
cmake --preset cpu
cmake --build --preset cpu -j 8
ctest --preset cpu -R depth_speckle_contracts --output-on-failure
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
`reliability`, `rviz_config`, `depth_speckle_min_pixels`, and
`depth_speckle_delta_m` as needed. Both standard and publisher-only launch
default to the executable in `build/ros-daxa-native` **of the checked-out repository**.
For an isolated build, supply `publisher_executable:=/absolute/path/to/executable`
with the publisher-only launch; standard launch requires the default path.
The speckle filter is disabled by default (`depth_speckle_min_pixels:=0`).
Example: `depth_speckle_min_pixels:=50 depth_speckle_delta_m:=0.05`.
This filters only the published 32FC1 depth, setting small disconnected
clusters to NaN; it does not change GPU stereo estimates or recover missing
geometry.

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
`build/ros-daxa-native/stereoforge_ros2_depth_publisher` relative to this checkout.
Live camera launch still requires the actual calibration manifest and clock-map
from the capture run; this repository does not fabricate those artifacts.

Set `rmw_implementation=rmw_fastrtps_cpp` explicitly for a DDS fallback; Zenoh
is intentionally the default. All communicating ROS processes must use
compatible middleware and config settings.

## Fedora 44 / Strix Halo Vulkan runtime

The standard ROS launch uses the same native Mesa/RADV device and host DRM
libraries for both RViz **and** the Daxa publisher. It sets
`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`,
`LIBGL_DRIVERS_PATH=/usr/lib64/dri`, and preloads the host
`libdrm.so.2` and `libdrm_amdgpu.so.1`. These settings are deliberately
specific to the verified Fedora workstation; check the paths before using
this launch on another distribution. Zenoh router startup remains separate;
clients retry discovery with `ZENOH_ROUTER_CHECK_ATTEMPTS=20`.

To diagnose GPU availability without ROS, compare the native and Pixi
`vulkaninfo --summary` output and run the built `daxa_probe`. An AMD GPU in
`vulkaninfo` is necessary but does not establish that the Daxa device was
successfully created. `DaxaDepthConfig::required_wave_size` already defaults
to 32; no shader changes are needed for that setting. When benchmarking FPS,
stop competing GPU compute jobs and check utilization. The Zenoh session and
router JSON5 configurations in `ros_ws/` explicitly disable shared memory,
so no system-wide `ulimit -l` changes are needed for that mode.
