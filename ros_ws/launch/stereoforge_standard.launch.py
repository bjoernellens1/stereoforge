"""Launch the standard StereoForge Zenoh camera workflow (with optional RealSense D435i bridge)."""

import glob
import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


_OPTIONAL_VALUE_ARGUMENTS = (
    ("mode", "--mode"),
    ("device_serial", "--device-serial"),
    ("device_id", "--device-id"),
    ("capture_fps", "--capture-fps"),
    ("depth_rate", "--depth-rate"),
    ("disparities", "--disparities"),
    ("stereo_variant", "--stereo-variant"),
    ("pipeline", "--pipeline"),
    ("coarse_manifest", "--coarse-manifest"),
    ("descriptor_mode", "--descriptor-mode"),
    ("fast_mode", "--fast-mode"),
    ("ultra_fast", "--ultra-fast"),
    ("coarse_schedule", "--coarse-schedule"),
    ("coarse_right_reference_mode", "--coarse-right-reference-mode"),
    ("refinement_subpixel", "--refinement-subpixel"),
    ("continuous_subpixel", "--continuous-subpixel"),
    ("adaptive_tiles", "--adaptive-tiles"),
    ("min_disparity_bound", "--min-disparity-bound"),
    ("max_disparity_bound", "--max-disparity-bound"),
    ("selective_repair", "--selective-repair"),
    ("planar_recovery", "--planar-recovery"),
    ("completed_depth_topic", "--completed-depth-topic"),
    ("completed_provenance_topic", "--completed-provenance-topic"),
    ("reliability", "--reliability"),
    ("depth_speckle_min_pixels", "--depth-speckle-min-pixels"),
    ("depth_speckle_delta_m", "--depth-speckle-delta-m"),
    ("depth_speckle_delta_px", "--depth-speckle-delta-px"),
)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_x11_display_env() -> dict[str, str]:
    env: dict[str, str] = {"DISPLAY": os.environ.get("DISPLAY") or ":0"}
    xauth = os.environ.get("XAUTHORITY", "")
    if not xauth or not Path(xauth).exists():
        candidates = sorted(glob.glob(f"/run/user/{os.getuid()}/xauth_*"))
        if candidates:
            xauth = candidates[0]
    if xauth:
        env["XAUTHORITY"] = xauth
    return env


def _prepare_camera_and_trace(repo_root: Path, source: str, trace_path_str: str) -> None:
    trace_path = Path(trace_path_str)
    try:
        if trace_path.exists() and trace_path.resolve().is_relative_to(Path("/tmp")):
            trace_path.unlink()
    except OSError:
        pass

    if source == "camera":
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        try:
            from benchmarks.live_and_e2e_qualification_v9 import ensure_zed2_camera_active

            ensure_zed2_camera_active(force_reset=False)
        except Exception:
            pass


def _launch_standard(context):
    repo_root = Path(__file__).resolve().parents[2]
    values = {
        name: LaunchConfiguration(name).perform(context)
        for name in (
            "manifest",
            "clock_map",
            "frame_id",
            "run_id",
            "trace",
            "source",
            "replay",
            *(name for name, _ in _OPTIONAL_VALUE_ARGUMENTS),
            "zenoh_router_config",
            "zenoh_session_config",
            "rviz_config",
            "with_realsense",
        )
    }
    if values["pipeline"] != "full-sgm" and not values["coarse_manifest"]:
        default_coarse = (
            repo_root
            / "artifacts"
            / "depth-video"
            / "v8-review-milestone"
            / "fixtures"
            / "zed2_320x180"
            / "manifest.json"
        )
        if default_coarse.exists():
            values["coarse_manifest"] = str(default_coarse)

    required = ("manifest", "clock_map", "frame_id", "run_id", "trace")
    missing = [name for name in required if not values[name]]
    if values["source"] == "replay" and not values["replay"]:
        missing.append("replay")
    if missing:
        raise RuntimeError(
            "stereoforge_standard requires launch arguments: " + ", ".join(missing)
        )

    _prepare_camera_and_trace(repo_root, values["source"], values["trace"])

    publisher = [
        str(repo_root / "build" / "ros-daxa-native" / "stereoforge_ros2_depth_publisher"),
        "--source",
        values["source"],
        "--manifest",
        values["manifest"],
        "--clock-map",
        values["clock_map"],
        "--frame-id",
        values["frame_id"],
        "--run-id",
        values["run_id"],
        "--trace",
        values["trace"],
        "--depth-topic",
        "/stereoforge/depth",
        "--rgb-topic",
        "/stereoforge/rectified_rgb",
    ]
    if values["source"] == "replay":
        publisher.extend(["--replay", values["replay"]])
    for name, option in _OPTIONAL_VALUE_ARGUMENTS:
        if values[name]:
            publisher.extend([option, values[name]])

    publisher_env = {
        "RMW_IMPLEMENTATION": "rmw_zenoh_cpp",
        "ZENOH_SESSION_CONFIG_URI": str(repo_root / "ros_ws" / "zenoh_shm_session.json5"),
        "ZENOH_ROUTER_CONFIG_URI": str(repo_root / "ros_ws" / "zenoh_shm_router.json5"),
    }
    if values["zenoh_router_config"]:
        publisher_env["ZENOH_ROUTER_CONFIG_URI"] = values["zenoh_router_config"]
    if values["zenoh_session_config"]:
        publisher_env["ZENOH_SESSION_CONFIG_URI"] = values["zenoh_session_config"]

    # Fedora 44 / Strix Halo: use the host Mesa/DRM stack for BOTH GPU
    # processes. Pixi's bundled libdrm can shadow the system driver.
    publisher_env["VK_DRIVER_FILES"] = "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
    publisher_env["LIBGL_DRIVERS_PATH"] = "/usr/lib64/dri"
    publisher_env["LD_PRELOAD"] = "/usr/lib64/libdrm.so.2:/usr/lib64/libdrm_amdgpu.so.1"
    publisher_env["ZENOH_ROUTER_CHECK_ATTEMPTS"] = "20"
    rviz_env = {**publisher_env, **_resolve_x11_display_env()}

    processes = [
        ExecuteProcess(
            cmd=[str(repo_root / "scripts" / "stereoforge-ros"), "router"],
            additional_env=publisher_env,
            output="screen",
        ),
        ExecuteProcess(
            cmd=publisher,
            additional_env=publisher_env,
            output="screen",
        ),
        ExecuteProcess(
            cmd=["rviz2", "-d", values["rviz_config"]],
            additional_env=rviz_env,
            output="screen",
        ),
    ]
    if _as_bool(values["with_realsense"]):
        processes.append(
            ExecuteProcess(
                cmd=[sys.executable, str(repo_root / "scripts" / "realsense_zed2_ros2_bridge.py")],
                additional_env=publisher_env,
                output="screen",
            )
        )
    return processes


def generate_launch_description():
    repo_root = Path(__file__).resolve().parents[2]
    default_manifest = str(
        repo_root / "configs" / "cameras" / "zed2" / "fixtures" / "SN26041639-fixture" / "manifest.json"
    )
    default_clock_map = str(
        repo_root / "configs" / "cameras" / "zed2" / "clock_maps" / "SN26041639-clock-map.json"
    )
    arguments = [
        DeclareLaunchArgument(
            "source",
            default_value="camera",
            choices=["camera", "replay"],
            description="Capture source. Use replay only with replay=DIR.",
        ),
        DeclareLaunchArgument(
            "manifest",
            default_value=default_manifest if Path(default_manifest).exists() else "",
            description="Required calibration manifest path.",
        ),
        DeclareLaunchArgument(
            "clock_map",
            default_value=default_clock_map if Path(default_clock_map).exists() else "",
            description="Required capture-to-ROS clock mapping path.",
        ),
        DeclareLaunchArgument(
            "frame_id",
            default_value="stereoforge_camera",
            description="Required ROS image frame ID.",
        ),
        DeclareLaunchArgument(
            "run_id",
            default_value="live-zed2",
            description="Required unique delivery run ID.",
        ),
        DeclareLaunchArgument(
            "trace",
            default_value="/tmp/stereoforge-trace.json",
            description="Required new JSON delivery trace path.",
        ),
        DeclareLaunchArgument(
            "replay",
            default_value="",
            description="Replay directory; required when source=replay.",
        ),
        DeclareLaunchArgument("capture_fps", default_value="60"),
        DeclareLaunchArgument("depth_rate", default_value="adaptive"),
        DeclareLaunchArgument(
            "mode",
            default_value="depth-rgb",
            choices=["depth", "rgb", "depth-rgb"],
        ),
        DeclareLaunchArgument(
            "stereo_variant",
            default_value="packed-specialized",
            choices=["packed-specialized", "packed-specialized-overwrite", "packed-specialized-overwrite-quad"],
        ),
        DeclareLaunchArgument(
            "disparities",
            default_value="128",
            description="Stereo disparity search range (e.g. 128 or 256).",
        ),
        DeclareLaunchArgument(
            "pipeline",
            default_value="experimental-dual-refinement",
            choices=["full-sgm", "experimental-refinement", "experimental-dual-refinement"],
        ),
        DeclareLaunchArgument(
            "coarse_manifest",
            default_value="",
            description="Required only for an experimental refinement pipeline.",
        ),
        DeclareLaunchArgument(
            "descriptor_mode",
            default_value="ortho-hybrid",
            choices=["", "census", "ortho-hybrid"],
        ),
        DeclareLaunchArgument("fast_mode", default_value="1"),
        DeclareLaunchArgument("ultra_fast", default_value="1"),
        DeclareLaunchArgument(
            "coarse_schedule",
            default_value="shared-census-sequential",
            choices=["", "sequential", "paired-per-path", "layered-per-path", "shared-census-sequential"],
        ),
        DeclareLaunchArgument(
            "coarse_right_reference_mode",
            default_value="left-aggregated-selection",
            choices=["", "independent-sgm", "left-aggregated-selection"],
        ),
        DeclareLaunchArgument("refinement_subpixel", default_value="1"),
        DeclareLaunchArgument("continuous_subpixel", default_value="0"),
        DeclareLaunchArgument("adaptive_tiles", default_value="1"),
        DeclareLaunchArgument("min_disparity_bound", default_value="0"),
        DeclareLaunchArgument("max_disparity_bound", default_value="0"),
        DeclareLaunchArgument("selective_repair", default_value="1"),
        DeclareLaunchArgument("planar_recovery", default_value="1"),
        DeclareLaunchArgument(
            "completed_depth_topic", default_value="/stereoforge/depth_completed"
        ),
        DeclareLaunchArgument(
            "completed_provenance_topic",
            default_value="/stereoforge/depth_completed_provenance",
        ),
        DeclareLaunchArgument("depth_speckle_min_pixels", default_value="0"),
        DeclareLaunchArgument("depth_speckle_delta_m", default_value="0.05"),
        DeclareLaunchArgument("depth_speckle_delta_px", default_value="1.25"),
        DeclareLaunchArgument(
            "reliability",
            default_value="best_effort",
            choices=["best_effort", "reliable"],
        ),
        DeclareLaunchArgument(
            "device_serial",
            default_value="26041639",
            description="Explicit camera serial override (e.g. 26041639).",
        ),
        DeclareLaunchArgument(
            "device_id",
            default_value="-1",
            description="Camera video device ID (-1 to scan).",
        ),
        DeclareLaunchArgument(
            "zenoh_router_config",
            default_value="",
            description="Optional existing JSON5 router config path.",
        ),
        DeclareLaunchArgument(
            "zenoh_session_config",
            default_value="",
            description="Optional existing JSON5 session config path.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(repo_root / "ros_ws" / "rviz" / "stereoforge.rviz"),
            description="RViz display configuration.",
        ),
        DeclareLaunchArgument(
            "with_realsense",
            default_value="true",
            description="Launch RealSense D435i + ZED 2 colorized depth bridge alongside StereoForge.",
        ),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_standard)])
