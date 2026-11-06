"""Launch the native StereoForge ROS 2 image publisher."""

import os
import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


_OPTIONAL_VALUE_ARGUMENTS = (
    ("mode", "--mode"),
    ("cloud", "--cloud"),
    ("cloud_stride", "--cloud-stride"),
    ("cloud_rate", "--cloud-rate"),
    ("cloud_topic", "--cloud-topic"),
    ("diagnostics", "--diagnostics"),
    ("diagnostic_prefix", "--diagnostic-prefix"),
    ("rig_session_id", "--rig-session-id"),
    ("duration_seconds", "--duration-seconds"),
    ("subscriber_wait_seconds", "--subscriber-wait-seconds"),
    ("max_frames", "--max-frames"),
    ("warmup", "--warmup"),
    ("gpu_timing_report", "--gpu-timing-report"),
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
    ("device_id", "--device-id"),
    ("device_serial", "--device-serial"),
    ("depth_topic", "--depth-topic"),
    ("rgb_topic", "--rgb-topic"),
    ("reliability", "--reliability"),
    ("depth_speckle_min_pixels", "--depth-speckle-min-pixels"),
    ("depth_speckle_delta_m", "--depth-speckle-delta-m"),
    ("depth_speckle_delta_px", "--depth-speckle-delta-px"),
)


def _as_bool(value):
    return value.strip().lower() in {"1", "true", "yes", "on"}


def ensure_zed2_awake_and_prepare_trace(repo_root: Path, source: str, trace_path_str: str) -> None:
    """Wake ZED 2 from USB autosuspend if source=camera, and remove stale /tmp trace files."""
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


def _launch_publisher(context):
    repo_root = Path(__file__).resolve().parents[2]
    values = {
        name: LaunchConfiguration(name).perform(context)
        for name in (
            "publisher_executable",
            "source",
            "manifest",
            "clock_map",
            "frame_id",
            "run_id",
            "trace",
            "replay",
            *(name for name, _ in _OPTIONAL_VALUE_ARGUMENTS),
            "validation",
            "thread_cpu_timing",
            "rmw_implementation",
            "zenoh_router_config",
            "zenoh_session_config",
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
    if not values["source"] or values["source"] not in {"camera", "replay"}:
        missing.append("source=camera|replay")
    if values["source"] == "replay" and not values["replay"]:
        missing.append("replay")
    if missing:
        raise RuntimeError(
            "stereoforge_publisher requires launch arguments: " + ", ".join(missing)
        )

    ensure_zed2_awake_and_prepare_trace(repo_root, values["source"], values["trace"])

    command = [
        values["publisher_executable"],
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
    ]
    if values["source"] == "replay":
        command.extend(["--replay", values["replay"]])
    for name, option in _OPTIONAL_VALUE_ARGUMENTS:
        if values[name]:
            command.extend([option, values[name]])
    if _as_bool(values["validation"]):
        command.append("--validation")
    if _as_bool(values["thread_cpu_timing"]):
        command.append("--thread-cpu-timing")

    environment = {
        "RMW_IMPLEMENTATION": values["rmw_implementation"],
        "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
        "LIBGL_DRIVERS_PATH": "/usr/lib64/dri",
        "LD_PRELOAD": "/usr/lib64/libdrm.so.2:/usr/lib64/libdrm_amdgpu.so.1",
    }
    if values["zenoh_router_config"]:
        environment["ZENOH_ROUTER_CONFIG_URI"] = values["zenoh_router_config"]
    if values["zenoh_session_config"]:
        environment["ZENOH_SESSION_CONFIG_URI"] = values["zenoh_session_config"]
    return [
        ExecuteProcess(
            cmd=command,
            additional_env=environment,
            output="screen",
        )
    ]


def generate_launch_description():
    repo_root = Path(__file__).resolve().parents[2]
    default_publisher = str(
        repo_root
        / "build"
        / "ros-daxa-native"
        / "stereoforge_ros2_depth_publisher"
    )
    default_manifest = str(
        repo_root / "configs" / "cameras" / "zed2" / "fixtures" / "SN26041639-fixture" / "manifest.json"
    )
    default_clock_map = str(
        repo_root / "configs" / "cameras" / "zed2" / "clock_maps" / "SN26041639-clock-map.json"
    )
    arguments = [
        DeclareLaunchArgument(
            "publisher_executable",
            default_value=default_publisher,
            description="Path to the built native StereoForge ROS 2 publisher.",
        ),
        DeclareLaunchArgument(
            "rmw_implementation",
            default_value="rmw_zenoh_cpp",
            description="Installed ROS middleware implementation; override for DDS fallback.",
        ),
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
            "mode",
            default_value="depth-rgb",
            choices=["depth", "rgb", "depth-rgb"],
        ),
        DeclareLaunchArgument("cloud", default_value="0", choices=["0", "1"]),
        DeclareLaunchArgument("cloud_stride", default_value="4"),
        DeclareLaunchArgument("cloud_rate", default_value="15"),
        DeclareLaunchArgument("cloud_topic", default_value="/stereoforge/points"),
        DeclareLaunchArgument("gpu_timing_report", default_value=""),
        DeclareLaunchArgument("diagnostics", default_value="0", choices=["0", "1"]),
        DeclareLaunchArgument("diagnostic_prefix", default_value="/stereoforge/diagnostics"),
        DeclareLaunchArgument("rig_session_id", default_value=""),
        DeclareLaunchArgument("duration_seconds", default_value="600"),
        DeclareLaunchArgument("subscriber_wait_seconds", default_value="0"),
        DeclareLaunchArgument("max_frames", default_value="100000"),
        DeclareLaunchArgument("warmup", default_value="5"),
        DeclareLaunchArgument("disparities", default_value="128"),
        DeclareLaunchArgument(
            "stereo_variant",
            default_value="packed-specialized",
            choices=["packed-specialized", "packed-specialized-overwrite", "packed-specialized-overwrite-quad"],
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
        DeclareLaunchArgument("device_id", default_value="-1"),
        DeclareLaunchArgument("device_serial", default_value="26041639"),
        DeclareLaunchArgument("depth_topic", default_value="/stereoforge/depth"),
        DeclareLaunchArgument(
            "rgb_topic", default_value="/stereoforge/rectified_rgb"
        ),
        DeclareLaunchArgument(
            "reliability",
            default_value="best_effort",
            choices=["best_effort", "reliable"],
        ),
        DeclareLaunchArgument("depth_speckle_min_pixels", default_value="0"),
        DeclareLaunchArgument("depth_speckle_delta_m", default_value="0.05"),
        DeclareLaunchArgument("depth_speckle_delta_px", default_value="1.25"),
        DeclareLaunchArgument("validation", default_value="false"),
        DeclareLaunchArgument("thread_cpu_timing", default_value="false"),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_publisher)])
