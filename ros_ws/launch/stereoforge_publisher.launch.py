"""Launch the native StereoForge ROS 2 image publisher."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


_OPTIONAL_VALUE_ARGUMENTS = (
    ("mode", "--mode"),
    ("duration_seconds", "--duration-seconds"),
    ("subscriber_wait_seconds", "--subscriber-wait-seconds"),
    ("max_frames", "--max-frames"),
    ("warmup", "--warmup"),
    ("disparities", "--disparities"),
    ("stereo_variant", "--stereo-variant"),
    ("pipeline", "--pipeline"),
    ("coarse_manifest", "--coarse-manifest"),
    ("device_id", "--device-id"),
    ("device_serial", "--device-serial"),
    ("depth_topic", "--depth-topic"),
    ("rgb_topic", "--rgb-topic"),
    ("reliability", "--reliability"),
)


def _as_bool(value):
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _launch_publisher(context):
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

    environment = {"RMW_IMPLEMENTATION": values["rmw_implementation"]}
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
    default_publisher = str(
        Path(__file__).resolve().parents[2]
        / "build"
        / "daxa"
        / "stereoforge_ros2_depth_publisher"
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
            description="Required calibration manifest path.",
        ),
        DeclareLaunchArgument(
            "clock_map",
            description="Required capture-to-ROS clock mapping path.",
        ),
        DeclareLaunchArgument(
            "frame_id",
            description="Required ROS image frame ID.",
        ),
        DeclareLaunchArgument(
            "run_id",
            description="Required unique delivery run ID.",
        ),
        DeclareLaunchArgument(
            "trace",
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
            default_value="depth",
            choices=["depth", "rgb", "depth-rgb"],
        ),
        DeclareLaunchArgument("duration_seconds", default_value="600"),
        DeclareLaunchArgument("subscriber_wait_seconds", default_value="0"),
        DeclareLaunchArgument("max_frames", default_value="100000"),
        DeclareLaunchArgument("warmup", default_value="5"),
        DeclareLaunchArgument("disparities", default_value="128"),
        DeclareLaunchArgument(
            "stereo_variant",
            default_value="packed-specialized",
            choices=["packed-specialized", "packed-specialized-overwrite"],
        ),
        DeclareLaunchArgument(
            "pipeline",
            default_value="full-sgm",
            choices=["full-sgm", "experimental-refinement", "experimental-dual-refinement"],
        ),
        DeclareLaunchArgument(
            "coarse_manifest",
            default_value="",
            description="Required only for an experimental refinement pipeline.",
        ),
        DeclareLaunchArgument("device_id", default_value="-1"),
        DeclareLaunchArgument("device_serial", default_value=""),
        DeclareLaunchArgument("depth_topic", default_value="/stereoforge/depth"),
        DeclareLaunchArgument(
            "rgb_topic", default_value="/stereoforge/rectified_rgb"
        ),
        DeclareLaunchArgument(
            "reliability",
            default_value="best_effort",
            choices=["best_effort", "reliable"],
        ),
        DeclareLaunchArgument("validation", default_value="false"),
        DeclareLaunchArgument("thread_cpu_timing", default_value="false"),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_publisher)])
