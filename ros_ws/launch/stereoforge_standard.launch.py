"""Launch the standard StereoForge Zenoh camera workflow."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


_OPTIONAL_VALUE_ARGUMENTS = (
    ("mode", "--mode"),
    ("device_serial", "--device-serial"),
    ("capture_fps", "--capture-fps"),
    ("depth_rate", "--depth-rate"),
    ("disparities", "--disparities"),
    ("stereo_variant", "--stereo-variant"),
    ("pipeline", "--pipeline"),
    ("coarse_manifest", "--coarse-manifest"),
    ("reliability", "--reliability"),
    ("depth_speckle_min_pixels", "--depth-speckle-min-pixels"),
    ("depth_speckle_delta_m", "--depth-speckle-delta-m"),
)


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
        )
    }

    required = ("manifest", "clock_map", "frame_id", "run_id", "trace")
    missing = [name for name in required if not values[name]]
    if values["source"] == "replay" and not values["replay"]:
        missing.append("replay")
    if missing:
        raise RuntimeError(
            "stereoforge_standard requires launch arguments: " + ", ".join(missing)
        )

    publisher = [
        str(repo_root / "build" / "ros-daxa" / "stereoforge_ros2_depth_publisher"),
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

    environment = {"RMW_IMPLEMENTATION": "rmw_zenoh_cpp"}
    if values["zenoh_router_config"]:
        environment["ZENOH_ROUTER_CONFIG_URI"] = values["zenoh_router_config"]
    if values["zenoh_session_config"]:
        environment["ZENOH_SESSION_CONFIG_URI"] = values["zenoh_session_config"]
    return [
        ExecuteProcess(
            cmd=[str(repo_root / "scripts" / "stereoforge-ros"), "router"],
            additional_env=environment,
            output="screen",
        ),
        ExecuteProcess(
            cmd=publisher,
            additional_env=environment,
            output="screen",
        ),
        ExecuteProcess(
            cmd=["rviz2", "-d", values["rviz_config"]],
            additional_env=environment,
            output="screen",
        ),
    ]


def generate_launch_description():
    repo_root = Path(__file__).resolve().parents[2]
    arguments = [
        DeclareLaunchArgument(
            "source",
            default_value="camera",
            choices=["camera", "replay"],
            description="Capture source. Use replay only with replay=DIR.",
        ),
        DeclareLaunchArgument("manifest", description="Required calibration manifest path."),
        DeclareLaunchArgument(
            "clock_map",
            description="Required capture-to-ROS clock mapping path.",
        ),
        DeclareLaunchArgument("frame_id", description="Required ROS image frame ID."),
        DeclareLaunchArgument("run_id", description="Required unique delivery run ID."),
        DeclareLaunchArgument("trace", description="Required new JSON delivery trace path."),
        DeclareLaunchArgument(
            "replay",
            default_value="",
            description="Replay directory; required when source=replay.",
        ),
        DeclareLaunchArgument("device_serial", default_value=""),
        DeclareLaunchArgument("capture_fps", default_value="30"),
        DeclareLaunchArgument("depth_rate", default_value="30"),
        DeclareLaunchArgument("disparities", default_value="128"),
        DeclareLaunchArgument(
            "mode",
            default_value="depth",
            choices=["depth", "rgb", "depth-rgb"],
        ),
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
        DeclareLaunchArgument("depth_speckle_min_pixels", default_value="0"),
        DeclareLaunchArgument("depth_speckle_delta_m", default_value="0.05"),
        DeclareLaunchArgument(
            "reliability",
            default_value="best_effort",
            choices=["best_effort", "reliable"],
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
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_standard)])
