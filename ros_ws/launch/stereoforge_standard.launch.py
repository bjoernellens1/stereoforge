"""Launch the standard StereoForge Zenoh camera workflow."""

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
    rviz_env = dict(publisher_env)

    return [
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
        DeclareLaunchArgument("capture_fps", default_value="30"),
        DeclareLaunchArgument("depth_rate", default_value="30"),
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
            "disparities",
            default_value="128",
            description="Stereo disparity search range (e.g. 128 or 256).",
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
            "device_serial",
            default_value="",
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
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_standard)])
