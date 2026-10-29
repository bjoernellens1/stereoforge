"""Launch the StereoForge publisher and its RViz image view together."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_dir = Path(__file__).resolve().parent
    publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(launch_dir / "stereoforge_publisher.launch.py")
        ),
        launch_arguments={
            name: LaunchConfiguration(name)
            for name in (
                "publisher_executable",
                "rmw_implementation",
                "zenoh_router_config",
                "zenoh_session_config",
                "source",
                "manifest",
                "clock_map",
                "frame_id",
                "run_id",
                "trace",
                "replay",
                "mode",
                "duration_seconds",
                "subscriber_wait_seconds",
                "max_frames",
                "warmup",
                "disparities",
                "stereo_variant",
                "pipeline",
                "coarse_manifest",
                "device_id",
                "depth_topic",
                "rgb_topic",
                "reliability",
                "validation",
                "thread_cpu_timing",
            )
        }.items(),
    )
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(launch_dir / "stereoforge_rviz.launch.py")),
        launch_arguments={
            "rviz_config": LaunchConfiguration("rviz_config"),
            "rmw_implementation": LaunchConfiguration("rmw_implementation"),
            "zenoh_router_config": LaunchConfiguration("zenoh_router_config"),
            "zenoh_session_config": LaunchConfiguration("zenoh_session_config"),
        }.items(),
    )
    arguments = [
        DeclareLaunchArgument(
            "publisher_executable",
            default_value=str(
                Path(__file__).resolve().parents[2]
                / "build"
                / "daxa"
                / "stereoforge_ros2_depth_publisher"
            ),
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
        DeclareLaunchArgument("manifest", description="Required calibration manifest path."),
        DeclareLaunchArgument("clock_map", description="Required capture-to-ROS clock mapping path."),
        DeclareLaunchArgument("frame_id", description="Required ROS image frame ID."),
        DeclareLaunchArgument("run_id", description="Required unique delivery run ID."),
        DeclareLaunchArgument("trace", description="Required new JSON delivery trace path."),
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
            "replay",
            default_value="",
            description="Replay directory; required when source=replay.",
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
        DeclareLaunchArgument("depth_topic", default_value="/stereoforge/depth"),
        DeclareLaunchArgument("rgb_topic", default_value="/stereoforge/rectified_rgb"),
        DeclareLaunchArgument(
            "reliability",
            default_value="best_effort",
            choices=["best_effort", "reliable"],
        ),
        DeclareLaunchArgument("validation", default_value="false"),
        DeclareLaunchArgument("thread_cpu_timing", default_value="false"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=str(
                Path(__file__).resolve().parents[1] / "rviz" / "stereoforge.rviz"
            ),
        ),
    ]
    return LaunchDescription(arguments + [publisher, rviz])
