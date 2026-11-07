"""Launch the StereoForge publisher and its RViz image view together."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


_FORWARDED_ARGS = (
    "profile",
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
    "descriptor_mode",
    "fast_mode",
    "coarse_schedule",
    "coarse_right_reference_mode",
    "refinement_subpixel",
    "selective_repair",
    "planar_recovery",
    "completed_depth_topic",
    "completed_provenance_topic",
    "device_id",
    "device_serial",
    "depth_topic",
    "rgb_topic",
    "reliability",
    "depth_speckle_min_pixels",
    "depth_speckle_delta_m",
    "depth_speckle_delta_px",
    "validation",
    "thread_cpu_timing",
)


def generate_launch_description():
    launch_dir = Path(__file__).resolve().parent
    repo_root = launch_dir.parents[1]
    default_manifest = str(
        repo_root / "configs" / "cameras" / "zed2" / "fixtures" / "SN26041639-fixture" / "manifest.json"
    )
    default_clock_map = str(
        repo_root / "configs" / "cameras" / "zed2" / "clock_maps" / "SN26041639-clock-map.json"
    )
    publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(launch_dir / "stereoforge_publisher.launch.py")
        ),
        launch_arguments={
            name: LaunchConfiguration(name) for name in _FORWARDED_ARGS
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
        DeclareLaunchArgument("profile", default_value="", choices=["", "quality-reference", "live-quality"],
                              description="Fixed unqualified development recipe; empty preserves custom defaults."),
        DeclareLaunchArgument(
            "publisher_executable",
            default_value=str(
                repo_root
                / "build"
                / "ros-daxa-native"
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
            default_value="depth-rgb",
            choices=["depth", "rgb", "depth-rgb"],
        ),
        DeclareLaunchArgument("duration_seconds", default_value="600"),
        DeclareLaunchArgument("subscriber_wait_seconds", default_value="0"),
        DeclareLaunchArgument("max_frames", default_value="100000"),
        DeclareLaunchArgument("warmup", default_value="5"),
        DeclareLaunchArgument("disparities", default_value=""),
        DeclareLaunchArgument(
            "stereo_variant",
            default_value="",
            choices=["", "packed-specialized", "packed-specialized-overwrite", "packed-specialized-overwrite-quad"],
        ),
        DeclareLaunchArgument(
            "pipeline",
            default_value="",
            choices=["", "full-sgm", "experimental-refinement", "experimental-dual-refinement"],
        ),
        DeclareLaunchArgument(
            "coarse_manifest",
            default_value="",
            description="Required only for an experimental refinement pipeline.",
        ),
        DeclareLaunchArgument(
            "descriptor_mode",
            default_value="",
            choices=["", "census", "ortho-hybrid"],
        ),
        DeclareLaunchArgument("fast_mode", default_value=""),
        DeclareLaunchArgument(
            "coarse_schedule",
            default_value="",
            choices=["", "sequential", "paired-per-path", "layered-per-path", "shared-census-sequential"],
        ),
        DeclareLaunchArgument(
            "coarse_right_reference_mode",
            default_value="",
            choices=["", "independent-sgm", "left-aggregated-selection"],
        ),
        DeclareLaunchArgument("refinement_subpixel", default_value=""),
        DeclareLaunchArgument("selective_repair", default_value=""),
        DeclareLaunchArgument("planar_recovery", default_value=""),
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
        DeclareLaunchArgument("rgb_topic", default_value="/stereoforge/rectified_rgb"),
        DeclareLaunchArgument(
            "reliability",
            default_value="best_effort",
            choices=["best_effort", "reliable"],
        ),
        DeclareLaunchArgument("depth_speckle_min_pixels", default_value=""),
        DeclareLaunchArgument("depth_speckle_delta_m", default_value=""),
        DeclareLaunchArgument("depth_speckle_delta_px", default_value=""),
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
