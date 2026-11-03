"""Launch RViz with the StereoForge image view."""

import glob
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def resolve_x11_display_env() -> dict[str, str]:
    """Ensure DISPLAY and XAUTHORITY point to the active desktop X11 session."""
    env: dict[str, str] = {}
    env["DISPLAY"] = os.environ.get("DISPLAY") or ":0"
    xauth = os.environ.get("XAUTHORITY", "")
    if not xauth or not Path(xauth).exists():
        candidates = sorted(glob.glob(f"/run/user/{os.getuid()}/xauth_*"))
        if candidates:
            xauth = candidates[0]
    if xauth:
        env["XAUTHORITY"] = xauth
    return env


def generate_launch_description():
    default_config = str(
        Path(__file__).resolve().parents[1] / "rviz" / "stereoforge.rviz"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_config,
                description="RViz configuration file.",
            ),
            DeclareLaunchArgument(
                "rmw_implementation",
                default_value="rmw_zenoh_cpp",
                description="Installed ROS middleware implementation; override for DDS fallback.",
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
            OpaqueFunction(function=_launch_rviz),
        ]
    )


def _launch_rviz(context):
    environment = {
        "RMW_IMPLEMENTATION": LaunchConfiguration("rmw_implementation").perform(context),
        **resolve_x11_display_env(),
    }
    router_config = LaunchConfiguration("zenoh_router_config").perform(context)
    session_config = LaunchConfiguration("zenoh_session_config").perform(context)
    if router_config:
        environment["ZENOH_ROUTER_CONFIG_URI"] = router_config
    if session_config:
        environment["ZENOH_SESSION_CONFIG_URI"] = session_config
    return [
        ExecuteProcess(
            cmd=["rviz2", "-d", LaunchConfiguration("rviz_config")],
            additional_env=environment,
            output="screen",
        )
    ]
