"""Launch the standalone Zenoh router; router ownership stays outside applications."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _launch_router(context):
    environment = {}
    config = LaunchConfiguration("zenoh_router_config").perform(context)
    if config:
        environment["ZENOH_ROUTER_CONFIG_URI"] = config
    return [
        ExecuteProcess(
            cmd=["ros2", "run", "rmw_zenoh_cpp", "rmw_zenohd"],
            additional_env=environment,
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "zenoh_router_config",
                default_value="",
                description="Optional existing JSON5 router config path.",
            ),
            OpaqueFunction(function=_launch_router),
        ]
    )
