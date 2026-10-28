"""ROS defaults are process-local, explicit, and independent of a ROS install."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "scripts/stereoforge-ros"
APPS = ("ros2_depth_publisher.cpp", "ros2_gpu_composed.cpp", "ros2_delivery_probe.cpp")


@pytest.fixture
def fake_ros(tmp_path):
    ros = tmp_path / "ros2"
    ros.write_text('#!/bin/sh\n'
                   'if [ "$1" = pkg ] && [ "$2" = prefix ]; then\n'
                   '  [ "$3" != missing_rmw ]; exit $?\n'
                   'fi\n'
                   'printf "router:%s:%s\\n" "$2" "$3"\n')
    ros.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ["PATH"])
    env.pop("RMW_IMPLEMENTATION", None)
    return env


def launch(env, *args):
    return subprocess.run([str(LAUNCH), *args], env=env, text=True, capture_output=True)


def test_launcher_default_and_overrides(fake_ros):
    command = ("--", "/bin/sh", "-c", 'printf %s "$RMW_IMPLEMENTATION"')
    assert launch(fake_ros, "run", *command).stdout == "rmw_zenoh_cpp"
    fake_ros["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    assert launch(fake_ros, "run", *command).stdout == "rmw_fastrtps_cpp"
    assert launch(fake_ros, "run", "--rmw", "rmw_other_cpp", *command).stdout == "rmw_other_cpp"


def test_launcher_fails_closed_and_keeps_router_foreground(fake_ros):
    failed = launch(fake_ros, "run", "--rmw", "missing_rmw", "--", "/bin/true")
    assert failed.returncode == 1 and "unavailable" in failed.stderr
    fake_ros["RMW_IMPLEMENTATION"] = ""
    assert launch(fake_ros, "run", "--", "/bin/true").returncode == 1
    router = launch(fake_ros, "router")
    assert router.returncode == 0 and router.stdout == "router:rmw_zenoh_cpp:rmw_zenohd\n"


def test_native_default_and_explicit_environment(tmp_path):
    cpp = tmp_path / "test.cpp"
    cpp.write_text('#include <stereoforge/ros_default_rmw.hpp>\n'
                   '#include <iostream>\n'
                   'int main() { stereoforge::set_default_ros_rmw(); '
                   'std::cout << std::getenv("RMW_IMPLEMENTATION"); }\n')
    binary = tmp_path / "test"
    subprocess.run(["c++", "-std=c++17", "-I", str(ROOT / "include"), str(cpp), "-o", str(binary)], check=True)
    env = dict(os.environ)
    env.pop("RMW_IMPLEMENTATION", None)
    assert subprocess.check_output([str(binary)], env=env, text=True) == "rmw_zenoh_cpp"
    env["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    assert subprocess.check_output([str(binary)], env=env, text=True) == "rmw_fastrtps_cpp"
    env["RMW_IMPLEMENTATION"] = ""
    result = subprocess.run([str(binary)], env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "RMW_IMPLEMENTATION is empty" in result.stderr


def test_all_native_entrypoints_set_default_before_ros_init():
    for app in APPS:
        source = (ROOT / "apps" / app).read_text()
        assert source.count("stereoforge::set_default_ros_rmw();") == 1
        assert source.index("stereoforge::set_default_ros_rmw();") < source.index("rclcpp::init(")
