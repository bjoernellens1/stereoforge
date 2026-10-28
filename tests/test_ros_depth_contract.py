"""Compile and run the real ROS-independent C++ metadata contract."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_ros_depth_contract(tmp_path):
    binary = tmp_path / 'ros-depth-contract'
    subprocess.run(['c++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'include'), str(ROOT / 'tests/ros_depth_contract.cpp'),
                    '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
