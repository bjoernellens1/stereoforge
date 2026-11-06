"""Exercise the real installed ROS launch expansion without starting a process."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SDK = Path('/home/bjoern/git/stereoforge/ros_ws/.pixi/envs/default')


def test_cloud_launch_preserves_explicit_export_controls(tmp_path):
    python = SDK / 'bin/python'
    if not python.exists():
        pytest.skip('native ROS SDK unavailable')
    script = r'''
import importlib.util,json,sys
from pathlib import Path
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
spec=importlib.util.spec_from_file_location('sf_launch',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
context=LaunchContext();description=module.generate_launch_description()
for entity in description.entities:
 if isinstance(entity,DeclareLaunchArgument):entity.execute(context)
def command(overrides):
 context.launch_configurations.update(overrides)
 action=module._launch_publisher(context)[0]
 return [perform_substitutions(context,part) for part in action.cmd]
base=dict(source='replay',manifest='fixture',clock_map='clock',replay='recording',frame_id='left_optical',run_id='launch-test',trace=sys.argv[2],pipeline='full-sgm')
default=command(base)
custom=command(dict(cloud='1',cloud_stride='3',cloud_rate='7',cloud_topic='/test/native_points'))
print(json.dumps(dict(default=default,custom=custom)))
'''
    env = os.environ | {'AMENT_PREFIX_PATH': str(SDK), 'LD_LIBRARY_PATH': str(SDK / 'lib')}
    result = subprocess.run([str(python), '-c', script,
                             str(ROOT / 'ros_ws/launch/stereoforge_publisher.launch.py'),
                             str(tmp_path / 'trace.json')], env=env,
                            capture_output=True, text=True, check=True)
    commands = json.loads(result.stdout)
    for name, expected in [('default', {'--cloud': '0', '--cloud-stride': '4', '--cloud-rate': '15', '--cloud-topic': '/stereoforge/points'}),
                           ('custom', {'--cloud': '1', '--cloud-stride': '3', '--cloud-rate': '7', '--cloud-topic': '/test/native_points'})]:
        cmd = commands[name]
        for option, value in expected.items():
            assert option in cmd, (option, cmd)
            assert cmd.count(option) == 1 and cmd[cmd.index(option) + 1] == value
        assert cmd[cmd.index('--pipeline') + 1] == 'full-sgm'
        assert cmd[cmd.index('--mode') + 1] == 'depth-rgb'
