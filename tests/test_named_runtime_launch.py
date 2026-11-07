"""Exercise real publisher/standard/camera launch forwarding without starting ROS."""
import json
import os
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
SDK=Path('/home/bjoern/git/stereoforge/ros_ws/.pixi/envs/default')

@pytest.fixture(scope='module')
def named_commands(tmp_path_factory):
    if not (SDK/'bin/python').exists(): pytest.skip('ROS launch SDK unavailable')
    temp=tmp_path_factory.mktemp('named_runtime_launch')
    script=r'''
import importlib.util,json,sys
from pathlib import Path
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument,IncludeLaunchDescription
from launch.utilities import perform_substitutions
root=Path(sys.argv[1]);temp=Path(sys.argv[2])
def module(name):
 spec=importlib.util.spec_from_file_location('sf_named_'+name,root/'ros_ws/launch'/('stereoforge_'+name+'.launch.py'))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
publisher=module('publisher');standard=module('standard');camera=module('camera')
def defaults(m):
 context=LaunchContext()
 for entity in m.generate_launch_description().entities:
  if isinstance(entity,DeclareLaunchArgument):entity.execute(context)
 return context
results={}
for name,m,callback in [('publisher',publisher,'_launch_publisher'),('standard',standard,'_launch_standard'),('camera',camera,None)]:
 def expand(overrides):
  context=defaults(m)
  context.launch_configurations.update(dict(source='replay',manifest='fixture',clock_map='clock',replay='recording',frame_id='left_optical',run_id='named-profile-test',trace=str(temp/(name+'-trace')),with_realsense='false',person_refine='false')|overrides)
  if name=='camera':
   include=next(entity for entity in m.generate_launch_description().entities if isinstance(entity,IncludeLaunchDescription))
   forwarded={key:value.perform(context) for key,value in include.launch_arguments}
   child=defaults(publisher);child.launch_configurations.update(forwarded);context=child
   actions=publisher._launch_publisher(context)
  else:actions=getattr(m,callback)(context)
  commands=[[perform_substitutions(context,part) for part in action.cmd] for action in actions]
  return next(command for command in commands if '--manifest' in command and '--clock-map' in command)
 rows={profile:expand(dict(profile=profile)) for profile in ['quality-reference','live-quality']}
 custom_default=expand({})
 trace=temp/(name+'-trace');trace.write_text('keep-invalid-named-profile')
 try:expand(dict(profile='live-quality',stereo_variant='packed-specialized'));rejected=False
 except RuntimeError:rejected=True
 results[name]=dict(commands=rows,custom_default=custom_default,rejected=rejected,trace_preserved=trace.exists() and trace.read_text()=='keep-invalid-named-profile')
print(json.dumps(results))
'''
    env=os.environ|{'AMENT_PREFIX_PATH':str(SDK),'LD_LIBRARY_PATH':str(SDK/'lib')}
    result=subprocess.run([str(SDK/'bin/python'),'-c',script,str(ROOT),str(temp)],env=env,cwd=temp,capture_output=True,text=True,check=True)
    return json.loads(result.stdout)

@pytest.mark.parametrize('name',['publisher','standard','camera'])
@pytest.mark.parametrize('profile',['quality-reference','live-quality'])
def test_named_profile_forwarded_without_legacy_control_defaults(named_commands,name,profile):
    command=named_commands[name]['commands'][profile]
    assert command[command.index('--profile')+1]==profile
    assert command[command.index('--pipeline')+1]=='full-sgm'
    for flag in ['--coarse-manifest','--disparities','--stereo-variant','--descriptor-mode',
                 '--capture-fps','--depth-rate','--fast-mode','--ultra-fast','--coarse-schedule',
                 '--coarse-right-reference-mode','--refinement-subpixel','--adaptive-tiles',
                 '--selective-repair','--planar-recovery','--guided-fill-px',
                 '--depth-speckle-min-pixels','--depth-speckle-delta-m','--depth-speckle-delta-px']:
        assert flag not in command,(name,flag,command)

@pytest.mark.parametrize('name',['publisher','standard','camera'])
def test_named_profile_conflict_rejected_before_trace_cleanup(named_commands,name):
    assert named_commands[name]['rejected'] and named_commands[name]['trace_preserved']


@pytest.mark.parametrize('name',['publisher','standard','camera'])
def test_all_launch_routes_preserve_existing_custom_default_recipe(named_commands,name):
    command=named_commands[name]['custom_default']
    expected={'--pipeline':'experimental-dual-refinement','--disparities':'128',
        '--stereo-variant':'packed-specialized','--descriptor-mode':'ortho-hybrid',
        '--coarse-schedule':'shared-census-sequential','--coarse-right-reference-mode':'left-aggregated-selection',
        '--fast-mode':'1','--ultra-fast':'1','--refinement-subpixel':'1',
        '--adaptive-tiles':'1','--selective-repair':'1','--planar-recovery':'1'}
    for flag,value in expected.items():assert command[command.index(flag)+1]==value
    assert '--profile' not in command
    if name=='standard':assert command[command.index('--depth-rate')+1]=='60'
    else:assert '--depth-rate' not in command
