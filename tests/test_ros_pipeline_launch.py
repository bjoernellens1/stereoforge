"""Expand the real ROS launch descriptions without executing their processes."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
SDK=Path('/home/bjoern/git/stereoforge/ros_ws/.pixi/envs/default')
REFINEMENT_FLAGS=(
    '--fast-mode','--ultra-fast','--refinement-subpixel','--continuous-subpixel',
    '--adaptive-tiles','--min-disparity-bound','--max-disparity-bound',
    '--selective-repair','--planar-recovery','--guided-fill-px',
)

@pytest.fixture(scope='module')
def launch_commands(tmp_path_factory):
    if not (SDK/'bin/python').exists():pytest.skip('ROS launch SDK unavailable')
    temp=tmp_path_factory.mktemp('launch_pipeline_binding')
    script=r'''
import importlib.util,json,sys
from pathlib import Path
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
root=Path(sys.argv[1]);temp=Path(sys.argv[2]);results={}
for name,function in [('publisher','_launch_publisher'),('standard','_launch_standard')]:
 path=root/'ros_ws/launch'/('stereoforge_'+name+'.launch.py')
 spec=importlib.util.spec_from_file_location('sf_'+name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 def expand(overrides):
  context=LaunchContext()
  for entity in module.generate_launch_description().entities:
   if isinstance(entity,DeclareLaunchArgument):entity.execute(context)
  base=dict(source='replay',manifest='fixture',clock_map='clock',replay='recording',frame_id='left_optical',run_id='binding-test',trace=str(temp/(name+'-trace.json')),with_realsense='false',person_refine='false')
  context.launch_configurations.update(base|overrides)
  actions=getattr(module,function)(context)
  commands=[[perform_substitutions(context,part) for part in action.cmd] for action in actions]
  return next(command for command in commands if '--manifest' in command and '--clock-map' in command)
 full=expand(dict(pipeline='full-sgm'))
 experimental=expand(dict(pipeline='experimental-dual-refinement',coarse_manifest='coarse'))
 custom=expand(dict(pipeline='full-sgm',descriptor_mode='ortho-hybrid',coarse_schedule='paired-per-path',coarse_right_reference_mode='independent-sgm'))
 trace=temp/(name+'-trace.json');trace.write_text('keep-on-invalid-request')
 try:
  expand(dict(pipeline='full-sgm',fast_mode='1'))
  rejected=False
 except RuntimeError:
  rejected=True
 results[name]=dict(full=full,experimental=experimental,custom=custom,rejected=rejected,trace_preserved=trace.exists() and trace.read_text()=='keep-on-invalid-request')
print(json.dumps(results))
'''
    env=os.environ|{'AMENT_PREFIX_PATH':str(SDK),'LD_LIBRARY_PATH':str(SDK/'lib')}
    result=subprocess.run([str(SDK/'bin/python'),'-c',script,str(ROOT),str(temp)],env=env,cwd=temp,capture_output=True,text=True,check=True)
    return json.loads(result.stdout)

@pytest.mark.parametrize('name',['publisher','standard'])
def test_full_sgm_launch_does_not_inject_refinement_defaults(launch_commands,name):
    command=launch_commands[name]['full']
    assert all(flag not in command for flag in REFINEMENT_FLAGS)
    for flag in ['--descriptor-mode','--coarse-schedule','--coarse-right-reference-mode']:assert flag not in command

@pytest.mark.parametrize('name',['publisher','standard'])
def test_experimental_launch_preserves_existing_recipe(launch_commands,name):
    command=launch_commands[name]['experimental']
    expected={'--descriptor-mode':'ortho-hybrid','--fast-mode':'1','--ultra-fast':'1',
              '--coarse-schedule':'shared-census-sequential','--coarse-right-reference-mode':'left-aggregated-selection',
              '--refinement-subpixel':'1','--continuous-subpixel':'0','--adaptive-tiles':'1',
              '--min-disparity-bound':'0','--max-disparity-bound':'0','--selective-repair':'1','--planar-recovery':'1'}
    for flag,value in expected.items():assert command.count(flag)==1 and command[command.index(flag)+1]==value
    # The publisher launch historically leaves guided fill at native default0.
    if '--guided-fill-px' in command:assert command[command.index('--guided-fill-px')+1]=='0'

@pytest.mark.parametrize('name',['publisher','standard'])
def test_full_sgm_launch_preserves_effective_common_controls(launch_commands,name):
    command=launch_commands[name]['custom']
    for flag,value in [('--descriptor-mode','ortho-hybrid'),('--coarse-schedule','paired-per-path'),('--coarse-right-reference-mode','independent-sgm')]:
        assert command[command.index(flag)+1]==value

@pytest.mark.parametrize('name',['publisher','standard'])
def test_invalid_launch_request_rejected_before_trace_cleanup(launch_commands,name):
    assert launch_commands[name]['rejected']
    assert launch_commands[name]['trace_preserved']


@pytest.fixture(scope='module')
def resolve_options():
    import importlib.util
    path=ROOT/'ros_ws/launch/stereoforge_launch_options.py'
    spec=importlib.util.spec_from_file_location('sf_pipeline_options_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_pipeline_options


def test_resolver_keeps_unset_full_sgm_controls_unset(resolve_options):
    requested={'pipeline':'full-sgm','fast_mode':'','descriptor_mode':''}
    assert resolve_options(requested)==requested


@pytest.mark.parametrize('flag',REFINEMENT_FLAGS)
def test_resolver_rejects_explicit_disabled_refinement_request(resolve_options,flag):
    with pytest.raises(RuntimeError,match='refinement-only options require'):
        resolve_options({'pipeline':'full-sgm',flag[2:].replace('-','_'):'0'})


def test_resolver_preserves_requested_values_and_does_not_mutate_them(resolve_options):
    requested={'pipeline':'experimental-refinement','mode':'depth-rgb',
               'fast_mode':'0','ultra_fast':'','descriptor_mode':''}
    original=dict(requested)
    actual=resolve_options(requested)
    assert requested==original
    assert actual['fast_mode']=='0'
    assert actual['ultra_fast']=='1'
    assert actual['descriptor_mode']=='ortho-hybrid'


@pytest.mark.parametrize('mode,planar',[('depth','1'),('depth-rgb','0')])
def test_resolver_rejects_guided_fill_without_inputs(resolve_options,mode,planar):
    with pytest.raises(RuntimeError,match='guided fill requires depth-rgb and planar recovery'):
        resolve_options({'pipeline':'experimental-refinement','mode':mode,
                         'guided_fill_px':'2','planar_recovery':planar})
