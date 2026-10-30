from pathlib import Path
import subprocess
import shutil
import pytest
from benchmarks.zed_comparison.report import _same_estimator

ROOT = Path(__file__).resolve().parents[1]


def test_ros_variant_parser_executable(tmp_path):
    compiler=shutil.which('c++')
    if not compiler: pytest.skip('C++ compiler unavailable')
    source=(ROOT/'apps/ros2_depth_publisher.cpp').read_text()
    parser=source[source.index('struct Options {'):source.index('std::string clock_id()')]
    cpp='''#include <stereoforge/depth_rate_controller.hpp>\n#include <algorithm>\n#include <cctype>\n#include <cmath>\n#include <cstdlib>\n#include <iostream>\n#include <stdexcept>\n#include <string>\n''' + parser + '''
int main(int argc,char** argv){try{auto o=parse(argc,argv);std::cout<<o.stereo_variant;return 0;}catch(const std::exception&){return 2;}}
'''
    path=tmp_path/'parser.cpp';path.write_text(cpp);exe=tmp_path/'parser'
    subprocess.run([compiler,'-std=c++20','-I',str(ROOT/'include'),str(path),'-o',str(exe)],check=True,capture_output=True,text=True)
    args=['--source','camera','--manifest','x','--clock-map','x','--frame-id','x','--run-id','x','--trace','x']
    for extra, expected, status in [([], 'packed-specialized',0),(['--stereo-variant','packed-specialized-overwrite'],'packed-specialized-overwrite',0),(['--stereo-variant','bad'],'',2)]:
        result=subprocess.run([str(exe),*args,*extra],capture_output=True,text=True)
        assert (result.stdout,result.returncode)==(expected,status)


def test_unavailable_estimator_never_matches_quality_profile():
    identity={'name':'StereoForge','commit':{'status':'available','value':'a'*40},'device':{'status':'available','value':{'device_uuid':'a'*32,'driver_uuid':'b'*32,'vendor_id':1,'device_id':1,'driver_version':1}},'settings':{'effective_estimator':{}}}
    assert not _same_estimator(identity,identity,{'status':'unavailable','reason':'experimental variant'})


def test_trace_declares_actual_variant_and_unavailable_candidate():
    source=(ROOT/'apps/ros2_depth_publisher.cpp').read_text()
    assert 'depth_config.stereo_variant=' in source
    assert '{"stereo_variant",depth?DaxaStereo::variant_name(depth_config.stereo_variant):"not_executed"}' in source
    assert 'experimental_variant_not_qualified_for_canonical_estimator_identity' in source


def test_delivery_evaluator_preserves_unavailable_estimator():
    from test_zed_ros2_delivery import traces, POLICY
    from benchmarks.zed_comparison.ros2_delivery import summarize_delivery
    events=traces()
    unavailable={'status':'unavailable','reason':'experimental_variant_not_qualified_for_canonical_estimator_identity'}
    events[0]['config']['effective_estimator']=unavailable
    result=summarize_delivery(events,policy=POLICY)
    assert result['runs'][0]['effective_estimator']==unavailable
