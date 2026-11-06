"""Run the production ROS diagnostic serialization contract without a ROS SDK."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_measured_depth_diagnostic_contract(tmp_path):
    binary = tmp_path / 'ros-depth-diagnostics'
    subprocess.run(['c++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'include'), str(ROOT / 'tests/ros_depth_diagnostics.cpp'),
                    '-o', str(binary)], check=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_publisher_diagnostic_options(tmp_path):
    source = (ROOT / 'apps/ros2_depth_publisher.cpp').read_text()
    parser = source[source.index('struct Options {'):source.index('std::string clock_id()')]
    cpp = ('#include <stereoforge/depth_rate_controller.hpp>\n'
           '#include <algorithm>\n#include <cctype>\n#include <cmath>\n'
           '#include <cstdlib>\n#include <iostream>\n#include <stdexcept>\n#include <string>\n'
           'using namespace stereoforge;\n' + parser +
           '\nint main(int argc,char** argv){try{parse(argc,argv);return 0;}'
           'catch(const std::exception& e){std::cerr<<e.what();return 2;}}\n')
    path = tmp_path / 'parser.cpp'
    path.write_text(cpp)
    binary = tmp_path / 'parser'
    subprocess.run(['c++', '-std=c++20', '-I', str(ROOT / 'include'),
                    str(path), '-o', str(binary)], check=True)
    required = ['--source', 'camera', '--manifest', 'x', '--clock-map', 'x',
                '--frame-id', 'x', '--run-id', 'x', '--trace', 'x']
    cases = [(['--gpu-timing-report', 'timing.json'], 0),
             (['--gpu-timing-report', 'x'], 2),
             (['--gpu-timing-report', 'timing.json', '--mode', 'rgb'], 2),
             (['--gpu-timing-report', 'timing.json', '--pipeline', 'experimental-refinement', '--coarse-manifest', 'coarse'], 2),
             (['--gpu-timing-report', 'timing.json', '--rate-report', 'timing.json'], 2),
             (['--diagnostics', '1'], 0),
             (['--diagnostics', '0'], 0),
             (['--diagnostics', '1', '--rig-session-id', 'development-session'], 0),
             (['--diagnostics', '1', '--mode', 'rgb'], 2),
             (['--diagnostics', '1', '--diagnostic-prefix', 'relative'], 2),
             (['--diagnostics', '1', '--diagnostic-prefix', '/trailing/'], 2),
             (['--diagnostics', '1', '--depth-topic', '/stereoforge/diagnostics/source_ids'], 2)]
    for extra, expected in cases:
        result = subprocess.run([str(binary), *required, *extra], capture_output=True, text=True)
        assert result.returncode == expected, (extra, result.stderr)


def test_publisher_measured_speckle_preserves_geometry(tmp_path):
    """The publisher's actual filter setup may reject, but never smooth measurements."""
    source = (ROOT / 'apps/ros2_depth_publisher.cpp').read_text()
    setup = source[source.index('    std::unique_ptr<DepthSpeckleFilter> depth_speckle;'):
                   source.index('    std::unique_ptr<GuidedDepthFill> guided_fill;')]
    cpp = r'''
#include <stereoforge/depth_speckle.hpp>
#include <memory>
#include <bit>
using namespace stereoforge;
int main(){
 const bool depth=true;
 struct {unsigned width=5,height=5;struct Metric{float fx=10,baseline_m=.1f;};
         Metric metric_calibration(){return {};}} maps;
 struct {unsigned depth_speckle_min_pixels=1;float depth_speckle_delta_m=.05f,depth_speckle_delta_px=1.25f;} o;
 const bool completed_exporter=true;
''' + setup + r'''
 std::vector<float> original(25,1.f);original[12]=1.1f;
 auto actual=original;
 depth_speckle->apply(std::as_writable_bytes(std::span(actual)));
 for(size_t i=0;i<actual.size();++i)
   if(std::isfinite(actual[i])&&std::bit_cast<uint32_t>(actual[i])!=std::bit_cast<uint32_t>(original[i]))return 1;
 return 0;
}
'''
    path = tmp_path / 'measured-speckle.cpp'
    path.write_text(cpp)
    binary = tmp_path / 'measured-speckle'
    subprocess.run(['c++', '-std=c++20', '-I', str(ROOT / 'include'),
                    str(path), '-o', str(binary)], check=True)
    result = subprocess.run([str(binary)], check=False)
    assert result.returncode == 0, 'publisher speckle smoothing changed measured geometry'


def test_publisher_completed_smoothing_has_spatial_provenance(tmp_path):
    source = (ROOT / 'apps/ros2_depth_publisher.cpp').read_text()
    begin = source.rfind('            if(', 0, source.index('// Keep provenance honest'))
    block = source[begin:source.index('            if(which==RosDeliveryOutput::Depth&&o.diagnostics)', begin)]
    cpp = r'''
#include <stereoforge/depth_speckle.hpp>
#include <bit>
using namespace stereoforge;
enum class RosDeliveryOutput{CompletedProvenance};
int main(){
 const auto which=RosDeliveryOutput::CompletedProvenance;
 struct {std::vector<uint8_t> data{1,1,6,1};} image;
 const std::array<float,4> original{1.f,1.f,2.f,1.f};
 const std::array<float,4> smoothed{1.f,1.01f,2.f,NAN};
 struct Image{std::vector<uint8_t> data;};
 Image completed_depth_image;completed_depth_image.data.resize(sizeof(smoothed));
 std::memcpy(completed_depth_image.data.data(),smoothed.data(),sizeof(smoothed));
 struct Frame{std::span<const std::byte> bytes;auto depth_bytes(){return bytes;}};
 struct Pending{Frame completed_output;} p{{std::as_bytes(std::span(original))}};
 std::vector<uint8_t> guided_filled;
''' + block + r'''
 return image.data==std::vector<uint8_t>{1,8,6,0}?0:1;
}
'''
    path = tmp_path / 'completed-provenance.cpp'
    path.write_text('#include <array>\n' + cpp)
    binary = tmp_path / 'completed-provenance'
    subprocess.run(['c++', '-std=c++20', '-I', str(ROOT / 'include'),
                    str(path), '-o', str(binary)], check=True)
    result = subprocess.run([str(binary)], check=False)
    assert result.returncode == 0, 'completed geometry changes need spatial provenance'


def test_publisher_records_execution_when_canonical_identity_unavailable(tmp_path):
    """Exercise the actual publisher JSON block for two unqualified configurations."""
    import os
    import pytest
    include = Path(os.environ.get('STEREOFORGE_JSON_INCLUDE',
                                  str(ROOT / 'build/daxa/_deps/sf_native_json-src')))
    if not (include / 'json.hpp').is_file():
        pytest.skip('native JSON build dependency unavailable')
    source = (ROOT / 'apps/ros2_depth_publisher.cpp').read_text()
    begin = source.index('    if(depth)implementation["settings"]["stereo_execution"]=')
    block = source[begin:source.index('    std::unique_ptr<RosDepthDiagnosticMetadata>', begin)]
    cpp = r'''
#include <json.hpp>
#include <string>
using json=nlohmann::json;
#define SF_DAXA_COMPILER_SHA256 "compiler-hash"
namespace stereoforge::native_estimator {
json shader_hashes(const std::string& directory,const std::string& prefix){return {{prefix,directory}};}
}
struct DaxaStereo {static std::string variant_name(int){return "packed-specialized-overwrite";}
                  static std::string shader_directory(){return "stereo-shaders";}};
struct DaxaDepthPipeline {static std::string shader_directory(){return "depth-shaders";}};
struct DaxaPreprocess {static std::string shader_directory(){return "preprocess-shaders";}};
int main(){
 const bool depth=true;
 struct {struct {unsigned width=1280,height=720,disparities=64,paths=4,p1=8,p2=32;bool subpixel=true;} stereo;
         float lr_tolerance=1;int stereo_variant=1;} depth_config;
 struct {std::string descriptor_mode="census",coarse_schedule="shared-census-sequential",coarse_right_mode="independent-sgm";} o;
 json implementation={{"settings",{{"effective_estimator",{{"status","unavailable"}}}}}};
 for(unsigned count:{64u,128u}){
 depth_config.stereo.disparities=count;
''' + block + r'''
 const auto& actual=implementation["settings"]["stereo_execution"];
 if(actual["disparities"]!=count||actual["paths"]!=4||actual["p1"]!=8||actual["p2"]!=32||actual["subpixel"]!=true)return 1;
 if(actual["schedule"]!=o.coarse_schedule||actual["right_reference_mode"]!=o.coarse_right_mode||actual["descriptor_mode"]!="census")return 2;
 for(auto key:{"stereo_spirv_sha256","depth_spirv_sha256","preprocess_spirv_sha256"})if(actual[key].empty())return 3;
 if(implementation["settings"]["effective_estimator"]["status"]!="unavailable")return 4;
 }
 return 0;
}
'''
    path = tmp_path / 'execution-configuration.cpp'
    path.write_text(cpp)
    binary = tmp_path / 'execution-configuration'
    subprocess.run(['c++', '-std=c++20', '-I', str(include), str(path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_gpu_timing_collection_is_optional_and_bounded(tmp_path):
    source = (ROOT / 'apps/ros2_depth_publisher.cpp').read_text()
    begin = source.index('struct GpuTimingSample {')
    record = source[begin:source.index('\n', begin)]
    begin = source.index('        if(gpu_timing_report&&!trace_full){')
    block = source[begin:source.index('        if(o.diagnostics&&!trace_full)', begin)]
    cpp = r'''
#include <cstdint>
#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>
using steady=int;
template<class T>uint64_t now_ns(){static uint64_t value=100;return ++value;}
''' + record + r'''
int main(){
 struct Timing{bool combined_gpu_timing_available=true;double combined_gpu_ms=32;struct {uint64_t start_tick=10;}preprocessing;uint64_t end_tick=42;uint32_t timestamp_bits=64;double timestamp_period=1000000;};
 unsigned calls=0;
 struct Depth{unsigned* calls;Timing value;auto timings(){++*calls;return value;}};
 struct Capture{uint64_t frame_id=(uint64_t(1)<<60)+3,camera_timestamp_ns=1000000001;};
 struct Output{uint64_t producer_ticket(){return 77;}};
 struct Pending{Depth depth;std::shared_ptr<Capture>capture;Output output;}p{{&calls,{}},std::make_shared<Capture>(),{}};
 struct Record{uint64_t publish_end_ns=99;};
 struct Trace{std::vector<Record> rows{{}};const auto& records(){return rows;}}trace;
 struct {unsigned max_frames=1;}o;
 bool trace_full=false;std::unique_ptr<int>gpu_timing_report;
 std::vector<GpuTimingSample>gpu_timings;gpu_timings.reserve(1);const auto* storage=gpu_timings.data();
 auto collect=[&](){
''' + block + r'''
 };
 collect();if(calls||!gpu_timings.empty())return 1;
 gpu_timing_report=std::make_unique<int>(1);collect();
 if(calls!=1||gpu_timings.size()!=1||gpu_timings.data()!=storage)return 2;
 const auto& sample=gpu_timings[0];
 if(sample.source_id!=int64_t(p.capture->frame_id)||sample.capture_ns!=p.capture->camera_timestamp_ns||sample.producer_ticket!=77||sample.published_ns!=99)return 3;
 if(sample.start_tick!=10||sample.end_tick!=42||sample.timestamp_bits!=64||sample.timestamp_period_ns!=1000000||sample.gpu_span_ms!=32||sample.query_end_ns<sample.query_begin_ns)return 4;
 bool rejected=false;try{collect();}catch(const std::runtime_error&){rejected=true;}
 if(!rejected||gpu_timings.size()!=1||gpu_timings.data()!=storage)return 5;
 gpu_timings.clear();p.depth.value.combined_gpu_timing_available=false;rejected=false;
 try{collect();}catch(const std::runtime_error&){rejected=true;}
 if(!rejected||!gpu_timings.empty())return 6;
 return 0;
}
'''
    path = tmp_path / 'gpu-timing-collection.cpp'
    path.write_text(cpp)
    binary = tmp_path / 'gpu-timing-collection'
    subprocess.run(['c++', '-std=c++20', str(path), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_publisher_drains_pending_frames_in_source_order(tmp_path):
    source = (ROOT / 'apps/ros2_depth_publisher.cpp').read_text()
    start = source.index('capture->stop();') + len('capture->stop();')
    drain = source[start:source.index('if(completed_exporter)completed_exporter->drain()', start)]
    start = source.index('    auto retire_pending=')
    helper = source[start:source.index('    try{\n        while(', start)]
    cpp = r'''
#include <array>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
struct Capture{uint64_t frame_id;};
struct Pending{bool output;std::shared_ptr<Capture>capture;bool ready;};
int main(){
 std::array<Pending,2>pending{{{true,std::make_shared<Capture>(Capture{922}),true},{true,std::make_shared<Capture>(Capture{920}),true}}};
 std::vector<uint64_t>published;bool trace_full=false;
 auto retire=[&](Pending& p,bool drain){if(p.output&&(drain||p.ready)){published.push_back(p.capture->frame_id);p.output=false;}};
''' + helper + drain + r'''
 if(published!=std::vector<uint64_t>{920,922})return 1;
 published.clear();pending={{{true,std::make_shared<Capture>(Capture{924}),true},{true,std::make_shared<Capture>(Capture{923}),false}}};
 retire_pending(false);if(!published.empty())return 2;
 pending[1].ready=true;retire_pending(false);
 if(published!=std::vector<uint64_t>{923,924})return 3;
 trace_full=true;pending={{{true,std::make_shared<Capture>(Capture{926}),false},{true,std::make_shared<Capture>(Capture{925}),false}}};
 retire_pending(true);if(pending[0].output||pending[1].output)return 4;
 pending[0].capture.reset();pending[1].capture.reset();retire_pending(false);
 return 0;
}
'''
    path = tmp_path / 'pending-source-order.cpp'
    path.write_text(cpp)
    binary = tmp_path / 'pending-source-order'
    subprocess.run(['c++', '-std=c++20', str(path), '-o', str(binary)], check=True)
    result = subprocess.run([str(binary)], check=False)
    assert result.returncode == 0, 'shutdown published newer slot before older source frame'
