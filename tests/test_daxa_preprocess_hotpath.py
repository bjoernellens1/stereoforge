"""CPU-only checks of the native staged preprocessing ownership and byte contract.

The allocation probe extracts production admission/retirement/copy methods and
substitutes only the Vulkan device/graph boundary. It excludes Daxa recording,
driver allocations, GPU work and physical capture.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / 'src/daxa_preprocess.cpp'
STEREO = ROOT / 'src/daxa_stereo.cpp'
FRAME = ROOT / 'include/stereoforge/daxa_frame.hpp'


def test_reused_raw_upload_orders_previous_transfer_write_and_compute_read():
    source = PREPROCESS.read_text()
    upload = body(source, '"bounded packed source upload").reads(views[Staging]).writes(views[Raw]).executes(')
    before_copy = upload.split('ti.recorder.copy_buffer_to_buffer(', 1)[0]
    barrier = re.search(r'ti\.recorder\.pipeline_barrier\s*\(\s*\{(.*?)\}\s*\)', before_copy, re.S)
    assert barrier, 'reused Raw requires a pre-copy dependency across submissions'
    fields = barrier.group(1)
    assert re.search(r'\.src_access\s*=\s*daxa::AccessConsts::TRANSFER_WRITE\s*\|\s*daxa::AccessConsts::COMPUTE_SHADER_READ', fields)
    assert re.search(r'\.dst_access\s*=\s*daxa::AccessConsts::TRANSFER_WRITE', fields)


def check_submission_hotpath(preprocess, stereo):
    enqueue = body(preprocess, 'DaxaFrame DaxaPreprocess::enqueue_at(')
    process = body(stereo, 'DaxaStereo::Frame DaxaStereo::process_at(')
    integrated = body(preprocess, 'DaxaStereo::Frame DaxaPreprocessStereo::enqueue(')
    forbidden = r'\b(?:make_shared|make_unique|create_buffer|create_memory|allocate|create_command_recorder|readback|inspect|wait_idle|wait_on_submit|malloc|calloc|realloc|resize|reserve|push_back)(?:<[^;()]*>)?\s*\('
    assert not re.search(forbidden, enqueue + process + integrated), 'frame submission must not allocate, wait or read back'
    assert enqueue.count('std::memcpy(') == 1, 'one explicit packed-source staging copy'
    assert 'std::memcpy(' not in process + integrated, 'GPU intake must not copy full planes'
    assert enqueue.index('validate_source(*source)') < enqueue.index('std::memcpy(')
    assert enqueue.index('detail::available_slot(') < enqueue.index('std::memcpy(')
    assert 'oldest_pending_submit_index(' not in enqueue + process, 'internal helpers use the shared completion snapshot'
    assert integrated.count('oldest_pending_submit_index(') == 1, 'one amortized completion query per integrated admission'
    assert integrated.index('collect_completed_inputs_at(oldest)') < integrated.index('preprocess_.enqueue_at(')
    assert integrated.index('collect_completed_sources_at(oldest)') < integrated.index('preprocess_.enqueue_at(')
    assert 'process_at(&input,oldest)' in integrated
    assert 'collect_garbage(' in process, 'retire completed Daxa command resources'
    assert enqueue.count('->execute(') == 1 and process.count('->execute(') == 1


def test_staging_and_gpu_intake_have_no_application_heap_wait_or_intermediate_copy():
    check_submission_hotpath(PREPROCESS.read_text(), STEREO.read_text())


@pytest.mark.parametrize('regression', ['d.wait_idle();', 'std::make_shared<int>();', 'std::vector<int> x;x.resize(1);'])
def test_hotpath_guard_detects_wait_and_allocation(regression):
    source = PREPROCESS.read_text()
    marker = 'DaxaFrame DaxaPreprocess::enqueue_at(CaptureLease source,uint64_t oldest){'
    assert marker in source
    with pytest.raises(AssertionError, match='must not allocate'):
        check_submission_hotpath(source.replace(marker, marker + regression, 1), STEREO.read_text())


def test_completion_guard_detects_repeated_device_polling():
    source = PREPROCESS.read_text()
    marker = 'DaxaFrame DaxaPreprocess::enqueue_at(CaptureLease source,uint64_t oldest){'
    with pytest.raises(AssertionError, match='shared completion snapshot'):
        check_submission_hotpath(source.replace(marker, marker + 'context.oldest_pending_submit_index();', 1), STEREO.read_text())


def check_public_completion_and_gc(preprocess, stereo):
    public = [body(preprocess, 'DaxaFrame DaxaPreprocess::enqueue('),
              body(preprocess, 'size_t DaxaPreprocess::collect_completed_sources('),
              body(preprocess, 'DaxaStereo::Frame DaxaPreprocessStereo::enqueue('),
              body(preprocess, 'size_t DaxaPreprocessStereo::collect('),
              body(stereo, 'DaxaStereo::Frame DaxaStereo::process()'),
              body(stereo, 'DaxaStereo::Frame DaxaStereo::process(const DaxaFrame&')]
    for entry in public:
        assert entry.index('require_thread()') < entry.index('oldest_pending_submit_index('), 'reject foreign host thread before device access'
        assert entry.count('oldest_pending_submit_index(') == 1
    assert 'collect_garbage(' in public[0], 'standalone preprocessing must retire deferred Daxa command resources'
    assert 'collect_garbage(' not in public[2], 'integration reuses Stereo GC instead of adding a second application GC'
    assert 'collect_garbage(' in body(stereo, 'DaxaStereo::Frame DaxaStereo::process_at(')


def test_public_admission_rejects_foreign_thread_and_bounds_framework_resources():
    check_public_completion_and_gc(PREPROCESS.read_text(), STEREO.read_text())


def test_gc_guard_catches_unbounded_standalone_command_retention():
    source = PREPROCESS.read_text()
    marker = 'impl_->context->device.collect_garbage();'
    assert marker in body(source, 'DaxaFrame DaxaPreprocess::enqueue(')
    with pytest.raises(AssertionError, match='deferred Daxa command'):
        check_public_completion_and_gc(source.replace(marker, '', 1), STEREO.read_text())


def check_config_owner(stereo):
    process = re.sub(r'\s+', '', body(stereo, 'DaxaStereo::Frame DaxaStereo::process_at('))
    assert 'maps==e.calibration' in process or 'maps.get()==e.calibration.get()' in process, 'same IDs do not establish the immutable map/profile owner'
    constructor = body(stereo, 'DaxaStereo::DaxaStereo(StereoConfig c,const DaxaFrame& prototype,')
    assert not re.search(r'impl_->\w+\s*=\s*prototype\b', constructor), 'constructor must not retain a prototype GPU frame'
    validate = body((ROOT/'src/daxa_frame.cpp').read_text(), 'void DaxaFrame::validate(')
    assert 'generation_==slot_->generation' in validate, 'stale typed generation must reject'
    assert 'input->owner_->context==e.context' in process


def test_gpu_intake_binds_actual_context_generation_and_configuration_owner():
    check_config_owner(STEREO.read_text())


def test_owner_guard_catches_accepting_same_ids_with_different_maps():
    source = STEREO.read_text()
    marker = 'maps==e.calibration&&'
    assert marker in source
    with pytest.raises(AssertionError, match='immutable map/profile owner'):
        check_config_owner(source.replace(marker, '', 1))


def test_benchmark_measures_integrated_retirement_copy_and_queries_without_reference_work():
    source = (ROOT/'apps/daxa_preprocess_benchmark.cpp').read_text()
    measured = body(source, 'for(uint32_t i=0;i<frames+warmup;++i)')
    assert measured.index('auto a=Clock::now()') < measured.index('input={}') < measured.index('pipeline->enqueue(source,&input)')
    assert 'stereo->process(' not in measured, 'benchmark must exercise the integration owner'
    for forbidden in ('readback(', 'daxa_preprocess_reference(', 'metadata(', 'resize(', 'push_back(', 'std::vector'):
        assert forbidden not in measured, 'diagnostic/full-reference work stays outside frame timing'
    assert measured.index('pre->timings(input)') < measured.index('auto d=Clock::now()')
    assert 'host_wall[n]=elapsed(a,d)' in measured
    assert 'query_ms[n]=elapsed(c,d)' in measured
    assert 'repeated_retained_record' in source, 'fixture repetition is not fresh source throughput'
    release = source.index('context->close()')
    assert source.index('pipeline.reset()') < release
    assert release < source.index('context->validation_errors||context->validation_warnings')


def check_stereo_diagnostic_gating(source):
    gate = 'if(check&&stereo&&options.contains("--dump-prefix"))'
    assert gate in source, 'stereo diagnostics require reference, dump and stereo together'
    diagnostic = body(source, gate)
    measured = body(source, 'for(uint32_t i=0;i<frames+warmup;++i)')
    assert source.index(gate) > source.index(measured)+len(measured), 'new stereo readback must follow every timed sample'
    assert source.count('stereo->readback(') == 1
    assert 'stereo->readback(output,true)' in diagnostic
    assert 'daxa_stereo_reference(config,expected.planes[4],expected.planes[5])' in diagnostic, 'scalar input comes from independently qualified fixture luma'
    assert diagnostic.index('verify_stereo_integer(') < diagnostic.index('stereo_integer_checked=true')
    assert diagnostic.index('verify_stereo_integer(') < diagnostic.index('stereo_packed_checked=true')
    for flag in ('stereo_integer_checked', 'stereo_packed_checked'):
        assert flag+'=false' in source and source.count(flag+'=true') == 1, 'resident runs cannot acquire diagnostic check flags'
    assert '"compound_float_reference_checked",false' in source


def test_stereo_integer_diagnostics_are_separate_from_resident_samples():
    check_stereo_diagnostic_gating((ROOT/'apps/daxa_preprocess_benchmark.cpp').read_text())


def test_diagnostic_guard_catches_readback_in_resident_mode():
    source = (ROOT/'apps/daxa_preprocess_benchmark.cpp').read_text()
    with pytest.raises(AssertionError, match='reference, dump and stereo'):
        check_stereo_diagnostic_gating(source.replace('if(check&&stereo&&options.contains("--dump-prefix"))', 'if(check&&stereo)', 1))


def test_diagnostic_guard_catches_falsely_qualified_resident_flags():
    source = (ROOT/'apps/daxa_preprocess_benchmark.cpp').read_text()
    with pytest.raises(AssertionError, match='resident runs'):
        check_stereo_diagnostic_gating(source.replace('auto diagnostic_begin=Clock::now();', 'stereo_integer_checked=true;auto diagnostic_begin=Clock::now();', 1))


def check_retirement(preprocess, stereo):
    sources = body(preprocess, 'size_t DaxaPreprocess::collect_completed_sources_at(')
    inputs = body(stereo, 'size_t DaxaStereo::collect_completed_inputs_at(')
    for section in (sources, inputs):
        assert 'for(auto& slot:e.slots)' in section, 'bounded slot scan, no frame history'
        assert not re.search(r'oldest_pending_submit_index\(|wait_|std::vector|push_back|erase\(', section)
    assert 'available_slot(slot.use_count(),slot->submit,oldest)' in sources, 'preprocess release needs no outstanding GPU lease'
    assert 'completed_submission(slot->submit,oldest)' in inputs
    release = inputs.index('slot->input={}')
    assert 'use_count()' not in inputs[:release], 'completed GPU input retires even when output metadata is retained'
    assert inputs.index('available_slot(slot.use_count(),slot->submit,oldest)') > release
    assert inputs.index('slot->source.reset(') > release, 'unleased completed output metadata releases after GPU-input retirement'
    process = body(stereo, 'DaxaStereo::Frame DaxaStereo::process_at(')
    assert 'slot->input=*input' in process and 'slot->source=input->source()' in process


def test_retirement_is_bounded_and_separates_gpu_input_from_metadata():
    check_retirement(PREPROCESS.read_text(), STEREO.read_text())


def test_retirement_guard_catches_output_reference_deadlock():
    source = STEREO.read_text()
    marker = 'if(slot->input&&detail::completed_submission(slot->submit,oldest))'
    assert marker in source
    with pytest.raises(AssertionError, match='even when output metadata'):
        check_retirement(PREPROCESS.read_text(), source.replace(marker, 'if(slot.use_count()==1&&slot->input&&detail::completed_submission(slot->submit,oldest))', 1))


def check_dependencies(preprocess, stereo):
    assert '.host.writes(views[Staging])' in preprocess
    assert '.reads(views[Staging]).writes(views[Raw])' in preprocess
    assert '.reads(views[Raw],views[Params]).writes(views[OriginalLeft],views[OriginalRight])' in preprocess
    assert '.reads(views[OriginalLeft],views[OriginalRight],views[MapLeftX]' in preprocess
    setup = body(stereo, 'void DaxaStereo::Impl::setup(')
    assert 'if(calibration&&stage<2)' in setup, 'either scheduled Census callback needs imported visibility'
    assert '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ' in setup
    assert '.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ' in setup, 'updated input-address parameters need host visibility'


def test_host_copy_compute_and_cross_graph_visibility_are_explicit():
    check_dependencies(PREPROCESS.read_text(), STEREO.read_text())


def test_dependency_guard_catches_first_census_order_assumption():
    source = STEREO.read_text()
    with pytest.raises(AssertionError, match='either scheduled Census'):
        check_dependencies(PREPROCESS.read_text(), source.replace('if(calibration&&stage<2)', 'if(calibration&&stage==0)', 1))


def check_shader_contract(shader, operations):
    decode = body(shader, 'void decode_yuyv(')
    rectify = body(shader, 'void rectify_gray(')
    assert decode.index('if(tid.x>=p.eye_width') < decode.index('raw_byte('), 'source loads need padded-dispatch bounds'
    assert rectify.index('if(tid.x>=p.width') < rectify.index('mx[i]'), 'map loads need padded-dispatch bounds'
    assert 'if(weights[tap]!=0&&!inside)full_support=false' in rectify, 'support counts only nonzero-weight taps'
    assert rectify.index('if(inside)') < rectify.index('original[uint(yy)'), 'zero-border loads must remain in bounds'
    assert rectify.index('channels=(channels+512)>>10') < rectify.index('preprocess_gray(color)'), 'RGB rounds before grayscale conversion'
    quantize = body(operations, 'int preprocess_quantize(')
    assert 'floor(scaled)' in quantize and 'fraction==0.5' in quantize and 'value&1' in quantize, 'map ties must round to even'
    gray = body(operations, 'uint preprocess_gray(')
    assert all(coefficient in gray for coefficient in ('9798', '19235', '3735', '16384', '>>15')), 'qualified gray15 profile'
    assert 'Support' not in body(STEREO.read_text(), 'void DaxaStereo::Impl::setup('), 'diagnostic support never changes frozen Census input'


def test_shader_preserves_rgb_rounding_borders_and_diagnostic_support():
    check_shader_contract((ROOT/'python/stereoforge/shaders/daxa/preprocess.slang').read_text(), (ROOT/'python/stereoforge/shaders/ops/preprocess.slang').read_text())


def test_shader_guard_detects_zero_weight_edge_invalidation():
    shader = (ROOT/'python/stereoforge/shaders/daxa/preprocess.slang').read_text()
    with pytest.raises(AssertionError, match='nonzero-weight'):
        check_shader_contract(shader.replace('if(weights[tap]!=0&&!inside)', 'if(!inside)'), (ROOT/'python/stereoforge/shaders/ops/preprocess.slang').read_text())


def test_optional_fixture_loader_dependencies_do_not_enter_stereo_target():
    cmake = (ROOT/'cmake/DaxaStereo.cmake').read_text()
    optional = cmake.index('option(STEREOFORGE_DAXA_PREPROCESS')
    assert re.search(r'option\(STEREOFORGE_DAXA_PREPROCESS[^\n]+ OFF\)', cmake)
    assert 'if(STEREOFORGE_DAXA_PREPROCESS)' in cmake[optional:]
    for dependency in ('OpenSSL', 'sf_native_json', 'src/daxa_preprocess.cpp', 'stereoforge_capture', 'Python'):
        assert dependency not in cmake[:optional], 'old stereo build must not acquire optional fixture/runtime dependencies'
    for source in (STEREO, ROOT/'src/daxa_frame.cpp'):
        assert not re.search(r'#include\s*[<"](?:openssl/|json\.hpp|Python\.h)', source.read_text())


def test_diagnostic_readback_releases_its_resources_instead_of_retaining_growth():
    readback = body(PREPROCESS.read_text(), 'DaxaPreprocessReadback DaxaPreprocess::readback(')
    assert 'e.allocate(' not in readback, 'diagnostic allocations must not enter retained resource vectors'
    assert 'destroy_buffer(' in readback and 'collect_garbage(' in readback
    assert 'auto memory=d.create_memory(' in readback


def test_actual_stereo_retirement_frees_capture_pool_after_output_release(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    helper = '\n'.join(definition(FRAME.read_text(), signature) for signature in (
        'inline bool completed_submission(uint64_t submitted,uint64_t oldest_pending)',
        'inline bool available_slot(long references,uint64_t submitted,uint64_t oldest_pending)'))
    retirement = definition(STEREO.read_text(), 'size_t DaxaStereo::collect_completed_inputs_at(uint64_t oldest)')
    code = r'''
#include <stereoforge/capture.hpp>
#include <iostream>
namespace stereoforge {
namespace detail { HELPER }
struct Slot {std::shared_ptr<int> input;CaptureLease source;uint64_t submit=4;};
struct Impl {std::vector<std::shared_ptr<Slot>> slots;void require_thread()const{}};
struct DaxaStereo {std::shared_ptr<Impl> impl_;size_t collect_completed_inputs_at(uint64_t);};
RETIREMENT
}
int main(){using namespace stereoforge;DaxaStereo engine;engine.impl_=std::make_shared<Impl>();
 std::array<std::weak_ptr<const CaptureFrame>,2> capture_pool;
 for(unsigned i=0;i<2;++i){auto source=std::make_shared<CaptureFrame>();capture_pool[i]=source;auto slot=std::make_shared<Slot>();slot->source=source;slot->input=std::make_shared<int>(1);engine.impl_->slots.push_back(slot);}
 auto output0=engine.impl_->slots[0],output1=engine.impl_->slots[1];
 if(engine.collect_completed_inputs_at(5)!=2||output0->input||output1->input||capture_pool[0].expired()||capture_pool[1].expired()){std::cerr<<"GPU input must retire while retained output metadata survives";return 1;}
 output0.reset();output1.reset();engine.collect_completed_inputs_at(5);
 if(!capture_pool[0].expired()||!capture_pool[1].expired()){std::cerr<<"completed unleased output metadata keeps Capture pool slots pinned";return 2;}
 auto source=std::make_shared<CaptureFrame>();capture_pool[0]=source;engine.impl_->slots[0]->source=std::move(source);engine.impl_->slots[0]->submit=6;engine.collect_completed_inputs_at(5);
 if(capture_pool[0].expired()){std::cerr<<"inflight metadata retired before completion";return 3;}
}
'''.replace('HELPER', helper).replace('RETIREMENT', retirement)
    cpp, binary = tmp_path/'retire.cpp', tmp_path/'retire'
    cpp.write_text(code)
    build = subprocess.run([compiler, '-std=c++20', '-O2', '-I', str(ROOT/'include'), str(cpp), '-o', str(binary)], capture_output=True, text=True, timeout=30)
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


SHIM = r'''
#include <stereoforge/capture.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <new>
#include <span>
#include <stdexcept>
std::atomic<bool> tracking=false;
std::atomic<unsigned long> allocations=0;
void* operator new(std::size_t n){if(tracking)++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);} void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);} void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
namespace daxa {constexpr unsigned QUEUE_MAIN=0;}
namespace stereoforge {
void require(bool b,const char* m){if(!b)throw std::invalid_argument(m);}
struct DaxaRectification {
 uint32_t eye_width=0,source_height=0,stride_bytes=0,width=0,height=0;
 std::string original_identity="fixture-original",rectified_identity="fixture-rectified";
 std::array<std::vector<float>,4> maps;
 uint64_t requested_bytes(unsigned)const;void validate(unsigned slots=1)const;void validate_source(const CaptureFrame&)const;
};
struct DaxaPreprocessReadback {std::array<std::vector<uint32_t>,8> planes;};
namespace detail { HELPERS }
struct Address {void* pointer;void* value()const{return pointer;}};
struct Device {
 std::array<void*,2> address{};uint64_t submitted=0;
 Address buffer_host_address(unsigned i){return {address.at(i)};}
 uint64_t latest_queue_submit_index(unsigned)const{return submitted;}
};
struct Context {Device device;};
struct Graph {Device* device;unsigned executes=0;void execute(std::initializer_list<int>){++executes;++device->submitted;}};
constexpr unsigned Staging=0;
struct Slot {CaptureLease source;uint64_t generation=0,submit=0;std::array<unsigned,1> buffers{};std::unique_ptr<Graph> graph;};
struct Impl {std::shared_ptr<Context> context;std::shared_ptr<DaxaRectification> calibration;std::vector<std::shared_ptr<Slot>> slots;uint64_t next_generation=1,retired_sources=0;double upload_cpu_ms=0;void require_thread()const{}};
struct DaxaFrame {std::shared_ptr<Impl> owner;std::shared_ptr<Slot> slot;DaxaFrame(std::shared_ptr<Impl> o,std::shared_ptr<Slot> s):owner(std::move(o)),slot(std::move(s)){};};
struct DaxaPreprocess {std::shared_ptr<Impl> impl_;DaxaFrame enqueue_at(CaptureLease,uint64_t);size_t collect_completed_sources_at(uint64_t);};
PRODUCTION_FUNCTIONS
}
using namespace stereoforge;
int main(int argc,char** argv){try{
 if(argc>1&&std::string(argv[1])=="--reference"){
  require(argc==8,"reference arguments");DaxaRectification c;c.eye_width=unsigned(std::stoul(argv[2]));c.source_height=unsigned(std::stoul(argv[3]));c.width=unsigned(std::stoul(argv[4]));c.height=unsigned(std::stoul(argv[5]));c.stride_bytes=unsigned(std::stoul(argv[6]));
  CaptureFrame frame;frame.source_width=c.eye_width*2;frame.height=c.source_height;frame.stride_bytes=c.stride_bytes;frame.calibration_identity=c.original_identity;frame.packed_yuyv.resize(size_t(c.stride_bytes)*c.source_height);
  std::ifstream f(argv[7],std::ios::binary);f.read(reinterpret_cast<char*>(frame.packed_yuyv.data()),frame.packed_yuyv.size());for(auto& map:c.maps){map.resize(size_t(c.width)*c.height);f.read(reinterpret_cast<char*>(map.data()),map.size()*sizeof(float));}require(bool(f),"fixture data read");auto out=daxa_preprocess_reference(frame,c);for(const auto& plane:out.planes)std::cout.write(reinterpret_cast<const char*>(plane.data()),plane.size()*4);return 0;
 }
 require(detail::available_slot(1,0,0)&&detail::available_slot(1,5,5),"free completed slot");
 require(!detail::available_slot(2,5,5)&&!detail::available_slot(1,6,5),"retained or inflight slot");
 DaxaPreprocess e;e.impl_=std::make_shared<Impl>();auto& impl=*e.impl_;impl.context=std::make_shared<Context>();impl.calibration=std::make_shared<DaxaRectification>();auto& c=*impl.calibration;c.eye_width=1280;c.source_height=720;c.stride_bytes=5120;c.width=1280;c.height=720;
 require(c.requested_bytes(1)==66355328&&c.requested_bytes(2)==103219456,"count-dependent retained maps/staging/planes memory budget");
 std::array<std::vector<uint8_t>,2> staging;for(unsigned i=0;i<2;++i){staging[i].resize(size_t(c.stride_bytes)*c.source_height);impl.context->device.address[i]=staging[i].data();auto slot=std::make_shared<Slot>();slot->buffers[Staging]=i;slot->graph=std::make_unique<Graph>();slot->graph->device=&impl.context->device;impl.slots.push_back(std::move(slot));}
 auto source=std::make_shared<CaptureFrame>();source->source_width=2560;source->height=720;source->stride_bytes=5120;source->calibration_identity=c.original_identity;source->packed_yuyv.resize(5120*720,173);source->controls_json=std::string(4096,'x');source->imu_samples.resize(128);
 auto rejected=[&](CaptureLease input,uint64_t oldest){auto submitted=impl.context->device.submitted;bool threw=false;try{e.enqueue_at(std::move(input),oldest);}catch(const std::exception&){threw=true;}require(threw&&impl.context->device.submitted==submitted,"rejected admission cannot submit work");};
 auto bad=std::make_shared<CaptureFrame>(*source);bad->calibration_identity="other";rejected(bad,1);bad->calibration_identity=c.original_identity;bad->stride_bytes+=1;rejected(bad,1);bad->stride_bytes=c.stride_bytes;bad->packed_yuyv.pop_back();rejected(bad,1);rejected({},1);
 auto one=[&]{auto oldest=impl.context->device.submitted;e.collect_completed_sources_at(oldest);auto frame=e.enqueue_at(source,oldest);require(frame.slot->source.get()==source.get(),"source metadata must use the original CaptureLease");auto& copied=staging[frame.slot->buffers[Staging]];asm volatile("" : : "r"(copied.data()) : "memory");require(copied.back()==173,"copied payload");};
 for(unsigned i=0;i<16;++i)one();allocations=0;tracking=true;auto start=std::chrono::steady_clock::now();for(unsigned i=0;i<256;++i)one();auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();tracking=false;
 {auto first=e.enqueue_at(source,impl.context->device.submitted);auto second=e.enqueue_at(source,impl.context->device.submitted);require(e.collect_completed_sources_at(impl.context->device.submitted)==0,"retained GPU slots cannot retire source");rejected(source,impl.context->device.submitted);}
 rejected(source,0);require(e.collect_completed_sources_at(impl.context->device.submitted)==2,"released completed slots retire in bounded scan");
 std::cout<<"{\"frames\":256,\"bytes_per_frame\":"<<source->packed_yuyv.size()<<",\"application_allocations\":"<<allocations<<",\"admission_copy_retirement_mean_ms\":"<<ms/256<<"}\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what();return 1;}}
'''


def definition(source, signature):
    return signature + '{' + body(source, signature) + '}'


@pytest.fixture(scope='module')
def cpu_probe(tmp_path_factory):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    source = PREPROCESS.read_text()
    helpers = '\n'.join(definition(FRAME.read_text(), signature) for signature in (
        'inline bool completed_submission(uint64_t submitted,uint64_t oldest_pending)',
        'inline bool available_slot(long references,uint64_t submitted,uint64_t oldest_pending)'))
    functions = '\n'.join(definition(text, signature) for text, signature in (
        ((ROOT/'src/capture.cpp').read_text(), 'void CaptureFrame::require_calibration()const'),
        (source, 'uint32_t rgb(uint8_t y,uint8_t u,uint8_t v)'),
        (source, 'int quantize(float map)'),
        (source, 'uint32_t gray(uint32_t c)'),
        (source, 'uint64_t DaxaRectification::requested_bytes(unsigned slots)const'),
        (source, 'void DaxaRectification::validate(unsigned slots)const'),
        (source, 'void DaxaRectification::validate_source(const CaptureFrame& f)const'),
        (source, 'DaxaPreprocessReadback daxa_preprocess_reference(const CaptureFrame& f,const DaxaRectification& c)'),
        (source, 'size_t DaxaPreprocess::collect_completed_sources_at(uint64_t oldest)'),
        (source, 'DaxaFrame DaxaPreprocess::enqueue_at(CaptureLease source,uint64_t oldest)')))
    directory = tmp_path_factory.mktemp('daxa-preprocess-cpu')
    cpp = directory/'probe.cpp'
    cpp.write_text(SHIM.replace('HELPERS', helpers).replace('PRODUCTION_FUNCTIONS', functions))
    binary = directory/'probe'
    result = subprocess.run([compiler, '-std=c++20', '-O3', '-I', str(ROOT/'include'), str(cpp), '-o', str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return binary


def test_actual_admission_copy_and_retirement_allocate_no_cpp_heap(cpu_probe):
    result = subprocess.run([str(cpu_probe)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    print(json.dumps(report, sort_keys=True))
    assert report['frames'] == 256 and report['bytes_per_frame'] == 1280*4*720
    assert report['application_allocations'] == 0
    assert report['admission_copy_retirement_mean_ms'] >= 0


def test_actual_cpu_reference_matches_installed_opencv_and_nonzero_tap_masks(cpu_probe, tmp_path):
    cv2 = pytest.importorskip('cv2')
    assert cv2.__version__ == '4.14.0', 'qualified CPU profile is explicitly OpenCV4.14.0'
    rng = np.random.default_rng(20261027)
    width, height = 14, 11
    packed = rng.integers(0, 256, (height, width*2, 2), dtype=np.uint8)
    yy, xx = np.indices((height, width), dtype=np.float32)
    maps = [xx.copy(), yy.copy(), xx+np.float32(.17), yy-np.float32(.39)]
    # Explicit signed half-ties and partial black-border blends.
    maps[2][0, :6] = np.array([-1.5/32, -.5/32, .5/32, 13, 13.5, 14], np.float32)
    fixture = tmp_path/'reference.bin'
    fixture.write_bytes(packed.tobytes()+b''.join(m.astype('<f4').tobytes() for m in maps))
    result = subprocess.run([str(cpu_probe), '--reference', str(width), str(height), str(width), str(height), str(width*4), str(fixture)], capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode()
    actual = np.frombuffer(result.stdout, '<u4').reshape(8, height, width)
    rgb = cv2.cvtColor(packed, cv2.COLOR_YUV2RGB_YUY2)
    for eye in (0, 1):
        original = rgb[:, eye*width:(eye+1)*width]
        rectified = cv2.remap(original, maps[eye*2], maps[eye*2+1], cv2.INTER_LINEAR)
        for index, expected in ((eye, original), (2+eye, rectified)):
            expected_packed = (expected[..., 0].astype(np.uint32) | expected[..., 1].astype(np.uint32)<<8 | expected[..., 2].astype(np.uint32)<<16 | np.uint32(0xff000000))
            np.testing.assert_array_equal(actual[index], expected_packed)
        np.testing.assert_array_equal(actual[4+eye], cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY).astype(np.uint32))
        qx = np.rint(maps[eye*2]*32).astype(np.int64)
        qy = np.rint(maps[eye*2+1]*32).astype(np.int64)
        x, y, fx, fy = qx>>5, qy>>5, qx&31, qy&31
        expected_support = np.ones((height, width), bool)
        for dx, dy, weight in ((0,0,(32-fx)*(32-fy)), (1,0,fx*(32-fy)), (0,1,(32-fx)*fy), (1,1,fx*fy)):
            expected_support &= (weight == 0) | ((x+dx >= 0) & (x+dx < width) & (y+dy >= 0) & (y+dy < height))
        np.testing.assert_array_equal(actual[6+eye], expected_support.astype(np.uint32))
    assert np.all(actual[6] == 1), 'identity edges ignore out-of-range zero-weight taps'
