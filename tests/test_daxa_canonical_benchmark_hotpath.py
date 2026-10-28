"""Actual benchmark loop CPU control only: mocked GPU calls cannot measure GPU speed."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

import pytest
from test_kernel_variant_hotpath import body
from test_zed_native_input import native_record

ROOT = Path(__file__).resolve().parents[1]
APP = 'apps/daxa_depth_rectified_benchmark.cpp'
ARTIFACTS = ROOT / 'artifacts/zed-task3/native-input-performance'
LOOP = 'for(unsigned i=0;i<frames+warmup;++i)'
FORBIDDEN = r'\b(?:new|malloc|calloc|realloc|make_shared|make_unique|resize|reserve|push_back|emplace_back|readback|wait_idle|wait_on_submit|load_canonical_record|verify_canonical_record_files|shader_hashes|daxa_preprocess_file_sha256|write|read|open|ifstream|ofstream)\b'


def check_loop(source):
    loop = body(source, LOOP)
    assert not re.search(FORBIDDEN, loop), 'forbidden per-frame work'
    assert loop.count('.wait()') == 2, 'only the mutually exclusive final output waits allowed'
    submit = loop.index('if(refinement)output=refinement->process(stereo,input.full());')
    wait = loop.index('if(refinement)output.wait();else stereo.wait();')
    assert submit < wait < loop.index('auto c=Clock::now();'), 'wait must follow all stages'
    assert '.wait()' not in loop[:wait], 'interstage wait'
    assert loop.index('auto pt=producer->timings(input)') > wait
    assert loop.index('output={};') < loop.index('stereo={};') < loop.index('input={};') < loop.index('input=producer->enqueue(record.frame)')
    assert 'samples[k][i-warmup]=row[k]' in loop
    assert 'stereo_stages[eye][k][i-warmup]=' in loop
    prefix = source[:source.index(LOOP)]
    assert 'for(auto& column:samples)column.resize(frames);' in prefix
    assert 'for(auto& eye:stereo_stages)for(auto& stage:eye)stage.resize(frames);' in prefix
    assert source.count('load_canonical_record(') == 1
    assert source.index('load_canonical_record(') < source.index('auto context=') < source.index(LOOP)
    assert source.index('verify_canonical_record_files(record)') > source.index('context->close()')
    for term in ('bits==pt.timestamp_bits', 'period==pt.timestamp_period', 'end-pt.start_tick', 'wrap_ns', 'std::isfinite(combined)'):
        assert term in loop


def test_reused_transitive_producer_depth_refinement_retention_and_hotpaths():
    from test_daxa_rectified_input_hotpath import checked_hotpaths, sources
    checked_hotpaths(sources())


def test_actual_loop_has_one_final_boundary_and_preallocated_storage():
    check_loop((ROOT / APP).read_text())


@pytest.mark.parametrize('injection', ['stereo.wait();', 'input.wait();', 'producer->readback(input);', 'samples[0].resize(frames);', 'samples[0].push_back(1);', 'load_canonical_record("x");', 'shader_hashes("x","y");', 'daxa_preprocess_file_sha256("x");', 'auto p=new int;'])
def test_mutations_reject_interstage_wait_alloc_io_and_readback(injection):
    source = (ROOT / APP).read_text()
    source = source.replace('auto b=Clock::now();', injection+'auto b=Clock::now();', 1)
    with pytest.raises(AssertionError):
        check_loop(source)


def check_depth_timing(source):
    fn = body(source, 'DaxaDepthTiming DaxaDepthPipeline::Frame::timings()const')
    assert fn.count('daxa_timeline_query_pool_query_results(') == 1
    assert not re.search(r'\b(?:execute|submit|wait|write_timestamp|create_\w+)\s*\(', fn)
    for term in ('out.start_tick=stereo.start_tick', 'out.end_tick=result[6]', 'out.timestamp_period=owner_->timestamp_period', 'out.timestamp_bits=owner_->timestamp_bits'):
        assert term in fn


def test_depth_exports_existing_query_domain_without_extra_work():
    check_depth_timing((ROOT / 'src/daxa_depth.cpp').read_text())


@pytest.mark.parametrize('injection', ['graph->execute();', 'daxa_timeline_query_pool_query_results(0);', 'frame.wait();'])
def test_mutations_reject_added_timing_queries_submit_and_wait(injection):
    source = (ROOT / 'src/daxa_depth.cpp').read_text()
    source = source.replace('DaxaDepthTiming out;', injection+'DaxaDepthTiming out;', 1)
    with pytest.raises(AssertionError):
        check_depth_timing(source)


MOCKS = r'''
#include <array>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <iostream>
#include <algorithm>
static uint64_t allocations=0,waits=0,producer_calls=0,depth_calls=0,refine_calls=0,violations=0;
static bool tracked=false,refined=false;static unsigned phase=0;
void* operator new(std::size_t n){if(tracked)++allocations;auto p=std::malloc(n);if(!p)throw std::bad_alloc();return p;}
void operator delete(void*p)noexcept{std::free(p);}void operator delete(void*p,std::size_t)noexcept{std::free(p);}
void* operator new[](std::size_t n){return ::operator new(n);}void operator delete[](void*p)noexcept{::operator delete(p);}
using Clock=std::chrono::steady_clock;
void check(bool v,const char*){if(!v)std::abort();}
struct DaxaStereoTiming{static constexpr unsigned max_stages=16;unsigned stage_count=3;double total_gpu_ms=.1;std::array<double,max_stages> stage_gpu_ms{};};
struct Timing{uint64_t start_tick=100,end_tick=400;unsigned timestamp_bits=64;double timestamp_period=1,upload_ms=.1,original_ms=.1,rectified_ms=.1,coarse_ms=.1,total_ms=.1,total_gpu_ms=.1,lr_gpu_ms=.1,metric_metadata_gpu_ms=.1,stereo_to_metric_gpu_ms=.1;DaxaStereoTiming left,right;std::array<double,5> stage_gpu_ms{};};
using DaxaRefinementTiming=Timing;
struct Frame{int tag=0;Frame full()const{return *this;}Frame coarse()const{return *this;}void wait()const{++waits;if(phase!=(refined?3u:2u)||tag!=(refined?3:2))++violations;phase=4;}Timing timings()const{if(phase!=4)++violations;return {};}};
struct DaxaRectifiedInput{using Frame=::Frame;void collect(){}Frame enqueue(int){phase=1;++producer_calls;return {1};}Timing timings(Frame){if(phase!=4)++violations;return {};}double last_upload_cpu_ms(){return .01;}};
struct DaxaDepthPipeline{using Frame=::Frame;void collect(){}Frame enqueue_preprocessed(Frame input){if(phase!=1||input.tag!=1)++violations;phase=2;++depth_calls;return {2};}};
struct DaxaRefinement{using Frame=::Frame;void collect(){}bool can_process(){return true;}Frame process(Frame stereo,Frame input){if(phase!=2||stereo.tag!=2||input.tag!=1)++violations;phase=3;++refine_calls;return {3};}};
'''


def test_extracted_complete_loop_cpu_microprobe(tmp_path):
    source = (ROOT / APP).read_text()
    check_loop(source)
    # Extract declarations, preallocation, handle lifetimes and complete loop.
    start = source.index('    enum : unsigned {Enqueue')
    end = source.index('    auto calibration=full.metric_calibration();', start)
    fragment = source[start:end]
    program = MOCKS + r'''
int main(){for(bool refine:{false,true}){
 const unsigned frames=300,warmup=30,repetitions=100;
 DaxaRectifiedInput p;DaxaDepthPipeline d;DaxaRefinement r;
 auto producer=&p;auto depth=&d;auto refinement=refine?&r:nullptr;
 struct{int frame=0;}record;
 std::array<double,repetitions> costs{};
 allocations=waits=producer_calls=depth_calls=refine_calls=violations=0;refined=refine;
 for(unsigned rep=0;rep<repetitions;++rep){
''' + fragment.replace(LOOP, 'std::array<double const*,Count> storage{};for(unsigned k=0;k<Count;++k)storage[k]=samples[k].data();auto control_start=Clock::now();tracked=true;\n    '+LOOP, 1) + r'''
 tracked=false;for(unsigned k=0;k<Count;++k)check(storage[k]==samples[k].data()&&samples[k].size()==frames,"sample storage moved");costs[rep]=std::chrono::duration<double,std::nano>(Clock::now()-control_start).count()/(frames+warmup);
 }
 std::sort(costs.begin(),costs.end());
 std::cout<<(refine?"coarse_refine":"full_sgm")<<" "<<costs[49]<<" "<<costs[94]<<" "<<allocations<<" "<<waits<<" "<<producer_calls<<" "<<depth_calls<<" "<<refine_calls<<" "<<violations<<"\n";
 if(allocations||violations||waits!=repetitions*(frames+warmup))return 2;
}}
'''
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    probe = tmp_path / 'probe.cpp'; exe = tmp_path / 'probe'
    probe.write_text(program)
    command = [compiler, '-std=c++20', '-O3', str(probe), '-o', str(exe)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    rows = []
    for line in result.stdout.splitlines():
        variant,p50,p95,alloc,wait,producer,depth,refine,violations = line.split()
        row = dict(variant=variant, control_ns_per_iteration_p50=float(p50), control_ns_per_iteration_p95=float(p95), warm_allocations=int(alloc), final_wait_calls=int(wait), producer_calls=int(producer), depth_calls=int(depth), refinement_calls=int(refine), stage_order_violations=int(violations))
        assert row['warm_allocations'] == row['stage_order_violations'] == 0
        assert row['final_wait_calls'] == row['producer_calls'] == row['depth_calls'] == 33000
        assert row['refinement_calls'] == (33000 if variant == 'coarse_refine' else 0)
        rows.append(row)
    assert len(rows) == 2
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / 'benchmark-loop-probe.cpp').write_text(program)
    paths = [APP, 'src/canonical_record.cpp', 'src/daxa_depth.cpp', 'include/stereoforge/daxa_depth.hpp', 'benchmarks/zed_comparison/native_input.py']
    report = dict(scope='Actual complete benchmark loop and preallocation extracted verbatim; mock producer/depth/refinement and timestamps; CPU control only, no Vulkan/driver/GPU/upload/loader/quality or FPS qualification.', frames=300, warmup=30, repetitions=100, rows=rows, forbidden_per_frame_operation_matches=0, explicit_wait_boundary='One final-output wait after all submissions, per iteration including warmup; mock wait has no blocking cost.', source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}, compiler_command=command)
    (ARTIFACTS / 'benchmark-loop-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(result.stdout)


def test_actual_loader_setup_only_admission_cost(native_record, json_include, tmp_path):
    from benchmarks.zed_comparison.native_input import write_native_input
    _,record,_,supplement = native_record
    manifest = write_native_input(record, supplement, tmp_path/'bundle')
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    code = r'''
#include <stereoforge/canonical_record.hpp>
#include <chrono>
#include <iostream>
int main(int argc,char**argv){try{
 auto a=std::chrono::steady_clock::now();
 auto record=stereoforge::load_canonical_record(argv[1]);
 auto b=std::chrono::steady_clock::now();
 uint64_t bytes=0;for(auto const& f:record.files)bytes+=f.size_bytes;
 if(!record.frame.admitted())return 2;
 auto c=std::chrono::steady_clock::now();
 stereoforge::verify_canonical_record_files(record);
 auto d=std::chrono::steady_clock::now();
 std::cout<<std::chrono::duration<double,std::milli>(b-a).count()<<" "<<std::chrono::duration<double,std::milli>(d-c).count()<<" "<<record.files.size()<<" "<<bytes<<"\n";
}catch(std::exception const&e){std::cerr<<e.what();return 1;}}
'''
    probe = tmp_path/'loader-probe.cpp'; exe = tmp_path/'loader-probe'
    probe.write_text(code)
    command = [compiler, '-std=c++20', '-O3', '-pthread', '-I'+str(ROOT/'include'), '-I'+str(json_include), str(probe), str(ROOT/'src/canonical_record.cpp'), str(ROOT/'src/daxa_calibration.cpp'), str(ROOT/'src/capture.cpp'), '-lcrypto', '-o', str(exe)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe), str(manifest)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    admission,verification,files,size = result.stdout.split()
    assert int(files) == 13 and int(size) > 0
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS/'loader-probe.cpp').write_text(code)
    report = dict(scope='First in-process admission of tiny 4x4 fixture using actual complete canonical_record/calibration/capture translation units; filesystem cache is warm after Python exporter. Setup-only cost; no cold disk-cache, 720p, steady frame, GPU, quality or FPS claim.', admission_ms=float(admission), after_run_verification_ms=float(verification), admitted_files=int(files), admitted_bytes=int(size), frame_load_calls=0, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['src/canonical_record.cpp','src/daxa_calibration.cpp','src/capture.cpp','benchmarks/zed_comparison/native_input.py']}, compiler_command=command)
    (ARTIFACTS/'loader-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(result.stdout)


def test_report_distinguishes_sgm_and_final_subpixel_settings():
    source = (ROOT / APP).read_text()
    assert '{"subpixel",refine?refinement_config.subpixel:depth_config.stereo.subpixel}' in source
    assert '{"sgm_subpixel",depth_config.stereo.subpixel}' in source
    assert 'result["refinement_subpixel"]=refinement_config.subpixel;' in source


def test_instantaneous_metadata_is_reported_after_measured_loop():
    src=(ROOT/'apps/daxa_depth_rectified_benchmark.cpp').read_text()
    assert 'DaxaTemporal' not in src
    assert 'result["temporal"]' not in body(src, LOOP)
    assert src.index('result["temporal"]') > src.index('auto calibration=full.metric_calibration()')
