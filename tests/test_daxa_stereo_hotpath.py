"""CPU-only functional/performance checks; these never construct a Vulkan device."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]


def check_resident_hotpath(source):
    process = body(source, 'DaxaStereo::Frame DaxaStereo::process(')
    timing = body(source, 'DaxaStereoTiming DaxaStereo::Frame::timings(')
    forbidden = r'\b(?:allocate|create_buffer|create_memory|create_compute_pipeline|create_command_recorder|make_shared|make_unique|get_query_results|wait_on_submit|wait_idle|readback|inspect|memcpy|malloc|calloc|realloc)(?:<[^;()]*>)?\s*\('
    assert not re.search(forbidden, process), 'resident process must not allocate, wait, or read back'
    assert process.count('->execute(') == 1
    assert 'collect_garbage(' in process, 'completed Daxa command resources must be retired'
    assert 'use_count()' in process and 'oldest_pending_submit_index(' in process, 'reuse needs both lease release and GPU completion'
    assert 'std::array<uint64_t,' in timing and 'daxa_timeline_query_pool_query_results(' in timing
    assert not re.search(r'get_query_results\s*\(|wait_on_submit\s*\(|std::vector', timing), 'timestamp access is nonblocking and uses caller-owned stack storage'


def test_resident_processing_and_timestamp_queries_have_no_own_heap_or_transfer_work():
    check_resident_hotpath((ROOT / 'src/daxa_stereo.cpp').read_text())


@pytest.mark.parametrize('regression', ['d.wait_idle();', 'std::make_shared<int>();', 'std::memcpy(a,b,n);'])
def test_resident_guard_detects_new_wait_allocation_or_full_copy(regression):
    source = (ROOT / 'src/daxa_stereo.cpp').read_text()
    marker = 'DaxaStereo::Frame DaxaStereo::process(){'
    assert marker in source
    with pytest.raises(AssertionError, match='must not allocate'):
        check_resident_hotpath(source.replace(marker, marker + regression, 1))


def test_benchmark_keeps_consumer_and_full_reference_export_outside_frame_timing():
    source = (ROOT / 'apps/daxa_stereo_benchmark.cpp').read_text()
    measured = body(source, 'for(uint32_t i=0;i<frames;++i)')
    for forbidden in ('inspect(', 'readback(', 'daxa_stereo_reference(', 'std::vector', 'resize('):
        assert forbidden not in measured
    assert measured.index('process()') < measured.index('frame.wait()') < measured.index('frame.timings()')
    assert source.index('inspect(frame)') > source.index(measured) + len(measured)


def test_wave_line_identity_does_not_assume_local_ids_equal_subgroup_ids():
    source = (ROOT / 'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    wave = body(source, 'void aggregate_wave(')
    assert not re.search(r'thread\.x\s*/\s*lanes', wave), 'Vulkan local invocation IDs do not determine subgroup membership'
    assert 'WaveGetWaveIndex()' in wave or 'SubgroupId' in source, 'wave line identity must use the real subgroup builtin'


def test_wave_pipeline_requires_full_supported_subgroups_and_explicit_size():
    source = (ROOT / 'src/daxa_stereo.cpp').read_text()
    setup = body(source, 'void DaxaStereo::Impl::setup(')
    assert 'ShaderCreateFlagBits::REQUIRE_FULL_SUBGROUPS' in setup
    assert 'shader.required_subgroup_size=wave_size' in setup
    for capability in ('computeFullSubgroups', 'requiredSubgroupSizeStages', 'maxComputeWorkgroupSubgroups'):
        assert capability in source


def test_benchmark_checks_diagnostics_after_frame_resource_and_device_teardown():
    source = (ROOT / 'apps/daxa_stereo_benchmark.cpp').read_text()
    release = source.index('frame={};engine.reset();context->close();')
    assert release < source.index('context->validation_errors.load()')
    assert release < source.index('context->validation_warnings.load()')
    assert 'if(errors||warnings)' in source


def check_packed_resource_and_word_ownership(source, shader):
    setup = body(source, 'void DaxaStereo::Impl::setup(')
    assert 'if(packing&&(b==Costs||b==Sums))continue;' in setup, 'packed setup must omit both dense planes'
    assert 's->buffers[Penalties]=allocate(' in setup
    packed = body(shader, 'void aggregate_packed_body<')
    assert 'WaveGetWaveIndex()' in packed
    assert 'Interlocked' not in packed, 'one lane owns the whole packed word; atomics are unnecessary'
    assert 'penalties[i*row_words+word]+=addition' in packed, 'accumulation must update the owned whole word'
    assert packed.index('WaveReadLaneAt(') < packed.index('if(d<p.disparities)'), 'masked tail lanes must participate in neighbor shuffles'


def test_packed_variant_omits_dense_planes_and_accumulates_owned_words():
    check_packed_resource_and_word_ownership((ROOT/'src/daxa_stereo.cpp').read_text(), (ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text())


def test_packed_resource_guard_catches_restoring_dense_allocations():
    source = (ROOT/'src/daxa_stereo.cpp').read_text()
    shader = (ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    branch = 'if(packing&&(b==Costs||b==Sums))continue;'
    assert branch in source
    with pytest.raises(AssertionError, match='omit both dense'):
        check_packed_resource_and_word_ownership(source.replace(branch,''),shader)


def test_packed_word_guard_catches_unnecessary_per_frame_atomics():
    source = (ROOT/'src/daxa_stereo.cpp').read_text()
    shader = (ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    marker = 'void aggregate_packed_body<let Components:uint,let Bits:uint>(uint3 gid){'
    assert marker in shader
    with pytest.raises(AssertionError, match='atomics are unnecessary'):
        check_packed_resource_and_word_ownership(source,shader.replace(marker,marker+'InterlockedAdd(penalties[0],0);'))


def check_specialized_address_arithmetic(shader):
    cost = body(shader, 'uint specialized_cost(')
    summed = body(shader, 'uint specialized_sum<')
    aggregate = body(shader, 'void aggregate_specialized_body<')
    select = body(shader, 'void select_specialized_body<')
    assert not re.search(r'[%/]|\bp\.',cost), 'specialized cost must use known x and hoisted inputs, without per-disparity division'
    assert 'x>=d&&reference_valid' in cost
    assert cost.index('if(supported)') < cost.index('right_valid[i-d]')
    assert 'packing_bits' not in summed and 'd/Components' in summed and 'd%Components' in summed, 'selection layout must use compile-time Components/Bits'
    assert 'word_base' in summed and 'specialized_cost(x,i,d,' in summed
    assert aggregate.index('uint reference=left_desc[i]') < aggregate.index('uint c=specialized_cost(')
    assert 'x=uint(xy.x)' in aggregate and 'WaveGetWaveIndex()' in aggregate
    assert select.index('uint x=i%width,word_base=') < select.index('for(uint d=lane;')
    assert 'packed_sum(' not in select
    for name,layout in (('aggregate_specialized8','4,8'),('aggregate_specialized16','2,16'),('select_specialized8','4,8'),('select_specialized16','2,16')):
        entry = body(shader,'void '+name+'(')
        assert '<'+layout+'>' in entry


def test_specialized_variant_hoists_pixel_inputs_and_selects_constant_layouts():
    check_specialized_address_arithmetic((ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text())


@pytest.mark.parametrize('regression', ['uint discarded=i%p.width;', 'uint discarded=i/p.width;'])
def test_specialized_guard_catches_restored_per_disparity_division(regression):
    shader=(ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    marker='uint specialized_cost(uint x,uint i,uint d,uint reference,bool reference_valid,uint* right_valid,uint* right_desc){'
    assert marker in shader
    with pytest.raises(AssertionError,match='without per-disparity division'):
        check_specialized_address_arithmetic(shader.replace(marker,marker+regression))


def test_specialized_guard_catches_runtime_packing_layout():
    shader=(ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    marker='uint specialized_sum<let Components:uint,let Bits:uint>(uint x,uint i,uint d,uint word_base,uint paths,uint reference,bool reference_valid,uint* right_valid,uint* right_desc,uint* words){'
    assert marker in shader
    with pytest.raises(AssertionError,match='compile-time'):
        check_specialized_address_arithmetic(shader.replace(marker,marker+'uint discarded=p.packing_bits;'))


def check_d128_state_and_admission(source,shader):
    admission=body(source,'void DaxaStereo::validate_shape(')
    assert 'variant!=DaxaSgmVariant::PackedD128||c.disparities==128' in admission, 'D128 must reject every other disparity count'
    assert 'validate_shape(c,slots,metric,variant)' in body(source,'void DaxaStereo::validate('), 'full input validation must retain shape admission'
    constructor=body(source,'DaxaStereo::DaxaStereo(')
    assert constructor.index('validate(')<constructor.index('std::make_shared<Impl>'), 'admission precedes device ownership'
    aggregate=body(shader,'void aggregate_d128_body<')
    assert 'disparities=128' in aggregate and 'p.disparities' not in aggregate, 'D128 aggregation must have a compile-time disparity bound'
    assert 'prev[4],next[4]' in aggregate and '8/Components' not in aggregate, 'D128 has at most four components of recurrence state'
    assert 'k<4/Components' in aggregate and 'r<4' in aggregate
    assert 'WaveGetWaveIndex()' in aggregate and 'Interlocked' not in aggregate
    assert aggregate.index('WaveReadLaneAt(')<aggregate.index('if(d<disparities)')
    assert 'words[i*row_words+word]+=addition' in aggregate
    for entry,layout in (('aggregate_d1288','4,8'),('aggregate_d12816','2,16')):
        assert '<'+layout+'>' in body(shader,'void '+entry+'(')
    assert 'select_d128' not in shader, 'D reuses the unchanged C selection entries'


def test_d128_has_bounded_state_and_rejects_other_dimensions_before_device():
    check_d128_state_and_admission((ROOT/'src/daxa_stereo.cpp').read_text(),(ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text())


def test_d128_guard_detects_lost_admission_check():
    source=(ROOT/'src/daxa_stereo.cpp').read_text()
    shader=(ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    condition='variant!=DaxaSgmVariant::PackedD128||c.disparities==128'
    assert condition in source
    with pytest.raises(AssertionError,match='reject every other'):
        check_d128_state_and_admission(source.replace(condition,'true'),shader)


@pytest.mark.parametrize('old,new,diagnostic',[('disparities=128','disparities=p.disparities','compile-time'),('prev[4],next[4]','prev[8],next[8]','four components')])
def test_d128_guard_detects_generic_state_regressions(old,new,diagnostic):
    source=(ROOT/'src/daxa_stereo.cpp').read_text()
    shader=(ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    start=shader.index('void aggregate_d128_body<')
    with pytest.raises(AssertionError,match=diagnostic):
        check_d128_state_and_admission(source,shader[:start]+shader[start:].replace(old,new))


def check_setup_preflight(source,app):
    shape=body(source,'void DaxaStereo::validate_shape(')
    assert not re.search(r'\b(?:make_shared|make_unique|create_buffer|create_memory|resize|reserve|malloc|new)\b',shape), 'shape preflight must remain pure and allocation-free'
    main=body(app,'int main(')
    assert main.index('DaxaStereo::validate_shape(')<main.index('std::array<std::vector<uint32_t>,2> luma'), 'shape preflight must precede full luma allocation/read'
    assert main.index('DaxaStereo::validate_shape(')<main.index('std::make_shared<DaxaFoundation>'), 'shape preflight must precede device construction'
    measured=body(app,'for(uint32_t i=0;i<frames;++i)')
    assert 'validate_shape' not in measured and 'weakly_canonical' not in measured and 'fresh_destination' not in measured


def test_cli_shape_preflight_stays_before_large_inputs_and_device():
    check_setup_preflight((ROOT/'src/daxa_stereo.cpp').read_text(),(ROOT/'apps/daxa_stereo_benchmark.cpp').read_text())


def test_cli_guard_catches_preflight_moved_after_input_allocation():
    source=(ROOT/'src/daxa_stereo.cpp').read_text()
    app=(ROOT/'apps/daxa_stereo_benchmark.cpp').read_text()
    call='DaxaStereo::validate_shape(config,slots,metric,selected_variant);'
    assert call in app
    app=app.replace(call,'').replace('std::array<std::vector<uint32_t>,2> luma;','std::array<std::vector<uint32_t>,2> luma;'+call)
    with pytest.raises(AssertionError,match='precede full luma'):
        check_setup_preflight(source,app)


def check_quad_collectives(shader):
    quad=body(shader,'void select_quad_body<')
    assert 'WaveGetWaveIndex()*(wave_size/4)+wave_lane/4' in quad, 'quad pixels must use actual subgroup and lane identity'
    assert 'linear_group(gid)*16' in quad and 'WaveGetLaneIndex()' in quad
    assert not re.search(r'GroupMemoryBarrier|groupshared|Interlocked',quad), 'quad selection must not restore LDS, barriers or atomics'
    assert quad.index('return;')>quad.rindex('WaveReadLaneAt('), 'all tail lanes must execute the uniform shuffles before return'
    assert 'source=wave_lane^stride' in quad and 'for(uint stride=2;stride>0;stride/=2)' in quad
    assert quad.rindex('WaveReadLaneAt(')<quad.index('if(lane<stride)'), 'merge predicates must not mask shuffle execution'
    assert 'if(i<width*p.height)' in quad and 'if(lane!=0||i>=width*p.height)return;' in quad
    for entry,layout in (('select_quad8','4,8'),('select_quad16','2,16')):
        assert '<'+layout+'>' in body(shader,'void '+entry+'(')


def test_quad_selection_uses_full_active_pixel_groups_without_lds():
    shader=(ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    source=(ROOT/'src/daxa_stereo.cpp').read_text()
    check_quad_collectives(shader)
    setup=body(source,'void DaxaStereo::Impl::setup(')
    assert 'i==17||i==18' in setup[:setup.index('shader.create_flags=')], 'both quad pipelines require full subgroups'
    assert 'PackedQuad?15:63' in setup and 'PackedQuad?16:64' in setup, 'quad dispatch must cover16 pixels per group'


@pytest.mark.parametrize('insert,diagnostic',[('return;','uniform shuffles'),('GroupMemoryBarrierWithGroupSync();','LDS, barriers')])
def test_quad_guard_catches_early_tail_exit_or_restored_barrier(insert,diagnostic):
    shader=(ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    marker='void select_quad_body<let Components:uint,let Bits:uint>(uint3 gid){'
    assert marker in shader
    with pytest.raises(AssertionError,match=diagnostic):
        check_quad_collectives(shader.replace(marker,marker+insert))


def test_quad_guard_catches_assumed_subgroup_identity():
    shader=(ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    start=shader.index('void select_quad_body<')
    with pytest.raises(AssertionError,match='actual subgroup'):
        check_quad_collectives(shader[:start]+shader[start:].replace('WaveGetWaveIndex()','0'))


def check_shuffle_minimum_only(shader):
    original=body(shader,'void aggregate_specialized_body<')
    changed=body(shader,'void aggregate_shuffle_body<')
    primitive=(r'uint minimum=local_min;\s*\[unroll\]for\(uint offset=1;offset<32;offset\*=2\)'
               r'\{uint other=WaveReadLaneAt\(minimum,lane\^offset\);minimum=min\(minimum,other\);\}\s*'
               r'if\(lanes==64\)\{uint other=WaveReadLaneAt\(minimum,lane\^32\);minimum=min\(minimum,other\);\}\s*uint x=')
    restored,count=re.subn(primitive,'uint minimum=WaveActiveMin(local_min),x=',changed)
    assert count==1, 'shuffle butterfly must execute five full-wave steps plus uniform sixth for64'
    assert re.sub(r'\s+','',restored)==re.sub(r'\s+','',original), 'only the minimum primitive may differ from C; preserve neutral tails, recurrence and addressing'
    assert 'WaveActiveMin' not in changed
    for entry,layout in (('aggregate_shuffle8','4,8'),('aggregate_shuffle16','2,16')):
        assert '<'+layout+'>' in body(shader,'void '+entry+'(')


def test_shuffle_variant_changes_only_full_wave_minimum():
    check_shuffle_minimum_only((ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text())
    setup=body((ROOT/'src/daxa_stereo.cpp').read_text(),'void DaxaStereo::Impl::setup(')
    assert 'i==19||i==20' in setup[:setup.index('shader.create_flags=')]


@pytest.mark.parametrize('old,new,diagnostic',[('offset<32','offset<16','five full-wave'),('if(lanes==64)','if(lane==0)','five full-wave'),('uint local_min=SGM_REDUCTION_INF','uint local_min=0','only the minimum')])
def test_shuffle_guard_detects_lost_partners_active_sources_or_neutral_value(old,new,diagnostic):
    shader=(ROOT/'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    start=shader.index('void aggregate_shuffle_body<')
    with pytest.raises(AssertionError,match=diagnostic):
        check_shuffle_minimum_only(shader[:start]+shader[start:].replace(old,new))


def definition(source, signature):
    start = source.index(signature)
    opening = source.index('{', start)
    return source[start:opening] + '{' + body(source, signature) + '}\n'


@pytest.fixture(scope='module')
def scalar_cpu_probe(tmp_path_factory):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable for CPU-only scalar/model probe')
    source = (ROOT / 'src/daxa_stereo.cpp').read_text()
    header = (ROOT / 'include/stereoforge/daxa_stereo.hpp').read_text()
    shader = (ROOT / 'python/stereoforge/shaders/daxa/stereo.slang').read_text()
    ops = (ROOT / 'python/stereoforge/shaders/ops/sgm.slang').read_text()
    # Compile the actual pure CPU definitions with a CPU-only declaration shim.
    # No Daxa headers, Vulkan library, runtime owner, queue or device is involved.
    program = r'''
#include <stereoforge/contracts.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <vector>
std::atomic<bool> tracking=false;std::atomic<unsigned> allocations=0;
void* operator new(std::size_t n){if(tracking)++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);}void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
namespace stereoforge {
enum class DaxaSgmVariant { Baseline,Wave,Packed,PackedSpecialized,PackedD128,PackedQuad,PackedShuffle };
class DaxaStereo {public:static void validate_shape(StereoConfig,unsigned,std::optional<MetricCalibration>,DaxaSgmVariant variant=DaxaSgmVariant::Baseline);static void validate(StereoConfig,std::span<const uint32_t>,std::span<const uint32_t>,unsigned,std::optional<MetricCalibration>,DaxaSgmVariant variant=DaxaSgmVariant::Baseline);static uint32_t packed_bits(StereoConfig);};
'''
    program += definition(header, 'struct DaxaStereoReadback') + ';\n'
    program += definition(header, 'struct DaxaQuadSelection') + ';\n'
    program += 'using uint=uint32_t;static const uint SGM_INVALID_COST=10000;uint countbits(uint x){return std::popcount(x);}\n'
    program += definition(ops, 'uint sgm_cost(') + definition(shader, 'uint specialized_cost(')
    directions = re.search(r'constexpr std::array<std::array<int,2>,8> directions.*?;', source)
    assert directions is not None
    program += 'namespace {\n' + directions.group() + '\n'
    for signature in ('void require(', 'float invalid_float(', 'bool inside(', 'uint32_t step(', 'uint32_t line_count(', 'std::array<int,2> line_start('):
        program += definition(source, signature)
    program += '}\n'
    for signature in ('void DaxaStereo::validate_shape(', 'void DaxaStereo::validate(', 'uint32_t DaxaStereo::packed_bits(', 'std::vector<uint32_t> daxa_sgm_packed_model(', 'std::vector<uint32_t> daxa_sgm_d128_model(', 'std::vector<uint32_t> daxa_wave_min_model(', 'DaxaQuadSelection daxa_sgm_quad_model(', 'std::array<std::vector<uint32_t>,2> daxa_stereo_fixture(',
                      'DaxaStereoReadback daxa_stereo_reference(', 'std::vector<uint32_t> daxa_sgm_wave_model('):
        program += definition(source, signature)
    program += r'''
}
using namespace stereoforge;
int main(int argc,char** argv){
  if(argc==3&&std::string(argv[1])=="--wave-min"){std::ifstream input(argv[2],std::ios::binary);uint count;input.read(reinterpret_cast<char*>(&count),4);std::vector<uint> values(count);input.read(reinterpret_cast<char*>(values.data()),count*4);if(!input)return 13;
    try{auto result=daxa_wave_min_model(values);std::cout.write(reinterpret_cast<const char*>(result.data()),result.size()*4);return std::cout?0:14;}catch(const std::invalid_argument&){return 12;}}
  if(argc==2&&std::string(argv[1])=="--d128-admission"){
    for(uint disparities:{1u,127u,128u,129u,256u}){StereoConfig c{9,7,disparities,4,8,32,true};std::vector<uint> luma(63),costs(63*disparities);bool admitted=true,modeled=true;
      try{DaxaStereo::validate(c,luma,luma,1,std::nullopt,DaxaSgmVariant::PackedD128);}catch(const std::invalid_argument&){admitted=false;}
      try{auto result=daxa_sgm_d128_model(c,costs,32);}catch(const std::invalid_argument&){modeled=false;}
      if(admitted!=(disparities==128)||modeled!=(disparities==128))return 8;
    }std::cout<<"only_d128_admitted\n";return 0;
  }
  if(argc==4){std::ifstream input(argv[2],std::ios::binary);uint32_t parameters[7];input.read(reinterpret_cast<char*>(parameters),sizeof(parameters));
    StereoConfig c{parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],true};std::vector<uint32_t> costs(size_t(c.width)*c.height*c.disparities);
    input.read(reinterpret_cast<char*>(costs.data()),costs.size()*4);if(!input)return 5;
    if(std::string(argv[1])=="--quad"){auto result=daxa_sgm_quad_model(c,costs,parameters[6]);std::ofstream output(argv[3],std::ios::binary);for(auto* values:{&result.best,&result.best_cost,&result.second})output.write(reinterpret_cast<const char*>(values->data()),values->size()*4);return output?0:10;}
    auto packed=std::string(argv[1])=="--d128"?daxa_sgm_d128_model(c,costs,parameters[6]):daxa_sgm_packed_model(c,costs,parameters[6]);
    std::ofstream output(argv[3],std::ios::binary);output.write(reinterpret_cast<const char*>(packed.data()),packed.size()*4);return output?0:6;
  }
  if(argc==2){StereoConfig c{37,11,33,8,8,32,true};auto f=daxa_stereo_fixture(c);auto r=daxa_stereo_reference(c,f[0],f[1],std::nullopt);
    std::vector<uint32_t> valid(f[0].size());for(size_t i=0;i<valid.size();++i)valid[i]=(i%5)!=0;
    for(uint i=0;i<valid.size();++i)for(uint d=0;d<256;++d){uint x=i%c.width;bool supported=x>=d&&valid[i]&&valid[i-d];
      uint expected=supported?uint(std::popcount(f[0][i]^f[1][i-d])):10000u;
      if(specialized_cost(x,i,d,f[0][i],valid[i]!=0,valid.data(),f[1].data())!=expected)return 7;}
    if(daxa_sgm_wave_model(c,r.costs,32)!=r.sums||daxa_sgm_wave_model(c,r.costs,64)!=r.sums)return 2;
    std::ofstream out(argv[1],std::ios::binary);auto write=[&](const auto& v){out.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(v[0]));};
    write(f[0]);write(f[1]);write(r.costs);write(r.sums);write(r.disparity);write(r.confidence);write(r.validity);return out?0:3;
  }
  StereoConfig c{160,90,128,4,8,32,true};auto f=daxa_stereo_fixture(c);auto r=daxa_stereo_reference(c,f[0],f[1],std::nullopt);
  struct Row{double ms;unsigned alloc;uint32_t check;};
  auto measure=[&](auto work){allocations=0;tracking=true;auto start=std::chrono::steady_clock::now();uint32_t check=0;
    for(unsigned i=0;i<3;++i){auto result=work();check^=result[result.size()/2];}
    auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();tracking=false;return Row{elapsed/3,allocations/3,check};};
  auto scalar=measure([&]{return daxa_stereo_reference(c,f[0],f[1],std::nullopt).sums;});
  auto wave32=measure([&]{return daxa_sgm_wave_model(c,r.costs,32);});auto wave64=measure([&]{return daxa_sgm_wave_model(c,r.costs,64);});
  auto packed32=measure([&]{return daxa_sgm_packed_model(c,r.costs,32);});auto packed64=measure([&]{return daxa_sgm_packed_model(c,r.costs,64);});
  auto d12832=measure([&]{return daxa_sgm_d128_model(c,r.costs,32);});auto d12864=measure([&]{return daxa_sgm_d128_model(c,r.costs,64);});
  auto quad32=measure([&]{return daxa_sgm_quad_model(c,r.sums,32).best_cost;});auto quad64=measure([&]{return daxa_sgm_quad_model(c,r.sums,64).best_cost;});
  auto min32=measure([&]{return daxa_wave_min_model(std::span<const uint>(r.sums.data(),32));});auto min64=measure([&]{return daxa_wave_min_model(std::span<const uint>(r.sums.data(),64));});
  if(quad32.check!=quad64.check)return 11;
  if(daxa_sgm_d128_model(c,r.costs,32)!=daxa_sgm_packed_model(c,r.costs,32)||daxa_sgm_d128_model(c,r.costs,64)!=daxa_sgm_packed_model(c,r.costs,64))return 9;
  if(scalar.check!=wave32.check||scalar.check!=wave64.check)return 4;
  std::cout<<"{\"width\":160,\"height\":90,\"disparities\":128,\"paths\":4,\"repetitions\":3,\"scalar_reference_ms\":"<<scalar.ms
    <<",\"wave32_model_ms\":"<<wave32.ms<<",\"wave64_model_ms\":"<<wave64.ms<<",\"scalar_allocations\":"<<scalar.alloc
    <<",\"wave32_allocations\":"<<wave32.alloc<<",\"wave64_allocations\":"<<wave64.alloc
    <<",\"packed32_model_ms\":"<<packed32.ms<<",\"packed64_model_ms\":"<<packed64.ms<<",\"packed32_allocations\":"<<packed32.alloc<<",\"packed64_allocations\":"<<packed64.alloc
    <<",\"d12832_model_ms\":"<<d12832.ms<<",\"d12864_model_ms\":"<<d12864.ms<<",\"d12832_allocations\":"<<d12832.alloc<<",\"d12864_allocations\":"<<d12864.alloc
    <<",\"quad32_model_ms\":"<<quad32.ms<<",\"quad64_model_ms\":"<<quad64.ms<<",\"quad32_allocations\":"<<quad32.alloc<<",\"quad64_allocations\":"<<quad64.alloc
    <<",\"min32_model_ms\":"<<min32.ms<<",\"min64_model_ms\":"<<min64.ms<<",\"min32_allocations\":"<<min32.alloc<<",\"min64_allocations\":"<<min64.alloc<<"}\n";
}
'''
    directory = tmp_path_factory.mktemp('daxa-stereo-cpu')
    code = directory / 'probe.cpp'
    code.write_text(program)
    binary = directory / 'probe'
    built = subprocess.run([compiler, '-std=c++20', '-O3', '-I', str(ROOT / 'include'), str(code), '-o', str(binary)],
                           capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    return binary, directory


def test_scalar_stereo_and_lane_chunk_models_match_the_existing_numpy_oracle(scalar_cpu_probe):
    from stereoforge.reference import census, matching_cost, aggregate_sgm, select_disparity
    binary, directory = scalar_cpu_probe
    output = directory / 'planes.bin'
    ran = subprocess.run([str(binary), str(output)], capture_output=True, text=True, timeout=30)
    assert ran.returncode == 0, ran.stderr
    words = np.fromfile(output, dtype='<u4')
    n, volume = 37 * 11, 37 * 11 * 33
    assert words.size == n * 5 + volume * 2
    left, right = words[:n].reshape(11, 37), words[n:2*n].reshape(11, 37)
    a, av = census(left)
    b, bv = census(right)
    cost = matching_cost(a, b, av, bv, 33)
    sums = aggregate_sgm(cost, paths=8, p1=8, p2=32)
    np.testing.assert_array_equal(words[2*n:2*n+volume].reshape(cost.shape), cost)
    np.testing.assert_array_equal(words[2*n+volume:2*n+2*volume].reshape(sums.shape), sums)
    disparity, confidence, valid = select_disparity(sums, subpixel=True)
    offset = 2*n + 2*volume
    np.testing.assert_array_equal(words[offset:offset+n], disparity.reshape(-1).view(np.uint32))
    np.testing.assert_array_equal(words[offset+n:offset+2*n], confidence.reshape(-1).view(np.uint32))
    np.testing.assert_array_equal(words[offset+2*n:], valid.reshape(-1).astype(np.uint32))


def test_cpu_diagnostic_reference_and_wave_model_microbenchmark(scalar_cpu_probe):
    binary, _ = scalar_cpu_probe
    ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert ran.returncode == 0, ran.stderr
    result = json.loads(ran.stdout)
    print('Daxa stereo pure CPU diagnostic probe:', json.dumps(result, sort_keys=True))
    assert result['repetitions'] == 3
    for field in ('scalar_reference_ms', 'wave32_model_ms', 'wave64_model_ms', 'packed32_model_ms', 'packed64_model_ms', 'd12832_model_ms', 'd12864_model_ms', 'quad32_model_ms', 'quad64_model_ms', 'min32_model_ms', 'min64_model_ms'):
        assert 0 < result[field] < 30_000


@pytest.mark.parametrize('disparities,paths,p2', [(1,4,0),(2,4,63),(3,8,31),(31,4,63),(32,4,64),(33,8,32),(127,4,63),(128,8,8191),(255,8,32),(256,4,64)])
def test_packed_penalty_words_reconstruct_independent_numpy_sums(scalar_cpu_probe, disparities, paths, p2):
    check_packed_penalty_numpy(scalar_cpu_probe,disparities,paths,p2,'--packed')


@pytest.mark.parametrize('paths,p2',[(paths,p2) for paths in (4,8) for p2 in (31,32,64,8191)])
def test_d128_four_component_model_reconstructs_independent_numpy_sums(scalar_cpu_probe,paths,p2):
    check_packed_penalty_numpy(scalar_cpu_probe,128,paths,p2,'--d128')


def test_d128_admission_and_model_reject_other_disparity_counts(scalar_cpu_probe):
    binary,_=scalar_cpu_probe
    result=subprocess.run([str(binary),'--d128-admission'],capture_output=True,text=True,check=True,timeout=30)
    assert result.stdout.strip()=='only_d128_admitted'


@pytest.mark.parametrize('disparities',[1,2,3,4,5,31,32,33,64,127,128,129,255,256])
@pytest.mark.parametrize('paths,p2',[(4,32),(8,64)])
def test_quad_model_matches_numpy_lowest_index_and_second_cost_with_tails(scalar_cpu_probe,disparities,paths,p2):
    binary,directory=scalar_cpu_probe
    height,width=(5,7) if paths==4 else (7,9)
    rng=np.random.default_rng(0x51554144+disparities+paths)
    sums=rng.integers(0,paths*20000+1,size=(height*width,disparities),dtype=np.uint32)
    sums[:4]=rng.integers(0,8,size=(4,disparities),dtype=np.uint32)
    sums[0]=0;sums[1]=paths*20000
    sums[2]=31;sums[2,-1]=0
    sums[3]=31;sums[3,0]=0;sums[3,-1]=0
    best=np.argmin(sums,axis=1).astype(np.uint32)
    best_cost=np.take_along_axis(sums,best[:,None],axis=1)[:,0]
    second=np.partition(sums,1,axis=1)[:,1] if disparities>1 else best_cost
    for lanes in (32,64):
        source=directory/f'quad-{disparities}-{paths}-{lanes}.input';target=source.with_suffix('.output')
        with source.open('wb') as stream:
            np.array([width,height,disparities,paths,8,p2,lanes],dtype='<u4').tofile(stream)
            sums.astype('<u4').tofile(stream)
        subprocess.run([str(binary),'--quad',str(source),str(target)],check=True,timeout=30)
        result=np.fromfile(target,dtype='<u4').reshape(3,height*width)
        for actual,expected in zip(result,(best,best_cost,second)):
            np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('lanes',[32,64])
def test_wave_min_actual_cpu_model_matches_independent_scalar_for_all_lanes(scalar_cpu_probe,lanes):
    binary,directory=scalar_cpu_probe;rng=np.random.default_rng(0x4d494e+lanes)
    samples=[np.full(lanes,1_000_000,dtype=np.uint32),np.zeros(lanes,dtype=np.uint32),np.full(lanes,0xffffffff,dtype=np.uint32)]
    for active in (1,2,3,lanes//2-1,lanes//2,lanes//2+1,lanes-1,lanes):
        values=np.full(lanes,1_000_000,dtype=np.uint32);values[:active]=rng.integers(0,20001,size=active,dtype=np.uint32);samples.append(values)
    for _ in range(12):samples.append(rng.integers(0,20001,size=lanes,dtype=np.uint32))
    values=np.full(lanes,1_000_000,dtype=np.uint32);values[-1]=0;samples.append(values)
    for trial,values in enumerate(samples):
        source=directory/f'min-{lanes}-{trial}.input'
        with source.open('wb') as stream:
            np.array([lanes],dtype='<u4').tofile(stream);values.astype('<u4').tofile(stream)
        result=subprocess.run([str(binary),'--wave-min',str(source)],capture_output=True,check=True,timeout=30)
        actual=np.frombuffer(result.stdout,dtype='<u4')
        np.testing.assert_array_equal(actual,np.full(lanes,int(values.min()),dtype=np.uint32))


@pytest.mark.parametrize('lanes',[0,1,31,33,63,65])
def test_wave_min_model_rejects_incomplete_or_unsupported_waves(scalar_cpu_probe,lanes):
    binary,directory=scalar_cpu_probe;source=directory/f'min-invalid-{lanes}.input'
    np.array([lanes]+[0]*lanes,dtype='<u4').tofile(source)
    result=subprocess.run([str(binary),'--wave-min',str(source)],capture_output=True,timeout=30)
    assert result.returncode==12


def check_packed_penalty_numpy(scalar_cpu_probe,disparities,paths,p2,model):
    from stereoforge.reference import aggregate_sgm
    binary, directory = scalar_cpu_probe
    rng = np.random.default_rng(0x5041434b + disparities + p2)
    costs = rng.integers(0, 10001, size=(7, 9, disparities), dtype=np.uint32)
    # Force a disparity to remain zero so unsupported neighbors reach P2 exactly.
    costs[..., 0] = 0
    sums = aggregate_sgm(costs, paths=paths, p1=min(8,p2), p2=p2)
    bits = 8 if paths*p2 <= 255 else 16
    components, mask = 32//bits, (1<<bits)-1
    words = (disparities+components-1)//components
    for lanes in (32,64):
        source = directory / f'{model.lstrip("-")}-{disparities}-{paths}-{p2}-{lanes}.input'
        target = source.with_suffix('.output')
        with source.open('wb') as stream:
            np.array([9,7,disparities,paths,min(8,p2),p2,lanes], dtype='<u4').tofile(stream)
            costs.astype('<u4').tofile(stream)
        subprocess.run([str(binary),model,str(source),str(target)],check=True,timeout=30)
        packed = np.fromfile(target,dtype='<u4').reshape(7,9,words)
        fields = np.stack([(packed>>np.uint32(i*bits))&np.uint32(mask) for i in range(components)],axis=-1).reshape(7,9,words*components)
        expected = sums-paths*costs
        assert np.all(expected <= paths*p2)
        np.testing.assert_array_equal(fields[...,:disparities],expected)
        assert not np.any(fields[...,disparities:]), 'unused fields must remain zero'
