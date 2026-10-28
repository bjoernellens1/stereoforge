"""Execute the producer's actual cold allocation owner with injected device failures."""
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest
from test_daxa_depth_hotpath import definition
from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]


def probe_program(source):
    resources = definition(source, 'struct Resources {') + ';\n'
    footprint = definition(source, 'uint64_t daxa_rectified_input_required_bytes(')
    return r'''
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
void require(bool v,const char* m){if(!v)throw std::invalid_argument(m);}
struct State {int call=0,fail=0,memory=0,next=1,destroyed=0,waits=0;std::set<int> live;void step(){if(++call==fail)throw std::runtime_error("injected allocation failure");}};
namespace daxa {
struct BufferId {int id=0;};
struct Token {std::shared_ptr<State> state;explicit Token(std::shared_ptr<State> s):state(std::move(s)){++state->memory;}~Token(){--state->memory;}};
struct MemoryBlock {std::shared_ptr<Token> token;};
struct BufferInfo {size_t size;std::string name;};
enum class MemoryFlagBits {HOST_ACCESS_RANDOM};
struct MemoryInfo {int requirements;MemoryFlagBits flags;};
struct BufferFromMemory {BufferInfo buffer_info;MemoryBlock memory_block;};
struct Device {
    std::shared_ptr<State> state;
    MemoryBlock create_memory(MemoryInfo){state->step();return {std::make_shared<Token>(state)};}
    BufferId create_buffer(BufferInfo){state->step();int id=state->next++;state->live.insert(id);return {id};}
    BufferId create_buffer_from_memory_block(BufferFromMemory f){return create_buffer(f.buffer_info);}
    bool is_id_valid(BufferId b){return state->live.contains(b.id);}
    void destroy_buffer(BufferId b){require(state->live.erase(b.id)==1,"double destruction");++state->destroyed;}
    void collect_garbage(){}
    void wait_idle(){++state->waits;}
};
}
struct DaxaFoundation {daxa::Device device;int coherent_readback_requirements(daxa::BufferInfo const&){return 0;}};
struct Plane {uint32_t width=4,height=4,stride_bytes=13;};
struct Source {std::array<Plane,4> rgb;std::array<Plane,2> support;};
struct Capture {std::shared_ptr<Source> canonical_rgb=std::make_shared<Source>();};
struct RectifiedRgbFrame {std::shared_ptr<Capture> source=std::make_shared<Capture>();};
struct DaxaRectification {uint32_t eye_width=4,source_height=4,width=4,height=4;};
struct DaxaRectifiedInputConfig {unsigned slots=2;};
struct Parameters {char bytes[200];};
void validate_daxa_rectified_input_geometry(const DaxaRectification&,const std::optional<DaxaRectification>&,const DaxaRectifiedInputConfig&,const RectifiedRgbFrame&){}
''' + resources + footprint + r'''
struct Harness {
    Resources resources;
    Harness(std::shared_ptr<DaxaFoundation> context,bool later_failure=false){
        resources.prepare(context,1000,10,5);
        for(unsigned i=0;i<10;++i)resources.allocate(10,"actual resource owner",i%2,1000);
        if(later_failure)throw std::runtime_error("injected later pipeline/graph setup failure");
    }
};
template<class F>void rejects(F f){bool rejected=false;try{f();}catch(std::exception const&){rejected=true;}require(rejected,"expected failure");}
int main(){
    auto state=std::make_shared<State>();auto context=std::make_shared<DaxaFoundation>();context->device.state=state;
    for(int nth=1;nth<=15;++nth){state->call=0;state->fail=nth;rejects([&]{Harness failed(context);});require(state->live.empty()&&state->memory==0,"nth allocation failure leaked device resources");}
    state->call=0;state->fail=0;rejects([&]{Harness failed(context,true);});require(state->live.empty()&&state->memory==0,"later setup exception leaked resources");
    {Harness good(context);require(state->live.size()==10&&state->memory==5,"successful resource allocation count");good.resources.wait();good.resources.release();good.resources.release();require(state->live.empty()&&state->memory==0,"idempotent explicit release");}
    int calls=state->call;
    {Resources owner;rejects([&]{owner.prepare(context,2ull*1024*1024*1024,10,5);});}
    {Resources owner;owner.prepare(context,10,1,1);rejects([&]{owner.allocate(11,"over budget",false,100);});rejects([&]{owner.allocate(5,"storage limit",false,4);});}
    {Resources owner;owner.prepare(context,10,0,0);rejects([&]{owner.allocate(1,"tracking capacity",false,100);});}
    {Resources owner;owner.prepare(context,10,1,0);rejects([&]{owner.allocate(1,"memory tracking capacity",true,100);});}
    require(state->call==calls&&state->live.empty()&&state->memory==0,"admission rejection must precede any device allocation");
    RectifiedRgbFrame source;for(auto& p:source.source->canonical_rgb->support)p.stride_bytes=7;
    DaxaRectification full,coarse;coarse.width=coarse.height=2;
    auto planned=daxa_rectified_input_required_bytes(full,coarse,{},source);
    // Four RGB extents=51->52 and two support extents=25->28.
    require(planned==2*(2*(4*52+2*28)+200+8*16+24*16+24*4),"exact footprint includes aligned upload, originals once and six coarse outputs");
    source.source->canonical_rgb->rgb[0].stride_bytes=UINT32_MAX;
    rejects([&]{daxa_rectified_input_required_bytes(full,coarse,{},source);});
    require(state->call==calls,"overbudget footprint must not touch device");
    require(state->waits>=17,"cold failure and teardown waits executed");
    std::cout<<"actual_resources_nth_failures=15 later_setup_failures=1 live_buffers=0 live_memory=0 preflight_reject_before_device=true\n";
}
'''


def compile_run(program):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.fail('C++ compiler required for actual allocation lifecycle probe')
    with tempfile.TemporaryDirectory(prefix='sf-rectified-lifecycle-') as directory:
        source = Path(directory) / 'probe.cpp'
        binary = Path(directory) / 'probe'
        source.write_text(program)
        built = subprocess.run([compiler, '-std=c++20', '-O2', str(source), '-o', str(binary)], capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        return subprocess.run([str(binary)], capture_output=True, text=True)


def test_actual_rectified_allocation_lifecycle():
    source = (ROOT / 'src/daxa_rectified_input.cpp').read_text()
    result = compile_run(probe_program(source))
    assert result.returncode == 0, result.stderr
    assert 'live_buffers=0 live_memory=0' in result.stdout


def test_preflight_and_teardown_order_guards():
    source = (ROOT / 'src/daxa_rectified_input.cpp').read_text()
    setup = body(source, 'Impl(DaxaRectification f,')
    assert setup.index('daxa_rectified_input_required_bytes(') < setup.index('resources.prepare(') < setup.index('create_compute_pipeline(') < setup.index('allocate(upload_bytes,')
    assert 'cfg.slots*(c?17u:11u)' in setup and 'cfg.slots*2u' in setup
    teardown = body(source, '~Impl()noexcept')
    assert teardown.index('resources.wait()') < teardown.index('slots.clear()') < teardown.index('pipelines={}') < teardown.index('resources.release()')
    assert source.index('} resources;') < source.index('struct Slot {')


@pytest.mark.parametrize('mutation', ['skip_release', 'skip_preallocation_budget', 'skip_reserved_tracking'])
def test_lifecycle_probe_rejects_mutations(mutation):
    source = (ROOT / 'src/daxa_rectified_input.cpp').read_text()
    if mutation == 'skip_release':
        source = source.replace('~Resources()noexcept{wait();release();}', '~Resources()noexcept{wait();}')
    elif mutation == 'skip_preallocation_budget':
        source = source.replace('require(bytes<=planned_bytes&&size<=planned_bytes-bytes,"canonical allocation exceeds preflight budget");', '')
    else:
        source = source.replace('require(owned.size()<owned.capacity()&&(!host||memory.size()<memory.capacity()),"canonical allocation tracking not reserved");', '')
    result = compile_run(probe_program(source))
    assert result.returncode != 0, mutation + ' escaped actual allocation lifecycle probe'
