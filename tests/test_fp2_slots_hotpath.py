"""CPU-only guards for FP2's synthetic two-slot fixture, not a live pipeline."""
from pathlib import Path
import re
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_slots_fixture.cpp'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'


def command_block(src):
    start=src.index('        for(size_t k=0;k<2;++k){auto recorder=')
    end=src.index('        require(submitted[0]<submitted[1]',start)
    return src[start:end]


def guard(src, owner):
    block=command_block(src)
    checks=['slots[k].parameter', 'pass<2?f[k].h:f[k].w',
            'commands[k]=recorder.complete_current_commands()',
            'd.submit_commands({.command_lists={&commands[k],1}})',
            'submitted[k]=d.latest_queue_submit_index(daxa::QUEUE_MAIN)',
            '.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE',
            '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE',
            '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ']
    for token in checks:
        if token not in block: raise ValueError('missing command invariant '+token)
    for token in ('wait_idle(', 'wait_on_submit(', 'memcpy(', 'memset(', 'buffer_host_address(', 'create_memory(', 'create_buffer(', 'destroy_buffer(', 'collect_garbage('):
        if token in block: raise ValueError('unexpected between submissions '+token)
    if src.index('validate_regions(regions);')>src.index('for(size_t k=0;k<2;++k){auto recorder='):
        raise ValueError('regions must validate before submissions')
    if 'slots[k]={sums,paths,parameter,params};' not in src:
        raise ValueError('slot parameter ownership')
    start=src.index('require(submitted[0]<submitted[1]')
    tail=src[start:]
    expected=['d.wait_on_submit(', 'std::memcmp(', 'owner.release();context.close()']
    if any(x not in tail for x in expected) or not tail.index(expected[0])<tail.index(expected[1])<tail.index(expected[2]):
        raise ValueError('completion/readback/lifetime order')
    if not re.search(r'"slot path mismatch"\);}\s*}\s*owner.release\(\);context.close\(\)', tail):
        raise ValueError('command and pipeline lifetime')
    if '.queue_submit_index=submitted[k]' not in tail or 'slots[k].sums' not in tail or 'slots[k].paths' not in tail:
        raise ValueError('per-slot completion/outputs')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in owner:
        raise ValueError('owner cleanup order')
    if '.byte_code_size=uint32_t(words.size())' not in src:
        raise ValueError('SPIR-V word count')


def test_current_sources():
    guard(APP.read_text(),OWNER.read_text())


@pytest.mark.parametrize('old,new',[
    ('slots[k].parameter','slots[0].parameter'),
    ('&commands[k],1','&commands[0],1'),
    ('submitted[k]=d.latest_queue_submit_index','submitted[0]=d.latest_queue_submit_index'),
    ('.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access', '.src_access=daxa::AccessConsts::HOST_READ,.dst_access'),
    ('.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE', '.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ'),
    ('d.submit_commands({.command_lists={&commands[k],1}});','d.wait_idle();d.submit_commands({.command_lists={&commands[k],1}});'),
    ('d.submit_commands({.command_lists={&commands[k],1}});','d.create_buffer();d.submit_commands({.command_lists={&commands[k],1}});'),
    ('.queue_submit_index=submitted[k]','.queue_submit_index=submitted[0]'),
    ('owner.release();context.close()','context.close();owner.release()'),
    ('validate_regions(regions);',''),
    ('"slot path mismatch");}\n    }','"slot path mismatch");}'),
    ('.byte_code_size=uint32_t(words.size())','.byte_code_size=uint32_t(words.size()*4)'),
])
def test_mutation_rejected(old,new):
    source=APP.read_text(); assert old in source
    with pytest.raises(ValueError): guard(source.replace(old,new,1),OWNER.read_text())


def test_fake_device_executes_actual_recording_and_submit_block(tmp_path):
    block=command_block(APP.read_text())
    cpp=r'''#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
namespace daxa {
struct AccessConsts {static constexpr int HOST_WRITE=1,COMPUTE_SHADER_READ=2,COMPUTE_SHADER_WRITE=4,HOST_READ=8;};
struct Barrier{int src_access,dst_access;};struct Dispatch{uint32_t x,y,z;};struct Info{const char* name;};struct Address{uint64_t a;uint64_t value()const{return a;}};
struct ExecutableCommandList{};struct Submission{struct List{ExecutableCommandList* p;int n;}command_lists;};
struct Recorder {static inline uint64_t barriers=0,dispatches=0,pushes=0;static inline uint64_t ids[8]{};void pipeline_barrier(Barrier){++barriers;}void set_pipeline(int){}template<class T>void push_constant(T v){ids[pushes%8]=v.address;++pushes;}void dispatch(Dispatch){++dispatches;}ExecutableCommandList complete_current_commands(){return {};}};
constexpr int QUEUE_MAIN=0;struct Device{uint64_t submits=0;Recorder create_command_recorder(Info){return {};}Address device_address(int a){return {uint64_t(a)};}void submit_commands(Submission){++submits;}uint64_t latest_queue_submit_index(int){return submits;}};
}
struct Push{uint64_t address;uint32_t pass;};struct Slot{int parameter;};struct F{uint32_t w=320,h=180;};
__attribute__((noinline)) static uint64_t run(daxa::Device& d){asm volatile("" ::: "memory");std::array<F,2> f{};std::array<Slot,2> slots{{{11},{22}}};int pipeline=0;std::array<daxa::ExecutableCommandList,2> commands{};
''' + block + r'''
return submitted[0]+submitted[1];}
int main(){daxa::Device d;constexpr unsigned n=100000;auto t=std::chrono::steady_clock::now();uint64_t result=0;for(unsigned i=0;i<n;++i)result+=run(d);auto e=std::chrono::steady_clock::now();
if(d.submits!=2*n || daxa::Recorder::barriers!=10*n || daxa::Recorder::dispatches!=8*n || daxa::Recorder::pushes!=8*n || daxa::Recorder::ids[0]!=11 || daxa::Recorder::ids[4]!=22 || result!=uint64_t(n)*(2*n+1))return 1;
std::printf("actual-source fake record+submit %.2f ns per pair; eight dispatches, ten barriers, two submissions, no fake waits\n",std::chrono::duration<double,std::nano>(e-t).count()/n);}
'''
    path=tmp_path/'fake.cpp';path.write_text(cpp); exe=tmp_path/'fake'
    subprocess.run(['c++','-std=c++20','-O2',str(path),'-o',str(exe)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    assert 'two submissions, no fake waits' in result.stdout
    print(result.stdout.strip())
