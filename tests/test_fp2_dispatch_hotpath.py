"""CPU-only guards for the isolated FP2 fixture; not a production performance test."""
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'apps/fp2_prototype_fixture.cpp'
SHADER = ROOT / 'python/stereoforge/shaders/daxa/fp2_banded.slang'


def dispatch_block(source):
    start = source.index('    auto recorder=d.create_command_recorder(')
    end = source.index('d.wait_idle();', start) + len('d.wait_idle();')
    return source[start:end]


def guard_shader(source):
    src = re.sub(r'\s+', '', source)
    required = ['for(uintt=0;t<prev_width;++t)', 'for(uintt=0;t<width;++t)',
                'uintd=lo[i]+t;', 'd-prev_lo<prev_width',
                'd-1-prev_lo<prev_width', 'd+1-prev_lo<prev_width',
                'prev_lo=lo[i];prev_width=width;', 'scratch[current+t]=v;',
                'uint64_tj=offsets[i]+t;']
    for token in required:
        if token not in src: raise ValueError('shader invariant: '+token)
    for pattern in (r'for\([^;]*;[^;]*(?:p\.disparities|128)[^;]*;', r'while\([^)]*p\.disparities'):
        if re.search(pattern, src): raise ValueError('dense disparity iteration')


def guard_host(source):
    block = dispatch_block(source)
    tokens = ['HOST_WRITE', 'COMPUTE_SHADER_READ', 'COMPUTE_SHADER_WRITE',
              'for(uint32_t pass=0;pass<4;++pass)', 'if(pass)recorder.pipeline_barrier',
              'recorder.dispatch({pass<2?f.h:f.w,1,1})', 'HOST_READ',
              'submit_commands', 'd.wait_idle()']
    for token in tokens:
        if token not in block: raise ValueError('host invariant: '+token)
    for token in ('create_buffer', 'create_memory', 'memcpy', 'memset', 'buffer_host_address', 'collect_garbage'):
        if token in block: raise ValueError('unexpected per-dispatch operation: '+token)
    interpass = block[block.index('if(pass)recorder.pipeline_barrier'):block.index('recorder.set_pipeline')]
    if '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE' not in interpass: raise ValueError('interpass compute source required')
    if '.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE' not in interpass: raise ValueError('interpass compute read/write destination required')
    barrier_pair = '.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    if block.count(barrier_pair) != 2: raise ValueError('compute read/write destination required')
    tail = source[source.index('d.wait_idle();') :]
    if not (tail.index('std::memcmp') < tail.index('owner.release()') < tail.index('context.close()')):
        raise ValueError('diagnostic resources not retained through wait/readback')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in source:
        raise ValueError('resource owner cleanup order')
    if not re.search(r'paths.size\(\)\*4\)==0,\"GPU paths mismatch\"\);\s*}\s*owner.release\(\);context.close\(\)', tail):
        raise ValueError('pipeline/commands outlive context')
    if block.count('d.wait_idle()') != 1 or source.count('d.wait_idle()') != 1: raise ValueError('unexpected wait count')


def test_actual_source_static_guards():
    guard_shader(SHADER.read_text())
    guard_host(APP.read_text())


@pytest.mark.parametrize('old,new', [
    ('t<prev_width', 't<p.disparities'),
    ('uint d=lo[i]+t', 'uint d=t'),
    ('d-prev_lo<prev_width', 'd<prev_width'),
    ('scratch[current+t]=v;', ''),
])
def test_shader_guard_rejects_mutations(old,new):
    s=SHADER.read_text(); assert old in s
    with pytest.raises(ValueError): guard_shader(s.replace(old,new,1))


@pytest.mark.parametrize('old,new', [
    ('HOST_WRITE','COMPUTE_SHADER_WRITE'),
    ('.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE', '.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ'),
    ('GPU paths mismatch"); }', 'GPU paths mismatch");'),
    ('owner.release();context.close()', 'context.close();owner.release()'),
    ('if(pass)recorder.pipeline_barrier','if(false)recorder.pipeline_barrier'),
    ('d.wait_idle();','d.wait_idle();d.wait_idle();'),
    ('recorder.set_pipeline(pipeline);','d.create_buffer();recorder.set_pipeline(pipeline);'),
])
def test_host_guard_rejects_mutations(old,new):
    s=APP.read_text(); assert old in s
    with pytest.raises((ValueError,IndexError)): guard_host(s.replace(old,new,1))


def test_actual_source_fake_device_dispatch(tmp_path):
    """Compile the unedited recorded command block with counting fake Daxa methods."""
    block=dispatch_block(APP.read_text())
    cpp=r'''#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
namespace daxa {
struct AccessConsts { static constexpr int HOST_WRITE=1, COMPUTE_SHADER_READ=2, COMPUTE_SHADER_WRITE=3, HOST_READ=4; };
struct Barrier {int src_access,dst_access;}; struct Dispatch {uint32_t x,y,z;};
struct Address {uint64_t value() const{return 4;}}; struct Info {const char* name;};
struct Cmd {}; struct Submission { struct List { Cmd* data; unsigned size; } command_lists; };
struct Recorder {static inline uint64_t barriers=0,dispatches=0,sets=0,pushes=0; void pipeline_barrier(Barrier){++barriers;} void set_pipeline(int){++sets;} template<class T>void push_constant(T){++pushes;} void dispatch(Dispatch){++dispatches;} Cmd complete_current_commands(){return {};}};
struct Device {static inline uint64_t submits=0,waits=0; Recorder create_command_recorder(Info){return {};} Address device_address(int){return {};} void submit_commands(Submission){++submits;} void wait_idle(){++waits;}};
}
struct Push {uint64_t address;uint32_t pass;};struct F {uint32_t w=7,h=5;};
__attribute__((noinline)) static void run(daxa::Device& d,F& f,int pipeline,int parameter) {
asm volatile("" ::: "memory");
''' + block + r'''
}
int main(){daxa::Device d;F f;constexpr unsigned n=100000;auto begin=std::chrono::steady_clock::now();for(unsigned i=0;i<n;++i)run(d,f,0,0);auto end=std::chrono::steady_clock::now();
if(daxa::Recorder::barriers!=n*5 || daxa::Recorder::dispatches!=n*4 || daxa::Recorder::sets!=n*4 || daxa::Recorder::pushes!=n*4 || daxa::Device::submits!=n || daxa::Device::waits!=n)return 1;
std::printf("fake actual-source recording %.2f ns/fixture; 4 passes, 5 barriers, 1 submit, 1 fake diagnostic wait\n",std::chrono::duration<double,std::nano>(end-begin).count()/n);}
'''
    source=tmp_path/'fake.cpp'; exe=tmp_path/'fake'; source.write_text(cpp)
    subprocess.run(['c++','-std=c++20','-O2',str(source),'-o',str(exe)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    assert '4 passes, 5 barriers, 1 submit, 1 fake diagnostic wait' in result.stdout
    print(result.stdout.strip())


def test_interpass_write_dependency_mutation_rejected():
    src=APP.read_text()
    marker='.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    assert src.count(marker) == 2
    position=src.rfind(marker)
    with pytest.raises(ValueError):
        guard_host(src[:position] + src[position:].replace(marker, '.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ', 1))


def test_interpass_source_dependency_mutation_rejected():
    src=APP.read_text()
    start=src.index('if(pass)recorder.pipeline_barrier')
    end=src.index('recorder.set_pipeline', start)
    old='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    assert old in src[start:end]
    changed=src[:start]+src[start:end].replace(old,'.src_access=daxa::AccessConsts::HOST_WRITE',1)+src[end:]
    with pytest.raises(ValueError): guard_host(changed)


def guard_spirv_word_count(source):
    assert re.search(r'\.byte_code\s*=\s*words\.data\(\)\s*,\s*\.byte_code_size\s*=\s*uint32_t\(words\.size\(\)\)', source), 'SPIR-V byte_code_size must be word count'


def test_pipeline_size_is_spirv_word_count():
    guard_spirv_word_count(APP.read_text())


def test_pipeline_size_guard_rejects_byte_count():
    source=APP.read_text()
    correct='.byte_code_size=uint32_t(words.size())'
    assert correct in source
    with pytest.raises(AssertionError):
        guard_spirv_word_count(source.replace(correct, '.byte_code_size=uint32_t(sizeof(uint32_t)*words.size())', 1))
