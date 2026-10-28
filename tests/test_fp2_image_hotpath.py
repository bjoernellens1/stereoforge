"""CPU-only checks of the experimental image-cost fixture, not live performance."""
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_image_fixture.cpp'
SHADER=ROOT/'python/stereoforge/shaders/daxa/fp2_image_cost.slang'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'


def block(source):
    start=source.index('     auto recorder=d.create_command_recorder(')
    return source[start:source.index('d.wait_idle();',start)+len('d.wait_idle();')]


def guard_shader(source):
    compact=''.join(source.split())
    required=['uinti=tid.x;if(i>=p.width*p.height)return;', 'for(uintd=lo[i];d<hi[i];++d)', 'p.eye==0?int(x)-int(d):int(x)+int(d)',
              'support[i]!=0', 'ox>=0&&ox<int(p.width)', 'other_support[j]!=0',
              'countbits(own[i]^other[j])', 'costs[offsets[i]+d-lo[i]]=value']
    for item in required:
        if item not in compact: raise ValueError('shader invariant '+item)


def guard_host(source, owner):
    commands=block(source)
    expected=['.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE',
              'recorder.set_pipeline(producer)', 'CostPush{d.device_address(cparam).value()}',
              'recorder.dispatch({(f.w*f.h+63)/64,1,1})',
              '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE',
              'for(uint32_t pass=0;pass<4;++pass)', 'if(pass)recorder.pipeline_barrier',
              'recorder.set_pipeline(recurrence)', 'Push{d.device_address(parameter).value(),pass}',
              '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ', 'd.submit_commands', 'd.wait_idle()']
    for item in expected:
        if item not in commands: raise ValueError('dispatch invariant '+item)
    a=commands.index('recorder.dispatch({(f.w*f.h+63)/64,1,1})')
    b=commands.index('for(uint32_t pass=0;pass<4;++pass)')
    if '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE' not in commands[a:b]:
        raise ValueError('cost to recurrence dependency')
    recurrence=commands[commands.index('if(pass)recorder.pipeline_barrier'):commands.index('recorder.set_pipeline(recurrence)')]
    if '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE' not in recurrence:
        raise ValueError('interpass dependency')
    for item in ('buffer_host_address','memcpy','memset','create_buffer','create_memory','wait_on_submit','collect_garbage'):
        if item in commands: raise ValueError('unexpected operation '+item)
    if source.count('d.wait_idle()')!=1: raise ValueError('extra wait')
    for item in ('std::memset(d.buffer_host_address(cost).value(),0xff', 'validate_regions(regions)',
                 'CostParams cp{p.lo,p.hi,p.offsets,p.costs,', '.byte_code_size=uint32_t(cw.size())',
                 '.byte_code_size=uint32_t(rw.size())'):
        if item not in source: raise ValueError('resource invariant '+item)
    tail=source[source.index('d.wait_idle();'):]
    if not tail.index('"path mismatch");}') < tail.index('owner.release();context.close()'):
        raise ValueError('pipeline lifetime')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in owner:
        raise ValueError('owner cleanup order')


def test_current_sources():
    guard_shader(SHADER.read_text());guard_host(APP.read_text(),OWNER.read_text())


@pytest.mark.parametrize('old,new',[
    ('p.eye==0?int(x)-int(d):int(x)+int(d)', 'p.eye==0?int(x)+int(d):int(x)-int(d)'),
    ('support[i]!=0', 'support[i]==0'),('other_support[j]!=0','other_support[j]==0'),
    ('costs[offsets[i]+d-lo[i]]','costs[offsets[i]+d]'),
    ('for(uint d=lo[i];d<hi[i];++d)','for(uint d=0;d<128;++d)'),
])
def test_shader_mutations(old,new):
    s=SHADER.read_text();assert old in s
    with pytest.raises(ValueError): guard_shader(s.replace(old,new,1))


@pytest.mark.parametrize('old,new',[
    ('std::memset(d.buffer_host_address(cost).value(),0xff','std::memset(d.buffer_host_address(cost).value(),0'),
    ('validate_regions(regions)',''),('CostParams cp{p.lo,p.hi,p.offsets,p.costs,','CostParams cp{p.lo,p.hi,p.offsets,p.sums,'),
    ('recorder.set_pipeline(producer)','d.wait_idle();recorder.set_pipeline(producer)'),
    ('recorder.set_pipeline(recurrence)','d.buffer_host_address(cost);recorder.set_pipeline(recurrence)'),
    ('.byte_code_size=uint32_t(cw.size())','.byte_code_size=uint32_t(cw.size()*4)'),
    ('.byte_code_size=uint32_t(rw.size())','.byte_code_size=uint32_t(rw.size()*4)'),
    ('owner.release();context.close()','context.close();owner.release()'),
])
def test_host_mutations(old,new):
    s=APP.read_text();assert old in s
    with pytest.raises((ValueError,IndexError)):guard_host(s.replace(old,new,1),OWNER.read_text())


def test_producer_recurrence_barrier_mutation():
    source=APP.read_text(); start=source.index('recorder.dispatch({(f.w*f.h+63)/64,1,1})');end=source.index('for(uint32_t pass=0;pass<4;++pass)',start)
    old='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE';assert old in source[start:end]
    changed=source[:start]+source[start:end].replace(old,'.src_access=daxa::AccessConsts::HOST_WRITE',1)+source[end:]
    with pytest.raises(ValueError):guard_host(changed,OWNER.read_text())


def test_fake_device_actual_source(tmp_path):
    real=block(APP.read_text())
    cpp=r'''#include <cstdint>
#include <chrono>
#include <cstdio>
namespace daxa {
struct AccessConsts{static constexpr int HOST_WRITE=1,COMPUTE_SHADER_READ=2,COMPUTE_SHADER_WRITE=4,HOST_READ=8;};struct Barrier{int src_access,dst_access;};struct Info{const char*name;};struct Dispatch{uint32_t x,y,z;};struct Addr{uint64_t value()const{return 7;}};struct Cmd{};struct Submit{struct List{Cmd*p;int n;}command_lists;};
struct Recorder{static inline uint64_t barriers=0,dispatches=0;void pipeline_barrier(Barrier){++barriers;}void set_pipeline(int){}template<class T>void push_constant(T){}void dispatch(Dispatch){++dispatches;}Cmd complete_current_commands(){return {};}};
struct Device{static inline uint64_t submits=0,waits=0;Recorder create_command_recorder(Info){return {};}Addr device_address(int){return {};}void submit_commands(Submit){++submits;}void wait_idle(){++waits;}};
}
struct Push{uint64_t p;uint32_t pass;};struct CostPush{uint64_t p;};struct F{uint32_t w=320,h=180;};
__attribute__((noinline))static void run(daxa::Device& d,F& f,int producer,int recurrence,int cparam,int parameter){asm volatile("" ::: "memory");
''' + real + r'''
}
int main(){daxa::Device d;F f;constexpr unsigned n=100000;auto t=std::chrono::steady_clock::now();for(unsigned i=0;i<n;++i)run(d,f,0,0,0,0);auto e=std::chrono::steady_clock::now();if(daxa::Recorder::barriers!=n*6||daxa::Recorder::dispatches!=n*5||daxa::Device::submits!=n||daxa::Device::waits!=n)return 1;std::printf("fake actual-source %.2f ns; 5 dispatches 6 barriers 1 submit 1 diagnostic fake wait\n",std::chrono::duration<double,std::nano>(e-t).count()/n);}
'''
    path=tmp_path/'fake.cpp';path.write_text(cpp);exe=tmp_path/'fake'
    subprocess.run(['c++','-std=c++20','-O2',str(path),'-o',str(exe)],check=True,capture_output=True,text=True)
    r=subprocess.run([str(exe)],check=True,capture_output=True,text=True);print(r.stdout.strip())
    assert '5 dispatches 6 barriers 1 submit 1 diagnostic fake wait' in r.stdout


def test_interpass_barrier_mutation():
    source=APP.read_text();start=source.index('if(pass)recorder.pipeline_barrier');end=source.index('recorder.set_pipeline(recurrence)',start)
    old='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE';assert old in source[start:end]
    changed=source[:start]+source[start:end].replace(old,'.src_access=daxa::AccessConsts::HOST_WRITE',1)+source[end:]
    with pytest.raises(ValueError):guard_host(changed,OWNER.read_text())


def test_dispatch_tail_guard_mutation():
    source=SHADER.read_text()
    old='if(i>=p.width*p.height)return;'
    assert old in source
    with pytest.raises(ValueError): guard_shader(source.replace(old,'',1))
