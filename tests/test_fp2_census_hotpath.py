"""CPU-only guards for test-only Census/cost/recurrence command chain."""
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_census_fixture.cpp'
SHADER=ROOT/'python/stereoforge/shaders/daxa/fp2_census.slang'
HELPER=ROOT/'python/stereoforge/shaders/ops/refinement.slang'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'


def block(s):
    start=s.index('     auto recorder=d.create_command_recorder(')
    return s[start:s.index('d.wait_idle();',start)+len('d.wait_idle();')]


def guard_shader(shader,helper):
    s=''.join(shader.split());h=''.join(helper.split())
    for token in ('#include"../ops/refinement.slang"','uinti=tid.x;if(i>=p.width*p.height)return;',
                  'refinement_support(support,p.width,p.height,x,y)',
                  'descriptor[i]=ok?refinement_census(gray,p.width,x,y):0;valid[i]=uint(ok);'):
        if token not in s: raise ValueError('census shader invariant '+token)
    for token in ('if(x<2||y<2||x+2>=width||y+2>=height)returnfalse;',
                  'for(intdy=-2;dy<=2;++dy)', 'for(intdx=-2;dx<=2;++dx)',
                  'if(dx==0&&dy==0)continue;', '<centre)<<(bit++);',
                  'support[uint(int(y)+dy)*width+uint(int(x)+dx)]==0'):
        if token not in h: raise ValueError('shared helper invariant '+token)


def guard_host(src,owner):
    cmd=block(src)
    needed=['CensusPush{d.device_address(census0).value()}', 'CensusPush{d.device_address(census1).value()}',
            'recorder.set_pipeline(census)', 'recorder.set_pipeline(producer)', 'recorder.set_pipeline(recurrence)',
            'CostPush{d.device_address(cparam).value()}', 'Push{d.device_address(parameter).value(),pass}',
            'for(uint32_t pass=0;pass<4;++pass)', 'd.submit_commands', 'd.wait_idle()']
    for item in needed:
        if item not in cmd: raise ValueError('command invariant '+item)
    if cmd.count('recorder.dispatch({(f.w*f.h+63)/64,1,1})')!=3: raise ValueError('two census and one cost dispatch required')
    if cmd.count('d.wait_idle()')!=1 or src.count('d.wait_idle()')!=1: raise ValueError('unexpected wait')
    for token in ('buffer_host_address','memcpy','memset','create_memory','create_buffer','wait_on_submit','collect_garbage'):
        if token in cmd: raise ValueError('forbidden command '+token)
    host='.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    gpu='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    finish='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ'
    positions=[cmd.index('recorder.set_pipeline(census)'),cmd.index('recorder.set_pipeline(producer)'),cmd.index('for(uint32_t pass=0;pass<4;++pass)'),cmd.index('recorder.set_pipeline(recurrence)')]
    last_census = cmd.index('recorder.dispatch({(f.w*f.h+63)/64,1,1})', cmd.index('CensusPush{d.device_address(census1).value()}'))
    if gpu not in cmd[last_census:positions[1]]: raise ValueError('barrier must follow second census dispatch')
    if host not in cmd[:positions[0]] or gpu not in cmd[positions[0]:positions[1]] or gpu not in cmd[positions[1]:positions[2]] or gpu not in cmd[positions[2]:positions[3]] or finish not in cmd[positions[3]:]:
        raise ValueError('pipeline dependency')
    for token in ('auto left=make_plane(f,eye),right=make_plane(f,1-eye)',
                  'std::memset(d.buffer_host_address(cost).value(),0xff',
                  'CensusParams c0{d.device_address(luma0).value(),d.device_address(support0).value(),d.device_address(own).value(),d.device_address(own_support).value()',
                  'CensusParams c1{d.device_address(luma1).value(),d.device_address(support1).value(),d.device_address(other).value(),d.device_address(other_support).value()',
                  'CostParams cp{p.lo,p.hi,p.offsets,p.costs,d.device_address(own).value(),d.device_address(other).value(),d.device_address(own_support).value(),d.device_address(other_support).value()',
                  'validate_regions(regions)', '.byte_code_size=uint32_t(sw.size())','.byte_code_size=uint32_t(cw.size())','.byte_code_size=uint32_t(rw.size())'):
        if token not in src: raise ValueError('resource invariant '+token)
    tail=src[src.index('d.wait_idle();'):]
    if not tail.index('"path mismatch");}')<tail.index('owner.release();context.close()'):raise ValueError('teardown order')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in owner:raise ValueError('cleanup order')


def test_sources():guard_shader(SHADER.read_text(),HELPER.read_text());guard_host(APP.read_text(),OWNER.read_text())


@pytest.mark.parametrize('old,new',[
    ('if(i>=p.width*p.height)return;',''),('refinement_support(support,p.width,p.height,x,y)','true'),
    ('descriptor[i]=ok?refinement_census(gray,p.width,x,y):0','descriptor[i]=refinement_census(gray,p.width,x,y)'),
])
def test_shader_mutations(old,new):
    s=SHADER.read_text();assert old in s
    with pytest.raises(ValueError):guard_shader(s.replace(old,new,1),HELPER.read_text())


@pytest.mark.parametrize('old,new',[
    ('<centre)<<(bit++)','>centre)<<(bit++)'),('if(dx==0&&dy==0)continue;',''),
    ('support[uint(int(y)+dy)*width+uint(int(x)+dx)]==0','support[uint(int(y)+dy)*width+uint(int(x)+dx)]!=0'),
])
def test_helper_mutations(old,new):
    s=HELPER.read_text();assert old in s
    with pytest.raises(ValueError):guard_shader(SHADER.read_text(),s.replace(old,new,1))


@pytest.mark.parametrize('old,new',[
    ('CensusPush{d.device_address(census1).value()}','CensusPush{d.device_address(census0).value()}'),
    ('recorder.set_pipeline(producer)','d.wait_idle();recorder.set_pipeline(producer)'),
    ('recorder.set_pipeline(recurrence)','d.buffer_host_address(cost);recorder.set_pipeline(recurrence)'),
    ('CensusParams c1{d.device_address(luma1).value()','CensusParams c1{d.device_address(luma0).value()'),
    ('validate_regions(regions)',''),
    ('right=make_plane(f,1-eye)','right=make_plane(f,eye)'),
    ('std::memset(d.buffer_host_address(cost).value(),0xff','std::memset(d.buffer_host_address(cost).value(),0'),('.byte_code_size=uint32_t(sw.size())','.byte_code_size=uint32_t(sw.size()*4)'),
    ('owner.release();context.close()','context.close();owner.release()'),
])
def test_host_mutations(old,new):
    s=APP.read_text();assert old in s
    with pytest.raises((ValueError,IndexError)):guard_host(s.replace(old,new,1),OWNER.read_text())


@pytest.mark.parametrize('start,end',[
    ('recorder.set_pipeline(census)','recorder.set_pipeline(producer)'),
    ('recorder.set_pipeline(producer)','for(uint32_t pass=0;pass<4;++pass)'),
    ('if(pass)recorder.pipeline_barrier','recorder.set_pipeline(recurrence)'),
])
def test_scoped_barrier_mutations(start,end):
    s=APP.read_text();a=s.index(start);b=s.index(end,a);old='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE';assert old in s[a:b]
    m=s[:a]+s[a:b].replace(old,'.src_access=daxa::AccessConsts::HOST_WRITE',1)+s[b:]
    with pytest.raises(ValueError):guard_host(m,OWNER.read_text())


def test_fake_actual_source(tmp_path):
    real=block(APP.read_text())
    cpp=r'''#include <cstdint>
#include <chrono>
#include <cstdio>
namespace daxa {
struct AccessConsts{static constexpr int HOST_WRITE=1,COMPUTE_SHADER_READ=2,COMPUTE_SHADER_WRITE=4,HOST_READ=8;};struct Barrier{int src_access,dst_access;};struct Info{const char*name;};struct Dispatch{uint32_t x,y,z;};struct Addr{uint64_t value()const{return 7;}};struct Cmd{};struct Submit{struct List{Cmd*p;int n;}command_lists;};
struct Recorder{static inline uint64_t barriers=0,dispatches=0;void pipeline_barrier(Barrier){++barriers;}void set_pipeline(int){}template<class T>void push_constant(T){}void dispatch(Dispatch){++dispatches;}Cmd complete_current_commands(){return {};}};
struct Device{static inline uint64_t submits=0,waits=0;Recorder create_command_recorder(Info){return {};}Addr device_address(int){return {};}void submit_commands(Submit){++submits;}void wait_idle(){++waits;}};
}
struct Push{uint64_t p;uint32_t pass;};struct CostPush{uint64_t p;};struct CensusPush{uint64_t p;};struct F{uint32_t w=320,h=180;};
__attribute__((noinline))static void run(daxa::Device& d,F& f,int census,int producer,int recurrence,int census0,int census1,int cparam,int parameter){asm volatile("" ::: "memory");
''' + real + r'''
}
int main(){daxa::Device d;F f;constexpr unsigned n=100000;auto t=std::chrono::steady_clock::now();for(unsigned i=0;i<n;++i)run(d,f,0,0,0,0,0,0,0);auto e=std::chrono::steady_clock::now();if(daxa::Recorder::barriers!=n*7||daxa::Recorder::dispatches!=n*7||daxa::Device::submits!=n||daxa::Device::waits!=n)return 1;std::printf("fake actual-source %.2f ns; 7 dispatches 7 barriers 1 submit 1 diagnostic fake wait\n",std::chrono::duration<double,std::nano>(e-t).count()/n);}
'''
    path=tmp_path/'fake.cpp';path.write_text(cpp);exe=tmp_path/'fake'
    subprocess.run(['c++','-std=c++20','-O2',str(path),'-o',str(exe)],check=True,capture_output=True,text=True)
    r=subprocess.run([str(exe)],check=True,capture_output=True,text=True);print(r.stdout.strip())
    assert '7 dispatches 7 barriers 1 submit 1 diagnostic fake wait' in r.stdout


def test_barrier_moved_between_census_dispatches_rejected():
    source=APP.read_text()
    gpu='recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE});'
    second='recorder.push_constant(CensusPush{d.device_address(census1).value()});'
    start=source.index(second); pos=source.index(gpu,start)
    changed=source[:start]+gpu+source[start:pos]+source[pos+len(gpu):]
    with pytest.raises(ValueError):guard_host(changed,OWNER.read_text())
