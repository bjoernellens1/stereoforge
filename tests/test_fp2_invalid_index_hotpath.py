"""CPU-only guard of deliberate metric-index error fixture; never a live path."""
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_selection_fixture.cpp'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'


def section(source):
    return source[source.index('static void gpu_invalid_metric_index()'):source.index('static void gpu_witnesses()')]


def block(source):
    part=section(source);start=part.index('auto recorder=d.create_command_recorder(')
    return part[start:part.index('d.wait_idle();',start)+len('d.wait_idle();')]


def guard(source,owner):
    part=section(source);cmd=block(source)
    for item in ('raw_data[0].winner=128','raw_data[0].valid=1','obs_data[0].lr_valid=1','support{1,1}',
                 'uint32_t zero=0','status=alloc(sizeof(zero),&zero)',
                 'make_metric_lut(63.125f,.25f,reinterpret_cast<uintptr_t>(d.get()))',
                 'validate_metric_lut(lut,63.125f,.25f,reinterpret_cast<uintptr_t>(d.get()))',
                 'lut_buffer=alloc(sizeof(lut.entries),lut.entries.data())',
                 'd.device_address(status).value(),d.device_address(lut_buffer).value()', 'validate_regions(regions)',
                 '.byte_code_size=uint32_t(words.size())'):
        if item not in part:raise ValueError('fixture invariant '+item)
    for item in ('.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE',
                 'recorder.set_pipeline(metric)','SelectPush{d.device_address(param).value()}','recorder.dispatch({1,1,1})',
                 '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ','d.submit_commands','d.wait_idle()'):
        if item not in cmd:raise ValueError('command invariant '+item)
    for item in ('buffer_host_address','memcpy','memset','create_buffer','create_memory','wait_on_submit','collect_garbage'):
        if item in cmd:raise ValueError('unexpected command operation '+item)
    if part.count('d.wait_idle()')!=1:raise ValueError('extra wait')
    t=part[part.index('d.wait_idle();'):]
    if not(t.index('d.buffer_host_address(status).value()')<t.index('owner.release();context.close()')<t.index('require(failure==2')<t.index('throw std::runtime_error("metric LUT index outside contract")')):raise ValueError('status/teardown')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in owner:raise ValueError('owner cleanup')


def test_source():guard(APP.read_text(),OWNER.read_text())


@pytest.mark.parametrize('old,new',[
 ('raw_data[0].winner=128','raw_data[0].winner=127'),('obs_data[0].lr_valid=1','obs_data[0].lr_valid=0'),
 ('status=alloc(sizeof(zero),&zero)','status=alloc(sizeof(zero),nullptr)'),('validate_metric_lut(lut,63.125f,.25f,reinterpret_cast<uintptr_t>(d.get()))','(void)lut'),
 ('recorder.set_pipeline(metric)','d.wait_idle();recorder.set_pipeline(metric)'),
 ('recorder.set_pipeline(metric)','d.buffer_host_address(status);recorder.set_pipeline(metric)'),
 ('.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ','.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::HOST_READ'),
 ('.byte_code_size=uint32_t(words.size())','.byte_code_size=uint32_t(words.size()*4)'),('require(failure==2','require(true'),
 ('owner.release();context.close()','context.close();owner.release()'),
])
def test_mutations(old,new):
    source=APP.read_text();a=source.index('static void gpu_invalid_metric_index()');b=source.index('static void gpu_witnesses()',a);assert old in source[a:b]
    with pytest.raises((ValueError,IndexError)):guard(source[:a]+source[a:b].replace(old,new,1)+source[b:],OWNER.read_text())


def test_fake_actual_source(tmp_path):
    real=block(APP.read_text())
    cpp=r'''#include <cstdint>
#include <chrono>
#include <cstdio>
namespace daxa{struct AccessConsts{static constexpr int HOST_WRITE=1,COMPUTE_SHADER_READ=2,COMPUTE_SHADER_WRITE=4,HOST_READ=8;};struct Barrier{int src_access,dst_access;};struct Info{const char*name;};struct Dispatch{uint32_t x,y,z;};struct Addr{uint64_t value()const{return 7;}};struct Cmd{};struct Submit{struct List{Cmd*p;int n;}command_lists;};struct Recorder{static inline uint64_t barriers=0,dispatches=0;void pipeline_barrier(Barrier){++barriers;}void set_pipeline(int){}template<class T>void push_constant(T){}void dispatch(Dispatch){++dispatches;}Cmd complete_current_commands(){return {};}};struct Device{static inline uint64_t submits=0,waits=0;Recorder create_command_recorder(Info){return {};}Addr device_address(int){return {};}void submit_commands(Submit){++submits;}void wait_idle(){++waits;}};}
struct SelectPush{uint64_t p;};
__attribute__((noinline))static void run(daxa::Device& d){asm volatile("" ::: "memory");int metric=0,param=0;
''' + real + r'''
}
int main(){daxa::Device d;constexpr unsigned n=100000;auto t=std::chrono::steady_clock::now();for(unsigned i=0;i<n;++i)run(d);auto e=std::chrono::steady_clock::now();if(daxa::Recorder::barriers!=n*2||daxa::Recorder::dispatches!=n||daxa::Device::submits!=n||daxa::Device::waits!=n)return 1;std::printf("invalid-index fake %.2f ns; 1 dispatch 2 barriers 1 submit 1 diagnostic fake wait\n",std::chrono::duration<double,std::nano>(e-t).count()/n);}
'''
    path=tmp_path/'fake.cpp';exe=tmp_path/'fake';path.write_text(cpp)
    subprocess.run(['c++','-std=c++20','-O2',str(path),'-o',str(exe)],check=True,capture_output=True,text=True)
    r=subprocess.run([str(exe)],check=True,capture_output=True,text=True);print(r.stdout.strip())
    assert '1 dispatch 2 barriers 1 submit 1 diagnostic fake wait' in r.stdout
