"""CPU-only profiling command guards; no GPU measurements."""
from pathlib import Path
import re
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_fused_stage.cpp'
BASE=ROOT/'apps/fp2_fused_fixture.cpp'
DIAG=ROOT/'benchmarks/fp2_fused_stage.py'


def block(s):
    a=s.index('auto recorder=d.create_command_recorder({.name="resident FP2"})')
    return s[a:s.index('return recorder.complete_current_commands();',a)+len('return recorder.complete_current_commands();')]


def guard(src,base,diag):
    cmd=block(src); ref=block(base)
    if re.findall(r'recorder\.set_pipeline\((\w+)\)',cmd)!=re.findall(r'recorder\.set_pipeline\((\w+)\)',ref):raise ValueError('pipeline order differs')
    if re.findall(r'recorder\.dispatch\((\{[^;]+?\})\)',cmd)!=re.findall(r'recorder\.dispatch\((\{[^;]+?\})\)',ref):raise ValueError('dispatch dimensions differ')
    mark=cmd[cmd.index('auto mark=[&]()'):cmd.index('recorder.write_timestamp({.query_pool=queries,.pipeline_stage=daxa::PipelineStageFlagBits::TOP_OF_PIPE')]
    for item in ('auto mark=[&](){recorder.pipeline_barrier(', 'if(instrumented)recorder.write_timestamp(', '.query_index=query++'):
        if item not in mark:raise ValueError('control/instrumented mark mismatch')
    for token in ('reset_timestamps({.query_pool=queries,.start_index=0,.count=17})','require(query==(instrumented?17u:2u)', 'for(uint32_t pass=0;pass<4;++pass)',
                  'ResetCostPush{d.device_address(sums).value(),uint32_t(max_words)}'):
        if token not in cmd:raise ValueError('recording invariant '+token)
    for token in ('buffer_host_address','memcpy','memset','create_buffer','create_memory','wait_on_submit','wait_idle','std::ofstream'):
        if token in cmd:raise ValueError('recording host operation')
    start=src.index('for(uint32_t frame=0;frame<iterations;++frame)');end=src.index('uint32_t failure=0;',start);loop=src[start:end]
    order=['auto commands=record(instrumented)','d.submit_commands','auto submit_id=d.latest_queue_submit_index','d.wait_on_submit','daxa_timeline_query_pool_query_results']
    positions=[loop.index(x) for x in order]
    if positions!=sorted(positions) or '.queue_submit_index=submit_id' not in loop:raise ValueError('query reuse before completion')
    for token in ('buffer_host_address','memcpy','memset','create_buffer','create_memory','std::ofstream'):
        if token in loop:raise ValueError('frame roundtrip')
    for token in ('uint32_t phase=frame<10?frame:frame-10', 'phase%4==1 || phase%4==2', 'instrumented?17:2', 'values[2*i+1]!=0',
                  'wall_ns<wrap_ns','(values[2*i]&~mask)==0','total_ticks*period<=wall_ns'):
        if token not in loop:raise ValueError('timing qualification '+token)
    tail=src[end:]
    if not tail.index('require(failure==0,"resident GPU status failure")')<tail.index('if(!timed)')<tail.index('owner.release();context.close()'):raise ValueError('status/lifetime')
    for item in ("proof.get('scope')!='fused stage parity'", "proof.get('manifest_sha256')!=manifest_sha", "proof.get('host')!=host", "proof.get('tiny')!=tiny", "proof.get('harness_sha256')!=harness_sha", "proof.get('baseline_sha256')!=baseline_sha", "proof['gpu_outputs'][key]!=baseline['gpu_outputs'][key]"):
        if item not in diag:raise ValueError('parity proof identity')


def current():return APP.read_text(),BASE.read_text(),DIAG.read_text()

def test_source():guard(*current())


@pytest.mark.parametrize('where,old,new',[
 ('src','if(instrumented)recorder.write_timestamp','recorder.write_timestamp'),
 ('src','auto mark=[&](){recorder.pipeline_barrier(', 'auto mark=[&](){if(instrumented)recorder.pipeline_barrier('),
 ('src','recorder.dispatch({pass<2?720u:1280u,1,1})','recorder.dispatch({1280u,1,1})'),
 ('src','recorder.set_pipeline(selection)','recorder.set_pipeline(metric)'),
 ('src','.queue_submit_index=submit_id','.queue_submit_index=0'),
 ('src','values[2*i+1]!=0','true'),('src','wall_ns<wrap_ns','true'),
 ('src','total_ticks*period<=wall_ns','true'),
 ('src','recorder.set_pipeline(metric)','d.buffer_host_address(raw);recorder.set_pipeline(metric)'),
 ('diag',"proof.get('host')!=host",'False'),
])
def test_mutations(where,old,new):
    src,base,diag=current();target=src if where=='src' else diag;assert old in target
    with pytest.raises((ValueError,IndexError)):guard(target.replace(old,new,1) if where=='src' else src,base,target.replace(old,new,1) if where=='diag' else diag)


def test_fake_actual_source(tmp_path):
    actual=block(APP.read_text())
    cpp=r'''#include <cstdint>
#include <cstddef>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <stdexcept>
namespace daxa{struct AccessConsts{static constexpr int HOST_WRITE=1,COMPUTE_SHADER_READ=2,COMPUTE_SHADER_WRITE=4,HOST_READ=8;};struct Barrier{int src_access,dst_access;};struct Info{const char*name;};struct Dispatch{uint32_t x,y,z;};struct Addr{uint64_t value()const{return 7;}};struct Cmd{};struct Query{int query_pool,start_index,count;};struct Timestamp{int query_pool,pipeline_stage,query_index;};struct PipelineStageFlagBits{static constexpr int TOP_OF_PIPE=1,BOTTOM_OF_PIPE=2;};struct Recorder{static inline uint64_t barriers=0,dispatches=0,timestamps=0;void pipeline_barrier(Barrier){++barriers;}void reset_timestamps(Query){}void write_timestamp(Timestamp){++timestamps;}void set_pipeline(int){}template<class T>void push_constant(T){}void dispatch(Dispatch){++dispatches;}Cmd complete_current_commands(){return {};}};struct Device{Recorder create_command_recorder(Info){return {};}Addr device_address(int){return {};}};}
struct Push{uint64_t p;uint32_t pass;};struct SelectPush{uint64_t p;};struct CostPush{uint64_t p;};struct CensusPush{uint64_t p;};struct ResetCostPush{uint64_t p;uint32_t n;};static void require(bool b,const char*){if(!b)throw std::runtime_error("query count");}
__attribute__((noinline))static daxa::Cmd run(daxa::Device& d,bool instrumented){asm volatile("" ::: "memory");uint64_t n=1280*720,max_words=100;int cp[2]={1,2},sel_params[2]={3,4},cost_params[2]={5,6},rec_params[2]={7,8};int queries=0,census=0,clear_cost=0,cost=0,sums=0,reset=0,matcher=0,recurrence=0,selection=0,lr=0,metric=0;
''' + actual + r'''
}
int main(){daxa::Device d;constexpr unsigned n=100000;auto t=std::chrono::steady_clock::now();for(unsigned i=0;i<n;++i){run(d,false);run(d,true);}auto e=std::chrono::steady_clock::now();if(daxa::Recorder::barriers!=n*64||daxa::Recorder::dispatches!=n*32||daxa::Recorder::timestamps!=n*19)return 1;std::printf("stage fake %.2f ns per pair; each 16 dispatches 32 barriers; 2/17 timestamps\n",std::chrono::duration<double,std::nano>(e-t).count()/n);}
'''
    path=tmp_path/'fake.cpp';exe=tmp_path/'fake';path.write_text(cpp)
    subprocess.run(['c++','-std=c++20','-O2',str(path),'-o',str(exe)],check=True,capture_output=True,text=True)
    r=subprocess.run([str(exe)],check=True,capture_output=True,text=True);print(r.stdout.strip())
    assert 'each 16 dispatches 32 barriers; 2/17 timestamps' in r.stdout
