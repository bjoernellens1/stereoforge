"""CPU-only resident diagnostic guards; no GPU or production performance claim."""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_resident_fixture.cpp'
DIAG=ROOT/'benchmarks/fp2_resident_diagnostic.py'
NOPATH=ROOT/'python/stereoforge/shaders/daxa/fp2_banded_no_path.slang'
SUM=ROOT/'python/stereoforge/shaders/daxa/fp2_sum_reset.slang'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'


def record(s):
    a=s.index('auto recorder=d.create_command_recorder({.name="resident FP2"})')
    b=s.index('return recorder.complete_current_commands();',a)+len('return recorder.complete_current_commands();')
    return s[a:b]


def guard(src,diag,nopath,reset,owner):
    code=''.join(nopath.split());reset=''.join(reset.split())
    if 'p.paths' in code or 'paths[' in code:raise ValueError('path writes')
    for item in ('for(uintt=0;t<prev_width;++t)','for(uintt=0;t<width;++t)','uintd=lo[i]+t;', 'scratch[current+t]=v;'):
        if item not in code:raise ValueError('compact recurrence '+item)
    if 'for(uinti=tid.x;i<pc.words;i+=stride)((uint*)pc.sums)[i]=0xffffffffu;' not in reset or 'pc.paths' in reset:raise ValueError('sum reset')
    for item in ('std::max(f[0].offsets.back(),f[1].offsets.back())','std::max(f[0].scratch_words,f[1].scratch_words)',
                 'max_words&&max_words<=65000000', 'size_index<sizes.size()&&bytes==sizes[size_index++]',
                 'sizes.size()==buffers.size()&&size_index==sizes.size()', 'admission(context,sizes)', 'validate_regions(regions)',
                 'addr(desc[eye]),addr(desc[1-eye]),addr(valid[eye]),addr(valid[1-eye])',
                 'uint32_t(eye),uint32_t(max_words),numerator,cx,uint32_t(source),uint32_t(source>>32),addr(status),addr(lut_buffer)',
                 'validate_metric_lut(lut,numerator,cx,reinterpret_cast<uintptr_t>(d.get()))'):
        if item not in src:raise ValueError('resource '+item)
    if 'paths=alloc(' in src or 'max_words*16' in src:raise ValueError('path allocation')
    for item in ('addr(sums),0,addr(scratch)', 'addr(sums),0,addr(raw)'):
        if item not in src:raise ValueError('null path pointer')
    for name in ('census','cost','rec','reset_cost','reset','select','lr','metric'):
        if f'.byte_code_size=uint32_t({name}_words.size())' not in src:raise ValueError('word count')
    cmd=record(src)
    for item in ('reset_timestamps({.query_pool=queries,.start_index=0,.count=2})','query_index=0','query_index=1',
                 'CensusPush{d.device_address(cp[eye]).value()}', 'if(eye)recorder.pipeline_barrier',
                 'ResetCostPush{d.device_address(cost).value(),uint32_t(max_words)}',
                 'ResetCostPush{d.device_address(sums).value(),uint32_t(max_words)}',
                 'CostPush{d.device_address(cost_params[eye]).value()}', 'Push{d.device_address(rec_params[eye]).value(),pass}',
                 'SelectPush{d.device_address(sel_params[eye]).value()}', 'for(uint32_t pass=0;pass<4;++pass)'):
        if item not in cmd:raise ValueError('recording '+item)
    host='HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    gpu='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    reuse='.src_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    marks=['recorder.set_pipeline(census)','for(uint32_t eye=0;eye<2;++eye)','recorder.set_pipeline(clear_cost)','recorder.set_pipeline(reset)','recorder.set_pipeline(matcher)','for(uint32_t pass=0;pass<4;++pass)','recorder.set_pipeline(recurrence)','recorder.set_pipeline(selection)','recorder.set_pipeline(lr)','recorder.set_pipeline(metric)'];p=[cmd.index(x) for x in marks]
    if not(host in cmd[:p[0]] and gpu in cmd[p[0]:p[1]] and reuse in cmd[p[1]:p[2]] and gpu in cmd[p[3]:p[4]] and gpu in cmd[p[4]:p[5]] and gpu in cmd[p[5]:p[6]] and gpu in cmd[p[6]:p[7]] and gpu in cmd[p[7]:p[8]] and gpu in cmd[p[8]:p[9]]):raise ValueError('barrier chain')
    for token in ('buffer_host_address','memcpy','memset','create_buffer','create_memory','wait_on_submit','wait_idle','std::ofstream'):
        if token in cmd:raise ValueError('host operation in recording '+token)
    start=src.index('for(uint32_t frame=0;frame<iterations;++frame)');end=src.index('uint32_t failure=0;',start);loop=src[start:end]
    if not(loop.index('auto commands=record()')<loop.index('d.submit_commands')<loop.index('auto submit_id=d.latest_queue_submit_index')<loop.index('d.wait_on_submit')<loop.index('daxa_timeline_query_pool_query_results')):raise ValueError('one-slot query lifetime')
    if '.queue_submit_index=submit_id' not in loop:raise ValueError('slot completion ticket')
    for token in ('buffer_host_address','memcpy','memset','create_buffer','create_memory','std::ofstream'):
        if token in loop:raise ValueError('frame readback/allocation')
    tail=src[end:]
    if not tail.index('require(failure==0,"resident GPU status failure")')<tail.index('if(!timed)')<tail.index('std::ofstream')<tail.index('owner.release();context.close()'):raise ValueError('debug/status order')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in owner:raise ValueError('lifetime cleanup')
    for item in ("proof.get('scope')!='resident parity'", "proof.get('manifest_sha256')!=manifest_sha", "proof.get('host')!=host", "proof.get('tiny')!=tiny", "proof.get('harness_sha256')!=harness_sha", "proof.get('baseline_sha256')!=baseline_sha", "proof['gpu_outputs'][key]!=baseline['gpu_outputs'][key]"):
        if item not in diag:raise ValueError('parity identity gate')


def current():return APP.read_text(),DIAG.read_text(),NOPATH.read_text(),SUM.read_text(),OWNER.read_text()

def test_source():guard(*current())


@pytest.mark.parametrize('which,old,new',[
 ('nopath','scratch[current+t]=v;','paths[j]=v;scratch[current+t]=v;'),
 ('reset','i<pc.words','i<pc.words/2'),('src','paths=alloc(','paths=alloc('),
 ('src','addr(sums),0,addr(scratch)','addr(sums),addr(paths),addr(scratch)'),
 ('src','ResetCostPush{d.device_address(sums).value(),uint32_t(max_words)}','ResetCostPush{d.device_address(sums).value(),16}'),
 ('src','d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=submit_id','d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=0'),
 ('src','recorder.set_pipeline(metric)','d.buffer_host_address(raw);recorder.set_pipeline(metric)'),
 ('diag',"proof.get('manifest_sha256')!=manifest_sha",'False'),
])
def test_mutations(which,old,new):
    src,diag,nopath,reset,owner=current()
    target={'src':src,'diag':diag,'nopath':nopath,'reset':reset}[which]
    if old==new: mutated=src.replace('auto cost=alloc(', 'auto paths=alloc(4,nullptr);auto cost=alloc(',1)
    else:
        assert old in target;mutated=target.replace(old,new,1)
    with pytest.raises((ValueError,IndexError)):
        guard(mutated if which=='src' else src,mutated if which=='diag' else diag,mutated if which=='nopath' else nopath,mutated if which=='reset' else reset,owner)


def test_fake_actual_recording(tmp_path):
    actual=record(APP.read_text())
    cpp=r'''#include <cstdint>
#include <cstddef>
#include <chrono>
#include <cstdio>
#include <algorithm>
namespace daxa{struct AccessConsts{static constexpr int HOST_WRITE=1,COMPUTE_SHADER_READ=2,COMPUTE_SHADER_WRITE=4,HOST_READ=8;};struct Barrier{int src_access,dst_access;};struct Info{const char*name;};struct Dispatch{uint32_t x,y,z;};struct Addr{uint64_t value()const{return 7;}};struct Cmd{};struct Query{int query_pool,start_index,count;};struct Timestamp{int query_pool,pipeline_stage,query_index;};struct PipelineStageFlagBits{static constexpr int TOP_OF_PIPE=1,BOTTOM_OF_PIPE=2;};struct Recorder{static inline uint64_t barriers=0,dispatches=0,queries=0,timestamps=0;void pipeline_barrier(Barrier){++barriers;}void reset_timestamps(Query){++queries;}void write_timestamp(Timestamp){++timestamps;}void set_pipeline(int){}template<class T>void push_constant(T){}void dispatch(Dispatch){++dispatches;}Cmd complete_current_commands(){return {};}};struct Device{Recorder create_command_recorder(Info){return {}; }Addr device_address(int){return {};}};}
struct Push{uint64_t p;uint32_t pass;};struct SelectPush{uint64_t p;};struct CostPush{uint64_t p;};struct CensusPush{uint64_t p;};struct ResetCostPush{uint64_t p;uint32_t n;};
__attribute__((noinline))static daxa::Cmd run(daxa::Device& d){asm volatile("" ::: "memory");uint64_t n=1280*720,max_words=100;int cp[2]={1,2},sel_params[2]={3,4},cost_params[2]={5,6},rec_params[2]={7,8};int queries=0,census=0,clear_cost=0,cost=0,sums=0,reset=0,matcher=0,recurrence=0,selection=0,lr=0,metric=0;
''' + actual + r'''
}
int main(){daxa::Device d;constexpr unsigned n=100000;auto t=std::chrono::steady_clock::now();for(unsigned i=0;i<n;++i)run(d);auto e=std::chrono::steady_clock::now();if(daxa::Recorder::barriers!=n*18||daxa::Recorder::dispatches!=n*20||daxa::Recorder::queries!=n||daxa::Recorder::timestamps!=n*2)return 1;std::printf("resident fake %.2f ns; 20 dispatches 18 barriers 1 query reset 2 timestamps, no submission/wait\n",std::chrono::duration<double,std::nano>(e-t).count()/n);}
'''
    path=tmp_path/'fake.cpp';exe=tmp_path/'fake';path.write_text(cpp)
    subprocess.run(['c++','-std=c++20','-O2',str(path),'-o',str(exe)],check=True,capture_output=True,text=True)
    r=subprocess.run([str(exe)],check=True,capture_output=True,text=True);print(r.stdout.strip())
    assert '20 dispatches 18 barriers 1 query reset 2 timestamps, no submission/wait' in r.stdout


@pytest.mark.parametrize('start,end,old',[
 ('if(eye)recorder.pipeline_barrier','recorder.set_pipeline(clear_cost)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(reset)','recorder.set_pipeline(matcher)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(matcher)','for(uint32_t pass=0;pass<4;++pass)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(selection)','recorder.set_pipeline(lr)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(lr)','recorder.set_pipeline(metric)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
])
def test_scoped_barrier_mutations(start,end,old):
    src,diag,nopath,reset,owner=current();a=src.index(start,src.index('auto recorder=d.create_command_recorder({.name="resident FP2"})'));b=src.index(end,a);assert old in src[a:b]
    with pytest.raises(ValueError):guard(src[:a]+src[a:b].replace(old,'.src_access=daxa::AccessConsts::HOST_WRITE',1)+src[b:],diag,nopath,reset,owner)


TINY=ROOT/'apps/fp2_resident_tiny.cpp'

def guard_tiny(src):
    s=src[src.index('static void gpu_selection('):]
    start=s.index('auto recorder=d.create_command_recorder(');end=s.index('d.wait_idle();',start)+len('d.wait_idle();');cmd=s[start:end]
    for token in ('debug_sum[eye]', 'recorder.copy_buffer_to_buffer({.src_buffer=sums,.dst_buffer=debug_sum[eye],.size=f[eye].offsets.back()*4})',
                  '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ|daxa::AccessConsts::COMPUTE_SHADER_READ',
                  '.src_access=daxa::AccessConsts::TRANSFER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE',
                  '.src_access=daxa::AccessConsts::TRANSFER_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ'):
        if token not in cmd:raise ValueError('tiny sum capture invariant')
    for token in ('buffer_host_address','memcpy','wait_idle();d.wait_idle','create_buffer','create_memory'):
        if token in cmd:raise ValueError('tiny intermediate host operation')
    tail=s[end:]
    if not(tail.index('require(failure==0')<tail.index('d.buffer_host_address(debug_sum[eye]).value()')<tail.index('oracle_out[eye].sums.data()')<tail.index('owner.release();context.close()')):raise ValueError('tiny parity/lifetime')
    return cmd


def test_tiny_source():guard_tiny(TINY.read_text())


@pytest.mark.parametrize('old,new',[
    ('.dst_buffer=debug_sum[eye]','.dst_buffer=debug_sum[0]'),
    ('.size=f[eye].offsets.back()*4','.size=4'),
    ('.src_access=daxa::AccessConsts::TRANSFER_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ','.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ'),
    ('oracle_out[eye].sums.data()','oracle_out[0].sums.data()'),
])
def test_tiny_mutations(old,new):
    src=TINY.read_text();assert old in src
    with pytest.raises(ValueError):guard_tiny(src.replace(old,new,1))
