"""CPU-only guards for integrated diagnostic; no production or GPU timing claims."""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_integrated_fixture.cpp'
RESET=ROOT/'python/stereoforge/shaders/daxa/fp2_integrated_reset.slang'
SELECT=ROOT/'python/stereoforge/shaders/daxa/fp2_selection.slang'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'


def block(source):
    start=source.index('     auto recorder=d.create_command_recorder(')
    return source[start:source.index('d.wait_idle();',start)+len('d.wait_idle();')]


def guard(src,reset,select,owner):
    compact=''.join(reset.split()); sel=''.join(select.split())
    for item in ('for(uinti=tid.x;i<pc.words;i+=stride)((uint*)pc.costs)[i]=0xffffffffu;',):
        if item not in compact:raise ValueError('cost reset capacity')
    for item in ('for(uinti=tid.x;i<p.max_words;i+=stride)', '((uint*)p.sums)[i]=INF', '((uint*)p.paths)[pass*p.max_words+i]=INF', '((Raw*)p.raw)[p.eye*n+i]=r'):
        if item not in sel:raise ValueError('sum/path reset or eye partition')
    for item in ('std::max(f[0].offsets.back(),f[1].offsets.back())','std::max(f[0].scratch_words,f[1].scratch_words)',
                 'max_words&&max_words<=65000000', 'size_index<sizes.size()&&bytes==sizes[size_index++]',
                 'sizes.size()==buffers.size()&&size_index==sizes.size()', 'admission(context,sizes)', 'validate_regions(regions)',
                 'addr(desc[eye]),addr(desc[1-eye]),addr(valid[eye]),addr(valid[1-eye])',
                 'uint32_t(eye),uint32_t(max_words),numerator,cx,uint32_t(source),uint32_t(source>>32),addr(status),addr(lut_buffer)',
                 'validate_metric_lut(lut,numerator,cx,reinterpret_cast<uintptr_t>(d.get()))'):
        if item not in src:raise ValueError('resource invariant '+item)
    if src.index('admission(context,sizes)')>src.index('auto cost=alloc('):raise ValueError('admission order')
    for name in ('census','cost','rec','reset_cost','reset','select','lr','metric'):
        if f'.byte_code_size=uint32_t({name}_words.size())' not in src:raise ValueError('shader word count')
    cmd=block(src)
    for item in ('CensusPush{d.device_address(cp[eye]).value()}', 'for(uint32_t eye=0;eye<2;++eye)',
                 'if(eye)recorder.pipeline_barrier', 'ResetCostPush{d.device_address(cost).value(),uint32_t(max_words)}',
                 'SelectPush{d.device_address(sel_params[eye]).value()}', 'CostPush{d.device_address(cost_params[eye]).value()}',
                 'Push{d.device_address(rec_params[eye]).value(),pass}', 'for(uint32_t pass=0;pass<4;++pass)',
                 'd.submit_commands', 'd.wait_idle()'):
        if item not in cmd:raise ValueError('command invariant '+item)
    gpu='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    reuse='.src_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    host='.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    finish='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ'
    marks=['recorder.set_pipeline(census)','for(uint32_t eye=0;eye<2;++eye)','recorder.set_pipeline(clear_cost)','recorder.set_pipeline(reset)','recorder.set_pipeline(matcher)','for(uint32_t pass=0;pass<4;++pass)','recorder.set_pipeline(recurrence)','recorder.set_pipeline(selection)','recorder.set_pipeline(lr)','recorder.set_pipeline(metric)']
    pos=[cmd.index(x) for x in marks];second=cmd.index('recorder.dispatch(',cmd.index('CensusPush{d.device_address(cp[eye]).value()}'))
    if not(host in cmd[:pos[0]] and gpu in cmd[second:pos[1]] and reuse in cmd[pos[1]:pos[2]] and gpu in cmd[pos[3]:pos[4]] and gpu in cmd[pos[4]:pos[5]] and gpu in cmd[pos[5]:pos[6]] and gpu in cmd[pos[6]:pos[7]] and gpu in cmd[pos[7]:pos[8]] and gpu in cmd[pos[8]:pos[9]] and finish in cmd[pos[9]:]):raise ValueError('stage barrier')
    for token in ('buffer_host_address','memcpy','memset','create_buffer','create_memory','wait_on_submit','collect_garbage','std::ofstream'):
        if token in cmd:raise ValueError('host operation in chain')
    if src.count('d.wait_idle()')!=1:raise ValueError('extra wait')
    tail=src[src.index('d.wait_idle();'):]
    if not tail.index('d.buffer_host_address(status).value()')<tail.index('require(failure==0,"integrated GPU status failure")')<tail.index('std::ofstream')<tail.index('owner.release();context.close()'):raise ValueError('status/readback/lifetime')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in owner:raise ValueError('owner cleanup')


def current():return APP.read_text(),RESET.read_text(),SELECT.read_text(),OWNER.read_text()

def test_source():guard(*current())


@pytest.mark.parametrize('old,new',[
 ('uint32_t(max_words)}','uint32_t(16)}'),('addr(desc[eye]),addr(desc[1-eye])','addr(desc[eye]),addr(desc[eye])'),
 ('if(eye)recorder.pipeline_barrier','if(false)recorder.pipeline_barrier'),
 ('size_index<sizes.size()&&bytes==sizes[size_index++]','true'),('validate_regions(regions)',''),
 ('validate_metric_lut(lut,numerator,cx,reinterpret_cast<uintptr_t>(d.get()))','(void)lut'),
 ('recorder.set_pipeline(matcher)','d.wait_idle();recorder.set_pipeline(matcher)'),
 ('recorder.set_pipeline(lr)','d.buffer_host_address(raw);recorder.set_pipeline(lr)'),
 ('.byte_code_size=uint32_t(metric_words.size())','.byte_code_size=uint32_t(metric_words.size()*4)'),
 ('require(failure==0,"integrated GPU status failure")','require(true,"ignored")'),
 ('owner.release();context.close()','context.close();owner.release()'),
])
def test_host_mutations(old,new):
    src,reset,select,owner=current();assert old in src
    with pytest.raises((ValueError,IndexError)):guard(src.replace(old,new,1),reset,select,owner)


@pytest.mark.parametrize('which,old,new',[
 ('reset','i<pc.words','i<pc.words/2'),('select','i<p.max_words','i<p.max_words/2'),
 ('select','((uint*)p.paths)[pass*p.max_words+i]=INF','((uint*)p.paths)[pass*p.max_words+i]=0'),
 ('select','((Raw*)p.raw)[p.eye*n+i]=r','((Raw*)p.raw)[i]=r'),
])
def test_shader_mutations(which,old,new):
    src,reset,select,owner=current();target=reset if which=='reset' else select;assert old in target
    with pytest.raises(ValueError):guard(src,target.replace(old,new,1) if which=='reset' else reset,target.replace(old,new,1) if which=='select' else select,owner)


def test_fake_actual_source(tmp_path):
    actual=block(APP.read_text())
    cpp=r'''#include <cstdint>
#include <cstddef>
#include <chrono>
#include <cstdio>
#include <algorithm>
namespace daxa{struct AccessConsts{static constexpr int HOST_WRITE=1,COMPUTE_SHADER_READ=2,COMPUTE_SHADER_WRITE=4,HOST_READ=8;};struct Barrier{int src_access,dst_access;};struct Info{const char*name;};struct Dispatch{uint32_t x,y,z;};struct Addr{uint64_t value()const{return 7;}};struct Cmd{};struct Submit{struct List{Cmd*p;int n;}command_lists;};struct Recorder{static inline uint64_t barriers=0,dispatches=0;void pipeline_barrier(Barrier){++barriers;}void set_pipeline(int){}template<class T>void push_constant(T){}void dispatch(Dispatch){++dispatches;}Cmd complete_current_commands(){return {};}};struct Device{static inline uint64_t submits=0,waits=0;Recorder create_command_recorder(Info){return {};}Addr device_address(int){return {};}void submit_commands(Submit){++submits;}void wait_idle(){++waits;}};}
struct Push{uint64_t p;uint32_t pass;};struct SelectPush{uint64_t p;};struct CostPush{uint64_t p;};struct CensusPush{uint64_t p;};struct ResetCostPush{uint64_t p;uint32_t n;};
__attribute__((noinline))static void run(daxa::Device& d){asm volatile("" ::: "memory");uint64_t n=1280*720,max_words=100;int cp[2]={1,2},sel_params[2]={3,4},cost_params[2]={5,6},rec_params[2]={7,8};int census=0,clear_cost=0,cost=0,reset=0,matcher=0,recurrence=0,selection=0,lr=0,metric=0;
''' + actual + r'''
}
int main(){daxa::Device d;constexpr unsigned n=100000;auto t=std::chrono::steady_clock::now();for(unsigned i=0;i<n;++i)run(d);auto e=std::chrono::steady_clock::now();if(daxa::Recorder::barriers!=n*18||daxa::Recorder::dispatches!=n*20||daxa::Device::submits!=n||daxa::Device::waits!=n)return 1;std::printf("integrated fake %.2f ns; 20 dispatches 18 barriers 1 submit 1 diagnostic fake wait\n",std::chrono::duration<double,std::nano>(e-t).count()/n);}
'''
    path=tmp_path/'fake.cpp';exe=tmp_path/'fake';path.write_text(cpp)
    subprocess.run(['c++','-std=c++20','-O2',str(path),'-o',str(exe)],check=True,capture_output=True,text=True)
    r=subprocess.run([str(exe)],check=True,capture_output=True,text=True);print(r.stdout.strip())
    assert '20 dispatches 18 barriers 1 submit 1 diagnostic fake wait' in r.stdout


@pytest.mark.parametrize('start,end,needle',[
 ('for(uint32_t eye=0;eye<2;++eye)','recorder.set_pipeline(clear_cost)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(reset)','recorder.set_pipeline(matcher)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(matcher)','for(uint32_t pass=0;pass<4;++pass)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(selection)','recorder.set_pipeline(lr)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(lr)','recorder.set_pipeline(metric)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
])
def test_stage_barrier_mutations(start,end,needle):
    src,reset,select,owner=current();a=src.index(start,src.index('auto recorder=d.create_command_recorder('));b=src.index(end,a);assert needle in src[a:b]
    with pytest.raises(ValueError):guard(src[:a]+src[a:b].replace(needle,'.src_access=daxa::AccessConsts::HOST_WRITE',1)+src[b:],reset,select,owner)
