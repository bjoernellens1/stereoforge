"""CPU-only guards for the fused four-path recurrence experiment."""
from pathlib import Path
import subprocess
import pytest
import test_fp2_resident_hotpath as resident
import test_fp2_matched_timing_hotpath as matched
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_fused_fixture.cpp';TINY=ROOT/'apps/fp2_fused_tiny.cpp'
SHADER=ROOT/'python/stereoforge/shaders/daxa/fp2_banded_fused.slang'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp';DIAG=ROOT/'benchmarks/fp2_fused_diagnostic.py';MATCH=ROOT/'benchmarks/fp2_fused_matched.py'


def shader_guard(source):
    s=''.join(source.split())
    for x in ('[numthreads(128,1,1)]','if(line>=(horizontal?p.height:p.width))return;',
              'previous[lane]=INF;GroupMemoryBarrierWithGroupSync();',
              'for(uintstride=64;stride>0;stride>>=1)', 'if(lane<p.disparities&&lane>=lo[i]&&lane<hi[i])',
              'offsets[i]+lane-lo[i]', 'uintc=INF;', 'p.eye==0?xy.x-int(lane):xy.x+int(lane)',
              'support[i]!=0&&target>=0&&target<int(p.width)', 'if(other_support[other_i]!=0)c=countbits(own[i]^other[other_i]);',
              'if(c!=INF)', 'if(minimum!=INF)', 'current[lane]=value;GroupMemoryBarrierWithGroupSync();previous[lane]=current[lane];GroupMemoryBarrierWithGroupSync();',
              'if(p.paths!=0)paths[uint64_t(pc.pass)*allocated+j]=value;'):
        if x not in s:raise ValueError('shader '+x)
    if s.count('return;')!=1 or s.count('GroupMemoryBarrierWithGroupSync();')!=5:raise ValueError('uniform barriers')


def host_guard(src,tiny,owner):
    for s in (src,tiny):
        for x in ('limits.maxComputeWorkGroupInvocations>=128','limits.maxComputeSharedMemorySize>=1536',
                  'vkGetPhysicalDeviceProperties(daxa_dvc_get_vk_physical_device(d.get()),&props)',
                  'admit_parallel_device(d,uint32_t(std::max<uint64_t>', 'validate_regions(regions)'):
            if x not in s:raise ValueError('device/resources '+x)
    for x in ('addr(desc[eye]),addr(desc[1-eye]),addr(valid[eye]),addr(valid[1-eye])',
              'uint32_t(eye)', 'addr(sums),0,addr(desc[eye])', 'validate_metric_lut(lut,numerator,cx,reinterpret_cast<uintptr_t>(d.get()))',
              'size_index<sizes.size()&&bytes==sizes[size_index++]','sizes.size()==buffers.size()&&size_index==sizes.size()'):
        if x not in src:raise ValueError('admission/eye '+x)
    if 'auto cost=alloc(' in src or 'cost_params' in src or 'recorder.set_pipeline(matcher)' in src:raise ValueError('materialized cost path')
    if '/fp2-fused.spv' not in src or 'SF_FP2_FUSED_SPIRV_SHA256' not in src:raise ValueError('fused shader identity')
    tail=src[src.index('uint32_t failure=0;',src.index('for(uint32_t frame=0;frame<iterations;++frame)')):]
    if not tail.index('d.buffer_host_address(status).value()')<tail.index('require(failure==0,"resident GPU status failure")')<tail.index('if(!timed)')<tail.index('owner.release();context.close()'):raise ValueError('status/teardown')
    begin=src.index('for(uint32_t frame=0;frame<iterations;++frame)');end=src.index('uint32_t failure=0;',begin);loop=src[begin:end]
    if not loop.index('auto commands=record()')<loop.index('d.submit_commands')<loop.index('.queue_submit_index=submit_id')<loop.index('daxa_timeline_query_pool_query_results'):raise ValueError('query completion')
    if any(x in loop for x in ('buffer_host_address','memcpy','create_buffer','std::ofstream')):raise ValueError('frame readback')
    cmd=resident.record(src)
    for forbidden in ('recorder.set_pipeline(matcher)','recorder.set_pipeline(clear_cost)','buffer_host_address','memcpy','wait_idle','wait_on_submit','create_buffer','create_memory'):
        if forbidden in cmd:raise ValueError('unexpected stage '+forbidden)
    for x in ('recorder.set_pipeline(census)','recorder.set_pipeline(reset)','recorder.set_pipeline(recurrence)',
              'for(uint32_t pass=0;pass<4;++pass)', 'Push{d.device_address(rec_params[eye]).value(),pass}',
              'recorder.set_pipeline(selection)','recorder.set_pipeline(lr)','recorder.set_pipeline(metric)',
              'recorder.dispatch({pass<2?720u:1280u,1,1})'):
        if x not in cmd:raise ValueError('command '+x)
    barrier='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    points=[cmd.index(x) for x in ('recorder.set_pipeline(census)','for(uint32_t eye=0;eye<2;++eye)','for(uint32_t pass=0;pass<4;++pass)','recorder.set_pipeline(recurrence)','recorder.set_pipeline(selection)','recorder.set_pipeline(lr)','recorder.set_pipeline(metric)')]
    for a,b in zip(points,points[1:]):
        if barrier not in cmd[a:b]:raise ValueError('stage barrier')
    for x in ('std::memcmp(d.buffer_host_address(debug_sum[eye]).value(),oracle_out[eye].sums.data()',
              'std::memcmp(d.buffer_host_address(debug_paths[eye]).value(),oracle_out[eye].paths.data()',
              'd.wait_idle();uint32_t failure=0'):
        if x not in tiny:raise ValueError('tiny parity')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in owner:raise ValueError('cleanup')


def test_sources():shader_guard(SHADER.read_text());host_guard(APP.read_text(),TINY.read_text(),OWNER.read_text())


@pytest.mark.parametrize('old,new',[
 ('[numthreads(128,1,1)]','[numthreads(64,1,1)]'),('p.eye==0?xy.x-int(lane):xy.x+int(lane)','p.eye==0?xy.x+int(lane):xy.x-int(lane)'),
 ('support[i]!=0 && target>=0 && target<int(p.width)','support[i]!=0'),('other_support[other_i]!=0','true'),
 ('uint c=INF;','uint c=0;'),('offsets[i]+lane-lo[i]','offsets[i]+lane'),
 ('if(line >= (horizontal?p.height:p.width)) return;','if(lane>0)return;'),('GroupMemoryBarrierWithGroupSync();',''),
])
def test_shader_mutations(old,new):
    s=SHADER.read_text();assert old in s
    with pytest.raises(ValueError):shader_guard(s.replace(old,new,1))


@pytest.mark.parametrize('which,old,new',[
 ('src','addr(desc[eye]),addr(desc[1-eye])','addr(desc[eye]),addr(desc[eye])'),
 ('src','addr(sums),0,addr(desc[eye])','addr(sums),addr(paths),addr(desc[eye])'),
 ('src','recorder.set_pipeline(recurrence)','d.wait_idle();recorder.set_pipeline(recurrence)'),
 ('src','recorder.set_pipeline(metric)','d.buffer_host_address(raw);recorder.set_pipeline(metric)'),
 ('tiny','oracle_out[eye].paths.data()','oracle_out[0].paths.data()'),
 ('src','limits.maxComputeWorkGroupInvocations>=128','true'),
 ('src','/fp2-fused.spv','/fp2-parallel.spv'),
 ('src','.queue_submit_index=submit_id','.queue_submit_index=0'),
 ('src','require(failure==0,"resident GPU status failure")','require(true,"ignored")'),
])
def test_host_mutations(which,old,new):
    src=APP.read_text();tiny=TINY.read_text();target=src if which=='src' else tiny;assert old in target
    with pytest.raises((ValueError,IndexError)):host_guard(target.replace(old,new,1) if which=='src' else src,target.replace(old,new,1) if which=='tiny' else tiny,OWNER.read_text())


def test_matched_proof_scope():
    d=DIAG.read_text();m=MATCH.read_text();matched.guard(m,work_labels=("'128 lanes and tree reduction per path pixel including inactive candidates; candidate recomputes descriptor cost in every path'",))
    for x in ("proof.get('scope')!='fused parity'", "proof.get('host')!=host", "proof.get('tiny')!=tiny", "proof['gpu_outputs'][key]!=baseline['gpu_outputs'][key]"):
        assert x in d


def test_fake_actual_recording(tmp_path):
    actual=resident.record(APP.read_text())
    cpp=r'''#include <cstdint>
#include <cstddef>
#include <chrono>
#include <cstdio>
#include <algorithm>
namespace daxa{struct AccessConsts{static constexpr int HOST_WRITE=1,COMPUTE_SHADER_READ=2,COMPUTE_SHADER_WRITE=4,HOST_READ=8;};struct Barrier{int src_access,dst_access;};struct Info{const char*name;};struct Dispatch{uint32_t x,y,z;};struct Addr{uint64_t value()const{return 7;}};struct Cmd{};struct Query{int query_pool,start_index,count;};struct Timestamp{int query_pool,pipeline_stage,query_index;};struct PipelineStageFlagBits{static constexpr int TOP_OF_PIPE=1,BOTTOM_OF_PIPE=2;};struct Recorder{static inline uint64_t barriers=0,dispatches=0,queries=0,timestamps=0;void pipeline_barrier(Barrier){++barriers;}void reset_timestamps(Query){++queries;}void write_timestamp(Timestamp){++timestamps;}void set_pipeline(int){}template<class T>void push_constant(T){}void dispatch(Dispatch){++dispatches;}Cmd complete_current_commands(){return {};}};struct Device{Recorder create_command_recorder(Info){return {};}Addr device_address(int){return {};}};}
struct Push{uint64_t p;uint32_t pass;};struct SelectPush{uint64_t p;};struct CensusPush{uint64_t p;};struct ResetCostPush{uint64_t p;uint32_t n;};
__attribute__((noinline))static daxa::Cmd run(daxa::Device& d){asm volatile("" ::: "memory");uint64_t n=1280*720,max_words=100;int cp[2]={1,2},sel_params[2]={3,4},rec_params[2]={7,8};int queries=0,census=0,sums=0,reset=0,recurrence=0,selection=0,lr=0,metric=0;
''' + actual + r'''
}
int main(){daxa::Device d;constexpr unsigned n=100000;auto t=std::chrono::steady_clock::now();for(unsigned i=0;i<n;++i)run(d);auto e=std::chrono::steady_clock::now();if(daxa::Recorder::barriers!=n*16||daxa::Recorder::dispatches!=n*16||daxa::Recorder::queries!=n||daxa::Recorder::timestamps!=n*2)return 1;std::printf("fused fake %.2f ns; 16 dispatches 16 barriers 1 query reset 2 timestamps\n",std::chrono::duration<double,std::nano>(e-t).count()/n);}
'''
    path=tmp_path/'fake.cpp';exe=tmp_path/'fake';path.write_text(cpp)
    subprocess.run(['c++','-std=c++20','-O2',str(path),'-o',str(exe)],check=True,capture_output=True,text=True)
    r=subprocess.run([str(exe)],check=True,capture_output=True,text=True);print(r.stdout.strip())
    assert '16 dispatches 16 barriers 1 query reset 2 timestamps' in r.stdout
