"""CPU-only experimental selection chain guards; not a production runtime benchmark."""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_selection_fixture.cpp'
SHADER=ROOT/'python/stereoforge/shaders/daxa/fp2_selection.slang'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'


def block(s):
    start=s.index('     auto recorder=d.create_command_recorder(')
    return s[start:s.index('d.wait_idle();',start)+len('d.wait_idle();')]


def guard_shader(source):
    s=''.join(source.split())
    required=['for(uinti=tid.x;i<p.max_words;i+=stride)', '((uint*)p.sums)[i]=INF',
              '((uint*)p.paths)[pass*p.max_words+i]=INF', 'if(i>=n)return;',
              'for(uintd=lo[i];d<hi[i];++d)', 'sums[offsets[i]+d-lo[i]]', '((Raw*)p.raw)[p.eye*n+i]=r',
              'if(t>=2*n)return;', 'eye!=0?int(x)+a.winner:int(x)-a.winner',
              'o.depth=NAN_BITS', 'o.metric_valid=0', 'o.provenance=0', 'o.source_low=INF;o.source_high=INF',
              'o.age_low=0;o.age_high=0', 'o.source_low=p.source_low;o.source_high=p.source_high',
              'a.winner<=0||a.winner>=int(p.disparities)', 'InterlockedOr(((uint*)p.status)[0],2,old)',
              'uint2entry=((uint2*)p.metric_lut)[uint(a.winner)]', 'if(entry.y!=0){o.depth=entry.x;',
              'if(v>224){uintold;InterlockedOr(((uint*)p.status)[0],1,old);continue;}',
              'r.confidence=confidence_rte(r.second-r.best,r.second);']
    for item in required:
        if item not in s:raise ValueError('shader invariant '+item)
    if s.count('if(t>=2*n)return;')!=2:raise ValueError('both observation tails')
    metric=s[s.index('voidmetric('):]
    if 'p.numerator/' in metric or 'precisefloat' in metric:raise ValueError('metric float division prohibited')


def guard_host(source,owner):
    live=source[source.index('static void gpu_selection('):]
    cmd=block(live)
    need=['for(uint32_t eye=0;eye<2;++eye)', 'if(eye)recorder.pipeline_barrier',
          'recorder.set_pipeline(reset)', 'recorder.set_pipeline(recurrence)', 'recorder.set_pipeline(selection)',
          'recorder.set_pipeline(lr)', 'recorder.set_pipeline(metric)', 'for(uint32_t pass=0;pass<4;++pass)',
          'SelectPush{d.device_address(sel_param[eye]).value()}', 'Push{d.device_address(rec_param[eye]).value(),pass}',
          'SelectPush{d.device_address(sel_param[0]).value()}', 'd.submit_commands','d.wait_idle()']
    for item in need:
        if item not in cmd:raise ValueError('command invariant '+item)
    host='.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    gpu='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    reuse='.src_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    finish='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ'
    a=cmd.index('for(uint32_t eye=0;eye<2;++eye)');b=cmd.index('recorder.set_pipeline(reset)');c=cmd.index('for(uint32_t pass=0;pass<4;++pass)');d=cmd.index('recorder.set_pipeline(recurrence)');e=cmd.index('recorder.set_pipeline(selection)');f=cmd.index('recorder.set_pipeline(lr)');g=cmd.index('recorder.set_pipeline(metric)')
    if not(host in cmd[:a] and reuse in cmd[a:b] and gpu in cmd[b:c] and gpu in cmd[c:d] and gpu in cmd[d:e] and gpu in cmd[e:f] and gpu in cmd[f:g] and finish in cmd[g:]):raise ValueError('stage barrier ordering')
    for token in ('buffer_host_address','memcpy','memset','create_buffer','create_memory','wait_on_submit','collect_garbage'):
        if token in cmd:raise ValueError('intermediate forbidden operation '+token)
    if live.count('d.wait_idle()')!=1:raise ValueError('extra wait')
    for token in ('rec_param[eye]=alloc(sizeof(p),&p)', 'sel_param[eye]=alloc(sizeof(sp),&sp)',
                  'uint32_t(eye),uint32_t(max_words),numerator,cx,SRC_LO,SRC_HI', 'validate_regions(regions)'):
        if token not in live:raise ValueError('resources '+token)
    for prefix in ('rec','reset','select','lr','metric'):
        if f'.byte_code_size=uint32_t({prefix}_words.size())' not in live:raise ValueError('SPIR-V word count')
    tail=live[live.index('d.wait_idle();'):]
    if not tail.index('d.wait_idle();')<tail.index('d.buffer_host_address(status).value()')<tail.index('require(failure==0,"aggregate outside score contract")')<tail.index('std::memcmp')<tail.index('owner.release();context.close()'):raise ValueError('status/teardown order')
    if 'uint32_t zero=0;auto status=alloc(sizeof(zero),&zero)' not in live or 'SRC_LO,SRC_HI,d.device_address(status).value()' not in live:raise ValueError('status initialization/ownership')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in owner:raise ValueError('owner cleanup')


def test_sources():guard_shader(SHADER.read_text());guard_host(APP.read_text(),OWNER.read_text())


@pytest.mark.parametrize('old,new',[
 ('((uint*)p.sums)[i]=INF','((uint*)p.sums)[i]=0'),('((Raw*)p.raw)[p.eye*n+i]=r','((Raw*)p.raw)[i]=r'),
 ('o.source_low=INF;o.source_high=INF','o.source_low=0;o.source_high=0'),
 ('o.source_low=p.source_low;o.source_high=p.source_high','o.source_low=p.source_low;o.source_high=0'),
 ('o.depth=NAN_BITS','o.depth=0'),('if(t>=2*n)return;',''),('sums[offsets[i]+d-lo[i]]','sums[offsets[i]+d]'),
])
def test_shader_mutations(old,new):
    source=SHADER.read_text();assert old in source
    with pytest.raises(ValueError):guard_shader(source.replace(old,new,1))


@pytest.mark.parametrize('old,new',[
 ('if(eye)recorder.pipeline_barrier','if(false)recorder.pipeline_barrier'),
 ('rec_param[eye]=alloc(sizeof(p),&p)','rec_param[0]=alloc(sizeof(p),&p)'),
 ('sel_param[eye]=alloc(sizeof(sp),&sp)','sel_param[0]=alloc(sizeof(sp),&sp)'),
 ('recorder.set_pipeline(selection)','d.wait_idle();recorder.set_pipeline(selection)'),
 ('recorder.set_pipeline(lr)','d.buffer_host_address(raw);recorder.set_pipeline(lr)'),
 ('validate_regions(regions)',''),('.byte_code_size=uint32_t(metric_words.size())','.byte_code_size=uint32_t(metric_words.size()*4)'),
 ('owner.release();context.close()','context.close();owner.release()'),
])
def test_host_mutations(old,new):
    source=APP.read_text();assert old in source
    start=source.index('static void gpu_selection(')
    with pytest.raises((ValueError,IndexError)):guard_host(source[:start]+source[start:].replace(old,new,1),OWNER.read_text())


@pytest.mark.parametrize('start,end,old',[
 ('if(eye)recorder.pipeline_barrier','recorder.set_pipeline(reset)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(reset)','for(uint32_t pass=0;pass<4;++pass)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(selection)','recorder.set_pipeline(lr)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
 ('recorder.set_pipeline(lr)','recorder.set_pipeline(metric)', '.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE'),
])
def test_barrier_mutations(start,end,old):
    source=APP.read_text();a=source.index(start,source.index('static void gpu_selection('));b=source.index(end,a);assert old in source[a:b]
    with pytest.raises(ValueError):guard_host(source[:a]+source[a:b].replace(old,'.src_access=daxa::AccessConsts::HOST_WRITE',1)+source[b:],OWNER.read_text())


def test_fake_actual_source(tmp_path):
    actual=block(APP.read_text())
    cpp=r'''#include <cstdint>
#include <chrono>
#include <cstdio>
#include <algorithm>
namespace daxa{struct AccessConsts{static constexpr int HOST_WRITE=1,COMPUTE_SHADER_READ=2,COMPUTE_SHADER_WRITE=4,HOST_READ=8;};struct Barrier{int src_access,dst_access;};struct Info{const char*name;};struct Dispatch{uint32_t x,y,z;};struct Addr{uint64_t value()const{return 7;}};struct Cmd{};struct Submit{struct List{Cmd*p;int n;}command_lists;};struct Recorder{static inline uint64_t barriers=0,dispatches=0;void pipeline_barrier(Barrier){++barriers;}void set_pipeline(int){}template<class T>void push_constant(T){}void dispatch(Dispatch){++dispatches;}Cmd complete_current_commands(){return {};}};struct Device{static inline uint64_t submits=0,waits=0;Recorder create_command_recorder(Info){return {};}Addr device_address(int){return {};}void submit_commands(Submit){++submits;}void wait_idle(){++waits;}};}
struct Push{uint64_t p;uint32_t pass;};struct SelectPush{uint64_t p;};struct F{uint32_t w=320,h=180;};
__attribute__((noinline))static void run(daxa::Device& d){asm volatile("" ::: "memory");F f[2];uint64_t max_words=100,n=320*180;int reset=0,recurrence=0,selection=0,lr=0,metric=0,sel_param[2]={1,2},rec_param[2]={3,4};
''' + actual + r'''
}
int main(){daxa::Device d;constexpr unsigned n=100000;auto t=std::chrono::steady_clock::now();for(unsigned i=0;i<n;++i)run(d);auto e=std::chrono::steady_clock::now();if(daxa::Recorder::barriers!=n*15||daxa::Recorder::dispatches!=n*14||daxa::Device::submits!=n||daxa::Device::waits!=n)return 1;std::printf("fake actual-source %.2f ns; 14 dispatches 15 barriers 1 submit 1 fake diagnostic wait\n",std::chrono::duration<double,std::nano>(e-t).count()/n);}
'''
    source=tmp_path/'fake.cpp';exe=tmp_path/'fake';source.write_text(cpp)
    subprocess.run(['c++','-std=c++20','-O2',str(source),'-o',str(exe)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True);print(result.stdout.strip())
    assert '14 dispatches 15 barriers 1 submit 1 fake diagnostic wait' in result.stdout


def guard_witness(source):
    part=source[source.index('static void gpu_witnesses()'):source.index('static void gpu_selection(')]
    start=part.index('auto recorder=d.create_command_recorder(')
    stop=part.index('d.wait_idle();',start)+len('d.wait_idle();')
    cmd=part[start:stop]
    gpu='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    host='.src_access=daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'
    finish='.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ'
    a=cmd.index('recorder.set_pipeline(selection)');b=cmd.index('recorder.set_pipeline(lr)');c=cmd.index('recorder.set_pipeline(metric)')
    if not(host in cmd[:a] and gpu in cmd[a:b] and gpu in cmd[b:c] and finish in cmd[c:]):raise ValueError('witness barriers')
    for token in ('for(size_t eye=0;eye<2;++eye)', 'd.device_address(param[eye]).value()', 'd.device_address(param[0]).value()', 'd.submit_commands','d.wait_idle()'):
        if token not in cmd:raise ValueError('witness command')
    for token in ('buffer_host_address','memcpy','memset','create_buffer','create_memory','wait_on_submit','collect_garbage','std::ofstream'):
        if token in cmd:raise ValueError('witness intermediate host operation')
    for word in ('select','lr','metric'):
        if f'.byte_code_size=uint32_t({word}_words.size())' not in part:raise ValueError('witness word count')
    tail=part[stop:]
    if not(tail.index('d.buffer_host_address(status).value()')<tail.index('require(failure==0,"witness aggregate outside score contract")')<tail.index('std::memcmp')<tail.index('SF_FP2_WITNESS_DIR')<tail.index('std::ofstream')<tail.index('witness raw mismatch')<tail.index('witness observation mismatch')<tail.index('owner.release();context.close()')):raise ValueError('witness diagnostic/status ordering')
    if 'uint32_t zero=0;auto status=alloc(sizeof(zero),&zero)' not in part or 'SRC_LO,SRC_HI,d.device_address(status).value()' not in part:raise ValueError('witness status initialization/ownership')
    return cmd


def test_witness_actual_source():guard_witness(APP.read_text())


@pytest.mark.parametrize('old,new',[
    ('d.device_address(param[eye]).value()','d.device_address(param[0]).value()'),
    ('recorder.set_pipeline(lr)','d.wait_idle();recorder.set_pipeline(lr)'),
    ('.byte_code_size=uint32_t(select_words.size())','.byte_code_size=uint32_t(select_words.size()*4)'),
    ('owner.release();context.close()','context.close();owner.release()'),
])
def test_witness_mutations(old,new):
    s=APP.read_text();a=s.index('static void gpu_witnesses()');b=s.index('static void gpu_selection(');assert old in s[a:b]
    with pytest.raises((ValueError,IndexError)):guard_witness(s[:a]+s[a:b].replace(old,new,1)+s[b:])


def test_witness_fake_actual_source(tmp_path):
    actual=guard_witness(APP.read_text())
    cpp=r'''#include <cstdint>
#include <cstddef>
#include <chrono>
#include <cstdio>
namespace daxa{struct AccessConsts{static constexpr int HOST_WRITE=1,COMPUTE_SHADER_READ=2,COMPUTE_SHADER_WRITE=4,HOST_READ=8;};struct Barrier{int src_access,dst_access;};struct Info{const char*name;};struct Dispatch{uint32_t x,y,z;};struct Addr{uint64_t value()const{return 7;}};struct Cmd{};struct Submit{struct List{Cmd*p;int n;}command_lists;};struct Recorder{static inline uint64_t barriers=0,dispatches=0;void pipeline_barrier(Barrier){++barriers;}void set_pipeline(int){}template<class T>void push_constant(T){}void dispatch(Dispatch){++dispatches;}Cmd complete_current_commands(){return {};}};struct Device{static inline uint64_t submits=0,waits=0;Recorder create_command_recorder(Info){return {};}Addr device_address(int){return {};}void submit_commands(Submit){++submits;}void wait_idle(){++waits;}};}
struct SelectPush{uint64_t p;};
__attribute__((noinline))static void run(daxa::Device& d){asm volatile("" ::: "memory");int selection=0,lr=0,metric=0,param[2]={1,2};
''' + actual + r'''
}
int main(){daxa::Device d;constexpr unsigned n=100000;auto t=std::chrono::steady_clock::now();for(unsigned i=0;i<n;++i)run(d);auto e=std::chrono::steady_clock::now();if(daxa::Recorder::barriers!=n*4||daxa::Recorder::dispatches!=n*4||daxa::Device::submits!=n||daxa::Device::waits!=n)return 1;std::printf("witness fake %.2f ns; 4 dispatches 4 barriers 1 submit 1 fake diagnostic wait\n",std::chrono::duration<double,std::nano>(e-t).count()/n);}
'''
    path=tmp_path/'fake-witness.cpp';exe=tmp_path/'fake-witness';path.write_text(cpp)
    subprocess.run(['c++','-std=c++20','-O2',str(path),'-o',str(exe)],check=True,capture_output=True,text=True)
    r=subprocess.run([str(exe)],check=True,capture_output=True,text=True);print(r.stdout.strip())
    assert '4 dispatches 4 barriers 1 submit 1 fake diagnostic wait' in r.stdout


def test_witness_debug_io_cannot_move_before_wait():
    source=APP.read_text(); a=source.index('static void gpu_witnesses()'); b=source.index('static void gpu_selection(')
    old='recorder.set_pipeline(lr)'; assert old in source[a:b]
    changed=source[:a]+source[a:b].replace(old,'std::ofstream debug;'+old,1)+source[b:]
    with pytest.raises(ValueError):guard_witness(changed)


def test_shader_score_bound_mutation():
    s=SHADER.read_text();old='if(v>224)';assert old in s
    with pytest.raises(ValueError):guard_shader(s.replace(old,'if(v>225)',1))


@pytest.mark.parametrize('start,end', [('static void gpu_witnesses()', 'static void gpu_selection('), ('static void gpu_selection(', 'int main(')])
@pytest.mark.parametrize('old,new', [('uint32_t zero=0;auto status=alloc(sizeof(zero),&zero)','uint32_t zero=1;auto status=alloc(sizeof(zero),&zero)'), ('require(failure==0,','require(true,')])
def test_status_mutations(start,end,old,new):
    s=APP.read_text();a=s.index(start);b=s.index(end,a);assert old in s[a:b]
    changed=s[:a]+s[a:b].replace(old,new,1)+s[b:]
    with pytest.raises((ValueError,IndexError)):
        (guard_witness(changed) if start=='static void gpu_witnesses()' else guard_host(changed,OWNER.read_text()))


def guard_overflow(source):
    for token in ('if(inject_witness_overflow)w.sums[0].sums[w.f[0].offsets[2]]=225', 'mode=="--gpu-overflow"', 'inject_witness_overflow=true;gpu_witnesses()'):
        if token not in source:raise ValueError('overflow witness provenance')


def test_overflow_witness_source():guard_overflow(APP.read_text())


def test_overflow_witness_mutation():
    source=APP.read_text();old='offsets[2]]=225';assert old in source
    with pytest.raises(ValueError):guard_overflow(source.replace(old,'offsets[2]]=224',1))


def guard_observation_dump(source):
    a=source.index('static void gpu_witnesses()');b=source.index('static void gpu_selection(',a)
    part=source[a:b];wait=part.index('d.wait_idle();');status=part.index('require(failure==0,"witness aggregate outside score contract")')
    mismatch=part.index('if(std::memcmp(d.buffer_host_address(obs).value()')
    env=part.index('SF_FP2_WITNESS_DIR',mismatch);out=part.index('std::ofstream',env);error=part.index('witness observation mismatch',out)
    if not wait<status<mismatch<env<out<error:raise ValueError('observation debug ordering')
    if 'gpu-obs.bin' not in part[out:error] or 'cpu-obs.bin' not in part[out:error] or 'require(bool(a)&&bool(b),"witness observation dump failed")' not in part[out:error]:raise ValueError('observation dump checks')


def test_observation_dump_source():guard_observation_dump(APP.read_text())


def test_observation_dump_status_bypass_mutation():
    source=APP.read_text();a=source.index('static void gpu_witnesses()');b=source.index('static void gpu_selection(',a)
    old='require(failure==0,"witness aggregate outside score contract")';assert old in source[a:b]
    with pytest.raises((ValueError,IndexError)):guard_observation_dump(source[:a]+source[a:b].replace(old,'require(true,"ignored")',1)+source[b:])


LUT=ROOT/'apps/fp2_metric_lut.hpp'

def guard_lut(host, header):
    for start,end,args in [('static void gpu_witnesses()', 'static void gpu_selection(', '63.125f,.25f'), ('static void gpu_selection(', 'int main(', 'numerator,cx')]:
        part=host[host.index(start):host.index(end,host.index(start))]
        make=f'make_metric_lut({args},reinterpret_cast<uintptr_t>(d.get()))'
        validate=f'validate_metric_lut(lut,{args},reinterpret_cast<uintptr_t>(d.get()))'
        upload='alloc(sizeof(lut.entries),lut.entries.data())'
        if part.count(make)!=1 or part.count(validate)!=1 or part.count(upload)!=1:raise ValueError('LUT setup/validation/upload')
        if not part.index(make)<part.index(validate)<part.index(upload)<part.index('auto recorder=d.create_command_recorder('):raise ValueError('LUT setup order')
        if 'd.device_address(lut_buffer).value()' not in part:raise ValueError('LUT address missing')
        if part.index('d.wait_idle();')>part.index('owner.release();context.close()'):raise ValueError('LUT lifetime')
    for token in ('table.numerator_bits!=expected.numerator_bits','table.offset_bits!=expected.offset_bits','table.device_identity!=expected.device_identity','table.depth_convention!=expected.depth_convention','table.entries[i].depth!=expected.entries[i].depth','table.entries[i].valid!=expected.entries[i].valid','std::fegetround()!=FE_TONEAREST'):
        if token not in header:raise ValueError('LUT identity/content check')


def test_lut_current():guard_lut(APP.read_text(),LUT.read_text())


@pytest.mark.parametrize('old,new',[
    ('table.numerator_bits!=expected.numerator_bits','false'),
    ('table.offset_bits!=expected.offset_bits','false'),
    ('table.device_identity!=expected.device_identity','false'),
    ('table.entries[i].depth!=expected.entries[i].depth','false'),
])
def test_lut_identity_mutations(old,new):
    h=LUT.read_text();assert old in h
    with pytest.raises(ValueError):guard_lut(APP.read_text(),h.replace(old,new,1))


def test_lut_host_immutable_upload_mutation():
    s=APP.read_text();a=s.index('static void gpu_selection(');old='alloc(sizeof(lut.entries),lut.entries.data())';assert old in s[a:]
    with pytest.raises(ValueError):guard_lut(s[:a]+s[a:].replace(old,'alloc(sizeof(lut.entries),nullptr)',1),LUT.read_text())


@pytest.mark.parametrize('old,new',[
    ('a.winner<=0||a.winner>=int(p.disparities)','a.winner<=0'),
    ('InterlockedOr(((uint*)p.status)[0],2,old)','old=0'),
    ('uint2 entry=((uint2*)p.metric_lut)[uint(a.winner)]','uint2 entry=((uint2*)p.metric_lut)[0]'),
])
def test_lut_shader_mutations(old,new):
    s=SHADER.read_text();assert old in s
    with pytest.raises(ValueError):guard_shader(s.replace(old,new,1))
