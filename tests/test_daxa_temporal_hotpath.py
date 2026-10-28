"""CPU-only executing production Temporal bookkeeping; no GPU performance claim."""
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import pytest
from test_daxa_depth_hotpath import definition
from test_kernel_variant_hotpath import body
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/daxa_temporal.cpp'


def check_hotpath(source):
    scopes=('DaxaTemporal::Frame DaxaTemporal::process(', 'size_t DaxaTemporal::collect(',
            'bool DaxaTemporal::can_process(', 'size_t collect_at(', 'void DaxaTemporal::reset(',
            'detail::TemporalSequence::Admission detail::TemporalSequence::prepare(',
            '.executes([this,raw,address,pass,count]')
    hot='\n'.join(body(source,s) for s in scopes)
    assert not re.search(r'\b(?:new|make_shared|make_unique|malloc|calloc|realloc|resize|reserve|push_back|emplace_back|wait\w*|readback\w*|create_\w+|destroy_\w+|register_buffer|add_task|memcpy|memmove|sha256|strict_json)\s*(?:<[^;()]*>)?\s*\(|\b(?:TaskGraph|vector|ifstream|ofstream)\b',hot), 'costly operation in actual temporal hotpath'
    assert not re.search(r'graph(?:->|\.)complete\s*\(',hot), 'graph rebuilt in hotpath'
    collection=body(source,'size_t DaxaTemporal::collect(')
    assert collection.count('collect_garbage(')==1 and collection.count('oldest_pending_submit_index(')==1
    process=body(source,scopes[0]);
    assert 'collect_garbage(' not in process
    assert process.count('oldest_pending_submit_index(')==1
    assert process.index('e.sequence.prepare(')<process.index('oldest_pending_submit_index(')<process.index('graph->execute(')<process.index('e.sequence.commit(')
    assert 's==e.history||s.use_count()!=1||!e.complete(*s,oldest)' in process
    allowed={'unsigned eye=0;eye<2;++eye','unsigned j=0;j<16;++j','auto& s:e.slots','unsigned p=0;p<6;++p','unsigned p=0;p<2;++p','unsigned p=0;p<11;++p'}
    assert set(re.findall(r'for\(([^)]*)\)',process))==allowed and 'while(' not in process, 'only fixed matrix/descriptor loops and bounded slot admission permitted'
    assert process.count('e.history->last_read=std::max(e.history->last_read,s->last_write)')==2
    assert 'catch(...){s->last_write=d.latest_queue_submit_index(' in process
    retirement=body(source,'size_t collect_at(')
    assert 's.use_count()==internal(s.get())&&complete(*s,oldest)' in retirement and 's->input={}' in retirement
    assert 'HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE' in source
    assert 'COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE' in source
    assert 'graph.submit({});graph.complete({});' in source
    assert 'std::array<daxa::ExternalTaskBuffer,6> fresh_nodes' in source
    assert 'std::array<daxa::ExternalTaskBuffer,11> history_nodes,output_nodes' in source
    assert 'std::array<daxa::ExternalTaskBuffer,2> support_nodes' in source
    assert 'std::array<daxa::ExternalTaskBuffer,4> scratch_nodes' in source
    adapter=body(source,'DaxaObservationFrame DaxaTemporal::Frame::observations(')
    assert 'slot_->input.observations()' in adapter and 'e.history' not in adapter
    assert 'std::copy_n(slot_->outputs.begin(),6,planes.begin())' in adapter
    assert 'DaxaDepthPipeline::Frame input' in body(source,'struct DaxaTemporal::Slot')
    state=(ROOT/'include/stereoforge/detail/daxa_temporal_state.hpp').read_text()
    assert 'device.reserve(65536);clock.reserve(65536)' in state
    assert 'temporal_tickets_complete' in state and 'active_history?2:1' in state


def test_temporal_static_hotpath_and_mutations():
    source=SOURCE.read_text();check_hotpath(source)
    marker='DaxaTemporal::Frame DaxaTemporal::process(DaxaDepthPipeline::Frame const& input,std::optional<DaxaTimedRigPose> const& pose){'
    for mutation in ('input.wait();','readback(input);','std::make_shared<int>();','std::vector<float> pixels(n);','graph.complete({});','for(unsigned pixel=0;pixel<fresh.width()*fresh.height();++pixel){}'):
        with pytest.raises(AssertionError):check_hotpath(source.replace(marker,marker+mutation,1))
    callback='.executes([this,raw,address,pass,count](daxa::TaskInterface ti){'
    assert callback in source
    for operation in ('ti.recorder.wait_idle();','std::make_shared<int>();'):
        with pytest.raises(AssertionError):check_hotpath(source.replace(callback,callback+operation,1))
    for old,new in [('s.use_count()!=1','false'),('e.history->last_read=std::max(e.history->last_read,s->last_write)',''),('s->input={}','')]:
        with pytest.raises(AssertionError):check_hotpath(source.replace(old,new))


def build_probe(tmp_path, mutation=None):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ compiler unavailable')
    source=SOURCE.read_text()
    if mutation:source=mutation(source)
    program=r'''
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
std::atomic<bool> track=false;unsigned allocations=0;
void* operator new(size_t n){if(track)++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}void operator delete(void* p,size_t)noexcept{std::free(p);}
namespace daxa {using BufferId=uint64_t;constexpr int QUEUE_MAIN=0;struct ExternalTaskBuffer{uint64_t id=0;unsigned updates=0;void set_buffer(uint64_t v){id=v;++updates;}};}
namespace stereoforge {
struct CaptureFrame{uint64_t frame_id=0,host_ingress_timestamp_ns=0;std::string device_identity,host_clock_domain;};using CaptureLease=std::shared_ptr<const CaptureFrame>;
struct Device{uint64_t latest=0,oldest=0,polls=0,waits=0,gc=0;void collect_garbage(){++gc;}void* get(){return this;}uint64_t oldest_pending_submit_index(){++polls;return oldest;}uint64_t latest_queue_submit_index(int){return latest;}struct Address{uint64_t id;uint64_t value(){return id;}};Address device_address(uint64_t id){return {id};}struct Host{uint64_t id;void* value(){return reinterpret_cast<void*>(id);}};Host buffer_host_address(uint64_t id){return {id};}};
struct DaxaCameraModel{double fx=1,fy=1,cx=0,cy=0;};
struct DaxaFoundation{Device device;};struct DaxaRectification{unsigned width=7,height=5;};
struct DaxaTemporalConfig{unsigned slots=2,max_contributors=4,max_reuses=4;int64_t max_age_ns=200000000;float relative_depth_tolerance=.03f,confidence_floor=.05f;void validate()const;};
struct DaxaTimedRigPose{std::array<double,16> world_from_rig{};int64_t timestamp_ns=0;std::string_view clock_domain;void validate(CaptureFrame const&)const;};
namespace detail{
SHARED_COMPLETION_HELPERS
}
enum class DaxaObservationPlane{Depth,Confidence,Validity,Provenance,SourceIds,AgeNs};enum class DaxaPreprocessPlane{LeftSupport,RightSupport};
struct Input{std::shared_ptr<DaxaFoundation> context;std::shared_ptr<const DaxaRectification> maps;CaptureLease source;uint64_t marker=10;};
struct Observation{std::shared_ptr<Input> input;auto context()const{return input->context;}auto map_owner()const{return input->maps;}unsigned width()const{return input->maps->width;}unsigned height()const{return input->maps->height;}uint64_t buffer(DaxaObservationPlane p)const{return input->marker+unsigned(p);}};
class DaxaDepthPipeline{public:struct Frame{std::shared_ptr<Input> input;explicit operator bool()const{return bool(input);}Observation observations()const{return {input};}CaptureLease source()const{return input->source;}struct Image{uint64_t id;auto buffer()const{return id;}};Image image(DaxaPreprocessPlane p)const{return {input->marker+20+unsigned(p)};}};};
}
'''
    state=(ROOT/'include/stereoforge/detail/daxa_temporal_state.hpp').read_text()
    program+='\n'.join(line for line in state.splitlines() if not line.startswith(('#include','#pragma')))
    program+='\nnamespace stereoforge {\nusing detail::temporal_require;using detail::TemporalMatrix;using detail::temporal_inverse;using detail::temporal_multiply;\n'
    program+=definition(source,'void identity(')+'\n'+definition(source,'struct Parameters')+';\n'+definition(source,'std::array<uint32_t,2> split(')
    program+=r'''
struct Graph{Device* device=nullptr;unsigned executions=0;int fail=0;void execute(std::initializer_list<int>){++executions;if(fail==1)throw std::runtime_error("before submit");++device->latest;if(fail==2)throw std::runtime_error("partial submit");}};
class DaxaTemporal{public:struct Slot{std::array<daxa::BufferId,11> outputs{};daxa::BufferId parameters{};std::array<daxa::ExternalTaskBuffer,6> fresh_nodes;std::array<daxa::ExternalTaskBuffer,11> history_nodes,output_nodes;std::array<daxa::ExternalTaskBuffer,2> support_nodes;std::array<daxa::ExternalTaskBuffer,4> scratch_nodes;std::unique_ptr<Graph> graph;DaxaDepthPipeline::Frame input;uint64_t generation=0,last_write=0,last_read=0;};
struct Impl{std::shared_ptr<DaxaFoundation> context;std::shared_ptr<const DaxaRectification> maps;DaxaTemporalConfig config;std::vector<std::shared_ptr<Slot>> slots;std::shared_ptr<Slot> history;detail::TemporalSequence sequence;std::array<TemporalMatrix,2> eye_to_rig,rig_to_eye;uint64_t next_generation=1,retired=0;void require_thread()const{}bool complete(Slot const&,uint64_t)const;long internal(Slot const*)const;size_t collect_at(uint64_t);[[noreturn]] void fail_backpressure(uint64_t)const;};
std::shared_ptr<Impl> impl_;class Frame{std::shared_ptr<Impl> owner_;std::shared_ptr<Slot> slot_;uint64_t generation_=0;friend class DaxaTemporal;Frame(std::shared_ptr<Impl>,std::shared_ptr<Slot>);void validate()const;public:Frame()=default;Frame(Frame const&)=default;Frame(Frame&&)noexcept=default;Frame& operator=(Frame)noexcept;uint64_t generation()const;explicit operator bool()const{return bool(owner_&&slot_);}};
Frame process(DaxaDepthPipeline::Frame const&,std::optional<DaxaTimedRigPose> const&);size_t collect();bool can_process()const;void reset();};
'''
    for signature,prefix in [('bool complete(','bool DaxaTemporal::Impl::complete('),('long internal(','long DaxaTemporal::Impl::internal('),('size_t collect_at(','size_t DaxaTemporal::Impl::collect_at('),('[[noreturn]] void fail_backpressure(','[[noreturn]] void DaxaTemporal::Impl::fail_backpressure(')]:
        program+=definition(source,signature).replace(signature,prefix,1)
    for signature in ('void DaxaTemporalConfig::validate(', 'void DaxaTimedRigPose::validate(', 'detail::TemporalSequence::Admission detail::TemporalSequence::prepare(', 'DaxaTemporal::Frame::Frame(', 'DaxaTemporal::Frame& DaxaTemporal::Frame::operator=(', 'void DaxaTemporal::Frame::validate(', 'uint64_t DaxaTemporal::Frame::generation(', 'DaxaTemporal::Frame DaxaTemporal::process(', 'size_t DaxaTemporal::collect(', 'bool DaxaTemporal::can_process(', 'void DaxaTemporal::reset('):program+=definition(source,signature)
    program+=r'''
}
using namespace stereoforge;
void check(bool v,char const* s){if(!v)throw std::runtime_error(s);}
int main(int argc,char** argv){try{size_t identity_bytes=argc>1?std::stoul(argv[1]):65536;
 auto context=std::make_shared<DaxaFoundation>();auto engine=std::make_shared<DaxaTemporal::Impl>();engine->context=context;engine->maps=std::make_shared<DaxaRectification>();DaxaTemporal stage;stage.impl_=engine;
 std::array<Parameters,2> params{};for(unsigned i=0;i<2;++i){auto slot=std::make_shared<DaxaTemporal::Slot>();for(unsigned p=0;p<11;++p)slot->outputs[p]=100+i*100+p;slot->parameters=reinterpret_cast<uintptr_t>(&params[i]);slot->graph=std::make_unique<Graph>(Graph{&context->device});engine->slots.push_back(std::move(slot));}
 std::array<std::shared_ptr<CaptureFrame>,3> captures;std::array<DaxaDepthPipeline::Frame,3> inputs;
 for(unsigned i=0;i<3;++i){captures[i]=std::make_shared<CaptureFrame>();captures[i]->device_identity=std::string(identity_bytes,'d');captures[i]->host_clock_domain=std::string(identity_bytes,'c');auto input=std::make_shared<Input>();input->context=context;input->maps=engine->maps;input->source=captures[i];input->marker=1000+100*i;inputs[i].input=input;}
 TemporalMatrix identity{};for(unsigned i=0;i<4;++i)identity[5*i]=1;for(unsigned eye=0;eye<2;++eye)engine->eye_to_rig[eye]=engine->rig_to_eye[eye]=identity;
 uint64_t id=0;auto pose=[&](unsigned i){captures[i]->frame_id=++id;captures[i]->host_ingress_timestamp_ns=id*1000;return DaxaTimedRigPose{identity,int64_t(id*1000),captures[i]->host_clock_domain};};auto finish=[&]{context->device.oldest=context->device.latest;};
 auto first=stage.process(inputs[0],pose(0));check(first.generation()==1,"generation association");++engine->slots[0]->generation;bool stale=false;try{first.generation();}catch(std::invalid_argument const&){stale=true;}--engine->slots[0]->generation;check(stale,"stale frame generation accepted");finish();auto second=stage.process(inputs[1],pose(1));finish();check(!stage.can_process(),"retained old consumer failed backpressure");check(engine->slots[0]->last_read==2,"history read ticket missing");check(stage.collect()==0,"retained capture retired");first={};check(stage.can_process(),"released old slot unavailable");check(stage.collect()==1,"completed nonhistory capture not retired");second={};check(stage.collect()==1&&!engine->history->input,"active history retained upstream capture");
 // Pinned Daxa reports UINT64_MAX when idle, but the latest COMPLETED
 // semaphore value while an upstream Depth submit is pending. Equality is complete.
 context->device.oldest=UINT64_MAX;check(stage.can_process(),"idle admission failed");
 auto completed_temporal=context->device.latest;++context->device.latest; // next Depth submit
 context->device.oldest=completed_temporal;
 check(stage.can_process(),"upstream submission incorrectly made completed Temporal read unavailable");
 auto equality=stage.process(inputs[0],pose(0));equality={};
 context->device.oldest=context->device.latest-1;
 check(!stage.can_process()&&stage.collect()==0,"genuinely pending Temporal read/write reused");
 finish();check(stage.collect()==1,"completed equality-regression source not retired");
 // A partial submit must retain both candidate input and previous history reads.
 auto* accepted=engine->history.get();auto accepted_id=engine->sequence.id;auto accepted_pose=engine->sequence.previous;auto* candidate=engine->slots[0].get()==accepted?engine->slots[1].get():engine->slots[0].get();candidate->graph->fail=2;bool failed=false;try{auto out=stage.process(inputs[2],pose(2));}catch(std::runtime_error const&){failed=true;}
 check(failed&&engine->history.get()==accepted&&engine->sequence.id==accepted_id&&engine->sequence.previous==accepted_pose,"failed submit promoted history/pose");check(candidate->last_write==context->device.latest&&accepted->last_read==candidate->last_write&&bool(candidate->input),"partial tickets/input not pinned");check(stage.collect()==0&&!stage.can_process(),"pending partial submit reused");finish();candidate->graph->fail=0;candidate=nullptr;accepted=nullptr;check(stage.collect()==1,"partial input not retired after completion");
 stage.reset();stage.collect();auto snapshots=context->device.polls;auto gc_start=context->device.gc;auto executions=context->device.latest;constexpr unsigned frames=10000;allocations=0;track=true;auto start=std::chrono::steady_clock::now();
 for(unsigned n=0;n<frames;++n){unsigned i=n%3;check(stage.can_process(),"warm admission failed");auto p=pose(i);if(n%17==0)captures[i]->device_identity[0]=(captures[i]->device_identity[0]=='d'?'e':'d');auto out=stage.process(inputs[i],p);finish();out={};stage.collect();}
 double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();track=false;
 check(allocations==0&&context->device.waits==0,"allocation or wait in warmed actual source");check(context->device.gc-gc_start==frames,"one GC required per explicit collect and none in process/admission");check(context->device.polls-snapshots==3*frames,"unexpected admission/process/collect snapshots");check(context->device.latest-executions==frames,"graph not reused");
 for(auto const& slot:engine->slots)check(!slot->input,"current RGB capture retained after completion");for(auto const& in:inputs)check(in.input.use_count()==1,"deep producer chain retained");check(engine->slots.size()==2,"slot registry grew");check(engine->sequence.device.capacity()>=65536&&engine->sequence.clock.capacity()>=65536,"epoch storage not reserved");
 // A read-only in-flight use protects storage even after reset.
 stage.reset();context->device.latest+=5;for(auto& slot:engine->slots)slot->last_read=context->device.latest;check(!stage.can_process(),"pending read ticket ignored after reset");context->device.oldest=context->device.latest;check(stage.can_process(),"read-complete slots not reusable");
 auto held=stage.process(inputs[0],pose(0));stage.reset();finish();check(stage.collect()==0,"reset retired external current RGB");held={};check(stage.collect()==1,"reset did not release completed current RGB");
 auto keep=stage.process(inputs[1],pose(1));auto weak=std::weak_ptr(engine);stage.impl_.reset();engine.reset();check(!weak.expired(),"Frame did not retain facade");keep={};check(weak.expired(),"Frame/Impl ownership cycle");
 std::cout<<"{\"frames\":"<<frames<<",\"identity_bytes\":"<<identity_bytes<<",\"application_allocations\":"<<allocations<<",\"implicit_waits\":0,\"snapshots\":"<<3*frames<<",\"explicit_collect_gc_calls\":"<<frames<<",\"mean_mock_host_ms\":"<<ms/frames<<",\"slots\":2,\"binding_nodes_per_slot\":34,\"partial_submit_pins_input_and_history\":true,\"current_RGB_retired\":true,\"scope\":\"actual_production_process_sequence_collect_admission_mock_GPU_and_producer\"}\n";
 }catch(std::exception const& e){track=false;std::cerr<<e.what()<<'\n';return 1;}}
'''
    frame_source=(ROOT/'include/stereoforge/daxa_frame.hpp').read_text()
    shared=definition(frame_source,'inline bool completed_submission(')+definition(frame_source,'inline bool available_slot(')
    program=program.replace('SHARED_COMPLETION_HELPERS',shared)
    code=tmp_path/'probe.cpp';code.write_text(program);binary=tmp_path/'probe'
    built=subprocess.run([compiler,'-std=c++20','-O3',str(code),'-o',str(binary)],capture_output=True,text=True,timeout=60)
    assert built.returncode==0,built.stderr
    return binary


@pytest.mark.parametrize('identity_bytes',[24,65536])
def test_actual_temporal_process_admission_retirement_probe(tmp_path,identity_bytes):
    result=subprocess.run([str(build_probe(tmp_path)),str(identity_bytes)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    row=json.loads(result.stdout);print('Actual Temporal CPU-only probe:',json.dumps(row,sort_keys=True))
    assert row['application_allocations']==row['implicit_waits']==0 and row['snapshots']==30000
    assert row['explicit_collect_gc_calls']==10000
    assert row['partial_submit_pins_input_and_history'] and row['current_RGB_retired']


@pytest.mark.parametrize('mutation', ['allocation','retirement','read_ticket'])
def test_executing_probe_detects_hotpath_regressions(tmp_path, mutation):
    def alter(source):
        if mutation=='allocation':
            marker='auto& e=*impl_;e.require_thread();auto fresh=input.observations();'
            assert marker in source
            return source.replace(marker,marker+'auto unexpected=std::make_shared<int>(9);',1)
        if mutation=='retirement':
            assert 's->input={}' in source
            return source.replace('s->input={}','')
        assert 'temporal_tickets_complete(s.last_write,s.last_read,oldest)' in source
        return source.replace('temporal_tickets_complete(s.last_write,s.last_read,oldest)','temporal_tickets_complete(s.last_write,0,oldest)')
    result=subprocess.run([str(build_probe(tmp_path,alter))],capture_output=True,text=True,timeout=30)
    assert result.returncode!=0, 'actual executing probe must detect the mutated production body'


def test_fixture_submission_rejections_and_controlled_upload_scope_are_honest():
    source=(ROOT/'apps/daxa_temporal_fixture.cpp').read_text()
    compact=re.sub(r'\s+','',source)
    assert '"performance_qualified",false' in compact and '"measured_stereo_output",false' in compact
    assert '"observations_overridden_for_fixture",true' in compact and '"support_overridden_for_fixture",true' in compact
    assert compact.index('auto const before_depth'.replace(' ','')) < compact.index('input=depth.enqueue(source)')
    assert 'before_depth==context->device.latest_queue_submit_index(daxa::QUEUE_MAIN)' in compact
    assert '"upstream host rejection submitted GPU work"'.replace(' ','') in compact
    assert 'before==context->device.latest_queue_submit_index(daxa::QUEUE_MAIN)' in compact
    upload=body(source,'void upload(')
    assert upload.index('frame.wait()') < upload.index('create_command_recorder(') < upload.index('copy_buffer_to_buffer(')
    assert 'TRANSFER_WRITE' in upload and 'COMPUTE_SHADER_READ' in upload
    assert 'diagnostic_owner_gather_not_GPU_ancestry_gather' in source


def check_benchmark(source):
    loop=body(source,'while(completed<uint64_t(frames)+warmup)')
    assert not re.search(r'\b(?:new|make_shared|make_unique|malloc|resize|reserve|push_back|readback|dump|memcpy|memmove|wait_idle|sha256|digest)\s*(?:<[^;()]*>)?\s*\(',loop)
    assert loop.index('view->collect()') < loop.index('temporal->collect()') < loop.index('depth->collect()') < loop.index('view->can_project()') < loop.index('depth->enqueue(capture)')
    assert loop.index('temporal->can_process()') < loop.index('capture->frame_id=') < loop.index('depth->enqueue(capture)')
    assert 'captures[index].use_count()==1' in loop
    assert loop.index('depth->enqueue(capture)') < loop.index('temporal->process(raw,') < loop.index('view->project(fused.observations())')
    assert loop.index('auto submitted=Clock::now()') < loop.index('projected.wait()') < loop.index('auto complete=Clock::now()') < loop.index('raw.timings()')
    assert source.index('for(auto& c:captures)') < source.index('auto measurement_start=')
    assert source.index('v.resize(frames)') < source.index('auto measurement_start=')
    assert 'depth_bytes+temporal_bytes+view_bytes<=2ull*1024*1024*1024' in source
    assert 'resident requested buffers grew during benchmark' in source
    assert source.index('auto measurement_end=') < source.index('view->readback(projected)')
    assert 'samples_do_not_establish_pipelined_or_live_FPS' in source
    assert 'unique_physical_frames' in source and 'driver_process_peak_memory_bytes' in source
    assert not re.search(r'Fresh|observations_overridden_for_fixture.*true|replace_once',source)


def test_actual_benchmark_admission_and_diagnostic_boundaries():
    source=(ROOT/'apps/daxa_temporal_benchmark.cpp').read_text();check_benchmark(source)
    marker='while(completed<uint64_t(frames)+warmup){'
    for operation in ('raw.wait_idle();','readback(raw);','std::make_shared<int>();'):
        with pytest.raises(AssertionError):check_benchmark(source.replace(marker,marker+operation,1))
    with pytest.raises(AssertionError):check_benchmark(source.replace('captures[index].use_count()==1','true'))



def test_temporal_benchmark_compiled_cold_shader_identities():
    binary=ROOT/'build/daxa/daxa_temporal_benchmark'
    if not binary.exists():pytest.skip('optional native temporal benchmark not built')
    env=dict(os.environ,VK_DRIVER_FILES='/nonexistent-temporal-cold-getter')
    result=subprocess.run([str(binary),'--shader-source-identities'],env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    pairs=dict(view_depth_shader_source_sha256_pair=('daxa/view_depth.slang','ops/projection.slang'),
               depth_shader_source_sha256_pair=('daxa/depth.slang','ops/stereo_observation.slang'),
               preprocess_shader_source_sha256_pair=('daxa/preprocess.slang','ops/preprocess.slang'),
               temporal_shader_source_sha256_pair=('daxa/temporal.slang','ops/temporal.slang'))
    expected={key:':'.join(hashlib.sha256((ROOT/'python/stereoforge/shaders'/name).read_bytes()).hexdigest() for name in names) for key,names in pairs.items()}
    assert json.loads(result.stdout)==expected
    directory=subprocess.run([str(binary),'--shader-directory'],env=env,capture_output=True,text=True,timeout=30)
    assert directory.returncode==0,directory.stderr
    for entry in ('clear','splat_z','splat_owner','gather_fuse'):
        assert (Path(directory.stdout.strip())/('daxa-temporal-'+entry+'.spv')).is_file()
