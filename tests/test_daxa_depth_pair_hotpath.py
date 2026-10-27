"""Independent paired-path source guards and CPU-only bookkeeping probes.

GPU calls are replaced only in extracted-source CPU probes; reported application
cost excludes Daxa implementation, driver work, shader execution and GPU waits.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_paired_per_path_is_an_explicit_schedule_with_sequential_default():
    header = (ROOT / "include/stereoforge/daxa_depth.hpp").read_text()
    assert "PairedPerPath" in header, "explicit paired-per-path schedule is not implemented"
    assert "Sequential" in header
    assert "DaxaDepthStereoSchedule::Sequential" in header, "existing sequential default must remain"

import json
import re
import shutil
import subprocess

import pytest
from test_kernel_variant_hotpath import body
from test_daxa_depth_hotpath import definition


def check_pair_hotpath(stereo):
    select = body(stereo, 'unsigned DaxaStereo::pair_slot_at(')
    process = body(stereo, 'std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at(')
    forbidden = r'\b(?:make_shared|make_unique|allocate|create_\w+|wait_idle|wait_on_submit|readback|inspect|memcpy|malloc|calloc|realloc|resize|reserve|push_back|metric_calibration|sha_file|strict_json)(?:<[^;()]*>)?\s*\('
    assert not re.search(forbidden, select+process), 'paired admission/submission cannot allocate, copy images, wait, read back or scan setup metadata'
    assert 'oldest_pending_submit_index(' not in select+process, 'caller supplies the shared admission snapshot'
    assert select.count('detail::available_slot(') == 2, 'both aligned producer slots must be available'
    assert process.count('detail::available_slot(') == 2
    assert process.index('detail::available_slot(') < process.index('collect_garbage(') < process.index('slot->input=input') < process.index('graph->execute(')
    assert 'input.owner_->context==l.context&&input.calibration()==l.calibration' in process
    assert 'slot->source=input.source()' in process and 'slot->input=input' in process
    assert 'ls->submit=rs->submit=d.latest_queue_submit_index(' in process, 'both producer leases must cover the paired submission'
    assert 'slot->per_reference_timings_available=false' in process
    assert 'return {Frame(impl_,ls),Frame(l.paired->right,rs)}' in process


def test_actual_pair_admission_and_submission_are_allocation_free_and_share_snapshot():
    check_pair_hotpath((ROOT/'src/daxa_stereo.cpp').read_text())


@pytest.mark.parametrize('mutation', ['std::make_shared<int>();', 'd.wait_idle();', 'memcpy(dst,src,n);', 'readback(frame);', 'strict_json(path);'])
def test_pair_guard_rejects_hidden_allocations_waits_copies_and_setup_scans(mutation):
    source=(ROOT/'src/daxa_stereo.cpp').read_text()
    marker='std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at(const DaxaFrame& input,uint64_t oldest,unsigned index){'
    assert marker in source
    with pytest.raises(AssertionError, match='cannot allocate'):
        check_pair_hotpath(source.replace(marker, marker+mutation, 1))


def check_pair_graph(stereo):
    setup = body(stereo, 'void DaxaStereo::setup_pair(')
    assert 'l.context==r.context' in setup and 'l.calibration==r.calibration' in setup
    assert 'l.paired=std::make_unique<Pair>()' in setup and 'pair.right=other.impl_' in setup
    assert 's->queries=d.create_timeline_query_pool(' in setup and 'graph.complete({})' in setup
    assert 'auto& slot=*engine.slots[index]' in setup, 'reuse existing resource buffers and parameters'
    assert not re.search(r'\ballocate\s*\(|create_buffer\s*\(|create_memory\s*\(', setup), 'paired graph adds no image/parameter allocations'
    assert 'task.reads_writes(views[eye][resource])' in setup, 'same-eye path updates retain explicit read/write hazards'
    assert 'path,line_count(a,dx,dy),{LeftDesc,RightDesc,LeftValid,RightValid},{},{Penalties}' in setup
    # The actual inner eye loop must contain dispatch only, never a timestamp or barrier.
    loop = body(setup, 'for(unsigned eye=0;eye<2;++eye){auto& engine=eye?r:l;ti.recorder.set_pipeline(')
    assert 'ti.recorder.dispatch(' in loop
    assert not re.search(r'write_timestamp|pipeline_barrier|reset_timestamps|wait_', loop), 'no inter-eye timestamp/barrier may serialize paired phase'
    start=setup.index('query_index=2+stage*2')
    end=setup.index('query_index=3+stage*2')
    assert start<setup.index(loop)<end, 'one phase timestamp pair must enclose both eye dispatches'
    captures=re.findall(r'task\.executes\(\[([^]]*)\]',setup)
    assert captures and all(not re.search(r'\b(?:input|prototype|pre|left|right)\b',c) for c in captures), 'setup callbacks retain no producer Frames'


def test_graph_setup_reuses_buffers_and_keeps_same_eye_hazards():
    check_pair_graph((ROOT/'src/daxa_stereo.cpp').read_text())


def test_phase_guard_rejects_inter_eye_timestamp():
    source=(ROOT/'src/daxa_stereo.cpp').read_text()
    marker='ti.recorder.dispatch({grid[0],grid[1],1});}'
    assert marker in body(source, 'void DaxaStereo::setup_pair(')
    with pytest.raises(AssertionError, match='inter-eye'):
        check_pair_graph(source.replace(marker, 'ti.recorder.dispatch({grid[0],grid[1],1});ti.recorder.write_timestamp({});}', 1))


def build_pair_bookkeeping_probe(tmp_path_factory, schedule='paired-per-path', paths=4, stereo_transform=None):
    compiler=shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable for extracted CPU-only bookkeeping probe')
    depth=(ROOT/'src/daxa_depth.cpp').read_text()
    pre=(ROOT/'src/daxa_preprocess.cpp').read_text()
    stereo=(ROOT/'src/daxa_stereo.cpp').read_text()
    if stereo_transform is not None:
        stereo = stereo_transform(stereo)
    frame=(ROOT/'include/stereoforge/daxa_frame.hpp').read_text()
    program=r'''
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
std::atomic<bool> tracking=false;std::atomic<unsigned> allocations=0;
void* operator new(std::size_t n){if(tracking)++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);}void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
namespace stereoforge {
void require(bool v,const char* msg){if(!v)throw std::invalid_argument(msg);}
namespace detail {
'''
    program += definition(frame, 'inline bool completed_submission(')+definition(frame, 'inline bool available_slot(')
    program += r'''
}
namespace daxa {using BufferId=uint64_t;constexpr int QUEUE_MAIN=0;}
struct CaptureFrame {uint64_t frame_id=0;std::vector<uint8_t> packed_yuyv;};using CaptureLease=std::shared_ptr<const CaptureFrame>;
struct Maps {std::string rectified_identity="rect";void validate_source(const CaptureFrame& f)const{require(f.packed_yuyv.size()==1280ull*2*720*2,"bad input");}};
struct Parameters {std::array<uint64_t,14> address{};std::array<uint64_t,8> inputs{};uint64_t source_id=0;};
struct Device {uint64_t oldest=1,latest=0,snapshots=0,gc=0,uploads=0;auto oldest_pending_submit_index(){++snapshots;return oldest;}auto latest_queue_submit_index(int){return latest;}void collect_garbage(){++gc;}
 struct Address{uint64_t id;void* value(){return reinterpret_cast<void*>(id);}};Address buffer_host_address(uint64_t id){return {id};}
 struct DeviceAddress{uint64_t id;uint64_t value(){return id;}};DeviceAddress device_address(uint64_t id){return {id};}};
struct Context{Device device;};
struct Graph {Device* device;bool upload=false;unsigned* started=nullptr;unsigned* finished=nullptr;unsigned stages=0;void execute(std::initializer_list<int>){++device->latest;if(upload)++device->uploads;if(started)*started=stages;if(finished)*finished=stages;}};
struct External {uint64_t id=0;void set_buffer(uint64_t value){id=value;}};
struct PreImpl;struct PreSlot {CaptureLease source;uint64_t submit=0,generation=0;std::array<uint64_t,1> buffers{};std::unique_ptr<Graph> graph;};
class DaxaFrame {std::shared_ptr<PreImpl> owner_;std::shared_ptr<PreSlot> slot_;friend class DaxaStereo;public:void validate()const{require(bool(slot_),"empty frame");}struct Plane{uint64_t id;uint64_t buffer()const{return id;}};template<class P>Plane plane(P p)const{return {uint64_t(unsigned(p)+100)};}DaxaFrame()=default;DaxaFrame(std::shared_ptr<PreImpl> o,std::shared_ptr<PreSlot> s):owner_(o),slot_(s){}explicit operator bool()const{return bool(slot_);}CaptureLease source()const{return slot_->source;}std::shared_ptr<const Maps> calibration()const;uint64_t context_id()const;};
struct PreImpl {std::shared_ptr<Context> context;std::shared_ptr<const Maps> calibration;std::vector<std::shared_ptr<PreSlot>> slots;uint64_t retired_sources=0,next_generation=1;double upload_cpu_ms=0;void require_thread()const{}};
std::shared_ptr<const Maps> DaxaFrame::calibration()const{return owner_->calibration;}uint64_t DaxaFrame::context_id()const{return reinterpret_cast<uintptr_t>(owner_->context.get());}
class DaxaPreprocess {public:std::shared_ptr<PreImpl> impl_;static constexpr unsigned Staging=0;DaxaFrame enqueue_at(CaptureLease,uint64_t);size_t collect_completed_sources_at(uint64_t);};
enum B:unsigned {Left,Right,LeftDesc,RightDesc,LeftValid,RightValid,Costs,Sums,Disparity,Confidence,Validity,Depth,DepthValid,Counters,Penalties,Params,BufferCount};
enum class DaxaDepthStereoSchedule{Sequential,PairedPerPath,LayeredPerPath,SharedCensusSequential};struct DaxaDepthConfig {DaxaDepthStereoSchedule stereo_schedule=DaxaDepthStereoSchedule::PairedPerPath;};
enum class DaxaStereoReference{Left,Right};enum class DaxaStereoPlane{Disparity,Confidence,Validity};enum class DaxaPreprocessPlane{LeftSupport,RightSupport,LeftLuma,RightLuma};
struct StereoSlot{std::array<uint64_t,BufferCount> buffers{};std::array<External,BufferCount> external{};bool per_reference_timings_available=true;Parameters parameters;DaxaFrame input;CaptureLease source;uint64_t submit=0,generation=0;};
struct StereoImpl;struct PairSlot {std::array<std::array<External,BufferCount>,2> external;std::unique_ptr<Graph> graph;unsigned started=0,finished=0;};struct Pair {unsigned stages=8;std::shared_ptr<StereoImpl> right;std::vector<std::unique_ptr<PairSlot>> slots;};
struct StereoImpl {std::unique_ptr<Pair> paired;std::shared_ptr<const Maps> calibration;unsigned stages=8;uint64_t next_generation=1;std::shared_ptr<Context> context;std::vector<std::shared_ptr<StereoSlot>> slots;DaxaStereoReference reference;void require_thread()const{}};
class DaxaStereo {public:std::shared_ptr<StereoImpl> impl_;class Frame {std::shared_ptr<StereoImpl> owner_;std::shared_ptr<StereoSlot> slot_;public:Frame()=default;Frame(std::shared_ptr<StereoImpl> o,std::shared_ptr<StereoSlot> s):owner_(o),slot_(s){}DaxaStereoReference reference()const{return owner_->reference;}CaptureLease source()const{return slot_->source;}uint64_t context_id()const{return reinterpret_cast<uintptr_t>(owner_->context.get());}std::shared_ptr<const Maps> calibration_owner()const{return slot_->input.calibration();}const std::string& calibration_identity()const{return slot_->input.calibration()->rectified_identity;}};
 Frame process_at(const DaxaFrame* pre,uint64_t oldest){for(auto& s:impl_->slots)if(detail::available_slot(s.use_count(),s->submit,oldest)){s->input=*pre;s->source=pre->source();s->submit=++impl_->context->device.latest;impl_->context->device.collect_garbage();return Frame(impl_,s);}throw std::runtime_error("stereo full");}
 unsigned pair_slot_at(uint64_t)const;std::array<Frame,2> process_pair_at(const DaxaFrame&,uint64_t,unsigned);size_t collect_completed_inputs_at(uint64_t);
};
class DaxaDepthPipeline {public:struct Slot {DaxaFrame preprocessing;DaxaStereo::Frame left,right;uint64_t submit=0,generation=0;std::chrono::steady_clock::time_point host_start;std::array<External,8> external_inputs{};Parameters data;uint64_t parameters=0;std::unique_ptr<Graph> graph;};
 struct Impl {DaxaDepthConfig config;std::shared_ptr<Context> context;std::unique_ptr<DaxaPreprocess> preprocessing;std::unique_ptr<DaxaStereo> left,right;std::vector<std::shared_ptr<Slot>> slots;uint64_t retired=0,next_generation=1;void require_thread()const{}size_t collect_at(uint64_t);
 std::array<uint64_t,8> input_buffers(const DaxaFrame&,const DaxaStereo::Frame&,const DaxaStereo::Frame&){return {};}};
 std::shared_ptr<Impl> impl_;
 class Frame {public:std::shared_ptr<Impl> owner_;std::shared_ptr<Slot> slot_;uint64_t generation_=0;Frame()=default;Frame(std::shared_ptr<Impl> o,std::shared_ptr<Slot> s):owner_(o),slot_(s),generation_(s->generation){}void validate()const;};
 Frame enqueue(CaptureLease);static void validate_pair(const DaxaFrame&,const DaxaStereo::Frame&,const DaxaStereo::Frame&);
};
'''
    program += definition(pre,'DaxaFrame DaxaPreprocess::enqueue_at(')
    program += definition(pre,'size_t DaxaPreprocess::collect_completed_sources_at(')
    program += definition(stereo,'size_t DaxaStereo::collect_completed_inputs_at(')
    program += definition(stereo,'unsigned DaxaStereo::pair_slot_at(')
    program += definition(stereo,'std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at(')
    # Actual production depth bookkeeping, pairing, frame validity and staging copy.
    program += definition(depth,'size_t collect_at(').replace('size_t collect_at(', 'size_t DaxaDepthPipeline::Impl::collect_at(',1)
    for signature in ('void DaxaDepthPipeline::validate_pair(', 'DaxaDepthPipeline::Frame DaxaDepthPipeline::enqueue(', 'void DaxaDepthPipeline::Frame::validate('):
        program += definition(depth,signature)
    program += r'''
}
using namespace stereoforge;
void check(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
int main(int argc,char** argv){try{
 auto context=std::make_shared<Context>();auto e=std::make_shared<DaxaDepthPipeline::Impl>();e->context=context;if(argc>1)e->config.stereo_schedule=std::string(argv[1])=="shared"?DaxaDepthStereoSchedule::SharedCensusSequential:DaxaDepthStereoSchedule::LayeredPerPath;
 e->preprocessing=std::make_unique<DaxaPreprocess>();e->preprocessing->impl_=std::make_shared<PreImpl>();auto pre=e->preprocessing->impl_;pre->context=context;pre->calibration=std::make_shared<Maps>();
 std::array<std::vector<uint8_t>,2> staging;for(auto& bytes:staging)bytes.resize(1280*2*720*2);
 for(unsigned i=0;i<2;++i){auto s=std::make_shared<PreSlot>();s->buffers[0]=reinterpret_cast<uintptr_t>(staging[i].data());s->graph=std::make_unique<Graph>(Graph{&context->device,true});pre->slots.push_back(std::move(s));}
 for(unsigned eye=0;eye<2;++eye){auto engine=std::make_unique<DaxaStereo>();engine->impl_=std::make_shared<StereoImpl>();engine->impl_->context=context;engine->impl_->calibration=pre->calibration;engine->impl_->reference=eye?DaxaStereoReference::Right:DaxaStereoReference::Left;for(unsigned i=0;i<2;++i){auto s=std::make_shared<StereoSlot>();s->buffers[Params]=reinterpret_cast<uintptr_t>(&s->parameters);engine->impl_->slots.push_back(std::move(s));}(eye?e->right:e->left)=std::move(engine);}
 e->left->impl_->paired=std::make_unique<Pair>();e->left->impl_->paired->right=e->right->impl_;for(unsigned i=0;i<2;++i){auto s=std::make_unique<PairSlot>();s->graph=std::make_unique<Graph>(Graph{&context->device,false,&s->started,&s->finished,8});e->left->impl_->paired->slots.push_back(std::move(s));}
 for(unsigned i=0;i<2;++i){auto s=std::make_shared<DaxaDepthPipeline::Slot>();s->parameters=reinterpret_cast<uintptr_t>(&s->data);s->graph=std::make_unique<Graph>(Graph{&context->device,false});e->slots.push_back(std::move(s));}
 auto owned=std::make_shared<CaptureFrame>();owned->frame_id=INT64_MAX;owned->packed_yuyv.resize(1280*2*720*2,173);CaptureLease source=owned;DaxaDepthPipeline pipeline;pipeline.impl_=e;
 auto complete=[&]{context->device.oldest=context->device.latest+1;};
 auto first=pipeline.enqueue(source);complete();auto second=pipeline.enqueue(source);complete();first.validate();second.validate();
 auto uploads=context->device.uploads,latest=context->device.latest;bool caught=false;try{auto third=pipeline.enqueue(source);}catch(const std::runtime_error&){caught=true;}
 check(caught&&context->device.uploads==uploads&&context->device.latest==latest,"full output ring uploaded/submitted");
 check(e->collect_at(context->device.oldest)==0&&bool(e->slots[0]->preprocessing)&&bool(e->slots[1]->preprocessing),"held output lost producers");
 // Independently retained image markers exhaust only the pre ring; no upload is permitted.
 DaxaFrame rgb0=first.slot_->preprocessing,rgb1=second.slot_->preprocessing;first={};second={};check(e->collect_at(context->device.oldest)==2,"unleased completed outputs did not retire");
 check(!e->slots[0]->preprocessing&&!e->slots[1]->preprocessing,"retired outputs pin hidden producers");
 uploads=context->device.uploads;latest=context->device.latest;caught=false;try{auto third=pipeline.enqueue(source);}catch(const std::runtime_error&){caught=true;}
 check(caught&&context->device.uploads==uploads&&context->device.latest==latest,"retained RGB rejection occurred after upload");
 rgb0={};auto third=pipeline.enqueue(source);third.validate();check(third.slot_->left.source()==source&&third.slot_->right.source()==source,"paired sources not retained");check(third.slot_->data.source_id==uint64_t(INT64_MAX),"ID truncated");
 third={};auto retired=e->collect_at(context->device.oldest);check(retired==0&&bool(e->slots[0]->preprocessing),"inflight producer retired");complete();rgb1={};e->collect_at(context->device.oldest);
 for(auto& s:pre->slots)check(!s->source,"completed unleased preprocessing keeps capture metadata pinned");
 for(auto* engine:{e->left.get(),e->right.get()})for(auto& s:engine->impl_->slots)check(!s->source&&!s->input,"completed unleased stereo keeps capture metadata pinned");
 auto bad=std::make_shared<CaptureFrame>();bad->frame_id=uint64_t(INT64_MAX)+1;auto snapshots=context->device.snapshots;uploads=context->device.uploads;caught=false;try{pipeline.enqueue(bad);}catch(const std::invalid_argument&){caught=true;}
 check(caught&&context->device.snapshots==snapshots&&context->device.uploads==uploads,"invalid signed ID reached device/upload");
 // Exhaust both aligned right producer slots; pair admission must reject before upload.
 std::array<std::shared_ptr<StereoSlot>,2> held_right{e->right->impl_->slots[0],e->right->impl_->slots[1]};uploads=context->device.uploads;latest=context->device.latest;caught=false;
 try{pipeline.enqueue(source);}catch(const std::exception&){caught=true;}check(caught&&context->device.uploads==uploads&&context->device.latest==latest,"aligned pair ring rejection occurred after upload");held_right={};
 // Either producer ring must reject before touching upstream upload/submission.
 std::array<std::shared_ptr<StereoSlot>,2> held_left{e->left->impl_->slots[0],e->left->impl_->slots[1]};uploads=context->device.uploads;latest=context->device.latest;caught=false;
 try{pipeline.enqueue(source);}catch(const std::exception&){caught=true;}check(caught&&context->device.uploads==uploads&&context->device.latest==latest,"aligned left ring rejection occurred after upload");held_left={};
 // Warm state and fixed allocation; extract actual depth/retirement/staging methods.
 constexpr unsigned frames=256;allocations=0;auto before_snapshots=context->device.snapshots;auto before_submissions=context->device.latest;auto before_gc=context->device.gc;tracking=true;auto start=std::chrono::steady_clock::now();
 for(unsigned i=0;i<frames;++i){auto out=pipeline.enqueue(source);out.validate();complete();out={};}
 e->collect_at(context->device.oldest);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();tracking=false;
 check(context->device.gc-before_gc==frames*2,"paired+depth framework garbage collection calls changed");check(context->device.latest-before_submissions==frames*3,"paired frame must submit preprocessing, pair, depth only");check(allocations==0,"application heap allocation in warmed successful frame path");check(context->device.snapshots-before_snapshots==frames,"more than one application admission snapshot");
 check(staging[0][0]==173&&staging[0].back()==173,"actual packed staging copy missing");
 std::cout<<"{\"frames\":"<<frames<<",\"packed_bytes_per_frame\":"<<owned->packed_yuyv.size()<<",\"mean_cpu_ms\":"<<ms/frames<<",\"application_allocations\":"<<allocations<<",\"admission_snapshots\":"<<context->device.snapshots-before_snapshots<<",\"full_output_and_rgb_backpressure_before_upload\":true,\"inflight_and_retained_producers_preserved\":true,\"scope\":\"actual_paired_and_depth_and_pre_bookkeeping_fake_GPU\"}\n";
 return 0;
}catch(const std::exception& ex){tracking=false;std::cerr<<ex.what()<<'\n';return 1;}}
'''
    if schedule == 'shared-census-sequential':
        assert paths in (4, 8)
        phases = 6 + 2 * paths
        program = program.replace('unsigned stages=8;', f'unsigned stages={phases};').replace('&s->finished,8}', f'&s->finished,{phases}'+'}')
    directory=tmp_path_factory.mktemp('daxa-depth-bookkeeping')
    code=directory/'probe.cpp';code.write_text(program);binary=directory/'probe'
    build=subprocess.run([compiler,'-std=c++20','-O3',str(code),'-o',str(binary)],capture_output=True,text=True,timeout=60)
    assert build.returncode==0,build.stderr
    return binary


@pytest.fixture(scope='module')
def pair_bookkeeping_probe(tmp_path_factory):
    return build_pair_bookkeeping_probe(tmp_path_factory)


@pytest.mark.parametrize('schedule',['paired-per-path','layered-per-path'])
def test_actual_pair_depth_admission_retirement_and_upload_have_zero_warm_allocations(pair_bookkeeping_probe,schedule):
    command=[str(pair_bookkeeping_probe)]+(['layered'] if schedule=='layered-per-path' else [])
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    row=json.loads(result.stdout)
    print('Actual '+schedule+' depth/pre bookkeeping CPU-only probe:',json.dumps(row,sort_keys=True))
    assert row['frames']==256 and row['packed_bytes_per_frame']==3686400
    assert row['application_allocations']==0 and row['admission_snapshots']==256
    assert row['full_output_and_rgb_backpressure_before_upload'] and row['inflight_and_retained_producers_preserved']
    assert 0<row['mean_cpu_ms']<1000


def check_pair_depth_admission(depth):
    enqueue=body(depth,'DaxaDepthPipeline::Frame DaxaDepthPipeline::enqueue(')
    assert enqueue.count('oldest_pending_submit_index(')==1
    assert enqueue.index('e.collect_at(oldest)')<enqueue.index('detail::available_slot(')<enqueue.index('e.left->pair_slot_at(oldest)')<enqueue.index('e.preprocessing->enqueue_at(')<enqueue.index('e.left->process_pair_at(')
    assert 'process_pair_at(pre,oldest,pair_index)' in enqueue
    assert 'slot->preprocessing=std::move(pre)' in enqueue and 'slot->left=std::move(left)' in enqueue and 'slot->right=std::move(right)' in enqueue
    setup=body(depth,'Impl(DaxaRectification maps,DaxaDepthConfig c,')
    assert setup.index('l={};r={};pre={};prototype.reset()')<setup.index('left->setup_pair(*right'), 'pair setup cannot retain prototype Frame/slot'
    timing=body(depth,'DaxaDepthTiming DaxaDepthPipeline::Frame::timings(')
    assert 'out.per_reference_stereo_timings_available=false' in timing
    assert 'out.paired_stereo=owner_->left->pair_timings(slot_->left,slot_->right)' in timing
    assert 'result[6]-out.preprocessing.start_tick' in timing, 'actual full queue span remains authoritative'


def test_depth_rejects_full_paired_ring_before_upload_and_keeps_complete_producers():
    check_pair_depth_admission((ROOT/'src/daxa_depth.cpp').read_text())


def test_admission_guard_catches_late_pair_rejection_after_upload():
    source=(ROOT/'src/daxa_depth.cpp').read_text()
    before='auto pair_index=e.config.stereo_schedule!=DaxaDepthStereoSchedule::Sequential?e.left->pair_slot_at(oldest):0u;'
    assert before in source
    # Move the actual admission into the downstream path: this mutation must fail.
    moved=source.replace(before,'auto pair_index=0u;',1).replace('auto pre=e.preprocessing->enqueue_at(std::move(source),oldest);','auto pre=e.preprocessing->enqueue_at(std::move(source),oldest);'+before,1)
    with pytest.raises(AssertionError):check_pair_depth_admission(moved)


def check_paired_reporting(app,stereo):
    timed=body(app,'for(uint32_t i=0;i<frames+warmup;++i)')
    assert timed.index('auto a=Clock::now()')<timed.index('output={}')<timed.index('pipeline->enqueue(source)')<timed.index('output.wait()')<timed.index('output.timings()')<timed.index('auto d=Clock::now()')
    assert not re.search(r'readback\w*\(|create_\w+\(|resize\(|push_back\(|strict_json|metric_calibration\(',timed), 'timed samples exclude diagnostics/setup work'
    assert '(!per_reference&&(k==8||k==9))?json(nullptr):summary(samples[k])' in app, 'paired per-eye timings must be explicit null'
    assert 'per_reference?json(nullptr):summary(pair_samples)' in app
    assert 'no_isolated_per_reference_times' in app
    assert 'SDK_GC_internal_scans_additional_included_host_wall' in app
    query=body(stereo,'DaxaStereoTiming DaxaStereo::pair_timings(')
    assert query.count('daxa_timeline_query_pool_query_results(')==1 and 'std::array<uint64_t,' in query
    assert not re.search(r'create_|wait_idle|wait_on_submit|std::vector|push_back|resize',query)
    assert 'require(slot_->per_reference_timings_available,' in body(stereo,'DaxaStereoTiming DaxaStereo::Frame::timings('), 'isolated producer timings cannot silently return stale sequential queries'


def test_paired_report_has_real_phase_times_and_unavailable_eye_times():
    check_paired_reporting((ROOT/'apps/daxa_depth_benchmark.cpp').read_text(),(ROOT/'src/daxa_stereo.cpp').read_text())


def test_report_guard_rejects_fabricated_zero_eye_timings():
    app=(ROOT/'apps/daxa_depth_benchmark.cpp').read_text()
    with pytest.raises(AssertionError,match='explicit null'):
        check_paired_reporting(app.replace('(!per_reference&&(k==8||k==9))?json(nullptr):summary(samples[k])','summary(samples[k])'),(ROOT/'src/daxa_stereo.cpp').read_text())


def test_coordinator_owns_only_one_way_engine_reference_and_no_producer_markers():
    source=(ROOT/'src/daxa_stereo.cpp').read_text()
    coordinator=body(source,'struct DaxaStereo::Pair')
    assert 'std::shared_ptr<Impl> right' in coordinator
    assert not re.search(r'\b(?:DaxaFrame|CaptureLease|Frame)\b|std::shared_ptr<Impl>\s+(?:left|owner)',coordinator), 'coordinator cannot retain its owning engine or producer leases'
    destruction=body(source,'~Impl()')
    assert destruction.index('paired.reset()')<destruction.index('slots.clear()')<destruction.index('pipelines={}'), 'raw graph references must retire before producer storage/pipelines'


def test_both_slot_guard_rejects_one_sided_pair_admission():
    source=(ROOT/'src/daxa_stereo.cpp').read_text()
    right='&&detail::available_slot(r.slots[index].use_count(),r.slots[index]->submit,oldest)'
    assert right in source
    with pytest.raises(AssertionError,match='both aligned'):
        check_pair_hotpath(source.replace(right,'',1))


def test_layered_per_path_is_an_explicit_separate_schedule():
    header=(ROOT/'include/stereoforge/daxa_depth.hpp').read_text()
    assert 'LayeredPerPath' in header, 'true layered schedule has not been implemented'
    assert 'DaxaDepthStereoSchedule::Sequential' in header


def check_layered_shader(wrapper,helper):
    for name in ('aggregate_layered_body','select_layered_body'):
        fn=body(wrapper,'void '+name+'<')
        assert 'if(gid.z>=2)return;' in fn, 'uniform layer guard must precede loads and collectives'
        assert fn.index('if(gid.z>=2)return;')<fn.index('*(Parameters*)'), 'invalid eye must not dereference parameters'
        assert 'if(gid.z==0)' in fn and 'layered_pc.left_parameters' in fn and 'layered_pc.right_parameters' in fn
        assert not re.search(r'tid\.[xyz].*(?:==|>=)\s*2|WaveGetLaneIndex|GroupMemoryBarrier',fn), 'eye selection must be workgroup-uniform before shared helper'
    assert '[numthreads(64,1,1)]' in wrapper and '[numthreads(256,1,1)]' in wrapper
    aggregate=body(helper,'void aggregate_specialized_body<')
    select=body(helper,'void select_specialized_body<')
    assert 'gid.z' not in aggregate+select, 'eye layer cannot enter XY scanline/pixel flattening'
    assert '(gid.y*groups_x+gid.x)*(64/lanes)+WaveGetWaveIndex()' in aggregate
    assert '(gid.y*groups_x+gid.x)*64+tid.x/4' in select
    assert not re.search(r'\bpc\.|layered_pc|\bp\.',helper), 'shared body consumes explicit parameters, not wrapper-global push state'
    assert 'Parameters parameters,uint path,uint groups_x,uint3 gid' in helper
    assert 'Parameters parameters,uint groups_x,uint3 gid,uint3 tid' in helper
    assert 'WaveActiveMin(local_min)' in aggregate and 'words[i*row_words+word]+=addition' in aggregate
    assert 'reference_cost<RightReference>' in aggregate and 'reference_sum<Components,Bits,RightReference>' in select


def test_layered_eye_guard_is_uniform_and_shared_helper_flattens_only_xy():
    check_layered_shader((ROOT/'python/stereoforge/shaders/daxa/stereo_layered.slang').read_text(),(ROOT/'python/stereoforge/shaders/ops/sgm_specialized.slang').read_text())


@pytest.mark.parametrize('mutation',['remove_guard','load_first','flatten_z'])
def test_layered_shader_guard_detects_invalid_layer_load_and_cross_eye_index(mutation):
    wrapper=(ROOT/'python/stereoforge/shaders/daxa/stereo_layered.slang').read_text()
    helper=(ROOT/'python/stereoforge/shaders/ops/sgm_specialized.slang').read_text()
    if mutation=='remove_guard':wrapper=wrapper.replace('if(gid.z>=2)return;','',1)
    if mutation=='load_first':wrapper=wrapper.replace('if(gid.z>=2)return;','auto bad=*(Parameters*)layered_pc.left_parameters;if(gid.z>=2)return;',1)
    if mutation=='flatten_z':helper=helper.replace('gid.y*groups_x+gid.x','gid.z*groups_x+gid.y*groups_x+gid.x',1)
    with pytest.raises(AssertionError):check_layered_shader(wrapper,helper)


@pytest.fixture(scope='module')
def actual_layered_wrapper_probe(tmp_path_factory):
    compiler=shutil.which('c++')
    if compiler is None:pytest.skip('C++ compiler unavailable')
    wrapper=(ROOT/'python/stereoforge/shaders/daxa/stereo_layered.slang').read_text()
    helper=(ROOT/'python/stereoforge/shaders/ops/sgm_specialized.slang').read_text()
    aggregate=definition(wrapper,'void aggregate_layered_body<').replace('void aggregate_layered_body<let Components:uint,let Bits:uint>','template<unsigned Components,unsigned Bits> void aggregate_layered_body')
    select=definition(wrapper,'void select_layered_body<').replace('void select_layered_body<let Components:uint,let Bits:uint>','template<unsigned Components,unsigned Bits> void select_layered_body')
    line=re.search(r'uint line=([^;]+);',body(helper,'void aggregate_specialized_body<')).group(1)
    pixel=re.search(r'uint i=(.*?),lane=',body(helper,'void select_specialized_body<')).group(1)
    program=r'''
#include <cstdint>
#include <iostream>
#include <stdexcept>
using uint=uint32_t;struct uint3{uint x,y,z;};struct Parameters{uint eye;};
struct LayeredPush {uint64_t left_parameters,right_parameters;uint pass,groups_x;} layered_pc;
unsigned calls=0,last_eye=99,last_pass=99,last_groups=0;uint3 last_gid{},last_tid{};
template<unsigned Components,unsigned Bits,bool Right>void aggregate_specialized_body(Parameters p,uint path,uint groups,uint3 gid){if(Right!=bool(p.eye))throw std::runtime_error("wrong aggregate eye");++calls;last_eye=p.eye;last_pass=path;last_groups=groups;last_gid=gid;}
template<unsigned Components,unsigned Bits,bool Right>void select_specialized_body(Parameters p,uint groups,uint3 gid,uint3 tid){if(Right!=bool(p.eye))throw std::runtime_error("wrong select eye");++calls;last_eye=p.eye;last_groups=groups;last_gid=gid;last_tid=tid;}
'''+aggregate+select+r'''
unsigned wave_index=0;uint WaveGetWaveIndex(){return wave_index;}
uint actual_line(uint3 gid,uint groups_x,uint lanes){return '''+line+r''';}
uint actual_pixel(uint3 gid,uint3 tid,uint groups_x){return '''+pixel+r''';}
int main(){try{Parameters left{0},right{1};layered_pc={uint64_t(&left),uint64_t(&right),3,7};uint64_t cases=0;
for(uint z:{0u,1u})for(uint y:{0u,1u,17u})for(uint x:{0u,3u,6u}){
 uint3 gid{x,y,z};aggregate_layered_body<4,8>(gid);if(last_eye!=z||last_pass!=3||last_groups!=7)return 1;select_layered_body<2,16>(gid,{255,0,0});if(last_eye!=z||last_tid.x!=255)return 2;
 for(uint groups:{1u,7u,65535u})for(uint lanes:{32u,64u}){for(wave_index=0;wave_index<64/lanes;++wave_index){if(actual_line(gid,groups,lanes)!=(uint64_t(y)*groups+x)*(64/lanes)+wave_index)return 3;++cases;}
 for(uint t=0;t<256;++t){if(actual_pixel(gid,{t,0,0},groups)!=(uint64_t(y)*groups+x)*64+t/4)return 4;++cases;}}
}
// Invalid layers must return before dereferencing null addresses.
layered_pc.left_parameters=layered_pc.right_parameters=0;auto before=calls;
for(uint z:{2u,3u,0xffffffffu}){aggregate_layered_body<4,8>({0,0,z});select_layered_body<2,16>({0,0,z},{0,0,0});}
if(calls!=before)return 5;
std::cout<<"{\"mapping_cases\":"<<cases<<",\"invalid_layers_rejected_before_parameter_load\":true,\"scope\":\"actual_shader_wrapper_routing_and_helper_index_expressions_CPU_only_no_GPU_collective_execution\"}\n";
return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 6;}}
'''
    directory=tmp_path_factory.mktemp('daxa-layered-wrapper');code=directory/'probe.cpp';code.write_text(program);binary=directory/'probe'
    result=subprocess.run([compiler,'-std=c++20','-O3',str(code),'-o',str(binary)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    return binary


def test_actual_layered_wrapper_routes_whole_eye_and_rejects_invalid_z(actual_layered_wrapper_probe):
    result=subprocess.run([str(actual_layered_wrapper_probe)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    row=json.loads(result.stdout);print('Actual layered wrapper CPU probe:',json.dumps(row,sort_keys=True))
    assert row['invalid_layers_rejected_before_parameter_load'] and row['mapping_cases']==27810


# 2094237 frozen bodies, normalized only for explicit parameter/pass/XY/output
# pointer bindings. Recompilation still needs independent GPU output requalification.
FROZEN_SPECIALIZED_BODIES = {'uint specialized_cost(': 'f6d46f8cddc720304e80381da9aaf383bba937ed767c2054c470db9a18acb2fb', 'uint specialized_sum<': 'bcac4328ecfb99f8dbca9e81a1dc1e9b3b96c97406abb15d5cea083d2325a962', 'uint reference_cost<': '40186619ca3e76b2839ad47eac4db8ac630810aed582ea4b3f7d86b5c259ac1c', 'uint reference_sum<': 'e5685489b7cfb7a3e3b1ecf0dbb3e2eceeb64fff2b1c43a5727fcb2edbd0f401', 'void aggregate_specialized_body<': '541aa9f96b16c9ea60a92c7ed08a977dde88728fcf955478d2d1f59558fa790a', 'void select_specialized_body<': 'e9c3e29a986a66eb1a660c510d67bc713fb98a6f30ea3417b1f4029b57cb58fb'}


def canonical_shared_body(text):
    text=re.sub(r'\s+','',re.sub(r'//[^\n]*','',text).replace('parameters.','p.').replace('sgm_direction(path)','sgm_direction(pc.pass)').replace('(gid.y*groups_x+gid.x)','linear_group(gid)'))
    if 'float*output_disparity=' in text:
        binding='float*output_disparity=(float*)p.disparity_address;float*output_confidence=(float*)p.confidence_address;uint*output_validity=(uint*)p.validity_address;'
        assert binding in text, 'explicit output bindings must retain exact frozen address/type'
        text=text.replace(binding,'').replace('output_disparity[','disparity[').replace('output_confidence[','confidence[').replace('output_validity[','validity[')
    return text


def test_shared_specialized_math_matches_frozen_pre_extraction_bodies():
    import hashlib
    helper=(ROOT/'python/stereoforge/shaders/ops/sgm_specialized.slang').read_text()
    for signature,expected in FROZEN_SPECIALIZED_BODIES.items():
        assert hashlib.sha256(canonical_shared_body(body(helper,signature)).encode()).hexdigest()==expected, signature+' frozen math/output expression drift'


def check_layered_host(source,app):
    setup=body(source,'void DaxaStereo::setup_pair(')
    assert 'pair.layered=layered' in setup
    assert 'bool layered_phase=pair.layered&&stage>=3' in setup, 'only aggregation/selection use eye-layer dispatch'
    assert 'l.group_limits[2]>=2' in setup and 'maxPushConstantsSize>=sizeof(LayeredPush)' in setup
    assert 'static_assert(sizeof(LayeredPush)==24&&offsetof(LayeredPush,pass)==16)' in source
    assert 'static_assert(sizeof(Push)==16)' in source, 'legacy push ABI remains separate'
    assert 'if(i<2){shader.create_flags=daxa::ShaderCreateFlagBits::REQUIRE_FULL_SUBGROUPS;shader.required_subgroup_size=l.wave_size;}' in setup, 'aggregate retains required full subgroup size; selection retains legacy policy'
    branch=body(setup,'if(layered_phase)')
    assert branch.count('ti.recorder.dispatch(')==1 and 'dispatch({grid[0],grid[1],2})' in branch, 'each layered phase is one real dispatch with two workgroup layers'
    assert 'LayeredPush{addresses[0],addresses[1],pass,grid[0]}' in branch
    assert not re.search(r'for\(|write_timestamp|pipeline_barrier|wait_',branch)
    assert 'else for(unsigned eye=0;eye<2;++eye)' in setup, 'paired control keeps two legacy reference dispatches'
    assert 'two_reference_dispatches_census_clear; z2_single_dispatch_aggregation_selection' in app, 'report exact actual dispatch form'
    assert 'layered_per_path_phase_intervals_include_both_references; no_isolated_per_reference_times' in app


def test_layered_host_records_one_z2_dispatch_and_keeps_legacy_push_and_fullwave_policy():
    check_layered_host((ROOT/'src/daxa_stereo.cpp').read_text(),(ROOT/'apps/daxa_depth_benchmark.cpp').read_text())


@pytest.mark.parametrize('old,new', [('dispatch({grid[0],grid[1],2})','dispatch({grid[0],grid[1],1})'), ('pair.layered&&stage>=3','pair.layered'), ('l.group_limits[2]>=2','true'), ('sizeof(LayeredPush)==24','sizeof(LayeredPush)==16')])
def test_layered_host_guard_detects_missing_eye_wrong_phase_or_abi(old,new):
    source=(ROOT/'src/daxa_stereo.cpp').read_text();assert old in source
    with pytest.raises(AssertionError):check_layered_host(source.replace(old,new,1),(ROOT/'apps/daxa_depth_benchmark.cpp').read_text())


def check_layered_identity(cmake,source,app):
    for path,macro in [('daxa/stereo_parameters.slang','PARAMETERS'),('ops/sgm_specialized.slang','SPECIALIZED'),('daxa/stereo_layered.slang','LAYERED')]:
        assert path in cmake and re.search(r'file\(SHA256[^\n]*'+re.escape(path)+r'[^\n]*SF_DAXA_STEREO_'+macro+'_SHA256',cmake), 'all shared/wrapper bytes need compile-time identities'
        assert 'SF_DAXA_STEREO_'+macro+'_SHA256="${SF_DAXA_STEREO_'+macro+'_SHA256}"' in cmake
    configure=re.search(r'set_property\(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS[^\n]*',cmake).group()
    assert 'stereo_layered.slang' in configure and '${sf_daxa_stereo_shared}' in configure
    assert 'DEPENDS "${source}" shaders/ops/sgm.slang ${sf_daxa_stereo_shared}' in cmake
    shared=body(source,'std::string DaxaStereo::specialized_shader_identity(')
    assert 'SF_DAXA_STEREO_PARAMETERS_SHA256' in shared and 'SF_DAXA_STEREO_SPECIALIZED_SHA256' in shared
    assert 'SF_DAXA_STEREO_LAYERED_SHA256' in body(source,'std::string DaxaStereo::layered_shader_identity(')
    for key in ('stereo_shared_shader_source_sha256_pair','stereo_layered_shader_source_sha256'):assert key in app
    assert app.index('--shader-source-identities')<app.index('std::make_shared<DaxaFoundation>'), 'identity getter must remain CPU-only'


def test_runtime_identities_and_rebuild_rules_bind_all_new_shared_shader_sources():
    check_layered_identity((ROOT/'cmake/DaxaStereo.cmake').read_text(),(ROOT/'src/daxa_stereo.cpp').read_text(),(ROOT/'apps/daxa_depth_benchmark.cpp').read_text())


def test_identity_guard_rejects_unbound_new_helper_hash():
    cmake=(ROOT/'cmake/DaxaStereo.cmake').read_text()
    with pytest.raises(AssertionError):
        check_layered_identity(cmake.replace('SF_DAXA_STEREO_SPECIALIZED_SHA256','DISCARDED_HASH'),(ROOT/'src/daxa_stereo.cpp').read_text(),(ROOT/'apps/daxa_depth_benchmark.cpp').read_text())


def test_built_cpu_identity_getter_matches_actual_wrapper_helper_and_parameters():
    import hashlib
    import os
    binary=ROOT/'build/daxa/daxa_depth_benchmark'
    if not binary.is_file():pytest.skip('optional Daxa benchmark is not built')
    env=dict(os.environ,VK_DRIVER_FILES='/nonexistent-stereoforge-CPU-only-identity.json')
    result=subprocess.run([str(binary),'--shader-source-identities'],capture_output=True,text=True,env=env,timeout=10)
    assert result.returncode==0 and not result.stderr,result.stderr
    def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    assert json.loads(result.stdout)=={
        'stereo_shader_source_sha256_pair':sha('python/stereoforge/shaders/daxa/stereo.slang')+':'+sha('python/stereoforge/shaders/ops/sgm.slang'),
        'stereo_shared_shader_source_sha256_pair':sha('python/stereoforge/shaders/daxa/stereo_parameters.slang')+':'+sha('python/stereoforge/shaders/ops/sgm_specialized.slang'),
        'stereo_layered_shader_source_sha256':sha('python/stereoforge/shaders/daxa/stereo_layered.slang')}
