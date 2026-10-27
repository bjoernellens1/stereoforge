"""Independent CPU-only depth admission/retirement and performance guards.

Extracted-source probes replace only GPU device/graph/producers; allocation and
CPU bookkeeping measurements exclude Daxa internals, drivers and GPU execution.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]


def definition(source, signature):
    start = source.index(signature)
    opening = source.index('{', start)
    return source[start:opening] + '{' + body(source, signature) + '}\n'


def check_geometry_stays_setup_only(preprocess, stereo):
    hot = ''.join(body(preprocess, sig) for sig in (
        'void DaxaRectification::validate_source(',
        'DaxaFrame DaxaPreprocess::enqueue_at(',
        'size_t DaxaPreprocess::collect_completed_sources_at('))
    hot += body(stereo, 'DaxaStereo::Frame DaxaStereo::process_at(')
    assert not re.search(r'metric_calibration\(|sha256|sha_file|strict_json|R1\[|R2\[|P1\[|P2\[|\.geometry', hot), 'matrix/hash/JSON geometry admission is setup-only'
    admission = body(preprocess, 'MetricCalibration DaxaRectification::metric_calibration(')
    assert 'geometry.has_value()' in admission
    assert 'g.original_identity==original_identity' in admission and 'g.rectified_identity==rectified_identity' in admission
    assert 'g.metric_calibration()' in admission


def test_retained_geometry_does_not_add_frame_matrix_hash_or_json_work():
    check_geometry_stays_setup_only((ROOT/'src/daxa_preprocess.cpp').read_text(), (ROOT/'src/daxa_stereo.cpp').read_text())


def test_geometry_guard_rejects_revalidation_on_every_capture():
    pre = (ROOT/'src/daxa_preprocess.cpp').read_text()
    marker = 'void DaxaRectification::validate_source(const CaptureFrame& f)const{'
    assert marker in pre
    with pytest.raises(AssertionError, match='setup-only'):
        check_geometry_stays_setup_only(pre.replace(marker, marker+'metric_calibration();', 1), (ROOT/'src/daxa_stereo.cpp').read_text())


def check_preprocessing_span_reuses_the_existing_query(preprocess):
    timing = body(preprocess, 'DaxaPreprocessTiming DaxaPreprocess::timings(')
    assert timing.count('daxa_timeline_query_pool_query_results(') == 1, 'span uses the existing single timestamp result fetch'
    assert 'slot->queries.get(),0,6,result.data()' in timing, 'retain six GPU timestamps rather than adding a query'
    assert 'result[0]' in timing and 'result[10]' in timing, 'raw first/last timestamps must come from returned values'
    assert 'timestamp_period' in timing and 'timestamp_bits' in timing
    assert not re.search(r'wait_on_submit|wait_idle|std::vector|create_timeline_query_pool|write_timestamp', timing)


def test_preprocessing_span_adds_no_gpu_query_or_synchronization():
    check_preprocessing_span_reuses_the_existing_query((ROOT/'src/daxa_preprocess.cpp').read_text())


def test_span_guard_rejects_duplicate_fetch():
    pre = (ROOT/'src/daxa_preprocess.cpp').read_text()
    marker = 'DaxaPreprocessTiming DaxaPreprocess::timings(const DaxaFrame& frame)const{'
    with pytest.raises(AssertionError, match='single timestamp'):
        check_preprocessing_span_reuses_the_existing_query(pre.replace(marker, marker+'daxa_timeline_query_pool_query_results();', 1))


def check_depth_submission(source, header):
    enqueue = body(source, 'DaxaDepthPipeline::Frame DaxaDepthPipeline::enqueue(')
    retire = body(source, 'size_t collect_at(')
    forbidden = r'\b(?:make_shared|make_unique|allocate|create_buffer|create_memory|create_command_recorder|wait_idle|wait_on_submit|readback|inspect|memcpy|malloc|calloc|realloc|resize|reserve|push_back|metric_calibration|sha_file|strict_json)(?:<[^;()]*>)?\s*\('
    assert not re.search(forbidden, enqueue+retire), 'depth admission/retirement must not allocate, wait, copy images, or validate geometry'
    assert enqueue.index('require_thread()') < enqueue.index('oldest_pending_submit_index(')
    assert enqueue.count('oldest_pending_submit_index(') == 1, 'one explicit application admission snapshot'
    assert 'oldest_pending_submit_index(' not in retire
    assert 'available_slot(' in enqueue and 'available_slot(' in retire, 'retained consumers block reuse and retirement'
    assert enqueue.index('e.collect_at(oldest)') < enqueue.index('available_slot(') < enqueue.index('enqueue_at('), 'retire then reject full output rings before uploading'
    assert enqueue.count('process_at(&pre,oldest)') == 2
    assert 'source->frame_id<=uint64_t(INT64_MAX)' in enqueue
    assert 'slot->preprocessing=std::move(pre)' in enqueue and 'slot->left=std::move(left)' in enqueue and 'slot->right=std::move(right)' in enqueue
    assert retire.index('available_slot(slot.use_count(),slot->submit,oldest)') < retire.index('slot->left={}')
    assert retire.index('slot->preprocessing={}') < retire.index('left->collect_completed_inputs_at(oldest)') < retire.index('preprocessing->collect_completed_sources_at(oldest)'), 'release completed unleased output producers before stereo/pre source retirement'
    getters = re.findall(r'DaxaStereo::Frame\s+(\w+)\s*\(', header)
    assert getters == ['stereo_frame'], 'only the explicit shared-schedule Stereo lease may escape aligned depth slots'
    getter = body(source, 'DaxaStereo::Frame DaxaDepthPipeline::Frame::stereo_frame(')
    assert 'owner_->config.stereo_schedule==DaxaDepthStereoSchedule::SharedCensusSequential' in getter, 'independent leases require aligned shared-schedule admission'
    assert getter.index('validate()') < getter.index('SharedCensusSequential') < getter.index('return ')
    assert 'reference==DaxaStereoReference::Left||reference==DaxaStereoReference::Right' in getter
    assert 'return reference==DaxaStereoReference::Left?slot_->left:slot_->right' in getter
    assert 'DaxaImageView image(' in header, 'independent RGB retention must be explicit'
    assert 'slot_->preprocessing.plane(p)' in body(source, 'DaxaImageView DaxaDepthPipeline::Frame::image(')


def test_combined_submission_and_retirement_are_bounded_and_ordered():
    check_depth_submission((ROOT/'src/daxa_depth.cpp').read_text(), (ROOT/'include/stereoforge/daxa_depth.hpp').read_text())


@pytest.mark.parametrize('regression', ['e.context->device.wait_idle();', 'std::make_shared<int>();', 'std::memcpy(a,b,n);', 'e.metric_calibration();'])
def test_depth_guard_catches_extra_wait_allocation_copy_or_geometry_scan(regression):
    source=(ROOT/'src/daxa_depth.cpp').read_text()
    marker='DaxaDepthPipeline::Frame DaxaDepthPipeline::enqueue(CaptureLease source){'
    assert marker in source
    with pytest.raises(AssertionError, match='must not allocate'):
        check_depth_submission(source.replace(marker,marker+regression,1),(ROOT/'include/stereoforge/daxa_depth.hpp').read_text())


def test_depth_guard_catches_completed_but_retained_output_overwrite():
    source=(ROOT/'src/daxa_depth.cpp').read_text()
    marker='detail::available_slot(slot.use_count(),slot->submit,oldest)'
    with pytest.raises(AssertionError):
        check_depth_submission(source.replace(marker,'detail::completed_submission(slot->submit,oldest)'),(ROOT/'include/stereoforge/daxa_depth.hpp').read_text())


def test_callbacks_and_cross_graph_visibility_do_not_retain_prototype_frames():
    source=(ROOT/'src/daxa_depth.cpp').read_text()
    callbacks=re.findall(r'task\.executes\(\[([^]]*)\]',source)
    assert callbacks and all(not re.search(r'\b(?:pre|prototype|left|right|l|r)\b',c) for c in callbacks), 'graph callbacks cannot hide frame/producer ownership cycles'
    assert 'l={};r={};pre={};prototype.reset();' in source, 'setup prototype leases must be released'
    assert 'AccessConsts::HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE' in source
    assert '.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ' in source
    assert 'task.reads(inputs[6]).reads(inputs[7])' in source
    assert 'for(unsigned p=0;p<3;++p)task.reads(outputs[p])' in source


@pytest.fixture(scope='module')
def depth_bookkeeping_probe(tmp_path_factory):
    compiler=shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable for extracted CPU-only bookkeeping probe')
    depth=(ROOT/'src/daxa_depth.cpp').read_text()
    pre=(ROOT/'src/daxa_preprocess.cpp').read_text()
    stereo=(ROOT/'src/daxa_stereo.cpp').read_text()
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
struct Parameters {std::array<uint64_t,8> inputs{};uint64_t source_id=0;};
struct Device {uint64_t oldest=1,latest=0,snapshots=0,gc=0,uploads=0;auto oldest_pending_submit_index(){++snapshots;return oldest;}auto latest_queue_submit_index(int){return latest;}void collect_garbage(){++gc;}
 struct Address{uint64_t id;void* value(){return reinterpret_cast<void*>(id);}};Address buffer_host_address(uint64_t id){return {id};}
 struct DeviceAddress{uint64_t id;uint64_t value(){return id;}};DeviceAddress device_address(uint64_t id){return {id};}};
struct Context{Device device;};
struct Graph {Device* device;bool upload=false;void execute(std::initializer_list<int>){++device->latest;if(upload)++device->uploads;}};
struct External {uint64_t id=0;void set_buffer(uint64_t value){id=value;}};
struct PreImpl;struct PreSlot {CaptureLease source;uint64_t submit=0,generation=0;std::array<uint64_t,1> buffers{};std::unique_ptr<Graph> graph;};
class DaxaFrame {std::shared_ptr<PreImpl> owner_;std::shared_ptr<PreSlot> slot_;public:DaxaFrame()=default;DaxaFrame(std::shared_ptr<PreImpl> o,std::shared_ptr<PreSlot> s):owner_(o),slot_(s){}explicit operator bool()const{return bool(slot_);}CaptureLease source()const{return slot_->source;}std::shared_ptr<const Maps> calibration()const;uint64_t context_id()const;};
struct PreImpl {std::shared_ptr<Context> context;std::shared_ptr<const Maps> calibration;std::vector<std::shared_ptr<PreSlot>> slots;uint64_t retired_sources=0,next_generation=1;double upload_cpu_ms=0;void require_thread()const{}};
std::shared_ptr<const Maps> DaxaFrame::calibration()const{return owner_->calibration;}uint64_t DaxaFrame::context_id()const{return reinterpret_cast<uintptr_t>(owner_->context.get());}
class DaxaPreprocess {public:std::shared_ptr<PreImpl> impl_;static constexpr unsigned Staging=0;DaxaFrame enqueue_at(CaptureLease,uint64_t);size_t collect_completed_sources_at(uint64_t);};
enum class DaxaDepthStereoSchedule{Sequential,PairedPerPath};struct DaxaDepthConfig{DaxaDepthStereoSchedule stereo_schedule=DaxaDepthStereoSchedule::Sequential;};
enum class DaxaStereoReference{Left,Right};enum class DaxaStereoPlane{Disparity,Confidence,Validity};enum class DaxaPreprocessPlane{LeftSupport,RightSupport};
struct StereoSlot{DaxaFrame input;CaptureLease source;uint64_t submit=0,generation=0;};
struct StereoImpl {std::shared_ptr<Context> context;std::vector<std::shared_ptr<StereoSlot>> slots;DaxaStereoReference reference;void require_thread()const{}};
class DaxaStereo {public:std::shared_ptr<StereoImpl> impl_;class Frame {std::shared_ptr<StereoImpl> owner_;std::shared_ptr<StereoSlot> slot_;public:Frame()=default;Frame(std::shared_ptr<StereoImpl> o,std::shared_ptr<StereoSlot> s):owner_(o),slot_(s){}DaxaStereoReference reference()const{return owner_->reference;}CaptureLease source()const{return slot_->source;}uint64_t context_id()const{return reinterpret_cast<uintptr_t>(owner_->context.get());}std::shared_ptr<const Maps> calibration_owner()const{return slot_->input.calibration();}const std::string& calibration_identity()const{return slot_->input.calibration()->rectified_identity;}};
 Frame process_at(const DaxaFrame* pre,uint64_t oldest){for(auto& s:impl_->slots)if(detail::available_slot(s.use_count(),s->submit,oldest)){s->input=*pre;s->source=pre->source();s->submit=++impl_->context->device.latest;impl_->context->device.collect_garbage();return Frame(impl_,s);}throw std::runtime_error("stereo full");}
 unsigned pair_slot_at(uint64_t)const{throw std::runtime_error("unused sequential probe pair admission");}
 std::array<Frame,2> process_pair_at(const DaxaFrame&,uint64_t,unsigned){throw std::runtime_error("unused sequential probe paired submission");}
 size_t collect_completed_inputs_at(uint64_t);
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
    # Actual production depth bookkeeping, pairing, frame validity and staging copy.
    program += definition(depth,'size_t collect_at(').replace('size_t collect_at(', 'size_t DaxaDepthPipeline::Impl::collect_at(',1)
    for signature in ('void DaxaDepthPipeline::validate_pair(', 'DaxaDepthPipeline::Frame DaxaDepthPipeline::enqueue(', 'void DaxaDepthPipeline::Frame::validate('):
        program += definition(depth,signature)
    program += r'''
}
using namespace stereoforge;
void check(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
int main(){try{
 auto context=std::make_shared<Context>();auto e=std::make_shared<DaxaDepthPipeline::Impl>();e->context=context;
 e->preprocessing=std::make_unique<DaxaPreprocess>();e->preprocessing->impl_=std::make_shared<PreImpl>();auto pre=e->preprocessing->impl_;pre->context=context;pre->calibration=std::make_shared<Maps>();
 std::array<std::vector<uint8_t>,2> staging;for(auto& bytes:staging)bytes.resize(1280*2*720*2);
 for(unsigned i=0;i<2;++i){auto s=std::make_shared<PreSlot>();s->buffers[0]=reinterpret_cast<uintptr_t>(staging[i].data());s->graph=std::make_unique<Graph>(Graph{&context->device,true});pre->slots.push_back(std::move(s));}
 for(unsigned eye=0;eye<2;++eye){auto engine=std::make_unique<DaxaStereo>();engine->impl_=std::make_shared<StereoImpl>();engine->impl_->context=context;engine->impl_->reference=eye?DaxaStereoReference::Right:DaxaStereoReference::Left;for(unsigned i=0;i<2;++i)engine->impl_->slots.push_back(std::make_shared<StereoSlot>());(eye?e->right:e->left)=std::move(engine);}
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
 rgb0={};auto third=pipeline.enqueue(source);third.validate();check(third.slot_->data.source_id==uint64_t(INT64_MAX),"ID truncated");
 third={};auto retired=e->collect_at(context->device.oldest);check(retired==0&&bool(e->slots[0]->preprocessing),"inflight producer retired");complete();rgb1={};e->collect_at(context->device.oldest);
 for(auto& s:pre->slots)check(!s->source,"completed unleased preprocessing keeps capture metadata pinned");
 for(auto* engine:{e->left.get(),e->right.get()})for(auto& s:engine->impl_->slots)check(!s->source&&!s->input,"completed unleased stereo keeps capture metadata pinned");
 auto bad=std::make_shared<CaptureFrame>();bad->frame_id=uint64_t(INT64_MAX)+1;auto snapshots=context->device.snapshots;uploads=context->device.uploads;caught=false;try{pipeline.enqueue(bad);}catch(const std::invalid_argument&){caught=true;}
 check(caught&&context->device.snapshots==snapshots&&context->device.uploads==uploads,"invalid signed ID reached device/upload");
 // Warm state and fixed allocation; extract actual depth/retirement/staging methods.
 constexpr unsigned frames=256;allocations=0;auto before_snapshots=context->device.snapshots;tracking=true;auto start=std::chrono::steady_clock::now();
 for(unsigned i=0;i<frames;++i){auto out=pipeline.enqueue(source);out.validate();complete();out={};}
 e->collect_at(context->device.oldest);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();tracking=false;
 check(allocations==0,"application heap allocation in warmed successful frame path");check(context->device.snapshots-before_snapshots==frames,"more than one application admission snapshot");
 check(staging[0][0]==173&&staging[0].back()==173,"actual packed staging copy missing");
 std::cout<<"{\"frames\":"<<frames<<",\"packed_bytes_per_frame\":"<<owned->packed_yuyv.size()<<",\"mean_cpu_ms\":"<<ms/frames<<",\"application_allocations\":"<<allocations<<",\"admission_snapshots\":"<<context->device.snapshots-before_snapshots<<",\"full_output_and_rgb_backpressure_before_upload\":true,\"inflight_and_retained_producers_preserved\":true,\"scope\":\"actual_depth_and_pre_bookkeeping_fake_GPU_and_stereo_submission\"}\n";
 return 0;
}catch(const std::exception& ex){tracking=false;std::cerr<<ex.what()<<'\n';return 1;}}
'''
    directory=tmp_path_factory.mktemp('daxa-depth-bookkeeping')
    code=directory/'probe.cpp';code.write_text(program);binary=directory/'probe'
    build=subprocess.run([compiler,'-std=c++20','-O3',str(code),'-o',str(binary)],capture_output=True,text=True,timeout=60)
    assert build.returncode==0,build.stderr
    return binary


def test_actual_depth_admission_retirement_staging_have_zero_warm_allocations(depth_bookkeeping_probe):
    result=subprocess.run([str(depth_bookkeeping_probe)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    row=json.loads(result.stdout)
    print('Actual depth/pre bookkeeping CPU-only probe:',json.dumps(row,sort_keys=True))
    assert row['frames']==256 and row['packed_bytes_per_frame']==3686400
    assert row['application_allocations']==0 and row['admission_snapshots']==256
    assert row['full_output_and_rgb_backpressure_before_upload'] and row['inflight_and_retained_producers_preserved']
    assert 0<row['mean_cpu_ms']<1000


def check_depth_timing(source):
    timing=body(source,'DaxaDepthTiming DaxaDepthPipeline::Frame::timings(')
    assert timing.count('daxa_timeline_query_pool_query_results(')==1
    assert 'std::array<uint64_t,8>' in timing and 'slot_->queries.get(),0,4,result.data()' in timing
    assert not re.search(r'get_query_results\(|wait_idle\(|wait_on_submit\(|std::vector|create_',timing)
    assert 'result[6]-out.preprocessing.start_tick' in timing, 'combined timing must span the actual queue rather than adding stage summaries'
    assert 'out.preprocessing.timestamp_bits==owner_->timestamp_bits' in timing
    assert 'out.preprocessing.timestamp_period==double(owner_->timestamp_period)' in timing
    assert 'elapsed_ns<wrap_ns' in timing, 'reject ambiguous complete timestamp wrap intervals'
    getter=body(source,'MetricCalibration DaxaDepthPipeline::Frame::metric_calibration(')
    assert 'return owner_->metric;' in getter and 'metric_calibration()' not in getter, 'immutable metric values are validated only during setup'


def test_combined_gpu_timestamp_span_includes_inter_graph_gaps_without_new_query():
    check_depth_timing((ROOT/'src/daxa_depth.cpp').read_text())


def test_timing_guard_rejects_fabricated_sum_of_stage_percentiles():
    source=(ROOT/'src/daxa_depth.cpp').read_text()
    with pytest.raises(AssertionError,match='actual queue'):
        check_depth_timing(source.replace('result[6]-out.preprocessing.start_tick','result[6]-result[0]'))


def test_benchmark_uses_actual_combined_pipeline_and_accounts_for_collection_and_query_cost():
    app=(ROOT/'apps/daxa_depth_benchmark.cpp').read_text()
    timed=body(app,'for(uint32_t i=0;i<frames+warmup;++i)')
    assert timed.index('auto a=Clock::now()')<timed.index('output={}')<timed.index('pipeline->enqueue(source)')<timed.index('output.wait()')<timed.index('output.timings()')<timed.index('auto d=Clock::now()')
    assert 'std::chrono::duration<double,std::milli>(d-a).count()' in timed
    assert 'std::chrono::duration<double,std::milli>(d-c).count()' in timed
    for forbidden in ('readback(', 'readback_images(', 'readback_stereo(', 'daxa_depth_reference(', 'daxa_stereo_reference(', 'metric_calibration(', 'std::vector', 'push_back(', 'resize(', 'json'):
        assert forbidden not in timed,'diagnostics/geometry/image work is outside every timed sample'
    assert app.index('auto diagnostic_begin=Clock::now()')>app.index(timed)+len(timed)
    release=app.index('context->close()')
    assert app.index('pipeline.reset()')<release<app.index('context->validation_errors==0&&context->validation_warnings==0')
    assert 'repeated_retained_record_dual_reference_LR_metric_no_RGBD_synthesis' in app
    assert 'metric_compound_float_reference_checked",false' in app and 'stereo_compound_float_reference_checked",false' in app


def check_lr_integer_indexing(ops):
    target=body(ops,'bool observation_target(')
    assert not re.search(r'floor\s*\(\s*float\s*\(x\)',target), 'do not round large pixel coordinates plus fractional float disparity in FP32'
    compact=re.sub(r'\s+','',target)
    assert 'floor(d)' in target and 'right?d>=threshold:d>threshold' in compact and 'floatthreshold=float(base)+0.5f' in compact, 'indexing uses exact half-integer comparisons with asymmetric half ties'
    assert 'abs(d)>float(width+1)' in target, 'finite/range admission must guard integer conversion'
    assert target.index('abs(d)>float(width+1)')<target.index('floor(d)'), 'range admission precedes the safe integer conversion'
    assert target.index('isfinite(d)') < target.index('floor(d)'), 'finite/range admission precedes integer conversion'


def test_lr_indexing_preserves_float32_nextafter_values_at_large_pixel_coordinates():
    check_lr_integer_indexing((ROOT/'python/stereoforge/shaders/ops/stereo_observation.slang').read_text())


@pytest.fixture(scope='module')
def actual_lr_index_probe(tmp_path_factory):
    compiler=shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable for extracted pure shader index probe')
    ops=(ROOT/'python/stereoforge/shaders/ops/stereo_observation.slang').read_text()
    helper=definition(ops,'bool observation_target(').replace('out uint target','uint& target')
    cpp=r'''
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
using uint=uint32_t;using std::isfinite;using std::floor;
'''+helper+r'''
int main(){uint64_t cases=0;for(uint width:{1u,9u,37u,1001u,32766u})for(uint x:{0u,width/2,width-1})for(bool right:{false,true}){
 for(int b=-257;b<=257;++b){float half=float(b)+.5f;for(float d:{std::nextafter(half,-INFINITY),half,std::nextafter(half,INFINITY)}){
  double exact=std::floor(double(x)+(right?double(d):-double(d))+.5);bool expected=exact>=0&&exact<double(width);uint target=0;bool actual=observation_target(width,x,d,right,target);++cases;
  if(actual!=expected||(actual&&target!=uint(exact))){std::cerr<<"width="<<width<<" x="<<x<<" d="<<std::hexfloat<<d<<" right="<<right<<" expected="<<exact<<" actual="<<actual<<":"<<target<<'\n';return 1;}
 }}
 for(float d:{NAN,INFINITY,-INFINITY,std::numeric_limits<float>::max(),-std::numeric_limits<float>::max()}){uint target=0;if(observation_target(width,x,d,right,target))return 2;}
}std::cout<<"{\"exact_double_oracle_cases\":"<<cases<<",\"scope\":\"actual_shader_index_helper_CPU_only\"}\n";}
'''
    directory=tmp_path_factory.mktemp('daxa-depth-index');code=directory/'probe.cpp';code.write_text(cpp);binary=directory/'probe'
    result=subprocess.run([compiler,'-std=c++20','-O3',str(code),'-o',str(binary)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    return binary


def test_actual_shader_lr_index_helper_matches_independent_double_oracle(actual_lr_index_probe):
    result=subprocess.run([str(actual_lr_index_probe)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    row=json.loads(result.stdout)
    print('Actual LR indexing CPU-only probe:',json.dumps(row,sort_keys=True))
    assert row['exact_double_oracle_cases']==46350


@pytest.mark.parametrize('old,new', [('right?d>=threshold:d>threshold','right?d>threshold:d>=threshold'), ('abs(d)>float(width+1)','false'), ('float threshold=float(base)+0.5f','float threshold=d-floor(d)')])
def test_index_guard_detects_swapped_half_ties_unbounded_cast_or_lost_threshold(old,new):
    ops=(ROOT/'python/stereoforge/shaders/ops/stereo_observation.slang').read_text()
    assert old in ops
    with pytest.raises(AssertionError):
        check_lr_integer_indexing(ops.replace(old,new,1))


def test_scalar_index_oracle_uses_double_coordinates_independently_of_shader_delta():
    scalar=body((ROOT/'src/daxa_depth.cpp').read_text(),'bool target(')
    assert 'double(x)' in scalar and 'double(d)' in scalar and 'std::floor(' in scalar
    assert 'threshold' not in scalar and 'float value' not in scalar


def check_combined_diagnostics(source, app):
    measured=body(app,'for(uint32_t i=0;i<frames+warmup;++i)')
    diagnostic=body(app,'if(reference||options.contains("--dump-prefix"))')
    assert app.index(diagnostic)>app.index(measured)+len(measured)
    checked=body(diagnostic,'if(reference)')
    assert 'pipeline->check_stereo_integer(output,expected)' in checked
    assert checked.index('check_stereo_integer(')<checked.index(';checked=true;')
    assert 'bool checked=false' in app
    for key in ('both_stereo_integer_reference_checked','both_stereo_packed_penalties_reference_checked'):
        assert '"'+key+'",checked' in app
    integer=body(source,'void DaxaDepthPipeline::check_stereo_integer(')
    assert 'for(unsigned eye=0;eye<2;++eye)' in integer
    assert 'images.planes[4],images.planes[5]' in integer, 'both CPU oracles use independent expected OpenCV luma'
    assert 'DaxaStereoReference::Right:DaxaStereoReference::Left' in integer
    for pair in ('actual.census==scalar.census','actual.census_valid==scalar.census_valid','actual.costs==scalar.costs','actual.sums==scalar.sums','actual.validity==scalar.validity','actual.packed_penalties==packed'):
        assert pair in integer
    assert 'scalar.sums[index]-base' in integer and 'impl_->config.stereo.paths*scalar.costs[index]' in integer, 'direct actual packed words compare to independently packed scalar residuals'
    assert 'actual.disparity' not in integer and 'actual.confidence' not in integer
    assert 'auto final_source=output.source()' in app and 'metadata(*final_source)' in app


def test_both_stereo_integer_and_direct_word_proof_is_after_timing_and_uses_expected_luma():
    check_combined_diagnostics((ROOT/'src/daxa_depth.cpp').read_text(),(ROOT/'apps/daxa_depth_benchmark.cpp').read_text())


def test_diagnostic_guard_catches_reconstructed_gpu_volume_pretending_to_prove_packed_words():
    source=(ROOT/'src/daxa_depth.cpp').read_text()
    with pytest.raises(AssertionError,match='independently packed scalar'):
        check_combined_diagnostics(source.replace('scalar.sums[index]-base','actual.sums[index]-base'),(ROOT/'apps/daxa_depth_benchmark.cpp').read_text())


def test_diagnostic_guard_catches_source_metadata_echo_instead_of_output_association():
    app=(ROOT/'apps/daxa_depth_benchmark.cpp').read_text()
    with pytest.raises(AssertionError):
        check_combined_diagnostics((ROOT/'src/daxa_depth.cpp').read_text(),app.replace('auto final_source=output.source()','auto final_source=source'))


def test_depth_lease_guard_rejects_removing_shared_schedule_admission_restriction():
    source = (ROOT/'src/daxa_depth.cpp').read_text()
    restriction = 'owner_->config.stereo_schedule==DaxaDepthStereoSchedule::SharedCensusSequential'
    assert restriction in source
    with pytest.raises(AssertionError, match='aligned shared-schedule'):
        check_depth_submission(source.replace(restriction, 'true', 1), (ROOT/'include/stereoforge/daxa_depth.hpp').read_text())


def test_depth_lease_guard_rejects_another_hidden_producer_escape():
    source = (ROOT/'src/daxa_depth.cpp').read_text()
    header = (ROOT/'include/stereoforge/daxa_depth.hpp').read_text() + '\nDaxaStereo::Frame hidden_producer();\n'
    with pytest.raises(AssertionError, match='only the explicit'):
        check_depth_submission(source, header)
