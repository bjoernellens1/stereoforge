"""Actual-source guided Refinement CPU bookkeeping, with a mocked GPU only.

These measurements cover application allocation/control cost, not Daxa driver
allocation or GPU latency. Production methods, common lease/Depth adapters and
inclusive completion helpers are extracted on every compile.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path
import pytest
from test_daxa_depth_hotpath import definition
from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/daxa_refinement.cpp'


def check_hotpath(source):
    signatures = ('DaxaRefinement::Frame DaxaRefinement::process(',
                  'bool DaxaRefinement::can_process(', 'size_t DaxaRefinement::collect(',
                  'size_t collect_at(', 'std::array<daxa::BufferId,5> input_buffers(',
                  'DaxaObservationFrame DaxaRefinement::Frame::observations(',
                  '.executes([this,raw,address,pass,count]')
    hot = '\n'.join(body(source, s) for s in signatures)
    assert not re.search(r'\b(?:new|make_shared|make_unique|malloc|calloc|realloc|resize|reserve|push_back|emplace_back|wait\w*|readback\w*|create_\w+|destroy_\w+|register_buffer|add_task|memcpy|memmove|sha256|strict_json|validate_daxa_refinement_geometry|validate_daxa_rectified_input_geometry)\s*(?:<[^;()]*>)?\s*\(|\b(?:TaskGraph|vector|ifstream|ofstream)\b', hot)
    assert not re.search(r'graph(?:->|\.)complete\s*\(', hot)
    assert not re.search(r'\b(?:new|delete)\b', hot)
    process = body(source, signatures[0])
    assert process.count('oldest_pending_submit_index(') == 1
    assert process.index('input.observations()') < process.index('oldest_pending_submit_index(') < process.index('graph->execute(')
    assert set(re.findall(r'for\(([^)]*)\)', process)) == {'auto& s:e.slots', 'unsigned p=0;p<5;++p'}
    assert 'while(' not in hot
    assert 'detail::available_slot(s.use_count(),s->submit,boundary)' in process
    assert 'catch(...){s->submit=d.latest_queue_submit_index(daxa::QUEUE_MAIN);throw;}' in process
    assert process.index('s->input=input') < process.index('graph->execute(')
    collection = body(source, 'size_t DaxaRefinement::collect(')
    assert collection.count('oldest_pending_submit_index(') == collection.count('collect_garbage(') == 1
    retirement = body(source, 'size_t collect_at(')
    assert 'detail::available_slot(s.use_count(),s->submit,boundary)' in retirement and 's->input={}' in retirement and 's->full_input={}' in retirement
    adapter = body(source, signatures[5])
    assert 'std::array<daxa::BufferId,6>' in adapter and 'std::array<DaxaObservationImageView,4>' in adapter
    assert '&slot_->generation,generation_,slot_->submit' in adapter
    assert 'slot_->input.observations()' in adapter
    assert 'HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE' in source
    assert 'graph.submit({});graph.complete({});' in source
    canonical = body(source, 'DaxaRefinement::Frame DaxaRefinement::process(DaxaDepthPipeline::Frame const& input,DaxaFrame const& full)')
    assert not re.search(r'\b(?:new|make_shared|make_unique|resize|reserve|wait\w*|readback\w*|memcpy|sha256|validate_daxa_rectified_input_geometry)\s*(?:<[^;()]*>)?\s*\(|\bvector\b', canonical)
    assert canonical.count('oldest_pending_submit_index(') == 1
    assert canonical.index('source==full.source()') < canonical.index('oldest_pending_submit_index(')
    assert canonical.index('s->input=input') < canonical.index('graph->execute(')
    assert canonical.index('s->full_input=full') < canonical.index('graph->execute(')
    assert 'full.calibration()==e.maps' in canonical
    assert 'catch(...){s->submit=d.latest_queue_submit_index(daxa::QUEUE_MAIN);throw;}' in canonical


def test_no_per_frame_alloc_wait_readback_matrix_validation():
    source = SOURCE.read_text()
    check_hotpath(source)
    marker = 'auto& e=*impl_;e.require_thread();auto observation=input.observations();'
    for operation in ('input.wait();', 'readback(input);', 'std::make_shared<int>();',
                      'std::vector<float> image(1000);', 'graph.complete({});',
                      'validate_daxa_refinement_geometry(a,b);', 'for(unsigned pixel=0;pixel<width*height;++pixel){}'):
        assert marker in source
        with pytest.raises(AssertionError):
            check_hotpath(source.replace(marker, marker + operation, 1))
    for operation in ('std::make_shared<int>();', 'ti.recorder.wait_idle();'):
        marker = '.executes([this,raw,address,pass,count](daxa::TaskInterface ti){'
        with pytest.raises(AssertionError):
            check_hotpath(source.replace(marker, marker + operation, 1))


def build_probe(tmp_path, mutate=lambda s: s):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    source = mutate(SOURCE.read_text())
    frame = (ROOT / 'include/stereoforge/daxa_frame.hpp').read_text()
    common_header = (ROOT / 'include/stereoforge/daxa_observation.hpp').read_text()
    common_source = (ROOT / 'src/daxa_observation.cpp').read_text()
    header = (ROOT / 'include/stereoforge/daxa_refinement.hpp').read_text()
    enums = '\n'.join(re.search(r'enum class ' + name + r'[^;]+;', text)[0] for name, text in (
        ('DaxaPreprocessPlane', frame), ('DaxaPlaneFormat', frame),
        ('DaxaDepthPlane', (ROOT / 'include/stereoforge/daxa_depth.hpp').read_text()),
        ('DaxaRightReferenceMode', (ROOT / 'include/stereoforge/daxa_stereo.hpp').read_text()),
        ('DaxaStereoEstimator', (ROOT / 'include/stereoforge/daxa_stereo.hpp').read_text())))
    program = r'''
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
std::atomic<bool> track=false;unsigned allocations=0;
void* operator new(size_t n){if(track)++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}void operator delete(void* p,size_t)noexcept{std::free(p);}
namespace daxa {
constexpr int QUEUE_MAIN=0;using TimelineQueryPool=int;
struct BufferId{uint64_t id=0;bool is_empty()const{return !id;}bool operator==(BufferId const&)const=default;};
struct ExternalTaskBuffer{BufferId id;unsigned updates=0;void set_buffer(BufferId v){id=v;++updates;}};
struct TaskGraph;
}
namespace stereoforge {
struct Device{uint64_t latest=0,boundary=0,polls=0,waits=0,gc=0;void* get(){return this;}uint64_t oldest_pending_submit_index(){++polls;return boundary;}uint64_t latest_queue_submit_index(int){return latest;}void collect_garbage(){++gc;}struct Address{uint64_t id;uint64_t value(){return id;}};Address device_address(daxa::BufferId id){return {id.id};}struct Host{uint64_t id;void* value(){return reinterpret_cast<void*>(id);}};Host buffer_host_address(daxa::BufferId id){return {id.id};}struct Wait{int queue;uint64_t queue_submit_index,timeout;};void wait_on_submit(Wait){++waits;}};
struct DaxaFoundation{Device device;};
struct DaxaRectification{unsigned width=8,height=6,eye_width=20,source_height=16,stride_bytes=160;std::string original_identity,rectified_identity;};
enum class CaptureSourceKind{PackedYuyv,CanonicalRgbMetadata};struct CanonicalRgbSource;
struct CaptureFrame{CaptureSourceKind source_kind=CaptureSourceKind::PackedYuyv;std::shared_ptr<const CanonicalRgbSource> canonical_rgb;std::vector<uint8_t> packed_yuyv;uint64_t frame_id=7;unsigned source_width=40,height=16,stride_bytes=160;std::string calibration_identity,device_identity,host_clock_domain;};using CaptureLease=std::shared_ptr<const CaptureFrame>;
namespace detail{
'''
    program += definition(frame, 'inline bool completed_submission(') + definition(frame, 'inline bool available_slot(')
    program += '}\n' + enums + '}\n' + common_header[common_header.index('namespace stereoforge {'):]
    program += r'''
namespace stereoforge {
struct Pre{std::shared_ptr<const DaxaRectification> maps;CaptureLease src;auto calibration()const{return maps;}auto source()const{return src;}struct Plane{daxa::BufferId id;auto buffer()const{return id;}};Plane plane(DaxaPreprocessPlane p)const{return {{100+unsigned(p)}};}};
class DaxaDepthPipeline{public:struct Owner{std::shared_ptr<DaxaFoundation> context;};struct Slot{std::array<daxa::BufferId,9> outputs{};uint64_t generation=1,submit=1;Pre preprocessing;};class Frame{public:std::shared_ptr<Owner> owner_;std::shared_ptr<Slot> slot_;uint64_t generation_=1;void validate()const{if(!slot_||generation_!=slot_->generation)throw std::invalid_argument("stale input");}explicit operator bool()const{return bool(slot_);}DaxaObservationFrame observations()const;CaptureLease source()const{return slot_->preprocessing.src;}DaxaRightReferenceMode right_reference_mode()const{return DaxaRightReferenceMode::IndependentSGM;}daxa::BufferId buffer(DaxaDepthPlane p)const{return slot_->outputs[unsigned(p)];}struct Image{daxa::BufferId id;auto buffer()const{return id;}};Image image(DaxaPreprocessPlane p)const{return {slot_->preprocessing.plane(p).buffer()};}void wait()const{++owner_->context->device.waits;}};};
}
'''
    program += r'''
namespace stereoforge {
class DaxaFrame{public:
 std::shared_ptr<DaxaFoundation> context;std::shared_ptr<const DaxaRectification> maps;CaptureLease src;std::shared_ptr<int> lifetime;std::array<daxa::BufferId,8> buffers{};
 explicit operator bool()const{return bool(src);}auto calibration()const{return maps;}auto source()const{return src;}
 uint64_t context_id()const{return reinterpret_cast<uintptr_t>(context->device.get());}
 struct Image{daxa::BufferId id;auto buffer()const{return id;}};
 Image plane(DaxaPreprocessPlane p)const{return {buffers[unsigned(p)]};}
};
}
'''
    canonical_header = (ROOT / 'include/stereoforge/canonical_rgb.hpp').read_text()
    program += canonical_header[canonical_header.index('namespace stereoforge {'):canonical_header.index('inline void validate_canonical_rgb_source(')] + '}\n'
    program += '\nnamespace stereoforge {\n' + definition(canonical_header, 'inline bool canonical_rgb_grid_association(') + '\n}\n'
    program += re.sub(r'\brequire\b', 'observation_require', common_source[common_source.index('namespace stereoforge {'):])
    program += '\nnamespace stereoforge {\n' + definition((ROOT / 'src/daxa_depth.cpp').read_text(), 'DaxaObservationFrame DaxaDepthPipeline::Frame::observations(') + '\n}\n'
    program += header[header.index('namespace stereoforge {'):].replace('class DaxaRefinement {', 'class DaxaRefinement { public:', 1)
    program += r'''
namespace daxa {struct TaskGraph{stereoforge::Device* device=nullptr;unsigned executions=0;int fail=0;void execute(std::initializer_list<int>){++executions;if(fail==1)throw std::runtime_error("before submit");++device->latest;if(fail==2)throw std::runtime_error("partial submit");}};}
namespace stereoforge {
'''
    program += definition(source, 'void require(') + '\nconstexpr unsigned planes=30;\n'
    program += definition(source, 'constexpr unsigned element_bytes(') + '\n' + definition(source, 'struct Parameters') + ';\n'
    program += definition(source, 'struct DaxaRefinement::Slot') + ';\n'
    program += r'''
struct DaxaRefinement::Impl{
 DaxaRefinementConfig config{};bool canonical=false;CaptureSourceKind full_source_kind=CaptureSourceKind::PackedYuyv;
 std::shared_ptr<DaxaFoundation> context;std::shared_ptr<const DaxaRectification> coarse_maps,maps;
 std::vector<std::shared_ptr<Slot>> slots;uint64_t next_generation=1,retired=0;
 const std::thread::id thread=std::this_thread::get_id();
'''
    for signature in ('void require_thread(', 'size_t collect_at(', 'std::array<daxa::BufferId,5> input_buffers(', 'std::array<daxa::BufferId,6> full_buffers('):
        program += definition(source, signature)
    program += '};\n'
    for signature in ('DaxaRefinement::Frame::Frame(', 'DaxaRefinement::Frame& DaxaRefinement::Frame::operator=(',
                      'void DaxaRefinement::Frame::validate(', 'DaxaRefinement::Frame DaxaRefinement::process(',
                      'bool DaxaRefinement::can_process(', 'size_t DaxaRefinement::collect(',
                      'bool DaxaRefinement::Frame::ready(', 'uint64_t DaxaRefinement::Frame::generation(',
                      'uint64_t DaxaRefinement::Frame::submit_ticket(', 'uint32_t DaxaRefinement::Frame::width(',
                      'uint32_t DaxaRefinement::Frame::height(', 'daxa::BufferId DaxaRefinement::Frame::buffer(',
                      'size_t DaxaRefinement::Frame::size_bytes(', 'DaxaObservationFrame DaxaRefinement::Frame::observations('):
        program += definition(source, signature)
    program += definition(source, 'DaxaRefinement::Frame DaxaRefinement::process(DaxaDepthPipeline::Frame const& input,DaxaFrame const& full)')
    # Constructor setup is the only mock boundary: no Vulkan, maps or shader setup.
    program += r'''
DaxaRefinement::DaxaRefinement(std::shared_ptr<DaxaFoundation>,DaxaRectification,DaxaDepthPipeline::Frame const&,DaxaRefinementConfig){}
DaxaRefinement::DaxaRefinement(std::shared_ptr<DaxaFoundation>,DaxaDepthPipeline::Frame const&,DaxaFrame const&,DaxaRefinementConfig){}
}
using namespace stereoforge;
void check(bool value,char const* message){if(!value)throw std::runtime_error(message);}
int main(int argc,char** argv){try{
 size_t bytes=argc>1?std::stoul(argv[1]):65536;auto context=std::make_shared<DaxaFoundation>();auto engine=std::make_shared<DaxaRefinement::Impl>();engine->context=context;
 auto maps=std::make_shared<DaxaRectification>();maps->original_identity=std::string(bytes,'c');engine->coarse_maps=maps;
 auto target=std::make_shared<DaxaRectification>(*maps);target->width*=2;target->height*=2;engine->maps=target;
 std::array<DaxaDepthPipeline::Frame,3> inputs;std::array<std::shared_ptr<CaptureFrame>,3> captures;
 for(unsigned i=0;i<3;++i){captures[i]=std::make_shared<CaptureFrame>();captures[i]->frame_id=i+1;captures[i]->calibration_identity=maps->original_identity;captures[i]->device_identity=std::string(bytes,'d');captures[i]->host_clock_domain=std::string(bytes,'t');auto owner=std::make_shared<DaxaDepthPipeline::Owner>();owner->context=context;auto slot=std::make_shared<DaxaDepthPipeline::Slot>();for(unsigned p=0;p<9;++p)slot->outputs[p]={1000+100*i+p};slot->preprocessing={maps,captures[i]};inputs[i]={owner,slot,1};}
 std::array<Parameters,2> params{};for(unsigned i=0;i<2;++i){auto slot=std::make_shared<DaxaRefinement::Slot>();for(unsigned p=0;p<30;++p)slot->output[p]={2000+100*i+p};slot->parameters={reinterpret_cast<uintptr_t>(&params[i])};slot->graph=std::make_unique<daxa::TaskGraph>();slot->graph->device=&context->device;engine->slots.push_back(std::move(slot));}
 captures[2]->frame_id=uint64_t(INT64_MAX);
 DaxaRefinement stage(context,{},inputs[0]);stage.impl_=engine;
 auto finish=[&]{context->device.boundary=context->device.latest;};
 auto first=stage.process(inputs[0]);check(first.generation()==1&&!first.ready(),"generation/pending readiness");finish();auto second=stage.process(inputs[1]);finish();check(!stage.can_process()&&stage.collect()==0,"external outputs failed backpressure");
 auto common=first.observations();first={};check(!stage.can_process()&&stage.collect()==0,"common lease not retaining producer slot");common={};check(stage.can_process()&&stage.collect()==1,"inclusive equality not reusable");second={};check(stage.collect()==1,"released completed source not retired");
 auto pending=stage.process(inputs[0]);pending={};check(bool(engine->slots[0]->input),"source missing");check(stage.collect()==0,"pending source retired");
 // One slot remains unused: pin its future ticket too to test actual admission.
 engine->slots[1]->submit=context->device.latest;check(!stage.can_process(),"pending ticket ignored");finish();check(stage.can_process()&&stage.collect()==1,"pending source not released on equality");engine->slots[1]->submit=0;
 auto* candidate=engine->slots[0].get();candidate->graph->fail=2;auto previous=context->device.latest;bool failed=false;try{auto out=stage.process(inputs[2]);}catch(std::runtime_error const&){failed=true;}
 check(failed&&context->device.latest==previous+1&&candidate->submit==context->device.latest&&bool(candidate->input),"partial actual ticket/input not pinned");check(stage.collect()==0,"partial source retired before completion");finish();candidate->graph->fail=0;check(stage.collect()==1,"partial source not retired on completion");
 candidate->graph->fail=1;failed=false;previous=context->device.latest;try{auto out=stage.process(inputs[2]);}catch(std::runtime_error const&){failed=true;}check(failed&&candidate->submit==previous,"before-submit failure invented ticket");candidate->graph->fail=0;check(stage.collect()==1,"no-submit source not released");
 auto stale=stage.process(inputs[0]);++engine->slots[0]->generation;bool rejected=false;try{stale.generation();}catch(std::invalid_argument const&){rejected=true;}--engine->slots[0]->generation;check(rejected,"stale generation accepted");finish();stale={};stage.collect();
 auto polls=context->device.polls,gc=context->device.gc,latest=context->device.latest;auto updates=engine->slots[0]->input_nodes[0].updates;constexpr unsigned frames=10000;uint64_t checksum=0;allocations=0;track=true;auto start=std::chrono::steady_clock::now();
 for(unsigned n=0;n<frames;++n){stage.collect();check(stage.can_process(),"preupload admission failed");auto out=stage.process(inputs[n%3]);auto observation=out.observations();auto copied=observation;auto moved=std::move(copied);copied=observation;copied={};check(observation.context()==context&&observation.map_owner()==target&&observation.source()==captures[n%3],"common owner/source association");check(observation.width()==16&&observation.height()==12&&observation.layers()==2,"target observation geometry");check(observation.ancestry().kind()==DaxaObservationAncestryKind::SingleMeasured&&observation.ancestry().capacity()==1,"wrong independent ancestry");for(unsigned p=0;p<6;++p){check(observation.buffer(DaxaObservationPlane(p))==engine->slots[0]->output[p+3],"metric plane binding");checksum+=observation.size_bytes(DaxaObservationPlane(p));}for(unsigned p=0;p<4;++p){auto image=observation.image(DaxaObservationImage(p));check(image.width()==(p<2?20:16)&&image.height()==(p<2?16:12),"RGB extent");auto expected=p<2?100+unsigned(p==0?DaxaPreprocessPlane::OriginalLeft:DaxaPreprocessPlane::OriginalRight):engine->slots[0]->output[p==2?26:29].id;check(image.buffer().id==expected,"RGB buffer association");checksum+=image.buffer().id;}check(params[0].source_id==captures[n%3]->frame_id,"current ID lost");for(unsigned p=0;p<5;++p)check(engine->slots[0]->input_nodes[p].updates==updates+n+1,"unexpected binding update count");finish();}
 double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();track=false;
 check(allocations==0&&context->device.waits==0,"allocation or implicit wait in warmed actual source");check(context->device.polls-polls==3*frames,"admission/process/collect poll count");check(context->device.gc-gc==frames,"explicit collect GC count");check(context->device.latest-latest==frames,"graph reuse count");check(engine->slots.size()==2,"slot registry grew");stage.collect();for(auto const& s:engine->slots)check(!s->input,"source retained after completion");for(auto const& input:inputs)check(input.slot_.use_count()==1,"deep upstream source chain retained");
 // Canonical mode retains both sources, updates six full nodes and rejects
 // unrelated leases/exact owners before submission. No setup/GPU is mocked as evidence.
 engine->canonical=true;DaxaFrame full;full.context=context;full.maps=target;full.src=captures[0];for(unsigned p=0;p<8;++p)full.buffers[p]={4000+p};
 auto canonical=stage.process(inputs[0],full);auto current=context->device.latest;engine->full_source_kind=CaptureSourceKind::CanonicalRgbMetadata;bool kind_bad=false;try{stage.process(inputs[0],full);}catch(std::invalid_argument const&){kind_bad=true;}check(kind_bad&&context->device.latest==current,"wrong full source kind admitted");engine->full_source_kind=CaptureSourceKind::PackedYuyv;auto full_bad=full;full_bad.src=captures[1];bool bad=false;try{stage.process(inputs[0],full_bad);}catch(std::invalid_argument const&){bad=true;}check(bad&&context->device.latest==current,"foreign full source admitted");
 full_bad=full;full_bad.maps=std::make_shared<DaxaRectification>(*target);bad=false;try{stage.process(inputs[0],full_bad);}catch(std::invalid_argument const&){bad=true;}check(bad&&context->device.latest==current,"foreign full owner admitted");
 canonical={};check(bool(engine->slots[0]->full_input)&&stage.collect()==0,"pending full input released");finish();check(stage.collect()==1&&!engine->slots[0]->full_input,"completed full input retained");
 candidate->graph->fail=2;previous=context->device.latest;failed=false;try{stage.process(inputs[0],full);}catch(std::runtime_error const&){failed=true;}check(failed&&candidate->submit==previous+1&&bool(candidate->input)&&bool(candidate->full_input),"partial canonical sources not pinned");check(stage.collect()==0,"partial full source released before equality");finish();candidate->graph->fail=0;check(stage.collect()==1&&!candidate->full_input,"partial canonical source not retired");
 auto canonical_polls=context->device.polls;allocations=0;track=true;for(unsigned n=0;n<10000;++n){stage.collect();check(stage.can_process(),"canonical preupload admission failed");auto out=stage.process(inputs[0],full);auto lease=out.observations();finish();}track=false;check(allocations==0&&context->device.waits==0&&context->device.polls-canonical_polls==30000,"canonical hotpath allocation/wait/poll regression");stage.collect();
 full_bad={};full.lifetime=std::make_shared<int>(7);auto full_lifetime=std::weak_ptr(full.lifetime);auto retained_full=stage.process(inputs[0],full);auto retained_common=retained_full.observations();full={};retained_full={};finish();check(stage.collect()==0&&!full_lifetime.expired(),"common lease failed to retain full producer");retained_common={};check(stage.collect()==1&&full_lifetime.expired(),"full producer leaked after common retirement");
 engine->canonical=false;
 auto held=stage.process(inputs[1]);finish();auto lease=held.observations();auto weak=std::weak_ptr(engine);held={};stage.impl_.reset();engine.reset();check(!weak.expired()&&lease.width()==16&&lease.source()==captures[1],"facade destruction invalidated common lease");lease={};check(weak.expired(),"producer ownership cycle");
 std::cout<<"{\"frames\":"<<frames<<",\"identity_bytes\":"<<bytes<<",\"application_allocations\":"<<allocations<<",\"implicit_waits\":0,\"snapshots\":"<<3*frames<<",\"explicit_collect_gc_calls\":"<<frames<<",\"slots\":2,\"binding_nodes_per_slot\":40,\"mean_mock_host_ms\":"<<ms/frames<<",\"checksum\":"<<checksum<<",\"partial_submit_pins_source\":true,\"common_lease_survives_facade\":true,\"scope\":\"actual production process/admission/retirement/Depth+common adapters; mocked GPU/setup only\"}\n";
 }catch(std::exception const& e){track=false;std::cerr<<e.what()<<'\n';return 1;}}
'''
    code = tmp_path / 'probe.cpp'
    code.write_text(program)
    binary = tmp_path / 'probe'
    built = subprocess.run([compiler, '-std=c++20', '-O3', str(code), '-o', str(binary)], capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    return binary


@pytest.mark.parametrize('identity_bytes', [24, 65536])
def test_actual_process_admission_retirement(tmp_path, identity_bytes):
    run = subprocess.run([str(build_probe(tmp_path)), str(identity_bytes)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    row = json.loads(run.stdout)
    print('Actual Refinement CPU-only probe:', json.dumps(row, sort_keys=True))
    assert row['frames'] == 10000 and row['application_allocations'] == row['implicit_waits'] == 0
    assert row['snapshots'] == 30000 and row['partial_submit_pins_source'] and row['common_lease_survives_facade']


@pytest.mark.parametrize('mutation', ['allocation', 'wait', 'pending', 'external', 'partial_ticket', 'retirement', 'full_retirement', 'full_retention', 'full_source', 'full_owner'])
def test_executing_probe_detects_regressions(tmp_path, mutation):
    def alter(source):
        if mutation in ('allocation', 'wait'):
            marker = 'auto& e=*impl_;e.require_thread();auto observation=input.observations();'
            assert marker in source
            return source.replace(marker, marker + ('auto unexpected=std::make_shared<int>(9);' if mutation == 'allocation' else 'input.wait();'), 1)
        if mutation == 'pending':
            return source.replace('detail::available_slot(s.use_count(),s->submit,boundary)', 'detail::available_slot(s.use_count(),0,boundary)')
        if mutation == 'external':
            return source.replace('detail::available_slot(s.use_count(),s->submit,boundary)', 'detail::available_slot(1,s->submit,boundary)')
        if mutation == 'partial_ticket':
            marker = 'catch(...){s->submit=d.latest_queue_submit_index(daxa::QUEUE_MAIN);throw;}'
            assert marker in source
            return source.replace(marker, 'catch(...){s->submit=0;throw;}')
        if mutation == 'full_retirement':
            return source.replace('s->full_input={}', '')
        if mutation == 'full_retention':
            return source.replace('s->full_input=full;', '')
        if mutation == 'full_source':
            return source.replace('source==full.source()', 'true')
        if mutation == 'full_owner':
            return source.replace('full.calibration()==e.maps', 'true')
        assert 's->input={}' in source
        return source.replace('s->input={}', '')
    run = subprocess.run([str(build_probe(tmp_path, alter))], capture_output=True, text=True, timeout=30)
    assert run.returncode != 0, 'executing actual-source mutation must fail'


def check_benchmark(source):
    loop = body(source, 'for(unsigned i=0;i<frames+warmup;++i)')
    assert not re.search(r'\b(?:new|make_shared|make_unique|malloc|calloc|resize|reserve|push_back|readback|dump|memcpy|memmove|wait_idle|sha256|strict_json)\b', loop)
    assert 'stage->can_process()' in loop
    assert loop.index('stage->collect()') < loop.index('stage->can_process()') < loop.index('depth->enqueue(prepared.source)') < loop.index('stage->process(coarse)')
    assert loop.index('auto b=Clock::now()') < loop.index('output.wait()') < loop.index('auto c=Clock::now()') < loop.index('coarse.timings()')
    assert source.index('v.resize(frames)') < source.index('for(unsigned i=0;i<frames+warmup;++i)')
    assert source.index('stage->readback(output)') > source.index('auto result=prepared.report()')
    assert 'including gaps' in source and 'excludes driver allocator' in source


def test_benchmark_admits_before_upload_and_separates_diagnostics():
    source = (ROOT / 'apps/daxa_refinement_benchmark.cpp').read_text()
    check_benchmark(source)
    marker = 'for(unsigned i=0;i<frames+warmup;++i){'
    for operation in ('stage->readback(output);', 'std::make_shared<int>();', 'context->device.wait_idle();'):
        with pytest.raises(AssertionError):
            check_benchmark(source.replace(marker, marker + operation, 1))
    with pytest.raises(AssertionError):
        check_benchmark(source.replace('stage->can_process()', 'true'))


def test_canonical_pack_exact_copy_offsets(tmp_path):
    source = SOURCE.read_text()
    actual = definition(source, 'struct CanonicalPackCopy') + ';\n' + definition(source, 'constexpr std::array<CanonicalPackCopy,6> canonical_pack_copies(')
    program = tmp_path / 'pack.cpp'
    program.write_text('#include <array>\n#include <cstddef>\n#include <cassert>\n' + actual + r'''
int main(){
 for(size_t pixels: {size_t(1),size_t(35),size_t(1280*720)}){
  auto c=canonical_pack_copies(pixels);const unsigned destinations[]={26,29,27,27,28,28};
  for(unsigned i=0;i<6;++i){assert(c[i].source==i);assert(c[i].output==destinations[i]);assert(c[i].size==pixels*4);assert(c[i].offset==((i==3||i==5)?pixels*4:0));}
 }
}
''')
    binary = tmp_path / 'pack'
    subprocess.run(['c++', '-std=c++20', '-Wall', '-Wextra', '-Werror', str(program), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    assert 'if(!canonical)for(unsigned p=0;p<4;++p){map_buffers' in source
    assert 'for(unsigned pass=canonical?1u:0u;pass<passes;++pass){auto words=shader' in source
    pack = body(source, '.executes([raw,copies,full_views,out]')
    assert 'copy_buffer_to_buffer' in pack and 'dispatch' not in pack
    assert 'dst_offset=copy.offset' in pack and 'size=copy.size' in pack
    assert not re.search(r'\b(?:wait|readback|make_shared|resize|sha256)\b', pack)


@pytest.mark.parametrize('mutation', [None, 'removed', 'wrong_destination'])
def test_canonical_pack_executing_barrier_order(tmp_path, mutation):
    source = SOURCE.read_text()
    pack = body(source, '.executes([raw,copies,full_views,out]')
    barrier = 'ti.recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ});'
    assert pack.count(barrier) == 1
    if mutation == 'removed':
        pack = pack.replace(barrier, '')
    elif mutation == 'wrong_destination':
        pack = pack.replace('dst_access=daxa::AccessConsts::TRANSFER_READ', 'dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ')
    actual = definition(source, 'struct CanonicalPackCopy') + ';\n' + definition(source, 'constexpr std::array<CanonicalPackCopy,6> canonical_pack_copies(')
    program = tmp_path / 'barrier.cpp'
    program.write_text(r'''
#include <array>
#include <cstddef>
#include <cassert>
namespace daxa {
struct AccessConsts {static constexpr unsigned COMPUTE_SHADER_WRITE=1,TRANSFER_READ=2,COMPUTE_SHADER_READ=4;};
enum class PipelineStageFlagBits {TOP_OF_PIPE,TRANSFER};
}
struct Barrier {unsigned src_access,dst_access;};
struct Reset {unsigned query_pool,start_index,count;};
struct Timestamp {unsigned query_pool;daxa::PipelineStageFlagBits pipeline_stage;unsigned query_index;};
struct Copy {unsigned src_buffer,dst_buffer;size_t dst_offset,size;};
struct Recorder {
 unsigned barrier_count=0,copies=0,sequence=0,last_barrier=0;
 std::array<Copy,6> recorded{};
 void reset_timestamps(Reset){++sequence;}
 void write_timestamp(Timestamp){++sequence;}
 void pipeline_barrier(Barrier b){assert(b.src_access==daxa::AccessConsts::COMPUTE_SHADER_WRITE);assert(b.dst_access==daxa::AccessConsts::TRANSFER_READ);assert(copies==0);++barrier_count;last_barrier=++sequence;}
 void copy_buffer_to_buffer(Copy c){assert(barrier_count==1&&last_barrier&&last_barrier<++sequence);assert(copies<6);recorded[copies++]=c;}
};
struct TaskInterface {Recorder& recorder;struct Buffer {unsigned id;};Buffer get(unsigned v){return {v};}};
struct Slot {unsigned queries=7;};
''' + actual + '\nint main(){\n' + r'''
 for(size_t pixels:{size_t(1),size_t(35),size_t(1280*720)}){
  auto copies=canonical_pack_copies(pixels);std::array<unsigned,6> full_views{100,101,102,103,104,105};std::array<unsigned,30> out{};for(unsigned p=0;p<30;++p)out[p]=200+p;
  Slot slot;auto* raw=&slot;constexpr unsigned passes=5;Recorder recorder;TaskInterface ti{recorder};
''' + pack + r'''
  assert(recorder.barrier_count==1&&recorder.copies==6);
  for(unsigned i=0;i<6;++i){auto c=recorder.recorded[i];assert(c.src_buffer==100+i);assert(c.dst_buffer==200+copies[i].output);assert(c.dst_offset==copies[i].offset);assert(c.size==pixels*4);}
 }
}
''')
    binary = tmp_path / 'barrier'
    subprocess.run(['c++', '-std=c++20', '-Wall', '-Wextra', '-Werror', str(program), '-o', str(binary)], check=True)
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert (run.returncode == 0) == (mutation is None), run.stderr
