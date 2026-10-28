"""CPU-only guards/probes for retained calibrated DaxaView submission.

The probe extracts production C++ methods; only GPU/device/graph and DepthFrame
storage are mocked. The common observation lease is probed separately. Host timings exclude GPU/driver work and prove no GPU speed.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_daxa_depth_hotpath import definition
from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]


def check_view_hotpath(source):
    signatures = ('void DaxaView::Impl::validate_frame(', 'void DaxaView::Impl::bind(',
                  'size_t collect_at(', 'size_t DaxaView::collect(', 'bool DaxaView::can_project(',
                  'DaxaView::Frame DaxaView::project(DaxaObservationFrame const&')
    assert body(source, 'DaxaView::Frame DaxaView::project(DaxaDepthPipeline::Frame const&').strip() == 'return project(input.observations());'
    hot = '\n'.join(body(source, signature) for signature in signatures)
    forbidden = r'\b(?:make_shared|make_unique|new|malloc|calloc|realloc|aligned_alloc|allocate|create_\w+|destroy_\w+|wait\w*|readback\w*|inspect|upload|sha256|sha_file|strict_json|metric_calibration|prepare_daxa_view_calibration|memcpy|memmove|resize|reserve|push_back|emplace_back|register_buffer|add_task|submit_commands)\s*(?:<[^;()]*>)?\s*\(|\b(?:vector|ifstream|ofstream|TaskGraph|ExternalTaskBuffer)\b|\.geometry|\b(?:R1|R2|P1|P2)\['
    assert not re.search(forbidden, hot) and not re.search(r'\bnew\b|\b\w*(?:sha256|hash|strict_json)\w*\s*\(|(?:maps|calibration)->validate\s*\(', hot), 'no per-frame allocation, wait, readback, matrix validation, hash/JSON or image copy'
    validate = body(source, 'void DaxaView::Impl::validate_frame(')
    assert 'frame.calibration()==maps' in validate and 'frame.context_id()' in validate
    assert 'frame.width()==config.source_width' in validate and 'frame.layers()==2' in validate
    assert not re.search(r'for\([^:;]*[xy]|while\(', validate), 'input validation is bounded metadata, not a pixel scan'
    bind = body(source, 'void DaxaView::Impl::bind(')
    assert bind.count('buffer_host_address(') == 1 and 'slot.buffers.find("parameters")' in bind, 'only preallocated scalar parameter storage may be mapped'
    assert '.set_buffer(id)' in bind and 'slot.buffers.find(name)' in bind and 'slot.external.find(name)' in bind
    assert not re.search(r'\b(?:slot\.buffers|slot\.external)\s*\[|\.emplace\(|\.insert\(', bind), 'hot bindings must reuse existing nodes'
    assert 'for(' not in bind and 'while(' not in bind, 'binding is a fixed scalar address update'
    assert 'slot.input=frame' in bind and 'slot.source=frame.source()' in bind and 'slot.timestamp=' in bind
    retire = body(source, 'size_t collect_at(')
    assert retire.index('detail::available_slot(slot.use_count(),slot->submit,pending)') < retire.index('slot->input={}')
    assert 'detail::available_slot(slot.use_count(),slot->submit,pending)' in retire and 'slot->source.reset()' in retire
    assert retire.count('for(') == 1 and 'while(' not in retire
    assert 'oldest_pending_submit_index' not in retire
    admission = body(source, 'bool DaxaView::can_project(')
    assert admission.count('for(') == 1 and 'while(' not in admission
    assert admission.count('oldest_pending_submit_index(') == 1 and 'collect_at(pending)' in admission
    project = body(source, 'DaxaView::Frame DaxaView::project(DaxaObservationFrame const&')
    assert project.count('for(') == 1 and 'while(' not in project, 'only bounded slot admission may loop'
    assert project.count('oldest_pending_submit_index(') == 1 and 'collect_at(pending)' in project
    assert project.index('e.validate_frame(input)') < project.index('oldest_pending_submit_index(') < project.index('e.bind(') < project.index('graph->execute(')
    assert 'detail::available_slot(s.use_count(),s->submit,pending)' in project
    assert 's->generation=e.next_generation++' in project and 's->visited==8&&s->finished==8' in project


def test_no_per_frame_alloc_wait_readback_matrix_validation():
    check_view_hotpath((ROOT/'src/daxa_view.cpp').read_text())


@pytest.mark.parametrize('mutation', ['std::make_shared<int>();', 'frame.wait();', 'readback(frame);',
                                    'sha256(bytes);', 'daxa_preprocess_file_sha256(path);', 'maps->validate();',
                                    'prepare_daxa_view_calibration(maps,c,v);',
                                    'std::memcpy(dst,src,n);', 'std::vector<float> pixels(n);',
                                    'for(unsigned pixel=0;pixel<width*height;++pixel){}'])
def test_guard_rejects_costly_work_in_actual_typed_project(mutation):
    source = (ROOT/'src/daxa_view.cpp').read_text()
    marker = 'DaxaView::Frame DaxaView::project(DaxaObservationFrame const& input){'
    assert marker in source
    with pytest.raises(AssertionError):
        check_view_hotpath(source.replace(marker, marker+mutation, 1))


def test_preallocated_graph_and_frame_ownership_reuse_setup_resources():
    source = (ROOT/'src/daxa_view.cpp').read_text()
    slot = body(source, 'struct DaxaView::Slot')
    assert 'DaxaObservationFrame input' in slot and 'CaptureLease source' in slot
    assert 'std::vector<std::shared_ptr<Slot>> slots' in source
    setup = body(source, 'void DaxaView::Impl::setup_slots(')
    assert 'slots.push_back(std::make_shared<Slot>())' in setup
    assert 'slot.external.emplace(' in setup and 'graph.register_buffer(' in setup and 'graph.complete({})' in setup
    captures = re.findall(r'\.executes\(\[([^]]*)\]', setup)
    assert captures and all(not re.search(r'\b(?:prototype|frame|input)\b', capture) for capture in captures)
    constructor = body(source, 'DaxaView::DaxaView(GeometryConfig c,VirtualView const& v,std::shared_ptr<DaxaFoundation> context,DaxaObservationFrame const&')
    assert 'e.maps=std::move(maps)' in constructor and 'e.setup_slots(' in constructor
    assert not re.search(r'\b(?:slot\.input|slot->input)\s*=\s*prototype', constructor), 'setup cannot retain the constructor prototype'


def check_fixture_boundary(benchmark, fixture, common):
    measured = body(benchmark, 'for(uint32_t i=0;i<frames+warmup;++i)')
    assert measured.index('view->can_project()') < measured.index('pipeline->enqueue(') < measured.index('view->project(input)')
    assert 'view->collect()' not in measured, 'one view admission snapshot precedes upstream upload'
    assert 'readback(' not in measured and 'dump(' not in measured
    assert not re.search(r'Observations|replace_once|observations-prefix|copy_buffer_to_buffer|observations_overridden_for_fixture.*true', benchmark)
    assert 'report(prepared,false)' in benchmark
    assert 'replace_once(input,*context)' in fixture and 'report(prepared,true)' in fixture
    assert fixture.index('observations.replace_once(') < fixture.index('std::make_unique<DaxaView>')
    assert 'replacement_submit' in fixture and 'wait_on_submit(' in fixture
    assert not re.search(r'for\([^;]*;[^;]*(?:frames|warmup)|performance_qualified.*true|gpu_chain_queue_span_ms', fixture)
    assert '"observations_overridden_for_fixture",overridden' in common and '"measured_stereo_output",!overridden' in common
    assert '"performance_qualified",false' in common and 'synthetic_known_geometry' in common


def test_synthetic_uploads_are_separate_from_actual_resident_chain_timing():
    check_fixture_boundary((ROOT/'apps/daxa_view_depth_benchmark.cpp').read_text(),
                           (ROOT/'apps/daxa_view_depth_fixture.cpp').read_text(),
                           (ROOT/'apps/daxa_view_depth_common.hpp').read_text())


def build_view_probe(tmp_path_factory, transform=None):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable for extracted View bookkeeping probe')
    source = (ROOT/'src/daxa_view.cpp').read_text()
    if transform:
        source = transform(source)
    program = r'''
#include <stereoforge/daxa_view_parameters.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
std::atomic<bool> tracking=false;std::atomic<unsigned> allocations=0;
void* operator new(std::size_t n){if(tracking)++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}void operator delete(void* p)noexcept{std::free(p);}void operator delete[](void* p)noexcept{std::free(p);}void operator delete(void* p,std::size_t)noexcept{std::free(p);}void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
namespace stereoforge {
void require(bool b,const char* m){if(!b)throw std::invalid_argument(m);}
namespace detail{
SHARED_COMPLETION_HELPERS
}
namespace daxa {using BufferId=uint64_t;constexpr int QUEUE_MAIN=0;}
struct CaptureFrame{uint64_t frame_id=0,host_ingress_timestamp_ns=0;};using CaptureLease=std::shared_ptr<const CaptureFrame>;
struct Maps{};
struct Device{uint64_t oldest=0,latest=0,snapshots=0,gc=0;void collect_garbage(){++gc;}void* get(){return this;}uint64_t oldest_pending_submit_index(){++snapshots;return oldest;}uint64_t latest_queue_submit_index(int){return latest;}struct Host{uint64_t id;void* value(){return reinterpret_cast<void*>(id);}};Host buffer_host_address(uint64_t id){return {id};}struct Address{uint64_t id;uint64_t value(){return id;}};Address device_address(uint64_t id){return {id};}};
struct Context{Device device;};
enum class DaxaObservationPlane{Depth,Confidence,Validity,Provenance,SourceIds,AgeNs};
enum class DaxaObservationImage{OriginalLeft,OriginalRight,RectifiedLeft,RectifiedRight};
struct Input{std::shared_ptr<Context> context;std::shared_ptr<const Maps> maps;CaptureLease source;uint64_t marker=0;unsigned width=1280,height=720;};
class DaxaObservationFrame{public:std::shared_ptr<Context> owner;std::shared_ptr<Input> input;uint64_t generation=1;explicit operator bool()const{return bool(input);}uint64_t context_id()const{return reinterpret_cast<uintptr_t>(input->context->device.get());}auto calibration()const{return input->maps;}unsigned width()const{return input->width;}unsigned height()const{return input->height;}unsigned layers()const{return 2;}size_t size_bytes(DaxaObservationPlane p)const{return 2ull*width()*height()*(unsigned(p)>=4?8:4);}CaptureLease source()const{return input->source;}uint64_t buffer(DaxaObservationPlane p)const{return input->marker+unsigned(p)+1;}struct Image{uint64_t id;uint64_t buffer()const{return id;}};Image image(DaxaObservationImage p)const{return {input->marker+unsigned(p)+20};}};
struct External{uint64_t id=0;void set_buffer(uint64_t value){id=value;}};
struct Graph{Device* device;unsigned* visited;unsigned* finished;unsigned executions=0;void execute(std::initializer_list<int>){++executions;++device->latest;*visited=*finished=8;}};
class DaxaView{public:struct Slot{std::map<std::string,daxa::BufferId,std::less<>> buffers;std::map<std::string,External,std::less<>> external;std::unique_ptr<Graph> graph;DaxaObservationFrame input;CaptureLease source;int64_t timestamp=0;uint64_t submit=0,generation=0;unsigned visited=0,finished=0;};
 struct Impl{std::shared_ptr<Context> context;std::shared_ptr<const Maps> maps;GeometryConfig config{1280,720,1280,720};bool typed=true;uint64_t next_generation=1;std::vector<std::shared_ptr<Slot>> slots;void require_thread()const{}void validate_frame(DaxaObservationFrame const&)const;void bind(Slot&,DaxaObservationFrame const&);size_t collect_at(uint64_t);};
 std::shared_ptr<Impl> impl_;class Frame{std::shared_ptr<Impl> owner_;std::shared_ptr<Slot> slot_;uint64_t generation_=0;friend class DaxaView;Frame(std::shared_ptr<Impl>,std::shared_ptr<Slot>);void validate()const;public:Frame()=default;Frame(Frame const&)=default;Frame(Frame&&)noexcept=default;Frame& operator=(Frame)noexcept;CaptureLease source()const;int64_t timestamp_ns()const;uint64_t generation()const;};size_t collect();bool can_project()const;Frame project(DaxaObservationFrame const&);
};
'''
    program += definition(source, 'size_t collect_at(').replace('size_t collect_at(', 'size_t DaxaView::Impl::collect_at(', 1)
    for signature in ('void DaxaView::Impl::validate_frame(', 'void DaxaView::Impl::bind(',
                      'DaxaView::Frame::Frame(', 'DaxaView::Frame& DaxaView::Frame::operator=(', 'void DaxaView::Frame::validate(',
                      'CaptureLease DaxaView::Frame::source(', 'int64_t DaxaView::Frame::timestamp_ns(',
                      'uint64_t DaxaView::Frame::generation(', 'size_t DaxaView::collect(',
                      'bool DaxaView::can_project(', 'DaxaView::Frame DaxaView::project(DaxaObservationFrame const&'):
        program += definition(source, signature)
    program += r'''
}
using namespace stereoforge;
void check(bool b,const char* m){if(!b)throw std::runtime_error(m);}
int main(){try{
 auto context=std::make_shared<Context>();auto engine=std::make_shared<DaxaView::Impl>();engine->context=context;engine->maps=std::make_shared<Maps>();DaxaView view;view.impl_=engine;
 std::array<detail::DepthViewParameters,2> parameters{};
 for(unsigned i=0;i<2;++i){auto slot=std::make_shared<DaxaView::Slot>();for(auto name:{"parameters","source_depth","source_confidence","source_provenance","source_ids","source_ages","source_rgb","rectified_right_rgb","original_left_rgb","original_right_rgb"}){slot->buffers.emplace(name,0);slot->external.emplace(name,External{});}slot->buffers.find("parameters")->second=reinterpret_cast<uintptr_t>(&parameters[i]);slot->graph=std::make_unique<Graph>(Graph{&context->device,&slot->visited,&slot->finished});engine->slots.push_back(std::move(slot));}
 std::array<DaxaObservationFrame,2> inputs;for(unsigned i=0;i<2;++i){auto s=std::make_shared<CaptureFrame>();s->frame_id=7;s->host_ingress_timestamp_ns=100+i;auto in=std::make_shared<Input>();in->context=context;in->maps=engine->maps;in->source=s;in->marker=1000+100*i;inputs[i].owner=context;inputs[i].input=std::move(in);}
 auto complete=[&]{context->device.oldest=context->device.latest;};
 check(view.can_project(),"initial admission failed");auto first=view.project(inputs[0]);complete();auto second=view.project(inputs[1]);complete();check(first.source()==inputs[0].source()&&second.source()==inputs[1].source(),"slot source metadata overwritten");check(first.timestamp_ns()==100&&second.timestamp_ns()==101,"slot timestamp changed");check(first.generation()!=second.generation(),"same source ID generations aliased");
 auto submissions=context->device.latest;unsigned upstream_uploads=0;auto offer=[&]{if(!view.can_project())return false;++upstream_uploads;return true;};check(!offer()&&upstream_uploads==0&&context->device.latest==submissions,"retained ring accepted upstream upload");check(view.collect()==0,"retained input retired");first={};second={};check(view.collect()==2,"completed unleased inputs not retired");for(auto& s:engine->slots)check(!s->input&&!s->source,"retirement kept hidden producer");
 auto inflight=view.project(inputs[0]);inflight={};submissions=context->device.latest;check(bool(engine->slots[0]->input),"inflight producer lost");check(view.collect()==0,"inflight input retired");complete();check(view.collect()==1,"completed producer not retired");
 auto delayed0=view.project(inputs[0]);auto delayed1=view.project(inputs[1]);delayed0={};delayed1={};check(!offer()&&upstream_uploads==0,"inflight ring accepted upstream upload");check(view.collect()==0,"full inflight ring retired early");complete();check(view.collect()==2,"delayed inputs did not retire");submissions=context->device.latest;

 auto foreign=inputs[0];foreign.input=std::make_shared<Input>(*inputs[0].input);foreign.input->maps=std::make_shared<Maps>();auto snapshots=context->device.snapshots;bool caught=false;try{auto bad=view.project(foreign);}catch(std::invalid_argument const&){caught=true;}check(caught&&context->device.latest==submissions&&context->device.snapshots==snapshots,"foreign owner reached admission/submission");foreign={};
 foreign.input=std::make_shared<Input>(*inputs[0].input);foreign.input->context=std::make_shared<Context>();snapshots=context->device.snapshots;caught=false;try{auto bad=view.project(foreign);}catch(std::invalid_argument const&){caught=true;}check(caught&&context->device.latest==submissions&&context->device.snapshots==snapshots,"foreign device reached admission/submission");foreign={};
 foreign.input=std::make_shared<Input>(*inputs[0].input);foreign.input->width=1279;snapshots=context->device.snapshots;caught=false;try{auto bad=view.project(foreign);}catch(std::invalid_argument const&){caught=true;}check(caught&&context->device.latest==submissions&&context->device.snapshots==snapshots,"foreign shape reached admission/submission");foreign={};
 constexpr unsigned frames=10000;auto before=context->device.snapshots;auto executions=engine->slots[0]->graph->executions+engine->slots[1]->graph->executions;auto before_gc=context->device.gc;allocations=0;tracking=true;auto start=std::chrono::steady_clock::now();
 for(unsigned i=0;i<frames;++i){check(view.can_project(),"warm admission failed");auto out=view.project(inputs[i%2]);check(out.source()==inputs[i%2].source(),"actual bind dropped producer");complete();out={};}
 view.collect();double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();tracking=false;
 check(allocations==0,"application allocation in warmed project/admission/retirement");check(context->device.gc-before_gc==frames,"actual project must reclaim Daxa pools once per frame");check(context->device.snapshots-before==2*frames+1,"view poll count changed: one upstream admission plus one project snapshot per frame and final collect");check(engine->slots[0]->graph->executions+engine->slots[1]->graph->executions-executions==frames,"preallocated graph not reused");
 for(auto& s:engine->slots){check(s->buffers.size()==10&&s->external.size()==10,"binding registry grew");check(!s->input&&!s->source,"warm completed source retained");}for(auto& in:inputs)check(in.input.use_count()==1,"source producer pinned after collect");check(parameters[0].view.address_source_depth==inputs[1].buffer(DaxaObservationPlane::Depth),"actual parameter addresses not rebound");check(parameters[0].original_right_rgb==inputs[1].image(DaxaObservationImage::OriginalRight).buffer(),"original RGB namespace not rebound");
 auto keep=view.project(inputs[1]);auto retained=std::weak_ptr<DaxaView::Impl>(engine);view.impl_.reset();engine.reset();check(!retained.expired()&&keep.source()==inputs[1].source(),"output did not retain engine and producer");keep={};check(retained.expired(),"view lease ownership cycle");
 std::cout<<"{\"frames\":"<<frames<<",\"application_allocations\":"<<allocations<<",\"view_admission_and_project_snapshots\":"<<2*frames<<",\"final_collect_snapshots\":1,\"mean_cpu_ms\":"<<ms/frames<<",\"retained_sources_bounded\":true,\"registry_nodes_per_slot\":20,\"graph_reused\":true,\"retained_ring_rejects_before_upstream_upload\":true,\"scope\":\"actual_View_project_bind_validate_admission_retirement_fake_GPU_no_image_copy\"}\n";
 return 0;
 }catch(std::exception const& e){tracking=false;std::cerr<<e.what()<<'\n';return 1;}}
'''
    setup = body(source, 'void DaxaView::Impl::setup_slots(')
    constructor = body(source, 'DaxaView::DaxaView(GeometryConfig c,VirtualView const& v,std::shared_ptr<DaxaFoundation> context,DaxaObservationFrame const&')
    names = set(re.findall(r'e\.inputs\["([^"]+)"\]', constructor))
    names.update(re.findall(r'e\.upload\(recorder,"([^"]+)"', constructor))
    names.update(re.findall(r'slot\.buffers\["([^"]+)"\]', setup))
    for values in re.findall(r'for\(char const\* name:\{([^}]+)\}\)slot\.buffers\[name\]', setup):
        names.update(re.findall(r'"([^"]+)"', values))
    assert len(names) >= 25, 'probe must include the full actual setup binding registry, not only the dynamic subset'
    old_names = '"parameters","source_depth","source_confidence","source_provenance","source_ids","source_ages","source_rgb","rectified_right_rgb","original_left_rgb","original_right_rgb"'
    assert old_names in program
    program = program.replace(old_names, ','.join(json.dumps(name) for name in sorted(names)))
    program = program.replace('s->buffers.size()==10&&s->external.size()==10', f's->buffers.size()=={len(names)}&&s->external.size()=={len(names)}')
    program = program.replace(':20,', ':'+str(2*len(names))+',')
    directory = tmp_path_factory.mktemp('view-depth-bookkeeping')
    frame_source=(ROOT/'include/stereoforge/daxa_frame.hpp').read_text()
    shared=definition(frame_source,'inline bool completed_submission(')+definition(frame_source,'inline bool available_slot(')
    program=program.replace('SHARED_COMPLETION_HELPERS',shared)
    code = directory/'probe.cpp';code.write_text(program);binary = directory/'probe'
    result = subprocess.run([compiler, '-std=c++20', '-O3', '-I'+str(ROOT/'include'), str(code), '-o', str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return binary


def test_actual_view_admission_projection_and_retirement_cpu_probe(tmp_path_factory):
    binary = build_view_probe(tmp_path_factory)
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    print('Actual retained View CPU-only probe:', json.dumps(row, sort_keys=True))
    assert row['frames'] == 10000 and row['application_allocations'] == 0
    assert row['view_admission_and_project_snapshots'] == 20000
    assert row['graph_reused'] and row['retained_sources_bounded'] and row['retained_ring_rejects_before_upstream_upload']
    assert 0 < row['mean_cpu_ms'] < 1000


@pytest.mark.parametrize('mutation', ['allocation', 'retained_admission', 'upstream_admission'])
def test_actual_view_probe_detects_hot_allocation_and_retained_slot_overwrite(tmp_path_factory, mutation):
    def transform(source):
        marker = 'DaxaView::Frame DaxaView::project(DaxaObservationFrame const& input){'
        assert marker in source
        if mutation == 'allocation':
            return source.replace(marker, marker+'auto allocation_guard=std::make_shared<int>(42);', 1)
        if mutation == 'upstream_admission':
            start = source.index('bool DaxaView::can_project(')
            before = 'detail::available_slot(slot.use_count(),slot->submit,pending)'
            assert before in source[start:]
            return source[:start]+source[start:].replace(before, 'true', 1)
        start = source.index(marker)
        before = 'detail::available_slot(s.use_count(),s->submit,pending)'
        assert before in source[start:]
        return source[:start]+source[start:].replace(before, 'true', 1)
    binary = build_view_probe(tmp_path_factory, transform)
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert {'allocation': 'application allocation', 'retained_admission': 'stale/empty Daxa view frame', 'upstream_admission': 'retained ring accepted upstream upload'}[mutation] in result.stderr


def test_cold_view_depth_getters_bind_actual_compiled_shader_sources_without_vulkan():
    import hashlib
    import os
    sha = lambda name: hashlib.sha256((ROOT/'python/stereoforge/shaders'/name).read_bytes()).hexdigest()
    expected = {
        'view_depth_shader_source_sha256_pair': sha('daxa/view_depth.slang')+':'+sha('ops/projection.slang'),
        'depth_shader_source_sha256_pair': sha('daxa/depth.slang')+':'+sha('ops/stereo_observation.slang'),
        'preprocess_shader_source_sha256_pair': sha('daxa/preprocess.slang')+':'+sha('ops/preprocess.slang'),
    }
    common = (ROOT/'apps/daxa_view_depth_common.hpp').read_text()
    assert 'bool getter(' in common and '--shader-source-identities' in common
    for name in ('daxa_view_depth_benchmark', 'daxa_view_depth_fixture'):
        source = (ROOT/'apps'/f'{name}.cpp').read_text()
        prepared = re.search(r'Prepared prepared\(options(?:,true)?\)', source)
        assert prepared is not None
        assert source.index('getter(argc,argv)') < prepared.start() < source.index('std::make_shared<DaxaFoundation>')
        binary = ROOT/'build/daxa'/name
        if not binary.is_file():
            pytest.skip('optional View depth binaries not built')
        result = subprocess.run([str(binary), '--shader-source-identities'], capture_output=True, text=True,
                                env=dict(os.environ, VK_DRIVER_FILES='/nonexistent-stereoforge-View-depth-getter.json'), timeout=10)
        assert result.returncode == 0 and not result.stderr, result.stderr
        assert json.loads(result.stdout) == expected


def test_actual_setup_default_pose_uses_rig_midpoint_but_fixture_left_origin(tmp_path):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable for setup-only pose probe')
    common = (ROOT/'apps/daxa_view_depth_common.hpp').read_text()
    helper = definition(common, 'inline Pose4 default_original_pose(')
    program = '#include <stereoforge/geometry.hpp>\nnamespace stereoforge {struct DaxaStereoCalibration {std::array<double,16> right_to_left;};}\nusing namespace stereoforge;\n'+helper+r'''
int main(){DaxaStereoCalibration calibration{};
 calibration.right_to_left={0,-1,0,-.12,1,0,0,.04,0,0,1,-.06,0,0,0,1};
 const Pose4 expected{1,0,0,.06f,0,1,0,-.02f,0,0,1,.03f,0,0,0,1};
 if(default_original_pose(calibration,false)!=expected)return 1;
 if(default_original_pose(calibration,true)!=identity_pose())return 2;
 calibration.right_to_left[3]=0;calibration.right_to_left[7]=0;calibration.right_to_left[11]=0;
 if(default_original_pose(calibration,false)!=identity_pose())return 3;
}
'''
    code = tmp_path/'pose.cpp';code.write_text(program);binary = tmp_path/'pose'
    result = subprocess.run([compiler, '-std=c++20', '-O2', '-I'+str(ROOT/'include'), str(code), '-o', str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    prepared = body(common, 'Prepared(Options const& o,bool fixture_input=false)')
    assert prepared.index('default_original_pose(g.original,fixture_input)') < prepared.index('if(o.values.contains("--virtual-from-original-left"))') < prepared.index('cache=prepare_daxa_view_calibration(')
    override = body(prepared, 'if(o.values.contains("--virtual-from-original-left"))')
    assert 'pose_policy="explicit_virtual_from_original_left"' in override and 'view.source_to_virtual_left[i]=real(' in override
    assert 'pose_policy=fixture_input?"fixture_original_left_origin":"rig_midpoint_original_left_axes"' in prepared
    assert 'Prepared prepared(options,true)' in (ROOT/'apps/daxa_view_depth_fixture.cpp').read_text()
    assert 'Prepared prepared(options);' in (ROOT/'apps/daxa_view_depth_benchmark.cpp').read_text()
