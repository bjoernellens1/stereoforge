"""Actual-source CPU control probes; fake GPU calls exclude framework/driver cost."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tempfile

import pytest
from test_daxa_depth_hotpath import definition
from test_kernel_variant_hotpath import body
from test_daxa_depth_pair_hotpath import build_pair_bookkeeping_probe

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/zed-task3/performance'
FORBIDDEN = r'\b(?:make_shared|make_unique|allocate|create_\w+|wait_idle|wait_on_submit|readback|malloc|calloc|realloc|resize|reserve|push_back|emplace_back|metric_calibration|sha_file|strict_json)(?:<[^;()]*>)?\s*\(|\bnew\s+'


def sources():
    return {name: (ROOT / name).read_text() for name in (
        'src/daxa_depth.cpp', 'src/daxa_stereo.cpp', 'src/daxa_frame.cpp',
        'src/daxa_rectified_input.cpp', 'src/daxa_refinement.cpp',
        'include/stereoforge/daxa_frame.hpp', 'include/stereoforge/canonical_rgb.hpp',
        'src/capture.cpp', 'src/daxa_observation.cpp')}


def checked_hotpaths(src):
    selected = {
        'src/daxa_depth.cpp': ['DaxaDepthPipeline::Frame DaxaDepthPipeline::enqueue_preprocessed(', 'DaxaDepthPipeline::Frame DaxaDepthPipeline::submit_preprocessed(', 'void validate_input(', 'size_t collect_at(', 'std::array<daxa::BufferId,8> input_buffers(', 'void DaxaDepthPipeline::validate_pair(', 'DaxaObservationFrame DaxaDepthPipeline::Frame::observations(', 'void DaxaDepthPipeline::Frame::validate('],
        'src/daxa_stereo.cpp': ['unsigned DaxaStereo::pair_slot_at(', 'std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at(', 'size_t DaxaStereo::collect_completed_inputs_at(', 'DaxaStereo::Frame DaxaStereo::process_at('],
        'src/daxa_rectified_input.cpp': ['DaxaRectifiedInput::Frame DaxaRectifiedInput::enqueue(', 'bool available(', 'size_t collect_at(', 'void dispatch(daxa::TaskInterface', '.executes([slot,source_view=views[0],destination_view=views[1],size=upload_bytes]'],
        'src/daxa_frame.cpp': ['void DaxaFrame::validate(', 'CaptureLease DaxaFrame::source(', 'DaxaFrame::calibration(', 'DaxaFrame::plane(', 'DaxaImageView::buffer('],
        'src/daxa_observation.cpp': ['void DaxaObservationFrame::validate(', 'DaxaObservationFrame::DaxaObservationFrame('],
        'src/daxa_refinement.cpp': ['DaxaRefinement::Frame DaxaRefinement::process(DaxaDepthPipeline::Frame const& input,DaxaFrame const& full)', 'size_t collect_at(', 'std::array<daxa::BufferId,5> input_buffers(', 'std::array<daxa::BufferId,6> full_buffers(', 'DaxaObservationFrame DaxaRefinement::Frame::observations(', 'constexpr std::array<CanonicalPackCopy,6> canonical_pack_copies(', '.executes([raw,copies,full_views,out]'],
    }
    # Include every application helper on these enqueue paths, not only public
    # method bodies. Pixel/support scanning is setup-only and guarded below.
    for path, signatures in selected.items():
        for signature in signatures:
            fn = body(src[path], signature)
            assert not re.search(FORBIDDEN, fn), path + ':' + signature + ' allocates/waits/reads back'
            if path != 'src/daxa_rectified_input.cpp':
                assert not re.search(r'\b(?:memcpy|memset)\s*\(', fn)
    canonical = src['include/stereoforge/canonical_rgb.hpp']
    assert not re.search(FORBIDDEN, body(canonical, 'inline bool canonical_rgb_grid_association('))
    validate = body(canonical, 'inline void validate_canonical_rgb_source(')
    assert not re.search(FORBIDDEN, validate)
    assert 'if(scan_support)for(' in validate, 'support scan must be setup-only'
    assert 'bool scan_support=false' in canonical
    assert not re.search(FORBIDDEN, body(src['src/capture.cpp'], 'void CaptureFrame::require_calibration('))
    enqueue = body(src['src/daxa_rectified_input.cpp'], 'DaxaRectifiedInput::Frame DaxaRectifiedInput::enqueue(')
    assert 'input.admitted()' in enqueue and enqueue.index('input.admitted()') < enqueue.index('oldest_pending_submit_index(')
    assert not re.search(FORBIDDEN, body(canonical, 'bool admitted('))
    assert not re.search(r'validate_canonical_rgb_source\([^;]*,\s*true\s*\)', enqueue)
    assert not re.search(r'\b(?:gray|color|grid|validate_daxa_rectified_input_geometry)\s*\(', enqueue)
    assert 'slot->full->source=input.source' in enqueue and 'slot->coarse->source=input.source' in enqueue
    assert enqueue.index('slot->full->source=input.source') < enqueue.index('graph->execute(')
    assert 'slot->full->submit=slot->submit' in enqueue and enqueue.count('slot->submit=e.context->device.latest_queue_submit_index') == 2
    submit = body(src['src/daxa_depth.cpp'], 'DaxaDepthPipeline::Frame DaxaDepthPipeline::submit_preprocessed(')
    assert 'slot->preprocessing=std::move(pre)' in submit
    assert submit.index('slot->preprocessing=std::move(pre)') < submit.index('graph->execute(')
    assert 'slot->submit=d.latest_queue_submit_index' in body(submit, 'catch(...)')
    refinement = body(src['src/daxa_refinement.cpp'], 'DaxaRefinement::Frame DaxaRefinement::process(DaxaDepthPipeline::Frame const& input,DaxaFrame const& full)')
    assert 's->input=input;s->full_input=full' in refinement
    assert refinement.index('s->input=input;s->full_input=full') < refinement.index('graph->execute(')
    assert 's->submit=d.latest_queue_submit_index' in body(refinement, 'catch(...)')
    pack = body(src['src/daxa_refinement.cpp'], '.executes([raw,copies,full_views,out]')
    barrier = re.search(r'pipeline_barrier\(\{\.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,\.dst_access=daxa::AccessConsts::TRANSFER_READ\}\)', pack)
    assert barrier and barrier.start() < pack.index('copy_buffer_to_buffer('), 'cross-graph input writes must precede canonical transfer reads'
    available = body(src['src/daxa_rectified_input.cpp'], 'bool available(')
    for marker in ('s.use_count()==1', 's->full.use_count()==1', 's->coarse.use_count()==1', 'detail::completed_submission(s->submit,oldest)'):
        assert marker in available
    completed = body(src['include/stereoforge/daxa_frame.hpp'], 'inline bool completed_submission(')
    assert 'submitted<=oldest_pending' in completed


def test_transitive_application_control_paths_have_no_allocations_waits_or_readback():
    checked_hotpaths(sources())


@pytest.mark.parametrize('injection', ['std::make_shared<int>();', 'd.wait_idle();', 'readback(frame);', 'v.resize(9);', 'auto p=new int;', 'malloc(16);'])
def test_guard_rejects_hidden_helper_work(injection):
    src = sources()
    marker = 'void validate_input(const DaxaFrame& input)const{'
    assert marker in src['src/daxa_depth.cpp']
    src['src/daxa_depth.cpp'] = src['src/daxa_depth.cpp'].replace(marker, marker + injection, 1)
    with pytest.raises(AssertionError, match='allocates/waits/reads back'):
        checked_hotpaths(src)


@pytest.mark.parametrize('path,old,new', [
    ('src/daxa_depth.cpp', 'slot->preprocessing=std::move(pre)', 'slot->preprocessing={}'),
    ('src/daxa_rectified_input.cpp', 'slot->full->source=input.source', 'slot->full->source={}'),
    ('src/daxa_refinement.cpp', 's->input=input;s->full_input=full', 's->input=input;s->full_input={}'),
    ('include/stereoforge/daxa_frame.hpp', 'submitted<=oldest_pending', 'submitted<oldest_pending'),
])
def test_guard_rejects_lost_retention_and_exclusive_ticket_semantics(path, old, new):
    src = sources(); assert old in src[path]; src[path] = src[path].replace(old, new, 1)
    with pytest.raises(AssertionError): checked_hotpaths(src)


class ArtifactFactory:
    def __init__(self, directory): self.directory = directory
    def mktemp(self, name):
        result = self.directory / name; result.mkdir(); return result


def compile_run(code, name):
    compiler = shutil.which('c++')
    if not compiler: pytest.skip('C++ compiler unavailable')
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=name+'-', dir=ARTIFACTS) as tmp:
        directory=Path(tmp); source=directory/'probe.cpp'; binary=directory/'probe'; source.write_text(code)
        result=subprocess.run([compiler, '-std=c++20', '-O3', str(source), '-o', str(binary)], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        result=subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
        (ARTIFACTS/(name+'-probe.cpp')).write_text(code)
        return result


@pytest.fixture(scope='module')
def external_depth_program():
    # Reuse the established fake-device fixture, including actual paired stereo
    # functions. Replace its raw main/setup with the external-input exercise.
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='harness-', dir=ARTIFACTS) as tmp:
        binary=build_pair_bookkeeping_probe(ArtifactFactory(Path(tmp)))
        text=binary.with_name('probe.cpp').read_text().split('int main(int argc,char** argv)')[0]
    text=text.replace('struct CaptureFrame {uint64_t frame_id=0;', 'struct CaptureFrame {uint32_t source_width=2560,height=720;std::string calibration_identity="original";uint64_t frame_id=0;')
    text=text.replace('struct Maps {std::string rectified_identity="rect";', 'struct Maps {uint32_t eye_width=1280,source_height=720,width=640,height=360;std::string original_identity="original",rectified_identity="rect";')
    text=text.replace('friend class DaxaStereo;public:', 'friend class DaxaStereo;friend class DaxaDepthPipeline;public:')
    text=text.replace('struct DaxaDepthConfig {DaxaDepthStereoSchedule', 'struct DaxaDepthConfig {struct {uint32_t width=640,height=360;} stereo;DaxaDepthStereoSchedule')
    text=text.replace('DaxaStereoReference reference()const', 'uint64_t buffer(DaxaStereoPlane p)const{return uint64_t(unsigned(p)+110);}DaxaStereoReference reference()const')
    text=text.replace('void require_thread()const{}size_t collect_at(uint64_t);', 'std::shared_ptr<const Maps> calibration;void validate_input(const DaxaFrame&)const;void require_thread()const{}size_t collect_at(uint64_t);')
    text=text.replace(' std::array<uint64_t,8> input_buffers(const DaxaFrame&,const DaxaStereo::Frame&,const DaxaStereo::Frame&){return {};}', ' std::array<uint64_t,8> input_buffers(const DaxaFrame&,const DaxaStereo::Frame&,const DaxaStereo::Frame&);')
    text=text.replace('Frame enqueue(CaptureLease);', 'Frame enqueue(CaptureLease);Frame enqueue_preprocessed(const DaxaFrame&);')
    # Insert definitions before namespace closes; actual validation/input binding.
    extra=definition((ROOT/'src/daxa_depth.cpp').read_text(), 'void validate_input(').replace('void validate_input(', 'void DaxaDepthPipeline::Impl::validate_input(',1)
    extra+=definition((ROOT/'src/daxa_depth.cpp').read_text(), 'std::array<daxa::BufferId,8> input_buffers(').replace(' input_buffers(', ' DaxaDepthPipeline::Impl::input_buffers(',1)
    extra+=definition((ROOT/'src/daxa_depth.cpp').read_text(), 'DaxaDepthPipeline::Frame DaxaDepthPipeline::enqueue_preprocessed(')
    text=text.replace('}\nusing namespace stereoforge;\n', extra+'}\nusing namespace stereoforge;\n')
    return text


DEPTH_MAIN = r'''
int main(){try{
 auto context=std::make_shared<Context>();auto e=std::make_shared<DaxaDepthPipeline::Impl>();e->context=context;
 auto pre=std::make_shared<PreImpl>();pre->context=context;pre->calibration=std::make_shared<Maps>();e->calibration=pre->calibration;
 auto owned=std::make_shared<CaptureFrame>();owned->frame_id=INT64_MAX;CaptureLease source=owned;
 auto producer=std::make_shared<PreSlot>();producer->source=source;DaxaFrame input(pre,producer);
 for(unsigned eye=0;eye<2;++eye){auto engine=std::make_unique<DaxaStereo>();engine->impl_=std::make_shared<StereoImpl>();engine->impl_->context=context;engine->impl_->calibration=pre->calibration;engine->impl_->reference=eye?DaxaStereoReference::Right:DaxaStereoReference::Left;for(unsigned i=0;i<2;++i){auto s=std::make_shared<StereoSlot>();s->buffers[Params]=reinterpret_cast<uintptr_t>(&s->parameters);engine->impl_->slots.push_back(std::move(s));}(eye?e->right:e->left)=std::move(engine);}
 e->left->impl_->paired=std::make_unique<Pair>();e->left->impl_->paired->right=e->right->impl_;for(unsigned i=0;i<2;++i){auto s=std::make_unique<PairSlot>();s->graph=std::make_unique<Graph>(Graph{&context->device,false,&s->started,&s->finished,8});e->left->impl_->paired->slots.push_back(std::move(s));}
 for(unsigned i=0;i<2;++i){auto s=std::make_shared<DaxaDepthPipeline::Slot>();s->parameters=reinterpret_cast<uintptr_t>(&s->data);s->graph=std::make_unique<Graph>(Graph{&context->device,false});e->slots.push_back(std::move(s));}
 DaxaDepthPipeline pipeline;pipeline.impl_=e;
 auto complete=[&]{context->device.oldest=context->device.latest;};
 auto first=pipeline.enqueue_preprocessed(input);complete();auto second=pipeline.enqueue_preprocessed(input);complete();
 auto latest=context->device.latest;bool caught=false;try{pipeline.enqueue_preprocessed(input);}catch(const std::runtime_error&){caught=true;}
 check(caught&&context->device.latest==latest,"held output ring admitted");
 first={};second={};e->collect_at(context->device.oldest);
 // Exact inclusive boundary must allow reuse; incomplete ticket must not retire.
 auto out=pipeline.enqueue_preprocessed(input);out={};check(e->collect_at(context->device.oldest)==0,"inflight depth lost input");
 check(bool(e->slots[0]->preprocessing),"input not retained");complete();check(e->collect_at(context->device.oldest)==1,"inclusive completed depth did not retire");
 // Aligned stereo backpressure must reject without submitting depth.
 std::array<std::shared_ptr<StereoSlot>,2> held{e->right->impl_->slots[0],e->right->impl_->slots[1]};latest=context->device.latest;caught=false;
 try{pipeline.enqueue_preprocessed(input);}catch(const std::runtime_error&){caught=true;}check(caught&&context->device.latest==latest,"pair admission submitted");held={};
 // Partial metric submission: the real catch must retain input and latest ticket.
 e->slots[0]->graph->fail=true;caught=false;try{pipeline.enqueue_preprocessed(input);}catch(const std::runtime_error&){caught=true;}
 check(caught&&e->slots[0]->preprocessing&&e->slots[0]->submit==context->device.latest,"partial depth retention lost");
 check(e->collect_at(context->device.oldest)==0,"partial depth retired early");e->slots[0]->graph->fail=false;complete();e->collect_at(context->device.oldest);
 constexpr unsigned frames=65536;
 for(int run=0;run<3;++run){allocations=0;auto snapshots=context->device.snapshots,submissions=context->device.latest;tracking=true;auto start=std::chrono::steady_clock::now();
 for(unsigned i=0;i<frames;++i){auto result=pipeline.enqueue_preprocessed(input);result.validate();complete();result={};}e->collect_at(context->device.oldest);
 auto ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/frames;tracking=false;
 check(allocations==0,"warm control allocated");check(context->device.snapshots-snapshots==frames,"multiple admission snapshots");check(context->device.latest-submissions==2*frames,"unexpected submission count");
 std::cout<<"{\"scope\":\"actual_external_depth_and_paired_stereo_application_control_fake_GPU\",\"run\":"<<run<<",\"frames\":"<<frames<<",\"cpu_ns_per_frame\":"<<ns<<",\"application_allocations\":"<<allocations<<",\"application_waits\":"<<context->device.waits<<",\"application_readbacks\":"<<context->device.readbacks<<",\"inclusive_completion_retention_partial_submit\":true}\n";}
 return 0;
}catch(const std::exception& e){tracking=false;std::cerr<<e.what()<<'\n';return 1;}}
'''


def test_actual_external_depth_control_cost_and_retention(external_depth_program):
    # Fake execute increments a completion ticket before the injected partial
    # submission failure; no application algorithm is reimplemented here.
    code=external_depth_program.replace('uint64_t oldest=0,latest=0,snapshots=0,gc=0,uploads=0;', 'uint64_t oldest=0,latest=0,snapshots=0,gc=0,uploads=0,waits=0,readbacks=0;').replace('unsigned stages=0;void execute', 'unsigned stages=0;bool fail=false;void execute')
    code=code.replace('if(finished)*finished=stages;}', 'if(finished)*finished=stages;if(fail)throw std::runtime_error("partial fake submit");}')
    result=compile_run(code+DEPTH_MAIN, 'external-depth')
    assert result.returncode==0,result.stderr
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows)==3 and all(row['application_allocations']==0 for row in rows)
    report={'samples': rows, 'source_sha256': {path:hashlib.sha256(text.encode()).hexdigest() for path,text in sources().items()}}
    (ARTIFACTS/'external-depth.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.stdout)


def producer_program(base):
    src=sources(); header=src['include/stereoforge/canonical_rgb.hpp']
    types=header[header.index('inline constexpr'):header.index('inline void validate_canonical_rgb_source')]
    forward='enum class CaptureSourceKind {PackedYuyv,CanonicalRgbMetadata};struct CanonicalRgbSource;'
    base=base.replace('struct CaptureFrame {',forward+'struct CaptureFrame {CaptureSourceKind source_kind=CaptureSourceKind::CanonicalRgbMetadata;std::shared_ptr<const CanonicalRgbSource> canonical_rgb;uint32_t stride_bytes=0;uint64_t camera_timestamp_ns=1;std::string camera_clock_domain="camera",host_clock_domain="host";void require_calibration()const{if(calibration_identity.empty())throw std::invalid_argument("missing calibration");}')
    base=base.replace('using CaptureLease=std::shared_ptr<const CaptureFrame>;', 'using CaptureLease=std::shared_ptr<const CaptureFrame>;'+types+definition(header,'inline void validate_canonical_rgb_source(')+definition(header,'inline bool canonical_rgb_grid_association(')+definition(header,'class RectifiedRgbFrame')+';')
    base=base.replace('#include <array>','#include <span>\n#include <limits>\n#include <array>')
    base=base.replace('uint64_t oldest=0,latest=0,snapshots=0,gc=0,uploads=0;', 'uint64_t oldest=0,latest=0,snapshots=0,gc=0,uploads=0,waits=0,readbacks=0;')
    base=base.replace('unsigned stages=0;void execute', 'unsigned stages=0;bool fail=false;void execute').replace('if(finished)*finished=stages;}', 'if(finished)*finished=stages;if(fail)throw std::runtime_error("partial submit");}')
    code=r'''
class DaxaRectifiedInput {public:struct Impl:PreImpl {
 struct Slot {std::shared_ptr<PreSlot> full=std::make_shared<PreSlot>(),coarse=std::make_shared<PreSlot>();uint64_t staging=0,submit=0;std::unique_ptr<Graph> graph;};
 std::shared_ptr<const Maps> full,coarse;std::array<CanonicalRgbPlane,6> descriptors;std::array<size_t,6> offsets{},extents{};std::vector<std::shared_ptr<Slot>> slots;
 bool available(const std::shared_ptr<Slot>&,uint64_t)const;size_t collect_at(uint64_t);
};
 std::shared_ptr<Impl> impl_;struct Frame {std::shared_ptr<Impl> owner_;std::shared_ptr<void> slot_;DaxaFrame full_,coarse_;Frame(std::shared_ptr<Impl>o,std::shared_ptr<void>s,DaxaFrame f,DaxaFrame c):owner_(o),slot_(s),full_(f),coarse_(c){}Frame()=default;};Frame enqueue(const RectifiedRgbFrame&);
};
bool real_copy=false;uint64_t copied_bytes=0,copy_calls=0;
void* probe_memcpy(void* d,const void* s,size_t n){++copy_calls;copied_bytes+=n;if(real_copy)return std::memcpy(d,s,n);return d;}
void* probe_memset(void* d,int v,size_t n){if(real_copy)return std::memset(d,v,n);return d;}
'''
    code+=definition(src['src/daxa_rectified_input.cpp'],'bool available(').replace('bool available(', 'bool DaxaRectifiedInput::Impl::available(',1)
    code+=definition(src['src/daxa_rectified_input.cpp'],'size_t collect_at(').replace('size_t collect_at(', 'size_t DaxaRectifiedInput::Impl::collect_at(',1)
    code+=definition(src['src/daxa_rectified_input.cpp'],'DaxaRectifiedInput::Frame DaxaRectifiedInput::enqueue(').replace('std::memcpy(', 'probe_memcpy(').replace('std::memset(', 'probe_memset(')
    base=base.replace('}\nusing namespace stereoforge;\n',code+'}\nusing namespace stereoforge;\n')
    return base+r'''
int main(){try{
 auto context=std::make_shared<Context>();auto e=std::make_shared<DaxaRectifiedInput::Impl>();e->context=context;
 auto full=std::make_shared<Maps>();full->width=1280;full->height=720;full->rectified_identity="full";auto coarse=std::make_shared<Maps>(*full);coarse->width=640;coarse->height=360;coarse->rectified_identity="coarse";e->full=full;e->coarse=coarse;
 auto side=std::make_shared<CanonicalRgbSource>();side->input_identity="input";side->original_pixel_identity="original pixels";side->rectified_pixel_identity="rectified pixels";side->original_calibration_identity="original";side->rectified_calibration_identity="full";side->coarse_rectified_calibration_identity="coarse";
 auto backing=std::make_shared<std::array<std::vector<uint8_t>,6>>();side->backing=backing;size_t total=0;
 for(unsigned i=0;i<6;++i){auto& plane=i<4?side->rgb[i]:side->support[i-4];plane.width=1280;plane.height=720;plane.stride_bytes=1280*(i<4?3:1);plane.format=i<4?CanonicalRgbFormat::Rgb8:CanonicalRgbFormat::SupportU8;(*backing)[i].resize(size_t(plane.stride_bytes)*720,i<4?173:1);plane.bytes=(*backing)[i];e->descriptors[i]=plane;e->extents[i]=plane.bytes.size();e->offsets[i]=total;total+=plane.bytes.size();}
 auto source=std::make_shared<CaptureFrame>();source->frame_id=INT64_MAX;source->canonical_rgb=side;RectifiedRgbFrame input{source};DaxaRectifiedInput producer;producer.impl_=e;
 std::array<std::vector<uint8_t>,2> staging;for(unsigned i=0;i<2;++i){staging[i].resize(total);auto s=std::make_shared<DaxaRectifiedInput::Impl::Slot>();s->staging=uintptr_t(staging[i].data());s->graph=std::make_unique<Graph>(Graph{&context->device,true});e->slots.push_back(s);}
 auto complete=[&]{context->device.oldest=context->device.latest;};
 // Actual header capability constructor deep-validates each second record
 // before the resident loop; rejection and construction costs are not timed.
 auto second_backing=std::make_shared<std::array<std::vector<uint8_t>,6>>(*backing);
 auto second_side=std::make_shared<CanonicalRgbSource>(*side);second_side->backing=second_backing;
 for(unsigned i=0;i<6;++i)(i<4?second_side->rgb[i]:second_side->support[i-4]).bytes=(*second_backing)[i];
 auto second_source=std::make_shared<CaptureFrame>(*source);second_source->frame_id=2;second_source->canonical_rgb=second_side;
 (*second_backing)[5].back()=2;bool bad_second=false;try{RectifiedRgbFrame invalid(second_source);}catch(const std::invalid_argument&){bad_second=true;}
 check(bad_second,"second record binary support accepted");(*second_backing)[5].back()=1;
 RectifiedRgbFrame admitted_second(second_source);check(admitted_second.admitted(),"valid second record not admitted");
 RectifiedRgbFrame bypass;bypass.source=second_source;auto before_admission=context->device.snapshots;bool bypassed=false;
 try{producer.enqueue(bypass);}catch(const std::invalid_argument&){bypassed=true;}
 check(bypassed&&context->device.snapshots==before_admission,"default capability bypass reached device");
 auto changed=input;changed.source=second_source;bypassed=false;try{producer.enqueue(changed);}catch(const std::invalid_argument&){bypassed=true;}
 check(bypassed&&context->device.snapshots==before_admission,"replacement capability bypass reached device");
 {auto first=producer.enqueue(input);complete();auto second=producer.enqueue(input);complete();auto latest=context->device.latest;bool reject=false;try{producer.enqueue(input);}catch(const std::runtime_error&){reject=true;}check(reject&&context->device.latest==latest,"outer retained slots reused");}
 e->collect_at(context->device.oldest);
 {auto outer=producer.enqueue(input);DaxaFrame child=outer.full_;DaxaFrame coarse_child=outer.coarse_;outer={};complete();check(e->collect_at(context->device.oldest)==0,"retained child released source");child={};check(e->collect_at(context->device.oldest)==0,"retained coarse child released source");coarse_child={};check(e->collect_at(context->device.oldest)==1,"inclusive completed child not retired");}
 e->slots[0]->graph->fail=true;bool rejected=false;try{producer.enqueue(input);}catch(const std::runtime_error&){rejected=true;}
 check(rejected&&e->slots[0]->full->source==source&&e->slots[0]->coarse->source==source&&e->slots[0]->submit==context->device.latest&&e->slots[0]->full->submit==context->device.latest&&e->slots[0]->coarse->submit==context->device.latest,"partial producer source/ticket lost");
 check(e->collect_at(context->device.oldest)==0,"partial producer retired early");complete();e->slots[0]->graph->fail=false;check(e->collect_at(context->device.oldest)==1,"partial source not retired at equality");
 for(bool copying:{false,true})for(int run=0;run<3;++run){real_copy=copying;const unsigned frames=copying?128u:65536u;allocations=0;copied_bytes=copy_calls=0;double upload_ns=0;auto snapshots=context->device.snapshots;tracking=true;auto start=std::chrono::steady_clock::now();
 for(unsigned i=0;i<frames;++i){auto out=producer.enqueue(input);upload_ns+=e->upload_cpu_ms*1e6;complete();out={};}e->collect_at(context->device.oldest);
 auto ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/frames;tracking=false;
 check(allocations==0&&context->device.snapshots-snapshots==frames,"producer allocated/repolled");check(copy_calls==frames*6&&copied_bytes==frames*total,"fixed prepared six-plane copy changed");
 if(copying)for(unsigned i=0;i<6;++i)check(std::memcmp(staging[0].data()+e->offsets[i],(*backing)[i].data(),e->extents[i])==0,"staging byte mismatch");
 std::cout<<"{\"scope\":\"actual_producer_application_control_fake_GPU_copy_boundary_observed\",\"copy_enabled\":"<<(copying?"true":"false")<<",\"run\":"<<run<<",\"frames\":"<<frames<<",\"cpu_ns_per_frame\":"<<ns<<",\"copy_interval_ns_per_frame\":"<<upload_ns/frames<<",\"prepared_bytes_per_frame\":"<<total<<",\"application_allocations\":"<<allocations<<",\"application_waits\":"<<context->device.waits<<",\"application_readbacks\":"<<context->device.readbacks<<",\"offline_admission_excluded\":true,\"second_record_binary_and_capability_bypass_rejected\":true,\"fixed_ring_slots\":2,\"retention_partial_submit_inclusive_ticket\":true}\n";
 }
 return 0;}catch(const std::exception& e){tracking=false;std::cerr<<e.what()<<'\n';return 1;}}
'''


def test_actual_producer_control_and_copy_cost(external_depth_program):
    result=compile_run(producer_program(external_depth_program),'canonical-producer')
    assert result.returncode==0,result.stderr
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows)==6 and all(row['application_allocations']==0 for row in rows)
    (ARTIFACTS/'canonical-producer.json').write_text(json.dumps({'samples':rows,'source_sha256':{p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources().items()}},indent=2)+'\n')
    print(result.stdout)


def test_actual_canonical_refinement_control_three_samples():
    from test_daxa_refinement_hotpath import build_probe
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='refinement-harness-', dir=ARTIFACTS) as tmp:
        binary=build_probe(Path(tmp));code=binary.with_name('probe.cpp').read_text()
    marker='auto canonical_polls=context->device.polls;allocations=0;track=true;for(unsigned n=0;n<10000;++n)'
    assert marker in code
    code=code.replace(marker,'for(int sample=0;sample<3;++sample){auto canonical_start=std::chrono::steady_clock::now();'+marker)
    end='"canonical hotpath allocation/wait/poll regression");stage.collect();'
    assert end in code
    code=code.replace(end,end+r'''
 auto canonical_ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-canonical_start).count()/10000;
 std::cout<<"{\"canonical_sample\":"<<sample<<",\"frames\":10000,\"cpu_ns_per_frame\":"<<canonical_ns<<",\"application_allocations\":"<<allocations<<",\"application_waits\":"<<context->device.waits<<",\"scope\":\"actual_canonical_refinement_process_collect_observation_control_fake_GPU\"}\n";
 }
''')
    result=compile_run(code,'canonical-refinement');assert result.returncode==0,result.stderr
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows)==4 and all(row.get('application_allocations')==0 for row in rows)
    (ARTIFACTS/'canonical-refinement.json').write_text(json.dumps({'samples':rows,'source_sha256':{p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources().items()}},indent=2)+'\n')
    print(result.stdout)


@pytest.mark.parametrize('mutation', ['allocation', 'input_retention', 'partial_ticket'])
def test_actual_external_probe_rejects_runtime_regressions(external_depth_program, mutation):
    code=external_depth_program.replace('uint64_t oldest=0,latest=0,snapshots=0,gc=0,uploads=0;', 'uint64_t oldest=0,latest=0,snapshots=0,gc=0,uploads=0,waits=0,readbacks=0;')
    code=code.replace('unsigned stages=0;void execute', 'unsigned stages=0;bool fail=false;void execute').replace('if(finished)*finished=stages;}', 'if(finished)*finished=stages;if(fail)throw std::runtime_error("partial fake submit");}')
    if mutation=='allocation':
        anchor='e.validate_input(input);';assert anchor in code
        code=code.replace(anchor,anchor+'auto extra_heap=std::make_shared<int>(17);if(*extra_heap!=17)throw std::runtime_error("bad allocation");',1)
    elif mutation=='input_retention':
        anchor='slot->preprocessing=std::move(pre)';assert anchor in code
        code=code.replace(anchor,'slot->preprocessing={}',1)
    else:
        fn=definition(code,'DaxaDepthPipeline::Frame DaxaDepthPipeline::submit_preprocessed(')
        anchor='catch(...){slot->submit=d.latest_queue_submit_index(daxa::QUEUE_MAIN);throw;}'
        assert anchor in fn
        code=code.replace(fn,fn.replace(anchor,'catch(...){throw;}',1))
    result=compile_run(code+DEPTH_MAIN,'external-depth-mutation-'+mutation)
    assert result.returncode != 0
    expected={'allocation':'warm control allocated','input_retention':'paired stereo backpressure','partial_ticket':'partial depth retention lost'}
    assert expected[mutation] in result.stderr, result.stderr


@pytest.mark.parametrize('replacement', ['', 'ti.recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ});'])
def test_transitive_pack_guard_rejects_missing_or_wrong_transfer_barrier(replacement):
    src=sources();barrier='ti.recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ});'
    assert barrier in src['src/daxa_refinement.cpp']
    src['src/daxa_refinement.cpp']=src['src/daxa_refinement.cpp'].replace(barrier,replacement,1)
    with pytest.raises(AssertionError,match='cross-graph input writes'):
        checked_hotpaths(src)


@pytest.mark.parametrize('mutation', ['second_record_scan', 'capability_bypass'])
def test_actual_header_admission_rejects_mutated_deep_scan_and_capability(external_depth_program, mutation):
    code=producer_program(external_depth_program)
    if mutation=='second_record_scan':
        anchor='validate_canonical_rgb_source(*source,true);';assert anchor in code
        code=code.replace(anchor,'validate_canonical_rgb_source(*source,false);',1)
        expected='second record binary support accepted'
    else:
        anchor='return source&&source==admitted_source_;';assert anchor in code
        code=code.replace(anchor,'return bool(source);',1)
        expected='default capability bypass reached device'
    result=compile_run(code,'canonical-admission-mutation-'+mutation)
    assert result.returncode!=0 and expected in result.stderr,result.stderr
