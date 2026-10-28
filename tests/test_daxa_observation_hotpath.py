"""Actual-source CPU lease probe. Mock device operations do not qualify GPU performance."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import pytest
from test_daxa_depth_hotpath import definition
ROOT=Path(__file__).resolve().parents[1]


def check_hot(source, adapter):
    hot=source
    for signature in ('void DaxaObservationFrame::wait(', 'bool DaxaObservationFrame::ready('):
        hot=hot.replace(definition(source,signature),'')
    hot+=adapter
    assert not re.search(r'\b(?:new|make_shared|make_unique|malloc|resize|reserve|push_back|memcpy|memmove|wait|ready|oldest_pending_submit_index|readback|create_context|sha256)\b', hot)
    assert 'std::array<daxa::BufferId,6>' in adapter and 'std::array<DaxaObservationImageView,4>' in adapter
    assert 'slot_->outputs[i+3]' in adapter and 'i<2?maps->eye_width:maps->width' in adapter
    assert '&slot_->generation,generation_' in adapter and 'slot_->submit' in adapter
    assert 'generation_==*live_generation_' in source


def test_guard_and_mutations():
    source=(ROOT/'src/daxa_observation.cpp').read_text()
    adapter=definition((ROOT/'src/daxa_depth.cpp').read_text(),'DaxaObservationFrame DaxaDepthPipeline::Frame::observations(')
    header=(ROOT/'include/stereoforge/daxa_observation.hpp').read_text()
    assert header.index('context_;') < header.index('owner_,lease_;') < header.index('maps_;')
    assert header.index('DaxaObservationFrame(std::shared_ptr') < header.index('public:',header.index('class DaxaObservationFrame {'))
    assert 'std::array<daxa::BufferId,6> planes_' in header and 'std::array<DaxaObservationImageView,4> images_' in header
    check_hot(source,adapter)
    for mutation in ['std::make_shared<int>();','ready();','memcpy(a,b,n);']:
        with pytest.raises(AssertionError):check_hot(source,adapter+mutation)
    with pytest.raises(AssertionError):check_hot(source,adapter.replace('outputs[i+3]','outputs[i]'))


def test_actual_adapter_lease_lifecycle_cpu_probe(tmp_path):
    compiler=shutil.which('c++')
    if not compiler:pytest.skip('C++ compiler unavailable')
    header=(ROOT/'include/stereoforge/daxa_observation.hpp').read_text()
    source=(ROOT/'src/daxa_observation.cpp').read_text()
    adapter=definition((ROOT/'src/daxa_depth.cpp').read_text(),'DaxaObservationFrame DaxaDepthPipeline::Frame::observations(')
    program=r'''
#include <array>
#include <memory>
#include <utility>
#include <stdexcept>
#include <string>
#include <atomic>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <stereoforge/canonical_rgb.hpp>
std::atomic<bool> track=false;unsigned allocations=0;
void* operator new(size_t n){if(track)++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}void operator delete(void* p,size_t)noexcept{std::free(p);}
namespace daxa {constexpr int QUEUE_MAIN=0;struct BufferId {uint64_t id=0;bool is_empty()const{return !id;}bool operator==(BufferId const&)const=default;};}
namespace stereoforge {
struct Device{unsigned polls=0,waits=0;void* get(){return this;}uint64_t oldest_pending_submit_index(){++polls;return 20;}struct Wait{int queue;uint64_t queue_submit_index,timeout;};void wait_on_submit(Wait){++waits;}};
struct DaxaFoundation{Device device;};
struct DaxaRectification{unsigned width=7,height=5,eye_width=11,source_height=13,stride_bytes=88;std::string original_identity=std::string(200,'c'),rectified_identity=std::string(200,'r');};
enum class DaxaPlaneFormat{Rgba8PackedUint32};
namespace detail{
SHARED_COMPLETION_HELPERS
}
}
'''+header[header.index('namespace stereoforge {'):]+r'''
namespace stereoforge {
enum class DaxaPreprocessPlane{OriginalLeft,OriginalRight,RectifiedLeft,RectifiedRight};
struct Pre{std::shared_ptr<const DaxaRectification> maps;CaptureLease src;auto calibration()const{return maps;}auto source()const{return src;}struct Plane{daxa::BufferId id;auto buffer()const{return id;}};Plane plane(DaxaPreprocessPlane p)const{return {{100+unsigned(p)}};}};
class DaxaDepthPipeline{public:struct Owner{std::shared_ptr<DaxaFoundation> context;};struct Slot{std::array<daxa::BufferId,9> outputs{};uint64_t generation=3,submit=9;Pre preprocessing;};class Frame{public:std::shared_ptr<Owner> owner_;std::shared_ptr<Slot> slot_;uint64_t generation_=3;void validate()const{if(!slot_||slot_->generation!=generation_)throw std::invalid_argument("stale depth");}DaxaObservationFrame observations()const;};};
}
'''+source[source.index('namespace stereoforge {'):]+ '\nnamespace stereoforge {\n'+adapter+'\n}\n'+r'''
using namespace stereoforge;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(){try{
 auto context=std::make_shared<DaxaFoundation>();auto owner=std::make_shared<DaxaDepthPipeline::Owner>();owner->context=context;
 auto slot=std::make_shared<DaxaDepthPipeline::Slot>();slot->preprocessing.maps=std::make_shared<DaxaRectification>();auto raw=std::make_shared<CaptureFrame>();raw->source_width=22;raw->height=13;raw->stride_bytes=88;raw->calibration_identity=std::string(200,'c');raw->camera_clock_domain=std::string(200,'t');slot->preprocessing.src=raw;for(unsigned i=0;i<9;++i)slot->outputs[i]={10+i};
 auto other_owner=std::make_shared<DaxaDepthPipeline::Owner>();other_owner->context=context;auto other_slot=std::make_shared<DaxaDepthPipeline::Slot>(*slot);DaxaDepthPipeline::Frame other{other_owner,other_slot};
 DaxaDepthPipeline::Frame input{owner,slot};auto initial=slot.use_count();uint64_t checksum=0;constexpr unsigned count=10000;
 track=true;auto start=std::chrono::steady_clock::now();
 for(unsigned n=0;n<count;++n){auto lease=input.observations();auto copy=lease;auto moved=std::move(copy);copy=lease;copy=copy;moved=copy;copy={};moved=other.observations();moved=lease;
 check(lease.generation()==3&&lease.submit_ticket()==9&&lease.layers()==2,"generation/ticket");check(lease.source()==slot->preprocessing.src&&lease.map_owner()==slot->preprocessing.maps&&lease.context()==context,"owner refs");
 for(unsigned i=0;i<6;++i){check(lease.buffer(DaxaObservationPlane(i))==slot->outputs[i+3],"plane association");check(lease.size_bytes(DaxaObservationPlane(i))==7*5*2*(i<4?4:8),"plane extent");checksum+=lease.buffer(DaxaObservationPlane(i)).id;}
 for(unsigned i=0;i<4;++i){auto image=lease.image(DaxaObservationImage(i));check(image.buffer().id==100+i&&image.width()==(i<2?11:7)&&image.height()==(i<2?13:5)&&image.row_stride_bytes()==image.width()*4&&image.format()==DaxaPlaneFormat::Rgba8PackedUint32,"RGB association");checksum+=image.size_bytes();}
 check(slot.use_count()==initial+2,"bounded retained slot refs");}
 double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();track=false;
 check(allocations==0&&context->device.polls==0&&context->device.waits==0,"hidden allocation or device polling");check(slot.use_count()==initial,"lease retained after iteration");
 auto canonical=std::make_shared<CaptureFrame>(*raw);canonical->source_kind=CaptureSourceKind::CanonicalRgbMetadata;canonical->stride_bytes=0;
 auto rgb=std::make_shared<CanonicalRgbSource>();rgb->backing=std::make_shared<int>(1);rgb->original_calibration_identity=raw->calibration_identity;rgb->rectified_calibration_identity=slot->preprocessing.maps->rectified_identity;
 for(unsigned i=0;i<4;++i){rgb->rgb[i].width=i<2?11:7;rgb->rgb[i].height=i<2?13:5;}canonical->canonical_rgb=rgb;slot->preprocessing.src=canonical;
 {auto lease=input.observations();check(lease.source()==canonical,"canonical source association");}
 rgb->rectified_calibration_identity="wrong";bool rejected=false;try{input.observations();}catch(std::invalid_argument const&){rejected=true;}check(rejected,"canonical calibration mismatch accepted");
 rgb->rectified_calibration_identity=slot->preprocessing.maps->rectified_identity;rgb->rgb[0].width=10;rejected=false;try{input.observations();}catch(std::invalid_argument const&){rejected=true;}check(rejected,"canonical original grid mismatch accepted");
 slot->preprocessing.src=raw;
 auto held=input.observations();slot->generation=4;bool stale=false;try{held.width();}catch(std::invalid_argument const&){stale=true;}check(stale,"stale generation accepted");slot->generation=3;
 auto weakslot=std::weak_ptr(slot);auto weakowner=std::weak_ptr(owner);input={};owner.reset();slot.reset();check(!weakslot.expired()&&!weakowner.expired()&&held.source(),"source producer not retained");held={};check(weakslot.expired()&&weakowner.expired(),"hidden ownership cycle");
 std::cout<<"{\"iterations\":"<<count<<",\"allocations\":"<<allocations<<",\"implicit_polls\":0,\"implicit_waits\":0,\"mean_cpu_ms\":"<<ms/count<<",\"checksum\":"<<checksum<<",\"planes\":6,\"rgb_descriptors\":4,\"stale_rejected\":true,\"retained_producer_released\":true}\n";
 }catch(std::exception const& e){track=false;std::cerr<<e.what();return 1;}}
'''
    frame_header=(ROOT/'include/stereoforge/daxa_frame.hpp').read_text()
    for name in ('DaxaPreprocessPlane','DaxaPlaneFormat'):
        actual=re.search(r'enum class '+name+r'[^;]+;',frame_header)[0]
        program=re.sub(r'enum class '+name+r'[^;]+;',lambda _:actual,program)
    frame_source=(ROOT/'include/stereoforge/daxa_frame.hpp').read_text()
    shared=definition(frame_source,'inline bool completed_submission(')+definition(frame_source,'inline bool available_slot(')
    program=program.replace('SHARED_COMPLETION_HELPERS',shared)
    code=tmp_path/'probe.cpp';code.write_text(program);binary=tmp_path/'probe'
    built=subprocess.run([compiler,'-std=c++20','-O3','-I',str(ROOT/'include'),str(code),'-o',str(binary)],capture_output=True,text=True,timeout=60)
    assert built.returncode==0,built.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    row=json.loads(result.stdout);print('Actual observation lease CPU-only probe:',json.dumps(row,sort_keys=True))
    assert row['iterations']==10000 and row['allocations']==row['implicit_polls']==row['implicit_waits']==0
    assert row['stale_rejected'] and row['retained_producer_released'] and row['checksum']>0
