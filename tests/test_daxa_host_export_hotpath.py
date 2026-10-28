"""Compile the entire production export implementation against bounded CPU API mocks.

Mocks model ticketed buffer copies, not GPU correctness or driver allocations.
"""
from pathlib import Path
import re
import shutil
import subprocess
import pytest
from test_daxa_depth_hotpath import definition

ROOT = Path(__file__).resolve().parents[1]


def test_hotpath_guards_and_mutations():
    source = (ROOT/'src/daxa_host_export.cpp').read_text()
    hot = definition(source, 'template<class Input>void submit(')
    hot += definition(source, 'DaxaHostExport::Frame DaxaHostExport::enqueue(const DaxaObservationFrame&')
    hot += definition(source, 'DaxaHostExport::Frame DaxaHostExport::enqueue(const DaxaFrame&')
    def check(text):
        assert not re.search(r'\b(create_buffer|create_memory|wait_on_submit|wait_idle|readback|memcpy|make_shared|resize|push_back)\s*\(', text)
        assert text.count('copy_buffer_to_buffer(') == 2
        assert '.src_offset=0,.dst_offset=0,.size=plane_bytes});' in text
        assert '.dst_offset=config.depth?plane_bytes:0,.size=plane_bytes});' in text
        assert 'AccessConsts::HOST_READ' in text
    check(hot)
    for mutation in ('d.create_buffer({});', 'd.wait_idle();', 'memcpy(a,b,n);', 'd.create_memory({});'):
        with pytest.raises(AssertionError):
            check(hot+mutation)
    for bad in (hot.replace('.size=plane_bytes', '.size=plane_bytes*2'), hot.replace('AccessConsts::HOST_READ', 'AccessConsts::TRANSFER_READ')):
        with pytest.raises(AssertionError):
            check(bad)
    assert 'std::array<std::shared_ptr<Slot>,3>' in source
    assert 'requirements.memory_type_bits=1u<<i' in source
    assert 's.copy_ticket=d.submit_commands' in source
    assert 'if(latest>before)s.copy_ticket=latest' in source


MOCK = r'''
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <variant>
#include <utility>
#include <limits>
#include <iostream>
#include <algorithm>
constexpr uint32_t VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT=1,VK_MEMORY_PROPERTY_HOST_COHERENT_BIT=2,VK_MEMORY_PROPERTY_HOST_CACHED_BIT=4;
struct VkPhysicalDeviceMemoryProperties {uint32_t memoryTypeCount=0;struct Type {uint32_t propertyFlags=0;} memoryTypes[32];};
VkPhysicalDeviceMemoryProperties properties;
void vkGetPhysicalDeviceMemoryProperties(void*,VkPhysicalDeviceMemoryProperties* out){*out=properties;}
void* daxa_dvc_get_vk_physical_device(void* d){return d;}
namespace daxa {
constexpr int QUEUE_MAIN=0;
struct BufferId{unsigned index=0;bool is_empty()const{return !index;}bool operator==(const BufferId&)const=default;};
struct MemoryRequirements{uint64_t size=0,alignment=4;uint32_t memory_type_bits=~0u;};
struct MemoryFlagBits{static constexpr int HOST_ACCESS_RANDOM=1;};
struct MemoryBlock{std::shared_ptr<int> storage;};
struct BufferInfo{size_t size=0;std::string name;};
struct MemoryInfo{MemoryRequirements requirements;int flags=0;};
struct BlockBufferInfo{BufferInfo buffer_info;MemoryBlock memory_block;};
struct Name{std::string name;};
struct AccessConsts{static constexpr int COMPUTE_SHADER_WRITE=1,TRANSFER_WRITE=2,TRANSFER_READ=4,HOST_READ=8;};
struct Barrier{int src_access,dst_access;};
struct BufferCopyInfo{BufferId src_buffer,dst_buffer;size_t src_offset=0,dst_offset=0,size=0;};
struct Command{std::array<BufferCopyInfo,2> copies{};unsigned count=0;};
struct Recorder{Command commands;std::vector<Barrier>* barriers;void pipeline_barrier(Barrier b){barriers->push_back(b);}void copy_buffer_to_buffer(BufferCopyInfo c){commands.copies[commands.count++]=c;}Command complete_current_commands(){return commands;}};
struct Submit{std::span<const Command> command_lists;};
struct Wait{int queue;uint64_t queue_submit_index,timeout;};
struct Device {
 std::array<std::vector<std::byte>,128> buffers{};unsigned next=1,allocations=0,memories=0,destroyed=0,recordings=0,submits=0,waits=0;
 unsigned fail_memory=0,fail_buffer=0;bool fail_mapping=false,fail_record=false,fail_submit=false,partial_submit=false,wait_failure=false;
 uint64_t latest=10,completed=10;uint32_t compatible=~0u;std::vector<Barrier> barriers;std::vector<BufferCopyInfo> submitted_copies;
 void* get(){return this;}
 MemoryRequirements buffer_memory_requirements(BufferInfo info){return {info.size,4,compatible};}
 MemoryBlock create_memory(MemoryInfo info){++memories;if(fail_memory==memories)throw std::runtime_error("memory failure");if(!(info.requirements.memory_type_bits&compatible))throw std::runtime_error("bad memory");return {std::make_shared<int>(1)};}
 BufferId create_buffer_from_memory_block(BlockBufferInfo info){++allocations;if(fail_buffer==allocations)throw std::runtime_error("buffer failure");auto id=next++;buffers[id].resize(info.buffer_info.size);return {id};}
 std::optional<std::byte*> buffer_host_address(BufferId b){if(fail_mapping)throw std::runtime_error("map failure");return buffers[b.index].data();}
 void destroy_buffer(BufferId b){++destroyed;buffers[b.index].clear();}
 void collect_garbage(){}
 Recorder create_command_recorder(Name){++recordings;if(fail_record)throw std::runtime_error("record failure");return {{},&barriers};}
 uint64_t submit_commands(Submit info){++submits;if(fail_submit&&!partial_submit)throw std::runtime_error("submit failed before acceptance");++latest;for(auto& command:info.command_lists)for(unsigned i=0;i<command.count;++i){auto c=command.copies[i];submitted_copies.push_back(c);if(c.src_offset+c.size>buffers[c.src_buffer.index].size()||c.dst_offset+c.size>buffers[c.dst_buffer.index].size())throw std::runtime_error("copy outside buffer");std::memcpy(buffers[c.dst_buffer.index].data()+c.dst_offset,buffers[c.src_buffer.index].data()+c.src_offset,c.size);}if(fail_submit)throw std::runtime_error("accepted then threw");return latest;}
 uint64_t latest_queue_submit_index(int){return latest;}
 uint64_t oldest_pending_submit_index(){return completed;}
 void wait_on_submit(Wait w){++waits;if(wait_failure)throw std::runtime_error("timeout");if(w.timeout!=30'000'000'000ull)throw std::runtime_error("unbounded timeout");completed=std::max(completed,w.queue_submit_index);}
 BufferId source(size_t size,unsigned seed){auto id=next++;buffers[id].resize(size);for(size_t i=0;i<size;++i)buffers[id][i]=std::byte((i+seed)%251);return {id};}
};
}
namespace stereoforge {
struct CaptureFrame {unsigned frame_id=1;};using CaptureLease=std::shared_ptr<const CaptureFrame>;
struct DaxaFoundation{daxa::Device device;};
struct DaxaRectification{uint32_t width=7,height=5;};
namespace detail{inline bool completed_submission(uint64_t submitted,uint64_t oldest){return !submitted||submitted<=oldest;}inline bool available_slot(long refs,uint64_t submitted,uint64_t oldest){return refs==1&&completed_submission(submitted,oldest);}}
enum class DaxaObservationPlane{Depth};enum class DaxaObservationImage{RectifiedLeft};enum class DaxaPreprocessPlane{RectifiedLeft};
struct MockImage{daxa::BufferId id;uint32_t w,h;auto buffer()const{return id;}auto width()const{return w;}auto height()const{return h;}};
struct InputState{std::shared_ptr<DaxaFoundation> context;std::shared_ptr<const DaxaRectification> maps;CaptureLease source;daxa::BufferId depth,rgb;uint64_t ticket=10;uint64_t generation=1;uint32_t w=7,h=5;};
struct DaxaFrame {std::shared_ptr<InputState> state;uint64_t generation_=1;void validate()const{if(!state||state->generation!=generation_)throw std::invalid_argument("stale producer");}explicit operator bool()const{return bool(state);}uint64_t context_id()const{validate();return reinterpret_cast<uintptr_t>(state->context->device.get());}auto calibration()const{validate();return state->maps;}auto source()const{validate();return state->source;}uint64_t submit_ticket()const{validate();return state->ticket;}auto plane(DaxaPreprocessPlane)const{validate();return MockImage{state->rgb,state->w,state->h};}};
struct DaxaObservationFrame:DaxaFrame {uint32_t width()const{validate();return state->w;}uint32_t height()const{validate();return state->h;}size_t size_bytes(DaxaObservationPlane)const{validate();return size_t(state->w)*state->h*8;}auto buffer(DaxaObservationPlane)const{validate();return state->depth;}auto image(DaxaObservationImage)const{validate();return MockImage{state->rgb,state->w,state->h};}};
}
'''

CASES = r'''
using namespace stereoforge;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class F>void rejects(F fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}check(failed,"expected rejection");}
DaxaObservationFrame input(std::shared_ptr<DaxaFoundation> c,std::shared_ptr<const DaxaRectification> maps){auto s=std::make_shared<InputState>();s->context=c;s->maps=maps;s->source=std::make_shared<CaptureFrame>();s->depth=c->device.source(280,3);s->rgb=c->device.source(140,29);uint32_t nan=0x7fc12345;std::memcpy(c->device.buffers[s->depth.index].data(),&nan,4);return {{s,1}};}
void setup(){properties={};properties.memoryTypeCount=3;properties.memoryTypes[0].propertyFlags=3;properties.memoryTypes[1].propertyFlags=7;properties.memoryTypes[2].propertyFlags=1;}
int main(){try{
 setup();auto c=std::make_shared<DaxaFoundation>();auto maps=std::make_shared<DaxaRectification>();auto p=input(c,maps);
 for(auto cfg:{DaxaHostExportConfig{1,true,false},DaxaHostExportConfig{4,true,false},DaxaHostExportConfig{2,false,false}})rejects([&]{DaxaHostExport x(c,p,cfg);});
 DaxaHostExport x(c,p,{2,true,true});check(x.memory().type_index==1&&x.memory().host_cached&&x.memory().property_flags==7&&x.memory().bytes_per_slot==280&&x.requested_bytes()==560,"cached exact memory accounting");
 auto allocated=c->device.allocations;auto memory=c->device.memories;auto a=x.enqueue(p);check(a.producer_ticket()==10&&a.copy_ticket()==11&&!a.ready(),"copy vs producer completion");rejects([&]{a.depth_bytes();});check(c->device.waits==0,"implicit wait");
 check(c->device.submitted_copies.size()==2,"requested copies");auto dc=c->device.submitted_copies[0],rc=c->device.submitted_copies[1];check(dc.src_buffer==p.state->depth&&dc.src_offset==0&&dc.dst_offset==0&&dc.size==140&&rc.src_buffer==p.state->rgb&&rc.dst_offset==140&&rc.size==140,"exact left-only offsets");
 auto b=x.enqueue(p);auto recordings=c->device.recordings,submits=c->device.submits;DaxaObservationFrame empty;rejects([&]{x.enqueue(empty);});check(c->device.recordings==recordings&&c->device.submits==submits,"full ring before source work");
 c->device.wait_failure=true;rejects([&]{a.wait();});check(!a.ready(),"failed wait cannot authorize read");c->device.wait_failure=false;a.wait();check(a.ready()&&!b.ready(),"inclusive completion boundary");check(std::equal(a.depth_bytes().begin(),a.depth_bytes().end(),c->device.buffers[p.state->depth.index].begin()),"bit exact NaN depth bytes");check(std::equal(a.rgba_bytes().begin(),a.rgba_bytes().end(),c->device.buffers[p.state->rgb.index].begin()),"bit exact RGBA");
 check(x.collect()==1&&x.collect()==0,"producer retirement once");auto held=a;a={};rejects([&]{x.enqueue(p);});held={};a=x.enqueue(p);check(a.copy_ticket()==13,"completed released slot reuse");
 check(c->device.allocations==allocated&&c->device.memories==memory,"no enqueue memory or buffers");
 auto producer=p.state;std::weak_ptr<InputState> weak=producer;p={};producer.reset();check(!weak.expired(),"source retained pending copy");x.drain();check(weak.expired(),"producer retired after completed copy despite retained host frames"); // p state is also referenced by original prototype metadata only until collect
 a={};b={};check(weak.expired(),"producer retired once all pending copy leases collected");
 p=input(c,maps);auto bad=p;bad.state=std::make_shared<InputState>(*p.state);bad.state->context=std::make_shared<DaxaFoundation>();rejects([&]{x.enqueue(bad);});bad.state->context=c;bad.state->maps=std::make_shared<DaxaRectification>();rejects([&]{x.enqueue(bad);});bad.state->maps=std::shared_ptr<const DaxaRectification>(maps.get(),[](const DaxaRectification*){});rejects([&]{x.enqueue(bad);});bad.state->maps=maps;bad.state->w=8;rejects([&]{x.enqueue(bad);});bad.state->w=7;++bad.state->generation;rejects([&]{x.enqueue(bad);});
 bool rejected=false;std::thread other([&]{try{x.collect();}catch(const std::exception&){rejected=true;}});other.join();check(rejected,"cross-thread collect rejected");rejected=false;std::thread other2([&]{try{x.enqueue(p);}catch(const std::exception&){rejected=true;}});other2.join();check(rejected,"cross-thread enqueue rejected");rejected=false;std::thread other3([&]{try{x.drain();}catch(const std::exception&){rejected=true;}});other3.join();check(rejected,"cross-thread drain rejected");
 auto stale=x.enqueue(p);auto slot=x.impl_->slots[0];++slot->generation;check(!stale,"stale checked bool");rejects([&]{stale.width();});--slot->generation;auto moved=std::move(stale);check(!stale&&bool(moved),"moved handle");rejects([&]{stale.wait();});moved.wait();moved={};slot.reset();x.collect();x.impl_->slots[0]->generation=UINT64_MAX;rejects([&]{x.enqueue(p);});x.impl_->slots[0]->generation=1;
 for(unsigned mode=0;mode<3;++mode){c->device.fail_record=mode==0;c->device.fail_submit=mode!=0;c->device.partial_submit=mode==2;auto prior=c->device.latest;rejects([&]{x.enqueue(p);});auto& s=*x.impl_->slots[0];if(mode==2){check(s.copy_ticket==prior+1&&s.producer.index()!=0,"partial submit retains slot/source");check(x.collect()==0,"partial ticket incomplete");x.drain();}else check(s.copy_ticket==0&&s.producer.index()==0,"unsubmitted producer released");c->device.fail_record=c->device.fail_submit=c->device.partial_submit=false;}
 c->device.completed=c->device.latest;
 {DaxaHostExport only(c,static_cast<const DaxaFrame&>(p),{2,false,true});auto first=c->device.submitted_copies.size();auto f=only.enqueue(static_cast<const DaxaFrame&>(p));f.wait();check(c->device.submitted_copies.size()==first+1&&f.depth_bytes().empty()&&f.rgba_bytes().size()==140,"RGB-only no depth");rejects([&]{DaxaHostExport invalid(c,static_cast<const DaxaFrame&>(p),{2,true,false});});}
 {DaxaHostExport depth(c,p,{2,true,false});auto first=c->device.submitted_copies.size();auto f=depth.enqueue(p);f.wait();check(c->device.submitted_copies.size()==first+1&&f.rgba_bytes().empty(),"depth-only no RGB");}
 DaxaHostExport::Frame retained;{DaxaHostExport temporary(c,p,{2,true,false});retained=temporary.enqueue(p);}retained.wait();auto copied=retained;retained={};check(copied.depth_bytes().size()==140&&copied.source()==p.source(),"destination survives exporter destruction");copied={};
 {auto cc=std::make_shared<DaxaFoundation>();auto pp=input(cc,maps);cc->device.compatible=1;DaxaHostExport fallback(cc,pp,{2,true,false});check(!fallback.memory().host_cached&&fallback.memory().type_index==0&&fallback.memory().property_flags==3,"coherent fallback");}
 {auto cc=std::make_shared<DaxaFoundation>();auto pp=input(cc,maps);cc->device.compatible=4;rejects([&]{DaxaHostExport unavailable(cc,pp,{2,true,false});});check(cc->device.allocations==0,"noncoherent rejected before allocation");}
 for(unsigned mode=0;mode<3;++mode){auto cc=std::make_shared<DaxaFoundation>();auto pp=input(cc,maps);cc->device.fail_memory=mode==0?2:0;cc->device.fail_buffer=mode==1?2:0;cc->device.fail_mapping=mode==2;rejects([&]{DaxaHostExport failure(cc,pp,{2,true,false});});check(cc->device.destroyed==(mode==2?1:1),"constructor cleans all created buffers");}
 std::cout<<"actual host export CPU cases passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
'''


@pytest.mark.parametrize("mutation", [None,"double_depth","rgb_offset","producer_ready","uncached_first","release_producer"])
def test_actual_source_cpu_probe(tmp_path, mutation):
    compiler = shutil.which('c++')
    assert compiler, 'actual-source CPU export probe requires C++20 compiler'
    header = (ROOT/'include/stereoforge/daxa_host_export.hpp').read_text()
    # Expose slot controls only to inject otherwise unreachable stale/overflow states.
    header = header[header.index('namespace stereoforge {'):].replace('    struct Slot;', 'public: struct Slot;', 1)
    source = (ROOT/'src/daxa_host_export.cpp').read_text()
    source = source[source.index('namespace stereoforge {'):]
    mutations = {
        'double_depth': ('.src_offset=0,.dst_offset=0,.size=plane_bytes}', '.src_offset=0,.dst_offset=0,.size=plane_bytes*2}'),
        'rgb_offset': ('.dst_offset=config.depth?plane_bytes:0', '.dst_offset=0'),
        'producer_ready': ('completed_submission(slot_->copy_ticket,', 'completed_submission(slot_->producer_ticket,'),
        'uncached_first': ('(pass||bool(flags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT))', 'true'),
        'release_producer': ('s.producer=input;', 's.producer=std::monostate{};'),
    }
    if mutation:
        before, after = mutations[mutation]
        assert before in source
        source = source.replace(before, after)
    cpp = tmp_path/'host_export_probe.cpp'
    cpp.write_text(MOCK+header+source+CASES)
    binary = tmp_path/'host_export_probe'
    command = [compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pthread', str(cpp), '-o', str(binary)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    if mutation:
        assert result.returncode != 0, f'production mutation escaped probe: {mutation}'
    else:
        assert result.returncode == 0, result.stdout+result.stderr
        assert 'actual host export CPU cases passed' in result.stdout


def test_native_frame_ticket_accessor_is_checked(tmp_path):
    source = (ROOT/'src/daxa_frame.cpp').read_text()
    method = definition(source, 'uint64_t DaxaFrame::submit_ticket(')
    program = r'''
#include <cstdint>
#include <memory>
#include <stdexcept>
namespace stereoforge {class DaxaFrame {public:struct Slot {uint64_t submit=17;};std::shared_ptr<Slot> slot_;void validate()const{if(!slot_)throw std::invalid_argument("empty");}uint64_t submit_ticket()const;};
'''+method+r'''}
int main(){stereoforge::DaxaFrame f;try{f.submit_ticket();return 1;}catch(const std::invalid_argument&){}f.slot_=std::make_shared<stereoforge::DaxaFrame::Slot>();return f.submit_ticket()==17?0:2;}
'''
    cpp=tmp_path/'ticket.cpp';cpp.write_text(program)
    binary=tmp_path/'ticket'
    result=subprocess.run(['c++','-std=c++20',str(cpp),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert subprocess.run([str(binary)],check=False).returncode==0
