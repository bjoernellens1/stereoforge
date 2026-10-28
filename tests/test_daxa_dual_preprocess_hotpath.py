"""CPU-only actual-source dual producer probe; fake device excludes Daxa and GPU cost."""
import json
import re
import subprocess
from pathlib import Path
import pytest
from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/daxa_dual_preprocess.cpp'


def check_hotpath(source):
    enqueue = body(source, 'DaxaDualPreprocess::Frame DaxaDualPreprocess::enqueue(')
    collect = body(source, 'size_t DaxaDualPreprocess::collect(')
    available = body(source, 'bool available(')
    forbidden = r'\b(?:make_shared|make_unique|create_buffer|allocate|readback|wait_idle|wait_on_submit|malloc|resize|reserve|push_back)\s*(?:<[^;()]*>)?\s*\('
    assert not re.search(forbidden, enqueue + collect), 'hot path allocates, waits or reads back'
    assert enqueue.count('std::memcpy(') == 1, 'one source copy required'
    assert enqueue.count('oldest_pending_submit_index(') == 1
    assert enqueue.index('validate_source(*source)') < enqueue.index('std::memcpy(')
    assert enqueue.index('e.available(slot,oldest)') < enqueue.index('std::memcpy(')
    assert 's->full.use_count()==1' in available and 's->coarse.use_count()==1' in available
    assert 'detail::completed_submission(s->submit,oldest)' in available
    assert 'catch(...)' in enqueue and 'slot->full->submit=slot->coarse->submit=slot->submit' in enqueue
    assert 'collect_garbage(' in enqueue or 'collect_garbage(' in collect, 'deferred graph resources require retirement'
    assert '.src_access=daxa::AccessConsts::TRANSFER_WRITE|daxa::AccessConsts::COMPUTE_SHADER_READ,.dst_access=daxa::AccessConsts::TRANSFER_WRITE' in source, 'reused raw buffer needs prior transfer and read dependency'
    assert 'for(unsigned resource=0;resource<ids.size();++resource){auto name="dual_"+std::to_string(index)+"_resource_"+std::to_string(resource);s->nodes.emplace_back(daxa::ExternalTaskBuffer({.buffer=ids[resource],.name=name}));' in source, 'each registered graph resource requires a distinct name'
    assert '.reads(v[1],v[2]).writes(v[4],v[5])' in source, 'decode dependencies changed'
    assert '.reads(v[2],v[4],v[5],v[18],v[19],v[20],v[21]).writes(v[6],v[7],v[8],v[9],v[10],v[11])' in source, 'full rectify dependencies changed'
    assert '.reads(v[3],v[4],v[5],v[22],v[23],v[24],v[25]).writes(v[12],v[13],v[14],v[15],v[16],v[17])' in source, 'coarse rectify dependencies changed'


def test_dual_hotpath_guard():
    check_hotpath(SOURCE.read_text())


@pytest.mark.parametrize('before,after', [
    ('std::memcpy(target,source->packed_yuyv.data(),bytes);', 'std::memcpy(target,source->packed_yuyv.data(),bytes);std::memcpy(target,source->packed_yuyv.data(),bytes);'),
    ('s->coarse.use_count()==1', 'true'),
    ('std::to_string(resource);s->nodes.emplace_back', 'std::to_string(0);s->nodes.emplace_back'),
    ('"dual_"+std::to_string(index)+"_resource_"+std::to_string(resource)', '"dual resource"'),
    ('TRANSFER_WRITE|daxa::AccessConsts::COMPUTE_SHADER_READ', 'TRANSFER_WRITE'),
    ('.reads(v[3],v[4],v[5],v[22],v[23],v[24],v[25])', '.reads(v[3],v[22],v[23],v[24],v[25])'),
    ('auto oldest=e.context->device.oldest_pending_submit_index();', 'e.context->device.wait_idle();auto oldest=e.context->device.oldest_pending_submit_index();'),
])
def test_dual_guard_rejects_mutations(before, after):
    source = SOURCE.read_text()
    assert before in source
    with pytest.raises(AssertionError):
        check_hotpath(source.replace(before, after, 1))


def definition(text, signature):
    start = text.index(signature)
    chunk = body(text, signature)
    begin = text.index('{', start)
    return text[start:begin] + '{' + chunk + '}'


def test_actual_enqueue_fake_device(tmp_path):
    source = SOURCE.read_text()
    available = definition(source, 'bool available(').replace('bool available(', 'bool Engine::available(', 1)
    collect = definition(source, 'size_t collect_at(').replace('size_t collect_at(', 'size_t Engine::collect_at(', 1)
    enqueue = definition(source, 'DaxaDualPreprocess::Frame DaxaDualPreprocess::enqueue(').replace('std::memcpy(', 'probe_copy(').replace('std::memset(', 'probe_set(')
    code = r'''
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <atomic>
#include <cstdlib>
std::atomic<unsigned long> allocations{0}; bool tracking=false;
void* operator new(std::size_t n){if(tracking)++allocations;if(void* p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept{std::free(p);} void operator delete(void* p,std::size_t) noexcept{std::free(p);}
void require(bool b,const char* m){if(!b)throw std::invalid_argument(m);}
namespace daxa {constexpr int QUEUE_MAIN=0;}
namespace detail {bool completed_submission(uint64_t ticket,uint64_t oldest){return !ticket||ticket<=oldest;}}
struct Capture {std::vector<uint8_t> packed_yuyv;}; using CaptureLease=std::shared_ptr<const Capture>;
struct Map {void validate_source(const Capture& s)const{require(s.packed_yuyv.size()==3686400,"bad source");}};
struct Child {CaptureLease source;uint64_t submit=0,generation=0;};
struct Device {uint64_t oldest=0,latest=0,polls=0,gc=0; uint64_t oldest_pending_submit_index(){++polls;return oldest;} uint64_t latest_queue_submit_index(int){return latest;} void collect_garbage(){++gc;} struct Address{uint8_t* p;uint8_t* value(){return p;}}; Address buffer_host_address(uintptr_t p){return {(uint8_t*)p};}};
struct Context {Device device;};
struct Graph {Device* d;bool fail=false;void execute(std::initializer_list<int>){++d->latest;if(fail)throw std::runtime_error("partial");}};
struct Engine {struct Slot{std::shared_ptr<Child> full=std::make_shared<Child>(),coarse=std::make_shared<Child>();uintptr_t staging=0;std::unique_ptr<Graph> graph;uint64_t submit=0;};std::shared_ptr<Context> context;std::shared_ptr<Map> full,coarse;std::vector<std::shared_ptr<Slot>> slots;uint64_t next_generation=1;void check_thread(){} bool available(const std::shared_ptr<Slot>&,uint64_t)const;size_t collect_at(uint64_t);};
struct DaxaFrame {std::shared_ptr<Engine> owner;std::shared_ptr<Child> slot;DaxaFrame()=default;DaxaFrame(std::shared_ptr<Engine> a,std::shared_ptr<Child> b):owner(a),slot(b){}};
struct DaxaDualPreprocess {struct Frame{DaxaFrame full,coarse;};std::shared_ptr<Engine> impl_;Frame enqueue(CaptureLease);};
uint64_t copies=0,bytes_copied=0;bool real_copy=false;
void* probe_copy(void* d,const void* s,size_t n){++copies;bytes_copied+=n;return real_copy?std::memcpy(d,s,n):d;}
void* probe_set(void* d,int v,size_t n){return real_copy?std::memset(d,v,n):d;}
'''+available+'\n'+collect+'\n'+enqueue+r'''
int main(){try {
 auto context=std::make_shared<Context>();auto e=std::make_shared<Engine>();e->context=context;e->full=std::make_shared<Map>();e->coarse=std::make_shared<Map>();
 std::array<std::vector<uint8_t>,2> staging;for(auto& b:staging){b.resize(3686400);auto s=std::make_shared<Engine::Slot>();s->staging=(uintptr_t)b.data();s->graph=std::make_unique<Graph>();s->graph->d=&context->device;e->slots.push_back(s);}
 auto source=std::make_shared<Capture>();source->packed_yuyv.resize(3686400,173);DaxaDualPreprocess stage{e};
 {auto a=stage.enqueue(source);context->device.oldest=context->device.latest;auto b=stage.enqueue(source);context->device.oldest=context->device.latest;bool rejected=false;try{stage.enqueue(source);}catch(const std::runtime_error&){rejected=true;}require(rejected,"retained children recycled");a.full={};rejected=false;try{stage.enqueue(source);}catch(const std::runtime_error&){rejected=true;}require(rejected,"coarse child did not retain slot");}
 e->collect_at(context->device.oldest);e->slots[0]->graph->fail=true;bool partial=false;try{stage.enqueue(source);}catch(const std::runtime_error&){partial=true;}require(partial&&e->slots[0]->full->source==source&&e->slots[0]->coarse->source==source&&e->slots[0]->submit==context->device.latest&&e->slots[0]->full->submit==context->device.latest&&e->slots[0]->coarse->submit==context->device.latest,"partial retention lost");require(!e->collect_at(context->device.oldest),"partial retired early");context->device.oldest=context->device.latest;e->slots[0]->graph->fail=false;require(e->collect_at(context->device.oldest)==1,"partial not retired on completion");
 for(bool copy:{false,true}){real_copy=copy;unsigned count=copy?128:10000;copies=bytes_copied=allocations=0;auto polls=context->device.polls;tracking=true;auto start=std::chrono::steady_clock::now();for(unsigned i=0;i<count;++i){auto out=stage.enqueue(source);context->device.oldest=context->device.latest;}auto ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/count;tracking=false;require(allocations==0&&copies==count&&bytes_copied==uint64_t(count)*3686400&&context->device.polls-polls==count,"hotpath regression");if(copy)require(std::memcmp(staging[0].data(),source->packed_yuyv.data(),3686400)==0,"copy bytes wrong");std::cout<<"{\"copy_enabled\":"<<(copy?"true":"false")<<",\"frames\":"<<count<<",\"ns_per_frame\":"<<ns<<",\"copies_per_frame\":1,\"bytes_per_frame\":3686400,\"allocations\":"<<allocations<<"}\n";}
 return 0;}catch(const std::exception& e){tracking=false;std::cerr<<e.what()<<'\n';return 1;}}
'''
    cpp=tmp_path/'probe.cpp'; binary=tmp_path/'probe'; cpp.write_text(code)
    build=subprocess.run(['c++','-std=c++20','-O2',str(cpp),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows)==2 and all(row['allocations']==0 for row in rows)
    print(result.stdout)
