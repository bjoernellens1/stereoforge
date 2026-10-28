"""Guard and CPU-only fake-stage probe for ROS refinement admission.

This measures extracted production admission statements with stub stages; it does
not measure ROS, Vulkan, camera, memory transfers, or end-to-end latency.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest
from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'apps/ros2_depth_publisher.cpp'


def admission(source):
    loop = body(source, 'while(rclcpp::ok()&&!trace_full')
    start = loop.index('if(depth){index->depth=pipeline->enqueue(f);')
    end = loop.index('index->export_enqueue_end=', start)
    return loop[start:end]


def check(source):
    section = admission(source)
    assert not re.search(r'\b(?:wait\w*|readback\w*|new|make_shared|make_unique|malloc|calloc|realloc|resize|reserve|memcpy|memmove|create_\w*)\s*(?:<[^;()]*>)?\s*\(', section)
    for fragment in ('index->depth=pipeline->enqueue(f)', 'index->full=full_preprocess->enqueue(f)',
                     'index->refined=refinement->process(index->depth,index->full)',
                     'exporter->enqueue(experimental?index->refined.observations():index->depth.observations())'):
        assert fragment in section
    assert section.index('index->depth=pipeline->enqueue(f)') < section.index('index->full=full_preprocess->enqueue(f)') < section.index('index->refined=refinement->process(') < section.index('exporter->enqueue(')
    retire = body(source, 'auto retire=')
    assert 'if(drain)p.output.wait();else if(!p.output.ready())return;' in retire
    releases = ['p.output={}', 'p.refined={}', 'p.full={}', 'p.depth={}', 'exporter->collect()', 'refinement->collect()', 'full_preprocess->collect_completed_sources()', 'pipeline->collect()']
    assert all(x in retire for x in releases)
    assert [retire.index(x) for x in releases] == sorted(retire.index(x) for x in releases)
    assert 'experimental?p.refined.ready():p.depth.ready()' in retire
    run = body(source, 'void run(')
    catch = 'catch(const std::exception& e){reason="error";failure=e.what();}'
    assert run.count(catch) >= 2
    cleanup = 'for(auto& p:pending){p.output={};p.refined={};p.full={};p.depth={};p.rgb={};p.capture={};}'
    assert cleanup in run
    assert 'exporter->drain()' in run and 'pipeline->drain()' in run
    assert run.index('exporter->drain()') < run.index('pipeline->drain()') < run.index(cleanup) < run.index('refinement.reset()') < run.index('full_preprocess.reset()') < run.index('pipeline.reset()') < run.index('context->close()')


def test_refinement_admission_and_retirement_contract():
    check(SOURCE.read_text())


@pytest.mark.parametrize('mutation', ['wait', 'allocation', 'wrong_source', 'release_order', 'wrong_ready', 'exception_cleanup'])
def test_guard_rejects_mutations(mutation):
    source = SOURCE.read_text()
    replacements = {
        'wait': ('index->full=full_preprocess->enqueue(f)', 'wait_idle();index->full=full_preprocess->enqueue(f)'),
        'allocation': ('index->full=full_preprocess->enqueue(f)', 'malloc(8);index->full=full_preprocess->enqueue(f)'),
        'wrong_source': ('index->full=full_preprocess->enqueue(f)', 'index->full=full_preprocess->enqueue(other)'),
        'release_order': ('p.output={};p.refined={};p.full={};p.depth={}', 'p.output={};p.full={};p.refined={};p.depth={}'),
        'wrong_ready': ('experimental?p.refined.ready():p.depth.ready()', 'p.depth.ready()'),
        'exception_cleanup': ('pipeline->drain();else preprocess->collect_completed_sources()', 'else preprocess->collect_completed_sources()'),
    }
    old, new = replacements[mutation]
    assert old in source
    with pytest.raises(AssertionError):
        check(source.replace(old, new, 1))


def test_actual_admission_fake_device_cpu_cost(tmp_path):
    code = admission(SOURCE.read_text())
    program = r'''
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
static uint64_t allocations=0; static bool track=false;
void* operator new(std::size_t n){if(track)++allocations;if(auto p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept{std::free(p);} void operator delete(void* p,std::size_t) noexcept{std::free(p);}
struct Frame { uint64_t id=0; Frame observations()const{return *this;} };
struct Stage { uint64_t enqueues=0,processes=0,waits=0; __attribute__((noinline)) Frame enqueue(Frame f){++enqueues;return f;} __attribute__((noinline)) Frame process(Frame a,Frame b){if(a.id!=b.id)std::abort();++processes;return a;} };
struct Pending{Frame depth,full,refined,rgb,output;uint64_t producer_enqueue_end=0;};
struct steady{}; template<class T> uint64_t now_ns(){return 1;}
int main(){ Stage a,b,c,d,e;auto *pipeline=&a,*full_preprocess=&b,*refinement=&c,*exporter=&d,*preprocess=&e;Pending slot,*index=&slot;bool depth=true,experimental=true;Frame f;constexpr uint64_t n=100000;auto start=std::chrono::steady_clock::now();track=true;
for(uint64_t i=1;i<=n;++i){f.id=i;
''' + code + r'''
if(index->output.id!=i)std::abort();}
track=false;double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/n;
std::printf("{\"frames\":%llu,\"ns_per_frame\":%.3f,\"allocations\":%llu,\"waits\":%llu,\"depth_enqueues\":%llu,\"full_enqueues\":%llu,\"refinements\":%llu,\"exports\":%llu}\n",(unsigned long long)n,ns,(unsigned long long)allocations,(unsigned long long)(a.waits+b.waits+c.waits+d.waits),(unsigned long long)a.enqueues,(unsigned long long)b.enqueues,(unsigned long long)c.processes,(unsigned long long)d.enqueues);
}
'''
    cpp = tmp_path / 'probe.cpp'
    binary = tmp_path / 'probe'
    cpp.write_text(program)
    subprocess.run(['c++', '-std=c++20', '-O2', str(cpp), '-o', str(binary)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    row = json.loads(result.stdout)
    print('ROS refinement fake-stage CPU admission:', json.dumps(row, sort_keys=True))
    assert row['allocations'] == row['waits'] == 0
    assert all(row[k] == row['frames'] for k in ('depth_enqueues', 'full_enqueues', 'refinements', 'exports'))
