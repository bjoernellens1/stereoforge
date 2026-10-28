"""CPU-only guard and fake-stage probe of the dual ROS admission path.

This measures publisher control flow, not Daxa, transfers, ROS, or GPU time.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest
from test_kernel_variant_hotpath import body

SOURCE = Path(__file__).resolve().parents[1] / 'apps/ros2_depth_publisher.cpp'


def dual_admission(source):
    loop = body(source, 'while(rclcpp::ok()&&!trace_full')
    start = loop.index('if(depth&&dual){')
    end = loop.index('\n            else if(depth)', start)
    return loop[start:end]


def check(source):
    section = dual_admission(source)
    assert not re.search(r'\b(?:wait\w*|readback\w*|new|make_shared|make_unique|malloc|calloc|realloc|resize|reserve|memcpy|memmove|create_\w*)\s*(?:<[^;()]*>)?\s*\(', section)
    pieces = ['auto pair=dual_preprocess->enqueue(f)',
              'index->depth=pipeline->enqueue_preprocessed(pair.coarse)',
              'index->full=pair.full',
              'index->refined=refinement->process(index->depth,index->full)',
              'index->output=exporter->enqueue(index->refined.observations())']
    assert all(section.count(x) == 1 for x in pieces)
    assert [section.index(x) for x in pieces] == sorted(section.index(x) for x in pieces)
    assert section.count('enqueue(') == 2
    retire = body(source, 'auto retire=')
    release = ['p.output={}', 'p.refined={}', 'p.full={}', 'p.depth={}', 'p.capture={}',
               'exporter->collect()', 'refinement->collect()', 'pipeline->collect()', 'dual_preprocess->collect()']
    assert all(x in retire for x in release)
    assert [retire.index(x) for x in release] == sorted(retire.index(x) for x in release)
    assert 'if(drain)p.output.wait();else if(!p.output.ready())return;' in retire
    run = body(source, 'void run(')
    run = run[run.index('auto window_end='): ]
    cleanup = ['exporter->drain()', 'pipeline->drain()', 'p.output={};p.refined={};p.full={};p.depth={}',
               'exporter.reset()', 'refinement.reset()', 'pipeline.reset()', 'dual_preprocess.reset()', 'context->close()']
    assert all(x in run for x in cleanup)
    assert [run.index(x) for x in cleanup] == sorted(run.index(x) for x in cleanup)
    assert run.count('catch(const std::exception& e){reason="error";failure=e.what();}') >= 2


def test_dual_admission_guard():
    check(SOURCE.read_text())


@pytest.mark.parametrize('before,after', [
    ('auto pair=dual_preprocess->enqueue(f)', 'wait_idle();auto pair=dual_preprocess->enqueue(f)'),
    ('auto pair=dual_preprocess->enqueue(f)', 'malloc(8);auto pair=dual_preprocess->enqueue(f)'),
    ('index->depth=pipeline->enqueue_preprocessed(pair.coarse)', 'index->depth=pipeline->enqueue_preprocessed(pair.full)'),
    ('index->full=pair.full', 'index->full=pair.coarse'),
    ('p.output={};p.refined={};p.full={};p.depth={}', 'p.output={};p.full={};p.refined={};p.depth={}'),
    ('pipeline.reset();dual_preprocess.reset()', 'dual_preprocess.reset();pipeline.reset()'),
])
def test_dual_guard_rejects_mutations(before, after):
    source = SOURCE.read_text()
    assert before in source
    with pytest.raises(AssertionError):
        check(source.replace(before, after, 1))


def test_actual_dual_admission_fake_device(tmp_path):
    section = dual_admission(SOURCE.read_text())
    program = r'''
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <new>
static uint64_t allocations=0; static bool track=false;
void* operator new(std::size_t n){if(track)++allocations;if(auto p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept{std::free(p);} void operator delete(void* p,std::size_t) noexcept{std::free(p);}
struct Frame { uint64_t id=0; Frame observations()const{return *this;} };
struct Pair{Frame full,coarse;};
struct Dual{uint64_t n=0; __attribute__((noinline)) Pair enqueue(Frame f){++n;return {f,f};}};
struct Depth{uint64_t n=0; __attribute__((noinline)) Frame enqueue_preprocessed(Frame f){++n;return f;}};
struct Ref{uint64_t n=0; __attribute__((noinline)) Frame process(Frame a,Frame b){if(a.id!=b.id)std::abort();++n;return a;}};
struct Export{uint64_t n=0; __attribute__((noinline)) Frame enqueue(Frame f){++n;return f;}};
struct Pending{Frame depth,full,refined,output;uint64_t producer_enqueue_end=0;};
struct steady{};template<class T> uint64_t now_ns(){return 1;}
int main(){Dual a;Depth b;Ref c;Export d;auto *dual_preprocess=&a;auto *pipeline=&b;auto *refinement=&c;auto *exporter=&d;Pending slot,*index=&slot;bool depth=true,dual=true;Frame f;constexpr uint64_t n=100000;auto begin=std::chrono::steady_clock::now();track=true;
for(uint64_t i=1;i<=n;++i){f.id=i;
''' + section + r'''
if(index->output.id!=i)std::abort();}
track=false;double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count()/n;
std::printf("{\"frames\":%llu,\"ns_per_frame\":%.3f,\"allocations\":%llu,\"dual_enqueues\":%llu,\"depth_enqueues\":%llu,\"refinements\":%llu,\"exports\":%llu}\n",(unsigned long long)n,ns,(unsigned long long)allocations,(unsigned long long)a.n,(unsigned long long)b.n,(unsigned long long)c.n,(unsigned long long)d.n);
}
'''
    cpp = tmp_path / 'probe.cpp'
    binary = tmp_path / 'probe'
    cpp.write_text(program)
    subprocess.run(['c++', '-std=c++20', '-O2', str(cpp), '-o', str(binary)], check=True, capture_output=True, text=True)
    row = json.loads(subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout)
    print('ROS dual fake-stage CPU admission:', json.dumps(row, sort_keys=True))
    assert row['allocations'] == 0
    assert all(row[k] == row['frames'] for k in ('dual_enqueues', 'depth_enqueues', 'refinements', 'exports'))
