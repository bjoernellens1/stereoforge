"""Actual offline timing/CPU sampler controls, not GPU or SDK performance."""
import ast
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import weakref

import pytest
from benchmarks.zed_comparison import performance as p, schema
from tests.test_zed_performance import qualified_abba, record, available, write_json
from test_zed_schema_io_guard import IOProbe, CHUNK
from test_zed_native_timing_metadata import timed_loop

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT/'artifacts/zed-task6/performance'
PYTHON = 'benchmarks/zed_comparison/performance.py'
SDK = 'benchmarks/zed_comparison/export_sdk.cpp'
SAMPLER = 'benchmarks/zed_comparison/sdk_benchmark.hpp'
NATIVE = {
    'apps/daxa_depth_benchmark.cpp':'887aaa07c1b0992c39c0b0c64138888f45fa1ec5c1f0391a0c01bced0ac6a8d3',
    'apps/daxa_refinement_benchmark.cpp':'fc1952562818ac2c1bd296f850048cdc90a6849a84b77732a88aa53b9bdbbe0f',
    'apps/daxa_depth_rectified_benchmark.cpp':'9ea025768c2bd2b27b359175acb2c5fc7ef0490090da69b96e9fa1d9702477f3',
}


def function(source, name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)


def guard_python(source):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Import):
            assert all(a.name in ('math',) for a in node.names),'non-CPU import'
        if isinstance(node,ast.ImportFrom):
            if node.level:
                permitted = node.level==1 and (
                    (node.module is None and all(a.name=='schema' for a in node.names)) or
                    (node.module=='native_observation' and all(a.name=='_settings' for a in node.names)) or
                    (node.module=='evaluate_temporal' and all(a.name=='temporal_settings' for a in node.names)))
                assert permitted,'non-CPU relative import'
            else:
                assert node.module in ('contextlib','hashlib','pathlib'),'non-CPU absolute import'
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
            assert name not in {'synchronize','readback','wait_idle','system','popen','__import__','import_module','eval','exec'},'device/process call'
    summary=function(source,'summarize_timing')
    text=ast.unparse(summary)
    comparisons={ast.unparse(node) for node in ast.walk(summary) if isinstance(node,ast.Compare)}
    assert 'total <= 1000000' in comparisons and 'len(records) <= 100000' in comparisons,'raw/record population bound lost'
    assert "s._integer(count, 'sample_count', 1, 100000)" in ast.unparse(function(source,'validate_timing_lane')),'per-run bound lost'
    # Run iteration admits files and shared evidence; global fingerprints only
    # after this loop. Per-record cache keys must retain binding dimensions.
    loop=next(n for n in ast.walk(summary) if isinstance(n,ast.For) and ast.unparse(n.iter)=='zip(records, roots)')
    loop_text=ast.unparse(loop)
    assert '.recheck()' not in loop_text,'whole-cache fingerprint scan per run'
    for kind in ('freeze','quality','protocol'):
        assert kind+'_key not in evidence_cache' in loop_text,'shared evidence repeatedly validated'
    assert 'file_sets.setdefault(' in loop_text
    assert 'ordered = sorted(values)' in ast.unparse(function(source,'_stats')),'chronological samples sorted in place'
    assert '.sort(' not in ast.unparse(function(source,'_stats'))
    assert 'ceil(' in ast.unparse(function(source,'_stats'))


def guard_sampler(source):
    sample=source[source.index('Samples<Code> sample('):source.index('// Post-loop report')]
    assert 'std::array<double, 100>' in source
    assert sample.count('HostClock::now()')==2 and sample.count('grab()')==1
    assert sample.index('const auto start')<sample.index('const Code code = grab()')<sample.index('const auto finish')
    assert 'result.samples[result.measured_count++]' in sample
    calls=set(re.findall(r'\b([A-Za-z_]\w*)\s*\(',sample))
    assert calls <= {'sample','if','for','invalid_argument','now','grab','count'}, 'sampler adds an unreviewed helper call'
    assert not re.search(r'\b(?:new|malloc|calloc|realloc|resize|reserve|push_back|write|read|retrieve\w*|getSVO\w*|getCUDA\w*|synchronize|wait_idle|report)\s*\(?',sample),'sampler allocates/queries/reads/waits/writes'


def guard_sdk_setup(source):
    sample=source.index('const auto samples = sf::zed_benchmark::sample')
    allocations=source.index('for (auto& image : images) image.alloc')
    assert sample<allocations,'opaque sampler prepares unused diagnostic CPU planes'
    # The benchmark returns before diagnostic allocations and retrieval loops.
    assert source.index('return 0;',sample)<allocations


def guard_native(source, expected):
    loop=timed_loop(source)
    assert hashlib.sha256(loop.encode()).hexdigest()==expected,'audited native timed loop changed'
    assert 'add_native_timing_metadata(' not in loop
    assert source.index('add_native_timing_metadata(')>source.index(loop)+len(loop)


def test_actual_timing_paths_keep_offline_and_frozen_loop_boundaries():
    guard_python((ROOT/PYTHON).read_text())
    guard_sampler((ROOT/SAMPLER).read_text())
    for path,digest in NATIVE.items():guard_native((ROOT/path).read_text(),digest)


def test_opaque_sdk_branch_skips_unused_diagnostic_plane_allocation():
    guard_sdk_setup((ROOT/SDK).read_text())


def test_sdk_setup_guard_rejects_restoring_unused_buffers_before_sample():
    source=(ROOT/SDK).read_text()
    anchor='const auto samples = sf::zed_benchmark::sample'
    mutant=source.replace(anchor,'for (auto& image : images) image.alloc(resolution, sl::MAT_TYPE::U8_C4, sl::MEM::CPU);\n            '+anchor,1)
    with pytest.raises(AssertionError):guard_sdk_setup(mutant)


@pytest.mark.parametrize('old,new',[
    ('import math','import torch\nimport math'),
    ('import math','from . import cuda\nimport math'),
    ('import math','from native_observation import _settings\nimport math'),
    ('total <= 1000000','total <= 1000000000'),
    ('len(records) <= 100000','len(records) <= 100000000'),
    ("'sample_count', 1, 100000","'sample_count', 1, 100000000"),
    ('if freeze_key not in evidence_cache:', 'if True:'),
    ('if quality_key not in evidence_cache:', 'if True:'),
    ('if protocol_key not in evidence_cache:', 'if True:'),
    ('            validate_timing_lane(record)','            files.recheck()\n            validate_timing_lane(record)'),
    ('ordered = sorted(values)','ordered = values; values.sort()'),
])
def test_python_mutations_reject_device_imports_unbounded_work_and_lost_caches(old,new):
    source=(ROOT/PYTHON).read_text();assert old in source
    with pytest.raises(AssertionError):guard_python(source.replace(old,new,1))


@pytest.mark.parametrize('injection',['samples.push_back(1);','device.synchronize();','camera.getSVOPosition();','retrieveImage();','write(file);','helper();'])
def test_sampler_mutations_reject_added_work(injection):
    source=(ROOT/SAMPLER).read_text()
    with pytest.raises(AssertionError):guard_sampler(source.replace('const auto start =',injection+'const auto start =',1))


@pytest.mark.parametrize('path',NATIVE)
def test_native_mutations_reject_any_loop_edit(path):
    source=(ROOT/path).read_text()
    loop=timed_loop(source)
    with pytest.raises(AssertionError):guard_native(source.replace(loop,loop.replace('{','{ device.wait_idle();',1),1),NATIVE[path])


def save(name, data):
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    paths=[PYTHON,SDK,SAMPLER,'apps/timing_metadata.hpp','benchmarks/zed_comparison/schema.py',*NATIVE]
    data['source_sha256']={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}
    (ARTIFACTS/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def test_actual_python_shared_file_and_evidence_scaling(tmp_path,monkeypatch):
    base=qualified_abba(tmp_path)
    # Large shared freeze artifact, hashed once across both profiles and all runs.
    artifact=tmp_path/'shared.bin';artifact.write_bytes(b'x'*(8*CHUNK))
    desc=dict(path=artifact.name,sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),size_bytes=artifact.stat().st_size)
    freeze=json.loads((tmp_path/'freeze.json').read_text())
    for profile in freeze['profiles']:profile['artifacts'].append(desc)
    frozen=write_json(tmp_path,'freeze.json',freeze)
    for row in base:row['sources']['freeze']=frozen
    # Repeat only the same four ABBA slots/content, with unique runs/blocks.
    rows=[];results=[]
    originals={name:getattr(p,name) for name in ('_freeze','_quality','_protocol')}
    for count in (8,128):
        records=[]
        for i in range(count):
            row=deepcopy(base[i%4]);row['run_id']=str(i);row['order']=dict(block=i//4,position=i%4);records.append(row)
        with monkeypatch.context() as patch:
            io=IOProbe(patch,tmp_path);calls=Counter();stats=[0];original_stat=schema._Files._stat
            for name,real in originals.items():
                def wrapped(*args,_name=name,_real=real,**kwargs):calls[_name]+=1;return _real(*args,**kwargs)
                patch.setattr(p,name,wrapped)
            def stat(self,name):stats[0]+=1;return original_stat(self,name)
            patch.setattr(schema._Files,'_stat',stat)
            start=time.perf_counter_ns();out=p.summarize_timing(records,root=tmp_path);elapsed=(time.perf_counter_ns()-start)/1e6
            assert io.opens['shared.bin']==1
            chunks=[(size,actual) for name,size,actual in io.reads if name=='shared.bin']
            assert len(chunks)==9 and sum(n for _,n in chunks)==8*CHUNK
            assert all(0<size<=CHUNK for size,_ in chunks)
            assert calls=={'_freeze':2,'_quality':2,'_protocol':1}
            assert out['comparisons'][0]['qualified'] and out['comparisons'][0]['ratio']==1
            assert len(out['admissions'])==count
            assert sum(len(v) for population in out['populations'] for v in population['samples'].values())==count*3*6
            rows.append(dict(runs=count,elapsed_ms=elapsed,stat_calls=stats[0],shared_hash_opens=io.opens['shared.bin'],shared_hash_reads=len(chunks),evidence_validation_calls=dict(calls)))
        results.append(out)
    assert rows[1]['stat_calls']<=rows[0]['stat_calls']*16,'quadratic fingerprint work'
    save('timing-admission',dict(scope='Actual Python native raw timing admission+freeze+quality+ABBA+pooled statistics on syntheticqualifiedCPUfixtures,8vs128runswith8MiBsharedartifact. Fixture-cachewarm; allartifacts hashonce percall; no hardware/GPU/SDK execution or qualityattestation.',rows=rows))
    print('actual timing admission',rows)


def test_actual_100000_sample_stats_cost_preserves_chronology():
    r=record();values=list(range(100000,0,-1));r['sample_count']=len(values)
    r['samples']['gpu_total_ms']=available(values);r['samples']['complete_wall_ms']=available(values)
    initial=list(values);times=[]
    for _ in range(3):
        start=time.perf_counter_ns();out=p.summarize_timing([r]);times.append((time.perf_counter_ns()-start)/1e6)
        population=out['populations'][0]
        stats=population['statistics']['samples']['complete_wall_ms']
        assert stats['p50_ms']==50000 and stats['p95_ms']==95000 and stats['p99_ms']==99000 and stats['max_ms']==100000
        assert values==initial and population['samples']['complete_wall_ms']==initial
        assert out['comparisons'][0]['qualified'] is False
    save('large-sample-stats',dict(scope='Actual structural admission+pooledrawnearest-rankstats on100000samples/column,2availablecolumns,nomissingrootqualification. CPUonly; nosyntheticGPUtimeclaim.',samples_per_column=100000,total_retained_raw_values=200000,first_call_ms=times[0],warm_calls_ms=times[1:]))
    print('100000 sample structural stats ms',times)


def test_actual_sdk_sampler_cpu_no_heap(tmp_path,json_include):
    compiler=shutil.which('c++')
    if compiler is None:pytest.skip('C++ compiler unavailable')
    program=r'''
#include "sdk_benchmark.hpp"
#include <iostream>
#include <cstdlib>
#include <new>
static bool tracking=false;static unsigned allocations=0;
void* operator new(std::size_t n){if(tracking)++allocations;void*p=std::malloc(n);if(!p)throw std::bad_alloc();return p;}
void operator delete(void*p)noexcept{std::free(p);}void operator delete(void*p,std::size_t)noexcept{std::free(p);}
int main(){constexpr int repeats=1000,warmup=5,frames=100;unsigned calls=0,measured=0;double checksum=0;
 auto start=std::chrono::steady_clock::now();tracking=true;
 for(int i=0;i<repeats;++i){auto result=sf::zed_benchmark::sample<int>(warmup,frames,[&]{++calls;return 0;},0);measured+=result.measured_count;checksum+=result.samples[frames-1];}
 tracking=false;auto elapsed=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/(repeats*(warmup+frames));
 if(allocations||calls!=repeats*(warmup+frames)||measured!=repeats*frames||checksum<0)return 2;
 std::cout<<elapsed<<" "<<allocations<<" "<<calls<<" "<<measured<<" "<<sizeof(sf::zed_benchmark::Samples<int>)<<"\n";
}
'''
    source=tmp_path/'sampler.cpp';exe=tmp_path/'sampler';source.write_text(program)
    command=[compiler,'-std=c++20','-O3','-I'+str(json_include),'-I'+str((ROOT/SAMPLER).parent),str(source),'-o',str(exe)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=60);assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=60);assert result.returncode==0,result.stderr
    ns,alloc,calls,measured,size=result.stdout.split();assert int(alloc)==0 and int(calls)==105000 and int(measured)==100000
    ARTIFACTS.mkdir(parents=True,exist_ok=True);(ARTIFACTS/'sdk-sampler-probe.cpp').write_text(program)
    save('sdk-sampler',dict(scope='Actual SDK-free sample helper withtrivialfakegrab andrealsteadyclock;1000x(5warmup+100measured),clock/controlcostonly,noSDK/driver/GPUcost or completionclaim. Postloopreportserializationexcluded.',cpu_ns_per_grab_iteration=float(ns),allocations=int(alloc),grab_calls=int(calls),measured_samples=int(measured),stack_sample_object_bytes=int(size),compiler_command=command))
    print('CPU SDK-free sampler ns/iteration',ns)


def test_actual_admission_releases_file_json_caches_after_return(tmp_path,monkeypatch):
    records=qualified_abba(tmp_path);refs=[];original=schema._Files.__init__
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs);refs.append(weakref.ref(self))
    monkeypatch.setattr(schema._Files,'__init__',initialize)
    out=p.summarize_timing(records,root=tmp_path)
    assert out['comparisons'][0]['qualified']
    gc.collect()
    assert refs and all(ref() is None for ref in refs),'file/JSON admission caches escape summary call'
