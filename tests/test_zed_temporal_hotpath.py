"""Actual offline temporal CPU probes; never a native/GPU frame-rate claim."""
import ast
from contextlib import ExitStack
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time
import weakref

import numpy as np
import pytest
from tests.test_zed_comparison_schema import record
from tests.test_zed_evaluate_temporal import sequence
from tests.test_zed_epipolar import envelope
from tests.test_zed_physical_reference import reference
from benchmarks.zed_comparison import schema
from benchmarks.zed_comparison.evaluate_temporal import _Moments, evaluate_temporal
from benchmarks.zed_comparison.epipolar import evaluate_epipolar
from benchmarks.zed_comparison.physical_reference import validate_physical_reference

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT/'artifacts/zed-task5/performance'
MODULES = ('evaluate_temporal','epipolar','physical_reference','truth','schema','compare_quality')


def sources():
    return {name:(ROOT/'benchmarks/zed_comparison'/f'{name}.py').read_text() for name in MODULES}


def function(source, name):
    return next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name == name)


def frame_loop(source):
    group = function(source, '_evaluate_group')
    return next(n for n in ast.walk(group) if isinstance(n,ast.For) and ast.unparse(n.iter)=='members')


def guard(src):
    allowed = {'contextlib','copy','hashlib','json','math','pathlib','time','numpy','__future__','itertools','stat','re','typing','dataclasses'}
    forbidden = {'synchronize','readback','wait_idle','wait_on_submit','cuda','vectorize','nditer','ndenumerate','fromiter','__import__','eval','exec','import_module','system','popen'}
    for name,text in src.items():
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                assert all(a.name.split('.')[0] in allowed for a in node.names), 'non-CPU dependency'
            if isinstance(node,ast.ImportFrom):
                if node.level:
                    assert node.module in ('schema','truth','compare_quality','quality_metrics','physical_reference','evaluate_temporal','epipolar') or (node.module is None and all(a.name=='schema' for a in node.names)), 'non-CPU local dependency'
                else:
                    assert node.module.split('.')[0] in allowed, 'non-CPU dependency'
            if isinstance(node,ast.Call):
                call = node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                assert call not in forbidden, 'device/runtime or Python pixel operation'
            if name in ('evaluate_temporal','epipolar'):
                if isinstance(node,ast.Attribute):
                    assert node.attr not in ('flat','flatiter','tolist'), 'Python pixel traversal'
                if isinstance(node,(ast.For,ast.comprehension)):
                    iteration = ast.unparse(node.iter)
                    assert not (iteration.startswith('range(') and any(t in iteration for t in ('.size','.shape','width','height'))), 'Python pixel loop'
                    assert iteration not in ('ids','values','depth','sample','points','track_ids','matches'), 'Python pixel loop'
    temporal = src['evaluate_temporal']
    loop = frame_loop(temporal); text = ast.unparse(loop)
    for node in ast.walk(loop):
        if isinstance(node,ast.Call):
            assert not (isinstance(node.func,ast.Name) and node.func.id in ('sorted','list') and node.args and ast.unparse(node.args[0])=='captures'), 'whole-sequence capture table rebuilt per frame'
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            callee = node.value.func
            assert not (isinstance(callee,ast.Attribute) and callee.attr=='recheck' and isinstance(callee.value,ast.Name) and callee.value.id in ('bundle','pf')), 'whole-dataset recheck inside frame loop'
    assert text.count('for p in paths')==2, 'record file subset missing'
    assert 'previous = (truth.copy(), sample.copy()) if stationary_grid else None' in text, 'previous frame bound changed'
    assert 'len(seen_tracks) <= maximum' in text, 'unbounded track state'
    update = ast.unparse(function(temporal,'update'))
    for term in ('len(union) <= self.maximum','np.searchsorted(', 'delta = values - self.mean[at]', 'self.mean[at] += delta / n', 'self.m2[at] += delta * (values - self.mean[at])'):
        assert term in update, 'bounded vectorized Welford contract lost'
    fast = next((node for node in ast.walk(function(temporal,'update')) if isinstance(node,ast.If) and isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Attribute) and node.test.func.attr=='array_equal'), None)
    assert fast is not None, 'unchanged-ID vector fastpath missing'
    fast_text = ast.unparse(fast)
    assert 'self.n += 1' in fast_text and 'self.mean += delta / self.n' in fast_text and 'self.m2 += delta * (values - self.mean)' in fast_text
    assert not any(term in fast_text for term in ('union1d(', 'searchsorted(', '[at]')), 'same-ID fastpath rebuilds association'
    assert "s._integer(limits['max_tracks'], 'max_tracks', 1, 2097152)" in ast.unparse(function(temporal,'_protocol'))
    assert "d['size_bytes'] <= MAX_FRAME_BYTES" in ast.unparse(function(temporal,'_protocol'))
    physical = ast.unparse(function(src['physical_reference'],'validate_physical_reference'))
    assert 'for path in checked:' in physical and 'files._unchanged(path, files.fingerprints[path])' in physical
    assert 'files.recheck()' not in physical, 'physical admission scans shared dataset cache'
    assert 'stream.read(HASH_CHUNK_BYTES)' in ast.unparse(function(src['schema'],'_hash'))


def test_actual_offline_sources_bound_frame_state_and_file_work():
    guard(sources())


@pytest.mark.parametrize('module,old,new', [
    ('evaluate_temporal','import numpy as np','import cupy as cp\nimport numpy as np'),
    ('epipolar','import numpy as np','import pyzed.sl as sl\nimport numpy as np'),
    ('evaluate_temporal','import numpy as np','from . import cuda\nimport numpy as np'),
    ('evaluate_temporal',"        identity=r['input_identity'];", "        device.wait_idle()\n        identity=r['input_identity'];"),
    ('evaluate_temporal',"        identity=r['input_identity'];", "        sorted(captures)\n        identity=r['input_identity'];"),
    ('evaluate_temporal',"        identity=r['input_identity'];", "        bundle.recheck()\n        identity=r['input_identity'];"),
    ('evaluate_temporal',"        identity=r['input_identity'];", "        for pixel in range(truth.size): pass\n        identity=r['input_identity'];"),
    ('evaluate_temporal','np.array_equal(ids,self.ids)','False'),
    ('evaluate_temporal','len(union)<=self.maximum','True'),
    ('evaluate_temporal','len(seen_tracks)<=maximum','True'),
    ('evaluate_temporal','self.m2[at]+=delta*(values-self.mean[at])','self.m2[at]+=values'),
    ('physical_reference','for path in checked:', 'files.recheck()\n    for path in checked:'),
    ('schema','stream.read(HASH_CHUNK_BYTES)','stream.read()'),
])
def test_mutations_reject_pixel_loops_device_calls_global_scans_and_unbounded_state(module,old,new):
    src=sources();assert old in src[module];src[module]=src[module].replace(old,new,1)
    with pytest.raises(AssertionError):guard(src)


def save(name, value):
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    value['source_sha256']={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources().items()}
    (ARTIFACTS/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def test_actual_720p_streaming_welford_cost_and_bounded_storage():
    count=1280*720;moments=_Moments(count);ids=np.arange(count,dtype=np.int64)
    timings=[];refs=[]
    for index in range(8):
        values=np.full(count,index*.01,dtype=np.float64);refs.append(weakref.ref(values))
        start=time.perf_counter_ns();moments.update(ids,values);timings.append((time.perf_counter_ns()-start)/1e6)
    del values;gc.collect();assert all(r() is None for r in refs)
    result=moments.report()
    assert result['track_count']==count and result['observation_count']==8*count
    assert result['mean_track_bias_m']==pytest.approx(.035)
    assert result['mean_track_error_std_m']==pytest.approx(np.std(np.arange(8)*.01,ddof=1))
    arrays={k:v for k,v in vars(moments).items() if isinstance(v,np.ndarray)}
    assert set(arrays)=={'ids','n','mean','m2'}
    assert all(len(v)==count for v in arrays.values())
    total=sum(v.nbytes for v in arrays.values());assert total==count*32
    save('moments-720p',dict(scope='Actual _Moments.update at921600independentIDs,8unique-capture-equivalentupdates,float64Welford. Firstallocation/update plus7warmupdates;32retainedbytes/track; no datasetpixelhistory, GPU/native/real-time/SDK or physicalqualification.',shape=[720,1280],first_update_ms=timings[0],warm_update_ms=timings[1:],retained_state_bytes=total,track_count=count,observation_count=result['observation_count'],mean_track_error_std_m=result['mean_track_error_std_m']))
    print('720p moment updates ms',timings)


def test_actual_moments_reject_growth_past_limit():
    state=_Moments(2);state.update(np.array([0,1]),np.array([1.,2.]))
    before={k:v.copy() for k,v in vars(state).items() if isinstance(v,np.ndarray)}
    with pytest.raises(ValueError,match='max_tracks'):state.update(np.array([2]),np.array([3.]))
    for key,value in before.items():np.testing.assert_array_equal(getattr(state,key),value)


def no_arrays(value):
    assert not isinstance(value,np.ndarray),'dataset pixel arrays retained in result'
    if isinstance(value,dict):
        for item in value.values():no_arrays(item)
    elif isinstance(value,list):
        for item in value:no_arrays(item)


def test_actual_temporal_evaluator_linear_fingerprints_and_closed_mappings(record,tmp_path,monkeypatch):
    original_stat=schema._Files._stat;original_map=np.memmap;counts=[];timings=[];maps=[];stats=[0];peak=[0]
    def stat(self,path):stats[0]+=1;return original_stat(self,path)
    def mapped(*args,**kwargs):
        mapping=original_map(*args,**kwargs);maps.append(mapping)
        peak[0]=max(peak[0],sum(not item._mmap.closed for item in maps));return mapping
    monkeypatch.setattr(schema._Files,'_stat',stat);monkeypatch.setattr(np,'memmap',mapped)
    for length in (2,16):
        data=sequence(record,tmp_path,depths=(1,)*length,explicit=True)
        records,truth,policy,protocol=data;stats[0]=0
        start=time.perf_counter_ns()
        out=evaluate_temporal(records,truth,policy,root=tmp_path,truth_root=tmp_path,protocol=protocol,protocol_root=tmp_path)
        timings.append((time.perf_counter_ns()-start)/1e6);counts.append(stats[0])
        row=out['groups'][0]
        assert row['unique_capture_count']==length and row['max_track_state']==4
        assert row['fresh_observation_count']==4*length
        assert row['fresh_error_jitter']['observation_count']==4*length
        assert all(item._mmap.closed for item in maps)
        no_arrays(out)
    assert counts[1]<=counts[0]*8,'quadratic whole-file fingerprint work'
    assert peak[0]<=12,'dataset memmaps retained across frames'
    save('evaluator-scaling',dict(scope='Actual full offline evaluator2vs16tiny2x2sequenceframeswithindependentexplicittracks,portablefilevalidation,truth,protocol,freshness,andWelford. Sharedfilefingerprintcounts and mappinglifetime; warm filesystemcachefromfixturegeneration. No full-resolutionevaluator/GPU/realtime/physicalqualification.',frames=[2,16],stat_calls=counts,elapsed_ms=timings,max_simultaneously_open_mappings=peak[0],all_mappings_closed=True))
    print('temporal2vs16statcalls,ms,maxmaps',counts,timings,peak[0])


def test_actual_physical_shared_cache_admission_does_not_scan_prior_files(reference,tmp_path,monkeypatch):
    with ExitStack() as stack:
        files=schema._Files(tmp_path,stack)
        for i in range(32):
            path=tmp_path/str(i);path.write_bytes(b'earlier frame')
            files.check(dict(path=str(i),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size))
        calls=[];original=files._stat
        def stat(path):calls.append(path);return original(path)
        monkeypatch.setattr(files,'_stat',stat)
        out=validate_physical_reference(reference,files=files)
        assert out['status']=='metadata_ready' and out['quality_qualified'] is False
        assert set(calls)=={'survey.bin'}


def test_actual_720p_epipolar_vectorized_diagnostic_cost():
    g=envelope();g['geometry'].update(width=1280,height=720)
    y,x=np.indices((720,1280));matches=np.column_stack((x.ravel(),y.ravel(),x.ravel(),y.ravel())).astype('<f8')
    timings=[]
    for _ in range(3):
        start=time.perf_counter_ns();out=evaluate_epipolar(matches,g);timings.append((time.perf_counter_ns()-start)/1e6)
        assert out['count']==720*1280 and out['vertical']['absolute_p95_px']==0
        assert out['quality_qualified'] is out['metric_truth'] is False
    save('epipolar-720p',dict(scope='Actual921600correspondencefloat64bulkepipolardiagnosticandhashing; syntheticidentitymatches; no metrictruth, physicalgeometryorGPUqualification.',first_call_ms=timings[0],warm_calls_ms=timings[1:],matches=720*1280,input_bytes=matches.nbytes))
    print('epipolar720p ms',timings)


def test_actual_same_ids_update_reuses_state_without_union_or_search(monkeypatch):
    state=_Moments(16);ids=np.arange(16,dtype=np.int64)
    first=np.linspace(-1,1,16);second=first+.02
    state.update(ids,first)
    addresses={name:getattr(state,name).ctypes.data for name in ('ids','n','mean','m2')}
    def forbidden(*args,**kwargs):
        raise AssertionError('same-ID update rebuilt/searched track association')
    monkeypatch.setattr(np,'union1d',forbidden)
    monkeypatch.setattr(np,'searchsorted',forbidden)
    state.update(ids,second)
    assert all(getattr(state,name).ctypes.data==ptr for name,ptr in addresses.items())
    np.testing.assert_array_equal(state.n,np.full(16,2))
    np.testing.assert_allclose(state.mean,(first+second)/2,atol=1e-15)
    np.testing.assert_allclose(state.m2,np.full(16,.0002),atol=1e-15)
