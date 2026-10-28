"""Offline CPU boundary guards and actual-source probes; no GPU/FPS qualification."""
import ast
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
from tests.test_zed_truth import make_truth
from tests.test_zed_native_input import native_record
from tests.test_zed_native_observation import executed
from benchmarks.zed_comparison import schema, truth, quality_metrics, native_observation

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT/'artifacts/zed-task4/performance'
MODULES = ('truth', 'quality_metrics', 'native_observation', 'compare_quality', 'schema', 'native_input')


def sources():
    return {name: (ROOT/'benchmarks/zed_comparison'/f'{name}.py').read_text() for name in MODULES}


def function(source, name):
    return next(node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)


def guard(source):
    allowed = {'__future__', 'contextlib', 'copy', 'hashlib', 'pathlib', 'math', 'json', 'itertools', 'shutil', 'stat', 're', 'numpy', 'argparse', 'os', 'sys', 'typing', 'dataclasses'}
    forbidden_calls = {'readback', 'synchronize', 'wait_idle', 'wait_on_submit', 'cuda', 'vectorize', 'nditer', 'ndenumerate', 'eval', 'exec', '__import__', 'import_module', 'fromiter', 'system', 'popen', 'Popen'}
    for name, text in source.items():
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split('.')[0] in allowed for alias in node.names), 'non-CPU dependency'
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    assert node.module in ('truth', 'quality_metrics', 'native_input') or (node.module is None and all(alias.name == 'schema' for alias in node.names)), 'non-CPU local dependency'
                else:
                    assert node.module.split('.')[0] in allowed, 'non-CPU dependency'
            if isinstance(node, ast.Call):
                callee = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
                assert callee not in forbidden_calls, 'device/runtime or Python pixel operation'
            if name in ('truth','quality_metrics','native_observation'):
                if isinstance(node, ast.Attribute):
                    assert node.attr not in ('flat', 'flatiter', 'tolist'), 'Python pixel traversal'
                if isinstance(node, (ast.For, ast.comprehension)):
                    iteration = ast.unparse(node.iter)
                    assert iteration not in {'depth', 'truth', 'sample', 'accepted', 'confidence', 'validity', 'valid', 'error', 'relative', 'mapping', 'converted'}, 'Python pixel traversal'
                    assert not (iteration.startswith('range(') and any(token in iteration for token in ('.shape', '.size', 'width', 'height', 'count'))), 'Python pixel loop'
    frame = function(source['truth'], 'frame')
    frame_text = ast.unparse(frame)
    assert "paths = self._frame_paths[input_identity]" in frame_text, 'per-frame file subset missing'
    assert frame_text.count('for p in paths') == 2, 'whole-dataset cache scan during frame materialization'
    assert 'stack.callback(mapping._mmap.close)' in frame_text, 'mapping lifetime unbounded'
    for node in ast.walk(frame):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assert not any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'self' for t in targets), 'materialized array cached on bundle'
    assert 'MAX_FRAME_BYTES = 512 * 1024 * 1024' in source['truth']
    assert "sum(d['size_bytes'] for d in planes) <= MAX_FRAME_BYTES" in source['truth']
    evaluation = ast.unparse(function(source['quality_metrics'], 'evaluate_frame'))
    assert 'len(thresholds) > 101' in evaluation and 'np.linspace(0, len(thresholds) - 1, 101, dtype=int)' in evaluation, 'confidence threshold loop unbounded'
    observation = ast.unparse(function(source['compare_quality'], '_observation'))
    assert observation.count('for p in paths') == 2, 'whole-dataset observation rechecks'
    recheck = ast.unparse(function(source['schema'], 'recheck'))
    assert '_hash(' not in recheck and 'self._unchanged(' in recheck, 'snapshot recheck rehashes files'
    hashing = ast.unparse(function(source['schema'], '_hash'))
    assert 'stream.read(HASH_CHUNK_BYTES)' in hashing, 'unbounded file hash read'
    snapshot = ast.unparse(function(source['native_input'], '_snapshot'))
    assert 'stream.read(schema.HASH_CHUNK_BYTES)' in snapshot, 'unbounded adapter hash read'
    adapter = ast.unparse(function(source['native_observation'], 'export_native_observation'))
    assert "mappings[field][:count].astype(np.uint8).tobytes()" in adapter, 'mask conversion must remain bulk NumPy'
    assert 'stream.read(count * np.dtype(dtype).itemsize)' in adapter, 'float bytes must remain exact bounded bulk read'


def test_actual_sources_preserve_offline_bounded_cpu_boundaries():
    guard(sources())


@pytest.mark.parametrize('module,old,new', [
    ('quality_metrics', 'import math', 'import torch\nimport math'),
    ('truth', 'import numpy as np', 'import cupy as cp\nimport numpy as np'),
    ('truth', 'import numpy as np', 'from . import cuda\nimport numpy as np'),
    ('quality_metrics', '    depth = np.asarray(depth, dtype=np.float64)', '    for pixel in depth.flat: pass\n    depth = np.asarray(depth, dtype=np.float64)'),
    ('quality_metrics', '    depth = np.asarray(depth, dtype=np.float64)', '    for row in range(depth.shape[0]): pass\n    depth = np.asarray(depth, dtype=np.float64)'),
    ('quality_metrics', '    depth = np.asarray(depth, dtype=np.float64)', '    device.synchronize()\n    depth = np.asarray(depth, dtype=np.float64)'),
    ('quality_metrics', 'len(thresholds)>101', 'len(thresholds)>1000000'),
    ('truth', 'for p in paths}', 'for p in self._hashes}'),
    ('truth', '            files.recheck()\n            try:', '            self.cached_frame = out\n            files.recheck()\n            try:'),
    ('truth', 'stack.callback(mapping._mmap.close)', 'pass # mapping close removed'),
    ('schema', 'stream.read(HASH_CHUNK_BYTES)', 'stream.read()'),
    ('native_input', 'stream.read(schema.HASH_CHUNK_BYTES)', 'stream.read()'),
    ('native_observation', "mappings[field][:count].astype(np.uint8).tobytes()", "bytes(int(pixel) for pixel in mappings[field][:count])"),
])
def test_mutation_guards_reject_runtime_loops_unbounded_io_and_retention(module, old, new):
    text = sources(); assert old in text[module]
    text[module] = text[module].replace(old, new, 1)
    with pytest.raises(AssertionError):
        guard(text)


def multi_truth(record, root, count):
    document = None; records = []
    for i in range(count):
        item = deepcopy(record); item['source_id'] -= i; item['ordinal'] = i
        item['input_identity'] = schema.input_identity(item)
        piece = make_truth(item, root)
        frame = piece['frames'][0]
        for desc in (frame['depth'], frame['masks']['sample']):
            old = root/desc['path']; name = str(i)+'-'+desc['path']
            (root/name).write_bytes(old.read_bytes()); desc['path'] = name
        if document is None:
            document = piece
        else:
            document['frames'].append(frame)
        records.append(item)
    return document, records


@pytest.mark.parametrize('count', [2, 16])
def test_truth_materialization_rechecks_only_frame_files_no_rehash_and_no_array_cache(record, tmp_path, monkeypatch, count):
    document, records = multi_truth(record, tmp_path, count)
    bundle = truth.load_truth(document, records, root=tmp_path)
    calls = []; misses = []; mappings = []
    original_stat = schema._Files._stat; original_hash = schema._Files._hash; original_map = np.memmap
    def stat(self, path):
        calls.append(path)
        return original_stat(self,path)
    def hashed(self, path):
        if path not in self.hashes: misses.append(path)
        return original_hash(self,path)
    def mapped(*args, **kwargs):
        value = original_map(*args, **kwargs); mappings.append(value)
        return value
    monkeypatch.setattr(schema._Files, '_stat', stat)
    monkeypatch.setattr(schema._Files, '_hash', hashed)
    monkeypatch.setattr(truth.np, 'memmap', mapped)
    target = records[-1]['input_identity']
    with bundle.frame(target) as frame:
        refs = [weakref.ref(frame['depth_m']), weakref.ref(frame['sample'])]
        assert not frame['depth_m'].flags.writeable
        assert all(not m._mmap.closed for m in mappings)
    assert all(m._mmap.closed for m in mappings)
    assert not misses, 'materialization rehashed admitted files'
    expected = bundle._frame_paths[target]
    assert set(calls) == expected and len(calls) <= 4*len(expected)+4
    assert all(not isinstance(v,np.ndarray) for v in vars(bundle).values())
    del frame
    gc.collect()
    assert all(ref() is None for ref in refs), 'bundle retained materialized arrays'


def hashes():
    return {name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources().items()}


def report(name, value):
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    value['source_sha256'] = hashes()
    (ARTIFACTS/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def test_actual_720p_dense_metric_control_and_confidence_cost():
    h,w = 720,1280
    y,x = np.ogrid[:h,:w]
    depth = (1.0 + x/w*5.0 + y/h).astype(np.float64)
    sample = np.ones((h,w),dtype=bool)
    confidence = np.broadcast_to(np.linspace(0,1,w),(h,w)).copy()
    predicted = depth + .025*np.sin(x/15) + .01*np.cos(y/20)
    frame = dict(depth_m=depth,sample=sample,geometry=dict(fx=700.,baseline_m=.12,cx_left=640.,cx_right=640.),geometric_support=sample,known_negative=None,regions={},boundary_edges=None)
    policy = dict(requested_range_m=[.3,8.],required_regions=['all'],freeze_parameters=dict(correctness=dict(mode='depth_tolerance',absolute_m=.05,relative=.01)))
    timings = []
    for index in range(4):
        start = time.perf_counter_ns()
        out = quality_metrics.evaluate_frame(predicted,sample,confidence,'high_is_good',frame,policy,search_bounds=dict(min_disparity=0,max_disparity_exclusive=128))
        elapsed = (time.perf_counter_ns()-start)/1e6
        assert len(out['confidence_curve']) == 101
        assert out['lanes']['requested']['truth_count'] == h*w
        assert out['confidence_curve'][-1]['accepted_count'] == h*w
        assert out['confidence_curve'][0]['accepted_count'] == h
        assert out['lanes']['requested']['correct_count'] == h*w
        timings.append(elapsed)
    report('metrics-720p',dict(scope='Actual NumPy evaluate_frame at 1280x720, dense sample, geometric/search lanes, correctness, distance bins and full101tied-threshold confidence curve. First call separate from3warmcalls; offline CPU only; no GPU/backend/60FPS/sensor/scientific parity claim.',shape=[h,w],confidence_groups=1280,reported_thresholds=101,first_call_ms=timings[0],warm_calls_ms=timings[1:],dense_truth_count=h*w))
    print('720p metrics milliseconds:',timings)


def test_actual_tiny_truth_and_adapter_cold_warm_cost(executed, tmp_path):
    record = executed[0]
    document = make_truth(record,tmp_path)
    start = time.perf_counter_ns(); bundle = truth.load_truth(document,[record],root=tmp_path)
    admission = (time.perf_counter_ns()-start)/1e6
    warm = []
    for _ in range(4):
        start = time.perf_counter_ns()
        with bundle.frame(record['input_identity']) as frame:
            assert frame['sample'].shape == (4,4)
        warm.append((time.perf_counter_ns()-start)/1e6)
    _,source,_,native_report,dumps = executed
    adaptation = []
    for index in range(3):
        start = time.perf_counter_ns()
        output = native_observation.export_native_observation(native_report,source,dumps,tmp_path/f'export-{index}')
        adaptation.append((time.perf_counter_ns()-start)/1e6)
        admitted = schema.load_records([output])[0]
        assert admitted['input_identity'] == executed[0]['input_identity']
    report('cold-boundaries',dict(scope='Actual tiny4x4truth and4x4native dump export fixture; first in-process calls plus subsequentcalls, warm filesystem cache from fixturegeneration. Explicit offline CPU copy/hash/validation/output boundary; no full-resolution throughput/cold-disk/GPU/FPS qualification.',truth_admission_ms=admission,truth_warm_materialization_ms=warm,adapter_first_ms=adaptation[0],adapter_subsequent_ms=adaptation[1:]))
    print('truth admission/warm, adapter ms:',admission,warm,adaptation)


def test_coordinator_rechecks_scale_linearly_and_result_has_no_pixel_arrays(record, tmp_path, monkeypatch):
    from benchmarks.zed_comparison.compare_quality import compare_quality
    original_stat = schema._Files._stat
    counts = []
    count = [0]
    def stat(self, path):
        count[0] += 1
        return original_stat(self, path)
    monkeypatch.setattr(schema._Files, '_stat', stat)
    def no_arrays(value):
        assert not isinstance(value, np.ndarray), 'coordinator retained pixel array in result'
        if isinstance(value, dict):
            for child in value.values(): no_arrays(child)
        elif isinstance(value, list):
            for child in value: no_arrays(child)
    for frames in (2, 16):
        document, records = multi_truth(record, tmp_path, frames)
        count[0] = 0
        result = compare_quality(records, document, None, root=tmp_path, truth_root=tmp_path)
        assert len(result['frames']) == frames
        no_arrays(result)
        counts.append(count[0])
    # 8x more input groups may require 8x work, never quadratic 64x checks.
    assert counts[1] <= counts[0] * 8
    report('coordinator-scaling', dict(scope='Actual coordinator2versus16tinyinputgroupswithsharedportabledependenciesanddistincttruthplanes; counts _Files._stat calls across admission, materialization and final rechecks; result recursively contains no ndarray. CPU only.', frames=[2,16], stat_calls=counts))
