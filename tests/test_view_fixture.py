import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

BENCHMARKS=Path(__file__).parents[1]/'benchmarks'
sys.path.insert(0,str(BENCHMARKS))
from export_view_fixture import export_fixture
from benchmark_view_fusion import fixture,input_identity


def test_native_fixture_bytes_header_and_hashes(tmp_path):
    target=tmp_path/'fixture';manifest=export_fixture(target,8,6)
    frame,left,camera,right=fixture(8,6)
    assert manifest['input_sha256']==input_identity(frame,left,camera,right)
    assert manifest['explicit_right_depth'] and manifest['projection']=='perspective'
    tokens=(target/'header.txt').read_text().split()
    assert list(map(int,tokens[:4]))==[8,6,8,6]
    np.testing.assert_allclose(list(map(float,tokens[4:8])),[camera.focal,camera.focal,3.5,2.5])
    np.testing.assert_array_equal(np.fromfile(target/'source_depth.f32',np.float32),
                                 np.concatenate([left.depth_m.ravel(),right.depth_m.ravel()]))
    np.testing.assert_array_equal(np.array(tokens[9:25],dtype=np.float32).reshape(4,4),np.eye(4))
    assert float(tokens[25+3])==np.float32(.12)
    for name,record in manifest['files'].items():
        payload=(target/name).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==record['sha256']
        if 'dtype' in record:
            assert len(payload)==np.prod(record['shape'])*np.dtype(record['dtype']).itemsize
    with np.load(target/'expected_strict.npz',allow_pickle=False) as expected:
        assert expected['rgb'].shape==(6,8,3)
        assert np.all(~expected['approximate'])
        assert 'source_ids' not in expected.files
    with pytest.raises(ValueError):export_fixture(target,8,6)
