"""CPU-only identity and recording-layout checks; these do not exercise Vulkan."""
import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.fp2_parallel_stage import verify_binary, check_proof

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'build/daxa/fp2_parallel_stage'
SOURCE = ROOT / 'apps/fp2_parallel_stage.cpp'


@pytest.mark.skipif(not BINARY.exists(), reason='optional profiler not built')
def test_profiler_identity():
    assert verify_binary(BINARY)['embedded']['host_sha256']


def test_query_recording_structure():
    s = SOURCE.read_text()
    assert '.query_count=19' in s
    assert 'require(query==(instrumented?19u:2u)' in s
    assert s.count('mark();') == 6  # loops generate 2 + 2*(1+1+4+1) + 1 marks
    assert 'uint32_t phase=frame<10?frame:frame-10' in s
    assert '(phase%4==1 || phase%4==2)' in s
    assert sum(i % 4 in (1, 2) for i in range(10)) == 5
    assert sum(i % 4 in (1, 2) for i in range(100)) == 50
    assert 'total_ticks*period<=wall_ns' in s
    assert 'values[2*i+1]!=0' in s


def test_proof_rejects_changed_hash(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text(json.dumps({'scope': 'stage parity'}))
    with pytest.raises(ValueError, match='missing or changed'):
        check_proof(path, '0'*64, 'm', {}, {}, 'h', 'b', {})


def test_compute_dispatch_contract_matches_resident():
    import re
    resident = (ROOT / 'apps/fp2_resident_fixture.cpp').read_text()
    profiler = SOURCE.read_text()
    pattern = r'recorder\.(?:set_pipeline|push_constant|dispatch)\([^;]+;'
    def commands(source):
        recording = source[source.index('auto record=[&]'):source.index('using Clock=')]
        return re.findall(pattern, recording)
    assert commands(profiler) == commands(resident)
    assert 'buffer_host_address' not in profiler[profiler.index('auto record=[&]'):profiler.index('uint32_t failure=')]


def test_parallel_device_and_upstream_proof_guards():
    s = SOURCE.read_text()
    assert 'admit_parallel_device(d,' in s
    assert 'maxComputeSharedMemorySize>=1536' in s
    script = (ROOT/'benchmarks/fp2_parallel_stage.py').read_text()
    assert 'parallel shader binding mismatch' in script
    assert 'parallel proof changed' in script
    assert "pp.get('tiny')!=tiny" in script


def test_upstream_proof_mutation_rejected(tmp_path, monkeypatch):
    import benchmarks.fp2_parallel_stage as p
    proof=tmp_path/'proof.json'
    data={'scope':'parallel parity','manifest_sha256':'m','baseline_sha256':'b','tiny':{'id':1},'host':{'embedded':{'no_path_shader_sha256':'s','no_path_spirv_sha256':'p'}}}
    proof.write_text(json.dumps(data))
    monkeypatch.setattr(p,'sha',lambda _: 'ce5610937f578f7e7a15c302374bb5efa69687f041739f282f0b63cfe5f07c18')
    binary={'embedded':{'no_path_shader_sha256':'s','no_path_spirv_sha256':'p'}}
    p.check_parallel_proof(proof,'m','b',{'id':1},binary)
    for changed in ({'embedded':{'no_path_shader_sha256':'bad','no_path_spirv_sha256':'p'}},{'embedded':{'no_path_shader_sha256':'s','no_path_spirv_sha256':'bad'}}):
        with pytest.raises(ValueError,match='shader binding'):p.check_parallel_proof(proof,'m','b',{'id':1},changed)
    with pytest.raises(ValueError,match='proof binding'):p.check_parallel_proof(proof,'m','b',{'id':2},binary)
