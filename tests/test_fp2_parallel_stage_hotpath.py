"""CPU-only guards for the parallel stage profiler; no GPU is run."""
from pathlib import Path
import pytest
import test_fp2_stage_hotpath as stage
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_parallel_stage.cpp'
BASE=ROOT/'apps/fp2_parallel_fixture.cpp'
DIAG=ROOT/'benchmarks/fp2_parallel_stage.py'


def guard(source,base,diag):
    # The shared stage checker requires the equivalent scope spelling.
    if "proof.get('scope')!='parallel stage parity'" not in diag:raise ValueError("parallel stage proof scope")
    stage.guard(source,base,diag.replace("proof.get('scope')!='parallel stage parity'", "proof.get('scope')!='stage parity'"))
    for token in ('/fp2-parallel.spv', 'SF_FP2_PARALLEL_SPIRV_SHA256',
                  'limits.maxComputeWorkGroupInvocations>=128', 'limits.maxComputeSharedMemorySize>=1536',
                  'admit_parallel_device(d,uint32_t(std::max<uint64_t>'):
        if token not in source:raise ValueError('parallel shader/device identity')
    for token in ("'no_path_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_banded_parallel.slang'",
                  "pp['host']['embedded'][field]!=binary['embedded'][field]",
                  "for field in ('no_path_shader_sha256','no_path_spirv_sha256')",
                  "proof.get('parallel_proof_sha256')!=", "timeout=120", "check=True,timeout=600"):
        if token not in diag:raise ValueError('proof identity/timeout')


def current():return APP.read_text(),BASE.read_text(),DIAG.read_text()

def test_source():guard(*current())


@pytest.mark.parametrize('which,old,new',[
 ('src','/fp2-parallel.spv','/fp2-resident-no_path.spv'),
 ('src','SF_FP2_PARALLEL_SPIRV_SHA256','SF_FP2_RESIDENT_NO_PATH_SPIRV_SHA256'),
 ('src','limits.maxComputeWorkGroupInvocations>=128','true'),
 ('src','if(instrumented)recorder.write_timestamp','recorder.write_timestamp'),
 ('src','.queue_submit_index=submit_id','.queue_submit_index=0'),
 ('src','values[2*i+1]!=0','true'),('src','wall_ns<wrap_ns','true'),
 ('src','recorder.set_pipeline(metric)','d.buffer_host_address(raw);recorder.set_pipeline(metric)'),
 ('diag',"pp['host']['embedded'][field]!=binary['embedded'][field]",'False'),
 ('diag',"proof.get('scope')!='parallel stage parity'",'False'),
 ('diag','check=True,timeout=600','check=True'),
])
def test_mutations(which,old,new):
    source,base,diag=current();target=source if which=='src' else diag;assert old in target
    with pytest.raises((ValueError,IndexError)):
        guard(target.replace(old,new,1) if which=='src' else source,base,target.replace(old,new,1) if which=='diag' else diag)


def test_fake_actual_source(tmp_path,monkeypatch):
    monkeypatch.setattr(stage,'APP',APP)
    stage.test_fake_actual_source(tmp_path)
