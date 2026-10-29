"""Independent CPU-only checks for fused profiler identity and stage dependencies."""
import re
from pathlib import Path
import pytest
import test_fp2_fused_stage_hotpath as profiler
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_fused_stage.cpp';BASE=ROOT/'apps/fp2_fused_fixture.cpp';DIAG=ROOT/'benchmarks/fp2_fused_stage.py'


def guard(src,base,diag):
    profiler.guard(src,base,diag)
    cmd=profiler.block(src)
    if 'if(eye)recorder.pipeline_barrier' not in cmd or 'if(pass)recorder.pipeline_barrier' not in cmd:raise ValueError('conditional reuse/path barriers')
    pattern=r'recorder\.pipeline_barrier\((\{[^;]+?\})\)'
    actual=re.findall(pattern,profiler.block(src)); expected=re.findall(pattern,profiler.block(base))
    if len(actual)!=len(expected)+2 or actual[1:-2]+actual[-1:]!=expected:raise ValueError('baseline barrier mismatch')
    extra='{.src_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE}'
    if actual[0]!=extra or actual[-2]!=extra:raise ValueError('profiler barrier identity')
    for token in ('/fp2-fused.spv','SF_FP2_FUSED_SPIRV_SHA256','admit_parallel_device(d,uint32_t(std::max<uint64_t>',
                  'validate_regions(regions)','validate_metric_lut(lut,numerator,cx,reinterpret_cast<uintptr_t>(d.get()))'):
        if token not in src:raise ValueError('device/shader identity')
    for token in ("fp2_banded_fused.slang", "'fp2-fused.spv'", "proof.get('fused_proof_sha256')!=", 'check_fused_proof(a.fused_proof',
                  "pp['host']['embedded'][field]!=binary['embedded'][field]", 'check=True,timeout=600', 'timeout=120'):
        if token not in diag:raise ValueError('proof/timeout '+token)


def current():return APP.read_text(),BASE.read_text(),DIAG.read_text()

def test_current():guard(*current())


@pytest.mark.parametrize('where,old,new',[
 ('src','/fp2-fused.spv','/fp2-parallel.spv'),
 ('src','SF_FP2_FUSED_SPIRV_SHA256','SF_FP2_PARALLEL_SPIRV_SHA256'),
 ('src','validate_regions(regions)',''),
 ('src','if(eye)recorder.pipeline_barrier','if(false)recorder.pipeline_barrier'),
 ('diag',"pp['host']['embedded'][field]!=binary['embedded'][field]",'False'),
 ('diag','check=True,timeout=600','check=True'),
])
def test_mutations(where,old,new):
    src,base,diag=current();target=src if where=='src' else diag;assert old in target
    with pytest.raises((ValueError,IndexError)):guard(target.replace(old,new,1) if where=='src' else src,base,target.replace(old,new,1) if where=='diag' else diag)


def test_actual_source_fake_device(tmp_path,monkeypatch):
    monkeypatch.setattr(profiler,'APP',APP)
    profiler.test_fake_actual_source(tmp_path)
