"""CPU-only workgroup and timing guards for the experimental ping-pong fixture."""
from pathlib import Path
import pytest
import test_fp2_parallel_hotpath as parallel
import test_fp2_resident_hotpath as resident
import test_fp2_matched_timing_hotpath as matched
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_pingpong_fixture.cpp'
TINY=ROOT/'apps/fp2_pingpong_tiny.cpp'
SHADER=ROOT/'python/stereoforge/shaders/daxa/fp2_banded_pingpong.slang'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'
MATCHED=ROOT/'benchmarks/fp2_pingpong_matched.py'
DIAG=ROOT/'benchmarks/fp2_pingpong_diagnostic.py'


def guard_shader(source):
    s=''.join(source.split())
    required=['[numthreads(128,1,1)]','groupshareduintstates[256];','groupshareduintreduction[128];',
              'if(line>=(horizontal?p.height:p.width))return;', 'boolswap=false;states[lane]=INF;GroupMemoryBarrierWithGroupSync();',
              'uintprevious=swap?128:0;uintcurrent=swap?0:128;', 'reduction[lane]=states[previous+lane];',
              'for(uintstride=64;stride>0;stride>>=1)', 'if(lane<stride)reduction[lane]=min(reduction[lane],reduction[lane+stride]);',
              'if(lane<p.disparities&&lane>=lo[i]&&lane<hi[i])', 'offsets[i]+lane-lo[i]',
              'if(c!=INF)', 'if(minimum!=INF)', 'states[previous+lane-1]!=INF', 'states[previous+lane+1]!=INF',
              'states[current+lane]=value;GroupMemoryBarrierWithGroupSync();swap=!swap;',
              'if(p.paths!=0)paths[uint64_t(pc.pass)*allocated+j]=value;']
    for item in required:
        if item not in s:raise ValueError('shader invariant '+item)
    if s.count('return;')!=1 or s.count('GroupMemoryBarrierWithGroupSync();')!=4:raise ValueError('uniform barriers')


def guard_proof(diag,matched_source):
    for item in ("proof.get('scope')!='pingpong parity'", "proof.get('host')!=host", "proof.get('tiny')!=tiny",
                 "proof.get('manifest_sha256')!=manifest_sha", "proof['gpu_outputs'][key]!=baseline['gpu_outputs'][key]"):
        if item not in diag:raise ValueError('proof identity')
    matched.guard(matched_source)
    if "'128 lanes and tree reduction per path pixel including inactive candidates'" not in matched_source:raise ValueError('full 128 lane work disclosure')


def guard_status(src):
    a=src.index('for(uint32_t frame=0;frame<iterations;++frame)');b=src.index('uint32_t failure=0;',a)
    tail=src[b:]
    if not tail.index('d.buffer_host_address(status).value()')<tail.index('require(failure==0,"resident GPU status failure")')<tail.index('if(!timed)')<tail.index('owner.release();context.close()'):raise ValueError('status or lifetime')


def test_current_sources():
    guard_shader(SHADER.read_text());parallel.guard_host(APP.read_text(),TINY.read_text(),OWNER.read_text());guard_status(APP.read_text());guard_proof(DIAG.read_text(),MATCHED.read_text())


@pytest.mark.parametrize('old,new',[
 ('[numthreads(128,1,1)]','[numthreads(64,1,1)]'),('states[lane]=INF;','states[lane]=0;'),
 ('uint previous=swap?128:0;','uint previous=swap?0:128;'),('uint current=swap?0:128;','uint current=swap?128:0;'),
 ('states[current+lane]=value;','states[previous+lane]=value;'),('swap=!swap;','swap=false;'),
 ('if(line >= (horizontal?p.height:p.width)) return;','if(lane>10)return;'),
 ('offsets[i]+lane-lo[i]','offsets[i]+lane'),('if(c!=INF)','if(true)'),
 ('if(minimum!=INF)','if(true)'),('GroupMemoryBarrierWithGroupSync();',''),
])
def test_shader_mutations(old,new):
    s=SHADER.read_text();assert old in s
    with pytest.raises(ValueError):guard_shader(s.replace(old,new,1))


@pytest.mark.parametrize('which,old,new',[
 ('host','addr(sums),0,addr(scratch)','addr(sums),addr(paths),addr(scratch)'),
 ('host','limits.maxComputeWorkGroupInvocations>=128','true'),
 ('host','recorder.dispatch({pass<2?720u:1280u,1,1})','recorder.dispatch({1280u,1,1})'),
 ('host','.queue_submit_index=submit_id','.queue_submit_index=0'),
 ('tiny','oracle_out[eye].paths.data()','oracle_out[0].paths.data()'),
 ('tiny','oracle_out[eye].sums.data()','oracle_out[0].sums.data()'),
])
def test_host_mutations(which,old,new):
    src=APP.read_text();tiny=TINY.read_text();target=src if which=='host' else tiny;assert old in target;changed=target.replace(old,new,1)
    with pytest.raises((ValueError,IndexError)):parallel.guard_host(changed if which=='host' else src,changed if which=='tiny' else tiny,OWNER.read_text())


def test_proof_mutation():
    d=DIAG.read_text();old="proof.get('host')!=host";assert old in d
    with pytest.raises(ValueError):guard_proof(d.replace(old,'False',1),MATCHED.read_text())


def test_fake_actual_source(tmp_path,monkeypatch):
    monkeypatch.setattr(resident,'APP',APP)
    resident.test_fake_actual_recording(tmp_path)


def test_status_abort_mutation():
    src=APP.read_text();old='require(failure==0,"resident GPU status failure")';assert old in src
    with pytest.raises((ValueError,IndexError)):guard_status(src.replace(old,'require(true,"ignored")',1))
