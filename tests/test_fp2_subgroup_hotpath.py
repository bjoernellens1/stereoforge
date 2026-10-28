"""CPU-only guards for the subgroup recurrence and diagnostic command path."""
from pathlib import Path
import pytest
import test_fp2_parallel_hotpath as parallel
import test_fp2_resident_hotpath as resident
import test_fp2_matched_timing_hotpath as matched
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_subgroup_fixture.cpp'
TINY=ROOT/'apps/fp2_subgroup_tiny.cpp'
SHADER=ROOT/'python/stereoforge/shaders/daxa/fp2_banded_subgroup.slang'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'
MATCHED=ROOT/'benchmarks/fp2_subgroup_matched.py'
DIAG=ROOT/'benchmarks/fp2_subgroup_diagnostic.py'


def guard_shader(src):
    s=''.join(src.split())
    for token in ('[numthreads(128,1,1)]','if(line>=(horizontal?p.height:p.width))return;',
                  'previous[lane]=INF;GroupMemoryBarrierWithGroupSync();',
                  'if(lane==0)group_minimum=INF;GroupMemoryBarrierWithGroupSync();',
                  'uintsubgroup_minimum=WaveActiveMin(previous[lane]);',
                  'if(WaveIsFirstLane())InterlockedMin(group_minimum,subgroup_minimum);GroupMemoryBarrierWithGroupSync();',
                  'uintminimum=group_minimum;', 'if(lane<p.disparities&&lane>=lo[i]&&lane<hi[i])',
                  'offsets[i]+lane-lo[i]', 'if(c!=INF)', 'if(minimum!=INF)',
                  'previous[lane-1]!=INF','previous[lane+1]!=INF',
                  'current[lane]=value;GroupMemoryBarrierWithGroupSync();previous[lane]=current[lane];GroupMemoryBarrierWithGroupSync();',
                  'if(p.paths!=0)paths[uint64_t(pc.pass)*allocated+j]=value;'):
        if token not in s:raise ValueError('shader invariant '+token)
    if s.count('return;')!=1 or s.count('GroupMemoryBarrierWithGroupSync();')!=5:raise ValueError('uniform group barrier')
    for forbidden in ('WaveGetLaneCount()', 'WaveGetLaneIndex()', 'lane/32','lane/64'):
        if forbidden in s:raise ValueError('subgroup layout assumption')


def guard_host(host,tiny,owner):
    for source in (host,tiny):
        for token in ('limits.maxComputeWorkGroupInvocations>=128','limits.maxComputeWorkGroupSize[0]>=128','limits.maxComputeSharedMemorySize>=1544',
                      'limits.maxComputeWorkGroupCount[0]>=groups', 'subgroup.subgroupSize>0 && subgroup.subgroupSize<=128',
                      'VK_SHADER_STAGE_COMPUTE_BIT','VK_SUBGROUP_FEATURE_BASIC_BIT','VK_SUBGROUP_FEATURE_ARITHMETIC_BIT',
                      'subgroup_size="<<subgroup.subgroupSize','subgroup_stages="<<subgroup.supportedStages',
                      'subgroup_operations="<<subgroup.supportedOperations',
                      'vkGetPhysicalDeviceProperties2(daxa_dvc_get_vk_physical_device(d.get()),&props)',
                      'admit_subgroup(subgroup)', 'admit_parallel_device(d,uint32_t(std::max<uint64_t>'):
            if token not in source:raise ValueError('subgroup capability')
    # Reuse the parallel host checker after normalizing equivalent device-query text.
    def normalized(s):
        return s.replace('limits.maxComputeSharedMemorySize>=1544','limits.maxComputeSharedMemorySize>=1536').replace('vkGetPhysicalDeviceProperties2(daxa_dvc_get_vk_physical_device(d.get()),&props)','vkGetPhysicalDeviceProperties(daxa_dvc_get_vk_physical_device(d.get()),&props)')
    parallel.guard_host(normalized(host),normalized(tiny),owner)
    tail=host[host.index('uint32_t failure=0;',host.index('for(uint32_t frame=0;frame<iterations;++frame)')):]
    if not tail.index('d.buffer_host_address(status).value()')<tail.index('require(failure==0,"resident GPU status failure")')<tail.index('if(!timed)')<tail.index('owner.release();context.close()'):raise ValueError('status ordering')
    if '/fp2-subgroup.spv' not in host or 'SF_FP2_SUBGROUP_SPIRV_SHA256' not in host:raise ValueError('subgroup binary identity')


def guard_proof(diag,matched_src):
    for token in ("proof.get('scope')!='subgroup parity'", "proof.get('host')!=host", "proof.get('tiny')!=tiny", "proof.get('manifest_sha256')!=manifest_sha", "proof['gpu_outputs'][key]!=baseline['gpu_outputs'][key]"):
        if token not in diag:raise ValueError('proof binding')
    if "fp2_banded_subgroup.slang" not in diag or "fp2-subgroup.spv" not in diag:raise ValueError('subgroup proof artifact')
    matched.guard(matched_src, work_labels=("'work_by_variant'", "'old':'128 lanes per path pixel; shared tree reduction, including inactive candidates'", "'new':'128 lanes per path pixel; subgroup minimum and leader atomic minimum, including inactive candidates'"))


def test_sources():guard_shader(SHADER.read_text());guard_host(APP.read_text(),TINY.read_text(),OWNER.read_text());guard_proof(DIAG.read_text(),MATCHED.read_text())


@pytest.mark.parametrize('old,new',[
 ('[numthreads(128,1,1)]','[numthreads(64,1,1)]'),
 ('if(lane==0) group_minimum=INF;','if(lane==0) group_minimum=0;'),
 ('WaveActiveMin(previous[lane])','WaveActiveMin(current[lane])'),
 ('if(WaveIsFirstLane()) InterlockedMin(group_minimum,subgroup_minimum);','if(lane==0) InterlockedMin(group_minimum,subgroup_minimum);'),
 ('if(line >= (horizontal?p.height:p.width)) return;','if(lane>10)return;'),
 ('offsets[i]+lane-lo[i]','offsets[i]+lane'),('GroupMemoryBarrierWithGroupSync();',''),
 ('current[lane]=value;','if(lane>3)return;current[lane]=value;'),
])
def test_shader_mutations(old,new):
    s=SHADER.read_text();assert old in s
    with pytest.raises(ValueError):guard_shader(s.replace(old,new,1))


@pytest.mark.parametrize('which,old,new',[
 ('host','VK_SUBGROUP_FEATURE_ARITHMETIC_BIT','VK_SUBGROUP_FEATURE_BALLOT_BIT'),
 ('host','VK_SUBGROUP_FEATURE_BASIC_BIT','VK_SUBGROUP_FEATURE_BALLOT_BIT'),
 ('host','limits.maxComputeWorkGroupInvocations>=128','true'),
 ('host','recorder.dispatch({pass<2?720u:1280u,1,1})','recorder.dispatch({1280u,1,1})'),
 ('host','.queue_submit_index=submit_id','.queue_submit_index=0'),
 ('tiny','oracle_out[eye].paths.data()','oracle_out[0].paths.data()'),
])
def test_host_mutations(which,old,new):
    host=APP.read_text();tiny=TINY.read_text();target=host if which=='host' else tiny;assert old in target;changed=target.replace(old,new,1)
    with pytest.raises((ValueError,IndexError)):guard_host(changed if which=='host' else host,changed if which=='tiny' else tiny,OWNER.read_text())


def test_fake_actual_source(tmp_path,monkeypatch):
    monkeypatch.setattr(resident,'APP',APP)
    resident.test_fake_actual_recording(tmp_path)


@pytest.mark.parametrize('old,new', [
    ("'old':'128 lanes per path pixel; shared tree reduction, including inactive candidates'", "'old':'incorrect'"),
    ("'new':'128 lanes per path pixel; subgroup minimum and leader atomic minimum, including inactive candidates'", "'new':'incorrect'"),
    ("'work_by_variant'", "'parallel_work'"),
])
def test_variant_work_label_mutation(old,new):
    source=MATCHED.read_text();assert old in source
    with pytest.raises(ValueError):guard_proof(DIAG.read_text(),source.replace(old,new,1))
