"""CPU-only parallel FP2 guards; no GPU/performance qualification."""
from pathlib import Path
import pytest
import test_fp2_resident_hotpath as resident
ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'apps/fp2_parallel_fixture.cpp'
TINY=ROOT/'apps/fp2_parallel_tiny.cpp'
SHADER=ROOT/'python/stereoforge/shaders/daxa/fp2_banded_parallel.slang'
OWNER=ROOT/'apps/fp2_prototype_fixture.cpp'


def guard_shader(shader):
    s=''.join(shader.split())
    for item in ('[numthreads(128,1,1)]','if(line>=(horizontal?p.height:p.width))return;',
                 'previous[lane]=INF;', 'for(uintstride=64;stride>0;stride>>=1)',
                 'if(lane<stride)reduction[lane]=min(reduction[lane],reduction[lane+stride]);',
                 'if(lane<p.disparities&&lane>=lo[i]&&lane<hi[i])',
                 'offsets[i]+lane-lo[i]', 'if(c!=INF)', 'if(minimum!=INF)',
                 'previous[lane-1]!=INF', 'previous[lane+1]!=INF',
                 'current[lane]=value;GroupMemoryBarrierWithGroupSync();previous[lane]=current[lane];GroupMemoryBarrierWithGroupSync();',
                 'if(p.paths!=0)paths[uint64_t(pc.pass)*allocated+j]=value;'):
        if item not in s:raise ValueError('shader invariant '+item)
    if s.count('return;')!=1 or s.count('GroupMemoryBarrierWithGroupSync();')!=5:raise ValueError('nonuniform return/barrier')


def guard_host(src,tiny,owner):
    cmd=resident.record(src)
    for target in (src,tiny):
        for item in ('limits.maxComputeWorkGroupInvocations>=128','limits.maxComputeWorkGroupSize[0]>=128',
                     'limits.maxComputeSharedMemorySize>=1536','limits.maxComputeWorkGroupCount[0]>=groups',
                     'vkGetPhysicalDeviceProperties(daxa_dvc_get_vk_physical_device(d.get()),&props)',
                     'admit_parallel_device(d,uint32_t(std::max<uint64_t>'):
            if item not in target:raise ValueError('device limits')
    for item in ('ResetCostPush{d.device_address(cost).value(),uint32_t(max_words)}',
                 'ResetCostPush{d.device_address(sums).value(),uint32_t(max_words)}',
                 'recorder.dispatch({pass<2?720u:1280u,1,1})', 'for(uint32_t pass=0;pass<4;++pass)',
                 'Push{d.device_address(rec_params[eye]).value(),pass}',
                 'if(eye)recorder.pipeline_barrier', 'reset_timestamps({.query_pool=queries,.start_index=0,.count=2})'):
        if item not in cmd:raise ValueError('dispatch invariant '+item)
    if 'addr(sums),0,addr(scratch)' not in src or 'addr(sums),0,addr(raw)' not in src:raise ValueError('diagnostic path disabled')
    for item in ('std::max(f[0].offsets.back(),f[1].offsets.back())','std::max(f[0].scratch_words,f[1].scratch_words)',
                 'validate_regions(regions)','admission(context,sizes)', 'validate_metric_lut(lut,numerator,cx,reinterpret_cast<uintptr_t>(d.get()))'):
        if item not in src:raise ValueError('resource invariant '+item)
    for token in ('buffer_host_address','memcpy','memset','create_buffer','create_memory','wait_on_submit','wait_idle','std::ofstream'):
        if token in cmd:raise ValueError('host operation in recording')
    for token in ('recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE})',
                  '.src_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::COMPUTE_SHADER_WRITE'):
        if token not in cmd:raise ValueError('barrier')
    loop=src[src.index('for(uint32_t frame=0;frame<iterations;++frame)'):src.index('uint32_t failure=0;',src.index('for(uint32_t frame=0;frame<iterations;++frame)'))]
    if not(loop.index('auto commands=record()')<loop.index('d.submit_commands')<loop.index('.queue_submit_index=submit_id')<loop.index('daxa_timeline_query_pool_query_results')):raise ValueError('completion')
    if any(x in loop for x in ('buffer_host_address','memcpy','create_buffer','std::ofstream')):raise ValueError('frame readback')
    for token in ('recorder.copy_buffer_to_buffer({.src_buffer=sums,.dst_buffer=debug_sum[eye],.size=f[eye].offsets.back()*4})',
                  'std::memcmp(d.buffer_host_address(debug_sum[eye]).value(),oracle_out[eye].sums.data()',
                  'std::memcmp(d.buffer_host_address(debug_paths[eye]).value(),oracle_out[eye].paths.data()',
                  'd.device_address(debug_paths[eye]).value()',
                  'd.wait_idle();uint32_t failure=0', 'validate_regions(regions)'):
        if token not in tiny:raise ValueError('tiny parity')
    if 'device.wait_idle();for(auto b:buffers){device.destroy_buffer(b);}device.collect_garbage();memory.clear();' not in owner:raise ValueError('owner cleanup')


def current():return APP.read_text(),TINY.read_text(),OWNER.read_text()

def test_sources():guard_shader(SHADER.read_text());guard_host(*current())


@pytest.mark.parametrize('old,new',[
 ('[numthreads(128,1,1)]','[numthreads(64,1,1)]'),
 ('if(line >= (horizontal?p.height:p.width)) return;','if(lane>10) return;'),
 ('previous[lane]=INF;','previous[lane]=0;'),
 ('if(c!=INF)','if(true)'),('if(minimum!=INF)','if(true)'),
 ('offsets[i]+lane-lo[i]','offsets[i]+lane'),
 ('current[lane]=value;','if(lane>0)return;current[lane]=value;'),
 ('GroupMemoryBarrierWithGroupSync();',''),
 ('if(p.paths!=0) paths[','if(true) paths['),
])
def test_shader_mutations(old,new):
    s=SHADER.read_text();assert old in s
    with pytest.raises(ValueError):guard_shader(s.replace(old,new,1))


@pytest.mark.parametrize('which,old,new',[
 ('src','limits.maxComputeWorkGroupInvocations>=128','true'),
 ('tiny','d.device_address(debug_paths[eye]).value()','d.device_address(debug_paths[0]).value()'),
 ('tiny','oracle_out[eye].paths.data()','oracle_out[0].paths.data()'),
 ('src','addr(sums),0,addr(scratch)','addr(sums),addr(paths),addr(scratch)'),
 ('src','recorder.dispatch({pass<2?720u:1280u,1,1})','recorder.dispatch({1280u,1,1})'),
 ('src','recorder.set_pipeline(metric)','d.buffer_host_address(raw);recorder.set_pipeline(metric)'),
 ('src','.queue_submit_index=submit_id','.queue_submit_index=0'),
 ('tiny','.dst_buffer=debug_sum[eye]','.dst_buffer=debug_sum[0]'),
 ('tiny','oracle_out[eye].sums.data()','oracle_out[0].sums.data()'),
])
def test_host_mutations(which,old,new):
    src,tiny,owner=current();target=src if which=='src' else tiny;assert old in target
    changed=target.replace(old,new,1)
    with pytest.raises((ValueError,IndexError)):guard_host(changed if which=='src' else src,changed if which=='tiny' else tiny,owner)


def test_fake_actual_source(tmp_path,monkeypatch):
    monkeypatch.setattr(resident,'APP',APP)
    resident.test_fake_actual_recording(tmp_path)


def guard_proof(diag):
    for token in ("proof.get('scope')!='parallel parity'", "proof.get('manifest_sha256')!=manifest_sha", "proof.get('host')!=host", "proof.get('tiny')!=tiny", "proof.get('harness_sha256')!=harness_sha", "proof.get('baseline_sha256')!=baseline_sha", "proof['gpu_outputs'][key]!=baseline['gpu_outputs'][key]"):
        if token not in diag:raise ValueError('parity proof identity')


def test_proof_source():guard_proof((ROOT/'benchmarks/fp2_parallel_diagnostic.py').read_text())


def test_proof_mutation():
    src=(ROOT/'benchmarks/fp2_parallel_diagnostic.py').read_text();old="proof.get('host')!=host";assert old in src
    with pytest.raises(ValueError):guard_proof(src.replace(old,'False',1))
