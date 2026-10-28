"""CPU model of lane recurrence and source identity, no Vulkan execution."""
from pathlib import Path
import numpy as np
import pytest
from benchmarks.compact_sgm_reference import dense_masked_sgm
from benchmarks.fp2_parallel_diagnostic import verify_binary, verify_tiny

ROOT = Path(__file__).resolve().parents[1]


def lane_model(cost, lo, hi):
    h,w,d=cost.shape
    out=np.full((4,h,w,d),np.inf)
    for direction,(dx,dy) in enumerate(((1,0),(-1,0),(0,1),(0,-1))):
        for sy in range(h):
            for sx in range(w):
                if 0 <= sx-dx < w and 0 <= sy-dy < h:
                    continue
                prev=np.full(128,np.inf)
                x,y=sx,sy
                while 0<=x<w and 0<=y<h:
                    minimum=np.min(prev)
                    cur=np.full(128,np.inf)
                    for lane in range(int(lo[y,x]),int(hi[y,x])):
                        c=cost[y,x,lane]
                        if not np.isfinite(c): continue
                        best=min(prev[lane],prev[lane-1]+8 if lane else np.inf,
                                 prev[lane+1]+8 if lane+1<d else np.inf,minimum+32)
                        cur[lane]=c if not np.isfinite(minimum) else c+best-minimum
                    out[direction,y,x]=cur[:d]
                    prev=cur
                    x+=dx;y+=dy
    return out


@pytest.mark.parametrize('seed', range(8))
def test_parallel_model_matches_independent_dense(seed):
    rng=np.random.default_rng(seed);h,w,d=5,7,128
    lo=rng.integers(0,129,(h,w));hi=np.array([rng.integers(v,129) for v in lo.flat]).reshape(h,w)
    lo[0,:4]=[0,100,0,127];hi[0,:4]=[0,128,16,128]
    cost=rng.integers(0,25,(h,w,d)).astype(float);cost[rng.random(cost.shape)<.15]=np.inf
    got=lane_model(cost,lo,hi);expected=dense_masked_sgm(cost,lo,hi).paths
    np.testing.assert_array_equal(got,expected)


def test_shader_has_uniform_barriers_and_declared_work():
    s=(ROOT/'python/stereoforge/shaders/daxa/fp2_banded_parallel.slang').read_text()
    assert '[numthreads(128,1,1)]' in s
    assert 'for(uint stride=64;stride>0;stride>>=1)' in s
    assert 'if(lane<stride) reduction[lane]' in s
    assert s.count('GroupMemoryBarrierWithGroupSync()') == 5
    assert 'if(p.paths!=0)' in s


def test_binaries_bind_parallel_shader():
    for name,check in [('fp2_parallel_fixture',verify_binary),('fp2_parallel_tiny',verify_tiny)]:
        path=ROOT/'build/daxa'/name
        if not path.exists(): pytest.skip('optional fixture not built')
        assert check(path)['embedded']['no_path_shader_sha256']


def test_limit_mutations_and_path_buffers():
    import subprocess
    tiny=ROOT/'build/daxa/fp2_parallel_tiny'
    if not tiny.exists(): pytest.skip('optional fixture not built')
    assert 'parallel limit mutations passed' in subprocess.check_output([str(tiny),'--test-parallel-limits'],text=True)
    for filename in ('fp2_parallel_tiny.cpp','fp2_parallel_fixture.cpp'):
        s=(ROOT/'apps'/filename).read_text()
        assert 'admit_parallel_device(d,' in s
        assert 'maxComputeSharedMemorySize>=1536' in s
        assert 'maxComputeWorkGroupInvocations>=128' in s
        assert 'maxComputeWorkGroupCount[0]>=groups' in s
    source=(ROOT/'apps/fp2_parallel_tiny.cpp').read_text()
    assert 'd.device_address(debug_paths[eye]).value()' in source
    assert 'oracle_out[eye].paths.data()' in source
    assert '"path or padding mismatch"' in source
    assert 'std::memset(d.buffer_host_address(debug_paths[eye]).value(),0xff' in source
