"""CPU witnesses for shared ping-pong recurrence; no GPU execution."""
from pathlib import Path
import numpy as np
import pytest
from benchmarks.compact_sgm_reference import dense_masked_sgm
from benchmarks.fp2_subgroup_diagnostic import verify_binary, verify_tiny
ROOT=Path(__file__).resolve().parents[1]


def subgroup_model(cost,lo,hi,partition):
    h,w,d=cost.shape;out=np.full((4,h,w,d),np.inf)
    for path,(dx,dy) in enumerate(((1,0),(-1,0),(0,1),(0,-1))):
        for sy in range(h):
            for sx in range(w):
                if 0<=sx-dx<w and 0<=sy-dy<h: continue
                states=np.full((2,128),np.inf);previous=0;x=sx;y=sy
                while 0<=x<w and 0<=y<h:
                    current=1-previous;minimum=min(np.min(states[previous,group]) for group in partition)
                    for lane in range(128):
                        value=np.inf
                        if lane<d and lo[y,x]<=lane<hi[y,x] and np.isfinite(cost[y,x,lane]):
                            best=min(states[previous,lane],states[previous,lane-1]+8 if lane else np.inf,states[previous,lane+1]+8 if lane+1<d else np.inf,minimum+32)
                            value=cost[y,x,lane] if not np.isfinite(minimum) else cost[y,x,lane]+best-minimum
                        states[current,lane]=value
                    out[path,y,x]=states[current,:d];previous=current;x+=dx;y+=dy
    return out


@pytest.mark.parametrize('seed',range(8))
def test_model_matches_independent_dense(seed):
    rng=np.random.default_rng(seed);h,w,d=5,7,128
    lo=rng.integers(0,129,(h,w));hi=np.array([rng.integers(v,129) for v in lo.flat]).reshape(h,w)
    lo[0,:4]=[0,100,0,127];hi[0,:4]=[0,128,16,128]
    cost=rng.integers(0,25,(h,w,d)).astype(float);cost[rng.random(cost.shape)<.15]=np.inf
    order=rng.permutation(128);size=32 if seed%2 else 64;partition=[order[i:i+size] for i in range(0,128,size)]
    np.testing.assert_array_equal(subgroup_model(cost,lo,hi,partition),dense_masked_sgm(cost,lo,hi).paths)


def test_shader_structure_and_binary_identities():
    s=(ROOT/'python/stereoforge/shaders/daxa/fp2_banded_subgroup.slang').read_text()
    assert 'WaveActiveMin(previous[lane])' in s
    assert 'if(lane==0) group_minimum=INF' in s
    assert 'if(WaveIsFirstLane()) InterlockedMin(group_minimum,subgroup_minimum)' in s
    assert s.count('GroupMemoryBarrierWithGroupSync()')==5
    for name,check in [('fp2_subgroup_fixture',verify_binary),('fp2_subgroup_tiny',verify_tiny)]:
        binary=ROOT/'build/daxa'/name
        if not binary.exists():pytest.skip('optional fixture absent')
        assert check(binary)['embedded']['no_path_shader_sha256']


def test_subgroup_capability_mutations():
    import subprocess
    binary=ROOT/'build/daxa/fp2_subgroup_tiny'
    if not binary.is_file(): pytest.skip('optional Daxa subgroup fixture not built')
    assert 'subgroup limit mutations passed' in subprocess.check_output([str(binary),'--test-subgroup-limits'],text=True)
