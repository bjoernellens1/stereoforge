from pathlib import Path
import subprocess
import numpy as np
import pytest
from benchmarks.fp2_fused_diagnostic import cost_work_for_eye,verify_binary,verify_tiny
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('eye',[0,1])
def test_work_counts_match_independent_scalar(eye):
    lo=np.array([[0,1,0,2],[1,0,1,0]]);hi=np.array([[3,3,2,3],[3,2,2,3]])
    own=np.array([[1,0,1,1],[1,1,1,1]]);other=np.array([[1,1,0,1],[1,0,1,1]])
    counts=[0,0,0,0]
    for y in range(2):
        for x in range(4):
            for d in range(lo[y,x],hi[y,x]):
                counts[0]+=1;t=x-d if eye==0 else x+d
                if not 0<=t<4:continue
                counts[1]+=1
                if not own[y,x]:continue
                counts[2]+=1
                if other[y,t]:counts[3]+=1
    got=cost_work_for_eye(lo,hi,own,other,eye,3)
    assert list(got.values())==[4*c for c in counts]


def test_bad_masks_are_rejected():
    x=np.array([[2]])
    with pytest.raises(ValueError):cost_work_for_eye(x*0,x*0,x,x,0)


def test_source_and_binary_identity():
    shader=(ROOT/'python/stereoforge/shaders/daxa/fp2_banded_fused.slang').read_text()
    assert 'p.eye==0?xy.x-int(lane):xy.x+int(lane)' in shader
    assert 'support[i]!=0' in shader and 'other_support[other_i]!=0' in shader
    assert 'countbits(own[i]^other[other_i])' in shader
    source=(ROOT/'apps/fp2_fused_fixture.cpp').read_text()
    recording=source[source.index('auto record='):source.index('using Clock=')]
    assert 'set_pipeline(matcher)' not in recording and 'set_pipeline(clear_cost)' not in recording
    assert 'auto cost=alloc(max_words' not in source
    for name,check in [('fp2_fused_fixture',verify_binary),('fp2_fused_tiny',verify_tiny)]:
        if not (ROOT/'build/daxa'/name).is_file(): pytest.skip('optional Daxa diagnostic fixture not built')
        assert check(ROOT/'build/daxa'/name)['embedded']['no_path_shader_sha256']
    assert 'fused cpu contracts passed' in subprocess.check_output([str(ROOT/'build/daxa/fp2_fused_tiny'),'--fused-cpu'],text=True)
