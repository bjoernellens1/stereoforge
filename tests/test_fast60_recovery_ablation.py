import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from fast60_recovery_ablation import generate,evaluate

def fixture():
    rng=np.random.default_rng(19)
    left=rng.integers(0,256,(48,80),dtype=np.uint8)
    right=np.zeros_like(left);right[:,:-8]=left[:,8:]
    coarse=np.full(left.shape,8,dtype=np.float32)
    return left,right,coarse

def test_progressive_modes_and_no_truth_leak():
    l,r,c=fixture()
    outs=[]
    for mode in ('coarse','anchors','propagate','rescue'):
        result=generate(l,r,c,mode=mode,disparities=16,stride=4,cap=12,spread=3)
        assert result[0].shape==(48,80,12)
        assert np.all((result[0]==-1)|((result[0]>=0)&(result[0]<16)))
        assert np.max((result[0]>=0).sum(axis=2))<=12
        outs.append(result)
    assert outs[0][2]['anchor_evaluations']==0
    assert outs[-1][2]['rescue_evaluations']>=0
    truth=np.full(l.shape,8,dtype=np.float32);risk=np.zeros(l.shape,bool)
    report=evaluate(*outs[-1],truth,risk,16)
    assert 0<=report['overall']['recall']<=1
    assert report['evaluations_vs_dense']>=0
    # Ground truth is used solely by the evaluator.
    truth[:]=5
    assert np.array_equal(outs[-1][0],generate(l,r,c,mode='rescue',disparities=16,stride=4,cap=12,spread=3)[0])

def test_reject_malformed():
    l,r,c=fixture()
    with pytest.raises(ValueError):generate(l,r,c[:4],mode='coarse')
    with pytest.raises(ValueError):generate(l,r,c,mode='bad')
    with pytest.raises(ValueError):generate(l,r,c,mode='rescue',cap=0)
