import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from fast60_multi_hypothesis import generate,evaluate

def scene():
    rng=np.random.default_rng(3)
    left=rng.integers(10,240,size=(48,96),dtype=np.uint8)
    right=np.zeros_like(left);right[:,:-8]=left[:,8:]
    coarse=np.full(left.shape,8,dtype=np.float32)
    return left,right,coarse

def test_known_shift_recall_and_work():
    l,r,c=scene()
    out,n,a=generate(l,r,c,disparities=16,spread=4,cap=12)
    truth=np.full(l.shape,8,dtype=np.float32)
    result=evaluate(out,truth)
    assert result['overall']['candidate_recall']>.9
    assert result['candidate_evaluations']<l.size*16
    assert a.count>0

def test_anchor_seeds_correct_missing_coarse():
    l,r,c=scene();c[:]=np.nan
    out,n,a=generate(l,r,c,disparities=16,spread=3)
    assert a.count>0
    for x,y,d in zip(a.left_x,a.y,a.disparity): assert d in out[y,x]

def test_no_truth_in_generator():
    l,r,c=scene()
    a,_,_=generate(l,r,c,disparities=16)
    truth=np.full(l.shape,8,dtype=np.float32)
    result=evaluate(a,truth,risk=np.zeros(l.shape,dtype=bool))
    assert result['risk']['candidate_recall'] is None
    assert result['non_risk']['candidate_recall'] is not None

def test_bad_inputs():
    l,r,c=scene()
    with pytest.raises(ValueError):generate(l,r,c,cap=0)
    with pytest.raises(ValueError):generate(l.astype(np.float32),r,c)
    with pytest.raises(ValueError):evaluate(np.zeros((3,3,2),dtype=np.int16),c)
