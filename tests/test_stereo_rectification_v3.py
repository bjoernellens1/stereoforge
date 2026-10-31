import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from stereo_rectification_v3 import sparse_patch_2d, orb_geometry, percentiles


def pair(d=12,dy=0,seed=77):
    rng=np.random.default_rng(seed)
    a=rng.integers(0,256,(300,460,3),dtype=np.uint8)
    b=np.zeros_like(a)
    # right(x-d,y+dy) = left(x,y) within bounds
    y0=max(0,-dy);y1=min(a.shape[0],a.shape[0]-dy)
    b[y0+dy:y1+dy,:-d]=a[y0:y1,d:]
    return a,b


def test_aligned_stereo_patch_recovers_horizontal_shift():
    l,r=pair()
    m,coords=sparse_patch_2d(l,r,max_disparity=32,max_dy=6,samples=100)
    assert m['high_confidence']>=65, m
    c=[p for p in coords if p['confident']]
    assert sum(abs(p['dy'])<=1 for p in c)/len(c)>.9
    assert sum(abs(p['dx']-12)<=1 for p in c)/len(c)>.9


def test_vertical_offset_is_visible():
    l,r=pair(d=14,dy=4)
    m,coords=sparse_patch_2d(l,r,max_disparity=32,max_dy=8,samples=100)
    assert m['high_confidence']>=60,m
    c=[p for p in coords if p['confident']]
    assert sum(abs(p['dy']+4)<=1 for p in c)/len(c)>.85
    assert m['high_confidence_fraction_abs_dy_gt_2_with_gain_gt_0_08']>.65


def test_orb_fundamental_rejects_bad_outliers():
    rng=np.random.default_rng(2)
    ps=[]
    for i in range(100):
        x=float(rng.uniform(80,420));y=float(rng.uniform(20,270)); d=float(rng.uniform(8,30))
        ps.append(((x,y),(x-d,y),24.0))
    for i in range(35):
        x=float(rng.uniform(80,420));y=float(rng.uniform(20,270));d=float(rng.uniform(8,30))
        ps.append(((x,y),(x-d,float(rng.uniform(20,270))),24.0))
    m,mask=orb_geometry(ps,64)
    assert m['fundamental_inliers']>=80,m
    assert m['abs_dy_fundamental_inliers']['p90']<1.0,m
    assert len(mask)==len(ps)


def test_rejects_malformed():
    l,r=pair()
    with pytest.raises(ValueError):sparse_patch_2d(l,r[:120])
    with pytest.raises(ValueError):sparse_patch_2d(l,r,patch_size=4)
    with pytest.raises(ValueError):sparse_patch_2d(l,r,max_dy=0)
    assert percentiles([])['p90'] is None
    assert orb_geometry([])[0]['fundamental_inliers']==0
