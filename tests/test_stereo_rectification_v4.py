import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from stereo_rectification_v3 import sparse_patch_2d
from stereo_rectification_v4 import reciprocal_samples, by_tiles, compare_sgm, read_sgm


def stereo(d=15,dy=0,seed=42):
    rng=np.random.default_rng(seed)
    l=rng.integers(0,256,(210,380,3),dtype=np.uint8)
    r=np.zeros_like(l)
    y0=max(0,-dy);y1=min(210,210-dy)
    r[y0+dy:y1+dy,:-d]=l[y0:y1,d:]
    return l,r

@pytest.mark.parametrize('dy',[0,4,-3])
def test_reciprocity_recovers_known_shift(dy):
    l,r=stereo(dy=dy)
    _,p=sparse_patch_2d(l,r,max_disparity=32,max_dy=6,samples=90)
    q=reciprocal_samples(l,r,p,max_disparity=32,vertical_radius=6)
    strict=[v for v in q if v['strict_reciprocal']]
    assert len(strict)>=55,(dy,len(strict))
    assert np.mean([abs(v['dy']+dy)<=1 for v in strict])>.95
    assert np.mean([abs(v['dx']-15)<=1 for v in strict])>.95
    assert np.mean([abs(v['cycle_dx'])<=1 and abs(v['cycle_dy'])<=1 for v in strict])==1.


def test_sgm_compare_diagnostic():
    h,w=40,80
    arr={'left_disparity':np.full((h,w),12,dtype='f4'),
         'right_disparity':np.full((h,w),12,dtype='f4'),
         'left_validity':np.ones((h,w),dtype='u4'),
         'right_validity':np.ones((h,w),dtype='u4')}
    points=[{'x':40,'y':20,'xr':28,'yr':20,'dx':12,'dy':0},
            {'x':42,'y':20,'xr':30,'yr':20,'dx':12,'dy':0}]
    x,_=compare_sgm(points,arr)
    assert x['both_match_independent_patch']==1
    arr['right_disparity'][20,30]=20
    x,_=compare_sgm(points,arr)
    assert x['left_matches_independent_patch']==1
    assert x['right_matches_independent_patch']==0.5
    assert x['left_right_agree_at_independent_patch']==0.5


def test_tile_counts_and_empty():
    r=by_tiles([{'x':2,'y':3,'dy':0,'gain_from_vertical_search':0},
                {'x':80,'y':75,'dy':6,'gain_from_vertical_search':0.2}],100,100,2,2)
    assert sum(e['count'] for e in r)==2
    assert r[0]['count']==1 and r[-1]['count']==1
    assert r[1]['abs_dy']['p90'] is None


def test_read_sgm_dump_shape(tmp_path):
    prefix=tmp_path/'run-1'
    for eye in ('left','right'):
        np.full(80,5,dtype='<f4').tofile(str(prefix)+f'.{eye}.disparity.f32')
        np.ones(80,dtype='<u4').tofile(str(prefix)+f'.{eye}.validity.u32')
    arr=read_sgm(prefix,10,8)
    assert arr['right_validity'].shape==(8,10)
    np.ones(4,dtype='<u4').tofile(str(prefix)+'.right.validity.u32')
    with pytest.raises(ValueError):read_sgm(prefix,10,8)
