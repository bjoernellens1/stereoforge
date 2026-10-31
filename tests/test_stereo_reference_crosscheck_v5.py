import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from stereo_reference_crosscheck_v5 import (
    valid_map,lr_pair,cv_sgbm,compare_view,native_maps,rgb_from_fixture,
    classify_texture)


def stereo(d=16,shift_y=0):
    rng=np.random.default_rng(42)
    left=rng.integers(0,256,(150,320,3),dtype=np.uint8)
    right=np.zeros_like(left)
    for y in range(left.shape[0]):
        ry=y+shift_y
        if 0<=ry<left.shape[0]:right[ry,:-d]=left[y,d:]
    return left,right


def test_physical_lr_and_wrong_match():
    h,w=12,40
    l=np.full((h,w),8.,dtype=np.float32)
    r=l.copy()
    v=np.ones((h,w),bool)
    ok,both,s=lr_pair(l,r,v,v)
    assert ok.sum()==h*(w-8)
    assert s['accepted_fraction_among_opposite_selected']==1
    r[2:9,4:10]=50
    ok,both,s=lr_pair(l,r,v,v)
    assert s['accepted_fraction_among_opposite_selected']<1
    assert ok.sum()<h*(w-8)


def test_validity_and_errors():
    l=np.array([[np.nan,0.,1.,9.]],np.float32)
    assert valid_map(l,disparities=8).tolist()==[[False,False,True,False]]
    with pytest.raises(ValueError):lr_pair(l,l,np.ones((2,2),bool),np.ones((2,2),bool))
    with pytest.raises(ValueError):lr_pair(l,l,np.ones((1,4),bool),np.ones((1,4),bool),tol=-1)


def test_cv_stereo_positive_in_both_views():
    l,r=stereo(d=16)
    dl,dr,seconds=cv_sgbm(l,r,disparities=64,block=5,uniqueness=5)
    vl=valid_map(dl,disparities=64);vr=valid_map(dr,disparities=64)
    assert seconds>=0 and np.median(dl[vl])==pytest.approx(16,abs=1)
    assert np.median(dr[vr])==pytest.approx(16,abs=1)
    _,_,stats=lr_pair(dl,dr,vl,vr)
    assert stats['accepted_fraction_among_opposite_selected']>.85


def test_vertical_error_reduces_support():
    l,r=stereo(d=16,shift_y=5)
    dl,dr,_=cv_sgbm(l,r,disparities=64,block=5,uniqueness=5)
    vl=valid_map(dl,disparities=64);vr=valid_map(dr,disparities=64)
    _,_,s=lr_pair(dl,dr,vl,vr)
    assert s['accepted_fraction_all_pixels']<.6


def test_native_shape_and_fixture_sha(tmp_path):
    prefix=tmp_path/'test'
    for eye in ('left','right'):
        np.full(120,11,np.float32).tofile(str(prefix)+f'.{eye}.disparity.f32')
        np.ones(120,np.uint32).tofile(str(prefix)+f'.{eye}.validity.u32')
    n=native_maps(prefix,(10,12))
    assert n['right_disparity'].shape==(10,12)
    np.ones(13,np.float32).tofile(str(prefix)+'.left.disparity.f32')
    with pytest.raises(ValueError):native_maps(prefix,(10,12))
    fixture=tmp_path/'fixture';(fixture/'expected').mkdir(parents=True)
    raw=np.ones((16,40,3),np.uint8).tobytes()
    (fixture/'expected/l.rgb8').write_bytes(raw)
    j={'rectified':{'height':16,'width':40},'expected':{'rect_left_rgb':{'file':'expected/l.rgb8','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}}
    assert rgb_from_fixture(fixture,'rect_left_rgb',j).shape==(16,40,3)
    j['expected']['rect_left_rgb']['sha256']='bad'
    with pytest.raises(ValueError):rgb_from_fixture(fixture,'rect_left_rgb',j)


def test_crossview_and_texture():
    l,r=stereo()
    n=np.full(l.shape[:2],16,np.float32)
    mask=np.ones(n.shape,bool)
    q=compare_view(n,n,mask,mask)
    assert q['agree_within_1px_fraction']==1
    bins=classify_texture(l,n,n,mask,mask,n,n,mask,mask)
    assert bins and max(x['native_lr_accepted'] for x in bins.values())>0
