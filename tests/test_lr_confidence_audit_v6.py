import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from lr_confidence_audit_v6 import (lr_pairs,confidence_report,census_5x5,census_winner,read_native,read_fixture_luma,census_audit)


def test_lr_confidence_ranks_agreeing_pairs():
    h,w=8,40
    left=np.full((h,w),6.,np.float32)
    right=np.full_like(left,6)
    right[:,5:15]=22
    valid=np.ones((h,w),np.uint32)
    lc=np.full_like(left,.9);rc=np.full_like(left,.9)
    rc[:,5:15]=.005
    both,accept,score,residual=lr_pairs(left,right,valid,valid,lc,rc)
    assert both.sum()==h*(w-6)
    report=confidence_report(both,accept,score,residual)
    assert report['native_lr_accepted_comparable']<1
    assert report['confidence_sweep'][-1]['lr_agreement_given_comparable']>=report['native_lr_accepted_comparable']


def test_reject_invalid_dimensions():
    arr=np.ones((8,10),np.float32)
    with pytest.raises(ValueError):lr_pairs(arr,arr,arr,arr,arr[:3],arr)
    with pytest.raises(ValueError):lr_pairs(arr,arr,arr,arr,arr,arr,tol=-1)
    with pytest.raises(ValueError):census_5x5(np.ones((4,10),np.uint32))
    with pytest.raises(ValueError):census_winner(arr.astype(np.uint32),arr.astype(np.uint32),np.array([4]),np.array([4]),4,'backwards')


def make_shift(d=8):
    rng=np.random.default_rng(12)
    left=rng.integers(0,256,(60,90),dtype=np.uint32)
    right=np.zeros_like(left)
    right[:,:-d]=left[:,d:]
    return left,right


def test_exact_census_known_shift():
    left,right=make_shift()
    ld=census_5x5(left);rd=census_5x5(right)
    ys=np.array([20,24,31]);xs=np.array([45,50,56]);
    disp,b,s,v=census_winner(ld,rd,ys,xs,32,'left')
    assert np.all(disp==8) and np.all(v)
    disp2,b2,s2,v2=census_winner(rd,ld,ys,xs,32,'right')
    assert np.all(disp2==8) and np.all(v2)


def test_tied_census_rejected():
    x=np.full((24,48),100,dtype=np.uint32)
    d=census_5x5(x)
    dep,first,second,valid=census_winner(d,d,np.array([12]),np.array([22]),16,'left')
    assert not valid[0] and first[0]==second[0]


def test_file_sha_and_read_native(tmp_path):
    root=tmp_path/'fixture';(root/'expected').mkdir(parents=True)
    data=np.ones((16,40),np.uint32)
    fields={}
    for eye in ('left','right'):
        path=root/'expected'/f'{eye}.u32';path.write_bytes(data.tobytes())
        fields[f'{eye}_luma']={'file':f'expected/{eye}.u32','bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (root/'manifest.json').write_text(json.dumps({'rectified':{'width':40,'height':16},'expected':fields}))
    x,y=read_fixture_luma(root)
    assert x.shape==(16,40) and y.shape==(16,40)
    prefix=tmp_path/'raw'
    for eye in ('left','right'):
        for f,ext,arr in [('disparity','f32',np.ones(640,np.float32)),('confidence','f32',np.ones(640,np.float32)),('validity','u32',np.ones(640,np.uint32))]:
            arr.tofile(f'{prefix}.{eye}.{f}.{ext}')
    o=read_native(prefix,(16,40))
    assert o['left_confidence'].shape==(16,40)
    (root/'expected/left.u32').write_bytes(b'broken')
    with pytest.raises(ValueError):read_fixture_luma(root)


def test_audit_unique_vs_native():
    left,right=make_shift(8)
    h,w=left.shape
    nat={f'{eye}_{field}':np.full((h,w),8 if field=='disparity' else 1,dtype=np.float32 if field=='disparity' else np.uint32) for eye in ('left','right') for field in ('disparity','validity')}
    report=census_audit(left,right,nat,32,8)
    assert report['left']['native_equals_raw_wta_1px_fraction']>0.7
    assert report['right']['native_equals_raw_wta_1px_fraction']>0.7
