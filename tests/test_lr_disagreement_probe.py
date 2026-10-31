import sys
from pathlib import Path
import numpy as np
import json
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from lr_disagreement_probe import compare,analyze,read_dump


def make_constant(h=24,w=56,d=8):
    l=np.full((h,w),d,np.float32)
    r=np.full_like(l,d)
    lv=np.ones((h,w),bool)
    rv=np.ones((h,w),bool)
    return l,r,lv,rv


def test_perfect_physical_match_and_boundaries():
    v=make_constant()
    r=compare(*v,stride=1)
    assert r['samples_left_selected']==24*56
    assert r['opposite_selected']==24*(56-8)
    assert r['accepted']==r['opposite_selected']
    assert r['absolute_residual_p90_px']==0
    x=analyze(*v,stride=2,radius=3,vertical_radius=2)
    assert x['baseline']['accepted_fraction_of_both_selected']==1
    assert x['total_left_valid_fraction']==1


def test_right_disparity_systematic_offset():
    l,r,lv,rv=make_constant()
    r[:]=13
    x=compare(l,r,lv,rv,stride=1)
    assert x['accepted']==0
    assert x['absolute_residual_p50_px']==5
    assert x['signed_disparity_residual_median_px']==-5


def test_horizontal_and_vertical_probes_identify_correspondence_shift():
    l,r,lv,rv=make_constant(h=20,w=64,d=8)
    # Sparse target support only at expected x-d+2: using a couple disjoint
    # scanline anchors prevents the reference itself from being universally valid.
    lv[:]=False
    rv[:]=False
    lv[3:17:2,20:55:7]=True
    yy,xx=np.where(lv)
    rv[yy,xx-8+2]=True
    x=analyze(l,r,lv,rv,stride=1,radius=3,vertical_radius=2)
    assert x['horizontal_offsets']['2']['accepted_fraction_of_left_selected']==1
    assert x['horizontal_offsets']['0']['accepted']==0
    rv[:]=False
    rv[yy+1,xx-8]=True
    x=analyze(l,r,lv,rv,stride=1,radius=2,vertical_radius=2)
    assert x['vertical_offsets']['1']['accepted_fraction_of_left_selected']==1
    assert x['vertical_offsets']['0']['accepted']==0


def test_reject_bad_shapes_and_nonpositive_tolerance():
    l,r,lv,rv=make_constant()
    import pytest
    with pytest.raises(ValueError): compare(l,r[:3],lv,rv)
    with pytest.raises(ValueError): compare(l,r,lv,rv,tolerance=-1)
    with pytest.raises(ValueError): compare(l,r,lv,rv,direction='invalid')


def test_binary_cli(tmp_path):
    l,r,lv,rv=make_constant()
    stem=tmp_path/'run-1'
    for suffix,arr in [('.left.disparity.f32',l),('.right.disparity.f32',r),
                       ('.left.validity.u32',lv.astype('<u4')),('.right.validity.u32',rv.astype('<u4'))]:
        np.asarray(arr).tofile(str(stem)+suffix)
    (tmp_path/'run-1.json').write_text(json.dumps({'width':l.shape[1],'height':l.shape[0]}))
    out=tmp_path/'out.json'
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'benchmarks'/'lr_disagreement_probe.py'),
                    '--report',str(tmp_path/'run-1.json'),'--output',str(out), '--stride','1'],check=True)
    data=json.loads(out.read_text())
    assert data['baseline']['accepted_fraction_of_both_selected']==1
    try:
        read_dump(stem,56,25)
    except ValueError: pass
    else: raise AssertionError('Wrong dump dimensions must fail')
