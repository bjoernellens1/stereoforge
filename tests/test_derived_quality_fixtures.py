"""CPU-only independent development geometry; no matcher or GPU invocation."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import derived_quality_fixtures as derived


def test_synthetic_geometry_zbuffer_uses_near_surface_and_marks_unseen():
    l=np.arange(16,dtype=np.uint8).reshape(2,8)
    d=np.full((2,8),2,dtype=np.uint32);d[:,4]=4
    case=derived.forward_warp_fixture(l,d)
    # Left x2,d2 and nearer x4,d4 both project to Right x0; x4 wins.
    assert case['right'][0,0]==l[0,4]
    assert case['right_disparity'][0,0]==4
    assert case['right_source_x'][0,0]==4
    assert case['left_occluded'][0,2] and not case['left_visible'][0,2]
    assert not case['right_visible'][0,6] and case['right_unseen'][0,6]
    assert not case['left_visible'][0,0] and case['left_off_image'][0,0]


@pytest.mark.parametrize('kind',['plane','slope','thin-occluder','photometric'])
def test_development_generator_deterministic_and_independent_gt(kind):
    case=derived.development_fixture(kind,32,12,seed=42)
    again=derived.development_fixture(kind,32,12,seed=42)
    for key in case:assert np.array_equal(case[key],again[key])
    y,x=np.nonzero(case['right_visible']);sx=case['right_source_x'][y,x]
    assert np.array_equal(sx-x,case['right_disparity'][y,x])
    assert np.array_equal(case['left_disparity'][y,sx],case['right_disparity'][y,x])
    if kind!='photometric':assert np.array_equal(case['right'][y,x],case['left'][y,sx])
    if kind=='thin-occluder':assert case['left_occluded'].any() and 9 in case['right_disparity']


def test_export_hashes_and_masks(tmp_path):
    import hashlib
    m=derived.export_fixture(tmp_path/'fixture','thin-occluder',32,12)
    assert m['promoted'] is False
    for row in m['planes'].values():
        data=(tmp_path/'fixture'/row['file']).read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    assert 'region_thin_foreground' in m['planes']
    assert 'region_left_thin_foreground' in m['planes']
    assert 'region_right_thin_foreground' in m['planes']


def test_region_scores_detect_false_lr_on_occluded_geometry():
    left=np.arange(32,dtype=np.uint8)[None,:].repeat(5,axis=0)
    disparity=np.full((5,32),2,'<u4');disparity[:,16:18]=9
    case=derived.forward_warp_fixture(left,disparity)
    raw=[{'disparity.f32':np.full((5,32),2.,'<f4').tobytes(),'validity.u32':np.ones((5,32),'<u4').tobytes()}]*2
    scores=derived.paired_region_scores(raw,case)
    assert scores['promoted'] is False
    left_scores=scores['regions']['left'];right_scores=scores['regions']['right']
    assert left_scores['left_occluded']['pixels']==10
    assert left_scores['left_occluded']['lr_accepted']==10
    assert left_scores['left_occluded']['lr_false_geometric_correspondence']==10
    # Right unseen x14/15 are LR accepted; off-right-border x30/31 are not.
    assert right_scores['right_unseen']['pixels']==20
    assert right_scores['right_unseen']['lr_accepted']==10
    assert right_scores['right_unseen']['lr_false_geometric_correspondence']==10
    assert right_scores['thin_foreground']['pixels']==10
    assert right_scores['thin_foreground']['lr_accepted']==10
    assert right_scores['thin_foreground']['lr_false_geometric_correspondence']==10
    assert right_scores['thin_foreground']['mean_absolute_disparity_error']==7


def test_eye_regions_use_projected_thin_foreground_and_right_gt_boundaries():
    left=np.arange(32,dtype=np.uint8)[None,:].repeat(5,axis=0)
    disparity=np.full((5,32),2,'<u4');disparity[:,16:18]=9
    case=derived.forward_warp_fixture(left,disparity)
    lregions=derived.regions(case,'left');rregions=derived.regions(case,'right')
    assert np.flatnonzero(lregions['thin_foreground'][2]).tolist()==[16,17]
    assert np.flatnonzero(rregions['thin_foreground'][2]).tolist()==[7,8]
    assert case['right_source_x'][2,7:9].tolist()==[16,17]
    assert np.flatnonzero(rregions['surface_boundary'][2]).tolist()==list(range(4,18))+list(range(27,32))
    assert np.flatnonzero(lregions['surface_boundary'][2]).tolist()==list(range(13,20))
    assert np.flatnonzero(rregions['right_unseen'][2]).tolist()==[14,15,30,31]
    assert np.flatnonzero(lregions['left_occluded'][2]).tolist()==[9,10]
