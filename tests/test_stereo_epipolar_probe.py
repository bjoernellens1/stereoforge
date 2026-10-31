import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from stereo_epipolar_probe import orb_epipolar, load_rgb

def test_orb_rectified_image_shift():
    import cv2
    rng=np.random.default_rng(314)
    gray=rng.integers(0,256,(270,450),np.uint8)
    # Well-aligned, signed-positive stereo pair with a constant 14px shift.
    rgb=np.repeat(gray[...,None],3,axis=2)
    right=np.zeros_like(rgb)
    right[:,:-14]=rgb[:,14:]
    result,pairs=orb_epipolar(rgb,right,max_disparity=64,nfeatures=2000)
    assert result['usable_matches']>200
    assert result['positive_disparity_fraction']>.95
    assert result['epipolar_p90_abs_dy_px']<2.0
    assert abs(result['positive_disparity_median_px']-14)<1.0


def test_inputs_rejected_and_blank_no_features():
    rgb=np.zeros((128,256,3),np.uint8)
    result,_=orb_epipolar(rgb,rgb)
    assert result['usable_matches']==0
    with pytest.raises(ValueError): orb_epipolar(rgb,rgb[:80])
    with pytest.raises(ValueError): orb_epipolar(rgb,rgb,max_disparity=999)


def test_pinned_fixture_schema_load_and_verify(tmp_path):
    import hashlib
    from stereo_epipolar_probe import load_rgb
    img=np.arange(10*12*3,dtype=np.uint8).reshape(10,12,3)
    blob=img.tobytes()
    (tmp_path/'expected').mkdir()
    (tmp_path/'expected'/'rect_left.rgb8').write_bytes(blob)
    doc={
        'schema':'stereoforge.daxa_preprocess_fixture.v1',
        'source':{'eye_width':12,'height':10},
        'rectified':{'width':12,'height':10},
        'expected':{'rect_left_rgb':{
            'file':'expected/rect_left.rgb8','bytes':len(blob),
            'sha256':hashlib.sha256(blob).hexdigest()
        }}
    }
    assert np.array_equal(load_rgb(tmp_path,'rect_left_rgb',doc),img)
    legacy={'rectification':doc['rectified'],'expected':doc['expected']}
    assert np.array_equal(load_rgb(tmp_path,'rect_left_rgb',legacy),img)
    from copy import deepcopy
    bad=deepcopy(doc);bad['expected']['rect_left_rgb']['sha256']='0'*64
    with pytest.raises(ValueError,match='checksum'):
        load_rgb(tmp_path,'rect_left_rgb',bad)
    with pytest.raises(ValueError,match='missing rectified'):
        load_rgb(tmp_path,'rect_left_rgb',{'expected':doc['expected']})


def test_new_fixture_dimensions_override_source_dimensions(tmp_path):
    import hashlib
    from stereo_epipolar_probe import load_rgb
    blob=bytes(8*6*3)
    (tmp_path/'left.bin').write_bytes(blob)
    doc={'rectified':{'width':8,'height':6},
         'source':{'eye_width':1280,'height':720},
         'expected':{'rect_left_rgb':{'file':'left.bin','bytes':len(blob),
                    'sha256':hashlib.sha256(blob).hexdigest()}}}
    assert load_rgb(tmp_path,'rect_left_rgb',doc).shape == (6,8,3)
