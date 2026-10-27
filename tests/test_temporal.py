import numpy as np
import pytest
from stereoforge.types import CameraModel, DepthFrame, Provenance
from stereoforge.temporal import reproject_depth, TemporalHistory, adaptive_bounds


def frame(depth=(1., 2., 3.), timestamp=0, source=1):
    z=np.array([depth], dtype=np.float32); shape=z.shape
    return DepthFrame(np.full(shape,5.,np.float32),z,np.full(shape,.8),np.isfinite(z),np.full(shape,Provenance.STEREO,np.uint8),np.full(shape,source,np.int64),np.zeros(shape,np.int64),timestamp,'cal')


def test_translation_zbuffer_and_age():
    f=frame((1.,4.,np.nan)); pose=np.eye(4); pose[0,3]=1
    out=reproject_depth(f,CameraModel(3,1,1,1,0,0),pose,10)
    assert out.valid.tolist()==[[False,True,False]]
    assert out.depth_m[0,1]==1
    assert out.age_ns[0,1]==10
    assert np.isnan(out.disparity).all()
    assert out.provenance[0,1]==Provenance.REPROJECTED


def test_rotation_and_invalid_inputs():
    f=frame(); p=np.diag([-1.,-1.,1.,1.])
    out=reproject_depth(f,CameraModel(3,1,1,1,0,0),p,0)
    assert out.valid.tolist()==[[True,False,False]]
    with pytest.raises(ValueError): reproject_depth(f,CameraModel(3,1,1,1,0,0),np.ones((4,4)),0)
    with pytest.raises(ValueError): reproject_depth(f,CameraModel(3,1,1,1,0,0),np.eye(4),-1)


def test_fusion_duplicate_occlusion_and_expiry():
    h=TemporalHistory(max_age_ns=20)
    current=frame(timestamp=10); prior=frame((1.02,8.,2.),timestamp=10)
    out=h.fuse(current,prior)
    assert out.depth_m[0,0]==current.depth_m[0,0]
    assert out.confidence[0,0]==current.confidence[0,0]
    assert out.depth_m[0,1]==2
    prior.source_ids[:]=2
    out=h.fuse(frame(timestamp=11),prior)
    assert 1<out.depth_m[0,0]<1.02
    assert np.isnan(out.disparity[0,0])
    missing=frame((np.nan, np.nan,np.nan),timestamp=12)
    prior.age_ns[:]=30
    assert not h.fuse(missing,prior).valid.any()


def test_history_resets_and_bounds():
    h=TemporalHistory(max_frames=2)
    for i in range(3): h.fuse(frame(timestamp=i),None)
    assert len(h.frames)==2
    changed=frame(timestamp=4); changed.calibration_id='other'
    out=h.fuse(changed,frame(timestamp=4))
    assert len(h.frames)==1
    assert out.calibration_id=='other'
    h.fuse(frame(timestamp=1),None)
    assert len(h.frames)==1
    h.reset(); assert not h.frames


def test_adaptive_ranges_and_refresh():
    f=frame(); f.confidence[0,1]=.1; f.disparity[0,2]=np.nan
    lo,hi,count=adaptive_bounds(f,16,1,radius=2)
    assert lo.tolist()==[[3,0,0]] and hi.tolist()==[[8,16,16]] and count==37
    lo,hi,count=adaptive_bounds(f,16,10)
    assert count==48 and (lo==0).all() and (hi==16).all()
    assert adaptive_bounds(None,16,1,shape=(2,3))[2]==96


def test_fused_ancestry_survives_warp_and_is_not_reaveraged():
    h=TemporalHistory()
    a=frame(timestamp=1,source=1)
    b=frame((1.02,2.02,3.02),timestamp=1,source=2)
    fused=h.fuse(a,b)
    warped=reproject_depth(fused,CameraModel(3,1,1,1,0,0),np.eye(4),2)
    assert (warped.provenance==Provenance.FUSED).all()
    again=h.fuse(frame(timestamp=2,source=1),warped)
    np.testing.assert_array_equal(again.depth_m,a.depth_m)
    np.testing.assert_array_equal(again.confidence,a.confidence)


def test_gap_reset_and_metric_predicted_disparity():
    h=TemporalHistory(max_age_ns=10)
    h.fuse(frame(timestamp=0),None)
    prior=frame(timestamp=20,source=2)
    out=h.fuse(frame((np.nan,np.nan,np.nan),timestamp=20),prior)
    assert not out.valid.any() and len(h.frames)==1
    pose=np.eye(4); pose[2,3]=1
    warped=reproject_depth(frame(),CameraModel(3,1,10,10,0,0),pose,1,
                           fx=10,baseline_m=.2,principal_offset=.25)
    assert warped.disparity[0,0]==pytest.approx(1.25)
    assert adaptive_bounds(warped,16,1,radius=1)[2]<48
    with pytest.raises(ValueError):
        reproject_depth(frame(),CameraModel(3,1,10,10,0,0),pose,1,fx=10)
