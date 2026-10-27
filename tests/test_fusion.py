import numpy as np
import pytest
from stereoforge.types import CameraModel, Calibration, StereoFrame, DepthFrame
from stereoforge.fusion import VirtualCamera, default_virtual_camera, synthesize


def scene(width=9, height=5, baseline=.2):
    camera = CameraModel(width,height,width/2,width/2,(width-1)/2,(height-1)/2)
    transform = np.eye(4); transform[0,3] = baseline
    calibration = Calibration(camera,camera,transform)
    rgb = np.arange(height*width*3,dtype=np.uint8).reshape(height,width,3)
    return StereoFrame(rgb,rgb.copy(),calibration,10,0)


def depth(frame, z=2., valid=None):
    shape=frame.left.shape[:2]
    if valid is None: valid=np.ones(shape,bool)
    values=np.broadcast_to(z,shape).astype(np.float32).copy()
    return DepthFrame(np.zeros(shape,np.float32),values,np.ones(shape,np.float32),valid,
        valid.astype(np.uint8),np.zeros(shape,np.uint64),np.zeros(shape,np.int64),frame.timestamp_ns,frame.calibration.identity)


def virtual(frame, **kwargs):
    c=frame.calibration.left
    return VirtualCamera(c.width,c.height,np.rad2deg(2*np.arctan(c.width/(2*c.fx))),**kwargs)


def test_identity_preserves_source_and_inputs():
    frame=scene(); original=frame.left.copy()
    view=synthesize(frame,depth(frame),virtual(frame),exposure_match=False)
    assert view.depth_valid.all()
    np.testing.assert_array_equal(view.rgb,frame.left)
    np.testing.assert_allclose(view.depth_m,2)
    np.testing.assert_array_equal(frame.left,original)
    assert not view.approximate.any()
    assert view.depth_convention=='axial_z'
    assert view.coverage['depth_fraction']==1


def test_midpoint_default_and_projection_shift():
    frame=scene()
    camera=default_virtual_camera(frame.calibration)
    assert camera.camera_to_rig[0,3]==pytest.approx(.1)
    translated=np.eye(4); translated[0,3]=1.
    view=synthesize(frame,depth(frame),virtual(frame,camera_to_rig=translated),exposure_match=False)
    assert not view.depth_valid[:,-2:].any()
    assert view.depth_valid.any()


def test_strict_holes_and_preview_masks():
    frame=scene(); invalid=np.zeros(frame.left.shape[:2],bool)
    strict=synthesize(frame,depth(frame,valid=invalid),virtual(frame))
    assert not strict.rgb_valid.any()
    assert np.isnan(strict.depth_m).all()
    preview=synthesize(frame,depth(frame,valid=invalid),virtual(frame),mode='preview')
    assert preview.rgb_valid.all() and preview.approximate.all()
    assert not preview.depth_valid.any()
    assert np.isnan(preview.depth_m).all()
    np.testing.assert_array_equal(preview.rgb,frame.left)


def test_zbuffer_keeps_nearest_surface():
    frame=scene(width=3,height=1)
    # All source pixels collapse onto one output pixel; nearer source wins.
    values=np.array([[3.,1.,2.]],np.float32)
    camera=VirtualCamera(1,1,160)
    view=synthesize(frame,depth(frame,values),camera,exposure_match=False)
    assert view.depth_m[0,0]==pytest.approx(1)
    np.testing.assert_array_equal(view.rgb[0,0],frame.left[0,1])


def test_cylindrical_depth_is_ray_range():
    frame=scene()
    view=synthesize(frame,depth(frame),virtual(frame,projection='cylindrical'),exposure_match=False)
    assert view.depth_convention=='ray_range'
    assert view.depth_m[2,4]==pytest.approx(2)
    assert np.nanmax(view.depth_m)>2


def test_right_without_observed_depth_adds_no_depth_coverage():
    frame=scene(); valid=np.zeros(frame.left.shape[:2],bool); valid[:,4]=True
    view=synthesize(frame,depth(frame,valid=valid),virtual(frame),exposure_match=False)
    assert view.depth_valid.sum()==5
    assert not view.depth_valid[:,0].any()


def test_explicit_right_depth_supports_right_periphery():
    frame=scene(); invalid=np.zeros(frame.left.shape[:2],bool)
    right=depth(frame)
    view=synthesize(frame,depth(frame,valid=invalid),virtual(frame),right_depth=right)
    assert view.depth_valid.any()
    assert np.all(view.source[view.depth_valid]==2)


def test_source_metadata_rejected():
    frame=scene(); d=depth(frame); d.calibration_id='other'
    with pytest.raises(ValueError,match='calibration'): synthesize(frame,d,virtual(frame))
    d=depth(frame); d.timestamp_ns+=1
    with pytest.raises(ValueError,match='timestamp'): synthesize(frame,d,virtual(frame))


@pytest.mark.parametrize('kwargs',[{'hfov_deg':180},{'hfov_deg':0},{'projection':'unknown'}, {'camera_to_rig':np.zeros((4,4))}])
def test_invalid_virtual_camera(kwargs):
    options=dict(width=9,height=5,hfov_deg=90); options.update(kwargs)
    with pytest.raises(ValueError): VirtualCamera(**options)


def test_foreground_wins_even_if_background_is_more_confident():
    frame=scene()
    frame.left[:]=[250,0,0]; frame.right[:]=[0,0,250]
    near=depth(frame,1.); near.confidence[:]=.1
    far=depth(frame,3.)
    view=synthesize(frame,near,virtual(frame),right_depth=far,exposure_match=False)
    # Center is supported by both, but incompatible depth cannot blend.
    np.testing.assert_array_equal(view.rgb[2,4],[250,0,0])
    assert view.depth_m[2,4]==pytest.approx(1)
    assert view.source[2,4]==1


def test_exposure_matching_only_on_consistent_geometry():
    frame=scene(); frame.left[:]=[100,100,100]; frame.right[:]=[50,50,50]
    view=synthesize(frame,depth(frame),virtual(frame),right_depth=depth(frame),exposure_match=True)
    np.testing.assert_array_equal(view.rgb[2,4],[100,100,100])
    assert view.source[2,4]==3


def test_distorted_raw_identity_is_preserved():
    frame=scene()
    c=frame.calibration.left
    distorted=CameraModel(c.width,c.height,c.fx,c.fy,c.cx,c.cy,(.05,-.01,.001,0.,0.))
    frame.calibration=Calibration(distorted,distorted,frame.calibration.right_to_left)
    view=synthesize(frame,depth(frame),virtual(frame),exposure_match=False)
    # A perspective output is undistorted, not identical to raw geometry; the
    # optical axis remains fixed under radial and tangential distortion.
    assert view.depth_valid[2,4]
    np.testing.assert_array_equal(view.rgb[2,4],frame.left[2,4])


def test_stabilization_cancels_rotation_and_records_pose_time():
    from stereoforge.fusion import stabilize_virtual_camera
    frame=scene(); camera=virtual(frame)
    pose=np.eye(4); pose[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
    stabilized=stabilize_virtual_camera(camera,pose,123)
    np.testing.assert_allclose(pose[:3,:3]@stabilized.camera_to_rig[:3,:3],np.eye(3))
    assert stabilized.pose_timestamp_ns==123
    assert camera.pose_timestamp_ns is None
    with pytest.raises(ValueError): stabilize_virtual_camera(camera,np.zeros((4,4)),123)


def test_color_blending_does_not_invent_fused_depth_provenance():
    frame=scene()
    view=synthesize(frame,depth(frame),virtual(frame))
    assert view.source[2,4]==3
    assert view.provenance[2,4]==1


def test_confidence_seam_selects_one_consistent_source():
    frame=scene(); frame.left[:]=[240,20,20]; frame.right[:]=[20,20,240]
    left=depth(frame); left.confidence[:]=.2
    right=depth(frame); right.confidence[:]=.8
    view=synthesize(frame,left,virtual(frame),right_depth=right,exposure_match=False)
    assert view.source[2,4]==2
    np.testing.assert_array_equal(view.rgb[2,4],[20,20,240])
    assert view.confidence[2,4]==pytest.approx(.8)
