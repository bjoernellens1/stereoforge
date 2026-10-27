"""Vulkan projection acceptance: strict geometry and deterministic visibility."""
import numpy as np
import pytest
from stereoforge.types import CameraModel, Calibration, StereoFrame, DepthFrame, Provenance
from stereoforge.fusion import VirtualCamera, synthesize
from stereoforge.gpu_geometry import VulkanViewSynthesizer


@pytest.fixture(scope='module')
def renderer():
    return VulkanViewSynthesizer()


def scene():
    camera=CameraModel(5,5,4,4,2,2)
    pose=np.eye(4); pose[0,3]=.1
    calibration=Calibration(camera,camera,pose)
    rgb=np.zeros((5,5,3),np.uint8); rgb[:,:,0]=180
    frame=StereoFrame(rgb,rgb.copy(),calibration,1,1)
    z=np.full((5,5),2.,np.float32)
    depth=DepthFrame(np.full_like(z,.2),z,np.full_like(z,.8),np.ones(z.shape,bool),
        np.full(z.shape,Provenance.STEREO,np.uint8),np.ones(z.shape,np.int64),
        np.zeros(z.shape,np.int64),1,calibration.identity)
    return frame,depth


def test_identity_strict_geometry(renderer):
    frame,depth=scene()
    camera=VirtualCamera(5,5,float(np.degrees(2*np.arctan(5/8))))
    actual=renderer.render(frame,depth,camera)
    expected=synthesize(frame,depth,camera,exposure_match=False)
    np.testing.assert_array_equal(actual.depth_valid,expected.depth_valid)
    np.testing.assert_allclose(actual.depth_m,expected.depth_m,atol=1e-6,equal_nan=True)
    np.testing.assert_array_equal(actual.rgb,frame.left)
    assert not actual.approximate.any()
    np.testing.assert_array_equal(actual.source,expected.source)


def test_occlusion_and_cylindrical_range(renderer):
    frame,depth=scene()
    # Narrow target collapses distinct source pixels: nearest surface must win.
    depth.depth_m[:]=3; depth.depth_m[2,2]=1
    camera=VirtualCamera(1,1,90.)
    actual=renderer.render(frame,depth,camera)
    assert actual.depth_valid[0,0] and actual.depth_m[0,0]==pytest.approx(1.)
    camera=VirtualCamera(5,5,90.,projection='cylindrical')
    actual=renderer.render(frame,depth,camera)
    expected=synthesize(frame,depth,camera,exposure_match=False)
    np.testing.assert_array_equal(actual.depth_valid,expected.depth_valid)
    np.testing.assert_allclose(actual.depth_m,expected.depth_m,atol=1e-5,equal_nan=True)
    assert actual.depth_convention=='ray_range'


def test_cylindrical_can_observe_behind_virtual_camera(renderer):
    frame,depth=scene()
    pose=np.diag([-1.,1.,-1.,1.])
    camera=VirtualCamera(17,9,350.,projection='cylindrical',camera_to_rig=pose)
    actual=renderer.render(frame,depth,camera)
    expected=synthesize(frame,depth,camera,exposure_match=False)
    assert expected.depth_valid.any()
    np.testing.assert_array_equal(actual.depth_valid,expected.depth_valid)
    np.testing.assert_allclose(actual.depth_m,expected.depth_m,atol=1e-5,equal_nan=True)

pytestmark = pytest.mark.gpu

def test_alternating_lens_calibrations_reuse_cached_rays(renderer):
    from dataclasses import replace
    left=CameraModel(5,5,4,4,2,2)
    right=replace(left,fx=4.2)
    first=renderer._rays(left)
    renderer._rays(right)
    assert renderer._rays(left) is first
