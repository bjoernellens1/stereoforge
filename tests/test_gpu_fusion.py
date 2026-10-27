"""Photometric Vulkan fusion parity with the scalar geometry oracle."""
import inspect
import numpy as np
import pytest
from stereoforge.gpu_geometry import VulkanViewSynthesizer
from stereoforge.fusion import VirtualCamera, synthesize
from test_gpu_geometry import scene


def test_fusion_render_contract():
    parameters=inspect.signature(VulkanViewSynthesizer.render).parameters
    assert parameters['mode'].default=='strict'
    assert parameters['exposure_match'].default is True


@pytest.fixture(scope='module')
def renderer():
    return VulkanViewSynthesizer()


def compare(renderer,frame,depth,camera,right=None,**kwargs):
    expected=synthesize(frame,depth,camera,right,**kwargs)
    actual=renderer.render(frame,depth,camera,right,**kwargs)
    for key in ('rgb_valid','depth_valid','source','provenance','approximate'):
        np.testing.assert_array_equal(getattr(actual,key),getattr(expected,key))
    np.testing.assert_allclose(actual.rgb,expected.rgb,atol=1)
    np.testing.assert_allclose(actual.depth_m,expected.depth_m,atol=2e-5,equal_nan=True)
    np.testing.assert_allclose(actual.confidence,expected.confidence,atol=1e-6)
    return actual


@pytest.mark.gpu
@pytest.mark.parametrize('projection',['perspective','cylindrical'])
@pytest.mark.parametrize('exposure',[True,False])
def test_fusion_surfaces_exposure_and_confidence(renderer,projection,exposure):
    import copy
    frame,depth=scene();right=copy.deepcopy(depth)
    frame.left[:]=[120,90,60];frame.right[:]=[60,70,100]
    right.confidence[:,2:]=.95
    right.depth_m[1,1]=.7;depth.depth_m[3,3]=.5
    depth.valid[0,0]=False;depth.depth_m[0,0]=np.nan
    depth.confidence[0,0]=0;depth.provenance[0,0]=0
    camera=VirtualCamera(5,5,75.,projection=projection)
    before=frame.left.copy(),frame.right.copy(),depth.depth_m.copy()
    first=compare(renderer,frame,depth,camera,right,exposure_match=exposure)
    second=compare(renderer,frame,depth,camera,right,exposure_match=exposure)
    np.testing.assert_array_equal(first.rgb,second.rgb)
    for original,current in zip(before,(frame.left,frame.right,depth.depth_m)):
        np.testing.assert_array_equal(original,current)


@pytest.mark.gpu
@pytest.mark.parametrize('projection',['perspective','cylindrical'])
def test_preview_holes_have_no_geometry_and_ignore_translation(renderer,projection):
    from dataclasses import replace
    frame,depth=scene()
    model=replace(frame.calibration.left,distortion=(.08,-.02,0.,0.,0.))
    frame.calibration=replace(frame.calibration,left=model,right=model)
    depth.calibration_id=frame.calibration.identity
    depth.valid[:]=False;depth.depth_m[:]=np.nan;depth.confidence[:]=0;depth.provenance[:]=0
    pose=np.eye(4);pose[0,3]=9
    camera=VirtualCamera(9,7,100.,projection=projection,camera_to_rig=pose)
    actual=compare(renderer,frame,depth,camera,mode='preview')
    assert actual.approximate.any() and not actual.depth_valid.any()
    assert not actual.confidence.any() and not actual.provenance.any()
    assert np.isnan(actual.depth_m).all()
    strict=compare(renderer,frame,depth,camera,mode='strict')
    assert not strict.rgb_valid.any()


def test_preview_configuration_and_bounded_calibration_cache():
    from stereoforge.pipeline import PipelineConfig
    PipelineConfig(output='wide_rgb',fusion_backend='vulkan',preview=True)
    with pytest.raises(ValueError,match='strict'):
        PipelineConfig(output='wide_rgbd',fusion_backend='vulkan',preview=True)
    renderer=object.__new__(VulkanViewSynthesizer)
    renderer._preview_cache={}
    frame,_=scene()
    first=VirtualCamera(5,5,70.)
    key,mapping=renderer._preview_map(frame.calibration,first)
    pose=np.eye(4);pose[0,3]=100
    translated=VirtualCamera(5,5,70.,camera_to_rig=pose)
    assert renderer._preview_map(frame.calibration,translated)[0]==key
    assert renderer._preview_map(frame.calibration,translated)[1] is mapping
    renderer._preview_map(frame.calibration,VirtualCamera(5,5,80.))
    renderer._preview_map(frame.calibration,VirtualCamera(5,5,90.))
    assert len(renderer._preview_cache)==2
    assert key not in renderer._preview_cache


@pytest.mark.gpu
def test_even_median_and_zero_confidence_blend(renderer):
    import copy
    frame,depth=scene();right=copy.deepcopy(depth)
    frame.left[:]=[100,0,255];frame.right[:]=[100,0,255]
    # Four nonsaturated ratios, whose even median must average middle two.
    frame.left[0,:4,0]=[50,80,140,200]
    frame.left[1:,:,0]=0;frame.right[1:,:,0]=0
    frame.left[0,4,0]=0;frame.right[0,4,0]=0
    depth.confidence[:]=0;right.confidence[:]=0
    camera=VirtualCamera(5,5,float(np.degrees(2*np.arctan(5/8))))
    result=compare(renderer,frame,depth,camera,right)
    assert (result.source[result.depth_valid]==3).all()


@pytest.mark.gpu
def test_even_median_clamps_after_averaging_order_statistics(renderer):
    import copy
    frame,depth=scene();right=copy.deepcopy(depth)
    frame.left[:]=0;frame.right[:]=0
    frame.left[0,0]=9;frame.right[0,0]=90
    frame.left[0,1]=240;frame.right[0,1]=80
    # Median([.1,3.])=1.55, while median(clamp([.1,3.]))=1.25.
    camera=VirtualCamera(5,5,float(np.degrees(2*np.arctan(5/8))))
    compare(renderer,frame,depth,camera,right)


@pytest.mark.gpu
def test_translated_distorted_view_preserves_disocclusion_holes(renderer):
    from dataclasses import replace
    import copy
    frame,depth=scene()
    model=replace(frame.calibration.left,distortion=(.08,-.02,.002,-.001,0.))
    frame.calibration=replace(frame.calibration,left=model,right=model)
    depth.calibration_id=frame.calibration.identity
    right=copy.deepcopy(depth)
    depth.depth_m[:,2]=.5;right.depth_m[:,2]=.5
    frame.left[:,2]=[20,180,30];frame.right[:,2]=[20,180,30]
    pose=np.eye(4);pose[0,3]=.17;pose[1,3]=.03
    camera=VirtualCamera(11,7,95.,camera_to_rig=pose)
    actual=compare(renderer,frame,depth,camera,right,exposure_match=False)
    assert actual.depth_valid.any() and (~actual.depth_valid).any()
    assert not actual.approximate.any()


def test_packed_readback_layout_decodes_independent_validity_and_nan():
    packed=np.zeros((1,2,4),np.uint32)
    packed[0,:,0]=[10|(20<<8)|(30<<16),40|(50<<8)|(60<<16)]
    packed[0,:,1]=np.array([2.,np.nan],np.float32).view(np.uint32)
    packed[0,:,2]=np.array([.8,0.],np.float32).view(np.uint32)
    packed[0,:,3]=[1|2|(3<<8)|(1<<16),2|4|(2<<8)]
    decoded=VulkanViewSynthesizer._decode_readback(packed)
    np.testing.assert_array_equal(decoded['rgb'],[[[10,20,30],[40,50,60]]])
    np.testing.assert_array_equal(decoded['depth_valid'],[[True,False]])
    np.testing.assert_array_equal(decoded['rgb_valid'],[[True,True]])
    np.testing.assert_array_equal(decoded['approximate'],[[False,True]])
    np.testing.assert_array_equal(decoded['source'],[[3,2]])
    np.testing.assert_array_equal(decoded['provenance'],[[1,0]])
    np.testing.assert_allclose(decoded['depth_m'],[[2.,np.nan]],equal_nan=True)
    np.testing.assert_allclose(decoded['confidence'],[[.8,0.]])


def test_injected_device_must_be_vulkan():
    from types import SimpleNamespace
    s=pytest.importorskip("slangpy")
    fake=SimpleNamespace(info=SimpleNamespace(type=s.DeviceType.cpu))
    with pytest.raises(ValueError,match='Vulkan'):
        VulkanViewSynthesizer(device=fake)


@pytest.mark.gpu
def test_documented_float32_half_pixel_boundary(renderer):
    """Nearest-pixel support can differ from the float64 oracle at a half edge.

    This fixture asserts each precision's result explicitly. Ordinary parity
    fixtures retain their exact mask/source checks and one-intensity RGB bound.
    """
    import copy
    frame,depth=scene()
    depth.valid[:]=False;depth.depth_m[:]=np.nan
    depth.confidence[:]=0;depth.provenance[:]=0
    depth.valid[2,2]=True;depth.depth_m[2,2]=2.
    depth.confidence[2,2]=.8;depth.provenance[2,2]=1
    right=copy.deepcopy(depth)
    right.valid[:]=False;right.depth_m[:]=np.nan
    right.confidence[:]=0;right.provenance[:]=0
    pose=np.eye(4);pose[0,3]=-.249999995
    camera=VirtualCamera(5,5,float(np.degrees(2*np.arctan(5/8))),camera_to_rig=pose)
    actual=renderer.render(frame,depth,camera,right,exposure_match=False)
    expected=synthesize(frame,depth,camera,right,exposure_match=False)
    u64=camera.focal*(-pose[0,3])/2+2
    u32=np.float32(camera.focal)*np.float32(-pose[0,3])/np.float32(2)+np.float32(2)
    assert u64<2.5 and u32==2.5 and abs(float(u32)-u64)<1e-4
    np.testing.assert_array_equal(np.argwhere(expected.depth_valid),[[2,2]])
    np.testing.assert_array_equal(np.argwhere(actual.depth_valid),[[2,3]])
    assert actual.source[2,3]==1 and actual.depth_m[2,3]==2.
    assert not actual.approximate.any()
