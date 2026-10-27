import pytest
import numpy as np
from stereoforge.types import CameraModel, DepthFrame, Provenance
from stereoforge.temporal import reproject_depth, TemporalHistory
from stereoforge.gpu_temporal import VulkanTemporal


def frame(source=1,timestamp=0):
    z=np.array([[1.,4.,np.nan]],np.float32)
    return DepthFrame(np.full(z.shape,3.,np.float32),z,np.full(z.shape,.8,np.float32),np.isfinite(z),np.full(z.shape,Provenance.STEREO,np.uint8),np.full(z.shape,source,np.int64),np.zeros(z.shape,np.int64),timestamp,'cal')


def equal(a,b):
    for name in ('depth_m','confidence','disparity'):
        np.testing.assert_allclose(getattr(a,name),getattr(b,name),atol=1e-6,equal_nan=True)
    for name in ('valid','provenance','source_ids','age_ns'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))


def test_reprojection_and_fusion_parity():
    gpu=VulkanTemporal();f=frame(); camera=CameraModel(3,1,1,1,0,0)
    pose=np.eye(4);pose[0,3]=1
    equal(gpu.reproject(f,camera,pose,10,fx=1,baseline_m=.2),reproject_depth(f,camera,pose,10,fx=1,baseline_m=.2))
    for source,age,prov in ((1,0,Provenance.STEREO),(2,0,Provenance.STEREO),(2,300000000,Provenance.STEREO),(2,0,Provenance.FUSED)):
        current=frame(timestamp=10);pred=frame(source,timestamp=10)
        pred.depth_m[pred.valid]*=1.01;pred.age_ns[:]=age;pred.provenance[:]=prov
        equal(gpu.fuse(current,pred),TemporalHistory().fuse(current,pred))
    equal(gpu.fuse(frame(),None),frame())


def test_fill_and_fused_warp_preserve_identity():
    gpu=VulkanTemporal();current=frame(timestamp=20);previous=frame(source=2**40+1,timestamp=10)
    current.valid[0,0]=False;current.depth_m[0,0]=np.nan
    previous.provenance[0,0]=Provenance.FUSED
    equal(gpu.fuse(current,previous),TemporalHistory().fuse(current,previous))
    camera=CameraModel(3,1,1,1,0,0)
    equal(gpu.reproject(previous,camera,np.eye(4),20),reproject_depth(previous,camera,np.eye(4),20))

pytestmark = pytest.mark.gpu
