import numpy as np
import pytest
from stereoforge.pipeline import Pipeline, PipelineConfig
from stereoforge.types import CameraModel, Calibration, StereoFrame

def pair(i=0):
    rng=np.random.default_rng(42)
    left=rng.integers(0,256,(24,64,3),dtype=np.uint8)
    right=np.zeros_like(left);right[:,:-4]=left[:,4:]
    c=CameraModel(64,24,60.,60.,31.5,11.5);t=np.eye(4);t[0,3]=.12
    return StereoFrame(left,right,Calibration(c,c,t),i*16000000+1,i)

def test_cpu_pipeline_metric_depth_and_wide_masks():
    p=Pipeline(PipelineConfig(backend='cpu_census',max_disparity=8,subpixel=False,output='wide_rgbd'))
    out=p.process(pair())
    assert np.nanmedian(out.depth.depth_m[5:-5,12:-8])==pytest.approx(1.8)
    assert out.view.depth_convention=='axial_z'
    assert not out.view.approximate.any()
    assert np.all(out.view.depth_valid<=out.view.rgb_valid)

def test_temporal_requires_explicit_motion_and_reports_work():
    p=Pipeline(PipelineConfig(backend='cpu_census',max_disparity=8,temporal=True,adaptive=True))
    p.process(pair())
    out=p.process(pair(1))
    assert out.metrics['motion_available'] is False
    assert out.metrics['candidate_evaluations']==24*64*8
    out=p.process(pair(2),previous_to_current=np.eye(4))
    assert out.metrics['motion_available'] is True

def test_config_rejects_vendor_compute_backends():
    for backend in ('cuda','rocm','hip','automatic'):
        with pytest.raises(ValueError):PipelineConfig(backend=backend)

def test_consumer_cannot_mutate_adaptive_history():
    p=Pipeline(PipelineConfig(backend='cpu_census',max_disparity=8,adaptive=True))
    result=p.process(pair())
    cached=p.history.frames[-1].valid.copy()
    result.depth.valid[:]=False
    np.testing.assert_array_equal(p.history.frames[-1].valid,cached)

def test_stabilization_rejects_stale_or_unbound_pose():
    from stereoforge.types import TimedPose
    p=Pipeline(PipelineConfig(backend='cpu_census',max_disparity=8,output='stabilized_wide_rgb'))
    with pytest.raises(ValueError,match='timestamp'):
        p.process(pair(),rig_to_world=TimedPose(np.eye(4),0,'host_monotonic'))
    with pytest.raises(ValueError,match='timestamped'):
        p.process(pair(),rig_to_world=np.eye(4))
