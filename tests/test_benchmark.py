import numpy as np
import pytest
from stereoforge.types import CameraModel, Calibration, StereoFrame
from stereoforge.benchmark import run_ablations


def frames():
    rng=np.random.default_rng(4); a=rng.integers(0,256,(12,24,3),dtype=np.uint8)
    b=np.zeros_like(a); b[:,:-2]=a[:,2:]
    camera=CameraModel(24,12,20.,20.,11.5,5.5)
    pose=np.eye(4); pose[0,3]=.12
    cal=Calibration(camera,camera,pose)
    return [StereoFrame(a.copy(),b.copy(),cal,100+i*10,i) for i in range(2)]


def test_four_ablations_same_input_and_full_refresh():
    inputs=frames(); before=inputs[0].left.copy()
    truth={i:np.full((12,24),1.2) for i in range(2)}
    result=run_ablations(inputs,max_disparity=8,motions={1:np.eye(4)},ground_truth=truth)
    assert len(result['ablations'])==4
    hashes={v['input_identity'] for v in result['ablations'].values()}
    assert hashes=={result['input_identity']}
    for mode in result['ablations'].values():
        assert mode['config']['output']=='rectified_rgb'
        assert mode['metric_ground_truth_available']
        assert mode['frames'][0]['candidate_evaluations']==12*24*8
        assert mode['frames'][0]['aggregation_candidates']==2*12*24*8*4
        assert mode['depth_mae_m'] is not None and mode['depth_mae_m']<.15
        assert mode['depth_rmse_m'] is not None
        assert mode['processing_ms']['p95']>=mode['processing_ms']['p50']>=0
    np.testing.assert_array_equal(inputs[0].left,before)
    again=run_ablations(inputs,max_disparity=8,motions={1:np.eye(4)})
    assert again['input_identity']==result['input_identity']


def test_missing_motion_and_ground_truth():
    with pytest.raises(ValueError,match='motion'): run_ablations(frames(),max_disparity=8)
    with pytest.raises(ValueError,match='motion'): run_ablations(frames(),max_disparity=8,motions={})
    result=run_ablations(frames()[:1],max_disparity=8)
    for mode in result['ablations'].values():
        assert not mode['metric_ground_truth_available']
        assert mode['depth_mae_m'] is None and mode['depth_rmse_m'] is None


def test_input_order_and_truth_shape():
    with pytest.raises(ValueError): run_ablations([],max_disparity=8)
    with pytest.raises(ValueError): run_ablations(frames()[::-1],max_disparity=8,motions={0:np.eye(4)})
    with pytest.raises(ValueError): run_ablations(frames()[:1],max_disparity=8,ground_truth={0:np.zeros((1,1))})


def test_identity_changes_with_rgb_bytes():
    first=frames()[:1]; second=frames()[:1]
    second[0].left[0,0,0]^=1
    a=run_ablations(first,max_disparity=8)
    b=run_ablations(second,max_disparity=8)
    assert a['input_identity']!=b['input_identity']


def test_fixed_support_and_common_accuracy(monkeypatch):
    from types import SimpleNamespace
    from stereoforge import benchmark
    from stereoforge.temporal import TemporalHistory
    # Explicit metric boundary fixture: dropping the difficult pixel must change
    # support completeness, while common accuracy always uses identical pixels.
    class FixturePipeline:
        def __init__(self,config):
            self.config=config; self.history=TemporalHistory()
        def process(self,frame,previous_to_current=None):
            shape=(12,24); z=np.full(shape,np.nan); valid=np.zeros(shape,bool)
            valid[0,0]=True; z[0,0]=2.
            if not self.config.temporal:
                valid[0,1]=True; z[0,1]=5.
            return SimpleNamespace(depth=SimpleNamespace(depth_m=z,valid=valid),metrics={
                'candidate_evaluations':1,'right_candidate_evaluations':1,'aggregation_candidates':1})
    monkeypatch.setattr(benchmark,'Pipeline',FixturePipeline)
    gt=np.full((12,24),np.nan); gt[0,:3]=[1.,1.,0.]
    out=run_ablations(frames()[:1],max_disparity=8,ground_truth={0:gt})
    for name,mode in out['ablations'].items():
        assert mode['ground_truth_support_count']==2
        assert mode['common_metric_samples']==1
        assert mode['common_depth_mae_m']==1.
        assert mode['common_depth_rmse_m']==1.
        assert mode['frames'][0]['common_metric_samples']==1
        expected=.5 if name in ('temporal','adaptive_temporal') else 1.
        assert mode['completeness_on_ground_truth_support']==expected
        assert mode['frames'][0]['completeness_on_ground_truth_support']==expected
    no_gt=run_ablations(frames()[:1],max_disparity=8)
    for mode in no_gt['ablations'].values():
        assert mode['ground_truth_support_count']==mode['common_metric_samples']==0
        assert mode['completeness_on_ground_truth_support'] is None
        assert mode['common_depth_mae_m'] is mode['common_depth_rmse_m'] is None


def test_identity_binds_camera_controls():
    from stereoforge.benchmark import _identity
    inputs=frames()[:1]
    before=_identity(inputs)
    inputs[0].camera_controls={'requested':{'exposure':100},'readback':{'exposure':101}}
    assert _identity(inputs)!=before
    same=_identity(inputs)
    inputs[0].camera_controls['readback']['exposure']=102
    assert _identity(inputs)!=same
