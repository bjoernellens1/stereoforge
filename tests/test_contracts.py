import numpy as np
import pytest
from stereoforge.types import CameraModel, Calibration, StereoFrame, DepthFrame, Provenance

def test_calibration_rejects_zero_baseline():
    c = CameraModel(4, 3, 4., 4., 1.5, 1.)
    with pytest.raises(ValueError, match="baseline"):
        Calibration(c, c, np.eye(4))

def test_depth_uses_rectified_principal_point_offset_and_invalidates_zero():
    from stereoforge.reference import depth_from_disparity
    d = np.array([[4., 0., -1., np.nan]], np.float32)
    z, valid = depth_from_disparity(d, 100., .12, 1.)
    assert z[0, 0] == pytest.approx(4.)
    assert valid.tolist() == [[True, False, False, False]]
    assert np.isnan(z[~valid]).all()

def test_frame_rejects_wrong_dimensions():
    c = CameraModel(4, 3, 4., 4., 1.5, 1.)
    t = np.eye(4); t[0, 3] = .12
    with pytest.raises(ValueError, match="shape"):
        StereoFrame(np.zeros((3, 3, 3), np.uint8), np.zeros((3, 4, 3), np.uint8), Calibration(c,c,t), 1, 0)

@pytest.mark.parametrize('change',[{'calibration_id':''},{'timestamp_ns':-1},{'source_ids':np.array([[-1]],np.int64)},{'provenance':np.array([[0]],np.uint8)}])
def test_valid_geometry_requires_calibration_and_observation_identity(change):
    fields=dict(disparity=np.array([[4]],np.float32),depth_m=np.array([[3]],np.float32),confidence=np.array([[.8]],np.float32),valid=np.array([[True]]),provenance=np.array([[1]],np.uint8),source_ids=np.array([[0]],np.int64),age_ns=np.array([[0]],np.int64),timestamp_ns=0,calibration_id='cal')
    fields.update(change)
    with pytest.raises(ValueError):DepthFrame(**fields)
