import json
import numpy as np
import pytest
from stereoforge.types import StereoFrame
from stereoforge.calibration import Rectifier
from stereoforge.recording import Recorder, Replay
from test_recording import frame, calibration, edit


def snapshot():
    return {'profile':'manual','revision':2,'requested':{'exposure':100},
        'readback':{'exposure':100},'read_errors':{},'write_errors':{},
        'matches':{'exposure':True},'applied':True,'sampled_at_host_ns':90,
        'clock_domain':'host_monotonic',
        'frame_association':'host_snapshot_not_exposure_synchronous'}


def test_nested_controls_roundtrip_and_snapshot(tmp_path):
    f=frame();f.camera_controls=snapshot()
    before=json.dumps(f.camera_controls,sort_keys=True,separators=(',',':'))
    with Recorder(tmp_path/'record',calibration(),{}) as out:
        out.write(f)
        f.camera_controls['requested']['exposure']=900
    restored=next(iter(Replay(tmp_path/'record')))
    assert json.dumps(restored.camera_controls,sort_keys=True,separators=(',',':'))==before
    np.testing.assert_array_equal(restored.left,f.left)
    np.testing.assert_array_equal(restored.right,f.right)


def test_old_schema_and_invalid_controls(tmp_path):
    with Recorder(tmp_path/'record',calibration(),{}) as out:out.write(frame())
    edit(tmp_path/'record',lambda d:d['frames'][0].pop('camera_controls',None))
    assert next(iter(Replay(tmp_path/'record'))).camera_controls=={}
    edit(tmp_path/'record',lambda d:d['frames'][0].update(camera_controls=[]))
    with pytest.raises(ValueError):Replay(tmp_path/'record')
    f=frame();f.camera_controls={'bad':float('nan')}
    with Recorder(tmp_path/'bad',calibration(),{}) as out:
        with pytest.raises(ValueError):out.write(f)
    with pytest.raises(ValueError):
        StereoFrame(f.left,f.right,f.calibration,1,1,camera_controls=[])


def test_rectification_copies_nested_metadata():
    f=frame();f.camera_controls=snapshot()
    rectified=Rectifier(f.calibration).rectify(f)
    assert rectified.camera_controls==f.camera_controls
    rectified.camera_controls['readback']['exposure']=900
    assert f.camera_controls['readback']['exposure']==100
