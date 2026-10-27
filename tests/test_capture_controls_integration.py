from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
from test_capture import calibration
from stereoforge.capture import ZedCamera


def native_module(state):
    class Native:
        serial=123
        sensors_available=False
        sensor_status='unavailable'
        sensor_dropped_count=0
        def __init__(self,**kwargs):state['opens']+=1
        def get_controls(self):
            state['queries']+=1
            return deepcopy(state['values'])
        def set_controls(self,settings):
            state['values'].update(settings)
            return deepcopy(state['values'])
        def read(self,timeout_ms):
            state['frames']+=1
            yuyv=np.zeros((2,8,2),np.uint8);yuyv[:,:,1]=128
            return dict(yuyv=yuyv,timestamp_ns=state['frames'],frame_id=state['frames'],sensor_samples=[])
        def close(self):state['closes']+=1
    return SimpleNamespace(ZedCapture=Native)

@pytest.fixture
def state(monkeypatch):
    import stereoforge.capture as capture
    state=dict(opens=0,closes=0,queries=0,frames=0,values={'auto_exposure_gain':True,'auto_white_balance':True})
    monkeypatch.setattr(capture,'_native_module',lambda:native_module(state))
    return state

def test_frame_attaches_cached_controls_without_per_frame_usb_poll(state):
    with ZedCamera(calibration(),width=4,height=2) as camera:
        assert state['queries']==1
        report=camera.camera_control.apply_profile('stereo_optimized',exposure=20,gain=30,white_balance=4500)
        assert report['applied']
        first=camera.read(); second=camera.read()
        assert state['queries']==1
        assert first.camera_controls['requested']['exposure_left']==20
        assert first.camera_controls['requested']['exposure_right']==20
        assert first.camera_controls['frame_association']=='host_snapshot_not_exposure_synchronous'
        first.camera_controls['readback']['exposure_left']=99
        assert second.camera_controls['readback']['exposure_left']==20
    with pytest.raises(RuntimeError,match='closed'):camera.camera_control

def test_invalid_cli_controls_rejected_before_camera_open(state,tmp_path):
    from stereoforge.cli import _camera
    f=tmp_path/'controls.json';f.write_text(json.dumps({'profile':'motion_optimized','exposure':101,'gain':20,'white_balance':4500}))
    args=SimpleNamespace(width=4,height=2,fps=60,device_id=-1,cache=None,calibration=None,camera_controls=str(f))
    with pytest.raises(ValueError):_camera(args)
    assert state['opens']==0

def test_camera_closes_when_requested_control_write_cannot_be_verified(state,tmp_path):
    from stereoforge.cli import _camera
    f=tmp_path/'controls.json';f.write_text(json.dumps({'profile':'automatic'}))
    cal=tmp_path/'cal.json';cal.write_text(json.dumps(calibration().to_dict()))
    state['values']['read_errors']=['device error']
    args=SimpleNamespace(width=4,height=2,fps=60,device_id=-1,cache=None,calibration=str(cal),camera_controls=str(f))
    with pytest.raises(RuntimeError,match='readback'):_camera(args)
    assert state['opens']==1 and state['closes']==1


@pytest.mark.parametrize('command,kind',[('controls','file'),('controls','directory'),('record','file'),('record','nonempty')])
def test_bad_output_rejected_before_camera(monkeypatch,tmp_path,command,kind):
    from stereoforge import cli
    output=tmp_path/'occupied'
    if kind=='file':output.write_text('preserve')
    else:
        output.mkdir()
        if kind=='nonempty':(output/'keep').write_text('preserve')
    calls=[]
    def forbidden(args):
        calls.append(args);raise RuntimeError('camera opened')
    monkeypatch.setattr(cli,'_camera',forbidden)
    with pytest.raises(SystemExit) as error:cli.main([command,'--output',str(output)])
    assert error.value.code==2 and calls==[]
    if kind=='file':assert output.read_text()=='preserve'


def test_controls_creates_parent_and_cleans_failed_reservation(monkeypatch,tmp_path):
    from stereoforge import cli
    class Camera:
        control_snapshot={'readback':{'gain_left':20}}
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(cli,'_camera',lambda args:Camera())
    output=tmp_path/'nested'/'controls.json'
    cli.main(['controls','--output',str(output)])
    assert json.loads(output.read_text())==Camera.control_snapshot
    def failed(args):raise RuntimeError('camera failure')
    monkeypatch.setattr(cli,'_camera',failed)
    missing=tmp_path/'another'/'controls.json'
    with pytest.raises(SystemExit):cli.main(['controls','--output',str(missing)])
    assert not missing.exists()
