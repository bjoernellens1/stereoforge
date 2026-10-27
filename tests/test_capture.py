import numpy as np
import pytest
from stereoforge.types import Calibration, CameraModel
from stereoforge.capture import yuyv_to_stereo, ZedCamera, discover_cameras


def calibration():
    camera=CameraModel(4,2,3.,3.,1.5,.5)
    transform=np.eye(4); transform[0,3]=.12
    return Calibration(camera,camera,transform,serial='123')


def test_yuyv_conversion_preserves_split_timestamp_and_samples():
    # Constant chroma and alternating luma distinguish original lens images.
    raw=np.zeros((2,8,2),np.uint8); raw[:,:,1]=128
    raw[:,:4,0]=16; raw[:,4:,0]=235
    original=raw.copy(); samples=[{'timestamp_ns':7,'sync':True}]
    frame=yuyv_to_stereo(raw,calibration(),10,5,samples)
    assert frame.left.shape==(2,4,3) and frame.right.shape==(2,4,3)
    assert np.max(frame.left)==0 and np.min(frame.right)==255
    assert frame.timestamp_ns==10 and frame.frame_id==5
    assert frame.clock_domain=='zed_open_capture'
    assert frame.sensor_samples==samples
    np.testing.assert_array_equal(raw,original)
    assert not np.shares_memory(frame.left,raw)


@pytest.mark.parametrize('raw',[np.zeros((2,7,2),np.uint8),np.zeros((2,8,3),np.uint8),np.zeros((2,8,2),np.float32)])
def test_wrong_yuyv_shape_or_dtype_rejected(raw):
    with pytest.raises(ValueError): yuyv_to_stereo(raw,calibration(),10,1,[])


def test_camera_closes_native_on_calibration_failure(monkeypatch):
    import stereoforge.capture as capture
    state={'closed':False}
    class Native:
        serial=123
        def __init__(self,**kwargs): pass
        def close(self): state['closed']=True
    class Module: ZedCapture=Native
    monkeypatch.setattr(capture,'_native_module',lambda:Module)
    monkeypatch.setattr(capture,'download_calibration',lambda *args: (_ for _ in ()).throw(ValueError('unavailable calibration')))
    with pytest.raises(ValueError,match='calibration'): ZedCamera()
    assert state['closed']


def test_discovery_reports_real_system_nodes():
    result=discover_cameras()
    assert isinstance(result,list)
    for node in result:
        assert node['device'].startswith('/dev/video')
        assert isinstance(node['name'],str)


def test_camera_refuses_wrong_serial(monkeypatch):
    import stereoforge.capture as capture
    class Native:
        serial=999
        def __init__(self,**kwargs): pass
        def close(self): pass
    class Module: ZedCapture=Native
    monkeypatch.setattr(capture,'_native_module',lambda:Module)
    with pytest.raises(ValueError,match='serial'): ZedCamera(calibration(),width=4,height=2)


def test_missing_native_extension_has_actionable_error(monkeypatch,tmp_path):
    import stereoforge.capture as capture
    monkeypatch.setenv('STEREOFORGE_NATIVE_DIR',str(tmp_path))
    with pytest.raises(RuntimeError,match='STEREOFORGE_CAPTURE=ON'): capture._native_module()


def test_native_adapter_read_and_context_close(monkeypatch):
    import stereoforge.capture as capture
    state={'closed':False}
    raw=np.zeros((2,8,2),np.uint8); raw[:,:,1]=128
    class Native:
        serial=123
        sensors_available=False
        sensor_status='sensor_open_failed_check_hidraw_permissions'
        sensor_dropped_count=0
        def __init__(self,**kwargs): pass
        def read(self,timeout_ms):
            return {'yuyv':raw,'timestamp_ns':10,'frame_id':9,'sensor_samples':[]}
        def close(self): state['closed']=True
    class Module: ZedCapture=Native
    monkeypatch.setattr(capture,'_native_module',lambda:Module)
    with ZedCamera(calibration(),width=4,height=2) as camera:
        frame=camera.read()
        assert frame.frame_id==9
        assert not camera.sensors_available
        assert 'sensor_open_failed' in camera.sensor_status
    assert state['closed']
    assert camera.sensor_status=='closed'
    with pytest.raises(RuntimeError,match='closed'): camera.read()
