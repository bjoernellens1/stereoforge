"""Offline native control validation. This suite never opens a camera."""
import pytest


@pytest.fixture(scope='module')
def native():
    from stereoforge.capture import _native_module
    try:
        return _native_module()
    except RuntimeError as error:
        pytest.skip(str(error))


def test_control_api_is_available_without_opening_camera(native):
    assert callable(native.validate_controls)
    assert hasattr(native.ZedCapture, 'get_controls')
    assert hasattr(native.ZedCapture, 'set_controls')


@pytest.mark.parametrize('settings', [
    {}, {'brightness':0, 'contrast':8, 'sharpness':8, 'saturation':0, 'hue':11, 'gamma':1},
    {'exposure_left':0,'exposure_right':100,'gain_left':100,'gain_right':0,'auto_exposure_gain':False},
    {'white_balance':2800,'auto_white_balance':False}, {'white_balance':6500},
    {'auto_exposure_gain':True,'auto_white_balance':True,'led':False},
    {'roi_left':[0,0,1280,720],'roi_right':None}, {'roi_left':(1269,710,11,10)},
])
def test_valid_native_requests(native, settings):
    assert native.validate_controls(1280,720,settings) is None


@pytest.mark.parametrize('settings', [
    {'unknown':1}, {'read_errors':[]}, {'write_errors':[]}, {1:2},
    {'brightness':True}, {'brightness':1.0}, {'brightness':-1}, {'contrast':9},
    {'sharpness':9}, {'saturation':9}, {'hue':12}, {'gamma':0}, {'gamma':10},
    {'exposure_left':101}, {'exposure_right':-1}, {'gain_left':101}, {'gain_right':-1},
    {'white_balance':2799}, {'white_balance':6501}, {'auto_exposure_gain':1},
    {'auto_white_balance':0}, {'led':1}, {'led':None}, {'brightness':2**100},
    {'roi_left':[]}, {'roi_left':[0,0,1,1]}, {'roi_left':[0,0,10,10]}, {'roi_left':[0,0,1281,720]}, {'roi_right':[0,0,1280,721]},
    {'roi_left':[-1,0,5,5]}, {'roi_left':[0,0,0,1]}, {'roi_left':[0,0,1,False]},
    {'roi_left':[0,0,1,1.0]}, {'roi_right':'0,0,20,20'},
    {'exposure_left':5,'auto_exposure_gain':True}, {'gain_right':5,'auto_exposure_gain':True},
    {'white_balance':4000,'auto_white_balance':True},
])
def test_invalid_native_requests_fail_before_io(native,settings):
    with pytest.raises((ValueError,TypeError,OverflowError)):
        native.validate_controls(1280,720,settings)


@pytest.mark.parametrize('width,height',[(0,720),(1280,0),(-1,720),(65536,720)])
def test_invalid_validation_dimensions(native,width,height):
    with pytest.raises(ValueError): native.validate_controls(width,height,{})


def test_qualifier_restores_manual_then_auto_after_write_exception():
    from benchmarks.qualify_controls import CONTROL_KEYS, qualify_controls
    baseline = {key: 4 for key in CONTROL_KEYS}
    baseline.update(auto_exposure_gain=True,auto_white_balance=True,led=True,
        exposure_left=20,exposure_right=21,gain_left=10,gain_right=11,
        white_balance=4000,gamma=4,roi_left=[0,0,1280,720],roi_right=[0,0,1280,720],
        read_errors=[],write_errors=[])
    class Camera:
        def __init__(self): self.values=baseline.copy();self.calls=[]
        def get_controls(self): return self.values.copy()
        def set_controls(self,settings):
            self.calls.append(settings.copy())
            if settings.get('brightness')==999: raise ValueError('invalid')
            self.values.update(settings)
            if 'auto_exposure_gain' in settings and settings['auto_exposure_gain'] is False:
                raise RuntimeError('partial write')
            return self.values.copy()
    camera=Camera()
    report=qualify_controls(camera,True)
    assert report['status']=='exercise_error'
    assert report['restore_errors']==[] and report['restore_mismatches']=={}
    assert set(camera.calls[-1])=={'auto_exposure_gain','auto_white_balance'}
    assert 'exposure_left' in camera.calls[-2] and 'auto_exposure_gain' not in camera.calls[-2]
    assert camera.values==baseline
    assert all('led' not in request for request in camera.calls)


def test_qualifier_read_only_does_not_write():
    from benchmarks.qualify_controls import qualify_controls
    class Camera:
        def get_controls(self): return {'read_errors':[]}
        def set_controls(self,settings): raise AssertionError('read-only qualifier wrote controls')
    assert qualify_controls(Camera())['status']=='readback_observed'


def test_native_snapshot_avoids_destructive_led_getter():
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/'src/zed_capture.cpp').read_text()
    body=source.split('ControlSnapshot controls_locked()',1)[1].split('void sensor_loop()',1)[0]
    assert 'getLEDstatus' not in body
    assert 'readback_limitations' in source
