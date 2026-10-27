import pytest
from stereoforge.controls import CameraControl


class Camera:
    def __init__(self):
        self.state={'exposure_left':10,'gain_left':20,'white_balance':4000,'auto_exposure_gain':True,
                    'auto_white_balance':True,'roi_left':[0,0,640,480]}
        self.reads=0;self.writes=0;self.closed=False;self.ignore=False
    def get_controls(self):
        self.reads+=1
        if self.closed:raise RuntimeError('closed')
        return {'read_errors':[],'write_errors':[],**self.state}
    def set_controls(self,settings):
        self.writes+=1
        if self.closed:raise RuntimeError('closed')
        if not self.ignore:self.state.update({k:([0,0,640,480] if k in ('roi_left','roi_right') and v is None else v) for k,v in settings.items()})
        return self.get_controls()


def test_read_set_and_independent_snapshot():
    native=Camera(); control=CameraControl(native,640,480)
    assert native.reads==0 and control.snapshot['applied'] is None
    read=control.read();assert read['readback']['exposure_left']==10 and read['applied'] is None
    result=control.set({'auto_exposure_gain':False,'exposure_left':30,'roi_left':None})
    assert result['applied'] and result['revision']==1 and result['matches']['roi_left']
    assert result['requested']['roi_left'] is None
    reads=native.reads; snapshot=control.snapshot;snapshot['readback']['roi_left'][0]=99
    assert control.snapshot['readback']['roi_left'][0]==0 and native.reads==reads
    assert result['frame_association']=='host_snapshot_not_exposure_synchronous'


@pytest.mark.parametrize('settings',[
    {'unknown':1},{'exposure_left':True},{'gain_left':101},{'white_balance':2799},{'gamma':0},
    {'brightness':9},{'hue':12},{'auto_white_balance':1},{'led':'on'},
    {'roi_left':[0,0,641,480]},{'roi_left':[0.,0,640,480]},{'roi_left':[0,0,0,480]},
    {'auto_exposure_gain':True,'exposure_left':20},{'auto_white_balance':True,'white_balance':4000},
])
def test_validation_before_native_write(settings):
    native=Camera();control=CameraControl(native,640,480)
    with pytest.raises(ValueError):control.set(settings)
    assert native.writes==native.reads==0


def test_profiles():
    native=Camera();control=CameraControl(native,640,480)
    for profile in ('stereo_optimized','motion_optimized'):
        with pytest.raises(ValueError):control.apply_profile(profile)
        report=control.apply_profile(profile,exposure=25,gain=30,white_balance=4500)
        assert report['profile']==profile and report['applied']
        assert report['requested']=={'auto_exposure_gain':False,'auto_white_balance':False,
            'exposure_left':25,'exposure_right':25,'gain_left':30,'gain_right':30,'white_balance':4500,'sharpness':0}
    assert control.apply_profile('automatic')['requested']=={'auto_exposure_gain':True,'auto_white_balance':True}
    with pytest.raises(ValueError):control.apply_profile('automatic',gain=2)
    with pytest.raises(ValueError):control.apply_profile('stereo-optimized')


def test_mismatch_unverified_and_errors():
    native=Camera();control=CameraControl(native,640,480);native.ignore=True
    report=control.set({'exposure_left':30});assert not report['applied'] and report['matches']['exposure_left'] is False
    native.state['exposure_left']=None
    report=control.set({'exposure_left':30});assert report['matches']['exposure_left'] is None and report['applied'] is None
    native.state['read_errors']=['usb read failed']
    assert control.set({'gain_left':20})['applied'] is False


def test_closed_does_not_assume_state():
    native=Camera();control=CameraControl(native,640,480);control.read();native.closed=True
    with pytest.raises(RuntimeError):control.set({'gain_left':50})
    assert control.snapshot['revision']==1 and control.snapshot['applied'] is False
    assert control.snapshot['readback']=={}


def test_write_error_and_exception_refresh_real_state():
    native=Camera();control=CameraControl(native,640,480)
    native.state['write_errors']=['one lens failed']
    result=control.set({'gain_left':30})
    assert result['readback']['gain_left']==30 and result['applied'] is False
    assert result['write_errors']==['one lens failed']
    assert 'write_errors' not in result['readback']
    native.state.pop('write_errors')
    def partial(settings):
        native.state['gain_left']=17
        raise RuntimeError('partial write')
    native.set_controls=partial
    with pytest.raises(RuntimeError,match='partial write'):control.set({'gain_left':40})
    snapshot=control.snapshot
    assert snapshot['revision']==2 and snapshot['requested']['gain_left']==40
    assert snapshot['readback']['gain_left']==17 and snapshot['applied'] is False


def test_normalized_boundaries_and_read_metadata():
    native=Camera();control=CameraControl(native,640,480)
    result=control.set({'brightness':0,'contrast':8,'sharpness':0,'saturation':8,
        'hue':11,'gamma':9,'exposure_left':0,'gain_left':100,'white_balance':6500,
        'auto_exposure_gain':False,'auto_white_balance':False,'led':True,
        'roi_left':(1,2,639,478)})
    assert result['applied'] is True and all(result['matches'].values())
    assert result['requested']['roi_left']==[1,2,639,478]
    report=control.read()
    assert report['applied'] is None and report['revision']==1
    assert report['sampled_at_host_ns']>=result['sampled_at_host_ns']


def test_native_keys_and_pure_validation():
    from stereoforge.controls import validate_controls, profile_settings
    for key in ('exposure','gain','roi'):
        with pytest.raises(ValueError):validate_controls({key:10},640,480)
    with pytest.raises(ValueError):validate_controls({'roi_right':[0,0,10,10]},640,480)
    assert validate_controls({'roi_right':[0,0,101,1]},640,480)=={'roi_right':[0,0,101,1]}
    settings=profile_settings('motion_optimized',exposure=25,gain=30,white_balance=4500)
    assert settings['exposure_left']==settings['exposure_right']==25
    assert settings['gain_left']==settings['gain_right']==30
    native=Camera();control=CameraControl(native,640,480)
    report=control.set({'exposure_left':10,'exposure_right':20,'gain_left':30,'gain_right':40})
    assert report['applied'] and all(report['matches'].values())
    assert report['readback']['exposure_right']==20 and report['readback']['gain_right']==40


def test_concurrent_writes_keep_requested_and_readback_associated():
    from concurrent.futures import ThreadPoolExecutor
    import time
    class SlowCamera(Camera):
        def set_controls(self,settings):
            self.writes+=1
            self.state.update(settings)
            time.sleep(.001)  # Model USB wait releasing the GIL after mutation.
            return self.get_controls()
    camera=SlowCamera();control=CameraControl(camera,640,480)
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports=list(pool.map(lambda gain:control.set({'gain_left':gain}),range(20)))
    assert sorted(r['revision'] for r in reports)==list(range(1,21))
    for report in reports:
        assert report['requested']['gain_left']==report['readback']['gain_left']
        assert report['applied'] is True
    assert control.revision==20


def test_right_roi_reset():
    native=Camera();control=CameraControl(native,640,480)
    report=control.set({'roi_right':None})
    assert report['matches']['roi_right'] is True


def test_direct_override_clears_profile_and_invalid_profile_preserves_intent():
    camera=Camera();control=CameraControl(camera,640,480)
    control.apply_profile('stereo_optimized',exposure=25,gain=30,white_balance=4500)
    with pytest.raises(ValueError):control.apply_profile('motion_optimized',exposure=101,gain=30,white_balance=4500)
    assert control.snapshot['profile']=='stereo_optimized' and control.revision==1
    report=control.set({'exposure_right':40})
    assert report['profile'] is None and report['revision']==2
    assert control.read()['profile'] is None
