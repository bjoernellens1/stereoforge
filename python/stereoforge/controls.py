"""Validated normalized camera controls and host-sampled readback reports.

Snapshots describe control-register readback, not exposure-synchronous metadata.
Manual profiles share a policy; their labels express intent, not quality guarantees.
"""
from copy import deepcopy
import time
from functools import wraps
from threading import RLock

_RANGES={**{k:(0,8) for k in ('brightness','contrast','sharpness','saturation')},
         'hue':(0,11),'gamma':(1,9),
         **{k:(0,100) for k in ('exposure_left','exposure_right','gain_left','gain_right')},
         'white_balance':(2800,6500)}
_BOOLS={'auto_exposure_gain','auto_white_balance','led'}


def validate_controls(settings,width,height):
    if type(width) is not int or type(height) is not int or min(width,height)<=0:
        raise ValueError('invalid source canvas')
    if not isinstance(settings,dict):raise ValueError('settings must be a dictionary')
    for key,value in settings.items():
        if key in _RANGES:
            low,high=_RANGES[key]
            if type(value) is not int or not low<=value<=high:
                raise ValueError(f'{key} must be an integer in [{low},{high}]')
        elif key in _BOOLS:
            if type(value) is not bool:raise ValueError(f'{key} must be bool')
        elif key in ('roi_left','roi_right'):
            if value is not None:
                if not isinstance(value,(tuple,list)) or len(value)!=4 or any(type(v) is not int for v in value):
                    raise ValueError('ROI requires four strict integers')
                x,y,w,h=value
                if x<0 or y<0 or w<=0 or h<=0 or w*h<=100 or x+w>width or y+h>height:
                    raise ValueError('ROI lies outside source canvas')
        else:raise ValueError(f'unknown control {key}')
    if settings.get('auto_exposure_gain') is True and any(k in settings for k in ('exposure_left','exposure_right','gain_left','gain_right')):
        raise ValueError('manual exposure/gain conflicts with automatic control')
    if settings.get('auto_white_balance') is True and 'white_balance' in settings:
        raise ValueError('manual white balance conflicts with automatic control')
    return {k:list(v) if k in ('roi_left','roi_right') and v is not None else v for k,v in settings.items()}


def profile_settings(profile,*,exposure=None,gain=None,white_balance=None,sharpness=0):
    """Expand a named profile to explicit normalized per-eye native controls."""
    if profile=='automatic':
        if any(v is not None for v in (exposure,gain,white_balance)) or type(sharpness) is not int or sharpness!=0:
            raise ValueError('automatic profile does not accept manual settings')
        settings={'auto_exposure_gain':True,'auto_white_balance':True}
    elif profile in ('stereo_optimized','motion_optimized'):
        if any(v is None for v in (exposure,gain,white_balance)):
            raise ValueError('manual profiles require explicit exposure, gain and white balance')
        settings=dict(auto_exposure_gain=False,auto_white_balance=False,
                      exposure_left=exposure,exposure_right=exposure,
                      gain_left=gain,gain_right=gain,white_balance=white_balance,sharpness=sharpness)
    else:raise ValueError('unknown camera profile')
    return validate_controls(settings,1,1)  # Profiles contain no canvas-dependent ROIs.


def _locked(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self._lock:
            return method(self,*args,**kwargs)
    return call


class CameraControl:
    def __init__(self,native,width,height):
        if type(width) is not int or type(height) is not int or min(width,height)<=0:
            raise ValueError('invalid source canvas')
        self._lock=RLock()
        self.native=native;self.width=width;self.height=height
        self._revision=0;self._profile=None;self._requested={}
        self._snapshot=self._report({},readonly=True)

    @property
    @_locked
    def revision(self):
        return self._revision

    @property
    @_locked
    def snapshot(self):
        return deepcopy(self._snapshot)

    def _report(self,values,readonly=False,write_errors=None):
        values=deepcopy(values)
        read_errors=values.pop('read_errors',[]) or []
        errors=values.pop('write_errors',[]) or []
        if write_errors:errors=list(errors)+list(write_errors)
        matches={}
        for key,wanted in self._requested.items():
            actual=values.get(key)
            expected=[0,0,self.width,self.height] if key in ('roi_left','roi_right') and wanted is None else wanted
            matches[key]=None if actual is None else actual==expected
        if readonly:
            applied=None
        elif read_errors or errors or any(v is False for v in matches.values()):
            applied=False
        elif any(v is None for v in matches.values()):
            applied=None
        else:
            applied=True
        return dict(profile=self._profile,revision=self._revision,
            requested=deepcopy(self._requested),readback=values,read_errors=read_errors,
            write_errors=errors,matches=matches,applied=applied,
            sampled_at_host_ns=time.monotonic_ns(),clock_domain='host_monotonic',
            frame_association='host_snapshot_not_exposure_synchronous')

    @_locked
    def read(self):
        try:
            values=self.native.get_controls()
            if not isinstance(values,dict):raise TypeError('control readback must be a dictionary')
            self._snapshot=self._report(values,readonly=True)
        except Exception as exc:
            self._snapshot=self._report({'read_errors':[str(exc)]},readonly=True)
            raise
        return self.snapshot

    def _validate(self,settings):
        return validate_controls(settings,self.width,self.height)

    @_locked
    def set(self,settings):
        return self._set(settings)

    def _set(self,settings,profile=None):
        # Called only while the public read/write lock is held.
        validated=self._validate(settings)
        self._profile=profile
        self._revision+=1;self._requested=deepcopy(validated)
        try:
            values=self.native.set_controls(deepcopy(validated))
            if not isinstance(values,dict):values=self.native.get_controls()
            if not isinstance(values,dict):raise TypeError('control readback must be a dictionary')
            self._snapshot=self._report(values)
        except Exception as exc:
            try:
                values=self.native.get_controls()
                if not isinstance(values,dict):raise TypeError('invalid control readback')
            except Exception as read_exc:
                values={'read_errors':[str(read_exc)]}
            self._snapshot=self._report(values,write_errors=[str(exc)])
            raise
        return self.snapshot

    @_locked
    def apply_profile(self,profile,*,exposure=None,gain=None,white_balance=None,sharpness=0):
        settings=profile_settings(profile,exposure=exposure,gain=gain,
                                  white_balance=white_balance,sharpness=sharpness)
        return self._set(settings,profile=profile)
