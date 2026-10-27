"""ZED Open Capture adapter. No proprietary SDK or guessed calibration fallback.

Sensor vectors retain upstream raw sensor axes: acceleration in m/s², angular
velocity in degrees/s, temperature in °C. A sync bit/timestamp is association
metadata, not an IMU-to-camera extrinsic calibration or a motion estimate.
"""
import importlib.util
import os
from pathlib import Path
import sysconfig
import cv2
import numpy as np
from .types import StereoFrame
from .calibration import download_calibration, parse_zed_conf


def _native_module():
    directory=Path(os.environ.get('STEREOFORGE_NATIVE_DIR',Path(__file__).resolve().parents[2]/'build'/'native'))
    filename=directory/('_zed_capture'+sysconfig.get_config_var('EXT_SUFFIX'))
    if not filename.is_file():
        raise RuntimeError('ZED capture extension is unavailable; build CMake with STEREOFORGE_CAPTURE=ON '
                           'and set STEREOFORGE_NATIVE_DIR to its native output directory')
    spec=importlib.util.spec_from_file_location('_zed_capture',filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_cameras():
    """Read V4L sysfs identity and USB speed; do not open arbitrary video nodes."""
    devices=[]
    for path in sorted(Path('/sys/class/video4linux').glob('video*')):
        name_path=path/'name'
        try:
            name=name_path.read_text().strip()
            physical=(path/'device').resolve()
            usb=None
            for ancestor in (physical,*physical.parents):
                if (ancestor/'idVendor').is_file():
                    usb={'vendor_id':(ancestor/'idVendor').read_text().strip(),
                         'product_id':(ancestor/'idProduct').read_text().strip(),
                         'speed_mbps':float((ancestor/'speed').read_text())}
                    break
            devices.append({'device':'/dev/'+path.name,'name':name,'usb':usb})
        except (OSError,ValueError):
            continue
    return devices


def yuyv_to_stereo(yuyv,calibration,timestamp_ns,frame_id,sensor_samples=None):
    """Convert copied packed side-by-side YUYV, preserving both raw lens canvases."""
    left,right=calibration.left,calibration.right
    if (left.width,left.height)!=(right.width,right.height):
        raise ValueError('ZED capture requires equal left/right image dimensions')
    if not isinstance(yuyv,np.ndarray) or yuyv.dtype!=np.uint8 or yuyv.shape!=(left.height,2*left.width,2):
        raise ValueError('expected uint8 side-by-side YUYV with calibrated dimensions')
    if left.width%2:
        raise ValueError('YUYV lens width must be even')
    rgb=cv2.cvtColor(np.ascontiguousarray(yuyv),cv2.COLOR_YUV2RGB_YUY2)
    return StereoFrame(rgb[:,:left.width].copy(),rgb[:,left.width:].copy(),calibration,
                       int(timestamp_ns),int(frame_id),'zed_open_capture',list(sensor_samples or []))


class ZedCamera:
    def __init__(self,calibration=None,width=1280,height=720,fps=60,device_id=-1,cache_dir=None):
        self._native=None
        native=_native_module().ZedCapture(width=width,height=height,fps=fps,device_id=device_id)
        try:
            self.serial=str(native.serial)
            if calibration is None:
                cache=Path(cache_dir or Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'stereoforge'/'calibration')
                path=download_calibration(self.serial,cache)
                calibration=parse_zed_conf(path,width,height,self.serial)
            if str(calibration.serial)!=self.serial:
                raise ValueError('calibration serial does not match attached camera')
            if any((c.width,c.height)!=(width,height) for c in (calibration.left,calibration.right)):
                raise ValueError('calibration dimensions do not match capture mode')
            self.calibration=calibration
            self._native=native
        except BaseException:
            native.close()
            raise

    @property
    def sensors_available(self):
        return bool(self._native is not None and self._native.sensors_available)

    @property
    def sensor_status(self):
        return 'closed' if self._native is None else self._native.sensor_status

    @property
    def sensor_dropped_count(self):
        return 0 if self._native is None else self._native.sensor_dropped_count

    def read(self,timeout_ms=1000):
        if self._native is None:
            raise RuntimeError('camera is closed')
        result=self._native.read(timeout_ms)
        return yuyv_to_stereo(result['yuyv'],self.calibration,result['timestamp_ns'],result['frame_id'],result['sensor_samples'])

    def close(self):
        native=self._native
        self._native=None
        if native is not None:
            native.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
