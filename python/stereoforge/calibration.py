"""Calibration loading and rectification; all translations are metres.

ZED INI Rodrigues(RX,CV,RZ) is the left-to-right optical rotation.
The raw right-camera pose uses its transpose and translation (+Baseline,TY,TZ).
On fixed ZED 2 correspondences the corrected rotation reduced rectified vertical
residual p50 from 1.3–1.4 to .53–.56 pixels; this does not qualify metric depth
or establish unsuffixed translation conventions for other camera families.
"""
import configparser
import json
from pathlib import Path
import re
import urllib.request
import cv2
import numpy as np
from .types import Calibration, CameraModel, StereoFrame, DepthFrame


def load_calibration(path):
    try:
        return Calibration.from_dict(json.loads(Path(path).read_text()))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('missing or invalid calibration') from exc


def parse_zed_conf(path,width,height,serial):
    modes={(2208,1242):'2k',(1920,1080):'fhd',(1280,720):'hd',(672,376):'vga'}
    if (width,height) not in modes:
        raise ValueError('unsupported ZED resolution')
    mode=modes[width,height]
    try:
        parser=configparser.ConfigParser(interpolation=None)
        with Path(path).open() as stream: parser.read_file(stream)
        sections={name.lower():parser[name] for name in parser.sections()}
        def camera(side):
            values=sections[f'{side}_cam_{mode}']
            return CameraModel(width,height,*(float(values[k]) for k in ('fx','fy','cx','cy')),
                               tuple(float(values[k]) for k in ('k1','k2','p1','p2','k3')))
        left,right=camera('left'),camera('right')
        stereo=sections['stereo']
        baseline=float(stereo['baseline'])
        if not np.isfinite(baseline) or baseline<=0: raise ValueError('invalid baseline')
        pose=np.eye(4)
        pose[:3,:3]=cv2.Rodrigues(np.array([float(stereo[f'{k}_{mode}']) for k in ('rx','cv','rz')]))[0].T
        pose[:3,3]=np.array([baseline,float(stereo.get(f'ty_{mode}',0)),float(stereo.get(f'tz_{mode}',0))])*.001
        return Calibration(left,right,pose,str(serial))
    except (OSError, KeyError, TypeError, configparser.Error, cv2.error) as exc:
        raise ValueError('missing or invalid ZED calibration') from exc


def download_calibration(serial,cache_dir):
    serial=str(serial)
    if not re.fullmatch(r'[0-9]+',serial): raise ValueError('serial must be numeric')
    directory=Path(cache_dir); target=directory/f'SN{serial}.conf'
    if target.exists():
        if not target.is_file() or not target.stat().st_size: raise ValueError('empty cached calibration')
        return target
    with urllib.request.urlopen(f'https://calib.stereolabs.com/?SN={serial}',timeout=15) as response:
        content=response.read()
    if not content.strip(): raise ValueError('empty downloaded calibration')
    directory.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects an existing cache, including concurrent downloads.
    try:
        with target.open('xb') as stream: stream.write(content)
    except FileExistsError:
        if not target.stat().st_size: raise ValueError('empty cached calibration')
    return target


class Rectifier:
    def __init__(self,calibration):
        self.calibration=calibration
        left,right=calibration.left,calibration.right
        if (left.width,left.height)!=(right.width,right.height):
            raise ValueError('rectification requires matching image sizes')
        inverse=np.linalg.inv(calibration.right_to_left)
        self.R1,self.R2,self.P1,self.P2,_,_,_=cv2.stereoRectify(
            left.K,self._dist(left),right.K,self._dist(right),(left.width,left.height),
            inverse[:3,:3].copy(),inverse[:3,3].reshape(3,1).copy(),flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
        def model(p):
            return CameraModel(left.width,left.height,float(p[0,0]),float(p[1,1]),float(p[0,2]),float(p[1,2]))
        self.left_camera,self.right_camera=model(self.P1),model(self.P2)
        pose=np.eye(4); pose[0,3]=calibration.baseline_m
        self.rectified_calibration=Calibration(self.left_camera,self.right_camera,pose,calibration.serial)
        self._maps=[cv2.initUndistortRectifyMap(c.K,self._dist(c),r,p,(c.width,c.height),cv2.CV_32FC1)
                    for c,r,p in [(left,self.R1,self.P1),(right,self.R2,self.P2)]]

    @staticmethod
    def _dist(camera):
        return np.asarray(camera.distortion,dtype=float) if camera.distortion else np.zeros(5)

    def rectify(self,frame):
        if frame.calibration.identity!=self.calibration.identity: raise ValueError('calibration mismatch')
        images=[cv2.remap(image,*maps,cv2.INTER_LINEAR) for image,maps in zip((frame.left,frame.right),self._maps)]
        return StereoFrame(*images,self.rectified_calibration,frame.timestamp_ns,frame.frame_id,
                           frame.clock_domain,frame.sensor_samples.copy())

    def depth_to_raw(self,depth,side='left'):
        if side not in ('left','right'): raise ValueError('side must be left or right')
        if depth.calibration_id!=self.rectified_calibration.identity: raise ValueError('calibration mismatch')
        raw=getattr(self.calibration,side); rect=getattr(self,f'{side}_camera')
        rotation=self.R1 if side=='left' else self.R2
        shape=(raw.height,raw.width)
        if depth.depth_m.shape!=(rect.height,rect.width): raise ValueError('depth dimensions mismatch')
        z=np.full(shape,np.nan,np.float32); valid=np.zeros(shape,bool)
        confidence=np.zeros(shape,np.float32); provenance=np.zeros(shape,np.uint8)
        ids=np.full(shape,-1,np.int64); age=np.zeros(shape,np.int64)
        yy,xx=np.where(depth.valid)
        points=np.column_stack([(xx-rect.cx)/rect.fx,(yy-rect.cy)/rect.fy,np.ones(len(xx))])*depth.depth_m[yy,xx,None]
        points=points@rotation  # R.T maps rectified column vectors to raw coordinates.
        if len(points):
            projected=cv2.projectPoints(points,np.zeros(3),np.zeros(3),raw.K,self._dist(raw))[0][:,0]
            finite=np.isfinite(projected).all(axis=1)&np.isfinite(points).all(axis=1)&(points[:,2]>0)
            supported=finite & (projected[:,0]>=-.5) & (projected[:,0]<raw.width-.5) & (projected[:,1]>=-.5) & (projected[:,1]<raw.height-.5)
            source=np.flatnonzero(supported)
            pixels=np.floor(projected[source]+.5).astype(np.int64)
            target=pixels[:,1]*raw.width+pixels[:,0]
            # Primary target, then axial depth, then original raster index.
            order=np.lexsort((source,points[source,2],target))
            sorted_target=target[order]
            first=np.r_[True,sorted_target[1:]!=sorted_target[:-1]] if len(order) else np.empty(0,bool)
            winners=order[first]; selected=source[winners]
            x,y=pixels[winners].T
            sy,sx=yy[selected],xx[selected]
            z[y,x]=points[selected,2]; valid[y,x]=True
            confidence[y,x]=depth.confidence[sy,sx]; provenance[y,x]=depth.provenance[sy,sx]
            ids[y,x]=depth.source_ids[sy,sx]; age[y,x]=depth.age_ns[sy,sx]
        return DepthFrame(np.full(shape,np.nan,np.float32),z,confidence,valid,provenance,ids,age,
                          depth.timestamp_ns,self.calibration.identity)
