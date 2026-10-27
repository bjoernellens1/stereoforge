"""Warmed CPU references, not end-to-end or GPU performance claims.

Run: PYTHONPATH=python .venv/bin/python benchmarks/host_hotpaths.py
"""
import json
from time import perf_counter
import numpy as np
from stereoforge.types import CameraModel, Calibration, StereoFrame, DepthFrame
from stereoforge.calibration import Rectifier
from stereoforge.fusion import synthesize, default_virtual_camera
from stereoforge.temporal import reproject_depth


def measure(call):
    call()
    samples=[]
    for _ in range(3):
        start=perf_counter(); call(); samples.append((perf_counter()-start)*1000)
    return {'samples_ms':samples,'p50_ms':float(np.median(samples))}


def main():
    width,height=1280,720
    c=CameraModel(width,height,700.,700.,639.5,359.5)
    pose=np.eye(4); pose[0,3]=.12
    calibration=Calibration(c,c,pose)
    rectifier=Rectifier(calibration)
    shape=(height,width)
    def depth(identity):
        return DepthFrame(np.full(shape,42.,np.float32),np.full(shape,2.,np.float32),
                          np.ones(shape,np.float32),np.ones(shape,bool),np.ones(shape,np.uint8),
                          np.zeros(shape,np.int64),np.zeros(shape,np.int64),100,identity)
    rectified=depth(rectifier.rectified_calibration.identity)
    raw=depth(calibration.identity)
    rgb=np.full((*shape,3),128,np.uint8)
    frame=StereoFrame(rgb,rgb.copy(),calibration,100,0)
    virtual=default_virtual_camera(calibration)
    results={'resolution':[width,height],'repetitions':3,'warmups':1,
             'depth_to_raw':measure(lambda:rectifier.depth_to_raw(rectified)),
             'synthesize_strict_cpu':measure(lambda:synthesize(frame,raw,virtual)),
             'reproject_depth_cpu':measure(lambda:reproject_depth(rectified,c,np.eye(4),101))}
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
