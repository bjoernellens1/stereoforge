#!/usr/bin/env python3
"""Independent ORB epipolar smoke test from StereoForge rectification fixture.

ORB matches are only image correspondences, not ground truth. Repetitive
structures can produce outliers. No modification to native stereo shaders.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def load_rgb(fixture: Path, label: str, data: dict):
    # StereoForge's pinned daxa_preprocess_fixture.v1 stores output shape in
    # `rectified`, not `rectification`. Keep support for older fixtures.
    shape = data.get('rectified', data.get('rectification'))
    if not isinstance(shape, dict) or 'height' not in shape or 'width' not in shape:
        raise ValueError('Fixture missing rectified width/height')
    h, w = int(shape['height']), int(shape['width'])
    if h <= 0 or w <= 0 or h > 8192 or w > 8192:
        raise ValueError('Invalid rectified image dimensions')
    info=data['expected'][label]
    file=fixture / info['file']
    blob=file.read_bytes()
    if len(blob)!=int(info['bytes']) or hashlib.sha256(blob).hexdigest()!=info['sha256']:
        raise ValueError(f'Fixture payload checksum/size mismatch: {file}')
    if len(blob)!=w*h*3: raise ValueError(f'Unexpected RGB8 fixture dimensions: {file}')
    return np.frombuffer(blob,dtype=np.uint8).reshape((h,w,3)).copy()


def orb_epipolar(left, right, max_disparity=128, nfeatures=8000, max_distance=65):
    if left.shape!=right.shape or left.ndim!=3 or left.shape[-1]!=3 or left.dtype!=np.uint8 or right.dtype!=np.uint8:
        raise ValueError('Expected matching uint8 [height,width,3] RGB images')
    if not (1<=max_disparity<=256 and nfeatures>=100 and 1<=max_distance<=256):
        raise ValueError('Invalid ORB parameters')
    detector=cv2.ORB_create(nfeatures=nfeatures,fastThreshold=12,edgeThreshold=24)
    kp_l, desc_l = detector.detectAndCompute(cv2.cvtColor(left,cv2.COLOR_RGB2GRAY),None)
    kp_r, desc_r = detector.detectAndCompute(cv2.cvtColor(right,cv2.COLOR_RGB2GRAY),None)
    if desc_l is None or desc_r is None:
        return {'features_left':len(kp_l),'features_right':len(kp_r),'mutual_matches':0,'usable_matches':0,
                'epipolar_p50_abs_dy_px':None, 'epipolar_p90_abs_dy_px':None,
                'positive_disparity_fraction':None},[]
    matched=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(desc_l,desc_r)
    filtered=[m for m in matched if m.distance<=max_distance]
    if filtered:
        coords=np.array([[*kp_l[m.queryIdx].pt,*kp_r[m.trainIdx].pt] for m in filtered],dtype=np.float64)
        dx=coords[:,0]-coords[:,2];dy=coords[:,1]-coords[:,3]
        positive=(dx>0)&(dx<max_disparity)
        # Metric computed both on all descriptor matches and on geometrically
        # plausible positive-disparity pairs. The latter may hide failures.
        in_range=positive & (np.abs(dy)<=3.)
        row_errors=np.abs(dy)
        results={'features_left':len(kp_l),'features_right':len(kp_r),
                 'mutual_matches':len(matched),'usable_matches':len(filtered),
                 'epipolar_p50_abs_dy_px':float(np.median(row_errors)),
                 'epipolar_p90_abs_dy_px':float(np.quantile(row_errors,.9)),
                 'positive_disparity_fraction':float(np.mean(positive)),
                 'positive_and_row_within_3px_fraction':float(np.mean(in_range)),
                 'positive_disparity_median_px':float(np.median(dx[positive])) if positive.any() else None,
                 'positive_epipolar_p90_abs_dy_px':float(np.quantile(row_errors[positive],.9)) if positive.any() else None}
    else:
        results={'features_left':len(kp_l),'features_right':len(kp_r),
                 'mutual_matches':len(matched),'usable_matches':0,
                 'epipolar_p50_abs_dy_px':None,'epipolar_p90_abs_dy_px':None,
                 'positive_disparity_fraction':None}
    return results,[(kp_l[m.queryIdx].pt,kp_r[m.trainIdx].pt, float(m.distance)) for m in filtered]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True,help='directory containing manifest.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image',type=Path)
    p.add_argument('--max-disparity',type=int,default=128)
    p.add_argument('--nfeatures',type=int,default=8000)
    a=p.parse_args()
    if a.output.exists():p.error('output already exists')
    info=json.loads((a.fixture/'manifest.json').read_text())
    l=load_rgb(a.fixture,'rect_left_rgb',info)
    r=load_rgb(a.fixture,'rect_right_rgb',info)
    report,pairs=orb_epipolar(l,r,max_disparity=a.max_disparity,nfeatures=a.nfeatures)
    report['scope']='independent ORB matches from the pinned rectified fixture; not metric depth ground truth'
    report['schema']='stereoforge.epipolar_orb_probe.v1'
    report['fixture_manifest']=str(a.fixture/'manifest.json')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    if a.image:
        # Use the best 100 mutual matches sorted by Hamming distance.
        pairs=sorted(pairs,key=lambda pair:pair[2])[:100]
        draw=np.hstack([cv2.cvtColor(l,cv2.COLOR_RGB2BGR),cv2.cvtColor(r,cv2.COLOR_RGB2BGR)])
        h,w=l.shape[:2]
        for (lx,ly),(rx,ry),dist in pairs:
            cv2.line(draw,(round(lx),round(ly)),(round(rx+w),round(ry)),(0,220,60),1,cv2.LINE_AA)
        a.image.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(a.image),draw): raise RuntimeError('Failed to save ORB match image')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
