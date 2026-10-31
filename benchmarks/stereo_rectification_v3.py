#!/usr/bin/env python3
"""Independent stereo rectification diagnostics; NOT calibration or metric GT.

Unconstrained ORB+F-RANSAC detects descriptor outliers; independent 2D patch
matching compares an epipolar-only row with +/-v vertical searching. A patch
can still match the wrong repetitive feature; report support and ambiguity.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from stereo_epipolar_probe import load_rgb, orb_epipolar


def percentiles(a):
    a=np.asarray(a,dtype=np.float64)
    if not a.size:return {'count':0,'p50':None,'p90':None,'p95':None}
    return {'count':int(a.size),'p50':float(np.quantile(a,.5)),'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95))}


def orb_geometry(pairs, max_disparity=128, ransac_px=1.5):
    if len(pairs)<8:return {'usable':len(pairs),'positive':0,'fundamental_inliers':0,'diagnostic':'insufficient_features'},np.zeros(len(pairs),bool)
    coords=np.asarray([(*a,*b) for a,b,_ in pairs],dtype=np.float64)
    disparity=coords[:,0]-coords[:,2]
    dy=coords[:,1]-coords[:,3]
    positive=(disparity>=1)&(disparity<max_disparity)
    mask=np.zeros(len(pairs),bool)
    subset=np.flatnonzero(positive)
    if len(subset)>=8:
        F,inliers=cv2.findFundamentalMat(coords[subset,:2],coords[subset,2:4],cv2.FM_RANSAC,ransac_px,0.999)
        if F is not None and F.shape==(3,3) and inliers is not None and len(inliers)==len(subset):
            mask[subset[inliers.ravel().astype(bool)]]=True
    geom={'usable':len(pairs),'positive':int(positive.sum()),
          'positive_fraction':float(positive.mean()),
          'fundamental_inliers':int(mask.sum()),
          'fundamental_inlier_fraction_of_positive':float(mask.sum()/max(1,positive.sum())),
          'abs_dy_all':percentiles(np.abs(dy)),
          'abs_dy_positive':percentiles(np.abs(dy[positive])),
          'abs_dy_fundamental_inliers':percentiles(np.abs(dy[mask])),
          'signed_dy_fundamental_inliers':percentiles(dy[mask]),
          'fraction_inliers_with_abs_dy_le_1px':float((np.abs(dy[mask])<=1).mean()) if mask.any() else None,
          'fraction_inliers_with_abs_dy_le_2px':float((np.abs(dy[mask])<=2).mean()) if mask.any() else None,
          'note':'F-RANSAC is unconstrained (not forced horizontal). It can favor repetitive-feature matches.'}
    return geom,mask


def sparse_patch_2d(left,right,*,max_disparity=128,max_dy=8,patch_size=11,samples=280,
                    score_min=.7,uniqueness_min=.04,seed=19):
    if left.shape!=right.shape or left.ndim!=3 or left.shape[2]!=3 or left.dtype!=np.uint8 or right.dtype!=np.uint8:
        raise ValueError('expected two equal RGB8 [H,W,3] images')
    if not(3<=patch_size<=31 and patch_size%2 and 1<=max_disparity<=256 and 1<=max_dy<=32 and 8<=samples<=5000):
        raise ValueError('invalid patch settings')
    gl=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
    gr=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
    h,w=gl.shape
    r=patch_size//2
    if w<=max_disparity+patch_size+4 or h<=2*max_dy+patch_size+4:
        raise ValueError('image too small for sweep')
    pts=cv2.goodFeaturesToTrack(gl,maxCorners=min(12000,samples*12),qualityLevel=.008,minDistance=6,blockSize=5)
    if pts is None:return {'features':0,'candidates_evaluated':0,'high_confidence':0},[]
    pts=np.round(pts[:,0,:]).astype(int)
    rng=np.random.default_rng(seed)
    # Shuffle and spatially distribute to avoid all patches from one edge.
    idx=rng.permutation(len(pts))
    pts=pts[idx]
    results=[]
    evaluated=0
    for x,y in pts:
        if len(results)>=samples:break
        xlo=max(r,x-max_disparity);xhi=min(w-r-1,x-1)
        ylo=max(r,y-max_dy);yhi=min(h-r-1,y+max_dy)
        if xhi<xlo or yhi<ylo or y-r<0 or y+r>=h:continue
        patch=gl[y-r:y+r+1,x-r:x+r+1]
        if patch.std()<13:continue
        roi=gr[ylo-r:yhi+r+1,xlo-r:xhi+r+1]
        if roi.shape[0]<patch_size or roi.shape[1]<patch_size:continue
        response=cv2.matchTemplate(roi,patch,cv2.TM_CCOEFF_NORMED)
        evaluated+=int(response.size)
        best_ij=np.unravel_index(int(np.argmax(response)),response.shape)
        by,bx=best_ij
        best=float(response[by,bx])
        masked=response.copy()
        masked[max(0,by-2):by+3,max(0,bx-2):bx+3]=-1.0
        second=float(masked.max()) if masked.size else -1
        rx=xlo+bx;ry=ylo+by
        if ylo<=y<=yhi:
            same_row=float(response[y-ylo,:].max())
        else:same_row=None
        confidence=best>=score_min and best-second>=uniqueness_min
        results.append({'x':int(x),'y':int(y),'xr':int(rx),'yr':int(ry),
                        'dx':int(x-rx),'dy':int(y-ry),'best':best,'second':second,
                        'margin':best-second,'best_same_row':same_row,
                        'gain_from_vertical_search':best-same_row if same_row is not None else None,
                        'confident':bool(confidence)})
    conf=[p for p in results if p['confident']]
    dyd=np.array([p['dy'] for p in conf],float)
    gains=np.array([p['gain_from_vertical_search'] for p in conf],float)
    metrics={'features':int(len(pts)), 'candidate_patches_evaluated':len(results),
             'candidate_scores_evaluated':int(evaluated),
             'high_confidence':len(conf),'high_confidence_fraction':len(conf)/len(results) if results else None,
             'confidence_thresholds':{'score_min':score_min,'uniqueness_min':uniqueness_min},
             'signed_dy_high_confidence':percentiles(dyd),
             'abs_dy_high_confidence':percentiles(np.abs(dyd)),
             'vertical_gain_high_confidence':percentiles(gains),
             'high_confidence_fraction_abs_dy_le_1px':float((np.abs(dyd)<=1).mean()) if dyd.size else None,
             'high_confidence_fraction_abs_dy_le_2px':float((np.abs(dyd)<=2).mean()) if dyd.size else None,
             'high_confidence_fraction_abs_dy_gt_2_with_gain_gt_0_08':float(((np.abs(dyd)>2)&(gains>.08)).mean()) if dyd.size else None,
             'note':'2D NCC patch matcher searches disparity and vertical offset without row constraint. Confidence is heuristic; repetitive textures can pass.'}
    return metrics,results


def make_diagnostic(left,right,pairs,inliers,patches,path):
    h,w=left.shape[:2]
    # Downscale to make preview useful on 720p.
    out=np.hstack([cv2.cvtColor(left,cv2.COLOR_RGB2BGR),cv2.cvtColor(right,cv2.COLOR_RGB2BGR)])
    for i,(a,b,_) in enumerate(pairs):
        if i%max(1,len(pairs)//110):continue
        col=(60,215,60) if inliers[i] else (65,65,220)
        cv2.line(out,(round(a[0]),round(a[1])),(round(b[0])+w,round(b[1])),col,1,cv2.LINE_AA)
    for p in patches:
        if not p['confident']:continue
        col=(255,220,50) if abs(p['dy'])<=1 else (0,160,255)
        cv2.circle(out,(p['x'],p['y']),4,col,1)
        cv2.circle(out,(p['xr']+w,p['yr']),4,col,1)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),out):raise OSError('cannot write preview')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--image',type=Path)
    ap.add_argument('--max-disparity',type=int,default=128)
    ap.add_argument('--vertical-radius',type=int,default=8)
    ap.add_argument('--samples',type=int,default=280)
    ap.add_argument('--patch-size',type=int,default=11)
    ap.add_argument('--score-min',type=float,default=.70)
    ap.add_argument('--uniqueness-min',type=float,default=.04)
    a=ap.parse_args()
    if a.output.exists():ap.error('output already exists; use new output path')
    data=json.loads((a.fixture/'manifest.json').read_text())
    l=load_rgb(a.fixture,'rect_left_rgb',data)
    r=load_rgb(a.fixture,'rect_right_rgb',data)
    orig,matched=orb_epipolar(l,r,max_disparity=a.max_disparity)
    orb,inliers=orb_geometry(matched,a.max_disparity)
    patch,patches=sparse_patch_2d(l,r,max_disparity=a.max_disparity,max_dy=a.vertical_radius,
                                   samples=a.samples,patch_size=a.patch_size,
                                   score_min=a.score_min,uniqueness_min=a.uniqueness_min)
    report={'schema':'stereoforge.rectification_robust_v3','scope':'single precomputed rectified fixture; no ground truth or measured live camera geometry',
            'fixture':str(a.fixture),'orb_original':orig,'orb_geometry':orb,'patch_search':patch,
            'patch_high_confidence_matches':[{k:v for k,v in p.items() if k not in ('best_same_row',)} for p in patches if p['confident']][:500],
            'interpretation':'High-confidence NCC and fundamental matrix inliers are not absolute ground truth. Compare full distributions and overlay.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    if a.image:make_diagnostic(l,r,matched,inliers,patches,a.image)
    print('ORB matches',orb['usable'],'positive',orb['positive'],'F-inliers',orb['fundamental_inliers'])
    print('ORB F-inliers abs(dy) p50/p90:',orb['abs_dy_fundamental_inliers']['p50'],orb['abs_dy_fundamental_inliers']['p90'])
    print('Independent NCC patches:',patch['candidate_patches_evaluated'],'confident',patch['high_confidence'])
    print('NCC confident abs(dy) p50/p90:',patch['abs_dy_high_confidence']['p50'],patch['abs_dy_high_confidence']['p90'])
    print('NCC confident fraction abs(dy)>2 and vertical gain>.08:',patch['high_confidence_fraction_abs_dy_gt_2_with_gain_gt_0_08'])
    print('Saved',a.output)

if __name__=='__main__':main()
