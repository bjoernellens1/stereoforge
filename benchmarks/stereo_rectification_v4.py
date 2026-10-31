#!/usr/bin/env python3
"""StereoForge v4: reciprocal NCC rectification test + independently matched SGM.

Offline diagnostics only. Individual NCC matches can be confidently wrong,
particularly on repetitive texture. Do not use as metric ground truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from stereo_epipolar_probe import load_rgb
from stereo_rectification_v3 import sparse_patch_2d, percentiles


def best_patch(image, target, xlo, xhi, ylo, yhi, radius):
    h, w = image.shape
    xlo=max(radius,int(xlo));xhi=min(w-1-radius,int(xhi))
    ylo=max(radius,int(ylo));yhi=min(h-1-radius,int(yhi))
    if xhi<xlo or yhi<ylo:return None
    roi=image[ylo-radius:yhi+radius+1,xlo-radius:xhi+radius+1]
    response=cv2.matchTemplate(roi,target,cv2.TM_CCOEFF_NORMED)
    if not np.isfinite(response).all():return None
    by,bx=np.unravel_index(int(np.argmax(response)),response.shape)
    best=float(response[by,bx]);shadow=response.copy()
    shadow[max(0,by-2):by+3,max(0,bx-2):bx+3]=-1
    runnerup=float(shadow.max())
    return (xlo+int(bx), ylo+int(by), best, best-runnerup)


def reciprocal_samples(left,right,patches,*,max_disparity=128,vertical_radius=8,
                       patch_size=11,min_score=.70,min_margin=.04):
    sample=[]
    gl=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
    gr=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
    rad=patch_size//2
    for p in patches:
        if not p['confident']:continue
        x,y=p['xr'],p['yr']
        if x<rad or x>=gr.shape[1]-rad or y<rad or y>=gr.shape[0]-rad:continue
        match=best_patch(gl,gr[y-rad:y+rad+1,x-rad:x+rad+1],x+1,x+max_disparity,
                         y-vertical_radius,y+vertical_radius,rad)
        if match is None:continue
        mx,my,score,margin=match
        rec=abs(mx-p['x'])<=1 and abs(my-p['y'])<=1
        sample.append({**p,'back_x':mx,'back_y':my,'back_score':score,
                       'back_margin':margin,'cycle_dx':mx-p['x'],'cycle_dy':my-p['y'],
                       'reciprocal':bool(rec),
                       'strict_reciprocal':bool(rec and score>=min_score and margin>=min_margin)})
    return sample


def by_tiles(points,width,height,cols=4,rows=3):
    out=[]
    for j in range(rows):
        for i in range(cols):
            xlo=i*width/cols;xhi=(i+1)*width/cols
            ylo=j*height/rows;yhi=(j+1)*height/rows
            p=[q for q in points if xlo<=q['x']<xhi and ylo<=q['y']<yhi]
            dy=np.array([q['dy'] for q in p],float)
            gain=np.array([q['gain_from_vertical_search'] for q in p],float)
            out.append({'tile':[i,j],'count':len(p),'dy':percentiles(dy),
                        'abs_dy':percentiles(abs(dy)),
                        'fraction_gt2_and_gain08':float(((abs(dy)>2)&(gain>.08)).mean()) if len(p) else None})
    return out


def read_sgm(prefix,width,height):
    prefix=Path(prefix);shape=(height,width)
    arr={}
    for eye in ('left','right'):
        for name,dtype in [('disparity.f32','<f4'),('validity.u32','<u4')]:
            p=Path(str(prefix)+f'.{eye}.{name}')
            a=np.fromfile(p,dtype=dtype)
            if a.size!=width*height:raise ValueError(f'Wrong dump size: {p}: {a.size} != {width*height}')
            arr[f'{eye}_{name.split(".")[0]}']=a.reshape(shape)
    return arr


def compare_sgm(points,arrays,tolerance=1.,max_disparity=128):
    stats=[]
    h,w=arrays['left_disparity'].shape
    for p in points:
        xl,yl,xr,yr=(p[k] for k in ('x','y','xr','yr'))
        if not(0<=xl<w and 0<=xr<w and 0<=yl<h and 0<=yr<h):continue
        d=p['dx']
        l_ok=bool(arrays['left_validity'][yl,xl]);r_ok=bool(arrays['right_validity'][yr,xr])
        l=float(arrays['left_disparity'][yl,xl]);r=float(arrays['right_disparity'][yr,xr])
        stats.append({'x':xl,'y':yl,'dx':d,'dy':p['dy'], 'left_valid':l_ok,'right_valid':r_ok,
                      'left_disparity':l if np.isfinite(l) else None,
                      'right_disparity':r if np.isfinite(r) else None,
                      'left_agrees_with_ncc':bool(l_ok and np.isfinite(l) and abs(l-d)<=tolerance),
                      'right_agrees_with_ncc':bool(r_ok and np.isfinite(r) and abs(r-d)<=tolerance),
                      'left_right_agree':bool(l_ok and r_ok and np.isfinite(l) and np.isfinite(r) and abs(l-r)<=tolerance)})
    if not stats:return {'matches_compared':0},[]
    def ratio(k):return float(np.mean([v[k] for v in stats]))
    summary={'matches_compared':len(stats),'left_selection_valid':ratio('left_valid'),
             'right_selection_valid':ratio('right_valid'),
             'left_matches_independent_patch':ratio('left_agrees_with_ncc'),
             'right_matches_independent_patch':ratio('right_agrees_with_ncc'),
             'both_match_independent_patch':float(np.mean([v['left_agrees_with_ncc'] and v['right_agrees_with_ncc'] for v in stats])),
             'left_right_agree_at_independent_patch':ratio('left_right_agree'),
             'warning':'NCC support has survivorship/texture bias; these comparisons are diagnostics not ground truth'}
    return summary,stats


def overlay(left,points,path,cols=4,rows=3):
    image=cv2.cvtColor(left,cv2.COLOR_RGB2BGR).copy();h,w=image.shape[:2]
    for i in range(1,cols):cv2.line(image,(round(i*w/cols),0),(round(i*w/cols),h-1),(90,90,90),1)
    for j in range(1,rows):cv2.line(image,(0,round(j*h/rows)),(w-1,round(j*h/rows)),(90,90,90),1)
    for p in points:
        # Green row-aligned reciprocal; orange reciprocal shifted; red nonreciprocal
        col=(0,225,0) if abs(p['dy'])<=2 else (0,135,255)
        if not p['strict_reciprocal']:col=(20,20,240)
        cv2.circle(image,(p['x'],p['y']),3,col,1,cv2.LINE_AA)
        if abs(p['dy'])>2 and p['strict_reciprocal']:
            cv2.line(image,(p['x'],p['y']),(p['x'],int(np.clip(p['y']-p['dy'],0,h-1))),col,1)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image):raise RuntimeError(f'Cannot write {path}')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--image',type=Path)
    ap.add_argument('--sgm-prefix',type=Path)
    ap.add_argument('--max-disparity',type=int,default=128)
    ap.add_argument('--vertical-radius',type=int,default=8)
    ap.add_argument('--samples',type=int,default=350)
    ap.add_argument('--patch-size',type=int,default=11)
    ap.add_argument('--min-score',type=float,default=.70)
    ap.add_argument('--min-margin',type=float,default=.04)
    a=ap.parse_args()
    if a.output.exists():ap.error('output already exists; use a new output path')
    fixture=json.loads((a.fixture/'manifest.json').read_text())
    left=load_rgb(a.fixture,'rect_left_rgb',fixture);right=load_rgb(a.fixture,'rect_right_rgb',fixture)
    metrics,patches=sparse_patch_2d(left,right,max_disparity=a.max_disparity,max_dy=a.vertical_radius,
                                    samples=a.samples,patch_size=a.patch_size,
                                    score_min=a.min_score,uniqueness_min=a.min_margin)
    sample=reciprocal_samples(left,right,patches,max_disparity=a.max_disparity,
                              vertical_radius=a.vertical_radius,patch_size=a.patch_size,
                              min_score=a.min_score,min_margin=a.min_margin)
    strict=[p for p in sample if p['strict_reciprocal']]
    offsets=[p for p in strict if abs(p['dy'])>2 and p['gain_from_vertical_search']>.08]
    signed=np.array([p['dy'] for p in strict],float)
    out={'schema':'stereoforge.rectification_reciprocal_probe.v4',
         'scope':'single rectified fixture, offline patch diagnostics; neither NCC nor SGM is ground truth',
         'input':str(a.fixture),'config':{'max_disparity':a.max_disparity,'vertical_radius':a.vertical_radius,'samples':a.samples,'patch_size':a.patch_size},
         'forward':metrics,
         'reciprocal':{'forward_confident':len([p for p in patches if p['confident']]),
                       'reverse_attempted':len(sample),'strict_reciprocal':len(strict),
                       'signed_dy':percentiles(signed),'abs_dy':percentiles(abs(signed)),
                       'fraction_abs_dy_gt2_and_gain_gt08':len(offsets)/len(strict) if strict else None,
                       'count_abs_dy_gt2_and_gain_gt08':len(offsets)},
         'tiles':by_tiles(strict,left.shape[1],left.shape[0]),
         'examples':[{k:p[k] for k in ('x','y','xr','yr','dx','dy','best','back_score','back_margin','gain_from_vertical_search')} for p in strict[:75]],
         'warnings':['Reciprocal NCC matches may still be false in repetitive scenes',
                     'No target-derived correction should be applied without independent physical validation']}
    if a.sgm_prefix is not None:
        summary,results=compare_sgm(strict,read_sgm(a.sgm_prefix,left.shape[1],left.shape[0]))
        out['sgm_comparison']=summary
        out['sgm_examples']=results[:40]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    if a.image:overlay(left,sample,a.image)
    print(f"Forward NCC confident: {out['reciprocal']['forward_confident']} / {metrics['candidate_patches_evaluated']}")
    print(f"Strict reciprocal: {len(strict)} | abs(dy) p50/p90 = {out['reciprocal']['abs_dy']['p50']} / {out['reciprocal']['abs_dy']['p90']}")
    print(f"Reciprocal significant vertical gain: {len(offsets)} / {len(strict)}")
    if a.sgm_prefix:print('SGM comparison:',json.dumps(out['sgm_comparison']))
    print('JSON:',a.output)

if __name__=='__main__':main()
