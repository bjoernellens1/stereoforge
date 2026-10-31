#!/usr/bin/env python3
"""Independent OpenCV SGBM vs StereoForge native left/right diagnostic.

Uses identical pinned rectified RGB fixture and native per-view disparity dumps.
SGBM is a second estimator, not ground truth. Tests matcher consistency, not
metric depth accuracy or calibration. No production pipeline changes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np


def rgb_from_fixture(root: Path, name: str, manifest: dict) -> np.ndarray:
    shape = manifest.get('rectified') or manifest.get('rectification')
    if not shape or 'width' not in shape or 'height' not in shape:
        raise ValueError('Missing rectified shape in fixture manifest')
    h,w=int(shape['height']),int(shape['width'])
    if not (16 <= h <= 8192 and 32 <= w <= 8192):
        raise ValueError('Invalid fixture dimensions')
    item=manifest['expected'][name]
    rel=Path(item['file'])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe fixture path')
    f=root/rel
    data=f.read_bytes()
    if len(data)!=h*w*3 or len(data)!=int(item['bytes']) or hashlib.sha256(data).hexdigest()!=item['sha256']:
        raise ValueError('Fixture RGB payload size/hash mismatch: '+str(f))
    return np.frombuffer(data,dtype=np.uint8).reshape(h,w,3).copy()


def native_maps(prefix:Path, shape:tuple[int,int])->dict[str,np.ndarray]:
    out={}
    for eye in ('left','right'):
        for field,kind in (('disparity','<f4'),('validity','<u4')):
            file=Path(str(prefix)+f'.{eye}.{field}.'+('f32' if field=='disparity' else 'u32'))
            a=np.fromfile(file,dtype=kind)
            if a.size!=int(np.prod(shape)):
                raise ValueError(f'Incorrect native array shape: {file}')
            out[f'{eye}_{field}']=a.reshape(shape)
    return out


def cv_sgbm(left_rgb:np.ndarray,right_rgb:np.ndarray,*,disparities:int,block:int,uniqueness:int)->tuple[np.ndarray,np.ndarray,float]:
    if left_rgb.shape != right_rgb.shape or left_rgb.ndim!=3 or left_rgb.shape[2]!=3:
        raise ValueError('Images must be matching RGB')
    h,w=left_rgb.shape[:2]
    if disparities%16 or disparities<16 or disparities>256 or w<=disparities+block+4:
        raise ValueError('SGBM requires valid multiple-of-16 disparity range and wide images')
    if block not in (3,5,7,9,11) or not(0<=uniqueness<=30):
        raise ValueError('Invalid SGBM block or uniqueness setting')
    left=cv2.cvtColor(left_rgb,cv2.COLOR_RGB2GRAY)
    right=cv2.cvtColor(right_rgb,cv2.COLOR_RGB2GRAY)
    kwargs={'minDisparity':0,'numDisparities':disparities,'blockSize':block,
            'P1':8*block**2,'P2':32*block**2,'disp12MaxDiff':-1,
            'uniquenessRatio':uniqueness,'speckleWindowSize':0,'speckleRange':0,
            'preFilterCap':31,'mode':cv2.STEREO_SGBM_MODE_SGBM}
    matcher=cv2.StereoSGBM_create(**kwargs)
    t0=time.perf_counter()
    dl=matcher.compute(left,right).astype(np.float32)/16.
    # Right disparity must be positive: swap the cameras AND mirror image X.
    # Conventional SGBM always looks from its first image to x-d in the second.
    dr_flipped=matcher.compute(np.ascontiguousarray(right[:,::-1]),
                               np.ascontiguousarray(left[:,::-1])).astype(np.float32)/16.
    dr=np.ascontiguousarray(dr_flipped[:,::-1])
    elapsed=time.perf_counter()-t0
    return dl,dr,elapsed


def valid_map(d:np.ndarray, valid:np.ndarray|None=None, disparities:int=128)->np.ndarray:
    mask=np.isfinite(d)&(d>=1)&(d<disparities)
    if valid is not None:mask &= (valid!=0)
    return mask


def lr_pair(left:np.ndarray,right:np.ndarray,leftvalid:np.ndarray,rightvalid:np.ndarray,*,tol:float=1.)->tuple[np.ndarray,np.ndarray,dict]:
    if left.shape != right.shape or left.shape != leftvalid.shape or left.shape != rightvalid.shape:
        raise ValueError('LR dimensions differ')
    if left.ndim!=2 or not np.isfinite(tol) or tol<0:raise ValueError('Invalid LR input')
    h,w=left.shape
    yy,xx=np.indices((h,w),dtype=np.int32)
    # Input validity is passed explicitly; harden against invalid or nonfinite disparity.
    lv=np.asarray(leftvalid,bool)&np.isfinite(left)&(left>=1)
    rv=np.asarray(rightvalid,bool)&np.isfinite(right)&(right>=1)
    safe_d=np.where(lv,left,0).astype(np.float64)
    target_x=np.floor(xx.astype(np.float64)-safe_d+.5).astype(np.int32)
    inside=(target_x>=0)&(target_x<w)
    target=np.clip(target_x,0,w-1)
    opp=lv&inside&rv[yy,target]
    residual=np.where(opp,left-right[yy,target],np.nan)
    accepted=opp&(np.abs(residual)<=tol)
    arr=np.abs(residual[opp]);total=h*w
    stats={
        'left_selected_fraction':float(lv.mean()),'right_selected_fraction':float(rv.mean()),
        'opposite_selected_fraction_all_pixels':float(opp.mean()),
        'accepted_fraction_all_pixels':float(accepted.mean()),
        'accepted_fraction_among_opposite_selected':float(accepted.sum()/opp.sum()) if opp.any() else None,
        'abs_lr_error_p50_px':float(np.quantile(arr,.5)) if arr.size else None,
        'abs_lr_error_p90_px':float(np.quantile(arr,.9)) if arr.size else None,
        'count_compared':int(opp.sum()),'count_accepted':int(accepted.sum())
    }
    return accepted,opp,stats


def summarize_pair(left:np.ndarray, right:np.ndarray,lv:np.ndarray,rv:np.ndarray,*,tolerance:float=1.):
    return lr_pair(left,right,lv,rv,tol=tolerance)[2]


def compare_view(native:np.ndarray,reference:np.ndarray,native_valid:np.ndarray,ref_valid:np.ndarray,*,tol:float=1.) -> dict:
    support=native_valid&ref_valid
    difference=np.abs(native[support]-reference[support])
    return {'common_fraction':float(support.mean()),
            'abs_disparity_difference_p50_px':float(np.quantile(difference,.5)) if difference.size else None,
            'abs_disparity_difference_p90_px':float(np.quantile(difference,.9)) if difference.size else None,
            'agree_within_1px_fraction':float((difference<=tol).mean()) if difference.size else None}


def classify_texture(image:np.ndarray,native_l:np.ndarray,native_r:np.ndarray,lv:np.ndarray,rv:np.ndarray,
                     cv_l:np.ndarray,cv_r:np.ndarray,clv:np.ndarray,crv:np.ndarray)->dict:
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    f=gray.astype(np.float32)
    # Neighborhood standard deviation distinguishes smooth, weakly textured and structured regions.
    mean=cv2.boxFilter(f,-1,(9,9),normalize=True)
    square=cv2.boxFilter(f*f,-1,(9,9),normalize=True)
    texture=np.sqrt(np.maximum(square-mean*mean,0))
    na,_,_=lr_pair(native_l,native_r,lv,rv)
    ca,_,_=lr_pair(cv_l,cv_r,clv,crv)
    out={}
    for low,high in [(0,4),(4,8),(8,16),(16,32),(32,1000)]:
        mask=(texture>=low)&(texture<high)
        if mask.sum()<100:continue
        out[f'{low}:{high}']={'pixels':int(mask.sum()),'fraction_image':float(mask.mean()),
            'native_left_selected':float(lv[mask].mean()),'cv_left_selected':float(clv[mask].mean()),
            'native_lr_accepted':float(na[mask].mean()),'cv_lr_accepted':float(ca[mask].mean()),
            'native_lr_accepted_of_left_selected':float(na[mask].sum()/lv[mask].sum()) if lv[mask].any() else None,
            'cv_lr_accepted_of_left_selected':float(ca[mask].sum()/clv[mask].sum()) if clv[mask].any() else None}
    return out


def plot_diagnostic(image:np.ndarray,native_l:np.ndarray,native_r:np.ndarray,lv:np.ndarray,rv:np.ndarray,
                    cv_l:np.ndarray,cv_r:np.ndarray,clv:np.ndarray,crv:np.ndarray,path:Path):
    namask,_,_=lr_pair(native_l,native_r,lv,rv)
    camask,_,_=lr_pair(cv_l,cv_r,clv,crv)
    h,w=namask.shape
    # BGR panels: dim gray=unaccepted, green=LR accepted, blue=CV-only, orange=native-only.
    na=np.zeros((h,w,3),np.uint8);na[namask]=(0,215,0)
    ca=np.zeros((h,w,3),np.uint8);ca[camask]=(0,215,0)
    both=np.zeros((h,w,3),np.uint8)
    both[namask&camask]=(0,215,0)
    both[namask&~camask]=(0,155,255)
    both[~namask&camask]=(255,100,0)
    ims=[cv2.cvtColor(image,cv2.COLOR_RGB2BGR),na,ca,both]
    names=['Rectified left RGB','Native LR valid','OpenCV LR valid','Shared green / native orange / CV blue']
    for im,name in zip(ims,names):
        cv2.rectangle(im,(0,0),(min(w-1,620),30),(14,14,14),-1)
        cv2.putText(im,name,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    out=np.vstack([np.hstack(ims[:2]),np.hstack(ims[2:])]);path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),out):raise RuntimeError('Failed to save diagnostic image')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',required=True,type=Path)
    p.add_argument('--native-prefix',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--image',type=Path)
    p.add_argument('--disparities',type=int,default=128)
    p.add_argument('--blocks',type=int,nargs='+',default=[3,5])
    p.add_argument('--uniqueness',type=int,default=5)
    a=p.parse_args()
    if a.output.exists():p.error('output already exists; choose a new output path')
    fixture=json.loads((a.fixture/'manifest.json').read_text())
    left=rgb_from_fixture(a.fixture,'rect_left_rgb',fixture)
    right=rgb_from_fixture(a.fixture,'rect_right_rgb',fixture)
    shape=left.shape[:2]
    native=native_maps(a.native_prefix,shape)
    nl,nr=native['left_disparity'],native['right_disparity']
    nlv=valid_map(nl,native['left_validity'],a.disparities)
    nrv=valid_map(nr,native['right_validity'],a.disparities)
    result={'schema':'stereoforge.independent_stereo_crosscheck.v5',
        'warning':'OpenCV SGBM is a different estimator, NOT ground truth. High consistency need not imply correct depth.',
        'fixture':str(a.fixture),'native_prefix':str(a.native_prefix),
        'config':{'disparities':a.disparities,'blocks':a.blocks,'uniqueness':a.uniqueness,
                  'opencv_mode':'SGBM','speckle_filter':0,'opencv_lr_crosscheck':-1,'lr_tolerance_px':1},
        'native_native':summarize_pair(nl,nr,nlv,nrv), 'reference_profiles':{}}
    for block in a.blocks:
        cl,cr,seconds=cv_sgbm(left,right,disparities=a.disparities,block=block,uniqueness=a.uniqueness)
        clv=valid_map(cl,disparities=a.disparities);crv=valid_map(cr,disparities=a.disparities)
        name=f'block{block}_unique{a.uniqueness}'
        result['reference_profiles'][name]={
            'elapsed_cpu_seconds':seconds,
            'cv_cv':summarize_pair(cl,cr,clv,crv),
            'nativeL_cvR':summarize_pair(nl,cr,nlv,crv),
            'cvL_nativeR':summarize_pair(cl,nr,clv,nrv),
            'native_cv_left':compare_view(nl,cl,nlv,clv),
            'native_cv_right':compare_view(nr,cr,nrv,crv),
            'texture_bins':classify_texture(left,nl,nr,nlv,nrv,cl,cr,clv,crv)}
        if a.image and block==a.blocks[0]:
            plot_diagnostic(left,nl,nr,nlv,nrv,cl,cr,clv,crv,a.image)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Native/native LR {:.1%} accepted all; {:.1%} among both selected'.format(
        result['native_native']['accepted_fraction_all_pixels'],result['native_native']['accepted_fraction_among_opposite_selected'] or 0))
    for name,v in result['reference_profiles'].items():
        print(f'=== {name} (CPU {v["elapsed_cpu_seconds"]:.2f}s) ===')
        for mode in ('cv_cv','nativeL_cvR','cvL_nativeR'):
            x=v[mode]
            print('{:<14} accepted/all {:>6.1%} accepted/both {:>6.1%} compared {:>8}'.format(mode,x['accepted_fraction_all_pixels'],x['accepted_fraction_among_opposite_selected'] or 0,x['count_compared']))
        print('SGM/CV match left {:>5.1%} right {:>5.1%} (among jointly selected)'.format(
            v['native_cv_left']['agree_within_1px_fraction'] or 0,v['native_cv_right']['agree_within_1px_fraction'] or 0))
    print('Report:',a.output)
if __name__=='__main__':main()
