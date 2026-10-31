#!/usr/bin/env python3
"""Audit existing daxa_depth_benchmark --dump-prefix output.

No production changes. Report selection -> correspondence -> LR -> metric
losses, LR threshold sensitivity, and repeatability on identical fixture bytes.
A higher acceptance rate is NOT a geometric accuracy improvement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def read(prefix: Path, suffix, dtype, count):
    p=Path(str(prefix)+suffix)
    if not p.is_file() or p.stat().st_size != count*np.dtype(dtype).itemsize:
        raise ValueError(f'missing/wrong-sized benchmark dump: {p}')
    return np.fromfile(p,dtype)


def pct(a):return float(np.mean(a))


def qtile(a):
    a = np.asarray(a)[np.isfinite(a)]
    return {str(k):float(np.percentile(a,k)) for k in (10,50,90)} if len(a) else None


def classify_lr(own, own_valid, other, other_valid, tolerance, *, right=False):
    """Nearest-coordinate LR rule; report exclusive rejection categories."""
    if own.shape != other.shape or own.shape != own_valid.shape or own.shape != other_valid.shape:
        raise ValueError('stereo planes have different shapes')
    if own.ndim!=2 or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('expected 2D arrays and nonnegative finite tolerance')
    h,w=own.shape
    yy,xx=np.indices((h,w))
    finite=np.isfinite(own)
    safe=np.where(finite,own,0).astype(np.float64)
    target=np.floor(xx+(safe if right else -safe)+.5)
    inside=finite & (target>=0) & (target<w)
    # Bounds are checked before safe integer conversion, including extreme finite disparity.
    tx=np.where(inside,target,0).astype(np.intp)
    ov=other_valid[yy,tx].astype(bool) & np.isfinite(other[yy,tx])
    with np.errstate(invalid='ignore',over='ignore'):
        residual=np.abs(own-other[yy,tx])
    valid=own_valid.astype(bool)
    accepted=valid & inside & ov & (residual<=tolerance)
    categories={
        'selection_invalid':~valid,
        'selection_valid_nonfinite_disparity':valid & ~finite,
        'correspondence_out_of_bounds':valid & finite & ~inside,
        'correspondence_other_invalid':valid & inside & ~ov,
        'lr_disagreement':valid & inside & ov & (residual>tolerance),
        'lr_accepted':accepted,
    }
    total=np.zeros_like(own_valid,dtype=np.uint8)
    for mask in categories.values():total+=mask
    if not np.all(total==1):raise ValueError('LR audit categories do not form an exclusive partition')
    return accepted,categories,residual


def audit_one(path: Path):
    report=json.loads(path.read_text())
    if report.get('schema')!='stereoforge.daxa_depth.v1':
        raise ValueError('expected native daxa_depth_benchmark JSON')
    w,h=report['width'],report['height']
    if any(type(v)is not int or not 1<=v<=8192 for v in (w,h)):
        raise ValueError('invalid benchmark dimensions')
    if report.get('layers')!='left_then_right':raise ValueError('unexpected native layer order')
    prefix=path.with_suffix('');n=w*h;tol=float(report['lr_tolerance'])
    disp=[read(prefix,f'.{eye}.disparity.f32','<f4',n).reshape(h,w) for eye in ('left','right')]
    valid=[read(prefix,f'.{eye}.validity.u32','<u4',n).reshape(h,w)!=0 for eye in ('left','right')]
    confidence=[read(prefix,f'.{eye}.confidence.f32','<f4',n).reshape(h,w) for eye in ('left','right')]
    lr=read(prefix,'.lr_validity.u32','<u4',2*n).reshape(2,h,w)!=0
    metric=read(prefix,'.metric_validity.u32','<u4',2*n).reshape(2,h,w)!=0
    depth=read(prefix,'.depth.f32','<f4',2*n).reshape(2,h,w)
    result={'source_report':str(path),'scope':report.get('scope'),
            'width':w,'height':h,'disparities':report['disparities'],
            'p1':report['p1'],'p2':report['p2'],'paths':report['paths'],
            'lr_tolerance':tol,'source_id':report.get('source_id'),'eyes':{}}
    for eye,name in enumerate(('left','right')):
        accepted,cats,residual=classify_lr(disp[eye],valid[eye],disp[1-eye],valid[1-eye],tol,right=bool(eye))
        sweep={}
        for t in (.5,1.,2.,4.):
            a,_,_=classify_lr(disp[eye],valid[eye],disp[1-eye],valid[1-eye],t,right=bool(eye))
            sweep[str(t)]=pct(a)
        result['eyes'][name]={
            'selection_valid_fraction':pct(valid[eye]),
            'exclusive_lr_categories_fraction':{k:pct(v) for k,v in cats.items()},
            'native_lr_valid_fraction':pct(lr[eye]),
            'native_metric_valid_fraction':pct(metric[eye]),
            'lr_to_metric_rejected_fraction_all_pixels':pct(lr[eye] & ~metric[eye]),
            'metric_without_lr_pixels':int((metric[eye] & ~lr[eye]).sum()),
            'cpu_lr_mask_disagrees_with_native_pixels':int((accepted ^ lr[eye]).sum()),
            'nonfinite_or_nonpositive_depth_marked_valid_pixels':int((metric[eye] & (~np.isfinite(depth[eye]) | (depth[eye]<=0))).sum()),
            'selection_high_disparity_fraction_among_selected':float((disp[eye][valid[eye]]>=report['disparities']-1.5).mean()) if valid[eye].any() else None,
            'lr_tolerance_sweep_acceptance_not_accuracy':sweep,
            'confidence_on_lr_kept':qtile(confidence[eye][valid[eye]&lr[eye]]),
            'confidence_on_lr_rejected':qtile(confidence[eye][valid[eye]&~lr[eye]]),
            'note':'Remap/Census tap support and SGM cost volumes are not exported here; metric loss cannot be fully attributed without those planes.'}
    # Fingerprint stage outputs; repeated runs must have identical input/config before comparison.
    hashes={}
    suffixes=['.left.disparity.f32','.right.disparity.f32','.left.validity.u32','.right.validity.u32',
              '.lr_validity.u32','.metric_validity.u32','.depth.f32']
    for suffix in suffixes:
        hashes[suffix]=hashlib.sha256(Path(str(prefix)+suffix).read_bytes()).hexdigest()
    result['plane_sha256']=hashes
    identity={k:report.get(k) for k in ('width','height','disparities','p1','p2','paths','subpixel','required_wave_size','lr_tolerance',
                                       'metric_calibration','source_id','source_metadata','scope','slots','record_ordinal')}
    # Capture hashes/manifest/calibration identity may differ by existing report version.
    for k,v in report.items():
        if any(tag in k for tag in ('sha256','identity','schedule','reference_mode')):identity[k]=v
    return result,identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.output.exists():raise FileExistsError(a.output)
        results=[];ids=[]
        for path in a.reports:
            row,identity=audit_one(path);results.append(row);ids.append(identity)
        comparable=all(i==ids[0] for i in ids[1:])
        determinism=None
        if len(results)>1 and comparable:
            determinism={k:all(r['plane_sha256'][k]==results[0]['plane_sha256'][k] for r in results[1:])
                         for k in results[0]['plane_sha256']}
        output={'schema':'stereoforge.stereo_stage_audit.v2','runs':results,
                'same_reported_input_and_config':comparable,
                'bitwise_repeatability_by_stage':determinism,
                'warning':'This checks final snapshots across repeated benchmark invocations, not every frame or all scheduling patterns. No stereo ground truth is provided.'}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
        for row in results:
            print(row['source_report'])
            for name,e in row['eyes'].items():
                print(' ',name,'selection={:.1%} LR={:.1%} metric={:.1%} CPU-LR-mask-mismatches={}'.format(
                    e['selection_valid_fraction'],e['native_lr_valid_fraction'],e['native_metric_valid_fraction'],e['cpu_lr_mask_disagrees_with_native_pixels']))
                print('   LR rejection partition:',e['exclusive_lr_categories_fraction'])
        print('Repeated input/config comparable:',comparable)
        print('Bitwise repeated snapshots:',determinism)
    except (ValueError,OSError,KeyError) as e:p.exit(2,f'ERROR: {e}\n')


if __name__=='__main__':main()
