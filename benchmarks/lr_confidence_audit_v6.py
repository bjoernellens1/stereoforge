#!/usr/bin/env python3
"""StereoForge offline audit of native LR confidence and raw 5x5 Census evidence.

Never changes native disparity, no truth labels. Ranks stereo confidence and
checks whether raw Census costs are ambiguous at native selected disparities.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read_native(prefix: Path, shape: tuple[int,int]) -> dict:
    n = int(np.prod(shape))
    result = {}
    for eye in ('left','right'):
        for field, dtype, ext in [('disparity','<f4','f32'),('confidence','<f4','f32'),('validity','<u4','u32')]:
            p=Path(f'{prefix}.{eye}.{field}.{ext}')
            v=np.fromfile(p,dtype=dtype)
            if v.size!=n:
                raise ValueError(f'Unexpected size in {p}: {v.size} rather than {n}')
            result[f'{eye}_{field}']=v.reshape(shape)
    return result


def lr_pairs(left, right, lv, rv, lc, rc, tol=1.):
    if any(x.shape!=left.shape for x in (right,lv,rv,lc,rc)) or left.ndim!=2:
        raise ValueError('LR map dimensions mismatch')
    if not np.isfinite(tol) or tol<0:raise ValueError('Invalid LR tolerance')
    h,w=left.shape
    yy,xx=np.indices((h,w),dtype=np.int32)
    a=(lv!=0)&np.isfinite(left)&(left>0)
    b=(rv!=0)&np.isfinite(right)&(right>0)
    dest=np.floor(xx.astype(np.float64)-np.where(a,left,0).astype(np.float64)+.5).astype(np.int32)
    inside=(dest>=0)&(dest<w)
    t=np.clip(dest,0,w-1)
    both=a&inside&b[yy,t]
    score=np.minimum(np.where(np.isfinite(lc),lc,0),np.where(np.isfinite(rc[yy,t]),rc[yy,t],0))
    residual=np.where(both,left-right[yy,t],np.nan)
    accepted=both&(np.abs(residual)<=tol)
    return both, accepted, score, residual


def confidence_report(both, accepted, confidence, residual):
    if both.shape!=accepted.shape or both.shape!=confidence.shape or both.shape!=residual.shape:
        raise ValueError('shape mismatch')
    vals=np.maximum(confidence[both],0)
    diff=np.abs(residual[both])
    if vals.size==0:raise ValueError('No comparable stereo pairs')
    truth=np.asarray(accepted[both],bool)
    rows=[]
    # Quantile cuts represent observed pair-confidence coverage, not hypothetical reruns.
    for retained in (1., .9, .75, .5, .25, .1, .05):
        cutoff=float(np.quantile(vals,1-retained))
        mask=vals>=cutoff
        # Ties may give more retained pixels than nominal.
        rows.append({'target_retention':retained,'confidence_cutoff':cutoff,
          'actual_pair_retention':float(mask.mean()),'fraction_all_pixels_compared':float(mask.sum()/both.size),
          'fraction_all_pixels_accepted':float(np.count_nonzero(truth&mask)/both.size),
          'lr_agreement_given_comparable':float(np.mean(truth[mask])) if mask.any() else None,
          'severe_gt10px_fraction':float(np.mean(diff[mask]>10)) if mask.any() else None,
          'residual_p50_px':float(np.quantile(diff[mask],.5)) if mask.any() else None,
          'residual_p90_px':float(np.quantile(diff[mask],.9)) if mask.any() else None})
    return {'compared_pixels':int(both.sum()),
        'native_lr_accepted_all':float(accepted.mean()),
        'native_lr_accepted_comparable':float(accepted.sum()/both.sum()),
        'confidence_median_accepted':float(np.median(vals[truth])) if truth.any() else None,
        'confidence_median_rejected':float(np.median(vals[~truth])) if (~truth).any() else None,
        'confidence_sweep':rows}


def read_fixture_luma(folder:Path):
    meta=json.loads((folder/'manifest.json').read_text())
    shape=meta.get('rectified') or meta.get('rectification')
    if not isinstance(shape,dict):raise ValueError('rectified dimensions missing')
    h,w=int(shape['height']),int(shape['width'])
    out=[]
    for eye in ('left','right'):
        entry=meta['expected'][eye+'_luma']
        rel=Path(entry['file'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe fixture path')
        b=(folder/rel).read_bytes()
        if len(b)!=h*w*4 or len(b)!=entry['bytes'] or hashlib.sha256(b).hexdigest()!=entry['sha256']:
            raise ValueError('luma fixture hash/size mismatch')
        out.append(np.frombuffer(b,dtype='<u4').reshape(h,w).copy())
    return out


def census_5x5(luma):
    """Exact index order and strict less-than defined in shaders/ops/sgm.slang."""
    if luma.ndim!=2 or luma.shape[0]<5 or luma.shape[1]<5:raise ValueError('luma too small')
    h,w=luma.shape
    out=np.zeros((h,w),dtype=np.uint32)
    center=luma[2:h-2,2:w-2]
    bit=0
    for dy in range(-2,3):
        for dx in range(-2,3):
            if dy==0 and dx==0:continue
            out[2:h-2,2:w-2] |= (luma[2+dy:h-2+dy,2+dx:w-2+dx]<center).astype(np.uint32)<<bit
            bit+=1
    return out


def census_winner(ref,match,ys,xs,D,eye='left'):
    if ref.shape!=match.shape or ref.ndim!=2:raise ValueError('Census shape mismatch')
    h,w=ref.shape
    ys=np.asarray(ys,dtype=np.int32)
    xs=np.asarray(xs,dtype=np.int32)
    if ys.shape!=xs.shape or ys.ndim!=1:raise ValueError('sample coordinate mismatch')
    if eye not in ('left','right') or D<2 or D>256:raise ValueError('bad Census settings')
    n=ys.size
    best=np.full(n,255,np.int16)
    second=np.full(n,255,np.int16)
    best_d=np.zeros(n,np.int16)
    refval=ref[ys,xs]
    for d in range(1,D):
        target=xs+(-d if eye=='left' else d)
        ok=(target>=2)&(target<w-2)&(ys>=2)&(ys<h-2)
        cost=np.full(n,255,dtype=np.int16)
        idx=np.flatnonzero(ok)
        if idx.size:
            xor=np.bitwise_xor(refval[idx],match[ys[idx],target[idx]])
            cost[idx]=np.bitwise_count(xor).astype(np.int16)
        improved=cost<best
        second=np.where(improved,best,np.minimum(second,cost))
        best_d=np.where(improved,d,best_d)
        best=np.minimum(cost,best)
    valid=(best<=24)&(second>best)
    return best_d,best,second,valid


def census_audit(luma_left,luma_right, native, disparities, stride):
    if stride<1 or disparities<2:raise ValueError('invalid audit settings')
    l=census_5x5(luma_left)
    r=census_5x5(luma_right)
    h,w=l.shape
    yy,xx=np.mgrid[2:h-2:stride,2:w-2:stride]
    ys=yy.reshape(-1);xs=xx.reshape(-1)
    ld,lb,ls,lv=census_winner(l,r,ys,xs,disparities,'left')
    rd,rb,rs,rv=census_winner(r,l,ys,xs,disparities,'right')
    nld=native['left_disparity'][ys,xs];nrd=native['right_disparity'][ys,xs]
    nlvalid=(native['left_validity'][ys,xs]!=0)&np.isfinite(nld)
    nrvalid=(native['right_validity'][ys,xs]!=0)&np.isfinite(nrd)
    def per_eye(raw_d, best, second, unique, native_d, valid):
        support=unique&valid
        report={'samples':int(len(best)),'raw_census_unique_fraction':float(unique.mean()),
                'raw_census_tied_fraction':float(((best<255)&(second==best)).mean()),
                'raw_census_invalid_fraction':float((best==255).mean()),
                'native_valid_fraction':float(valid.mean()),
                'joint_unique_valid_fraction':float(support.mean()),
                'native_equals_raw_wta_1px_fraction':float((np.abs(native_d[support]-raw_d[support])<=1).mean()) if support.any() else None,
                'raw_cost_min_p50':float(np.median(best[best<255])) if np.any(best<255) else None}
        return report
    # Compare independently computed local Census disparities at correctly indexed right target.
    dest=xs-ld
    inbounds=lv&(dest>=2)&(dest<w-2)
    rhs=ys[inbounds];rxs=dest[inbounds].astype(np.int32)
    right_target,_,_,right_target_valid=census_winner(r,l,rhs,rxs,disparities,'right')
    agree=(np.abs(ld[inbounds]-right_target)<=1)&right_target_valid
    return {'stride':stride,'samples':int(len(ys)),
            'left':per_eye(ld,lb,ls,lv,nld,nlvalid),
            'right':per_eye(rd,rb,rs,rv,nrd,nrvalid),
            'raw_census_lr_comparable':int(np.count_nonzero(inbounds&lv)),
            'raw_census_lr_confirmed_fraction_among_unique_left':float(agree.sum()/lv.sum()) if lv.any() else None,
            'note':'Raw single-pixel 5x5 Census WTA is NOT SGM or ground truth. Tied minima indicate matching ambiguity.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',required=True,type=Path)
    p.add_argument('--native-prefix',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--disparities',type=int,default=128)
    p.add_argument('--stride',type=int,default=8)
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists')
    left,right=read_fixture_luma(a.fixture)
    native=read_native(a.native_prefix,left.shape)
    both,accepted,score,residual=lr_pairs(native['left_disparity'],native['right_disparity'],native['left_validity'],native['right_validity'],native['left_confidence'],native['right_confidence'])
    result={'schema':'stereoforge.lr_confidence_and_census_audit.v6',
      'warning':'Internal LR agreement/unique Census cost is not metric ground truth. Confidence filtering can discard valid fine structure.',
      'fixture':str(a.fixture),'prefix':str(a.native_prefix),'config':{'disparities':a.disparities,'stride':a.stride},
      'confidence':confidence_report(both,accepted,score,residual),
      'census':census_audit(left,right,native,a.disparities,a.stride)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Native LR accepted among comparable: {:.1%}'.format(result['confidence']['native_lr_accepted_comparable']))
    print('retained%  cutoff    agree%  compared/all  accepted/all  abs residual p90')
    for x in result['confidence']['confidence_sweep']:
        print('{:>8.0%} {:>8.4f} {:>7.1%} {:>12.1%} {:>13.1%} {:>15.2f}'.format(x['actual_pair_retention'],x['confidence_cutoff'],x['lr_agreement_given_comparable'],x['fraction_all_pixels_compared'],x['fraction_all_pixels_accepted'],x['residual_p90_px']))
    for eye in ('left','right'):
        c=result['census'][eye]
        print(f'{eye} Census WTA unique={c["raw_census_unique_fraction"]:.1%} tied={c["raw_census_tied_fraction"]:.1%} native-vs-WTA={c["native_equals_raw_wta_1px_fraction"]}')
    print('JSON:',a.output)

if __name__=='__main__':main()
