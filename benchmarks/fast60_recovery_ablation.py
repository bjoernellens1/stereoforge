#!/usr/bin/env python3
"""CPU research ablation: coarse, anchors, edge-aware 2D, sparse local rescue.

No truth/risk labels enter generation. Report includes *all* candidate work,
including unsuccessful sparse-search probes; no GPU FPS claim is made.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from fast60_anchor_hypotheses import AnchorConfig, match_anchors, census5, _hamming24

SCENES=('thin','occlusion','low_texture','repetitive','slanted')


def add(candidates, values, mask, cap):
    """Append a vectorized disparity proposal once per pixel (bounded, unique)."""
    h,w,n=candidates.shape
    m=mask & (values>=0) & np.isfinite(values)
    if not m.any(): return
    v=values.astype(np.int16,copy=False)
    occupied=(candidates==v[:,:,None]).any(axis=2)
    room=(candidates<0).any(axis=2)
    yy,xx=np.where(m & ~occupied & room)
    if not len(yy):return
    slot=(candidates[yy,xx]<0).argmax(axis=1)
    candidates[yy,xx,slot]=v[yy,xx]


def generate(left,right,coarse,*,mode,disparities=32,stride=4,radius=2,spread=8,
             cap=12,edge_threshold=60,rescue_stride=4,rescue_max_hamming=8,
             rescue_uniqueness=3):
    left=np.asarray(left);right=np.asarray(right);coarse=np.asarray(coarse)
    if left.ndim!=2 or left.dtype!=np.uint8 or right.shape!=left.shape or right.dtype!=np.uint8 or coarse.shape!=left.shape:
        raise ValueError('expected matching 2D images and coarse disparity')
    if mode not in ('coarse','anchors','propagate','rescue') or not(1<=disparities<=256 and 0<=radius<=16 and 0<=spread<=64 and 1<=cap<=64 and 1<=stride<=64 and 1<=rescue_stride<=64):
        raise ValueError('invalid configuration')
    h,w=left.shape
    candidates=np.full((h,w,cap),-1,dtype=np.int16)
    source=np.zeros((h,w),dtype=np.uint8)
    evidence={'coarse_evaluations':0,'anchor_evaluations':0,'rescue_evaluations':0,'anchors':0,'rescue_accepted':0}
    valid=np.isfinite(coarse)&(coarse>=0)&(coarse<disparities)
    center=np.zeros((h,w),dtype=np.int16)
    center[valid]=np.floor(coarse[valid]+.5).astype(np.int16)
    for offset in range(-radius,radius+1):
        values=center+offset
        mask=valid & (values>=0)&(values<disparities)
        evidence['coarse_evaluations']+=int(mask.sum())
        add(candidates,values,mask,cap)
    source[valid]=1
    if mode=='coarse':return candidates,source,evidence
    anchors=match_anchors(left,right,AnchorConfig(disparities=disparities,stride=stride))
    evidence['anchors']=anchors.count;evidence['anchor_evaluations']=anchors.evaluated
    seeds=np.full((h,w),-1,dtype=np.int16)
    seeds[anchors.y,anchors.left_x]=anchors.disparity.astype(np.int16)
    seedmask=seeds>=0
    add(candidates,seeds,seedmask,cap)
    source[seedmask]=2
    if mode=='anchors':return candidates,source,evidence
    # Propagate independent seed proposals through 2D 4-neighbour image paths.
    # Limit distance and cut propagation at strong intensity discontinuities.
    frontier=seeds.copy()
    for _ in range(spread):
        next_frontier=np.full_like(frontier,-1)
        for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
            y0=max(0,dy);y1=h+min(0,dy);x0=max(0,dx);x1=w+min(0,dx)
            sy0=y0-dy;sy1=y1-dy;sx0=x0-dx;sx1=x1-dx
            v=frontier[sy0:sy1,sx0:sx1]
            compatible=np.abs(left[y0:y1,x0:x1].astype(np.int16)-left[sy0:sy1,sx0:sx1].astype(np.int16))<=edge_threshold
            target=next_frontier[y0:y1,x0:x1]
            ok=(v>=0)&compatible&(target<0)
            target[ok]=v[ok]
        # Seeds cannot be replaced; propagation can reach the same pixel from different paths.
        todo=(next_frontier>=0)
        add(candidates,next_frontier,todo,cap)
        source[(source==0)&todo]=3
        frontier=next_frontier
        if not todo.any():break
    if mode=='propagate':return candidates,source,evidence
    # Sparse full-resolution independent re-search where evidence is absent or
    # image structures indicate a discontinuity. No access to true disparity.
    ld,lv,grad=census5(left);rd,rv,_=census5(right)
    yy,xx=np.indices((h,w))
    uncertain=(~valid)|(grad>=80)|((candidates>=0).sum(axis=2)<=radius*2+1)
    probe=uncertain & lv & (grad>=12) & (xx%rescue_stride==2%rescue_stride) & (yy%rescue_stride==2%rescue_stride)
    ys,xs=np.where(probe)
    for y,x in zip(ys,xs):
        lo=max(2,int(x)-disparities+1)
        rx=np.arange(lo,int(x)+1,dtype=np.int32)
        support=rv[y,rx]
        if support.sum()<2:continue
        cost=_hamming24(ld[y,x]^rd[y,rx]);cost[~support]=255
        evidence['rescue_evaluations']+=int(support.sum())
        best=np.argpartition(cost,1)[:2];best=best[np.argsort(cost[best])]
        if int(cost[best[0]])>rescue_max_hamming or int(cost[best[1]])-int(cost[best[0]])<rescue_uniqueness:continue
        d=int(x-rx[best[0]])
        for delta in range(-radius,radius+1):
            v=d+delta
            if 0<=v<disparities:
                proposal=np.full((1,1),v,dtype=np.int16)
                # Explicit per-pixel cap/uniqueness without allocating a full plane.
                slots=candidates[y,x]
                if v not in slots and (slots<0).any():slots[np.flatnonzero(slots<0)[0]]=v
        source[y,x]=4;evidence['rescue_accepted']+=1
    return candidates,source,evidence


def evaluate(candidates,source,evidence,truth,risk,disparities):
    support=np.isfinite(truth)&(truth>=0)&(truth<disparities)
    recall=support&np.any((candidates>=0)&(np.abs(candidates-truth[:,:,None])<=1),axis=2)
    count=(candidates>=0).sum(axis=2)
    def region(mask):
        n=int(mask.sum());return {'pixels':n,'recalled':int((recall&mask).sum()),'recall':float((recall&mask).sum()/n) if n else None}
    work=sum(evidence[k] for k in ('coarse_evaluations','anchor_evaluations','rescue_evaluations'))
    return {'overall':region(support),'risk':region(support & risk), 'non_risk':region(support & ~risk),
            'mean_candidates':float(count.mean()),'retained_candidates':int(count.sum()),
            'measured_cpu_candidate_evaluations':int(work),'evaluations_vs_dense':float(work/(truth.size*disparities)),
            'retained_vs_dense':float(count.sum()/(truth.size*disparities)), 'sources':evidence}


def run(input_dir,output,**kw):
    results=[]
    for name in SCENES:
        with np.load(Path(input_dir)/(name+'.npz'),allow_pickle=False) as d:
            left,right,coarse,truth,risk=(d[k].copy() for k in ('left','right','coarse_disparity','truth_disparity','risk_mask'))
        row={'scene':name}
        for mode in ('coarse','anchors','propagate','rescue'):
            c,s,e=generate(left,right,coarse,mode=mode,**kw)
            row[mode]=evaluate(c,s,e,truth,risk,kw['disparities'])
        results.append(row)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'schema':'stereoforge.fast60_recovery_ablation.v1','config':kw,'scenes':results},indent=2)+'\n')
    print('scene       mode        overall   risk    retained/D  evaluated/D')
    for row in results:
        for mode in ('coarse','anchors','propagate','rescue'):
            r=row[mode]
            print(f"{row['scene']:<11} {mode:<11} {r['overall']['recall']:>6.1%} {r['risk']['recall']:>6.1%} {r['retained_vs_dense']:>10.1%} {r['evaluations_vs_dense']:>11.1%}")
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--disparities',type=int,default=32);p.add_argument('--stride',type=int,default=4)
    p.add_argument('--radius',type=int,default=2);p.add_argument('--spread',type=int,default=8)
    p.add_argument('--cap',type=int,default=12);p.add_argument('--edge-threshold',type=int,default=60)
    p.add_argument('--rescue-stride',type=int,default=4)
    a=p.parse_args()
    run(a.input_dir,a.output,disparities=a.disparities,stride=a.stride,radius=a.radius,spread=a.spread,
        cap=a.cap,edge_threshold=a.edge_threshold,rescue_stride=a.rescue_stride)
if __name__=='__main__':main()
