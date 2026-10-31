"""CPU-only compact disparity candidate experiment; truth never enters generation."""
import argparse
import json
from pathlib import Path
import numpy as np
from fast60_anchor_hypotheses import AnchorConfig, match_anchors


def generate(left, right, coarse, *, disparities=32, stride=4,
             radius=2, spread=8, edge_threshold=60, cap=12, max_hamming=8):
    left=np.asarray(left); right=np.asarray(right); coarse=np.asarray(coarse)
    if left.ndim != 2 or left.dtype != np.uint8 or right.shape != left.shape or right.dtype != np.uint8 or coarse.shape != left.shape:
        raise ValueError('expected equal-sized HxW uint8 stereo and coarse disparity plane')
    if not (1 <= disparities <= 256 and 0 <= radius <= 16 and 0 <= spread <= 64 and 1 <= cap <= 64 and 0 <= edge_threshold <= 255):
        raise ValueError('invalid candidate configuration')
    h,w=left.shape
    if h<5 or w<5: raise ValueError('image too small')
    anchors=match_anchors(left,right,AnchorConfig(disparities=disparities,stride=stride,max_hamming=max_hamming))
    # Each candidate is an independent disparity index. -1 means unused.
    out=np.full((h,w,cap),-1,dtype=np.int16)
    counts=np.zeros((h,w),dtype=np.uint8)
    def add(y,x,d):
        if not (0<=d<disparities): return
        n=int(counts[y,x]); slot=out[y,x]
        if d in slot[:n] or n>=cap:return
        slot[n]=d;counts[y,x]=n+1
    # Coarse priors get a compact local search interval.
    yy,xx=np.where(np.isfinite(coarse)&(coarse>=0)&(coarse<disparities))
    for y,x in zip(yy,xx):
        center=int(np.floor(float(coarse[y,x])+.5))
        for d in range(center-radius,center+radius+1):add(y,x,d)
    # A source anchor can propagate on its scanline only while the centre-to-
    # centre image path remains locally smooth. Stop at contrast boundaries.
    for x,y,d in zip(anchors.left_x,anchors.y,anchors.disparity):
        x=int(x);y=int(y);d=int(d)
        add(y,x,d)
        for direction in (-1,1):
            previous=x
            for step in range(1,spread+1):
                col=x+direction*step
                if col<2 or col>=w-2:break
                if abs(int(left[y,col])-int(left[y,previous]))>edge_threshold:break
                add(y,col,d)
                previous=col
    return out,counts,anchors


def evaluate(candidates,truth,*,risk=None,tolerance=1):
    arr=np.asarray(candidates);truth=np.asarray(truth)
    if arr.ndim!=3 or arr.shape[:2]!=truth.shape or arr.dtype.kind not in 'iu':raise ValueError('shape/dtype mismatch')
    support=np.isfinite(truth)&(truth>=0)
    match=np.any((arr>=0)&(np.abs(arr-truth[:,:,None])<=tolerance),axis=2)&support
    count=np.sum(arr>=0,axis=2)
    def scope(mask):
        n=int(mask.sum()); return {'truth_pixels':n,'recalled_pixels':int((match&mask).sum()),'candidate_recall':float((match&mask).sum()/n) if n else None,
            'mean_candidates_per_pixel':float(count[mask].mean()) if n else None}
    report={'overall':scope(support),'candidate_evaluations':int(count.sum()),'mean_candidates_all_pixels':float(count.mean()),'max_candidates':int(count.max()),
            'max_capacity_candidate_slots':int(np.prod(truth.shape)*arr.shape[2])}
    if risk is not None:
        risk=np.asarray(risk)
        if risk.shape!=truth.shape or risk.dtype!=np.bool_:raise ValueError('risk mask mismatch')
        report['risk']=scope(support&risk)
        report['non_risk']=scope(support&~risk)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--disparities',type=int,default=32);p.add_argument('--stride',type=int,default=4)
    p.add_argument('--radius',type=int,default=2);p.add_argument('--spread',type=int,default=8)
    p.add_argument('--cap',type=int,default=12);p.add_argument('--edge-threshold',type=int,default=60)
    a=p.parse_args()
    with np.load(a.input,allow_pickle=False) as data:
        required={'left','right','coarse_disparity','truth_disparity'}
        if not required.issubset(data.files):raise ValueError('input missing required arrays')
        left=data['left'];right=data['right'];coarse=data['coarse_disparity'];truth=data['truth_disparity'];risk=data['risk_mask'] if 'risk_mask' in data else None
    out,counts,anchors=generate(left,right,coarse,disparities=a.disparities,stride=a.stride,radius=a.radius,
                                spread=a.spread,cap=a.cap,edge_threshold=a.edge_threshold)
    report=evaluate(out,truth,risk=risk)
    report['schema']='stereoforge.fast60_multi_hypothesis.v1'
    report['config']={'disparities':a.disparities,'stride':a.stride,'radius':a.radius,'spread':a.spread,'cap':a.cap,'edge_threshold':a.edge_threshold}
    report['anchor_count']=anchors.count
    report['dense_search_work_reference']=int(truth.size*a.disparities)
    report['candidate_fraction_of_dense']=report['candidate_evaluations']/report['dense_search_work_reference']
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('anchor_count','overall','risk','mean_candidates_all_pixels','candidate_fraction_of_dense') if k in report},indent=2))
if __name__=='__main__':main()
