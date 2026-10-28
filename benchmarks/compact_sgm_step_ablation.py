"""Offline reference and runner interface for sustained-step directional P2."""
from pathlib import Path
import subprocess
import numpy as np
from benchmarks.compact_sgm_reference import DIRECTIONS, _inputs
from benchmarks.compact_sgm_gradient_ablation import run_eye  # same image-mode wire format


def step_paths(cost, lo, hi, image, support, p1=8, p2=32):
    cost, lo, hi = _inputs(cost, lo, hi, p1, p2)
    image, support = np.asarray(image), np.asarray(support)
    if image.shape != cost.shape[:2] or image.dtype != np.uint8 or support.shape != image.shape or support.dtype != np.bool_:
        raise ValueError('expected uint8 luma and boolean support matching costs')
    h, w, d = cost.shape
    paths = np.full((4,h,w,d), np.inf)
    for k,(dx,dy) in enumerate(DIRECTIONS):
        for y in (range(h) if dy >= 0 else range(h-1,-1,-1)):
            for x in (range(w) if dx >= 0 else range(w-1,-1,-1)):
                py,px=y-dy,x-dx
                previous=paths[k,py,px] if 0<=py<h and 0<=px<w else None
                minimum=min(previous) if previous is not None else np.inf
                penalty=p2
                coords=((y-2*dy,x-2*dx),(py,px),(y,x),(y+dy,x+dx))
                if all(0<=cy<h and 0<=cx<w and support[cy,cx] for cy,cx in coords):
                    b,a,c,e=(int(image[cy,cx]) for cy,cx in coords)
                    if abs(c-a)>=64 and abs(a-b)<=16 and abs(e-c)<=16: penalty=p1
                for q in range(int(lo[y,x]),int(hi[y,x])):
                    c=cost[y,x,q]
                    if not np.isfinite(c): continue
                    if not np.isfinite(minimum): paths[k,y,x,q]=c
                    else: paths[k,y,x,q]=c+min(previous[q],previous[q-1]+p1 if q else np.inf,previous[q+1]+p1 if q+1<d else np.inf,minimum+penalty)-minimum
    return paths


def run_compiled(runner,directory,cost,lo,hi,image,support):
    cost,lo,hi=_inputs(cost,lo,hi,8,32)
    image,support=np.asarray(image),np.asarray(support)
    if image.dtype != np.uint8 or support.dtype != np.bool_ or image.shape != cost.shape[:2] or support.shape != image.shape: raise ValueError('invalid luma or support')
    h,w,d=cost.shape;root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    wire=np.full(cost.shape,np.iinfo(np.uint32).max,dtype='<u4');finite=np.isfinite(cost);wire[finite]=cost[finite].astype(np.uint32);wire.tofile(root/'cost')
    for name,a in [('lo',lo),('hi',hi),('luma',image),('own_support',support)]: np.asarray(a,dtype='<u4').tofile(root/name)
    subprocess.run([str(runner),str(root),str(h),str(w),str(d),'8','32'],check=True)
    def load(name,shape):
        a=np.fromfile(root/name,dtype='<u4').reshape(shape).astype(float);a[a==np.iinfo(np.uint32).max]=np.inf;return a
    return load('paths',(4,h,w,d)),load('sums',(h,w,d)),np.fromfile(root/'winners',dtype='<i4').reshape(h,w)
