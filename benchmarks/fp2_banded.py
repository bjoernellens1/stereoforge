"""Offline FP2 runner interface; dense unpacking is test-diagnostic only."""
from pathlib import Path
import json,os,subprocess
import numpy as np
from benchmarks.compact_sgm_reference import _inputs

def buckets(width):
    width=np.asarray(width)
    if not np.issubdtype(width.dtype,np.integer) or np.any(width<0) or np.any(width>128):raise ValueError('invalid band width')
    return np.select([width==0,width<=16,width<=32,width<=64],[0,16,32,64],default=128).astype(np.uint32)

def _prepare(root,lo,hi,shape,d):
    lo,hi=np.asarray(lo),np.asarray(hi)
    if lo.shape!=shape or hi.shape!=shape or not np.issubdtype(lo.dtype,np.integer) or not np.issubdtype(hi.dtype,np.integer) or np.any(lo<0) or np.any(lo>hi) or np.any(hi>d) or not 1<=d<=128:raise ValueError('invalid bounds')
    cap=buckets(hi-lo);offset=np.r_[0,np.cumsum(cap.ravel(),dtype=np.uint64)]
    np.asarray(lo,dtype='<u4').tofile(root/'lo');np.asarray(hi,dtype='<u4').tofile(root/'hi');return offset

def _read(root,lo,hi,d,dump):
    h,w=lo.shape;n=h*w;offset=np.fromfile(root/'offset',dtype='<u8');cap=buckets(hi-lo).ravel();expected=np.r_[0,np.cumsum(cap,dtype=np.uint64)]
    if not np.array_equal(offset,expected):raise ValueError('bad offset output')
    result={'winner':np.fromfile(root/'winner',dtype='<i4').reshape(h,w),'stats':json.loads((root/'stats.json').read_text())}
    for k in ('best','second','count'):result[k]=np.fromfile(root/k,dtype='<u4').reshape(h,w)
    if dump:
        total=int(offset[-1]);packed=np.fromfile(root/'paths',dtype='<u4').reshape(4,total);sums=np.fromfile(root/'sums',dtype='<u4');paths=np.full((4,h,w,d),np.inf);dense_sums=np.full((h,w,d),np.inf)
        for i,(l,u) in enumerate(zip(lo.ravel(),hi.ravel())):
            v=packed[:,int(offset[i]):int(offset[i])+int(u-l)].astype(float);v[v==np.iinfo(np.uint32).max]=np.inf
            z=sums[int(offset[i]):int(offset[i])+int(u-l)].astype(float);z[z==np.iinfo(np.uint32).max]=np.inf
            paths[:,i//w,i%w,l:u]=v;dense_sums[i//w,i%w,l:u]=z
        result.update(paths=paths,sums=dense_sums)
    return result

def _run(runner,root,h,w,d,eye,dump):
    env=os.environ.copy();env.pop('FP2_DUMP_PATHS',None)
    if dump:env['FP2_DUMP_PATHS']='1'
    args=[str(runner),str(root),str(h),str(w),str(d),'8','32']
    if eye is not None:args.append(str(eye))
    subprocess.run(args,env=env,check=True)

def run_cost(runner,directory,cost,lo,hi):
    cost,lo,hi=_inputs(cost,lo,hi,8,32);h,w,d=cost.shape;root=Path(directory);root.mkdir(parents=True,exist_ok=True);offset=_prepare(root,lo,hi,(h,w),d)
    packed=np.full(int(offset[-1]),np.iinfo(np.uint32).max,dtype='<u4')
    for i,(l,u) in enumerate(zip(lo.ravel(),hi.ravel())):
        v=cost[i//w,i%w,l:u];finite=np.isfinite(v);a=packed[int(offset[i]):int(offset[i])+int(u-l)];a[finite]=v[finite].astype(np.uint32)
    packed.tofile(root/'cost');_run(runner,root,h,w,d,None,True);return _read(root,lo,hi,d,True)

def run_image(runner,directory,own,other,lo,hi,eye,disparities,dump=False):
    if eye not in (0,1) or own.descriptor.shape!=other.descriptor.shape or any(a.shape!=own.descriptor.shape for a in (own.support,other.support)):raise ValueError('invalid inputs')
    h,w=own.descriptor.shape;root=Path(directory);root.mkdir(parents=True,exist_ok=True);_prepare(root,lo,hi,(h,w),disparities)
    for name,a in [('own',own.descriptor),('other',other.descriptor),('own_support',own.support),('other_support',other.support)]:np.asarray(a,dtype='<u4').tofile(root/name)
    _run(runner,root,h,w,disparities,eye,dump);return _read(root,np.asarray(lo),np.asarray(hi),disparities,dump)
