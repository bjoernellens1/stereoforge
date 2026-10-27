"""Independent scalar/NumPy stereo oracle; no GPU imports."""
import numpy as np

def depth_from_disparity(disparity, fx, baseline_m, principal_offset=0.):
    if not np.isfinite([fx,baseline_m,principal_offset]).all() or fx<=0 or baseline_m<=0:
        raise ValueError("invalid metric calibration")
    d=np.asarray(disparity,np.float32)
    denominator=d-principal_offset
    valid=np.isfinite(d)&(d>0)&(denominator>0)
    z=np.full(d.shape,np.nan,np.float32)
    np.divide(fx*baseline_m,denominator,out=z,where=valid)
    return z,valid

INVALID_COST=10000
DIRECTIONS=((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1))

def census(image):
    """5x5, 24-bit Census; border descriptors zero and explicitly invalid."""
    image=np.asarray(image)
    if image.ndim!=2:
        raise ValueError("census expects grayscale HxW")
    h,w=image.shape
    out=np.zeros((h,w),np.uint32); valid=np.zeros((h,w),bool)
    if h<5 or w<5:
        return out,valid
    center=image[2:h-2,2:w-2]; bit=0
    for dy in range(-2,3):
        for dx in range(-2,3):
            if dx==dy==0: continue
            out[2:-2,2:-2] |= (image[2+dy:h-2+dy,2+dx:w-2+dx]<center).astype(np.uint32)<<bit
            bit+=1
    valid[2:-2,2:-2]=True
    return out,valid

def matching_cost(left,right,left_valid,right_valid,max_disparity,lower=None,upper=None):
    h,w=left.shape
    if not 1<=max_disparity<=256: raise ValueError("disparity count must be 1..256")
    cost=np.full((h,w,max_disparity),INVALID_COST,np.uint32)
    for d in range(min(w,max_disparity)):
        xor=left[:,d:]^right[:,:w-d]
        counts=np.bitwise_count(xor).astype(np.uint32)
        ok=left_valid[:,d:] & right_valid[:,:w-d]
        if lower is not None: ok &= (d>=lower[:,d:]) & (d<upper[:,d:])
        cost[:,d:,d]=np.where(ok,counts,INVALID_COST)
    return cost

def aggregate_sgm(cost,paths=4,p1=8,p2=32):
    """Deliberately scalar directional oracle, independent of GPU scheduling."""
    if paths not in (4,8) or not 0<=p1<=p2: raise ValueError("invalid SGM parameters")
    h,w,D=cost.shape; out=np.zeros_like(cost,dtype=np.uint32)
    for dx,dy in DIRECTIONS[:paths]:
        for y in range(h):
            for x in range(w):
                if 0<=x-dx<w and 0<=y-dy<h: continue
                xx,yy=x,y; prev=None
                while 0<=xx<w and 0<=yy<h:
                    c=cost[yy,xx].astype(np.int64)
                    if prev is None: cur=c
                    else:
                        m=int(prev.min())
                        left=np.r_[1000000,prev[:-1]+p1]
                        right=np.r_[prev[1:]+p1,1000000]
                        cur=c+np.minimum(np.minimum(prev,left),np.minimum(right,m+p2))-m
                    out[yy,xx]+=cur.astype(np.uint32)
                    prev=cur; xx+=dx; yy+=dy
    return out

def select_disparity(aggregate,subpixel=True):
    D=aggregate.shape[2]
    best=np.argmin(aggregate,axis=2); score=np.take_along_axis(aggregate,best[...,None],axis=2)[...,0]
    second=np.partition(aggregate,1,axis=2)[...,1] if D>1 else score
    confidence=(second.astype(np.float32)-score)/np.maximum(second,1)
    valid=(score<INVALID_COST)&(best>0)&(second>score)
    disp=best.astype(np.float32)
    if subpixel and D>2:
        lo=np.take_along_axis(aggregate,np.maximum(best-1,0)[...,None],axis=2)[...,0].astype(np.float32)
        hi=np.take_along_axis(aggregate,np.minimum(best+1,D-1)[...,None],axis=2)[...,0].astype(np.float32)
        denominator=lo+hi-2*score.astype(np.float32)
        refine=(best>0)&(best<D-1)&(denominator>0)
        disp[refine]+=.5*(lo[refine]-hi[refine])/denominator[refine]
    disp[~valid]=np.nan; confidence[~valid]=0
    return disp,confidence.astype(np.float32),valid
