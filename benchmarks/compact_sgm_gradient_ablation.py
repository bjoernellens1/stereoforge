"""Offline independent-eye gradient-adaptive P2 SGM ablation; not a GPU matcher."""
import subprocess
from pathlib import Path
import numpy as np
from benchmarks.compact_sgm_reference import DIRECTIONS, _inputs


def adaptive_paths(cost, lo, hi, image, p1=8, p2=32):
    cost, lo, hi = _inputs(cost, lo, hi, p1, p2)
    image = np.asarray(image)
    if image.shape != cost.shape[:2] or image.dtype != np.uint8:
        raise ValueError('expected uint8 luma matching costs')
    h, w, d = cost.shape
    paths = np.full((4, h, w, d), np.inf)
    for k, (dx, dy) in enumerate(DIRECTIONS):
        for y in (range(h) if dy >= 0 else range(h-1, -1, -1)):
            for x in (range(w) if dx >= 0 else range(w-1, -1, -1)):
                py, px = y-dy, x-dx
                previous = paths[k,py,px] if 0 <= py < h and 0 <= px < w else None
                m = min(previous) if previous is not None else np.inf
                delta = abs(int(image[y,x])-int(image[py,px])) if previous is not None else 0
                penalty = max(p1, p2 * 16 // (16 + delta))
                for q in range(int(lo[y,x]), int(hi[y,x])):
                    c = cost[y,x,q]
                    if not np.isfinite(c):
                        continue
                    if not np.isfinite(m):
                        paths[k,y,x,q] = c
                    else:
                        paths[k,y,x,q] = c + min(previous[q], previous[q-1]+p1 if q else np.inf,
                                                   previous[q+1]+p1 if q+1<d else np.inf, m+penalty) - m
    return paths


def run_compiled(runner, directory, cost, lo, hi, image, eye):
    cost, lo, hi = _inputs(cost, lo, hi, 8, 32)
    if eye not in (0, 1) or np.asarray(image).dtype != np.uint8 or image.shape != cost.shape[:2]:
        raise ValueError('invalid eye or luma')
    h,w,d=cost.shape
    root=Path(directory); root.mkdir(parents=True, exist_ok=True)
    out=np.full(cost.shape, np.iinfo(np.uint32).max, dtype='<u4')
    finite=np.isfinite(cost); out[finite]=cost[finite].astype(np.uint32)
    out.tofile(root/'cost')
    for name,a in [('lo',lo), ('hi',hi), ('luma',image)]:
        np.asarray(a,dtype='<u4').tofile(root/name)
    subprocess.run([str(runner),str(root),str(h),str(w),str(d),'8','32'],check=True)
    def load(name,shape):
        a=np.fromfile(root/name,dtype='<u4').reshape(shape).astype(float)
        a[a==np.iinfo(np.uint32).max]=np.inf
        return a
    return load('paths',(4,h,w,d)),load('sums',(h,w,d)),np.fromfile(root/'winners',dtype='<i4').reshape(h,w)


def run_eye(runner, directory, own, other, lo, hi, image, eye, disparities):
    """Image-mode runner uses independent own/other Census descriptors."""
    from benchmarks.compact_sgm_quality import INF
    if (eye not in (0,1) or not np.issubdtype(image.dtype, np.unsignedinteger)
            or np.any(image > 255) or image.shape != own.descriptor.shape):
        raise ValueError('invalid image or eye')
    shape=image.shape
    if any(a.shape != shape for a in (other.descriptor, own.support, other.support, lo, hi)):
        raise ValueError('plane shape mismatch')
    root=Path(directory); root.mkdir(parents=True,exist_ok=True)
    for name, a in [('own',own.descriptor),('other',other.descriptor),('own_support',own.support),
                    ('other_support',other.support),('lo',lo),('hi',hi),('luma',image)]:
        np.asarray(a,dtype='<u4').tofile(root/name)
    h,w=shape
    subprocess.run([str(runner),str(root),str(h),str(w),str(disparities),'8','32',str(eye)],check=True)
    result={name: np.fromfile(root/name,dtype=dtype).reshape(shape)
            for name,dtype in [('winners','<i4'),('best','<u4'),('second','<u4'),('count','<u4')]}
    result['candidate_updates'],result['interval_slots']=map(int,(root/'stats').read_text().split())
    return result
