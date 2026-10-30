"""Work-proportional CSR SGM research reference with independent disparity hypotheses.

This is an actual sparse-state recurrence, not a full-D shader with inactive
lanes. CPU oracle and GPU Slang proof-of-concept use the same sorted CSR layout.
Quality and speed are NOT qualified by candidate counts; dense fallback is
intentional for pixels without any reliable hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class CandidatePlan:
    mask: np.ndarray
    fallback_pixels: int
    disagreement_pixels: int
    anchor_only_pixels: int
    candidate_slots: int
    dense_slots: int


@dataclass(frozen=True)
class CsrVolume:
    width: int
    height: int
    disparities: int
    offsets: np.ndarray
    hypotheses: np.ndarray
    costs: np.ndarray


@dataclass(frozen=True)
class CsrResult:
    paths: np.ndarray
    sums: np.ndarray
    disparity: np.ndarray
    valid: np.ndarray
    candidate_state_updates: int
    dense_state_slots: int


def candidate_plan(coarse: np.ndarray, coarse_valid: np.ndarray,
                   anchors: np.ndarray, anchor_valid: np.ndarray, disparities: int,
                   coarse_radius: int = 2, anchor_radius: int = 1) -> CandidatePlan:
    """Union *independent* high-resolution sparse anchors with coarse priors.

    Both priors must be measured/estimated without ground truth. Unknown pixels
    fall back to exhaustive disparity search so they cannot silently disappear.
    """
    coarse, anchors = (np.asarray(v) for v in (coarse, anchors))
    valid_c, valid_a = (np.asarray(v) for v in (coarse_valid, anchor_valid))
    if (coarse.ndim != 2 or not all(coarse.shape) or
        anchors.shape != coarse.shape or valid_c.shape != coarse.shape or
        valid_a.shape != coarse.shape or
        valid_c.dtype != np.bool_ or valid_a.dtype != np.bool_ or
        not 1 <= disparities <= 4096 or not 0 <= coarse_radius < disparities or
        not 0 <= anchor_radius < disparities):
        raise ValueError("invalid candidate-plan shapes, masks or search widths")
    if (np.any(valid_c & (~np.isfinite(coarse) | (coarse < 0) | (coarse >= disparities))) or
        np.any(valid_a & (~np.isfinite(anchors) | (anchors < 0) | (anchors >= disparities)))):
        raise ValueError("nonfinite or out-of-bounds valid disparity prior")
    h, w = coarse.shape
    search = np.arange(disparities, dtype=np.int64)[None, None, :]
    # Round explicitly, using the same half-up convention for nonnegative d.
    safe_c = np.floor(np.where(valid_c, coarse, 0).astype(np.float64) + .5).astype(np.int64)
    safe_a = np.floor(np.where(valid_a, anchors, 0).astype(np.float64) + .5).astype(np.int64)
    from_c = valid_c[..., None] & (np.abs(search - safe_c[..., None]) <= coarse_radius)
    from_a = valid_a[..., None] & (np.abs(search - safe_a[..., None]) <= anchor_radius)
    missing = ~(valid_c | valid_a)
    mask = from_c | from_a | missing[..., None]
    overlap = valid_c & valid_a & (np.abs(safe_c - safe_a) > max(coarse_radius, anchor_radius))
    return CandidatePlan(mask, int(missing.sum()), int(overlap.sum()),
                         int((valid_a & ~valid_c).sum()), int(mask.sum()),
                         h * w * disparities)


def pack_csr(cost_cube: np.ndarray, candidates: np.ndarray) -> CsrVolume:
    cost = np.asarray(cost_cube, dtype=np.float64)
    mask = np.asarray(candidates)
    if (cost.ndim != 3 or not all(cost.shape) or mask.shape != cost.shape or mask.dtype != np.bool_):
        raise ValueError("expected nonempty HxWxD cost volume and matching boolean mask")
    if (np.any(np.isnan(cost)) or np.any(np.isneginf(cost)) or
        np.any(mask & ~np.isfinite(cost)) or np.any(mask & (cost < 0)) or
        np.any(mask & (cost > 10000)) or
        np.any(mask & (cost != np.floor(cost)))):
        raise ValueError("active costs must be finite integers between 0 and 10000")
    h, w, d = cost.shape
    flat_mask = mask.reshape(h*w,d)
    per_pixel = flat_mask.sum(axis=1, dtype=np.int64)
    offsets = np.empty(h*w+1, dtype=np.uint64)
    offsets[0]=0
    offsets[1:]=np.cumsum(per_pixel, dtype=np.uint64)
    if offsets[-1] > np.iinfo(np.uint32).max:
        raise ValueError("CSR candidate count exceeds uint32 GPU indexing")
    hypotheses = np.nonzero(flat_mask)[1].astype(np.uint32)
    costs = cost.reshape(h*w,d)[flat_mask].astype(np.uint32)
    return CsrVolume(w,h,d,offsets.astype(np.uint32),hypotheses,costs)


def validate_csr(volume: CsrVolume) -> None:
    w,h,d=volume.width,volume.height,volume.disparities
    offsets=np.asarray(volume.offsets)
    hypotheses=np.asarray(volume.hypotheses)
    costs=np.asarray(volume.costs)
    if (not all(type(x) is int and x > 0 for x in (w,h,d)) or
        w*h > np.iinfo(np.uint32).max or d > 4096 or
        offsets.dtype != np.uint32 or hypotheses.dtype != np.uint32 or
        costs.dtype != np.uint32 or offsets.shape != (w*h+1,) or
        hypotheses.ndim != 1 or costs.shape != hypotheses.shape or
        offsets[0] != 0 or offsets[-1] != len(hypotheses) or
        np.any(offsets[1:] < offsets[:-1]) or
        np.any(hypotheses >= d) or np.any(costs > 10000)):
        raise ValueError("invalid sorted CSR volume or GPU limits")
    for pixel in range(w*h):
        d_slice=hypotheses[offsets[pixel]:offsets[pixel+1]]
        if len(d_slice)>1 and np.any(d_slice[1:] <= d_slice[:-1]):
            raise ValueError("duplicate or unsorted per-pixel CSR disparity")


def csr_sgm(volume: CsrVolume,p1: int = 8,p2: int = 32) -> CsrResult:
    validate_csr(volume)
    if type(p1) is not int or type(p2) is not int or not 0 <= p1 <= p2 <= 10000:
        raise ValueError("invalid nonnegative integer SGM penalties")
    w,h=volume.width,volume.height
    offsets=volume.offsets
    d=volume.hypotheses
    cost=volume.costs
    total=len(cost)
    paths=np.zeros((4,total),dtype=np.uint32)
    for path,(dx,dy) in enumerate(DIRECTIONS):
        starts=((0 if dx>0 else w-1,y) for y in range(h)) if dx else (
            (x,0 if dy>0 else h-1) for x in range(w))
        for x0,y0 in starts:
            x,y=x0,y0
            previous: dict[int,int]={}
            while 0 <= x < w and 0 <= y < h:
                pixel=y*w+x
                begin,end=int(offsets[pixel]),int(offsets[pixel+1])
                minimum=min(previous.values()) if previous else None
                current={}
                for index in range(begin,end):
                    disp=int(d[index])
                    c=int(cost[index])
                    if minimum is None:
                        value=c
                    else:
                        value=c+min(previous.get(disp,2**31),
                                    previous.get(disp-1,2**31)+p1,
                                    previous.get(disp+1,2**31)+p1,
                                    minimum+p2)-minimum
                    current[disp]=value
                    paths[path,index]=value
                previous=current
                x+=dx;y+=dy
    sums=paths.astype(np.uint64).sum(axis=0)
    disparity=np.full((h,w),-1,dtype=np.int32)
    valid=np.zeros((h,w),dtype=np.bool_)
    for pixel in range(h*w):
        begin,end=int(offsets[pixel]),int(offsets[pixel+1])
        if begin==end:continue
        best=min(range(begin,end),key=lambda i:(int(sums[i]),int(d[i])))
        disparity.flat[pixel]=int(d[best])
        valid.flat[pixel]=True
    return CsrResult(paths,sums,disparity,valid,4*total,4*w*h*volume.disparities)


def normalized_work(result: CsrResult) -> float:
    return result.candidate_state_updates/result.dense_state_slots


@dataclass(frozen=True)
class CsrLRResult:
    left_valid: np.ndarray
    right_valid: np.ndarray


def csr_left_right_consistency(left: CsrResult, right: CsrResult,
                               tolerance: int = 1) -> CsrLRResult:
    """Symmetric integer left/right consistency with strict support.

    No inference, interpolation, or synthetic filling. The two CSR estimates
    must come from independently computed left and right camera hypotheses.
    """
    if (type(tolerance) is not int or tolerance < 0 or tolerance > 10 or
        left.disparity.ndim != 2 or right.disparity.shape != left.disparity.shape or
        left.valid.shape != left.disparity.shape or
        right.valid.shape != right.disparity.shape):
        raise ValueError("invalid independent CSR LR observations")
    h,w=left.disparity.shape
    if not h or not w:
        raise ValueError("empty CSR LR observations")
    x = np.broadcast_to(np.arange(w, dtype=np.int64), (h,w))
    ld,rd=left.disparity.astype(np.int64),right.disparity.astype(np.int64)
    if (np.any(left.valid & (ld < 0)) or np.any(right.valid & (rd < 0))):
        raise ValueError("valid disparities must be nonnegative")
    target_l=x-ld
    target_r=x+rd
    in_l=(target_l>=0)&(target_l<w)&left.valid
    in_r=(target_r>=0)&(target_r<w)&right.valid
    right_idx=np.clip(target_l,0,w-1)
    left_idx=np.clip(target_r,0,w-1)
    other_d_l=np.take_along_axis(rd,right_idx,axis=1)
    other_v_l=np.take_along_axis(right.valid,right_idx,axis=1)
    other_d_r=np.take_along_axis(ld,left_idx,axis=1)
    other_v_r=np.take_along_axis(left.valid,left_idx,axis=1)
    return CsrLRResult(in_l & other_v_l & (np.abs(ld-other_d_l)<=tolerance),
                       in_r & other_v_r & (np.abs(rd-other_d_r)<=tolerance))
