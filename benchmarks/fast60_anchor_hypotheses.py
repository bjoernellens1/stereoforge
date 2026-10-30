"""Independent full-resolution stereo anchors for Fast60 hypothesis experiments.

This CPU reference never reads coarse disparities or ground truth. It matches
5x5 Census descriptors at sparsely sampled, textured pixels and requires
mutual left/right correspondence. The output is a *candidate seed set*, not a
full-resolution depth map or a quality-qualified reconstruction method.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnchorConfig:
    disparities: int = 128
    stride: int = 4
    min_gradient: int = 16
    uniqueness_margin: int = 3
    max_hamming: int = 8
    lr_tolerance: int = 0

    def __post_init__(self):
        if not 1 <= self.disparities <= 256:
            raise ValueError("disparities must be in [1,256]")
        if not 1 <= self.stride <= 64:
            raise ValueError("stride must be in [1,64]")
        if not 0 <= self.min_gradient <= 510:
            raise ValueError("invalid gradient threshold")
        if not 0 <= self.uniqueness_margin <= 24 or not 0 <= self.max_hamming <= 24:
            raise ValueError("invalid Census thresholds")
        if not 0 <= self.lr_tolerance <= 2:
            raise ValueError("invalid left/right tolerance")


def census5(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 24-bit descriptors, valid support and simple gradient magnitude."""
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("expected HxW uint8 image")
    h, w = gray.shape
    if not 5 <= h <= 8192 or not 5 <= w <= 8192:
        raise ValueError("invalid image dimensions")
    desc = np.zeros((h, w), dtype=np.uint32)
    valid = np.zeros((h, w), dtype=bool)
    gradient = np.zeros((h, w), dtype=np.uint16)
    g = gray.astype(np.int16)
    gradient[1:-1, 1:-1] = (
        np.abs(g[1:-1, 2:] - g[1:-1, :-2]) +
        np.abs(g[2:, 1:-1] - g[:-2, 1:-1])
    ).astype(np.uint16)
    middle = g[2:-2, 2:-2]
    bits = desc[2:-2, 2:-2]
    bit = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            bits |= ((g[2+dy:h-2+dy, 2+dx:w-2+dx] < middle).astype(np.uint32) << bit)
            bit += 1
    valid[2:-2, 2:-2] = True
    return desc, valid, gradient


@dataclass(frozen=True)
class Anchors:
    # Coordinates in the rectified full-resolution LEFT image, except right_x.
    left_x: np.ndarray
    y: np.ndarray
    right_x: np.ndarray
    disparity: np.ndarray
    best_cost: np.ndarray
    second_cost: np.ndarray
    evaluated: int

    @property
    def count(self) -> int:
        return int(self.disparity.size)


def _hamming24(x: np.ndarray) -> np.ndarray:
    # Numpy bitwise_count is not available on all supported NumPy versions.
    x = np.asarray(x, dtype=np.uint32)
    n = np.zeros(x.shape, dtype=np.uint8)
    for _ in range(24):
        n += (x & np.uint32(1)).astype(np.uint8)
        x >>= np.uint32(1)
    return n


def match_anchors(left: np.ndarray, right: np.ndarray,
                  config: AnchorConfig = AnchorConfig()) -> Anchors:
    """Match independent, gradient-selected left anchors with mutual Census check.

    A right match is accepted only if a reverse exhaustive search selects the
    original left pixel (within the declared tolerance). Repeated textures and
    low-confidence matches are rejected. The disparity search remains dense at
    each selected anchor; no speedup is claimed from this CPU oracle.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("left/right dimensions differ")
    ld, lv, lg = census5(left)
    rd, rv, rg = census5(right)
    h, w = left.shape
    out = ([], [], [], [], [], [])
    evaluated = 0
    # Stable traversal order; no truth masks, labels, or coarse estimates.
    for y in range(2, h-2, config.stride):
        for x in range(2, w-2, config.stride):
            if not lv[y,x] or lg[y,x] < config.min_gradient:
                continue
            lo = max(2, x-(config.disparities-1))
            hi = x+1
            if hi-lo < 2:
                continue
            xs = np.arange(lo, hi, dtype=np.int32)
            support = rv[y,xs] & (rg[y,xs] >= config.min_gradient)
            if support.sum() < 2:
                continue
            costs = _hamming24(ld[y,x] ^ rd[y,xs])
            costs[~support] = np.uint8(255)
            order = np.argsort(costs, kind="stable")
            i0, i1 = int(order[0]), int(order[1])
            best, second = int(costs[i0]), int(costs[i1])
            evaluated += int(support.sum())
            if best > config.max_hamming or second-best < config.uniqueness_margin:
                continue
            xr = int(xs[i0])
            rev_xs = np.arange(xr, min(w-2, xr+config.disparities), dtype=np.int32)
            rev_support = lv[y,rev_xs] & (lg[y,rev_xs] >= config.min_gradient)
            if not np.any(rev_support):
                continue
            rev_costs = _hamming24(rd[y,xr] ^ ld[y,rev_xs])
            rev_costs[~rev_support] = np.uint8(255)
            evaluated += int(rev_support.sum())
            if abs(int(rev_xs[int(np.argmin(rev_costs))])-x) > config.lr_tolerance:
                continue
            for dst,value in zip(out,(x,y,xr,x-xr,best,second)):
                dst.append(value)
    arrays = [np.asarray(o,dtype=np.int32) for o in out]
    return Anchors(*arrays,evaluated=evaluated)


def candidate_recall(anchors: Anchors, truth_disparity: np.ndarray,
                     *, tolerance: float = 1.0) -> dict:
    """Evaluation-only function: truth must never influence anchor selection."""
    truth = np.asarray(truth_disparity)
    if truth.ndim != 2 or not np.issubdtype(truth.dtype,np.number):
        raise ValueError("expected numeric truth disparity plane")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid tolerance")
    if anchors.count and (np.max(anchors.y) >= truth.shape[0] or
                          np.max(anchors.left_x) >= truth.shape[1]):
        raise ValueError("truth dimensions do not cover anchors")
    values = truth[anchors.y,anchors.left_x]
    known = np.isfinite(values) & (values >= 0)
    good = known & (np.abs(values-anchors.disparity) <= tolerance)
    return {"anchors":anchors.count, "evaluated_candidates":anchors.evaluated,
            "known_anchors":int(known.sum()),"correct_anchors":int(good.sum()),
            "correct_fraction_on_known":float(good.sum()/known.sum()) if known.any() else None}
