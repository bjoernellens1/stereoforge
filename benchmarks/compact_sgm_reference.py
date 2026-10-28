"""Offline scalar experiment for interval-limited, four-path SGM.

Intervals are global disparity indices [lo, hi) at each pixel. +inf cost
means an invalid candidate, including within an interval. The predecessor
minimum ranges over its valid candidates, even when intervals are disjoint;
an empty predecessor resets the recurrence to the current raw costs.
Finite costs and penalties are integers in [0, 10000], ensuring exact bounded
float64 arithmetic. Outputs use +inf for invalid candidates. This is NOT the existing finite
INVALID_COST=10000 convention and is not a production or GPU implementation.
candidate_updates counts valid candidate evaluations across all four paths;
interval_slots counts all interval entries visited, including invalid costs;
dense_slots counts all disparity slots. These are not instruction or GPU work.
"""
from dataclasses import dataclass
import numpy as np

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class SgmResult:
    paths: np.ndarray
    sums: np.ndarray
    candidate_updates: int
    interval_slots: int
    dense_slots: int


def _inputs(cost, lo, hi, p1, p2):
    cost = np.asarray(cost, dtype=np.float64)
    lo, hi = np.asarray(lo), np.asarray(hi)
    if cost.ndim != 3 or not all(cost.shape) or lo.shape != cost.shape[:2] or hi.shape != lo.shape:
        raise ValueError('expected nonempty HxWxD costs and HxW intervals')
    if not np.issubdtype(lo.dtype, np.integer) or not np.issubdtype(hi.dtype, np.integer):
        raise ValueError('interval bounds must be integers')
    if np.any(lo < 0) or np.any(hi > cost.shape[2]) or np.any(lo > hi):
        raise ValueError('interval bounds outside disparity range')
    finite = cost[np.isfinite(cost)]
    if (np.any(np.isnan(cost)) or np.any(np.isneginf(cost)) or np.any(finite < 0)
            or np.any(finite > 10000) or np.any(finite != np.floor(finite))
            or not np.isfinite([p1, p2]).all() or not 0 <= p1 <= p2 <= 10000
            or p1 != int(p1) or p2 != int(p2)):
        raise ValueError('finite costs and penalties must be integers in [0, 10000]')
    return cost, lo, hi


def _result(paths, cost, lo, hi):
    active = ((np.arange(cost.shape[2]) >= lo[..., None]) &
              (np.arange(cost.shape[2]) < hi[..., None]) & np.isfinite(cost))
    return SgmResult(paths, paths.sum(axis=0), len(DIRECTIONS) * int(active.sum()),
                     len(DIRECTIONS) * int(np.sum(hi - lo)), len(DIRECTIONS) * cost.size)


def dense_masked_sgm(cost, lo, hi, p1=8, p2=32):
    """Dense scalar oracle: independently examine all global disparities."""
    cost, lo, hi = _inputs(cost, lo, hi, p1, p2)
    h, w, disparities = cost.shape
    paths = np.full((len(DIRECTIONS), h, w, disparities), np.inf)
    for path, (dx, dy) in enumerate(DIRECTIONS):
        ys = range(h) if dy >= 0 else range(h - 1, -1, -1)
        xs = range(w) if dx >= 0 else range(w - 1, -1, -1)
        for y in ys:
            for x in xs:
                px, py = x - dx, y - dy
                previous = paths[path, py, px] if 0 <= px < w and 0 <= py < h else None
                minimum = min(previous) if previous is not None else np.inf
                for d in range(disparities):
                    if not (lo[y, x] <= d < hi[y, x]) or not np.isfinite(cost[y, x, d]):
                        continue
                    if not np.isfinite(minimum):
                        paths[path, y, x, d] = cost[y, x, d]
                    else:
                        same = previous[d]
                        lower = previous[d - 1] + p1 if d else np.inf
                        upper = previous[d + 1] + p1 if d + 1 < disparities else np.inf
                        paths[path, y, x, d] = cost[y, x, d] + min(same, lower, upper, minimum + p2) - minimum
    return _result(paths, cost, lo, hi)


def compact_sgm(cost, lo, hi, p1=8, p2=32):
    """Scalar compact predecessor state, indexed by global disparity."""
    cost, lo, hi = _inputs(cost, lo, hi, p1, p2)
    h, w, disparities = cost.shape
    paths = np.full((len(DIRECTIONS), h, w, disparities), np.inf)
    for path, (dx, dy) in enumerate(DIRECTIONS):
        for sy in range(h):
            for sx in range(w):
                if 0 <= sx - dx < w and 0 <= sy - dy < h:
                    continue
                x, y, previous = sx, sy, {}
                while 0 <= x < w and 0 <= y < h:
                    current = {}
                    minimum = min(previous.values(), default=None)
                    for d in range(int(lo[y, x]), int(hi[y, x])):
                        c = cost[y, x, d]
                        if not np.isfinite(c):
                            continue
                        if minimum is None:
                            value = c
                        else:
                            value = c + min(previous.get(d, np.inf), previous.get(d-1, np.inf) + p1,
                                            previous.get(d+1, np.inf) + p1, minimum + p2) - minimum
                        current[d] = value
                        paths[path, y, x, d] = value
                    previous = current
                    x, y = x + dx, y + dy
    return _result(paths, cost, lo, hi)
