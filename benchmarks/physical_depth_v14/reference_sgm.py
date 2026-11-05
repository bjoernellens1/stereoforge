#!/usr/bin/env python3
"""Full-resolution single-pair references, without coarse-prior locking.

Compare normalized photometric/gradient and Census costs, 4/8 SGM paths and
explicit subpixel fits on identical inputs. Both eyes are independently matched.
Curvature and uniqueness are diagnostics, not calibrated absolute uncertainty.
Physical quality must be established before any production promotion.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

BIG = np.float32(60.0)  # cost assigned to out-of-image candidates


@dataclass
class Params:
    disparities: int = 128
    zsad_window: int = 5
    p1: float = 2.0
    p2: float = 14.0
    p2_min: float = 3.0
    edge_tau: float = 12.0  # luma step at which P2 is halved
    lr_tol: float = 1.0
    uniqueness: float = 0.0  # minimum relative gap to the nonadjacent second-best cost
    texture_scale: float = 10.0  # local luma std at which the data term is at half weight
    matching_cost: str = 'normalized'  # normalized photometric/gradient or 5x5 Census
    paths: int = 8
    subpixel: str = 'equiangular'  # explicit cost-fit ablation; no calibrated sigma
    lr_sampling: str = 'nearest'
    lr_max_jump: float = 1.0  # maximum right disparity change across contributing taps


def _prep(gray: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = gray.astype(np.float32)
    mu = cv2.boxFilter(g, -1, (win, win), borderType=cv2.BORDER_REFLECT)
    sd = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (win, win), borderType=cv2.BORDER_REFLECT) - mu * mu, 0))
    return (g - mu) / (sd + 8.0), g, sd


def cost_volume(left: np.ndarray, right: np.ndarray, p: Params) -> np.ndarray:
    """Left-reference (H,W,D) normalized photometric/gradient or Census costs.

    Census uses 24 comparisons, scaled to [0,20] with the full 5x5 window
    required in both images. Its border is unavailable rather than reflected.
    """
    h, w = left.shape
    vol = np.full((h, w, p.disparities), BIG, np.float32)
    if p.matching_cost == 'census':
        descriptors = []
        for gray in (left, right):
            census = np.zeros((h, w), np.uint32)
            center = gray[2:h-2, 2:w-2]
            if h >= 5 and w >= 5:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dx or dy:
                            census[2:h-2, 2:w-2] <<= 1
                            census[2:h-2, 2:w-2] |= gray[2+dy:h-2+dy, 2+dx:w-2+dx] < center
            descriptors.append(census)
        popcount = np.array([i.bit_count() for i in range(256)], np.uint8)
        for d in range(min(p.disparities, max(0, w-4))):
            xor = np.ascontiguousarray(descriptors[0][2:h-2, 2+d:w-2] ^ descriptors[1][2:h-2, 2:w-2-d])
            counts = popcount[xor.view(np.uint8).reshape(*xor.shape, 4)].sum(axis=-1)
            vol[2:h-2, 2+d:w-2, d] = counts * (20. / 24.)
        return vol
    lz, lg, lsd = _prep(left, p.zsad_window)
    rz, rg, _ = _prep(right, p.zsad_window)
    # Down-weight the data term where the left patch carries no texture so smoothness (not noise) decides there.
    texture = np.clip(cv2.boxFilter(lsd, -1, (9, 9)) / (cv2.boxFilter(lsd, -1, (9, 9)) + p.texture_scale), 0.15, 1.0)
    gl = cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gr = cv2.Sobel(rg, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    for d in range(p.disparities):
        if d >= w:
            break
        a = np.abs(lz[:, d:] - rz[:, : w - d])
        b = np.minimum(np.abs(gl[:, d:] - gr[:, : w - d]), 20.0) / 20.0 * 2.0
        c = 0.7 * cv2.boxFilter(np.minimum(a, 2.0), -1, (p.zsad_window, p.zsad_window), borderType=cv2.BORDER_REFLECT)
        c += 0.3 * cv2.boxFilter(b, -1, (3, 3), borderType=cv2.BORDER_REFLECT)
        vol[:, d:, d] = c * 20.0 * texture[:, d:]  # scale into a range where P1/P2 are meaningful
    return vol


def _step(prev: np.ndarray, cost: np.ndarray, p1: float, p2: np.ndarray) -> np.ndarray:
    """One SGM recursion for (N, D) arrays; p2 has shape (N, 1)."""
    m = prev.min(axis=1, keepdims=True)
    up = np.empty_like(prev)
    dn = np.empty_like(prev)
    up[:, 1:] = prev[:, :-1] + p1
    up[:, 0] = np.inf
    dn[:, :-1] = prev[:, 1:] + p1
    dn[:, -1] = np.inf
    best = np.minimum(np.minimum(prev, up), np.minimum(dn, m + p2))
    return cost + best - m


def aggregate(vol: np.ndarray, gray: np.ndarray, p: Params) -> np.ndarray:
    """4/8-path SGM with gradient-adaptive P2; summed (H,W,D) float32."""
    h, w, d = vol.shape
    g = gray.astype(np.float32)
    # P2 per path step depends on the luma step along the path direction.
    def p2_of(step_abs: np.ndarray) -> np.ndarray:
        return np.maximum(p.p2_min, p.p2 / (1.0 + step_abs / p.edge_tau))[:, None]

    total = np.zeros_like(vol)
    zero_n = lambda n: np.zeros((n, d), np.float32)
    # horizontal
    for sign in (1, -1):
        xs = range(w) if sign == 1 else range(w - 1, -1, -1)
        prev = zero_n(h)
        px = None
        for x in xs:
            step_abs = np.zeros(h, np.float32) if px is None else np.abs(g[:, x] - g[:, px])
            prev = _step(prev, vol[:, x, :], p.p1, p2_of(step_abs)) if px is not None else vol[:, x, :].copy()
            total[:, x, :] += prev
            px = x
    # vertical
    for sign in (1, -1):
        ys = range(h) if sign == 1 else range(h - 1, -1, -1)
        prev, py = None, None
        for y in ys:
            if py is None:
                prev = vol[y].copy()
            else:
                prev = _step(prev, vol[y], p.p1, p2_of(np.abs(g[y] - g[py])))
            total[y] += prev
            py = y
    if p.paths == 4:
        return total
    # diagonals: predecessor at (y - sy, x - sx)
    for sy in (1, -1):
        for sx in (1, -1):
            ys = range(h) if sy == 1 else range(h - 1, -1, -1)
            prev, py = None, None
            for y in ys:
                if py is None:
                    prev = vol[y].copy()
                else:
                    shifted = np.zeros_like(prev)
                    shifted_g = g[py].copy()
                    if sx == 1:
                        shifted[1:] = prev[:-1]
                        shifted_g[1:] = g[py][:-1]
                    else:
                        shifted[:-1] = prev[1:]
                        shifted_g[:-1] = g[py][1:]
                    nxt = _step(shifted, vol[y], p.p1, p2_of(np.abs(g[y] - shifted_g)))
                    edge = slice(0, 1) if sx == 1 else slice(w - 1, w)
                    nxt[edge] = vol[y][edge]  # path starts at the image border
                    prev = nxt
                total[y] += prev
                py = y
    return total


def subpixel_and_sigma(agg: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Winner-take-all with explicit equiangular, parabolic or no refinement.

    Returns (disparity, curvature, uniqueness_ratio, minimum_cost).
    Curvature is a diagnostic; it is not calibrated absolute uncertainty.
    """
    h, w, d = agg.shape
    wta = agg.argmin(axis=2)
    rows, cols = np.mgrid[0:h, 0:w]
    c0 = agg[rows, cols, wta]
    cm = agg[rows, cols, np.clip(wta - 1, 0, d - 1)]
    cp = agg[rows, cols, np.clip(wta + 1, 0, d - 1)]
    denom = cm - 2 * c0 + cp
    # Equiangular fits a V-shaped cost; parabola is a separate tested hypothesis.
    slope = np.maximum(cm, cp) - c0
    fit_denom = denom if p.subpixel == 'parabolic' else slope
    off = np.where(fit_denom > 1e-6, 0.5 * (cm - cp) / np.maximum(fit_denom, 1e-6), 0.0)
    if p.subpixel == 'none':
        off.fill(0)
    off = np.clip(off, -0.5, 0.5)
    interior = (wta > 0) & (wta < d - 1)
    disp = np.where(interior, wta + off, wta.astype(np.float32)).astype(np.float32)
    curv = np.where(interior, np.maximum(denom, 0.0), 0.0).astype(np.float32)
    # Second-best minimum away from the winner (>= 2 px) for a uniqueness test.
    masked = agg.copy()
    for k in (-1, 0, 1):
        idx = np.clip(wta + k, 0, d - 1)
        masked[rows, cols, idx] = np.inf
    second = masked.min(axis=2)
    uniq = ((second - c0) / np.maximum(np.abs(c0), 1e-3)).astype(np.float32)
    return disp, curv, uniq, c0.astype(np.float32)


def sample_right_disparity(left, right, right_valid, *, mode='nearest', max_jump=1.):
    """Sample independent right evidence at x-d, without clipping correspondences.

    Linear sampling requires every contributing tap to be reliable and finite,
    with no disparity jump greater than max_jump. Exact integers need one tap.
    Returns sampled disparity, sample validity, in-image and discontinuity masks.
    """
    left, right, right_valid = np.asarray(left), np.asarray(right), np.asarray(right_valid, bool)
    if left.ndim != 2 or left.shape != right.shape or right.shape != right_valid.shape or min(left.shape) < 1:
        raise ValueError('LR sampling requires equal nonempty stereo grids')
    if mode not in ('nearest', 'linear') or not np.isfinite(max_jump) or max_jump < 0:
        raise ValueError('invalid LR sampling mode or disparity jump')
    h, w = left.shape
    coordinate = np.arange(w)[None, :] - left
    inside = np.isfinite(coordinate) & (coordinate >= 0) & (coordinate <= w-1)
    safe = np.where(inside, coordinate, 0.)  # safe indexing never grants support
    rows = np.arange(h)[:, None]
    edge = np.zeros(left.shape, dtype=bool)
    if mode == 'nearest':
        ix = np.rint(safe).astype(np.intp)
        sampled = right[rows, ix]
        supported = inside & right_valid[rows, ix] & np.isfinite(sampled)
    else:
        lo = np.floor(safe).astype(np.intp)
        hi = np.minimum(lo+1, w-1)
        fraction = safe-lo
        interpolate = fraction != 0
        a, b = right[rows, lo], right[rows, hi]
        va, vb = right_valid[rows, lo] & np.isfinite(a), right_valid[rows, hi] & np.isfinite(b)
        a = np.where(va, a, 0.)
        b = np.where(vb, b, 0.)
        edge = inside & interpolate & va & vb & (np.abs(a-b) > max_jump)
        supported = inside & va & (~interpolate | vb) & ~edge
        sampled = a + fraction*(b-a)
    return np.where(supported, sampled, np.nan), supported, inside, edge


def estimate(left: np.ndarray, right: np.ndarray, p: Params | None = None, timings: dict | None = None) -> dict[str, np.ndarray]:
    p = p or Params()
    if (left.ndim != 2 or left.shape != right.shape or min(left.shape) < 2
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("stereo inputs must be finite, equal-sized grayscale images")
    if not isinstance(p.disparities, int) or p.disparities < 4:
        raise ValueError("reference requires at least four disparities")
    if p.paths not in (4, 8) or p.matching_cost not in ('normalized', 'census') or p.subpixel not in ('equiangular', 'parabolic', 'none'):
        raise ValueError('invalid path count, matching cost or subpixel fit')
    if (p.lr_sampling not in ('nearest', 'linear') or not np.isfinite(p.lr_max_jump)
            or p.lr_max_jump < 0 or not np.isfinite(p.lr_tol) or p.lr_tol < 0):
        raise ValueError('invalid LR sampling configuration')
    t0 = time.perf_counter()
    vol = cost_volume(left, right, p)
    t1 = time.perf_counter()
    agg = aggregate(vol, left, p)
    del vol
    t2 = time.perf_counter()
    disp, curv, uniq, cmin = subpixel_and_sigma(agg, p)
    del agg
    # Swapping AND mirroring makes positive x_left-x_right disparity a
    # positive disparity in right-reference coordinates. Aggregate afresh in
    # the right image: reindexing left path costs is correlated evidence.
    right_ref, left_target = right[:, ::-1], left[:, ::-1]
    right_volume = cost_volume(right_ref, left_target, p)
    right_agg = aggregate(right_volume, right_ref, p)
    del right_volume
    right_disp, _, right_uniq, _ = subpixel_and_sigma(right_agg, p)
    right_valid = ((right_uniq > p.uniqueness) & (right_disp > .5)
                   & (right_disp < p.disparities - 1))[:, ::-1].copy()
    support = np.ones(left.shape, dtype=bool)
    if p.matching_cost == 'census':
        support[:2] = support[-2:] = False
        support[:, :2] = support[:, -2:] = False
        right_valid &= support
    disp_r = right_disp[:, ::-1].copy()
    right_correspondence = np.arange(left.shape[1])[None, :] + disp_r
    right_valid &= np.isfinite(right_correspondence) & (right_correspondence >= 0) & (right_correspondence <= left.shape[1]-1)
    del right_agg
    sampled, right_supported, in_image, discontinuity = sample_right_disparity(
        disp, disp_r, right_valid, mode=p.lr_sampling, max_jump=p.lr_max_jump)
    lr_diff = np.where(right_supported, np.abs(disp-sampled), np.inf)
    boundary = (disp <= 0.5) | (disp >= p.disparities - 1)
    ambiguous = uniq <= p.uniqueness
    valid = support & in_image & ~boundary & ~ambiguous & (lr_diff <= p.lr_tol)
    occluded = (lr_diff > p.lr_tol) & (disp > 0.5)
    if timings is not None:
        timings.update(cost_s=t1 - t0, aggregate_s=t2 - t1, post_and_right_s=time.perf_counter() - t2)
    return {"disparity": disp, "right_disparity": disp_r, "right_validity": right_valid, "valid": valid,
            "occluded": occluded, "curvature": curv, "uniqueness": uniq,
            "cost": cmin, "lr_diff": lr_diff, "out_of_image": ~in_image,
            "range_boundary": boundary, "ambiguous": ambiguous, "support": support,
            "lr_discontinuity": discontinuity, "lr_support_failure": in_image & ~right_supported & ~discontinuity}
