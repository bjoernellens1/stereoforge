#!/usr/bin/env python3
"""Phase 2 prototype: reference-quality full-resolution stereo (numpy/OpenCV, slow but exact) with uncertainty.

Design (see implementation_plan.md): no coarse-prior locking; cost = normalised ZSAD + gradient SAD over the whole
disparity range; 8-path SGM with image-gradient-adaptive P2; two-way left-right check with occlusion map;
continuous sub-pixel from a parabola on the *aggregated* cost; per-pixel sigma_d from cost-curve curvature.
This is the algorithmic reference to be validated against physical metrics before porting to the GPU.
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
    uniqueness: float = 0.0  # optional min relative gap best vs second-best; sigma already expresses ambiguity
    texture_scale: float = 10.0  # local luma std at which the data term is at half weight


def _prep(gray: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = gray.astype(np.float32)
    mu = cv2.boxFilter(g, -1, (win, win), borderType=cv2.BORDER_REFLECT)
    sd = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (win, win), borderType=cv2.BORDER_REFLECT) - mu * mu, 0))
    return (g - mu) / (sd + 8.0), g, sd


def cost_volume(left: np.ndarray, right: np.ndarray, p: Params) -> np.ndarray:
    """Left-reference cost volume (H, W, D): 0.7 * boxed normalised ZSAD + 0.3 * boxed gradient SAD (clipped)."""
    lz, lg, lsd = _prep(left, p.zsad_window)
    rz, rg, _ = _prep(right, p.zsad_window)
    # Down-weight the data term where the left patch carries no texture so smoothness (not noise) decides there.
    texture = np.clip(cv2.boxFilter(lsd, -1, (9, 9)) / (cv2.boxFilter(lsd, -1, (9, 9)) + p.texture_scale), 0.15, 1.0)
    gl = cv2.Sobel(lg, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gr = cv2.Sobel(rg, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    h, w = left.shape
    vol = np.full((h, w, p.disparities), BIG, np.float32)
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
    """8-path SGM with gradient-adaptive P2; returns summed aggregated cost (H, W, D) float32."""
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
    """Winner-take-all with parabola refinement on the aggregated cost.

    Returns (disparity, sigma_d, curvature, uniqueness_ratio). sigma_d = 1/sqrt(curvature_per_cost_unit) scaled later by calibration.
    """
    h, w, d = agg.shape
    wta = agg.argmin(axis=2)
    rows, cols = np.mgrid[0:h, 0:w]
    c0 = agg[rows, cols, wta]
    cm = agg[rows, cols, np.clip(wta - 1, 0, d - 1)]
    cp = agg[rows, cols, np.clip(wta + 1, 0, d - 1)]
    denom = cm - 2 * c0 + cp
    # Equiangular (V-shaped) interpolation: the matching cost is SAD-like, a parabola would pull estimates to integers.
    slope = np.maximum(cm, cp) - c0
    off = np.where(slope > 1e-6, 0.5 * (cm - cp) / np.maximum(slope, 1e-6), 0.0)
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


def right_disparity_from_left_volume(agg: np.ndarray) -> np.ndarray:
    """WTA right-reference disparity from the left-reference aggregated volume: C_R(x_r, d) = C_L(x_r + d, d)."""
    h, w, d = agg.shape
    right = np.full((h, w, d), np.inf, np.float32)
    for k in range(d):
        right[:, : w - k, k] = agg[:, k:, k]
    return right.argmin(axis=2).astype(np.float32)


def estimate(left: np.ndarray, right: np.ndarray, p: Params | None = None, timings: dict | None = None) -> dict[str, np.ndarray]:
    p = p or Params()
    t0 = time.perf_counter()
    vol = cost_volume(left, right, p)
    t1 = time.perf_counter()
    agg = aggregate(vol, left, p)
    del vol
    t2 = time.perf_counter()
    disp, curv, uniq, cmin = subpixel_and_sigma(agg, p)
    disp_r = right_disparity_from_left_volume(agg)
    del agg
    h, w = disp.shape
    xs = np.arange(w)[None, :].repeat(h, 0)
    xr = np.clip(np.round(xs - disp).astype(int), 0, w - 1)
    lr_diff = np.abs(disp - disp_r[np.arange(h)[:, None], xr])
    valid = (lr_diff <= p.lr_tol) & (uniq > p.uniqueness) & (disp > 0.5)
    occluded = (lr_diff > p.lr_tol) & (disp > 0.5)
    if timings is not None:
        timings.update(cost_s=t1 - t0, aggregate_s=t2 - t1, post_s=time.perf_counter() - t2)
    return {"disparity": disp, "valid": valid, "occluded": occluded, "curvature": curv, "uniqueness": uniq, "cost": cmin, "lr_diff": lr_diff}
