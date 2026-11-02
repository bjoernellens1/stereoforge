"""CPU reference for conservative, provenance-tagged StereoForge planar completion.

Diagnostic oracle only. It does not replace measured depth, claim GT correctness,
run on the GPU, or apply completion to normal ROS depth output.
All disparities are in pixels on their respective image grids.
"""
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PlanarRecoveryConfig:
    scale: int = 4
    coarse_min_support: int = 5
    coarse_plane_rmse_px: float = 0.55
    coarse_lr_tolerance_px: float = 1.0
    min_measured_anchors: int = 10
    min_populated_quadrants: int = 3
    anchor_disparity_tolerance_px: float = 3.0
    anchor_plane_rmse_px: float = 0.8
    anchor_prior_difference_px: float = 2.5
    patch_mean_difference_dn: float = 12.0
    max_luma_gradient_dn: float = 32.0
    min_disparity_px: float = 1.0
    max_disparity_px: float = 127.0
    window_size: int = 41


@dataclass
class PlanarRecoveryResult:
    depth_disparity: np.ndarray
    provenance: np.ndarray
    inferred_confidence: np.ndarray
    candidate_mask: np.ndarray
    coarse_prediction: np.ndarray
    diagnostics: dict[str, int]


def _box(src: np.ndarray, size: int) -> np.ndarray:
    return cv2.boxFilter(src.astype(np.float64), cv2.CV_64F, (size, size), normalize=False,
                         borderType=cv2.BORDER_CONSTANT)


def _plane_fit_moments(n: np.ndarray, sx: np.ndarray, sy: np.ndarray,
                       sxx: np.ndarray, sxy: np.ndarray, syy: np.ndarray,
                       sz: np.ndarray, sxz: np.ndarray, syz: np.ndarray,
                       szz: np.ndarray, min_n: int) -> tuple[np.ndarray, ...]:
    safe_n = np.maximum(n, 1.0)
    mx = sx / safe_n
    my = sy / safe_n
    mz = sz / safe_n
    cxx = sxx - sx * mx
    cxy = sxy - sx * my
    cyy = syy - sy * my
    cxz = sxz - sx * mz
    cyz = syz - sy * mz
    det = cxx * cyy - cxy * cxy
    well_conditioned = (n >= min_n) & (det > 1e-4 * np.maximum(cxx * cyy, 1.0))
    safe_det = np.where(well_conditioned, det, 1.0)
    a = np.where(well_conditioned, (cxz * cyy - cyz * cxy) / safe_det, 0.0)
    b = np.where(well_conditioned, (cyz * cxx - cxz * cxy) / safe_det, 0.0)
    mse = np.maximum(0.0, (szz - sz * mz - a * cxz - b * cyz) / safe_n)
    return a, b, mz - a * mx - b * my, np.sqrt(mse), well_conditioned


def _coarse_local_planes(coarse_d: np.ndarray, coarse_v: np.ndarray,
                         min_n: int) -> tuple[np.ndarray, ...]:
    h, w = coarse_d.shape
    d = np.pad(np.where(coarse_v & np.isfinite(coarse_d), coarse_d, 0.0), 1)
    v = np.pad((coarse_v & np.isfinite(coarse_d) & (coarse_d > 0)).astype(np.float64), 1)
    n = np.zeros((h, w), np.float64)
    sx = np.zeros_like(n); sy = np.zeros_like(n)
    sxx = np.zeros_like(n); sxy = np.zeros_like(n); syy = np.zeros_like(n)
    sz = np.zeros_like(n); sxz = np.zeros_like(n); syz = np.zeros_like(n)
    szz = np.zeros_like(n)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = slice(1 + dy, 1 + dy + h); xx = slice(1 + dx, 1 + dx + w)
            support = v[yy, xx]; obs = d[yy, xx] * support
            n += support; sx += dx * support; sy += dy * support
            sxx += dx * dx * support; sxy += dx * dy * support; syy += dy * dy * support
            sz += obs; sxz += dx * obs; syz += dy * obs; szz += obs * d[yy, xx]
    return (*_plane_fit_moments(n, sx, sy, sxx, sxy, syy, sz, sxz, syz, szz, min_n), n)


def _quadrant_occupancy(valid: np.ndarray, radius: int) -> np.ndarray:
    h, w = valid.shape
    integ = cv2.integral(valid.astype(np.uint8), sdepth=cv2.CV_32S)
    y, x = np.indices((h, w))
    counts = np.zeros((h, w), np.uint8)
    for dx0, dx1, dy0, dy1 in [(-radius, 0, -radius, 0), (1, radius + 1, -radius, 0),
                                (-radius, 0, 1, radius + 1), (1, radius + 1, 1, radius + 1)]:
        x0 = np.clip(x + dx0, 0, w); x1 = np.clip(x + dx1, 0, w)
        y0 = np.clip(y + dy0, 0, h); y1 = np.clip(y + dy1, 0, h)
        cnt = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        counts += (cnt >= 2).astype(np.uint8)
    return counts


def recover_planar_reference(
    measured_disp: np.ndarray, measured_valid: np.ndarray,
    coarse_left: np.ndarray, coarse_right: np.ndarray,
    coarse_left_valid: np.ndarray, coarse_right_valid: np.ndarray,
    rectified_left_gray: np.ndarray, rectified_right_gray: np.ndarray,
    *, config: PlanarRecoveryConfig = PlanarRecoveryConfig(),
) -> PlanarRecoveryResult:
    """Recover a separate inferred layer using coarse stereo + 2D affine anchor fits.

    Requires coarse stereo, multi-region anchors, and a photometric check.
    It does not validate a full-resolution descriptor at the PROPOSED disparity;
    that is required in the future Slang implementation. Never use this reference
    as an accepted robotics measurement or quote the output as ground truth.
    """
    d = np.asarray(measured_disp, dtype=np.float32)
    v = np.asarray(measured_valid, dtype=bool) & np.isfinite(d) & (d > 0)
    left = np.asarray(rectified_left_gray, dtype=np.float32)
    right = np.asarray(rectified_right_gray, dtype=np.float32)
    coarse_l = np.asarray(coarse_left, dtype=np.float32)
    coarse_r = np.asarray(coarse_right, dtype=np.float32)
    h, w = d.shape
    s = config.scale
    if any(a.shape != (h, w) for a in (v, left, right)):
        raise ValueError('full-resolution input shapes disagree')
    if h % s or w % s or coarse_l.shape != (h // s, w // s) or coarse_r.shape != coarse_l.shape:
        raise ValueError('coarse/full grid must be exact scale')
    if config.window_size < 5 or config.window_size % 2 != 1:
        raise ValueError('anchor window size must be odd and >=5')
    hc, wc = coarse_l.shape
    ca, cb, cc, crms, cok, ccount = _coarse_local_planes(coarse_l, coarse_left_valid,
                                                          config.coarse_min_support)
    yy, xx = np.indices((h, w), dtype=np.float32)
    xc = (xx + 0.5) / s - 0.5
    yc = (yy + 0.5) / s - 0.5
    ic = np.clip(np.rint(xc).astype(np.int32), 0, wc-1)
    jc = np.clip(np.rint(yc).astype(np.int32), 0, hc-1)
    predicted = s * (ca[jc, ic] * (xc - ic) + cb[jc, ic] * (yc - jc) + cc[jc, ic])
    prior = cok[jc, ic] & (crms[jc, ic] <= config.coarse_plane_rmse_px)
    prior &= (predicted >= config.min_disparity_px) & (predicted <= config.max_disparity_px)
    xr_coarse = xc - predicted / s
    right_valid_f = np.asarray(coarse_right_valid, np.float32)
    right_depth = np.where(np.isfinite(coarse_r), coarse_r, 0.0)
    r_d = cv2.remap(right_depth, xr_coarse.astype(np.float32), yc.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
    r_v = cv2.remap(right_valid_f, xr_coarse.astype(np.float32), yc.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
    prior &= (xr_coarse >= 1) & (xr_coarse < wc-2) & (r_v >= 0.99)
    prior &= np.abs(predicted / s - r_d) <= config.coarse_lr_tolerance_px
    anchors = v & prior & (np.abs(d - predicted) <= config.anchor_disparity_tolerance_px)
    win = config.window_size; rad = win // 2
    n = _box(anchors, win)
    x64 = xx.astype(np.float64); y64 = yy.astype(np.float64)
    z = np.where(anchors, d, 0.0).astype(np.float64)
    a = anchors.astype(np.float64)
    sx = _box(a*x64,win); sy=_box(a*y64,win)
    sxx=_box(a*x64*x64,win); syy=_box(a*y64*y64,win); sxy=_box(a*x64*y64,win)
    sz=_box(z,win); sxz=_box(z*x64,win); syz=_box(z*y64,win); szz=_box(z*z,win)
    aa,bb,cc_full,arms,anchor_ok = _plane_fit_moments(n,sx,sy,sxx,sxy,syy,sz,sxz,syz,szz,config.min_measured_anchors)
    proposed = (aa*x64 + bb*y64 + cc_full).astype(np.float32)
    support = anchor_ok & (arms <= config.anchor_plane_rmse_px)
    support &= _quadrant_occupancy(anchors,rad) >= config.min_populated_quadrants
    support &= np.abs(proposed - predicted) <= config.anchor_prior_difference_px
    support &= (proposed >= config.min_disparity_px) & (proposed <= config.max_disparity_px)
    # Full-resolution descriptor cost must be independently checked at proposed disparity in GPU implementation.
    xl_match=xx-proposed
    l_mean=cv2.blur(left,(5,5)); r_mean=cv2.blur(right,(5,5))
    r_warp=cv2.remap(r_mean,xl_match.astype(np.float32),yy,cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
    photometric_ok=np.abs(l_mean-r_warp)<=config.patch_mean_difference_dn
    gx=cv2.Sobel(left,cv2.CV_32F,1,0,ksize=3,scale=0.125)
    gy=cv2.Sobel(left,cv2.CV_32F,0,1,ksize=3,scale=0.125)
    edge_ok=np.hypot(gx,gy)<=config.max_luma_gradient_dn
    inferred=(~v)&prior&support&photometric_ok&edge_ok&(xl_match>=4)&(xl_match<w-5)&(yy>=4)&(yy<h-4)
    out=np.full((h,w),np.nan,np.float32)
    out[v]=d[v]
    out[inferred]=proposed[inferred]
    provenance=np.where(v,1,np.where(inferred,2,0)).astype(np.uint8)
    confidence=np.zeros((h,w),np.float32)
    confidence[inferred]=np.clip(1-arms[inferred]/max(config.anchor_plane_rmse_px,1e-3),0,1)
    return PlanarRecoveryResult(out,provenance,confidence,inferred,predicted.astype(np.float32),{
        'measured_pixels':int(v.sum()),'coarse_supported_pixels':int(prior.sum()),
        'anchor_supported_pixels':int(((~v)&prior&support).sum()),
        'inferred_pixels':int(inferred.sum()),'unknown_pixels':int((provenance==0).sum()),
    })
