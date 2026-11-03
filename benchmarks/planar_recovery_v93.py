"""CPU reference and diagnostic evaluator for StereoForge v9.3 provenance-tagged planar completion
and the D-Y vs D-UV vs D-YUV vs D-UV-Shuffled causal tie-break ablation.

Diagnostic reference only:
- Measured stereo observations keep provenance in {0 (INVALID), 1 (STEREO)}.
- Planar-inferred pixels are tagged with Provenance.PLANAR_INFERRED = 6 (never colliding with
  Provenance.REPROJECTED = 2) and written only to the separate completed layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from stereoforge.types import Provenance


@dataclass(frozen=True)
class PlanarRecoveryConfig:
    scale: int = 4
    coarse_min_support: int = 5
    coarse_plane_rmse_px: float = 0.55
    coarse_lr_tolerance_px: float = 1.0
    use_coarse_affine_fit: bool = True
    use_ray_crossing_guard: bool = True
    use_ortho_descriptor_check: bool = True
    max_ortho_cost_5tap: int = 28
    min_measured_anchors: int = 10
    min_populated_quadrants: int = 3
    anchor_disparity_tolerance_px: float = 3.0
    anchor_plane_rmse_px: float = 0.65
    anchor_prior_difference_px: float = 2.4
    patch_mean_difference_dn: float = 12.0
    max_luma_gradient_dn: float = 28.0
    min_disparity_px: float = 1.5
    max_disparity_px: float = 127.0
    window_size: int = 19


@dataclass
class PlanarRecoveryResult:
    measured_disparity: np.ndarray
    measured_provenance: np.ndarray
    completed_disparity: np.ndarray
    completed_provenance: np.ndarray
    inferred_confidence: np.ndarray
    candidate_mask: np.ndarray
    coarse_prediction: np.ndarray
    diagnostics: dict[str, int]

    @property
    def depth_disparity(self) -> np.ndarray:
        return self.completed_disparity

    @property
    def provenance(self) -> np.ndarray:
        return self.completed_provenance


def _box(src: np.ndarray, size: int) -> np.ndarray:
    return cv2.boxFilter(
        src.astype(np.float64),
        cv2.CV_64F,
        (size, size),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )


def _plane_fit_moments(
    n: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
    syy: np.ndarray,
    sz: np.ndarray,
    sxz: np.ndarray,
    syz: np.ndarray,
    szz: np.ndarray,
    min_n: int,
) -> tuple[np.ndarray, ...]:
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


def _coarse_local_planes(
    coarse_d: np.ndarray, coarse_v: np.ndarray, min_n: int
) -> tuple[np.ndarray, ...]:
    h, w = coarse_d.shape
    d = np.pad(np.where(coarse_v & np.isfinite(coarse_d), coarse_d, 0.0), 1)
    v = np.pad((coarse_v & np.isfinite(coarse_d) & (coarse_d > 0)).astype(np.float64), 1)
    n = np.zeros((h, w), np.float64)
    sx = np.zeros_like(n)
    sy = np.zeros_like(n)
    sxx = np.zeros_like(n)
    sxy = np.zeros_like(n)
    syy = np.zeros_like(n)
    sz = np.zeros_like(n)
    sxz = np.zeros_like(n)
    syz = np.zeros_like(n)
    szz = np.zeros_like(n)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy = slice(1 + dy, 1 + dy + h)
            xx = slice(1 + dx, 1 + dx + w)
            support = v[yy, xx]
            obs = d[yy, xx] * support
            n += support
            sx += dx * support
            sy += dy * support
            sxx += dx * dx * support
            sxy += dx * dy * support
            syy += dy * dy * support
            sz += obs
            sxz += dx * obs
            syz += dy * obs
            szz += obs * d[yy, xx]
    return (*_plane_fit_moments(n, sx, sy, sxx, sxy, syy, sz, sxz, syz, szz, min_n), n)


def _quadrant_occupancy(valid: np.ndarray, radius: int) -> np.ndarray:
    h, w = valid.shape
    integ = cv2.integral(valid.astype(np.uint8), sdepth=cv2.CV_32S)
    y, x = np.indices((h, w))
    counts = np.zeros((h, w), np.uint8)
    for dx0, dx1, dy0, dy1 in [
        (-radius, 0, -radius, 0),
        (1, radius + 1, -radius, 0),
        (-radius, 0, 1, radius + 1),
        (1, radius + 1, 1, radius + 1),
    ]:
        x0 = np.clip(x + dx0, 0, w)
        x1 = np.clip(x + dx1, 0, w)
        y0 = np.clip(y + dy0, 0, h)
        y1 = np.clip(y + dy1, 0, h)
        cnt = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
        counts += (cnt >= 1).astype(np.uint8)
    return counts


def compute_ortho_thermo32(gray: np.ndarray) -> np.ndarray:
    """Compute exact 32-bit ortho_thermo32 descriptor matching ops/refinement.slang."""
    g = np.asarray(gray, dtype=np.int32)
    h, w = g.shape
    pad = np.pad(g, 4, mode="edge")
    center = pad[4 : 4 + h, 4 : 4 + w]
    code = np.zeros((h, w), dtype=np.uint32)
    bit = 0
    for dx in (-4, -3, -2, -1, 1, 2, 3, 4):
        diff = pad[4 : 4 + h, 4 + dx : 4 + dx + w] - center
        for thresh in (0, 2, 6):
            code |= (diff > thresh).astype(np.uint32) << np.uint32(bit)
            bit += 1
    for dy in (-4, -3, -2, -1, 1, 2, 3, 4):
        diff = pad[4 + dy : 4 + dy + h, 4 : 4 + w] - center
        code |= (diff > 0).astype(np.uint32) << np.uint32(bit)
        bit += 1
    return code


def _popcount32(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> 1) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333))
    return (((x + (x >> 4)) & np.uint32(0x0F0F0F0F)) * np.uint32(0x01010101)) >> np.uint32(24)


def recover_planar_reference(
    measured_disp: np.ndarray,
    measured_valid: np.ndarray,
    coarse_left: np.ndarray,
    coarse_right: np.ndarray,
    coarse_left_valid: np.ndarray,
    coarse_right_valid: np.ndarray,
    rectified_left_gray: np.ndarray,
    rectified_right_gray: np.ndarray,
    *,
    config: PlanarRecoveryConfig = PlanarRecoveryConfig(),
) -> PlanarRecoveryResult:
    """Recover a separate provenance-tagged completed layer (Provenance.PLANAR_INFERRED = 6)
    without modifying the measured stereo layer (Provenance.STEREO = 1)."""
    d = np.asarray(measured_disp, dtype=np.float32)
    v = np.asarray(measured_valid, dtype=bool) & np.isfinite(d) & (d > 0)
    left = np.asarray(rectified_left_gray, dtype=np.float32)
    right = np.asarray(rectified_right_gray, dtype=np.float32)
    coarse_l = np.asarray(coarse_left, dtype=np.float32)
    coarse_r = np.asarray(coarse_right, dtype=np.float32)
    h, w = d.shape
    s = config.scale
    if any(a.shape != (h, w) for a in (v, left, right)):
        raise ValueError("full-resolution input shapes disagree")
    if h % s or w % s or coarse_l.shape != (h // s, w // s) or coarse_r.shape != coarse_l.shape:
        raise ValueError("coarse/full grid must be exact scale")
    if config.window_size < 5 or config.window_size % 2 != 1:
        raise ValueError("anchor window size must be odd and >=5")

    hc, wc = coarse_l.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    xc = (xx + 0.5) / s - 0.5
    yc = (yy + 0.5) / s - 0.5
    ic = np.clip(np.rint(xc).astype(np.int32), 0, wc - 1)
    jc = np.clip(np.rint(yc).astype(np.int32), 0, hc - 1)

    if config.use_coarse_affine_fit:
        ca, cb, cc, crms, cok, _ = _coarse_local_planes(
            coarse_l, coarse_left_valid, config.coarse_min_support
        )
        predicted = s * (ca[jc, ic] * (xc - ic) + cb[jc, ic] * (yc - jc) + cc[jc, ic])
        prior = cok[jc, ic] & (crms[jc, ic] <= config.coarse_plane_rmse_px)
    else:
        cl_valid_f = np.asarray(coarse_left_valid, dtype=np.float32)
        cl_clean = np.where(np.isfinite(coarse_l) & coarse_left_valid, coarse_l, 0.0).astype(np.float32)
        w_sum = cv2.remap(cl_valid_f, xc, yc, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        d_sum = cv2.remap(cl_clean * cl_valid_f, xc, yc, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        predicted = np.where(w_sum >= 0.70, s * (d_sum / np.maximum(w_sum, 1e-5)), np.nan)
        prior = w_sum >= 0.70

    prior &= np.isfinite(predicted) & (predicted >= config.min_disparity_px) & (predicted <= config.max_disparity_px)
    xr_coarse = xc - predicted / s
    right_valid_f = np.asarray(coarse_right_valid, np.float32)
    right_depth = np.where(np.isfinite(coarse_r), coarse_r, 0.0)
    r_d = cv2.remap(
        right_depth, xr_coarse.astype(np.float32), yc.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    r_v = cv2.remap(
        right_valid_f, xr_coarse.astype(np.float32), yc.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    prior &= (xr_coarse >= 1) & (xr_coarse < wc - 2) & (r_v >= 0.99)
    prior &= np.abs(predicted / s - r_d) <= config.coarse_lr_tolerance_px

    anchors = v & prior & (np.abs(d - predicted) <= config.anchor_disparity_tolerance_px)
    win = config.window_size
    rad = win // 2
    n = _box(anchors, win)
    x64 = xx.astype(np.float64)
    y64 = yy.astype(np.float64)
    z = np.where(anchors, d, 0.0).astype(np.float64)
    a = anchors.astype(np.float64)
    sx = _box(a * x64, win)
    sy = _box(a * y64, win)
    sxx = _box(a * x64 * x64, win)
    syy = _box(a * y64 * y64, win)
    sxy = _box(a * x64 * y64, win)
    sz = _box(z, win)
    sxz = _box(z * x64, win)
    syz = _box(z * y64, win)
    szz = _box(z * z, win)
    aa, bb, cc_full, arms, anchor_ok = _plane_fit_moments(
        n, sx, sy, sxx, sxy, syy, sz, sxz, syz, szz, config.min_measured_anchors
    )
    proposed = (aa * x64 + bb * y64 + cc_full).astype(np.float32)
    support = anchor_ok & (arms <= config.anchor_plane_rmse_px) & (np.abs(aa) <= 0.22) & (np.abs(bb) <= 0.55)
    support &= _quadrant_occupancy(anchors, rad) >= config.min_populated_quadrants
    support &= np.abs(proposed - predicted) <= config.anchor_prior_difference_px
    support &= (proposed >= config.min_disparity_px) & (proposed <= config.max_disparity_px)

    if config.use_ray_crossing_guard:
        # Midpoint step-discontinuity guard: reject candidates whose local 5x5 neighborhood spans a step edge
        d_pad = np.where(v, d, predicted)
        d_max5 = cv2.dilate(d_pad.astype(np.float32), np.ones((5, 5), np.uint8))
        d_min5 = cv2.erode(d_pad.astype(np.float32), np.ones((5, 5), np.uint8))
        support &= (d_max5 - d_min5) <= 3.2

    xl_match = xx - proposed
    l_mean = cv2.blur(left, (5, 5))
    r_mean = cv2.blur(right, (5, 5))
    r_warp = cv2.remap(r_mean, xl_match.astype(np.float32), yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    r_pix = cv2.remap(right, xl_match.astype(np.float32), yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    photometric_ok = (np.abs(l_mean - r_warp) <= config.patch_mean_difference_dn) & (np.abs(left - r_pix) <= 18.0)

    if config.use_ortho_descriptor_check:
        c_left = compute_ortho_thermo32(left)
        c_right = compute_ortho_thermo32(right)
        rx_int = np.clip(np.rint(xl_match).astype(np.int32), 1, w - 2)
        ry_int = np.clip(yy.astype(np.int32), 1, h - 2)
        base = _popcount32(c_left ^ c_right[ry_int, rx_int])
        m_l = _popcount32(np.roll(c_left, 1, axis=1) ^ c_right[ry_int, np.clip(rx_int - 1, 0, w - 1)])
        m_r = _popcount32(np.roll(c_left, -1, axis=1) ^ c_right[ry_int, np.clip(rx_int + 1, 0, w - 1)])
        m_u = _popcount32(np.roll(c_left, 1, axis=0) ^ c_right[np.clip(ry_int - 1, 0, h - 1), rx_int])
        m_d = _popcount32(np.roll(c_left, -1, axis=0) ^ c_right[np.clip(ry_int + 1, 0, h - 1), rx_int])
        cost_5tap = 2 * base + m_l + m_r + m_u + m_d
        photometric_ok &= cost_5tap <= config.max_ortho_cost_5tap
    else:
        cost_5tap = np.zeros((h, w), dtype=np.uint32)

    gx = np.abs(np.roll(left, -1, axis=1) - np.roll(left, 1, axis=1))
    gy = np.abs(np.roll(left, -1, axis=0) - np.roll(left, 1, axis=0))
    edge_ok = (gx + gy) <= config.max_luma_gradient_dn
    inferred = (~v) & prior & support & photometric_ok & edge_ok & (xl_match >= 4) & (xl_match < w - 5) & (yy >= 4) & (yy < h - 4)

    measured_out = np.where(v, d, np.nan).astype(np.float32)
    measured_prov = np.where(v, int(Provenance.STEREO), int(Provenance.INVALID)).astype(np.uint8)

    completed_out = measured_out.copy()
    completed_out[inferred] = proposed[inferred]
    completed_prov = np.where(
        v,
        int(Provenance.STEREO),
        np.where(inferred, int(Provenance.PLANAR_INFERRED), int(Provenance.INVALID)),
    ).astype(np.uint8)

    confidence = np.zeros((h, w), np.float32)
    confidence[inferred] = np.clip(
        (1.0 - arms[inferred] / max(config.anchor_plane_rmse_px, 1e-3))
        * (1.0 - cost_5tap[inferred].astype(np.float32) / max(config.max_ortho_cost_5tap, 1))
        * np.minimum(n[inferred] / 16.0, 1.0),
        0.05,
        0.85,
    )
    return PlanarRecoveryResult(
        measured_disparity=measured_out,
        measured_provenance=measured_prov,
        completed_disparity=completed_out,
        completed_provenance=completed_prov,
        inferred_confidence=confidence,
        candidate_mask=inferred,
        coarse_prediction=predicted.astype(np.float32),
        diagnostics={
            "measured_pixels": int(v.sum()),
            "coarse_supported_pixels": int(prior.sum()),
            "anchor_supported_pixels": int(((~v) & prior & support).sum()),
            "inferred_pixels": int(inferred.sum()),
            "completed_valid_pixels": int((completed_prov != 0).sum()),
            "unknown_pixels": int((completed_prov == 0).sum()),
        },
    )


def evaluate_color_tiebreak_causal_ablation(
    rect_rgb_left: np.ndarray,
    rect_rgb_right: np.ndarray,
    base_disparity: np.ndarray,
    base_valid: np.ndarray,
    reason_mask: np.ndarray,
    coarse_prediction: np.ndarray,
    coarse_valid: np.ndarray,
    *,
    zncc_ref_disp: np.ndarray | None = None,
    zncc_ref_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the causal D-Y vs D-UV vs D-YUV vs D-UV-Shuffled tie-break ablation on ambiguous pixels.

    Tests whether Variant D's coverage gain comes from luminance (D-Y) vs independent chroma (D-UV).
    """
    ycrcb_l = cv2.cvtColor(rect_rgb_left, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ycrcb_r = cv2.cvtColor(rect_rgb_right, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y_l, cr_l, cb_l = ycrcb_l[..., 0], ycrcb_l[..., 1], ycrcb_l[..., 2]
    y_r, cr_r, cb_r = ycrcb_r[..., 0], ycrcb_r[..., 1], ycrcb_r[..., 2]

    # Shuffled chroma control: permute (Cr, Cb) across the image so chroma has identical marginal distribution but zero spatial correspondence
    rng = np.random.default_rng(20261102)
    perm = rng.permutation(cr_r.size)
    cr_r_shuf = cr_r.reshape(-1)[perm].reshape(cr_r.shape)
    cb_r_shuf = cb_r.reshape(-1)[perm].reshape(cb_r.shape)

    c_left = compute_ortho_thermo32(y_l)
    c_right = compute_ortho_thermo32(y_r)

    h, w = y_l.shape
    ambig = (~base_valid) & ((reason_mask & (16 | 32 | 64)) != 0) & coarse_valid & np.isfinite(coarse_prediction)
    ay, ax = np.nonzero(ambig)
    centre = np.rint(coarse_prediction[ay, ax]).astype(np.int32)

    # Compute local luma & chroma gradient magnitudes for Q1..Q4 quadrant classification
    luma_grad = np.hypot(
        cv2.Sobel(y_l, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(y_l, cv2.CV_32F, 0, 1, ksize=3),
    )
    chroma_grad = np.hypot(
        cv2.Sobel(cr_l, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(cb_l, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(cr_l, cv2.CV_32F, 0, 1, ksize=3) + cv2.Sobel(cb_l, cv2.CV_32F, 0, 1, ksize=3),
    )
    l_med = float(np.median(luma_grad))
    c_med = float(np.median(chroma_grad))

    variants = {
        "D_Y_luma_only": (1.0, 0.0, False),
        "D_UV_chroma_only": (0.0, 1.0, False),
        "D_YUV_combined": (1.0, 1.0, False),
        "D_UV_shuffled_control": (0.0, 1.0, True),
    }

    results: dict[str, Any] = {
        "baseline_valid_pixels": int(base_valid.sum()),
        "ambiguous_candidate_pixels": int(ambig.sum()),
        "variants": {},
    }

    for vname, (wy, wuv, shuffled) in variants.items():
        r_cr = cr_r_shuf if shuffled else cr_r
        r_cb = cb_r_shuf if shuffled else cb_r
        best_cost = np.full(ay.shape, 1e9, dtype=np.float32)
        second_cost = np.full(ay.shape, 1e9, dtype=np.float32)
        best_d = np.zeros(ay.shape, dtype=np.float32)

        for offset in range(-2, 3):
            d_cand = np.clip(centre + offset, 1, 127)
            rx = np.clip(ax - d_cand, 1, w - 2)
            base_h = _popcount32(c_left[ay, ax] ^ c_right[ay, rx]).astype(np.float32) * 6.0
            dy_diff = np.abs(y_l[ay, ax] - y_r[ay, rx]) / 4.0
            duv_diff = (np.abs(cr_l[ay, ax] - r_cr[ay, rx]) + np.abs(cb_l[ay, ax] - r_cb[ay, rx])) / 4.0
            cost = base_h + np.abs(offset) + wy * dy_diff + wuv * duv_diff

            better = cost < best_cost
            second_cost = np.where(better, best_cost, np.minimum(second_cost, cost))
            best_cost = np.where(better, cost, best_cost)
            best_d = np.where(better, d_cand.astype(np.float32), best_d)

        admitted = (second_cost - best_cost) >= 1.0
        adm_y = ay[admitted]
        adm_x = ax[admitted]
        adm_d = best_d[admitted]

        lg = luma_grad[adm_y, adm_x]
        cg = chroma_grad[adm_y, adm_x]
        q1 = int(((lg < l_med) & (cg < c_med)).sum())
        q2 = int(((lg < l_med) & (cg >= c_med)).sum())
        q3 = int(((lg >= l_med) & (cg < c_med)).sum())
        q4 = int(((lg >= l_med) & (cg >= c_med)).sum())
        total_gain = int(admitted.sum())

        zncc_bad1 = None
        if zncc_ref_disp is not None and zncc_ref_valid is not None and total_gain > 0:
            z_ok = zncc_ref_valid[adm_y, adm_x]
            if int(z_ok.sum()) > 0:
                zncc_bad1 = round(float((np.abs(adm_d[z_ok] - zncc_ref_disp[adm_y[z_ok], adm_x[z_ok]]) > 1.0).mean() * 100.0), 2)

        adm_rx = np.clip(adm_x - adm_d.astype(np.int32), 0, w - 1)
        epipolar_luma_mae = (
            round(float(np.mean(np.abs(y_l[adm_y, adm_x] - y_r[adm_y, adm_rx]))), 3)
            if total_gain > 0
            else None
        )
        epipolar_chroma_mae = (
            round(
                float(
                    np.mean(
                        0.5
                        * (
                            np.abs(cr_l[adm_y, adm_x] - cr_r[adm_y, adm_rx])
                            + np.abs(cb_l[adm_y, adm_x] - cb_r[adm_y, adm_rx])
                        )
                    )
                ),
                3,
            )
            if total_gain > 0
            else None
        )

        results["variants"][vname] = {
            "admitted_pixels": total_gain,
            "total_valid_pixels": int(base_valid.sum()) + total_gain,
            "coverage_gain_pct": round(100.0 * total_gain / (h * w), 3),
            "q1_low_luma_low_chroma": q1,
            "q2_low_luma_high_chroma": q2,
            "q3_high_luma_low_chroma": q3,
            "q4_high_luma_high_chroma": q4,
            "low_chroma_share_pct": round(100.0 * (q1 + q3) / max(total_gain, 1), 2),
            "high_chroma_share_pct": round(100.0 * (q2 + q4) / max(total_gain, 1), 2),
            "newly_admitted_zncc_bad_1px_pct": zncc_bad1,
            "newly_admitted_epipolar_luma_mae_dn": epipolar_luma_mae,
            "newly_admitted_epipolar_chroma_mae_dn": epipolar_chroma_mae,
        }
    return results


def _fit_3d_plane_ransac(
    disp: np.ndarray,
    mask: np.ndarray,
    intr: dict[str, float],
    thresh_mm: float = 3.5,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a 3D plane via RANSAC + ordinary least squares in projective disparity space (x, y, d),
    converting the affine disparity plane d(x, y) = a*x + b*y + c analytically to the 3D unit normal
    (n_x, n_y, n_z) and offset d_off. Because (x, y) are exact pixel coordinates and measurement noise
    is strictly in d, projective-disparity RANSAC avoids errors-in-variables eigenvector flips on narrow strips."""
    h, w = disp.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    fx, fy, cx, cy, baseline_m, cx_offset = (
        intr["fx"],
        intr["fy"],
        intr["cx"],
        intr["cy"],
        intr["baseline_m"],
        intr.get("cx_offset", 0.0),
    )
    fxb = fx * baseline_m * 1000.0
    valid = mask & np.isfinite(disp) & (disp > 0.5)
    vy, vx = np.nonzero(valid)
    dv = disp[vy, vx].astype(np.float64)
    xv, yv = vx.astype(np.float64), vy.astype(np.float64)
    A = np.column_stack([xv, yv, np.ones_like(xv)])

    # Convert millimeter threshold at median depth to disparity threshold (clamped to [0.45, 1.25] px)
    med_d = float(np.median(dv)) if len(dv) > 0 else 64.0
    med_z_mm = fxb / max(med_d - cx_offset, 1.0)
    thresh_px = float(np.clip(thresh_mm * max(med_d - cx_offset, 1.0) / max(med_z_mm, 1.0), 0.45, 1.25))

    rng = np.random.default_rng(42)
    best_inliers = np.zeros(len(dv), dtype=bool)
    if len(dv) >= 3:
        for _ in range(300):
            idx = rng.choice(len(dv), size=3, replace=False)
            A3 = A[idx]
            if abs(float(np.linalg.det(A3))) < 1e-3:
                continue
            abc = np.linalg.solve(A3, dv[idx])
            inl = np.abs(A @ abc - dv) <= thresh_px
            if inl.sum() > best_inliers.sum():
                best_inliers = inl

    if int(best_inliers.sum()) < 3:
        best_inliers = np.ones(len(dv), dtype=bool)

    abc, _, _, _ = np.linalg.lstsq(A[best_inliers], dv[best_inliers], rcond=None)
    a, b, c = float(abc[0]), float(abc[1]), float(abc[2])
    d_gt_map = (a * xx + b * yy + c).astype(np.float32)
    nt = np.array([a * fx, b * fy, (c - cx_offset) + a * cx + b * cy], dtype=np.float64)
    n_len = max(float(np.linalg.norm(nt)), 1e-12)
    normal = -nt / n_len
    d_off = fxb / n_len
    if normal[2] > 0:
        normal = -normal
        d_off = -d_off
    return normal, d_off, d_gt_map


def _eval_plane_cohort(
    disp: np.ndarray,
    mask: np.ndarray,
    roi_total_pixels: int,
    d_gt_map: np.ndarray,
    n_gt: np.ndarray,
    d_off_gt: float,
    intr: dict[str, float],
) -> dict[str, Any]:
    fx, fy, cx, cy, baseline_m, cx_offset = (
        intr["fx"],
        intr["fy"],
        intr["cx"],
        intr["cy"],
        intr["baseline_m"],
        intr.get("cx_offset", 0.0),
    )
    fxb = fx * baseline_m * 1000.0
    valid = mask & np.isfinite(disp) & (disp > 0.5)
    n_px = int(valid.sum())
    cov_pct = round(100.0 * n_px / max(1, roi_total_pixels), 2)
    if n_px < 6:
        return {
            "pixels": n_px,
            "roi_coverage_pct": cov_pct,
            "disp_mae_px": None,
            "disp_rmse_px": None,
            "depth_bias_mm": None,
            "depth_mae_mm": None,
            "depth_rmse_mm": None,
            "orth_plane_residual_rmse_mm": None,
            "normal_angle_err_deg": None,
        }
    vy, vx = np.nonzero(valid)
    d_meas = disp[vy, vx].astype(np.float64)
    d_ref = d_gt_map[vy, vx].astype(np.float64)
    disp_err = d_meas - d_ref
    z_meas_mm = fxb / np.maximum(d_meas - cx_offset, 0.5)
    z_ref_mm = fxb / np.maximum(d_ref - cx_offset, 0.5)
    z_diff = z_meas_mm - z_ref_mm
    X = (vx.astype(np.float64) - cx) * z_meas_mm / fx
    Y = (vy.astype(np.float64) - cy) * z_meas_mm / fy
    pts = np.column_stack([X, Y, z_meas_mm])
    orth_res_mm = np.abs(pts @ n_gt + d_off_gt)
    # Fit cohort normal using RANSAC inliers so isolated boundary outliers do not skew normal angle error
    normal, _, _ = _fit_3d_plane_ransac(disp, valid, intr, thresh_mm=3.5)
    cos_ang = np.clip(np.abs(np.dot(normal, n_gt)), 0.0, 1.0)
    return {
        "pixels": n_px,
        "roi_coverage_pct": cov_pct,
        "disp_mae_px": round(float(np.mean(np.abs(disp_err))), 3),
        "disp_rmse_px": round(float(np.sqrt(np.mean(disp_err ** 2))), 3),
        "depth_bias_mm": round(float(np.mean(z_diff)), 2),
        "depth_mae_mm": round(float(np.mean(np.abs(z_diff))), 2),
        "depth_rmse_mm": round(float(np.sqrt(np.mean(z_diff ** 2))), 2),
        "orth_plane_residual_rmse_mm": round(float(np.sqrt(np.mean(orth_res_mm ** 2))), 2),
        "normal_angle_err_deg": round(float(np.degrees(np.arccos(cos_ang))), 2),
    }


def run_v93_qualification_suite(work_dir: Any, desk_env_dir: Any | None = None) -> dict[str, Any]:
    """Execute the full commit-bound StereoForge v9.3 GPU qualification & ablation suite."""
    import json
    import subprocess
    from pathlib import Path
    from benchmarks.visual_accuracy_audit_v9 import (
        COARSE_4X_MANIFEST,
        DEFAULT_MANIFEST,
        DUAL_BENCH,
        REFINE_BENCH,
        build_independent_zncc_epipolar_reference,
        compute_dense_epipolar_and_planar_maps,
        current_provenance_manifest,
        evaluate_known_ground_truth_suite_720p,
        gpu_env,
    )

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    env = gpu_env()
    h, w = 720, 1280

    # 1. Run 240 FPS Hardware GPU Qualification across 5 interleaved trials (planar_recovery=0 vs planar_recovery=1)
    num_interleaved_trials = 5
    trial_docs: dict[str, list[dict[str, Any]]] = {
        "dual4x_selective_repair_only": [],
        "dual4x_selective_repair_plus_planar": [],
    }
    for trial in range(num_interleaved_trials):
        for label, planar_flag in (
            ("dual4x_selective_repair_only", "0"),
            ("dual4x_selective_repair_plus_planar", "1"),
        ):
            tmp_p = work / f"perf_{label}_t{trial}.json"
            tmp_p.unlink(missing_ok=True)
            subprocess.run(
                [
                    str(DUAL_BENCH),
                    "--coarse-manifest", str(COARSE_4X_MANIFEST),
                    "--target-manifest", str(DEFAULT_MANIFEST),
                    "--coarse-disparities", "32",
                    "--descriptor-mode", "ortho-hybrid",
                    "--fast-mode", "1",
                    "--selective-repair", "1",
                    "--planar-recovery", planar_flag,
                    "--planar-min-anchors", "10",
                    "--planar-coarse-affine", "1",
                    "--planar-ray-crossing-guard", "1",
                    "--planar-coarse-lr-guard", "1",
                    "--radius", "2",
                    "--subpixel", "1",
                    "--coarse-schedule", "shared-census-sequential",
                    "--coarse-right-reference-mode", "left-aggregated-selection",
                    "--frames", "120",
                    "--warmup", "20",
                    "--validation",
                    "--output", str(tmp_p),
                ],
                env=env,
                check=True,
            )
            trial_docs[label].append(json.loads(tmp_p.read_text()))
            tmp_p.unlink(missing_ok=True)

    gpu_perf: dict[str, Any] = {}
    for label, docs in trial_docs.items():
        # Select the median trial by gpu_combined_queue_span_ms.p95 (no best-of cherry-picking)
        docs_sorted = sorted(docs, key=lambda d: float(d["gpu_combined_queue_span_ms"]["p95"]))
        median_doc = docs_sorted[len(docs_sorted) // 2]
        out_p = work / f"perf_{label}.json"
        out_p.write_text(json.dumps(median_doc, indent=2) + "\n")
        doc = median_doc
        p50_trials = [round(float(d["gpu_combined_queue_span_ms"]["p50"]), 4) for d in docs]
        p95_trials = [round(float(d["gpu_combined_queue_span_ms"]["p95"]), 4) for d in docs]
        p99_trials = [round(float(d["gpu_combined_queue_span_ms"]["p99"]), 4) for d in docs]
        host_p95_trials = [round(float(d["host_wall_ms"]["p95"]), 4) for d in docs]
        planar_p50_trials = [
            round(float(d.get("gpu_planar_recovery_ms", {}).get("p50", 0.0)), 4) for d in docs
        ]
        gpu_perf[label] = {
            "interleaved_trials_count": num_interleaved_trials,
            "selection_policy": "median_by_gpu_combined_p95_across_5_interleaved_trials",
            "gpu_combined_queue_span_ms": doc["gpu_combined_queue_span_ms"],
            "gpu_stage_sum_ms": doc["gpu_stage_sum_ms"],
            "host_wall_ms": doc["host_wall_ms"],
            "gpu_planar_recovery_ms": doc.get("gpu_planar_recovery_ms"),
            "gpu_stage_timings_ms": doc["gpu_stage_timings_ms"],
            "interleaved_trial_distributions": {
                "gpu_combined_p50_ms": p50_trials,
                "gpu_combined_p95_ms": p95_trials,
                "gpu_combined_p99_ms": p99_trials,
                "host_wall_p95_ms": host_p95_trials,
                "gpu_planar_p50_ms": planar_p50_trials,
                "gpu_combined_p95_mean_ms": round(float(np.mean(p95_trials)), 4),
                "gpu_combined_p95_std_ms": round(float(np.std(p95_trials)), 4),
                "all_trials_meet_240fps_gpu_p95": all(x <= 4.167 for x in p95_trials),
            },
            "measured_valid_pixels": doc.get("measured_valid_pixels"),
            "completed_valid_pixels": doc.get("completed_valid_pixels"),
            "inferred_planar_pixels": doc.get("inferred_planar_pixels"),
            "meets_240fps_4_167ms_gpu_p50": float(doc["gpu_combined_queue_span_ms"]["p50"]) <= 4.167,
            "meets_240fps_4_167ms_gpu_p95": float(doc["gpu_combined_queue_span_ms"]["p95"]) <= 4.167,
            "meets_240fps_4_167ms_gpu_p99": float(doc["gpu_combined_queue_span_ms"]["p99"]) <= 4.167,
            "dual_vs_separate_preprocess_30_planes_exact": doc["dual_vs_separate_preprocess_30_planes_exact"],
            "validation_errors_after_teardown": doc["validation_errors_after_teardown"],
        }

    # 2. Desk Scene GPU Ablation (if desk_env_dir is provided)
    desk_ablation: dict[str, Any] = {}
    color_ablation: dict[str, Any] = {}
    if desk_env_dir is not None and Path(desk_env_dir).exists():
        d_env = Path(desk_env_dir)
        c_man = d_env / "coarse/manifest.json"
        t_man = d_env / "target/manifest.json"
        zed_intr = {
            "fx": 527.7903039292192,
            "fy": 527.7903039292192,
            "cx": 648.5904159545898,
            "cy": 352.4145050048828,
            "baseline_m": 0.119444,
            "cx_offset": 0.0,
        }
        tag_poly = np.array([[785, 404], [831, 404], [831, 450], [785, 450]], dtype=np.int32)
        tag_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(tag_mask, [tag_poly], 1)
        tag_mask_b = tag_mask.astype(bool)
        # In-range ChArUco board & horizontal tabletop before d=127 hardware disparity limit at y=464
        char_mask_b = np.zeros((h, w), dtype=bool)
        char_mask_b[450:463, 730:895] = True
        desk_clean_mask = np.zeros((h, w), dtype=bool)
        desk_clean_mask[445:463, 600:925] = True
        desk_clean_mask &= ~tag_mask_b
        desk_broad_mask = np.zeros((h, w), dtype=bool)
        desk_broad_mask[380:520, 480:920] = True

        ablation_specs = [
            ("v93_measured_selective_repair", "1", "0", "10", "1", "1", "1"),
            ("v93_measured_no_repair", "0", "0", "10", "1", "1", "1"),
            ("dea13a1_unguarded_planar", "1", "1", "5", "0", "0", "0"),
            ("v93_planar_a5_bilinear", "1", "1", "5", "0", "1", "1"),
            ("v93_planar_a5_affine3x3", "1", "1", "5", "1", "1", "1"),
            ("v93_planar_a10_bilinear", "1", "1", "10", "0", "1", "1"),
            ("v93_planar_a10_affine3x3_default", "1", "1", "10", "1", "1", "1"),
            ("v93_planar_a16_affine3x3", "1", "1", "16", "1", "1", "1"),
        ]

        ref_disp_p1 = None
        ref_val_p1 = None
        luma_l_desk = None
        luma_r_desk = None
        zncc_d_desk = None
        zncc_v_desk = None
        planes_measured_consistency: dict[str, tuple[np.ndarray, float, np.ndarray, np.ndarray]] = {}
        planes_independent_zncc: dict[str, tuple[np.ndarray, float, np.ndarray, np.ndarray, int]] = {}

        for cfg_name, rep_f, plan_f, min_a, c_aff, ray_g, clr_g in ablation_specs:
            pref = work / f"desk_{cfg_name}"
            for old_f in work.glob(f"desk_{cfg_name}*"):
                old_f.unlink()
            subprocess.run(
                [
                    str(REFINE_BENCH),
                    "--coarse-manifest", str(c_man),
                    "--target-manifest", str(t_man),
                    "--variant", "raw_borrow",
                    "--frames", "1", "--warmup", "0",
                    "--coarse-disparities", "32",
                    "--coarse-schedule", "layered-per-path",
                    "--coarse-right-reference-mode", "independent-sgm",
                    "--descriptor-mode", "ortho-hybrid",
                    "--fast-mode", "0",
                    "--selective-repair", rep_f,
                    "--planar-recovery", plan_f,
                    "--planar-min-anchors", min_a,
                    "--planar-coarse-affine", c_aff,
                    "--planar-ray-crossing-guard", ray_g,
                    "--planar-coarse-lr-guard", clr_g,
                    "--radius", "2", "--subpixel", "1",
                    "--dump-prefix", str(pref),
                    "--output", str(pref) + ".json",
                ],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            p_str = str(pref)
            d_meas = np.fromfile(p_str + ".disparity.f32", dtype="<f4").reshape(2, h, w)[0]
            v_meas = np.fromfile(p_str + ".validity.u32", dtype="<u4").reshape(2, h, w)[0] != 0
            prov_meas = np.fromfile(p_str + ".provenance.u32", dtype="<u4").reshape(2, h, w)[0]

            if (Path(p_str + ".completed_disparity.f32")).exists():
                d_comp = np.fromfile(p_str + ".completed_disparity.f32", dtype="<f4").reshape(-1, h, w)[0]
                v_comp = np.fromfile(p_str + ".completed_validity.u32", dtype="<u4").reshape(-1, h, w)[0] != 0
                prov_comp = np.fromfile(p_str + ".completed_provenance.u32", dtype="<u4").reshape(-1, h, w)[0]
            else:
                d_comp = d_meas
                v_comp = v_meas
                prov_comp = prov_meas

            if cfg_name == "v93_measured_selective_repair":
                ref_disp_p1 = d_meas.copy()
                ref_val_p1 = v_meas.copy()
                luma_pair = np.fromfile(p_str + ".luma.u32", dtype="<u4").reshape(2, h, w)
                luma_l_desk = luma_pair[0]
                luma_r_desk = luma_pair[1]
                zncc_d_desk, zncc_v_desk = build_independent_zncc_epipolar_reference(
                    luma_l_desk, luma_r_desk, max_disp=128
                )
                for pname, rmask, th_mm in (
                    ("upright_apriltag_target", tag_mask_b, 2.5),
                    ("charuco_board_inrange", char_mask_b, 2.5),
                    ("horizontal_desk_plane_clean", desk_clean_mask, 3.5),
                ):
                    n_m, d_off_m, d_map_m = _fit_3d_plane_ransac(d_meas, rmask & v_meas, zed_intr, thresh_mm=th_mm)
                    planes_measured_consistency[pname] = (n_m, d_off_m, d_map_m, rmask)
                    z_mask = rmask & zncc_v_desk
                    n_z, d_off_z, d_map_z = _fit_3d_plane_ransac(zncc_d_desk, z_mask, zed_intr, thresh_mm=th_mm)
                    planes_independent_zncc[pname] = (n_z, d_off_z, d_map_z, rmask, int(z_mask.sum()))

                # Run D-Y vs D-UV vs D-YUV vs D-UV-Shuffled causal tie-break ablation on desk scene
                rgb_l = np.fromfile(p_str + ".rectified_left.u32", dtype="<u4").reshape(h, w)
                rgb_r = np.fromfile(p_str + ".rectified_right.u32", dtype="<u4").reshape(h, w)
                rgb_l_u8 = np.stack([rgb_l & 255, (rgb_l >> 8) & 255, (rgb_l >> 16) & 255], axis=-1).astype(np.uint8)
                rgb_r_u8 = np.stack([rgb_r & 255, (rgb_r >> 8) & 255, (rgb_r >> 16) & 255], axis=-1).astype(np.uint8)
                reason_arr = np.fromfile(p_str + ".reason.u32", dtype="<u4").reshape(2, h, w)[0]
                pred_arr = np.fromfile(p_str + ".prediction.f32", dtype="<f4").reshape(2, h, w)[0]
                pred_v_arr = np.fromfile(p_str + ".prediction_validity.u32", dtype="<u4").reshape(2, h, w)[0] != 0
                color_ablation["desk_scene"] = evaluate_color_tiebreak_causal_ablation(
                    rgb_l_u8,
                    rgb_r_u8,
                    d_meas,
                    v_meas,
                    reason_arr,
                    pred_arr,
                    pred_v_arr,
                    zncc_ref_disp=zncc_d_desk,
                    zncc_ref_valid=zncc_v_desk,
                )

            inferred_mask = prov_comp == int(Provenance.PLANAR_INFERRED)
            meas_exact = (
                bool(np.array_equal(v_meas, ref_val_p1))
                and bool(np.all(prov_meas <= 1))
                if (ref_val_p1 is not None and rep_f == "1")
                else None
            )

            in_bounds_map, reproj_map, zncc5_map, plan3_map = compute_dense_epipolar_and_planar_maps(
                d_comp, v_comp, luma_l_desk, luma_r_desk
            )
            inf_eval = inferred_mask & in_bounds_map if in_bounds_map is not None else inferred_mask
            inf_reproj_mae = (
                round(float(np.nanmean(reproj_map[inf_eval])), 3)
                if (reproj_map is not None and int(inf_eval.sum()) > 0)
                else None
            )
            inf_zncc5 = (
                round(float(np.nanmean(zncc5_map[inf_eval])), 4)
                if (zncc5_map is not None and int(inf_eval.sum()) > 0)
                else None
            )
            inf_plan3_pct = (
                round(100.0 * float(np.mean(plan3_map[inf_eval])), 2)
                if (plan3_map is not None and int(inf_eval.sum()) > 0)
                else None
            )

            plane_evals_zncc = {}
            for pname, (n_z, d_off_z, d_map_z, rmask, z_sup) in planes_independent_zncc.items():
                roi_px = int(rmask.sum())
                plane_evals_zncc[pname] = {
                    "reference_kind": "independent_multiscale_zncc_epipolar_ransac_plane",
                    "independent_zncc_support_pixels": z_sup,
                    "measured_prov1": _eval_plane_cohort(d_comp, rmask & (prov_comp == 1), roi_px, d_map_z, n_z, d_off_z, zed_intr),
                    "inferred_prov6": _eval_plane_cohort(d_comp, rmask & (prov_comp == 6), roi_px, d_map_z, n_z, d_off_z, zed_intr),
                    "completed_total": _eval_plane_cohort(d_comp, rmask & v_comp, roi_px, d_map_z, n_z, d_off_z, zed_intr),
                }

            plane_evals_consistency = {}
            for pname, (n_m, d_off_m, d_map_m, rmask) in planes_measured_consistency.items():
                roi_px = int(rmask.sum())
                plane_evals_consistency[pname] = {
                    "reference_kind": "self_consistency_to_measured_stereo_ransac_plane",
                    "measured_prov1": _eval_plane_cohort(d_comp, rmask & (prov_comp == 1), roi_px, d_map_m, n_m, d_off_m, zed_intr),
                    "inferred_prov6": _eval_plane_cohort(d_comp, rmask & (prov_comp == 6), roi_px, d_map_m, n_m, d_off_m, zed_intr),
                    "completed_total": _eval_plane_cohort(d_comp, rmask & v_comp, roi_px, d_map_m, n_m, d_off_m, zed_intr),
                }

            b_tot = int(desk_broad_mask.sum())
            b_p1 = int((desk_broad_mask & (prov_comp == 1)).sum())
            b_p6 = int((desk_broad_mask & (prov_comp == 6)).sum())
            desk_ablation[cfg_name] = {
                "measured_plane_bit_exact_with_nonplanar": meas_exact,
                "full_frame_measured_valid_pixels": int((prov_comp == 1).sum()),
                "full_frame_inferred_planar_pixels_prov6": int(inferred_mask.sum()),
                "full_frame_completed_valid_pixels": int(v_comp.sum()),
                "broad_desk_roi_measured_pct": round(100.0 * b_p1 / b_tot, 2),
                "broad_desk_roi_inferred_gain_pct": round(100.0 * b_p6 / b_tot, 2),
                "broad_desk_roi_completed_pct": round(100.0 * (b_p1 + b_p6) / b_tot, 2),
                "inferred_cohort_epipolar_reproj_mae_dn": inf_reproj_mae,
                "inferred_cohort_patch_zncc_5x5_mean": inf_zncc5,
                "inferred_cohort_local_3x3_planar_inlier_pct": inf_plan3_pct,
                "planes": plane_evals_zncc,
                "independent_zncc_reference_planes": plane_evals_zncc,
                "consistency_to_measured_planes": plane_evals_consistency,
            }

    # 3. Canonical 5-Scene 1280x720 Synthetic Ground-Truth Suite (with Planar Recovery cohort evaluation)
    gt_suite = evaluate_known_ground_truth_suite_720p(work)

    return {
        "schema": "stereoforge.v9_3_planar_qualification.v1",
        "provenance": current_provenance_manifest(),
        "provenance_enum_contract": {
            "INVALID": int(Provenance.INVALID),
            "STEREO": int(Provenance.STEREO),
            "REPROJECTED": int(Provenance.REPROJECTED),
            "FUSED": int(Provenance.FUSED),
            "NEURAL": int(Provenance.NEURAL),
            "PLANAR_INFERRED": int(Provenance.PLANAR_INFERRED),
            "planar_never_collides_with_reprojected": int(Provenance.PLANAR_INFERRED) != int(Provenance.REPROJECTED),
        },
        "gpu_240fps_qualification": gpu_perf,
        "desk_scene_gpu_ablation": desk_ablation,
        "color_tiebreak_causal_ablation": color_ablation,
        "canonical_5scene_720p_gt_suite": gt_suite["aggregate_across_5_scenes"],
        "canonical_5scene_720p_per_scene_inferred_cohorts": {
            sname: sdata["newly_admitted_repair_pixels_vs_gt"]
            for sname, sdata in gt_suite["per_scene"].items()
        },
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    scratch_desk = Path("/home/bjoern/.gemini/antigravity-cli/brain/6a5df588-91c3-4b8d-8e19-d968e1de4f3f/scratch/desk_fixture_env")
    out_dir = root / "benchmarks/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_v93_qualification_suite(Path("/tmp/sf_v93_qual"), desk_env_dir=scratch_desk if scratch_desk.exists() else None)
    out_file = out_dir / "v9_3_planar_qualification.json"
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_file}")

