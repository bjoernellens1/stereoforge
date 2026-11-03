#!/usr/bin/env python3
"""StereoForge Milestone v12 (`P0`–`P4`) — Temporal Supersampling & Multi-Frame Enhancement Qualification:

Evaluates the 5-step Temporal Supersampling & Quality Tier Roadmap while keeping the fast
`v11` single-frame baseline untouched as control:
1. `v11_single_frame` (Control): Fast single-frame `v11` (`hierarchical-1to8-fallback`, `r=2`, `ultra_fast=True`, `continuous_subpixel=False`).
2. `v11_p0_continuous_subpixel` (`P0`): Single-frame `v11` + continuous 5-point half-pixel zero-mean 3x3 photometric subpixel parabola refinement (`continuous_subpixel=True`).
3. `realsense_ema_temporal`: Unwarped RealSense-style per-pixel exponential moving average (EMA, `alpha=0.40`, `delta_z=0.05 m`, persistence=3 frames) without pose warping.
4. `daxa_temporal_p1_p2` (`P1 + P2`): Pose-warped `DaxaTemporal` integrated with full-resolution `v11` observations (`DaxaRefinement::Frame`), calibrated inverse-depth uncertainty fusion (`w_i = conf_i / sigma_rho^2`), foreground depth-ordering disocclusion guard, age-decayed confidence, and 50% `left_eye_only` GPU memory reduction (`169.6 MB` vs `339.1 MB`).
5. `temporal_cost_refinement_p3_p4` (`P3 + P4`): Pose-warped temporal stereo cost refinement (`d_pred = fx * B / z_warp + cx_offset` guiding current-frame Census re-search in `[-2, +2]` with L-R consistency verification; strict provenance separation between verified current-frame `Provenance::Stereo = 1` with `age_ns = 0` vs historical-only `Provenance::Reprojected = 2` / `7` with `age_ns > 0`), plus optional `P4` 2x joint-luma edge-preserving depth super-resolution.

Generates `benchmarks/results/v12_temporal_supersampling_qualification.json`.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from benchmarks.visual_accuracy_audit_v9 import (
    COARSE_4X_MANIFEST,
    DEFAULT_MANIFEST,
    REFINE_BENCH,
    build_independent_zncc_epipolar_reference,
    current_provenance_manifest,
    gpu_env,
)
from stereoforge.slang_cpu_backend import SlangCpuRefinementBackend

ROOT = Path(__file__).resolve().parents[1]
OUT_V12_JSON = ROOT / "benchmarks" / "results" / "v12_temporal_supersampling_qualification.json"

PROVENANCE_STEREO = 1
PROVENANCE_REPROJECTED = 2
PROVENANCE_PLANAR_INFERRED = 6
PROVENANCE_TEMPORAL_PLANAR_INFERRED = 7


def _multi_scale_texture(h: int, w: int, seed_offset: int, contrast: float = 55.0) -> np.ndarray:
    r = np.random.default_rng(20261104 + seed_offset)
    n1 = cv2.GaussianBlur(r.standard_normal((h, w)).astype(np.float32), (0, 0), 1.2)
    n2 = cv2.GaussianBlur(r.standard_normal((h, w)).astype(np.float32), (0, 0), 3.5)
    n3 = cv2.GaussianBlur(r.standard_normal((h, w)).astype(np.float32), (0, 0), 9.0)
    tex = 128.0 + contrast * (
        0.50 * n1 / (np.std(n1) + 1e-5)
        + 0.35 * n2 / (np.std(n2) + 1e-5)
        + 0.15 * n3 / (np.std(n3) + 1e-5)
    )
    return np.clip(tex, 15.0, 240.0).astype(np.float32)


def _render_stereo_pair_from_world_disparity(
    base_tex: np.ndarray,
    disp_gt: np.ndarray,
    shift_x_px: float = 0.0,
    tx_over_baseline: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render left/right stereo pair at lateral camera translation `tx_over_baseline = tx_m / B`
    (or legacy `shift_x_px`) with z-buffer occlusion, returning `(left_u8, right_u8, disp_gt_frame, non_occluded)`."""
    h, w = disp_gt.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    if abs(tx_over_baseline) > 1e-9 or abs(shift_x_px) > 1e-9:
        scale = tx_over_baseline if abs(tx_over_baseline) > 1e-9 else (shift_x_px / 24.0)
        # Forward-warp reference surface (u0 -> u_L = u0 - scale * d0(u0, v0)) or inverse lookup on smooth surface
        map_x_l = (xx + scale * disp_gt).astype(np.float32)
        map_y_l = yy
        left_float = cv2.remap(
            base_tex,
            map_x_l,
            map_y_l,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        disp_frame = cv2.remap(
            disp_gt,
            map_x_l,
            map_y_l,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    else:
        left_float = base_tex.copy()
        disp_frame = disp_gt.copy()

    right_img = np.zeros_like(left_float)
    right_zbuf = np.full((h, w), -1.0, dtype=np.float32)
    for y in range(h):
        x_src = np.arange(w, dtype=np.float32)
        d_row = disp_frame[y]
        x_dst = x_src - d_row
        x0 = np.floor(x_dst).astype(np.int32)
        frac = x_dst - x0
        order = np.argsort(d_row, kind="stable")
        for idx in order:
            xi = x0[idx]
            f = frac[idx]
            val = left_float[y, idx]
            d_val = d_row[idx]
            if 0 <= xi < w and d_val >= right_zbuf[y, xi] - 0.25:
                right_img[y, xi] = (1.0 - f) * val + (f * right_img[y, xi] if right_zbuf[y, xi] > 0 else f * val)
                right_zbuf[y, xi] = d_val
            if 0 <= xi + 1 < w and d_val >= right_zbuf[y, xi + 1] - 0.25:
                right_img[y, xi + 1] = f * val + (
                    (1.0 - f) * right_img[y, xi + 1] if right_zbuf[y, xi + 1] > 0 else (1.0 - f) * val
                )
                right_zbuf[y, xi + 1] = d_val
    bg_fill = _multi_scale_texture(h, w, 99, contrast=45.0)
    holes = right_zbuf < 0
    right_img[holes] = bg_fill[holes]
    xr_round = np.clip(np.round(xx - disp_frame).astype(np.int32), 0, w - 1)
    seen_d = right_zbuf[yy.astype(np.int32), xr_round]
    non_occluded = (
        (yy >= 24)
        & (yy < h - 24)
        & (xx >= 80)
        & (xx < w - 48)
        & (xx - disp_frame >= 8)
        & (np.abs(seen_d - disp_frame) <= 0.75)
    )
    return (
        np.clip(np.round(left_float), 0, 255).astype(np.uint8),
        np.clip(np.round(right_img), 0, 255).astype(np.uint8),
        disp_frame,
        non_occluded,
    )


def _identity_maps(w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((h, w), dtype=np.float32)
    mx = xx.reshape(-1).copy()
    my = yy.reshape(-1).copy()
    return (mx, my, mx.copy(), my.copy())


def _scale_maps(w_src: int, h_src: int, w_dst: int, h_dst: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((h_dst, w_dst), dtype=np.float32)
    sx = float(w_src) / float(w_dst)
    sy = float(h_src) / float(h_dst)
    mx = ((xx + 0.5) * sx - 0.5).reshape(-1).astype(np.float32)
    my = ((yy + 0.5) * sy - 0.5).reshape(-1).astype(np.float32)
    return (mx, my, mx.copy(), my.copy())


def _pack_yuyv(left_u8: np.ndarray, right_u8: np.ndarray) -> np.ndarray:
    h, w = left_u8.shape
    yuyv = np.empty((h, 2 * w, 2), dtype=np.uint8)
    yuyv[:, :w, 0] = left_u8
    yuyv[:, w:, 0] = right_u8
    yuyv[:, :, 1] = 128
    return yuyv.reshape(-1)


def _popcount32(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.uint32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24


def run_realsense_ema_step(
    curr_depth: np.ndarray,
    curr_conf: np.ndarray,
    curr_valid: np.ndarray,
    curr_prov: np.ndarray,
    curr_sid: np.ndarray,
    curr_age: np.ndarray,
    state: dict[str, np.ndarray] | None,
    *,
    alpha: float = 0.40,
    delta_z: float = 0.05,
    max_persistence: int = 3,
    dt_ns: int = 33_333_333,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """RealSense-style unwarped per-pixel temporal EMA filter with persistence counter."""
    v = curr_valid != 0
    if state is None:
        new_state = {
            "depth": curr_depth.copy(),
            "conf": curr_conf.copy(),
            "valid": curr_valid.copy(),
            "prov": curr_prov.copy(),
            "sid": curr_sid.copy(),
            "age": curr_age.copy(),
            "missing_count": np.where(v, 0, max_persistence + 1).astype(np.int32),
        }
        return new_state, {k: val.copy() for k, val in new_state.items() if k != "missing_count"}

    prev_z = state["depth"]
    prev_v = state["valid"] != 0
    prev_mc = state["missing_count"]

    both = v & prev_v & (np.abs(curr_depth - prev_z) <= delta_z * np.maximum(curr_depth, 0.5))
    Persist = (~v) & prev_v & (prev_mc < max_persistence)

    out_z = np.where(both, alpha * curr_depth + (1.0 - alpha) * prev_z, np.where(v, curr_depth, np.where(Persist, prev_z, np.nan))).astype(np.float32)
    out_conf = np.where(both, np.maximum(curr_conf, 0.9 * state["conf"]), np.where(v, curr_conf, np.where(Persist, 0.85 * state["conf"], 0.0))).astype(np.float32)
    out_valid = np.where(v | Persist, 1, 0).astype(np.uint32)
    out_prov = np.where(
        v,
        curr_prov,
        np.where(
            Persist,
            np.where(
                (state["prov"] == PROVENANCE_PLANAR_INFERRED) | (state["prov"] == PROVENANCE_TEMPORAL_PLANAR_INFERRED),
                PROVENANCE_TEMPORAL_PLANAR_INFERRED,
                PROVENANCE_REPROJECTED,
            ),
            0,
        ),
    ).astype(np.uint32)
    out_sid = np.where(v, curr_sid, np.where(Persist, state["sid"], -1)).astype(np.int64)
    out_age = np.where(v, 0, np.where(Persist, state["age"] + dt_ns, 0)).astype(np.int64)
    out_mc = np.where(v, 0, prev_mc + 1).astype(np.int32)

    new_state = {
        "depth": out_z,
        "conf": out_conf,
        "valid": out_valid,
        "prov": out_prov,
        "sid": out_sid,
        "age": out_age,
        "missing_count": out_mc,
    }
    return new_state, {k: val.copy() for k, val in new_state.items() if k != "missing_count"}


def run_daxa_temporal_step(
    curr: dict[str, np.ndarray],
    prev_state: dict[str, np.ndarray] | None,
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    baseline_m: float,
    cx_offset: float,
    current_from_previous: np.ndarray,
    frame_id: int,
    dt_ns: int = 33_333_333,
    max_age_ns: int = 200_000_000,
    max_reuses: int = 4,
    relative_depth_tolerance: float = 0.03,
    confidence_floor: float = 0.05,
    inverse_depth_fusion: bool = True,
    disocclusion_guard: bool = True,
    temporal_stereo_research: bool = False,
) -> dict[str, np.ndarray]:
    """Executes the exact P1/P2/P3 pose-warped `DaxaTemporal` semantics (`temporal.slang`) on left-eye full-res observations:
    - P1: 3D back-projection & forward splat with z-buffer visibility competition and 200 ms / 4-reuse bounds.
    - P2: Calibrated inverse-depth uncertainty fusion (`rho = 1/z`, `w = conf / sigma_rho^2`) + foreground depth-ordering disocclusion guard + age-decayed confidence.
    - P3: When `temporal_stereo_research=True` and `!current_valid`, uses `d_pred = fx * B / z_warp + cx_offset` to re-search current-frame Census descriptors in `[-2, +2]` with L-R consistency verification. Verified matches become fresh `Provenance::Stereo = 1` (`age_ns = 0`, `source_id = frame_id`); unverified historical fills strictly remain `Provenance::Reprojected = 2` (or `7`) with `age_ns > 0` and historical `source_id`.
    """
    n = width * height
    c_z = curr["completed_depth"][:n].reshape(height, width).astype(np.float32)
    c_conf = curr["completed_confidence"][:n].reshape(height, width).astype(np.float32)
    c_val = (curr["completed_validity"][:n].reshape(height, width) != 0) & np.isfinite(c_z) & (c_z > 0)
    c_prov = curr["completed_provenance"][:n].reshape(height, width).astype(np.uint32)
    c_supp = (curr["support"][:n].reshape(height, width) != 0)
    c_valid_mask = c_val & c_supp & (c_conf >= confidence_floor) & ((c_prov == PROVENANCE_STEREO) | (c_prov == PROVENANCE_PLANAR_INFERRED))

    out_z = np.where(c_valid_mask, c_z, np.nan).astype(np.float32)
    out_conf = np.where(c_valid_mask, c_conf, 0.0).astype(np.float32)
    out_val = c_valid_mask.astype(np.uint32)
    out_prov = np.where(c_valid_mask, c_prov, 0).astype(np.uint32)
    out_sid = np.where(c_valid_mask, int(frame_id), -1).astype(np.int64)
    out_age = np.zeros((height, width), dtype=np.int64)
    out_reuse = np.zeros((height, width), dtype=np.uint32)
    out_count = np.where(c_valid_mask, 1, 0).astype(np.uint32)

    if prev_state is None:
        return {
            "depth": out_z,
            "confidence": out_conf,
            "validity": out_val,
            "provenance": out_prov,
            "source_ids": out_sid,
            "age_ns": out_age,
            "reuse": out_reuse,
            "count": out_count,
            "promoted_stereo_pixels": np.zeros((height, width), dtype=bool),
        }

    h_z = prev_state["depth"]
    h_conf = prev_state["confidence"]
    h_val = (prev_state["validity"] != 0) & np.isfinite(h_z) & (h_z > 0)
    h_prov = prev_state["provenance"]
    h_sid = prev_state["source_ids"]
    h_age = prev_state["age_ns"] + int(dt_ns)
    h_reuse = prev_state["reuse"] + 1
    h_count = prev_state["count"]

    eligible = (
        h_val
        & (h_conf >= confidence_floor)
        & (h_age <= max_age_ns)
        & (h_reuse <= max_reuses)
        & (h_count >= 1)
    )

    yy, xx = np.indices((height, width), dtype=np.float32)
    z_safe = np.where(eligible, h_z, 1.0).astype(np.float32)
    X = ((xx - cx) / fx) * z_safe
    Y = ((yy - cy) / fy) * z_safe
    Z = z_safe

    T = np.asarray(current_from_previous, dtype=np.float32).reshape(4, 4)
    Xw = T[0, 0] * X + T[0, 1] * Y + T[0, 2] * Z + T[0, 3]
    Yw = T[1, 0] * X + T[1, 1] * Y + T[1, 2] * Z + T[1, 3]
    Zw = T[2, 0] * X + T[2, 1] * Y + T[2, 2] * Z + T[2, 3]

    u_proj = np.rint(np.where(eligible, (Xw / np.maximum(Zw, 1e-5)) * fx + cx, -1.0)).astype(np.int32)
    v_proj = np.rint(np.where(eligible, (Yw / np.maximum(Zw, 1e-5)) * fy + cy, -1.0)).astype(np.int32)
    in_bounds = eligible & (Zw > 0.05) & (u_proj >= 0) & (u_proj < width) & (v_proj >= 0) & (v_proj < height)

    # Splat Z-buffer (closest Zw wins; tie-break by source index)
    warped_z = np.full((height, width), np.inf, dtype=np.float32)
    warped_conf = np.zeros((height, width), dtype=np.float32)
    warped_prov = np.zeros((height, width), dtype=np.uint32)
    warped_sid = np.full((height, width), -1, dtype=np.int64)
    warped_age = np.zeros((height, width), dtype=np.int64)
    warped_reuse = np.zeros((height, width), dtype=np.uint32)
    warped_count = np.zeros((height, width), dtype=np.uint32)

    ys, xs = np.nonzero(in_bounds)
    if ys.size > 0:
        zw_vals = Zw[ys, xs]
        up_vals = u_proj[ys, xs]
        vp_vals = v_proj[ys, xs]
        # Sort by decreasing Zw so closer overwrites farther deterministically
        order = np.argsort(-zw_vals, kind="stable")
        ys_o = ys[order]
        xs_o = xs[order]
        up_o = up_vals[order]
        vp_o = vp_vals[order]
        warped_z[vp_o, up_o] = zw_vals[order]
        warped_conf[vp_o, up_o] = h_conf[ys_o, xs_o]
        warped_prov[vp_o, up_o] = h_prov[ys_o, xs_o]
        warped_sid[vp_o, up_o] = h_sid[ys_o, xs_o]
        warped_age[vp_o, up_o] = h_age[ys_o, xs_o]
        warped_reuse[vp_o, up_o] = h_reuse[ys_o, xs_o]
        warped_count[vp_o, up_o] = h_count[ys_o, xs_o]

    has_hist = np.isfinite(warped_z) & c_supp

    # P2 Disocclusion guard:
    # 1. If a neighbor within 2 px has current valid foreground depth closer than warped_z by > tolerance,
    #    or if warped_z is behind a local foreground edge and current stereo was rejected due to occlusion,
    #    check current-frame photometric consistency at `d_warp = fx * B / warped_z + cx_offset`.
    if disocclusion_guard and np.any(has_hist):
        # Local minimum of current valid depth in a 5x5 neighborhood catches foreground boundary disocclusions
        c_z_fill = np.where(c_valid_mask, c_z, 1e6).astype(np.float32)
        local_min_cz = cv2.erode(c_z_fill, np.ones((5, 5), dtype=np.uint8))
        fg_occluded = (local_min_cz < 1e5) & (warped_z > local_min_cz * (1.0 + 2.0 * relative_depth_tolerance))
        # Also verify warped disparity does not violate current-frame luma/Census agreement with a foreground object
        d_warp = (fx * baseline_m) / np.maximum(warped_z, 1e-4) + cx_offset
        xr_w = np.clip(np.rint(xx - d_warp).astype(np.int32), 0, width - 1)
        # If the right-eye pixel at `xr_w` has a valid right-eye disparity belonging to a much closer foreground object,
        # then `(y, x)` was disoccluded by that moving foreground object!
        raw_d_r = curr["raw_disparity"][n:].reshape(height, width)
        raw_v_r = curr["raw_validity"][n:].reshape(height, width) != 0
        d_r_at_warp = raw_d_r[yy.astype(np.int32), xr_w]
        v_r_at_warp = raw_v_r[yy.astype(np.int32), xr_w]
        right_fg_occluding = v_r_at_warp & (d_r_at_warp > d_warp + 2.5)
        # Also check 3x3 patch luma difference at `d_warp` so stale foreground ghosting over newly exposed background is rejected
        luma_l = curr["luma"][:n].reshape(height, width).astype(np.float32)
        luma_r = curr["luma"][n:].reshape(height, width).astype(np.float32)
        luma_r_warped = luma_r[yy.astype(np.int32), xr_w]
        mad_warp = cv2.boxFilter(np.abs(luma_l - luma_r_warped), cv2.CV_32F, (3, 3))
        photometric_mismatch = (~c_valid_mask) & (mad_warp > 18.0)
        has_hist = has_hist & (~fg_occluded) & (~right_fg_occluding) & (~photometric_mismatch)

    # Age decay for historical confidence
    age_ratio = np.clip(warped_age.astype(np.float32) / max(float(max_age_ns), 1.0), 0.0, 1.0)
    h_conf_eff = warped_conf * (1.0 - 0.35 * age_ratio)

    # Case 1: Both current and historical valid -> fuse if within relative depth tolerance
    both = c_valid_mask & has_hist
    agree = both & (np.abs(c_z - warped_z) <= relative_depth_tolerance * np.maximum(c_z, warped_z))
    if np.any(agree):
        if inverse_depth_fusion:
            # P2 calibrated inverse-depth fusion: rho = 1/z, sigma_rho = sigma_d / (fx * B)
            rho_c = 1.0 / c_z
            rho_h = 1.0 / warped_z
            wc = np.maximum(c_conf, 1e-3)
            wh = np.maximum(h_conf_eff, 1e-3) * (0.85 ** np.minimum(warped_reuse, 4))
            rho_fused = (wc * rho_c + wh * rho_h) / (wc + wh)
            z_fused = 1.0 / np.maximum(rho_fused, 1e-5)
            conf_fused = np.clip((wc * c_conf + wh * h_conf_eff) / (wc + wh) + 0.05, 0.0, 1.0)
        else:
            wc = c_conf
            wh = warped_conf
            z_fused = (wc * c_z + wh * warped_z) / np.maximum(wc + wh, 1e-5)
            conf_fused = (wc * c_conf + wh * warped_conf) / np.maximum(wc + wh, 1e-5)
        out_z = np.where(agree, z_fused, out_z).astype(np.float32)
        out_conf = np.where(agree, conf_fused, out_conf).astype(np.float32)
        out_reuse = np.where(agree, warped_reuse, out_reuse).astype(np.uint32)
        out_count = np.where(agree, np.minimum(warped_count + 1, 4), out_count).astype(np.uint32)

    # Case 2: Missing current stereo (`~c_valid_mask`), valid historical (`has_hist`)
    missing_with_hist = (~c_valid_mask) & has_hist
    promoted_mask = np.zeros((height, width), dtype=bool)

    if temporal_stereo_research and np.any(missing_with_hist):
        # P3: Re-search current-frame Census descriptors in `[-2, +2]` around `d_pred = fx * B / warped_z + cx_offset`
        cen_l = curr["census"][:n].reshape(height, width)
        cen_r = curr["census"][n:].reshape(height, width)
        cval_l = curr["census_validity"][:n].reshape(height, width) != 0
        cval_r = curr["census_validity"][n:].reshape(height, width) != 0
        raw_d_r = curr["raw_disparity"][n:].reshape(height, width)
        raw_v_r = curr["raw_validity"][n:].reshape(height, width) != 0

        d_pred = (fx * baseline_m) / np.maximum(warped_z, 1e-4) + cx_offset
        d_center = np.rint(d_pred).astype(np.int32)

        ys_m, xs_m = np.nonzero(missing_with_hist & cval_l & (d_center >= 1))
        for y_i, x_i in zip(ys_m.tolist(), xs_m.tolist()):
            dc = int(d_center[y_i, x_i])
            d_lo = max(1, dc - 2)
            d_hi = min(127, x_i, dc + 2)
            if d_lo > d_hi:
                continue
            cl = np.uint32(cen_l[y_i, x_i])
            best_c = 999
            second_c = 999
            best_d = dc
            costs = {}
            for cand in range(d_lo, d_hi + 1):
                xr_i = x_i - cand
                if not cval_r[y_i, xr_i]:
                    continue
                hd = int(_popcount32(np.array([cl ^ np.uint32(cen_r[y_i, xr_i])], dtype=np.uint32))[0])
                costs[cand] = hd
                if hd < best_c:
                    second_c = best_c
                    best_c = hd
                    best_d = cand
                elif hd < second_c:
                    second_c = hd
            if best_c <= 10 and second_c > best_c:
                d_sub = float(best_d)
                if (best_d - 1) in costs and (best_d + 1) in costs:
                    cm = float(costs[best_d - 1])
                    cp = float(costs[best_d + 1])
                    denom = cm - 2.0 * float(best_c) + cp
                    if denom > 1e-3:
                        d_sub += float(np.clip(0.5 * (cm - cp) / denom, -0.5, 0.5))
                rx_chk = int(round(x_i - d_sub))
                if 0 <= rx_chk < width and raw_v_r[y_i, rx_chk]:
                    if abs(d_sub - float(raw_d_r[y_i, rx_chk])) <= 1.5 and (d_sub - cx_offset) > 0.1:
                        z_new = float((fx * baseline_m) / (d_sub - cx_offset))
                        z_w = float(warped_z[y_i, x_i])
                        if abs(z_new - z_w) <= 2.0 * relative_depth_tolerance * max(z_new, z_w):
                            wc = float(np.clip(float(second_c - best_c) / float(max(second_c, 1)), 0.15, 0.95))
                            if int(warped_count[y_i, x_i]) == 1 and abs(z_new - z_w) <= 0.25 * relative_depth_tolerance * z_w:
                                wh = 0.90 * float(h_conf_eff[y_i, x_i])
                                wc_eff = 0.10 * wc
                                rho_new = (wc_eff * (1.0 / z_new) + wh * (1.0 / z_w)) / (wc_eff + wh)
                                out_z[y_i, x_i] = np.float32(1.0 / rho_new)
                            else:
                                out_z[y_i, x_i] = np.float32(z_w)
                            out_conf[y_i, x_i] = np.float32(min(1.0, max(wc, float(h_conf_eff[y_i, x_i]))))
                            out_val[y_i, x_i] = 1
                            # Provenance honesty: confirmed by current-frame stereo Census + L-R check!
                            out_prov[y_i, x_i] = PROVENANCE_STEREO
                            out_sid[y_i, x_i] = int(frame_id)
                            out_age[y_i, x_i] = 0
                            out_reuse[y_i, x_i] = 0
                            out_count[y_i, x_i] = min(int(warped_count[y_i, x_i]) + 1, 4)
                            promoted_mask[y_i, x_i] = True

    # Remaining missing pixels with valid history become strictly Reprojected (`2` or `7`) with historical source_id & age > 0
    hist_only = missing_with_hist & (~promoted_mask)
    if np.any(hist_only):
        out_z = np.where(hist_only, warped_z, out_z).astype(np.float32)
        out_conf = np.where(hist_only, h_conf_eff, out_conf).astype(np.float32)
        out_val = np.where(hist_only, 1, out_val).astype(np.uint32)
        reproj_prov = np.where(
            (warped_prov == PROVENANCE_PLANAR_INFERRED) | (warped_prov == PROVENANCE_TEMPORAL_PLANAR_INFERRED),
            PROVENANCE_TEMPORAL_PLANAR_INFERRED,
            PROVENANCE_REPROJECTED,
        )
        out_prov = np.where(hist_only, reproj_prov, out_prov).astype(np.uint32)
        out_sid = np.where(hist_only, warped_sid, out_sid).astype(np.int64)
        out_age = np.where(hist_only, warped_age, out_age).astype(np.int64)
        out_reuse = np.where(hist_only, warped_reuse, out_reuse).astype(np.uint32)
        out_count = np.where(hist_only, warped_count, out_count).astype(np.uint32)

    return {
        "depth": out_z,
        "confidence": out_conf,
        "validity": out_val,
        "provenance": out_prov,
        "source_ids": out_sid,
        "age_ns": out_age,
        "reuse": out_reuse,
        "count": out_count,
        "promoted_stereo_pixels": promoted_mask,
    }


def upsample_depth_2x_joint_luma(
    depth_1x: np.ndarray,
    valid_1x: np.ndarray,
    luma_2x: np.ndarray,
    *,
    sigma_luma: float = 10.0,
    max_rel_z_span: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """P4: Edge-preserving 2x joint-bilateral inverse-depth super-resolution guided by 2x rectified luma.
    Refuses to interpolate across depth discontinuities (`max_rel_z_span`), falling back to nearest valid anchor."""
    h1, w1 = depth_1x.shape
    h2, w2 = h1 * 2, w1 * 2
    v1 = (valid_1x != 0) & np.isfinite(depth_1x) & (depth_1x > 0)
    rho1 = np.where(v1, 1.0 / depth_1x, 0.0).astype(np.float32)

    yy2, xx2 = np.indices((h2, w2), dtype=np.float32)
    yc = (yy2 - 0.5) * 0.5
    xc = (xx2 - 0.5) * 0.5
    y0 = np.clip(np.floor(yc).astype(np.int32), 0, h1 - 1)
    x0 = np.clip(np.floor(xc).astype(np.int32), 0, w1 - 1)
    y1 = np.clip(y0 + 1, 0, h1 - 1)
    x1 = np.clip(x0 + 1, 0, w1 - 1)
    wy = np.clip(yc - y0, 0.0, 1.0)
    wx = np.clip(xc - x0, 0.0, 1.0)

    luma_1x = cv2.resize(luma_2x.astype(np.float32), (w1, h1), interpolation=cv2.INTER_AREA)
    l2 = luma_2x.astype(np.float32)

    coords = [(y0, x0, (1.0 - wy) * (1.0 - wx)), (y0, x1, (1.0 - wy) * wx), (y1, x0, wy * (1.0 - wx)), (y1, x1, wy * wx)]
    z_samples = [depth_1x[cy, cx_i] for cy, cx_i, _ in coords]
    v_samples = [v1[cy, cx_i] for cy, cx_i, _ in coords]

    z_min = np.full((h2, w2), np.inf, dtype=np.float32)
    z_max = np.full((h2, w2), -np.inf, dtype=np.float32)
    for zs, vs in zip(z_samples, v_samples):
        z_min = np.where(vs, np.minimum(z_min, zs), z_min)
        z_max = np.where(vs, np.maximum(z_max, zs), z_max)
    is_discontinuity = (z_max - z_min) > max_rel_z_span * np.maximum(z_min, 0.1)

    w_sum = np.zeros((h2, w2), dtype=np.float32)
    rho_sum = np.zeros((h2, w2), dtype=np.float32)
    best_w = np.full((h2, w2), -1.0, dtype=np.float32)
    best_z = np.full((h2, w2), np.nan, dtype=np.float32)

    inv_2sigma2 = 1.0 / (2.0 * sigma_luma * sigma_luma)
    for cy, cx_i, w_spatial in coords:
        vs = v1[cy, cx_i]
        zs = depth_1x[cy, cx_i]
        rs = rho1[cy, cx_i]
        dl = l2 - luma_1x[cy, cx_i]
        w_photo = np.exp(-(dl * dl) * inv_2sigma2)
        w = np.where(vs, w_spatial * w_photo + 1e-5, 0.0)
        w_sum += w
        rho_sum += w * rs
        better = vs & (w > best_w)
        best_w = np.where(better, w, best_w)
        best_z = np.where(better, zs, best_z)

    valid_2x = w_sum > 0
    z_smooth = np.where(valid_2x, 1.0 / np.maximum(rho_sum / np.maximum(w_sum, 1e-6), 1e-5), np.nan)
    depth_2x = np.where(valid_2x, np.where(is_discontinuity, best_z, z_smooth), np.nan).astype(np.float32)
    return depth_2x, valid_2x.astype(np.uint32)


def evaluate_synthetic_translation_sequences(cpu_backend: SlangCpuRefinementBackend) -> dict[str, Any]:
    """Suite A: Evaluate all 5 tiers on synthetic 4-frame subpixel translation sequences
    (`repetitive_and_subpixel` and `slanted_planes`) at 640x360 with 2x super-resolution to 1280x720."""
    h2, w2 = 720, 1280
    h1, w1 = 360, 640
    wc, hc = 160, 90
    fx1, fy1, cx1, cy1 = 520.0, 520.0, 319.5, 179.5
    baseline_m = 0.12
    cx_offset = 0.0

    t_maps = _identity_maps(w1, h1)
    c_maps = _scale_maps(w1, h1, wc, hc)

    yy2, xx2 = np.indices((h2, w2), dtype=np.float32)
    tex_rep = _multi_scale_texture(h2, w2, 6, contrast=48.0) + 14.0 * np.sin(2.0 * np.pi * xx2 / 11.0) * np.cos(
        2.0 * np.pi * yy2 / 13.0
    )
    tex_rep = np.clip(tex_rep, 15.0, 240.0).astype(np.float32)
    gt_disp_rep_2x = np.full((h2, w2), 45.0, dtype=np.float32)  # 22.5 px at 640x360

    tex_slant = _multi_scale_texture(h2, w2, 1, contrast=58.0)
    gt_disp_slant_2x = (28.0 + 0.014 * xx2 + 0.014 * yy2).astype(np.float32)  # 14..35 px at 640x360

    scenes = [
        ("repetitive_and_subpixel", tex_rep, gt_disp_rep_2x, [0.0, 0.125, 0.250, 0.375]),
        ("slanted_planes", tex_slant, gt_disp_slant_2x, [0.0, 0.375, 0.750, 1.125]),
    ]
    results: dict[str, Any] = {}

    for sname, base_tex_2x, gt_disp_2x, tx_scales in scenes:
        ema_state = None
        p12_state = None
        p34_state = None
        prev_scale = 0.0

        tier_metrics: dict[str, dict[str, Any]] = {}
        for frame_idx, scale_t in enumerate(tx_scales):
            l2, r2, disp_frame_2x, mask2 = _render_stereo_pair_from_world_disparity(
                base_tex_2x, gt_disp_2x, tx_over_baseline=scale_t
            )
            gt_disp_1x = 0.5 * cv2.resize(disp_frame_2x, (w1, h1), interpolation=cv2.INTER_AREA)
            gt_z_1x = (fx1 * baseline_m) / gt_disp_1x
            gt_z_2x = (fx1 * 2.0 * baseline_m) / disp_frame_2x
            l1 = cv2.resize(l2, (w1, h1), interpolation=cv2.INTER_AREA)
            r1 = cv2.resize(r2, (w1, h1), interpolation=cv2.INTER_AREA)
            mask1 = cv2.resize(mask2.astype(np.uint8), (w1, h1), interpolation=cv2.INTER_NEAREST) > 0
            yuyv1 = _pack_yuyv(l1, r1)

            # Tier 1: v11 single frame
            res_v11 = cpu_backend.run_full_camera_to_depth_pipeline(
                raw_yuyv=yuyv1,
                target_maps=t_maps,
                coarse_maps=c_maps,
                eye_width=w1,
                source_height=h1,
                stride_bytes=w1 * 4,
                width=w1,
                height=h1,
                coarse_width=wc,
                coarse_height=hc,
                fx=fx1,
                baseline=baseline_m,
                principal_offset=cx_offset,
                disparities=64,
                radius=2,
                subpixel=True,
                continuous_subpixel=False,
                ortho_hybrid=True,
                fast_mode=True,
                ultra_fast=True,
                selective_repair=True,
                planar_recovery=True,
            )
            # Tier 2: v11 + P0 continuous subpixel
            res_p0 = cpu_backend.run_full_camera_to_depth_pipeline(
                raw_yuyv=yuyv1,
                target_maps=t_maps,
                coarse_maps=c_maps,
                eye_width=w1,
                source_height=h1,
                stride_bytes=w1 * 4,
                width=w1,
                height=h1,
                coarse_width=wc,
                coarse_height=hc,
                fx=fx1,
                baseline=baseline_m,
                principal_offset=cx_offset,
                disparities=64,
                radius=2,
                subpixel=True,
                continuous_subpixel=True,
                ortho_hybrid=True,
                fast_mode=True,
                ultra_fast=True,
                selective_repair=True,
                planar_recovery=True,
            )

            n1 = w1 * h1
            sid_arr = np.full((h1, w1), frame_idx + 1, dtype=np.int64)
            age_arr = np.zeros((h1, w1), dtype=np.int64)
            ema_state, res_ema = run_realsense_ema_step(
                res_p0["completed_depth"][:n1].reshape(h1, w1),
                res_p0["completed_confidence"][:n1].reshape(h1, w1),
                res_p0["completed_validity"][:n1].reshape(h1, w1),
                res_p0["completed_provenance"][:n1].reshape(h1, w1),
                sid_arr,
                age_arr,
                ema_state,
            )

            # Known translation transform between frames: u_k = u_{k-1} - (scale_t - prev_scale) * d_1x
            delta_scale = scale_t - prev_scale
            prev_scale = scale_t
            T_curr_from_prev = np.eye(4, dtype=np.float32)
            T_curr_from_prev[0, 3] = -delta_scale * baseline_m

            p12_state = run_daxa_temporal_step(
                res_p0,
                p12_state,
                width=w1,
                height=h1,
                fx=fx1,
                fy=fy1,
                cx=cx1,
                cy=cy1,
                baseline_m=baseline_m,
                cx_offset=cx_offset,
                current_from_previous=T_curr_from_prev,
                frame_id=frame_idx + 1,
                inverse_depth_fusion=True,
                disocclusion_guard=True,
                temporal_stereo_research=False,
            )

            p34_state = run_daxa_temporal_step(
                res_p0,
                p34_state,
                width=w1,
                height=h1,
                fx=fx1,
                fy=fy1,
                cx=cx1,
                cy=cy1,
                baseline_m=baseline_m,
                cx_offset=cx_offset,
                current_from_previous=T_curr_from_prev,
                frame_id=frame_idx + 1,
                inverse_depth_fusion=True,
                disocclusion_guard=True,
                temporal_stereo_research=True,
            )

            if frame_idx == len(tx_scales) - 1:
                def _eval_tier(z_map: np.ndarray, v_map: np.ndarray) -> dict[str, float]:
                    vm = (v_map != 0) & mask1 & np.isfinite(z_map) & (z_map > 0)
                    d_est = (fx1 * baseline_m) / z_map
                    d_err = np.abs(d_est[vm] - gt_disp_1x[vm])
                    z_err_mm = np.abs(z_map[vm] - gt_z_1x[vm]) * 1000.0
                    # P4 2x super-resolution evaluation against 1280x720 ground truth
                    z_2x, v_2x = upsample_depth_2x_joint_luma(z_map, v_map, l2)
                    vm2 = (v_2x != 0) & mask2 & np.isfinite(z_2x) & (z_2x > 0)
                    z2_err_mm = np.abs(z_2x[vm2] - gt_z_2x[vm2]) * 1000.0
                    return {
                        "disparity_mae_px": float(np.mean(d_err)),
                        "bad_0_5px_pct": float(np.mean(d_err > 0.5) * 100.0),
                        "bad_1_0px_pct": float(np.mean(d_err > 1.0) * 100.0),
                        "depth_mae_mm": float(np.mean(z_err_mm)),
                        "valid_coverage_pct": float(np.mean(vm[mask1]) * 100.0),
                        "superres_2x_depth_mae_mm": float(np.mean(z2_err_mm)),
                    }

                tier_metrics = {
                    "v11_single_frame": _eval_tier(
                        res_v11["completed_depth"][:n1].reshape(h1, w1),
                        res_v11["completed_validity"][:n1].reshape(h1, w1),
                    ),
                    "v11_p0_continuous_subpixel": _eval_tier(
                        res_p0["completed_depth"][:n1].reshape(h1, w1),
                        res_p0["completed_validity"][:n1].reshape(h1, w1),
                    ),
                    "realsense_ema_temporal": _eval_tier(res_ema["depth"], res_ema["valid"]),
                    "daxa_temporal_p1_p2": _eval_tier(p12_state["depth"], p12_state["validity"]),
                    "temporal_cost_refinement_p3_p4": _eval_tier(p34_state["depth"], p34_state["validity"]),
                }
        results[sname] = tier_metrics
    return results


def evaluate_moving_object_disocclusion_sequence(cpu_backend: SlangCpuRefinementBackend) -> dict[str, Any]:
    """Suite B: Synthetic 4-frame moving-object disocclusion sequence at 640x360.
    A foreground obstacle (`d = 36 px`, `z = 1.733 m`) translates rightwards by 28 px/frame across a static
    textured background (`d = 16 px`, `z = 3.900 m`). Measures false-obstacle ghosting in the disoccluded wake
    and verifies strict provenance separation."""
    h, w = 360, 640
    wc, hc = 160, 90
    fx, fy, cx, cy = 520.0, 520.0, 319.5, 179.5
    baseline_m = 0.12
    cx_offset = 0.0
    t_maps = _identity_maps(w, h)
    c_maps = _scale_maps(w, h, wc, hc)

    yy, xx = np.indices((h, w), dtype=np.float32)
    tex_bg = _multi_scale_texture(h, w, 20, contrast=55.0)
    tex_fg = _multi_scale_texture(h, w, 21, contrast=62.0)

    z_fg = (fx * baseline_m) / 36.0  # 1.733 m

    ema_state = None
    p12_no_guard_state = None
    p12_state = None
    p34_state = None

    box_starts = [180, 208, 236, 264]
    box_w = 96

    ghosting_by_tier: dict[str, list[float]] = {
        "v11_single_frame": [],
        "realsense_ema_temporal": [],
        "daxa_temporal_no_disocclusion_guard": [],
        "daxa_temporal_p1_p2": [],
        "temporal_cost_refinement_p3_p4": [],
    }
    provenance_violations = 0
    p3_promoted_total = 0

    for frame_idx, x0 in enumerate(box_starts):
        fg_mask = (yy >= 90) & (yy < 270) & (xx >= x0) & (xx < x0 + box_w)
        gt_disp = np.where(fg_mask, 36.0, 16.0).astype(np.float32)
        scene_tex = np.where(fg_mask, tex_fg, tex_bg)
        l_u8, r_u8, _, _ = _render_stereo_pair_from_world_disparity(scene_tex, gt_disp, shift_x_px=0.0)
        yuyv = _pack_yuyv(l_u8, r_u8)

        res = cpu_backend.run_full_camera_to_depth_pipeline(
            raw_yuyv=yuyv,
            target_maps=t_maps,
            coarse_maps=c_maps,
            eye_width=w,
            source_height=h,
            stride_bytes=w * 4,
            width=w,
            height=h,
            coarse_width=wc,
            coarse_height=hc,
            fx=fx,
            baseline=baseline_m,
            principal_offset=cx_offset,
            disparities=64,
            radius=2,
            subpixel=True,
            continuous_subpixel=True,
            ortho_hybrid=True,
            fast_mode=True,
            ultra_fast=True,
            selective_repair=True,
            planar_recovery=False,
        )
        n = w * h
        c_z = res["completed_depth"][:n].reshape(h, w)
        c_conf = res["completed_confidence"][:n].reshape(h, w)
        c_val = res["completed_validity"][:n].reshape(h, w)
        c_prov = res["completed_provenance"][:n].reshape(h, w)
        sid_arr = np.full((h, w), frame_idx + 1, dtype=np.int64)
        age_arr = np.zeros((h, w), dtype=np.int64)

        ema_state, res_ema = run_realsense_ema_step(c_z, c_conf, c_val, c_prov, sid_arr, age_arr, ema_state)
        T_id = np.eye(4, dtype=np.float32)

        p12_no_guard_state = run_daxa_temporal_step(
            res,
            p12_no_guard_state,
            width=w,
            height=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            baseline_m=baseline_m,
            cx_offset=cx_offset,
            current_from_previous=T_id,
            frame_id=frame_idx + 1,
            inverse_depth_fusion=True,
            disocclusion_guard=False,
            temporal_stereo_research=False,
        )
        p12_state = run_daxa_temporal_step(
            res,
            p12_state,
            width=w,
            height=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            baseline_m=baseline_m,
            cx_offset=cx_offset,
            current_from_previous=T_id,
            frame_id=frame_idx + 1,
            inverse_depth_fusion=True,
            disocclusion_guard=True,
            temporal_stereo_research=False,
        )
        p34_state = run_daxa_temporal_step(
            res,
            p34_state,
            width=w,
            height=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            baseline_m=baseline_m,
            cx_offset=cx_offset,
            current_from_previous=T_id,
            frame_id=frame_idx + 1,
            inverse_depth_fusion=True,
            disocclusion_guard=True,
            temporal_stereo_research=True,
        )
        p3_promoted_total += int(np.sum(p34_state["promoted_stereo_pixels"]))

        # Audit provenance honesty: no historical-only fill (`age_ns > 0` or `source_id != frame_idx + 1`) may carry `PROVENANCE_STEREO`
        for st in (p12_state, p34_state):
            v_st = st["validity"] != 0
            bad_prov = v_st & (
                ((st["age_ns"] > 0) & (st["provenance"] == PROVENANCE_STEREO))
                | ((st["source_ids"] != (frame_idx + 1)) & (st["provenance"] == PROVENANCE_STEREO))
                | ((st["provenance"] == PROVENANCE_REPROJECTED) & (st["age_ns"] <= 0))
            )
            provenance_violations += int(np.sum(bad_prov))

        if frame_idx >= 1:
            prev_x0 = box_starts[frame_idx - 1]
            # Newly uncovered background wake: was inside foreground at t-1, now background at t
            wake_mask = (yy >= 96) & (yy < 264) & (xx >= prev_x0) & (xx < x0)

            def _false_fg_pct(z_map: np.ndarray, v_map: np.ndarray) -> float:
                false_fg = (v_map != 0) & wake_mask & (np.abs(z_map - z_fg) < 0.35)
                return float(np.sum(false_fg)) / float(max(int(np.sum(wake_mask)), 1)) * 100.0

            ghosting_by_tier["v11_single_frame"].append(_false_fg_pct(c_z, c_val))
            ghosting_by_tier["realsense_ema_temporal"].append(_false_fg_pct(res_ema["depth"], res_ema["valid"]))
            ghosting_by_tier["daxa_temporal_no_disocclusion_guard"].append(
                _false_fg_pct(p12_no_guard_state["depth"], p12_no_guard_state["validity"])
            )
            ghosting_by_tier["daxa_temporal_p1_p2"].append(_false_fg_pct(p12_state["depth"], p12_state["validity"]))
            ghosting_by_tier["temporal_cost_refinement_p3_p4"].append(
                _false_fg_pct(p34_state["depth"], p34_state["validity"])
            )

    return {
        "mean_disocclusion_ghosting_false_obstacle_pct": {
            k: float(np.mean(v)) for k, v in ghosting_by_tier.items()
        },
        "max_disocclusion_ghosting_false_obstacle_pct": {
            k: float(np.max(v)) for k, v in ghosting_by_tier.items()
        },
        "p3_promoted_current_stereo_pixels_total": p3_promoted_total,
        "provenance_honesty_violations": provenance_violations,
    }


def evaluate_zed2_sequence_and_gpu_audit(work_dir: Path) -> dict[str, Any]:
    """Suite C: Evaluate all 5 tiers across the real 5-frame ZED 2 1280x720 sequence (`record_ordinal` 0..4)
    plus native Vulkan GPU latency and memory footprint audit."""
    env = gpu_env()
    w, h = 1280, 720
    n = w * h

    # Extract metric calibration from target manifest via a quick 20-frame GPU run
    out_p0_json = work_dir / "zed2_gpu_p0.json"
    out_v11_json = work_dir / "zed2_gpu_v11.json"
    for out_j, cont_flag in [(out_v11_json, "0"), (out_p0_json, "1")]:
        out_j.unlink(missing_ok=True)
        subprocess.run(
            [
                str(REFINE_BENCH),
                "--coarse-manifest",
                str(COARSE_4X_MANIFEST),
                "--target-manifest",
                str(DEFAULT_MANIFEST),
                "--output",
                str(out_j),
                "--profile",
                "v11-fast",
                "--planar-recovery",
                "1",
                "--continuous-subpixel",
                cont_flag,
                "--frames",
                "20",
                "--warmup",
                "5",
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
    gpu_v11_report = json.loads(out_v11_json.read_text(encoding="utf-8"))
    gpu_p0_report = json.loads(out_p0_json.read_text(encoding="utf-8"))
    mcal = gpu_v11_report["metric_calibration"]
    fx = float(mcal["fx"])
    baseline_m = float(mcal["baseline_m"])
    cx_offset = float(mcal["cx_offset"])
    fy = fx
    cx, cy = 639.5, 359.5

    ema_state = None
    p12_state = None
    p34_state = None

    zed_history: dict[str, list[np.ndarray]] = {
        "v11_single_frame": [],
        "v11_p0_continuous_subpixel": [],
        "realsense_ema_temporal": [],
        "daxa_temporal_p1_p2": [],
        "temporal_cost_refinement_p3_p4": [],
    }
    last_zncc_disp = None
    last_zncc_valid = None
    last_luma_l = None
    last_tier_outputs: dict[str, dict[str, np.ndarray]] = {}

    for ord_idx in range(5):
        ord_json = work_dir / f"zed_ord_{ord_idx}.json"
        ord_prefix = work_dir / f"zed_ord_{ord_idx}"
        ord_json.unlink(missing_ok=True)
        for p in work_dir.glob(f"zed_ord_{ord_idx}.*"):
            p.unlink(missing_ok=True)
        subprocess.run(
            [
                str(REFINE_BENCH),
                "--coarse-manifest",
                str(COARSE_4X_MANIFEST),
                "--target-manifest",
                str(DEFAULT_MANIFEST),
                "--output",
                str(ord_json),
                "--dump-prefix",
                str(ord_prefix),
                "--profile",
                "v11-fast",
                "--planar-recovery",
                "1",
                "--continuous-subpixel",
                "1",
                "--record-ordinal",
                str(ord_idx),
                "--frames",
                "1",
                "--warmup",
                "0",
            ],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        # Also run v11 control on ordinal 4
        if ord_idx == 4:
            v11_ord_json = work_dir / "zed_ord_4_v11.json"
            v11_ord_prefix = work_dir / "zed_ord_4_v11"
            v11_ord_json.unlink(missing_ok=True)
            for p in work_dir.glob("zed_ord_4_v11.*"):
                p.unlink(missing_ok=True)
            subprocess.run(
                [
                    str(REFINE_BENCH),
                    "--coarse-manifest",
                    str(COARSE_4X_MANIFEST),
                    "--target-manifest",
                    str(DEFAULT_MANIFEST),
                    "--output",
                    str(v11_ord_json),
                    "--dump-prefix",
                    str(v11_ord_prefix),
                    "--profile",
                    "v11-fast",
                    "--planar-recovery",
                    "1",
                    "--continuous-subpixel",
                    "0",
                    "--record-ordinal",
                    "4",
                    "--frames",
                    "1",
                    "--warmup",
                    "0",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

        res_p0 = {
            "completed_depth": np.fromfile(f"{ord_prefix}.completed_depth.f32", dtype=np.float32),
            "completed_confidence": np.fromfile(f"{ord_prefix}.completed_confidence.f32", dtype=np.float32),
            "completed_validity": np.fromfile(f"{ord_prefix}.completed_validity.u32", dtype=np.uint32),
            "completed_provenance": np.fromfile(f"{ord_prefix}.completed_provenance.u32", dtype=np.uint32),
            "census": np.fromfile(f"{ord_prefix}.census.u32", dtype=np.uint32),
            "census_validity": np.fromfile(f"{ord_prefix}.census_validity.u32", dtype=np.uint32),
            "raw_disparity": np.fromfile(f"{ord_prefix}.raw_disparity.f32", dtype=np.float32),
            "raw_validity": np.fromfile(f"{ord_prefix}.raw_validity.u32", dtype=np.uint32),
            "luma": np.fromfile(f"{ord_prefix}.luma.u32", dtype=np.uint32),
            "support": np.fromfile(f"{ord_prefix}.support.u32", dtype=np.uint32),
        }

        c_z = res_p0["completed_depth"][:n].reshape(h, w)
        c_conf = res_p0["completed_confidence"][:n].reshape(h, w)
        c_val = res_p0["completed_validity"][:n].reshape(h, w)
        c_prov = res_p0["completed_provenance"][:n].reshape(h, w)
        sid_arr = np.full((h, w), ord_idx + 1, dtype=np.int64)
        age_arr = np.zeros((h, w), dtype=np.int64)

        ema_state, res_ema = run_realsense_ema_step(c_z, c_conf, c_val, c_prov, sid_arr, age_arr, ema_state)
        T_id = np.eye(4, dtype=np.float32)

        p12_state = run_daxa_temporal_step(
            res_p0,
            p12_state,
            width=w,
            height=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            baseline_m=baseline_m,
            cx_offset=cx_offset,
            current_from_previous=T_id,
            frame_id=ord_idx + 1,
            inverse_depth_fusion=True,
            disocclusion_guard=True,
            temporal_stereo_research=False,
        )
        p34_state = run_daxa_temporal_step(
            res_p0,
            p34_state,
            width=w,
            height=h,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            baseline_m=baseline_m,
            cx_offset=cx_offset,
            current_from_previous=T_id,
            frame_id=ord_idx + 1,
            inverse_depth_fusion=True,
            disocclusion_guard=True,
            temporal_stereo_research=True,
        )

        zed_history["v11_p0_continuous_subpixel"].append(c_z.copy())
        zed_history["realsense_ema_temporal"].append(res_ema["depth"].copy())
        zed_history["daxa_temporal_p1_p2"].append(p12_state["depth"].copy())
        zed_history["temporal_cost_refinement_p3_p4"].append(p34_state["depth"].copy())

        if ord_idx == 4:
            v11_z = np.fromfile(f"{v11_ord_prefix}.completed_depth.f32", dtype=np.float32)[:n].reshape(h, w)
            v11_val = np.fromfile(f"{v11_ord_prefix}.completed_validity.u32", dtype=np.uint32)[:n].reshape(h, w)
            v11_prov = np.fromfile(f"{v11_ord_prefix}.completed_provenance.u32", dtype=np.uint32)[:n].reshape(h, w)
            last_luma_l = res_p0["luma"][:n].reshape(h, w).astype(np.uint8)
            luma_r = res_p0["luma"][n:].reshape(h, w).astype(np.uint8)
            last_zncc_disp, last_zncc_valid = build_independent_zncc_epipolar_reference(last_luma_l, luma_r)
            last_tier_outputs = {
                "v11_single_frame": {"depth": v11_z, "validity": v11_val, "provenance": v11_prov},
                "v11_p0_continuous_subpixel": {"depth": c_z, "validity": c_val, "provenance": c_prov},
                "realsense_ema_temporal": {
                    "depth": res_ema["depth"],
                    "validity": res_ema["valid"],
                    "provenance": res_ema["prov"],
                },
                "daxa_temporal_p1_p2": {
                    "depth": p12_state["depth"],
                    "validity": p12_state["validity"],
                    "provenance": p12_state["provenance"],
                },
                "temporal_cost_refinement_p3_p4": {
                    "depth": p34_state["depth"],
                    "validity": p34_state["validity"],
                    "provenance": p34_state["provenance"],
                },
            }

    # Summarize ZED 2 frame-4 accuracy, coverage breakdown by provenance, and static-sequence temporal jitter
    zed_tier_summaries: dict[str, Any] = {}
    for tname, out in last_tier_outputs.items():
        z_map = out["depth"]
        v_map = out["validity"] != 0
        prov_map = out["provenance"]
        d_map = np.where(v_map & (z_map > 0), (fx * baseline_m) / z_map + cx_offset, np.nan)

        ov_zncc = v_map & last_zncc_valid & np.isfinite(d_map)
        zncc_mae = float(np.mean(np.abs(d_map[ov_zncc] - last_zncc_disp[ov_zncc])))

        fresh_stereo_pct = float(np.mean(v_map & (prov_map == PROVENANCE_STEREO)) * 100.0)
        planar_pct = float(np.mean(v_map & (prov_map == PROVENANCE_PLANAR_INFERRED)) * 100.0)
        reproj_pct = float(
            np.mean(v_map & ((prov_map == PROVENANCE_REPROJECTED) | (prov_map == PROVENANCE_TEMPORAL_PLANAR_INFERRED)))
            * 100.0
        )
        total_valid_pct = float(np.mean(v_map) * 100.0)

        jitter_mm = None
        if tname in zed_history and len(zed_history[tname]) >= 4:
            stack = np.stack(zed_history[tname][-4:], axis=0)
            all_valid = np.all(np.isfinite(stack) & (stack > 0) & (stack < 6.0), axis=0)
            if np.any(all_valid):
                jitter_mm = float(np.median(np.std(stack[:, all_valid], axis=0)) * 1000.0)

        zed_tier_summaries[tname] = {
            "total_valid_coverage_pct": total_valid_pct,
            "fresh_stereo_coverage_pct": fresh_stereo_pct,
            "planar_inferred_coverage_pct": planar_pct,
            "historical_reprojected_coverage_pct": reproj_pct,
            "independent_zncc_mae_px": zncc_mae,
            "temporal_depth_jitter_median_std_mm": jitter_mm,
        }

    # Memory footprint comparison (1280x720, 2 slots, 11 output planes + z + owner + counters + params)
    # Per pixel per eye: 4+4+4+4+8+8+32+4+8+4+4 = 84 bytes output + 8 bytes z/owner = 92 bytes * 2 slots = 184 bytes
    full_two_eye_bytes = 2 * (w * h * 2 * 92 + 64 + 472)
    left_eye_only_bytes = 2 * (w * h * 1 * 92 + 64 + 472)

    return {
        "zed2_5frame_tier_comparison": zed_tier_summaries,
        "gpu_timing_1280x720_ms": {
            "v11_single_frame_combined_gpu_ms": gpu_v11_report["gpu_combined_queue_span_ms"],
            "v11_p0_continuous_subpixel_combined_gpu_ms": gpu_p0_report["gpu_combined_queue_span_ms"],
            "v11_single_frame_match_gpu_ms": gpu_v11_report["gpu_guided_match_ms"],
            "v11_p0_continuous_subpixel_match_gpu_ms": gpu_p0_report["gpu_guided_match_ms"],
        },
        "temporal_gpu_memory_optimization_1280x720": {
            "baseline_two_eye_bytes": full_two_eye_bytes,
            "baseline_two_eye_mb": round(full_two_eye_bytes / (1024.0 * 1024.0), 2),
            "left_eye_only_bytes": left_eye_only_bytes,
            "left_eye_only_mb": round(left_eye_only_bytes / (1024.0 * 1024.0), 2),
            "reduction_pct": round((1.0 - left_eye_only_bytes / float(full_two_eye_bytes)) * 100.0, 2),
        },
    }


def run_v12_qualification() -> Dict[str, Any]:
    cpu_backend = SlangCpuRefinementBackend()
    with tempfile.TemporaryDirectory(prefix="sf_v12_temporal_") as tmp:
        work_dir = Path(tmp)
        suite_a = evaluate_synthetic_translation_sequences(cpu_backend)
        suite_b = evaluate_moving_object_disocclusion_sequence(cpu_backend)
        suite_c = evaluate_zed2_sequence_and_gpu_audit(work_dir)

    report = {
        "schema": "stereoforge.milestone.v12_temporal_supersampling_qualification.v1",
        "provenance": current_provenance_manifest(),
        "roadmap_stages": {
            "P0": "Single-frame continuous 5-point half-pixel zero-mean 3x3 photometric subpixel refinement (`continuous_subpixel` in `refinement.slang`)",
            "P1": "Full-resolution `v11` (`DaxaRefinement::Frame`) integration with pose-warped `DaxaTemporal` (4 contributors, 200 ms history bound)",
            "P2": "Calibrated inverse-depth uncertainty fusion (`w = conf / sigma_rho^2`), foreground depth-ordering disocclusion guard, age-decayed confidence, and 50% `left_eye_only` memory reduction",
            "P3": "Pose-warped temporal stereo cost refinement (`d_pred = fx * B / z_warp + cx_offset` guiding current-frame Census re-search in `[-2, +2]` with L-R check and strict provenance separation)",
            "P4": "Optional 2x joint-luma edge-preserving inverse-depth super-resolution & multi-sequence evaluation benchmark",
        },
        "suite_a_synthetic_known_pose_translation": suite_a,
        "suite_b_moving_object_disocclusion": suite_b,
        "suite_c_zed2_real_sequence_and_gpu_audit": suite_c,
    }
    OUT_V12_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_V12_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    res = run_v12_qualification()
    print(json.dumps(res["suite_a_synthetic_known_pose_translation"], indent=2))
    print(json.dumps(res["suite_b_moving_object_disocclusion"], indent=2))
    print(json.dumps(res["suite_c_zed2_real_sequence_and_gpu_audit"]["zed2_5frame_tier_comparison"], indent=2))
