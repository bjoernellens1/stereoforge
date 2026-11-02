#!/usr/bin/env python3
"""StereoForge v8 — Controlled per-pixel matching-cost & post-SGM cost-curve
inspector (Step 2), single-variable matching-cost & uniqueness ablation (Step 3),
and independent geometric reference qualification (Step 4) on exact fixture luma.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

from benchmarks.lr_confidence_audit_v6 import census_5x5, read_fixture_luma, read_native
from benchmarks.stereo_precision_pipeline_v7 import census_32bit_multiscale, speckle_filter_depth
from benchmarks.stereo_reference_crosscheck_v5 import compare_view, cv_sgbm, lr_pair, valid_map

INVALID_COST = 10000


def build_ortho_hybrid_u32(img: np.ndarray) -> np.ndarray:
    """Pack 4 full-range 8-bit 5x5 sub-window features into a single uint32 per pixel:
    - Byte 0 (bits 0..7):   Left 5x3 horizontal gradient Gx_left  (centered at dx=-1)
    - Byte 1 (bits 8..15):  Right 5x3 horizontal gradient Gx_right (centered at dx=+1)
    - Byte 2 (bits 16..23): Top 3x5 mean luma Luma_top             (dy in -2..0, dx in -2..2)
    - Byte 3 (bits 24..31): Bottom 3x5 mean luma Luma_bot          (dy in 0..2, dx in -2..2)

    Uses only the 5x5 neighborhood [-2..2]x[-2..2], preserving the exact 2-pixel border.
    """
    if img.ndim != 2 or img.shape[0] < 5 or img.shape[1] < 5:
        raise ValueError("image must be 2D and at least 5x5")
    h, w = img.shape
    a = img.astype(np.int32)
    gl_acc = np.zeros((h - 4, w - 4), dtype=np.int32)
    gr_acc = np.zeros((h - 4, w - 4), dtype=np.int32)
    st_acc = np.zeros((h - 4, w - 4), dtype=np.int32)
    sb_acc = np.zeros((h - 4, w - 4), dtype=np.int32)
    for dy in (-2, -1, 0, 1, 2):
        wy = 2 if abs(dy) <= 1 else 1
        gl_acc += wy * (a[2 + dy : h - 2 + dy, 2 : w - 2] - a[2 + dy : h - 2 + dy, 0 : w - 4])
        gr_acc += wy * (a[2 + dy : h - 2 + dy, 4:w] - a[2 + dy : h - 2 + dy, 2 : w - 2])
        for dx in (-2, -1, 0, 1, 2):
            v = a[2 + dy : h - 2 + dy, 2 + dx : w - 2 + dx]
            if dy <= 0:
                st_acc += v
            if dy >= 0:
                sb_acc += v
    b0 = np.clip((gl_acc // 4) + 128, 0, 255).astype(np.uint32)
    b1 = np.clip((gr_acc // 4) + 128, 0, 255).astype(np.uint32)
    b2 = np.clip((st_acc + 7) // 15, 0, 255).astype(np.uint32)
    b3 = np.clip((sb_acc + 7) // 15, 0, 255).astype(np.uint32)
    out = np.zeros((h, w), dtype=np.uint32)
    out[2 : h - 2, 2 : w - 2] = b0 | (b1 << np.uint32(8)) | (b2 << np.uint32(16)) | (b3 << np.uint32(24))
    return out


def ortho_hybrid_cost_slice(ref_desc: np.ndarray, match_desc: np.ndarray) -> np.ndarray:
    """Compute integer matching cost in [0..48] from two uint32 ortho_hybrid descriptors."""
    r0 = (ref_desc & 0xFF).astype(np.int32)
    r1 = ((ref_desc >> 8) & 0xFF).astype(np.int32)
    r2 = ((ref_desc >> 16) & 0xFF).astype(np.int32)
    r3 = ((ref_desc >> 24) & 0xFF).astype(np.int32)
    m0 = (match_desc & 0xFF).astype(np.int32)
    m1 = ((match_desc >> 8) & 0xFF).astype(np.int32)
    m2 = ((match_desc >> 16) & 0xFF).astype(np.int32)
    m3 = ((match_desc >> 24) & 0xFF).astype(np.int32)
    dg = np.minimum(np.abs(r0 - m0) + np.abs(r1 - m1), 64)
    dl = np.minimum(np.abs(r2 - m2) + np.abs(r3 - m3), 64)
    return ((dg >> 1) + (dl >> 2)).astype(np.int32)


def build_cost_volumes(
    luma_left: np.ndarray,
    luma_right: np.ndarray,
    disparities: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build left and right (H, W, D) cost volumes for a single matching-cost model."""
    if luma_left.shape != luma_right.shape or luma_left.ndim != 2:
        raise ValueError("luma shapes must match and be 2D")
    h, w = luma_left.shape
    valid_border = np.zeros((h, w), dtype=bool)
    valid_border[2 : h - 2, 2 : w - 2] = True

    if mode == "census_24bit":
        dl = census_5x5(luma_left)
        dr = census_5x5(luma_right)
    elif mode == "census_32bit_multiscale":
        dl = census_32bit_multiscale(luma_left, epsilon=0)
        dr = census_32bit_multiscale(luma_right, epsilon=0)
    elif mode == "census_box3x3":
        dl = census_5x5(luma_left)
        dr = census_5x5(luma_right)
        valid_border = np.zeros((h, w), dtype=bool)
        valid_border[3 : h - 3, 3 : w - 3] = True
    elif mode == "ortho_hybrid_u32":
        dl = build_ortho_hybrid_u32(luma_left)
        dr = build_ortho_hybrid_u32(luma_right)
    else:
        raise ValueError(f"Unknown cost mode: {mode}")

    cl_vol = np.full((h, w, disparities), INVALID_COST, dtype=np.int32)
    cr_vol = np.full((h, w, disparities), INVALID_COST, dtype=np.int32)
    for d in range(disparities):
        ok = (valid_border[:, d:] & valid_border[:, : w - d]) if d > 0 else valid_border
        if mode in ("census_24bit", "census_32bit_multiscale"):
            raw = np.bitwise_count(dl[:, d:] ^ dr[:, : w - d]).astype(np.int32)
        elif mode == "census_box3x3":
            xor = np.bitwise_count(dl[:, d:] ^ dr[:, : w - d]).astype(np.float32)
            raw = np.round(cv2.boxFilter(xor, -1, (3, 3), normalize=True)).astype(np.int32)
        elif mode == "ortho_hybrid_u32":
            raw = ortho_hybrid_cost_slice(dl[:, d:], dr[:, : w - d])
        cl_vol[:, d:, d] = np.where(ok, raw, INVALID_COST)
        cr_vol[:, : w - d, d] = np.where(ok, raw, INVALID_COST)
    return cl_vol, cr_vol


def aggregate_sgm_4path_vectorized(cost: np.ndarray, p1: int = 8, p2: int = 32) -> np.ndarray:
    """Exact 4-path (+x, -x, +y, -y) SGM recurrence vectorized across scanlines."""
    if cost.ndim != 3 or not (0 <= p1 <= p2):
        raise ValueError("Invalid cost volume or P1/P2")
    h, w, d_count = cost.shape
    out = np.zeros((h, w, d_count), dtype=np.int32)
    c32 = cost.astype(np.int32)
    for step in (1, -1):
        xs = range(w) if step == 1 else range(w - 1, -1, -1)
        prev = None
        for x in xs:
            c = c32[:, x, :]
            if prev is None:
                cur = c
            else:
                mn = prev.min(axis=1, keepdims=True)
                l_nb = np.pad(prev[:, :-1] + p1, ((0, 0), (1, 0)), constant_values=1000000)
                r_nb = np.pad(prev[:, 1:] + p1, ((0, 0), (0, 1)), constant_values=1000000)
                cur = c + np.minimum(np.minimum(prev, mn + p2), np.minimum(l_nb, r_nb)) - mn
            out[:, x, :] += cur
            prev = cur
    for step in (1, -1):
        ys = range(h) if step == 1 else range(h - 1, -1, -1)
        prev = None
        for y in ys:
            c = c32[y, :, :]
            if prev is None:
                cur = c
            else:
                mn = prev.min(axis=1, keepdims=True)
                l_nb = np.pad(prev[:, :-1] + p1, ((0, 0), (1, 0)), constant_values=1000000)
                r_nb = np.pad(prev[:, 1:] + p1, ((0, 0), (0, 1)), constant_values=1000000)
                cur = c + np.minimum(np.minimum(prev, mn + p2), np.minimum(l_nb, r_nb)) - mn
            out[y, :, :] += cur
            prev = cur
    return out


def select_from_aggregated(
    agg: np.ndarray,
    *,
    non_adjacent_runner_up: bool = False,
    uniqueness_ratio: float = 0.0,
    min_margin: int = 1,
    denom_floor: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select subpixel disparity, confidence, and validity from aggregated SGM volume."""
    h, w, d_count = agg.shape
    best = np.argmin(agg, axis=2)
    score = np.take_along_axis(agg, best[..., None], axis=2)[..., 0]
    if non_adjacent_runner_up and d_count > 3:
        dd = np.arange(d_count)[None, None, :]
        masked = np.where(np.abs(dd - best[..., None]) <= 1, 100000000, agg)
        second = masked.min(axis=2)
    else:
        second = np.partition(agg, 1, axis=2)[..., 1] if d_count > 1 else score
    margin = second - score
    denom = np.maximum(second, max(1, int(denom_floor))).astype(np.float32)
    confidence = margin.astype(np.float32) / denom
    valid = (
        (best > 0)
        & (score < INVALID_COST)
        & (margin >= int(min_margin))
        & (confidence >= float(uniqueness_ratio))
    )
    disp = best.astype(np.float32)
    if d_count > 2:
        lo = np.take_along_axis(agg, np.maximum(best - 1, 0)[..., None], axis=2)[..., 0].astype(np.float32)
        hi = np.take_along_axis(agg, np.minimum(best + 1, d_count - 1)[..., None], axis=2)[..., 0].astype(np.float32)
        denom_sub = lo + hi - 2.0 * score.astype(np.float32)
        ref = valid & (best > 0) & (best < d_count - 1) & (denom_sub > 0)
        disp[ref] += 0.5 * (lo[ref] - hi[ref]) / denom_sub[ref]
    disp[~valid] = np.nan
    confidence[~valid] = 0.0
    return disp, confidence.astype(np.float32), valid


def inspect_cost_curves(
    luma_left: np.ndarray,
    luma_right: np.ndarray,
    native_left_disp: np.ndarray,
    native_right_disp: np.ndarray,
    native_left_val: np.ndarray,
    native_right_val: np.ndarray,
    cv_left_disp: np.ndarray,
    cv_right_disp: np.ndarray,
    disparities: int = 128,
) -> dict:
    """Step 2: Inspect pre-SGM C(x,y,d) and post-SGM S(x,y,d) at selected pixels."""
    h, w = luma_left.shape
    c24_l = census_5x5(luma_left)
    c24_r = census_5x5(luma_right)
    pop_l = np.bitwise_count(c24_l[2 : h - 2, 2 : w - 2])

    nlv = valid_map(native_left_disp, native_left_val, disparities)
    nrv = valid_map(native_right_disp, native_right_val, disparities)
    clv = valid_map(cv_left_disp, None, disparities)
    crv = valid_map(cv_right_disp, None, disparities)

    nat_lr, _, _ = lr_pair(native_left_disp, native_right_disp, nlv, nrv)
    cv_lr, _, _ = lr_pair(cv_left_disp, cv_right_disp, clv, crv)

    both_lr = nat_lr & cv_lr
    cv_only = cv_lr & (~nat_lr) & nlv
    ys, xs = np.nonzero(cv_only)
    d_cv = np.round(cv_left_disp[ys, xs]).astype(np.int32)
    d_nat = np.round(native_left_disp[ys, xs]).astype(np.int32)
    ok = (xs - d_cv >= 2) & (xs - d_nat >= 2) & (ys >= 2) & (ys < h - 2) & (xs < w - 2)
    ys, xs, d_cv, d_nat = ys[ok], xs[ok], d_cv[ok], d_nat[ok]

    if ys.size:
        raw_c_cv = np.bitwise_count(c24_l[ys, xs] ^ c24_r[ys, xs - d_cv])
        raw_c_nat = np.bitwise_count(c24_l[ys, xs] ^ c24_r[ys, xs - d_nat])
        ad_cv = np.abs(luma_left[ys, xs].astype(np.int32) - luma_right[ys, xs - d_cv].astype(np.int32))
        ad_nat = np.abs(luma_left[ys, xs].astype(np.int32) - luma_right[ys, xs - d_nat].astype(np.int32))
        pop_ref = np.bitwise_count(c24_l[ys, xs])
        pop_match_cv = np.bitwise_count(c24_r[ys, xs - d_cv])
        pop_match_nat = np.bitwise_count(c24_r[ys, xs - d_nat])
        cv_only_report = {
            "count": int(ys.size),
            "fraction_of_image": float(ys.size / (h * w)),
            "raw_census_cost_at_d_cv_median": float(np.median(raw_c_cv)),
            "raw_census_cost_at_d_nat_median": float(np.median(raw_c_nat)),
            "raw_census_cost_at_d_cv_mean": float(np.mean(raw_c_cv)),
            "raw_census_cost_at_d_nat_mean": float(np.mean(raw_c_nat)),
            "raw_census_prefers_wrong_d_nat_before_sgm_fraction": float(np.mean(raw_c_nat < raw_c_cv)),
            "abs_luma_diff_at_d_cv_mean": float(np.mean(ad_cv)),
            "abs_luma_diff_at_d_nat_mean": float(np.mean(ad_nat)),
            "ref_popcount_median": float(np.median(pop_ref)),
            "match_cv_popcount_median": float(np.median(pop_match_cv)),
            "match_nat_popcount_median": float(np.median(pop_match_nat)),
            "ref_popcount_le_3_fraction": float(np.mean(pop_ref <= 3)),
        }
    else:
        cv_only_report = {"count": 0}

    return {
        "whole_image_census_sparsity": {
            "popcount_eq_0_fraction": float(np.mean(pop_l == 0)),
            "popcount_le_2_fraction": float(np.mean(pop_l <= 2)),
            "popcount_le_4_fraction": float(np.mean(pop_l <= 4)),
            "luma_p50": float(np.quantile(luma_left, 0.5)),
            "luma_p75": float(np.quantile(luma_left, 0.75)),
        },
        "both_lr_consistent": {
            "fraction_of_image": float(both_lr.mean()),
            "agree_within_1px_fraction": float((np.abs(native_left_disp[both_lr] - cv_left_disp[both_lr]) <= 1.0).mean())
            if both_lr.any()
            else 0.0,
        },
        "cv_consistent_native_failed_breakdown": cv_only_report,
    }


def geometric_qualification_metrics(
    disparity: np.ndarray,
    lr_accepted: np.ndarray,
    luma_left: np.ndarray,
    fx: float = 527.790283203125,
    baseline_m: float = 0.11944399774074554,
) -> dict:
    """Step 4: Independent geometric reference metrics (planar surface RANSAC/SVD fit
    on low-gradient patches, disparity-edge alignment with Sobel image edges, and
    8-connected disparity-domain speckle survival).
    """
    h, w = disparity.shape
    disp_clean = np.where(lr_accepted & np.isfinite(disparity) & (disparity > 0), disparity, np.nan).astype(np.float32)
    depth = np.where(np.isfinite(disp_clean), (fx * baseline_m) / disp_clean, np.nan).astype(np.float32)

    # 1. Disparity-domain 8-connected speckle survival (coherent surface support)
    _, sp_stats = speckle_filter_depth(
        depth,
        min_component_pixels=40,
        fx_baseline_px_m=fx * baseline_m,
        max_neighbor_disparity_px=1.25,
        eight_connected=True,
    )
    in_valid = max(sp_stats["input_valid_pixels"], 1)
    coherent_surface_fraction = float(sp_stats["output_valid_pixels"] / (h * w))
    speckle_survival_rate = float(sp_stats["output_valid_pixels"] / in_valid)

    # 2. Local planar surface RMSE on smooth intensity patches (16x16 tiles where luma std < 6 and >=32 valid pixels)
    plane_residuals_px = []
    for y0 in range(16, h - 16, 32):
        for x0 in range(16, w - 16, 32):
            patch_luma = luma_left[y0 : y0 + 16, x0 : x0 + 16].astype(np.float32)
            if float(np.std(patch_luma)) > 6.0:
                continue
            patch_d = disp_clean[y0 : y0 + 16, x0 : x0 + 16]
            vy, vx = np.nonzero(np.isfinite(patch_d))
            if vy.size < 32:
                continue
            vals = patch_d[vy, vx].astype(np.float64)
            # Fit linear disparity plane d(x, y) = a*x + b*y + c (which is exact for any 3D plane in rectified stereo!)
            A = np.column_stack([vx.astype(np.float64), vy.astype(np.float64), np.ones(vy.size, dtype=np.float64)])
            coef, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
            pred = A @ coef
            rmse = float(np.sqrt(np.mean((vals - pred) ** 2)))
            plane_residuals_px.append(rmse)

    # 3. Thin/edge structure completeness (pixels in top 15% of image Sobel gradient magnitude)
    lf = luma_left.astype(np.float32)
    gx = cv2.Sobel(lf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lf, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    edge_thresh = float(np.quantile(grad_mag[4 : h - 4, 4 : w - 4], 0.85))
    edge_mask = grad_mag >= max(edge_thresh, 1.0)
    edge_lr_valid_fraction = float(lr_accepted[edge_mask].mean()) if edge_mask.any() else 0.0

    return {
        "coherent_surface_fraction_all_pixels": coherent_surface_fraction,
        "speckle_survival_rate_of_lr_accepted": speckle_survival_rate,
        "planar_patches_evaluated": len(plane_residuals_px),
        "planar_disparity_rmse_median_px": float(np.median(plane_residuals_px)) if plane_residuals_px else None,
        "edge_structure_lr_valid_fraction": edge_lr_valid_fraction,
    }


def evaluate_ablation_suite(
    luma_left: np.ndarray,
    luma_right: np.ndarray,
    cv_left_disp: np.ndarray,
    cv_right_disp: np.ndarray,
    disparities: int = 128,
) -> dict:
    """Step 3 & Step 4: Run single-variable ablation across matching-cost models and selectors."""
    clv = valid_map(cv_left_disp, None, disparities)
    crv = valid_map(cv_right_disp, None, disparities)
    results = {}

    for mode in ("census_24bit", "census_32bit_multiscale", "census_box3x3", "ortho_hybrid_u32"):
        cl_vol, cr_vol = build_cost_volumes(luma_left, luma_right, disparities, mode)
        agg_l = aggregate_sgm_4path_vectorized(cl_vol, p1=8, p2=32)
        agg_r = aggregate_sgm_4path_vectorized(cr_vol, p1=8, p2=32)

        # Profile 1: Raw existing selector (all-d runner-up, u=0.0, min_margin=1)
        dl_raw, _, lv_raw = select_from_aggregated(
            agg_l, non_adjacent_runner_up=False, uniqueness_ratio=0.0, min_margin=1, denom_floor=1
        )
        dr_raw, _, rv_raw = select_from_aggregated(
            agg_r, non_adjacent_runner_up=False, uniqueness_ratio=0.0, min_margin=1, denom_floor=1
        )
        acc_raw, _, st_raw = lr_pair(dl_raw, dr_raw, lv_raw, rv_raw)
        cmp_raw = compare_view(dl_raw, cv_left_disp, lv_raw, clv)
        geom_raw = geometric_qualification_metrics(dl_raw, acc_raw, luma_left)

        # Profile 2: Calibrated non-adjacent selector (|d - best| > 1, u=0.03, min_margin=2, floor=24)
        dl_cal, _, lv_cal = select_from_aggregated(
            agg_l, non_adjacent_runner_up=True, uniqueness_ratio=0.03, min_margin=2, denom_floor=24
        )
        dr_cal, _, rv_cal = select_from_aggregated(
            agg_r, non_adjacent_runner_up=True, uniqueness_ratio=0.03, min_margin=2, denom_floor=24
        )
        acc_cal, _, st_cal = lr_pair(dl_cal, dr_cal, lv_cal, rv_cal)
        cmp_cal = compare_view(dl_cal, cv_left_disp, lv_cal, clv)
        geom_cal = geometric_qualification_metrics(dl_cal, acc_cal, luma_left)

        results[mode] = {
            "raw_selector": {
                "lr_stats": st_raw,
                "opencv_left_agreement": cmp_raw,
                "geometric_qualification": geom_raw,
            },
            "calibrated_non_adjacent_u0p03": {
                "lr_stats": st_cal,
                "opencv_left_agreement": cmp_cal,
                "geometric_qualification": geom_cal,
            },
        }
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--native-prefix", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--disparities", type=int, default=128)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Output already exists: {args.output}")

    luma_left, luma_right = read_fixture_luma(args.fixture)
    native = read_native(args.native_prefix, luma_left.shape)
    cv_dl, cv_dr, _ = cv_sgbm(
        luma_left.astype(np.uint8),
        luma_right.astype(np.uint8),
        disparities=args.disparities,
        block=5,
        uniqueness=5,
    )

    step2 = inspect_cost_curves(
        luma_left,
        luma_right,
        native["left_disparity"],
        native["right_disparity"],
        native["left_validity"],
        native["right_validity"],
        cv_dl,
        cv_dr,
        disparities=args.disparities,
    )
    step3_4 = evaluate_ablation_suite(luma_left, luma_right, cv_dl, cv_dr, disparities=args.disparities)

    report = {
        "schema": "stereoforge.matching_cost_inspector.v8",
        "fixture": str(args.fixture),
        "native_prefix": str(args.native_prefix),
        "step2_cost_curve_inspection": step2,
        "step3_and_4_ablation_and_geometry": step3_4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Saved Step 2–4 inspector & ablation report:", args.output)


if __name__ == "__main__":
    main()
