"""StereoForge v9.4 — Dense Low-Texture Surface Recovery:
11-Stage Unrecovered-Pixel Taxonomy, Coplanarity-Aware Spatial Planar Propagation (Provenance = 6),
Multi-Frame Temporal Accumulation (Provenance = 2, 3), Combined Temporal + Planar Completion
(Provenance = 7), and Multi-Reference Geometric & Counterexample Qualification.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from benchmarks.planar_recovery_v93 import (
    _eval_plane_cohort,
    _fit_3d_plane_ransac,
    _popcount32,
    compute_ortho_thermo32,
)
from benchmarks.visual_accuracy_audit_v9 import (
    COARSE_4X_MANIFEST,
    DEFAULT_MANIFEST,
    DUAL_BENCH,
    REFINE_BENCH,
    build_independent_zncc_epipolar_reference,
    current_provenance_manifest,
    evaluate_known_ground_truth_suite_720p,
    gpu_env,
)
from stereoforge.types import Provenance


def _desk_rois(h: int = 720, w: int = 1280) -> dict[str, np.ndarray]:
    tag_poly = np.array([[785, 404], [831, 404], [831, 450], [785, 450]], dtype=np.int32)
    tag_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(tag_mask, [tag_poly], 1)
    tag_b = tag_mask.astype(bool)

    charuco_b = np.zeros((h, w), dtype=bool)
    charuco_b[450:463, 730:895] = True

    wood_clean_b = np.zeros((h, w), dtype=bool)
    wood_clean_b[445:463, 600:730] = True

    desk_clean_b = np.zeros((h, w), dtype=bool)
    desk_clean_b[445:463, 600:925] = True
    desk_clean_b &= ~tag_b

    broad_inrange_b = np.zeros((h, w), dtype=bool)
    broad_inrange_b[380:463, 480:920] = True

    broad_outofrange_b = np.zeros((h, w), dtype=bool)
    broad_outofrange_b[463:520, 480:920] = True

    return {
        "upright_apriltag_target": tag_b,
        "charuco_board_inrange": charuco_b,
        "horizontal_wood_desk_clean": wood_clean_b,
        "horizontal_desk_plane_clean": desk_clean_b,
        "broad_desk_inrange": broad_inrange_b,
        "broad_desk_outofrange_d_gt_127": broad_outofrange_b,
    }


def compute_unrecovered_pixel_taxonomy_and_relaxation_matrix(
    d_meas: np.ndarray,
    v_meas: np.ndarray,
    d_comp_v94: np.ndarray,
    prov_comp_v94: np.ndarray,
    pred_coarse: np.ndarray,
    pred_valid: np.ndarray,
    luma_l: np.ndarray,
    luma_r: np.ndarray,
    zncc_d_128: np.ndarray,
    zncc_v_128: np.ndarray,
    zncc_d_192: np.ndarray,
    zncc_v_192: np.ndarray,
    zed_intr: dict[str, float],
) -> dict[str, Any]:
    """Compute the 11-stage unrecovered-pixel rejection waterfall and single-guard relaxation
    ceiling matrix across all tabletop ROIs, plus the coarse SGM d_coarse=31.0 saturation diagnostic."""
    h, w = d_meas.shape
    rois = _desk_rois(h, w)
    l_f = luma_l.astype(np.float32)
    r_f = luma_r.astype(np.float32)
    c_l = compute_ortho_thermo32(l_f)
    c_r = compute_ortho_thermo32(r_f)

    # Precompute 4-neighbor and 45-deg diagonal pre-screen from v9.3 for exact attribution
    gx = np.abs(np.roll(l_f, -1, axis=1) - np.roll(l_f, 1, axis=1))
    gy = np.abs(np.roll(l_f, -1, axis=0) - np.roll(l_f, 1, axis=0))
    grad_l1 = gx + gy

    # 8-tap 45-degree diagonal qpre pre-screen used in v9.3
    qpre = np.zeros((h, w), dtype=np.uint8)
    for qbit, (dx1, dy1, dx2, dy2) in enumerate(
        [(-3, -3, -6, -6), (3, -3, 6, -6), (-3, 3, -6, 6), (3, 3, 6, 6)]
    ):
        v1 = np.roll(np.roll(v_meas, -dy1, axis=0), -dx1, axis=1)
        v2 = np.roll(np.roll(v_meas, -dy2, axis=0), -dx2, axis=1)
        qpre += (v1 | v2).astype(np.uint8)

    # 36-tap kOff stencil counts under v9.3 hard 18-DN cliff vs v9.4 soft-bilateral coplanar stencil
    k_off = [-9, -5, -2, 2, 5, 9]
    n_anc_v93 = np.zeros((h, w), dtype=np.int32)
    qmask_v93 = np.zeros((h, w), dtype=np.uint8)
    n_anc_v94 = np.zeros((h, w), dtype=np.int32)
    qmask_v94 = np.zeros((h, w), dtype=np.uint8)
    n_contra_v94 = np.zeros((h, w), dtype=np.int32)

    for dy in k_off:
        qy = 2 if dy > 0 else 0
        for dx in k_off:
            qbit = 1 << (qy + (1 if dx > 0 else 0))
            va = np.roll(np.roll(v_meas, -dy, axis=0), -dx, axis=1)
            da = np.roll(np.roll(d_meas, -dy, axis=0), -dx, axis=1)
            ga = np.roll(np.roll(l_f, -dy, axis=0), -dx, axis=1)
            pca = np.roll(np.roll(pred_coarse, -dy, axis=0), -dx, axis=1)
            pva = np.roll(np.roll(pred_valid, -dy, axis=0), -dx, axis=1)
            dc_ref = np.where(pva & np.isfinite(pca), pca, pred_coarse)
            u_res = np.abs(da - dc_ref)
            l_diff = np.abs(ga - l_f)
            dc_tol = np.where(pred_coarse >= 114.0, 2.6, 1.8)

            ok93 = va & (u_res <= 3.0) & (l_diff <= 18.0)
            n_anc_v93 += ok93.astype(np.int32)
            qmask_v93 |= np.where(ok93, np.uint8(qbit), np.uint8(0))

            ok94 = va & (u_res <= dc_tol)
            n_anc_v94 += ok94.astype(np.int32)
            qmask_v94 |= np.where(ok94, np.uint8(qbit), np.uint8(0))
            n_contra_v94 += (va & (u_res > 3.2)).astype(np.int32)

    qcnt_v93 = (
        ((qmask_v93 & 1) != 0).astype(np.int32)
        + ((qmask_v93 & 2) != 0).astype(np.int32)
        + ((qmask_v93 & 4) != 0).astype(np.int32)
        + ((qmask_v93 & 8) != 0).astype(np.int32)
    )
    qcnt_v94 = (
        ((qmask_v94 & 1) != 0).astype(np.int32)
        + ((qmask_v94 & 2) != 0).astype(np.int32)
        + ((qmask_v94 & 4) != 0).astype(np.int32)
        + ((qmask_v94 & 8) != 0).astype(np.int32)
    )

    # Epipolar evaluation at pred_coarse
    yy, xx = np.indices((h, w), dtype=np.float32)
    xr = np.clip(xx - np.where(np.isfinite(pred_coarse), pred_coarse, 0.0), 1.0, w - 2.0).astype(np.float32)
    r_pix = cv2.remap(r_f, xr, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    l_mean5 = cv2.blur(l_f, (5, 5))
    r_mean5 = cv2.remap(cv2.blur(r_f, (5, 5)), xr, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rx_i = np.clip(np.rint(xr).astype(np.int32), 1, w - 2)
    ry_i = np.clip(yy.astype(np.int32), 1, h - 2)
    base_h = _popcount32(c_l ^ c_r[ry_i, rx_i])
    cost_5tap = (
        2 * base_h
        + _popcount32(np.roll(c_l, 1, axis=1) ^ c_r[ry_i, np.clip(rx_i - 1, 0, w - 1)])
        + _popcount32(np.roll(c_l, -1, axis=1) ^ c_r[ry_i, np.clip(rx_i + 1, 0, w - 1)])
        + _popcount32(np.roll(c_l, 1, axis=0) ^ c_r[np.clip(ry_i - 1, 0, h - 1), rx_i])
        + _popcount32(np.roll(c_l, -1, axis=0) ^ c_r[np.clip(ry_i + 1, 0, h - 1), rx_i])
    )

    waterfall_by_roi: dict[str, Any] = {}
    for rname, rmask in rois.items():
        tot = int(rmask.sum())
        meas = int((rmask & v_meas).sum())
        holes = rmask & (~v_meas)
        n_holes = int(holes.sum())

        # Out-of-range ground truth check using extended 192-disparity ZNCC
        oor_mask = holes & ((yy >= 463) | (zncc_v_192 & (zncc_d_192 > 127.0)))
        in_range_holes = holes & (~oor_mask)
        no_coarse = in_range_holes & ((~pred_valid) | (~np.isfinite(pred_coarse)) | (pred_coarse >= 127.0))
        has_coarse = in_range_holes & (~no_coarse)

        # v9.3 waterfall stages on has_coarse
        fail_qpre93 = has_coarse & (qpre < 2)
        rem1 = has_coarse & (~fail_qpre93)
        fail_grad93 = rem1 & (grad_l1 > 28.0)
        rem2 = rem1 & (~fail_grad93)
        fail_anc93 = rem2 & ((n_anc_v93 < 10) | (qcnt_v93 < 3))
        rem3 = rem2 & (~fail_anc93)
        fail_epi_luma93 = rem3 & ((np.abs(l_f - r_pix) > 18.0) | (np.abs(l_mean5 - r_mean5) > 12.0))
        rem4 = rem3 & (~fail_epi_luma93)
        fail_epi_cost93 = rem4 & (cost_5tap > 28)
        pass_v93_est = rem4 & (~fail_epi_cost93)

        # v9.4 waterfall stages on has_coarse
        fail_contra94 = has_coarse & (n_contra_v94 > 0)
        rem94_1 = has_coarse & (~fail_contra94)
        fail_anc94 = rem94_1 & ((n_anc_v94 < 3) | (qcnt_v94 < 2))
        rem94_2 = rem94_1 & (~fail_anc94)
        fail_epi_luma94 = rem94_2 & (
            (np.abs(l_f - r_pix) > (18.0 + 0.325 * gx))
            | (np.abs(l_mean5 - r_mean5) > (12.0 + 0.175 * gx))
        )
        rem94_3 = rem94_2 & (~fail_epi_luma94)
        fail_epi_cost94 = rem94_3 & (cost_5tap > 30)
        admitted_v94 = rmask & (prov_comp_v94 == int(Provenance.PLANAR_INFERRED))

        waterfall_by_roi[rname] = {
            "total_roi_pixels": tot,
            "measured_stereo_prov1_pixels": meas,
            "measured_coverage_pct": round(100.0 * meas / max(tot, 1), 2),
            "unrecovered_hole_pixels": n_holes,
            "stage_1_out_of_disparity_range_d_gt_127_pixels": int(oor_mask.sum()),
            "stage_2_no_valid_coarse_prior_pixels": int(no_coarse.sum()),
            "stage_3_valid_coarse_candidates_pixels": int(has_coarse.sum()),
            "v93_guard_rejections_on_coarse_candidates": {
                "rejected_by_qpre_45deg_diagonal_prescreen": int(fail_qpre93.sum()),
                "rejected_by_blind_28dn_luma_gradient_gate": int(fail_grad93.sum()),
                "rejected_by_hard_18dn_anchor_count_and_3quadrants": int(fail_anc93.sum()),
                "rejected_by_fixed_epipolar_luma_gate": int(fail_epi_luma93.sum()),
                "rejected_by_ortho_cost_gt_28": int(fail_epi_cost93.sum()),
                "surviving_v93_candidates": int(pass_v93_est.sum()),
            },
            "v94_guard_rejections_on_coarse_candidates": {
                "rejected_by_step_edge_contradiction_guard": int(fail_contra94.sum()),
                "rejected_by_insufficient_coplanar_anchors": int(fail_anc94.sum()),
                "rejected_by_gradient_normalized_epipolar_luma": int(fail_epi_luma94.sum()),
                "rejected_by_ortho_cost_gt_30": int(fail_epi_cost94.sum()),
                "admitted_gpu_v94_prov6_pixels": int(admitted_v94.sum()),
                "completed_gpu_v94_coverage_pct": round(
                    100.0 * (meas + int(admitted_v94.sum())) / max(tot, 1), 2
                ),
            },
        }

    # Single-guard relaxation ceiling matrix on horizontal_desk_plane_clean
    desk_mask = rois["horizontal_desk_plane_clean"]
    n_z, d_off_z, d_map_z = _fit_3d_plane_ransac(
        zncc_d_128, desk_mask & zncc_v_128, zed_intr, thresh_mm=3.5
    )
    holes_desk = desk_mask & (~v_meas) & pred_valid & np.isfinite(pred_coarse) & (pred_coarse < 127.0)
    g_qpre = qpre >= 2
    g_grad = grad_l1 <= 28.0
    g_anc93 = (n_anc_v93 >= 10) & (qcnt_v93 >= 3)
    g_anc94 = (n_anc_v94 >= 5) & (qcnt_v94 >= 2) & (n_contra_v94 == 0)
    g_luma93 = (np.abs(l_f - r_pix) <= 18.0) & (np.abs(l_mean5 - r_mean5) <= 12.0)
    g_luma94 = (np.abs(l_f - r_pix) <= (18.0 + 0.325 * gx)) & (
        np.abs(l_mean5 - r_mean5) <= (12.0 + 0.175 * gx)
    )
    g_cost93 = cost_5tap <= 28
    g_cost94 = cost_5tap <= 30

    relaxation_variants = {
        "0_v93_all_guards_strict": holes_desk & g_qpre & g_grad & g_anc93 & g_luma93 & g_cost93,
        "1_relax_only_qpre_diagonal": holes_desk & g_grad & g_anc93 & g_luma93 & g_cost93,
        "2_relax_only_luma_gradient_gate": holes_desk & g_qpre & g_anc93 & g_luma93 & g_cost93,
        "3_relax_only_hard_18dn_anchor_cliff_to_coplanar_bilateral": (
            holes_desk & g_qpre & g_grad & g_anc94 & g_luma93 & g_cost93
        ),
        "4_relax_only_epipolar_to_gradient_normalized": (
            holes_desk & g_qpre & g_grad & g_anc93 & g_luma94 & g_cost94
        ),
        "5_relax_qpre_plus_gradient_plus_coplanar_anchors": (
            holes_desk & g_anc94 & g_luma93 & g_cost93
        ),
        "6_v94_combined_coplanar_bilateral_and_bt_epipolar": (
            holes_desk & g_anc94 & g_luma94 & g_cost94
        ),
        "7_unguarded_coarse_extrapolation_ceiling": holes_desk,
    }
    relaxation_matrix: dict[str, Any] = {}
    roi_tot = int(desk_mask.sum())
    for vkey, vmask in relaxation_variants.items():
        d_eval = np.where(vmask, pred_coarse, d_comp_v94)
        relaxation_matrix[vkey] = _eval_plane_cohort(
            d_eval, vmask, roi_tot, d_map_z, n_z, d_off_z, zed_intr
        )

    # Diagnostic quantification of coarse SGM best==31 (124.0 px) saturation on y >= 463
    oor_roi = rois["broad_desk_outofrange_d_gt_127"]
    oor_meas = oor_roi & v_meas
    oor_clamped_124 = oor_meas & (np.abs(d_meas - 124.0) <= 2.05)
    oor_zncc = oor_roi & zncc_v_192

    return {
        "waterfall_by_roi": waterfall_by_roi,
        "single_guard_relaxation_ceiling_matrix_desk_clean": relaxation_matrix,
        "near_field_d_gt_127_saturation_diagnostic": {
            "roi": "broad_desk_outofrange_d_gt_127 (y in [463, 520), x in [480, 920))",
            "total_pixels": int(oor_roi.sum()),
            "extended_zncc_192_valid_pixels": int(oor_zncc.sum()),
            "extended_zncc_192_disparity_p10_p50_p90_px": (
                [
                    round(float(np.percentile(zncc_d_192[oor_zncc], p)), 2)
                    for p in (10, 50, 90)
                ]
                if int(oor_zncc.sum()) > 0
                else None
            ),
            "measured_stereo_false_in_range_pixels": int(oor_meas.sum()),
            "measured_stereo_clamped_near_124px_pixels": int(oor_clamped_124.sum()),
            "clamped_share_of_measured_in_oor_pct": round(
                100.0 * int(oor_clamped_124.sum()) / max(int(oor_meas.sum()), 1), 2
            ),
            "root_cause": (
                "At 4x coarse resolution with D_coarse=32 (0..31 coarse px = 0..124 full-res px), "
                "select_output in stereo.slang skips subpixel parabola interpolation when best == 31 "
                "(best + 1 < disparities is false), clamping coarse_disparity to exact 31.000 (124.0 px). "
                "Full-res match then searches [122, 126] px and admits false matches at d≈124 px where true "
                "tabletop disparity is 132..170 px (Z < 0.496 m). Milestone v9.4 strictly isolates y >= 463 "
                "as out-of-range (never extrapolating beyond d = 127 px), while Milestone v10.2 evaluates "
                "D_coarse = 64 (256 px full-res range) to recover the near-field tabletop."
            ),
        },
    }


def run_multiframe_4mode_surface_recovery(
    bundle_dir: Path,
    coarse_manifest: Path,
    work_dir: Path,
    scene_name: str,
    rois: dict[str, np.ndarray],
    intr: dict[str, float],
    *,
    num_frames: int = 8,
    fast_mode: str = "0",
) -> dict[str, Any]:
    """Run all 8 frames of a stereo capture bundle through the GPU refinement + planar pipeline
    and compare the 4 recovery modes with strict provenance separation (0, 1, 2, 3, 6, 7):
      Mode 1: Measured Stereo Only (Provenance = 1)
      Mode 2: Spatial Planar Propagation Only (Provenance = 1, 6)
      Mode 3: Motion-Compensated Temporal Accumulation Only (Provenance = 1, 2, 3)
      Mode 4: Combined Temporal + Spatial Planar Propagation (Provenance = 1, 2, 3, 6, 7)
    """
    work = work_dir / f"seq_{scene_name}"
    work.mkdir(parents=True, exist_ok=True)
    env = gpu_env()
    h, w = 720, 1280
    cap_path = bundle_dir / "capture/capture.sfp"
    man_data = json.loads((bundle_dir / "manifest.json").read_text())

    d_meas_seq: list[np.ndarray] = []
    v_meas_seq: list[np.ndarray] = []
    d_spatial_seq: list[np.ndarray] = []
    v_spatial_seq: list[np.ndarray] = []
    prov_spatial_seq: list[np.ndarray] = []
    pred_seq: list[np.ndarray] = []
    pred_v_seq: list[np.ndarray] = []
    luma_l_seq: list[np.ndarray] = []
    luma_r_seq: list[np.ndarray] = []

    # Accumulate across the 8 recorded stereo frames ending on canonical record_ordinal = 0
    ordinals = list(reversed(range(num_frames)))
    for step_idx, ord_idx in enumerate(ordinals):
        pref = work / f"out_ord{ord_idx}"
        if not Path(str(pref) + ".completed_disparity.f32").exists():
            subprocess.run(
                [
                    str(REFINE_BENCH),
                    "--coarse-manifest", str(coarse_manifest),
                    "--target-manifest", str(bundle_dir / "manifest.json"),
                    "--record-ordinal", str(ord_idx),
                    "--variant", "raw_borrow",
                    "--frames", "1", "--warmup", "0",
                    "--coarse-disparities", "32",
                    "--coarse-schedule", "layered-per-path",
                    "--coarse-right-reference-mode", "independent-sgm",
                    "--descriptor-mode", "ortho-hybrid",
                    "--fast-mode", fast_mode,
                    "--selective-repair", "1",
                    "--planar-recovery", "1",
                    "--planar-min-anchors", "10",
                    "--planar-coarse-affine", "1",
                    "--planar-ray-crossing-guard", "1",
                    "--planar-coarse-lr-guard", "1",
                    "--radius", "2", "--subpixel", "1",
                    "--dump-prefix", str(pref),
                    "--output", str(pref) + ".json",
                ],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )

        p_str = str(pref)
        d_m = np.fromfile(p_str + ".disparity.f32", dtype="<f4").reshape(2, h, w)[0]
        v_m = np.fromfile(p_str + ".validity.u32", dtype="<u4").reshape(2, h, w)[0] != 0
        d_sp = np.fromfile(p_str + ".completed_disparity.f32", dtype="<f4").reshape(-1, h, w)[0]
        v_sp = np.fromfile(p_str + ".completed_validity.u32", dtype="<u4").reshape(-1, h, w)[0] != 0
        pr_sp = np.fromfile(p_str + ".completed_provenance.u32", dtype="<u4").reshape(-1, h, w)[0]
        pr_c = np.fromfile(p_str + ".prediction.f32", dtype="<f4").reshape(2, h, w)[0]
        pv_c = np.fromfile(p_str + ".prediction_validity.u32", dtype="<u4").reshape(2, h, w)[0] != 0
        l_pair = np.fromfile(p_str + ".luma.u32", dtype="<u4").reshape(2, h, w)

        d_meas_seq.append(d_m)
        v_meas_seq.append(v_m)
        d_spatial_seq.append(d_sp)
        v_spatial_seq.append(v_sp)
        prov_spatial_seq.append(pr_sp)
        pred_seq.append(pr_c)
        pred_v_seq.append(pv_c)
        luma_l_seq.append(l_pair[0].astype(np.float32))
        luma_r_seq.append(l_pair[1].astype(np.float32))

    # Mode 3: Motion-Compensated Temporal Accumulation across frames 0..num_frames-1
    # Matches temporal.slang: reproject previous valid state, verify photometric consistency,
    # fuse when current measured stereo is valid (Provenance.FUSED = 3), reproject with decay
    # when current stereo is invalid but history passes photometric + coarse prior verification (Provenance.REPROJECTED = 2).
    yy, xx = np.indices((h, w), dtype=np.float32)
    hist_d = np.full((h, w), np.nan, dtype=np.float32)
    hist_v = np.zeros((h, w), dtype=bool)
    hist_age = np.zeros((h, w), dtype=np.int32)
    hist_luma = luma_l_seq[0].copy()

    d_temp_seq: list[np.ndarray] = []
    v_temp_seq: list[np.ndarray] = []
    prov_temp_seq: list[np.ndarray] = []

    for f_idx in range(num_frames):
        dm = d_meas_seq[f_idx]
        vm = v_meas_seq[f_idx]
        pc = pred_seq[f_idx]
        pvc = pred_v_seq[f_idx]
        ll = luma_l_seq[f_idx]
        lr = luma_r_seq[f_idx]

        # Estimate subpixel global inter-frame camera translation (dx, dy) via phase correlation on luma
        if f_idx > 0:
            shift, _ = cv2.phaseCorrelate(hist_luma, ll)
            dx_cam, dy_cam = float(shift[0]), float(shift[1])
            if abs(dx_cam) > 4.0 or abs(dy_cam) > 4.0:
                dx_cam, dy_cam = 0.0, 0.0
            map_x = (xx - dx_cam).astype(np.float32)
            map_y = (yy - dy_cam).astype(np.float32)
            prev_d = cv2.remap(
                np.where(hist_v, hist_d, 0.0).astype(np.float32),
                map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            )
            prev_v = (
                cv2.remap(
                    hist_v.astype(np.uint8),
                    map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                )
                != 0
            )
            prev_l = cv2.remap(
                hist_luma, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
            prev_age = cv2.remap(
                hist_age.astype(np.int32),
                map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            )
        else:
            prev_d = np.full((h, w), np.nan, dtype=np.float32)
            prev_v = np.zeros((h, w), dtype=bool)
            prev_l = ll
            prev_age = np.zeros((h, w), dtype=np.int32)

        # Photometric + epipolar + coarse-prior verification for temporal reprojection
        xr_reproj = np.clip(xx - np.where(prev_v, prev_d, 0.0), 1.0, w - 2.0).astype(np.float32)
        r_reproj_luma = cv2.remap(lr, xr_reproj, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        temporal_photo_ok = (np.abs(ll - prev_l) <= 14.0) & (np.abs(ll - r_reproj_luma) <= 18.0)
        coarse_compat = (~pvc) | (np.abs(prev_d - pc) <= 2.4)

        fuse_mask = vm & prev_v & temporal_photo_ok & (np.abs(dm - prev_d) <= 1.6)
        reproj_mask = (~vm) & prev_v & (prev_age < 4) & temporal_photo_ok & coarse_compat

        cur_d = np.where(
            fuse_mask,
            0.65 * dm + 0.35 * prev_d,
            np.where(vm, dm, np.where(reproj_mask, prev_d, np.nan)),
        ).astype(np.float32)
        cur_v = vm | reproj_mask
        cur_prov = np.where(
            fuse_mask,
            int(Provenance.FUSED),
            np.where(
                vm,
                int(Provenance.STEREO),
                np.where(reproj_mask, int(Provenance.REPROJECTED), int(Provenance.INVALID)),
            ),
        ).astype(np.uint8)

        hist_d = cur_d.copy()
        hist_v = cur_v.copy()
        hist_age = np.where(vm, 0, np.where(reproj_mask, prev_age + 1, 99)).astype(np.int32)
        hist_luma = ll.copy()

        d_temp_seq.append(cur_d)
        v_temp_seq.append(cur_v)
        prov_temp_seq.append(cur_prov)

    # Mode 4: Combined Temporal + Spatial Planar Propagation on Final Frame (t = num_frames - 1)
    # Uses the temporally accumulated dense anchors (Provenance in {1, 2, 3}) to seed spatial planar completion.
    # Pixels already valid from Mode 2 (spatial planar from current-frame STEREO=1) retain Provenance.PLANAR_INFERRED = 6;
    # additional planar pixels unlocked by temporal anchors (REPROJECTED=2, FUSED=3) receive Provenance.TEMPORAL_PLANAR_INFERRED = 7.
    last = num_frames - 1
    d_m_last = d_meas_seq[last]
    v_m_last = v_meas_seq[last]
    d_sp_last = d_spatial_seq[last]
    v_sp_last = v_spatial_seq[last]
    prov_sp_last = prov_spatial_seq[last]
    d_tm_last = d_temp_seq[last]
    v_tm_last = v_temp_seq[last]
    prov_tm_last = prov_temp_seq[last]
    pc_last = pred_seq[last]
    pvc_last = pred_v_seq[last]
    ll_last = luma_l_seq[last]
    lr_last = luma_r_seq[last]

    # Spatial planar propagation seeded by temporal anchor set v_tm_last
    k_off = [-9, -5, -2, 2, 5, 9]
    sw = np.zeros((h, w), dtype=np.float32)
    su = np.zeros((h, w), dtype=np.float32)
    suu = np.zeros((h, w), dtype=np.float32)
    n_anc_tm = np.zeros((h, w), dtype=np.int32)
    n_temp_contrib = np.zeros((h, w), dtype=np.int32)
    n_contra_tm = np.zeros((h, w), dtype=np.int32)
    qmask_tm = np.zeros((h, w), dtype=np.uint8)
    dc_tol = np.where(pc_last >= 114.0, 2.6, 1.8).astype(np.float32)

    for dy in k_off:
        qy = 2 if dy > 0 else 0
        for dx in k_off:
            qbit = 1 << (qy + (1 if dx > 0 else 0))
            va = np.roll(np.roll(v_tm_last, -dy, axis=0), -dx, axis=1)
            da = np.roll(np.roll(d_tm_last, -dy, axis=0), -dx, axis=1)
            pa = np.roll(np.roll(prov_tm_last, -dy, axis=0), -dx, axis=1)
            ga = np.roll(np.roll(ll_last, -dy, axis=0), -dx, axis=1)
            pca = np.roll(np.roll(pc_last, -dy, axis=0), -dx, axis=1)
            pva = np.roll(np.roll(pvc_last, -dy, axis=0), -dx, axis=1)
            dc_a = np.where(pva & np.isfinite(pca), pca, pc_last)
            u_res = da - dc_a
            l_diff = np.abs(ga - ll_last)
            ok = va & np.isfinite(da) & (np.abs(u_res) <= dc_tol)
            contra = va & np.isfinite(da) & (np.abs(u_res) > 3.2)
            w_a = np.where(
                ok,
                1.0 / ((1.0 + 1.77 * u_res * u_res) * (1.0 + 0.0025 * l_diff * l_diff)),
                0.0,
            ).astype(np.float32)
            sw += w_a
            su += w_a * u_res
            suu += w_a * u_res * u_res
            n_anc_tm += ok.astype(np.int32)
            n_temp_contrib += (ok & (pa == int(Provenance.REPROJECTED))).astype(np.int32)
            n_contra_tm += contra.astype(np.int32)
            qmask_tm |= np.where(ok, np.uint8(qbit), np.uint8(0))

    qcnt_tm = (
        ((qmask_tm & 1) != 0).astype(np.int32)
        + ((qmask_tm & 2) != 0).astype(np.int32)
        + ((qmask_tm & 4) != 0).astype(np.int32)
        + ((qmask_tm & 8) != 0).astype(np.int32)
    )
    h_two_sided_tm = ((qmask_tm & 5) != 0) & ((qmask_tm & 10) != 0)
    mu_u = su / np.maximum(sw, 1e-5)
    rmse_u = np.sqrt(np.maximum(0.0, suu / np.maximum(sw, 1e-5) - mu_u * mu_u))
    d_prop_tm = pc_last + mu_u

    gx = 0.5 * np.abs(np.roll(ll_last, -1, axis=1) - np.roll(ll_last, 1, axis=1))
    xr_tm = np.clip(xx - np.where(np.isfinite(d_prop_tm), d_prop_tm, 0.0), 1.0, w - 2.0).astype(np.float32)
    r_pix_tm = cv2.remap(lr_last, xr_tm, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    l_m5 = cv2.blur(ll_last, (5, 5))
    r_m5 = cv2.remap(cv2.blur(lr_last, (5, 5)), xr_tm, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    c_l = compute_ortho_thermo32(ll_last)
    c_r = compute_ortho_thermo32(lr_last)
    rx_i = np.clip(np.rint(xr_tm).astype(np.int32), 1, w - 2)
    ry_i = np.clip(yy.astype(np.int32), 1, h - 2)
    base_h = _popcount32(c_l ^ c_r[ry_i, rx_i])
    cost_5tap = (
        2 * base_h
        + _popcount32(np.roll(c_l, 1, axis=1) ^ c_r[ry_i, np.clip(rx_i - 1, 0, w - 1)])
        + _popcount32(np.roll(c_l, -1, axis=1) ^ c_r[ry_i, np.clip(rx_i + 1, 0, w - 1)])
        + _popcount32(np.roll(c_l, 1, axis=0) ^ c_r[np.clip(ry_i - 1, 0, h - 1), rx_i])
        + _popcount32(np.roll(c_l, -1, axis=0) ^ c_r[np.clip(ry_i + 1, 0, h - 1), rx_i])
    )

    in_b = (yy >= 12) & (yy < h - 12) & (xx >= 12) & (xx < w - 12)
    tm_planar_cand = (
        in_b
        & (~v_tm_last)
        & pvc_last
        & np.isfinite(d_prop_tm)
        & (d_prop_tm >= 1.5)
        & (d_prop_tm < 127.0)
        & (n_contra_tm == 0)
        & h_two_sided_tm
        & (n_anc_tm >= 4)
        & (qcnt_tm >= 2)
        & (rmse_u <= 0.65)
        & (np.abs(mu_u) <= 2.2)
        & (np.abs(ll_last - r_pix_tm) <= (18.0 + 0.65 * gx))
        & (np.abs(l_m5 - r_m5) <= (12.0 + 0.35 * gx))
        & (cost_5tap <= 30)
    )

    # Combine Mode 3 (v_tm_last) + Mode 2 spatial planar (prov_sp_last == 6) + temporal-seeded planar (tm_planar_cand)
    sp_only_new = (~v_tm_last) & (prov_sp_last == int(Provenance.PLANAR_INFERRED))
    tm_seeded_new = (~v_tm_last) & (~sp_only_new) & tm_planar_cand

    d_comb_last = np.where(
        v_tm_last,
        d_tm_last,
        np.where(sp_only_new, d_sp_last, np.where(tm_seeded_new, d_prop_tm, np.nan)),
    ).astype(np.float32)
    v_comb_last = v_tm_last | sp_only_new | tm_seeded_new
    prov_comb_last = np.where(
        v_tm_last,
        prov_tm_last,
        np.where(
            sp_only_new,
            int(Provenance.PLANAR_INFERRED),
            np.where(
                tm_seeded_new,
                int(Provenance.TEMPORAL_PLANAR_INFERRED),
                int(Provenance.INVALID),
            ),
        ),
    ).astype(np.uint8)

    # Frame-by-frame temporal coverage progression & flicker stability across t = 0..num_frames-1
    progression = []
    for f_idx in range(num_frames):
        flicker_meas = (
            float(np.nanmedian(np.abs(d_meas_seq[f_idx][v_meas_seq[f_idx] & v_meas_seq[f_idx - 1]] - d_meas_seq[f_idx - 1][v_meas_seq[f_idx] & v_meas_seq[f_idx - 1]])))
            if f_idx > 0
            else 0.0
        )
        flicker_temp = (
            float(np.nanmedian(np.abs(d_temp_seq[f_idx][v_temp_seq[f_idx] & v_temp_seq[f_idx - 1]] - d_temp_seq[f_idx - 1][v_temp_seq[f_idx] & v_temp_seq[f_idx - 1]])))
            if f_idx > 0
            else 0.0
        )
        progression.append(
            {
                "frame_index": f_idx,
                "measured_valid_pixels": int(v_meas_seq[f_idx].sum()),
                "spatial_planar_completed_pixels": int(v_spatial_seq[f_idx].sum()),
                "temporal_accumulated_pixels": int(v_temp_seq[f_idx].sum()),
                "temporal_reprojected_prov2_pixels": int((prov_temp_seq[f_idx] == int(Provenance.REPROJECTED)).sum()),
                "temporal_fused_prov3_pixels": int((prov_temp_seq[f_idx] == int(Provenance.FUSED)).sum()),
                "interframe_disparity_flicker_p50_measured_px": round(flicker_meas, 4),
                "interframe_disparity_flicker_p50_temporal_px": round(flicker_temp, 4),
            }
        )

    # Independent ZNCC reference on final frame
    zncc_d_last, zncc_v_last = build_independent_zncc_epipolar_reference(
        ll_last.astype(np.uint32), lr_last.astype(np.uint32), max_disp=128
    )

    modes_last = {
        "mode_1_measured_stereo_only": (d_m_last, v_m_last, np.where(v_m_last, 1, 0).astype(np.uint8)),
        "mode_2_spatial_planar_propagation": (d_sp_last, v_sp_last, prov_sp_last),
        "mode_3_temporal_accumulation_8frame": (d_tm_last, v_tm_last, prov_tm_last),
        "mode_4_combined_temporal_plus_planar": (d_comb_last, v_comb_last, prov_comb_last),
    }

    roi_evaluations_zncc: dict[str, Any] = {}
    roi_evaluations_self_consistency: dict[str, Any] = {}

    for rname, rmask in rois.items():
        if rname == "broad_desk_outofrange_d_gt_127":
            continue
        roi_px = int(rmask.sum())
        z_sup = rmask & zncc_v_last
        m_sup = rmask & v_m_last
        if int(z_sup.sum()) < 12 or int(m_sup.sum()) < 12:
            continue
        th_mm = 2.5 if ("apriltag" in rname or "charuco" in rname) else 3.5
        n_z, d_off_z, d_map_z = _fit_3d_plane_ransac(zncc_d_last, z_sup, intr, thresh_mm=th_mm)
        n_m, d_off_m, d_map_m = _fit_3d_plane_ransac(d_m_last, m_sup, intr, thresh_mm=th_mm)

        roi_evaluations_zncc[rname] = {
            "reference_kind": "independent_multiscale_zncc_epipolar_ransac_plane",
            "independent_zncc_support_pixels": int(z_sup.sum()),
            "modes": {},
        }
        roi_evaluations_self_consistency[rname] = {
            "reference_kind": "self_consistency_to_measured_stereo_ransac_plane",
            "measured_stereo_support_pixels": int(m_sup.sum()),
            "modes": {},
        }

        for mname, (dm_arr, vm_arr, pr_arr) in modes_last.items():
            added_mask = rmask & vm_arr & (~v_m_last)
            roi_evaluations_zncc[rname]["modes"][mname] = {
                "completed_total": _eval_plane_cohort(dm_arr, rmask & vm_arr, roi_px, d_map_z, n_z, d_off_z, intr),
                "newly_recovered_only": _eval_plane_cohort(dm_arr, added_mask, roi_px, d_map_z, n_z, d_off_z, intr),
                "provenance_breakdown_pixels": {
                    "STEREO_1": int((rmask & (pr_arr == int(Provenance.STEREO))).sum()),
                    "REPROJECTED_2": int((rmask & (pr_arr == int(Provenance.REPROJECTED))).sum()),
                    "FUSED_3": int((rmask & (pr_arr == int(Provenance.FUSED))).sum()),
                    "PLANAR_INFERRED_6": int((rmask & (pr_arr == int(Provenance.PLANAR_INFERRED))).sum()),
                    "TEMPORAL_PLANAR_INFERRED_7": int((rmask & (pr_arr == int(Provenance.TEMPORAL_PLANAR_INFERRED))).sum()),
                    "INVALID_0": int((rmask & (pr_arr == int(Provenance.INVALID))).sum()),
                },
            }
            roi_evaluations_self_consistency[rname]["modes"][mname] = {
                "completed_total": _eval_plane_cohort(dm_arr, rmask & vm_arr, roi_px, d_map_m, n_m, d_off_m, intr),
                "newly_recovered_only": _eval_plane_cohort(dm_arr, added_mask, roi_px, d_map_m, n_m, d_off_m, intr),
            }

    return {
        "scene_name": scene_name,
        "frames_evaluated": num_frames,
        "temporal_progression_across_frames": progression,
        "final_frame_full_image_provenance_summary": {
            mname: {
                "total_valid_pixels": int(vm_arr.sum()),
                "coverage_pct": round(100.0 * int(vm_arr.sum()) / (h * w), 2),
                "STEREO_1": int((pr_arr == int(Provenance.STEREO)).sum()),
                "REPROJECTED_2": int((pr_arr == int(Provenance.REPROJECTED)).sum()),
                "FUSED_3": int((pr_arr == int(Provenance.FUSED)).sum()),
                "PLANAR_INFERRED_6": int((pr_arr == int(Provenance.PLANAR_INFERRED)).sum()),
                "TEMPORAL_PLANAR_INFERRED_7": int((pr_arr == int(Provenance.TEMPORAL_PLANAR_INFERRED)).sum()),
                "INVALID_0": int((pr_arr == int(Provenance.INVALID)).sum()),
            }
            for mname, (_, vm_arr, pr_arr) in modes_last.items()
        },
        "independent_zncc_reference_planes": roi_evaluations_zncc,
        "self_consistency_to_measured_planes": roi_evaluations_self_consistency,
        "_last_arrays": {
            "d_meas": d_m_last,
            "v_meas": v_m_last,
            "d_spatial": d_sp_last,
            "v_spatial": v_sp_last,
            "prov_spatial": prov_sp_last,
            "d_comb": d_comb_last,
            "v_comb": v_comb_last,
            "prov_comb": prov_comb_last,
            "pred": pc_last,
            "pred_v": pvc_last,
            "luma_l": ll_last,
            "luma_r": lr_last,
            "zncc_d_128": zncc_d_last,
            "zncc_v_128": zncc_v_last,
        },
    }


def evaluate_synthetic_woodgrain_and_counterexample_suite(work_dir: Path) -> dict[str, Any]:
    """Evaluate two targeted synthetic stress tests with exact ground truth:
    1. `slanted_tabletop_woodgrain_low_texture`: Slanted horizontal tabletop (d = 20 + 0.35*(y-300))
       with anisotropic low-contrast woodgrain texture across an 8-frame micro-jitter sequence.
    2. `narrow_gap_bars_counterexample`: Three vertical foreground bars (d = 36 px) separated by narrow
       background gaps (10 px, 14 px, 18 px wide at d = 16 px) to verify zero false gap-bridging.
    """
    work = work_dir / "syn_v94_stress"
    work.mkdir(parents=True, exist_ok=True)
    env = gpu_env()
    h, w = 720, 1280
    yy, xx = np.indices((h, w), dtype=np.float32)

    def warp_pair(left_f: np.ndarray, disp_gt: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        right_f = np.zeros_like(left_f)
        zbuf = np.full((h, w), -1.0, dtype=np.float32)
        for y in range(h):
            x_src = np.arange(w, dtype=np.float32)
            d_row = disp_gt[y]
            x_dst = x_src - d_row
            x0 = np.floor(x_dst).astype(np.int32)
            frac = x_dst - x0
            order = np.argsort(d_row, kind="stable")
            for idx in order:
                xi = x0[idx]
                f = frac[idx]
                val = left_f[y, idx]
                dv = d_row[idx]
                if 0 <= xi < w and dv >= zbuf[y, xi] - 0.25:
                    right_f[y, xi] = (1.0 - f) * val + (f * right_f[y, xi] if zbuf[y, xi] > 0 else f * val)
                    zbuf[y, xi] = dv
                if 0 <= xi + 1 < w and dv >= zbuf[y, xi + 1] - 0.25:
                    right_f[y, xi + 1] = f * val + ((1.0 - f) * right_f[y, xi + 1] if zbuf[y, xi + 1] > 0 else (1.0 - f) * val)
                    zbuf[y, xi + 1] = dv
        holes = zbuf < 0
        bg = 128.0 + 10.0 * rng.standard_normal((h, w)).astype(np.float32)
        right_f[holes] = bg[holes]
        # Add realistic sensor shot noise (sigma = 0.85 DN)
        l_u8 = np.clip(np.round(left_f + rng.normal(0.0, 0.85, (h, w))), 10, 245).astype(np.uint8)
        r_u8 = np.clip(np.round(right_f + rng.normal(0.0, 0.85, (h, w))), 10, 245).astype(np.uint8)
        xr_i = np.clip(np.round(xx - disp_gt).astype(np.int32), 0, w - 1)
        seen = zbuf[yy.astype(np.int32), xr_i]
        non_occ = (yy >= 40) & (yy < h - 40) & (xx >= 140) & (xx < w - 40) & (np.abs(seen - disp_gt) <= 0.75)
        return l_u8, r_u8, non_occ

    # 1. Slanted low-texture woodgrain tabletop (8 frames with independent sensor noise)
    rng_wg = np.random.default_rng(20261103)
    # Anisotropic horizontal woodgrain streaks (sigma_x = 14 px, sigma_y = 2.2 px, low contrast = 5.5 DN)
    raw_wg = rng_wg.standard_normal((h, w)).astype(np.float32)
    streaks = cv2.GaussianBlur(raw_wg, (0, 0), sigmaX=14.0, sigmaY=2.2)
    fine = cv2.GaussianBlur(rng_wg.standard_normal((h, w)).astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=1.2)
    tex_wg = 128.0 + 5.2 * (streaks / (np.std(streaks) + 1e-5)) + 2.4 * (fine / (np.std(fine) + 1e-5))
    gt_wg = np.clip(18.0 + 0.16 * np.maximum(yy - 120.0, 0.0) + 0.004 * (xx - 640.0), 18.0, 112.0).astype(np.float32)

    wg_meas_list = []
    wg_sp_list = []
    wg_v_meas_list = []
    wg_v_sp_list = []
    non_occ_wg = None
    for f_idx in range(4):
        l_u8, r_u8, non_occ_wg = warp_pair(tex_wg, gt_wg, 900 + f_idx)
        yuyv = np.empty((h, 2560, 2), dtype=np.uint8)
        yuyv[:, :1280, 0] = l_u8
        yuyv[:, 1280:, 0] = r_u8
        yuyv[:, :, 1] = 128
        yp = work / f"wg_f{f_idx}.yuyv"
        yp.write_bytes(yuyv.tobytes())
        pref = work / f"wg_out_f{f_idx}"
        if not Path(str(pref) + ".completed_disparity.f32").exists():
            subprocess.run(
                [
                    str(REFINE_BENCH),
                    "--coarse-manifest", str(COARSE_4X_MANIFEST),
                    "--target-manifest", str(DEFAULT_MANIFEST),
                    "--synthetic-yuyv", str(yp),
                    "--variant", "raw_borrow",
                    "--frames", "1", "--warmup", "0",
                    "--coarse-disparities", "32",
                    "--coarse-schedule", "layered-per-path",
                    "--coarse-right-reference-mode", "independent-sgm",
                    "--descriptor-mode", "ortho-hybrid",
                    "--fast-mode", "1",
                    "--selective-repair", "1",
                    "--planar-recovery", "1",
                    "--planar-min-anchors", "10",
                    "--planar-coarse-affine", "1",
                    "--planar-ray-crossing-guard", "1",
                    "--planar-coarse-lr-guard", "1",
                    "--radius", "2", "--subpixel", "1",
                    "--dump-prefix", str(pref),
                    "--output", str(pref) + ".json",
                ],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        p_str = str(pref)
        wg_meas_list.append(np.fromfile(p_str + ".disparity.f32", dtype="<f4").reshape(2, h, w)[0])
        wg_v_meas_list.append(np.fromfile(p_str + ".validity.u32", dtype="<u4").reshape(2, h, w)[0] != 0)
        wg_sp_list.append(np.fromfile(p_str + ".completed_disparity.f32", dtype="<f4").reshape(-1, h, w)[0])
        wg_v_sp_list.append(np.fromfile(p_str + ".completed_validity.u32", dtype="<u4").reshape(-1, h, w)[0] != 0)

    # Combine 4 frames temporally for woodgrain tabletop
    v_wg_meas = wg_v_meas_list[-1] & non_occ_wg
    d_wg_meas = wg_meas_list[-1]
    v_wg_sp = wg_v_sp_list[-1] & non_occ_wg
    d_wg_sp = wg_sp_list[-1]
    v_wg_temp = np.zeros((h, w), dtype=bool)
    d_wg_temp_sum = np.zeros((h, w), dtype=np.float32)
    d_wg_temp_cnt = np.zeros((h, w), dtype=np.float32)
    for dm_f, vm_f in zip(wg_meas_list, wg_v_meas_list):
        v_wg_temp |= vm_f
        d_wg_temp_sum += np.where(vm_f, dm_f, 0.0)
        d_wg_temp_cnt += vm_f.astype(np.float32)
    d_wg_temp = np.where(v_wg_temp, d_wg_temp_sum / np.maximum(d_wg_temp_cnt, 1.0), np.nan)
    v_wg_comb = (v_wg_temp | wg_v_sp_list[-1]) & non_occ_wg
    d_wg_comb = np.where(v_wg_temp, d_wg_temp, d_wg_sp)

    def _gt_eval(d_est: np.ndarray, v_est: np.ndarray, gt_arr: np.ndarray, mask_no: np.ndarray) -> dict[str, Any]:
        v_ok = v_est & mask_no & np.isfinite(d_est)
        err = np.abs(d_est[v_ok] - gt_arr[v_ok])
        tot_no = int(mask_no.sum())
        return {
            "valid_pixels": int(v_ok.sum()),
            "coverage_pct": round(100.0 * int(v_ok.sum()) / max(tot_no, 1), 2),
            "mae_px": round(float(np.mean(err)), 3) if err.size else 0.0,
            "rmse_px": round(float(np.sqrt(np.mean(err ** 2))), 3) if err.size else 0.0,
            "bad_1px_pct": round(100.0 * float((err > 1.0).mean()), 2) if err.size else 0.0,
            "bad_2px_pct": round(100.0 * float((err > 2.0).mean()), 2) if err.size else 0.0,
        }

    # 2. Narrow-gap foreground bars counterexample (gaps of width 12, 16, 20 px between d=28 bars over d=20 background)
    tex_bg = 85.0 + 24.0 * cv2.GaussianBlur(rng_wg.standard_normal((h, w)).astype(np.float32), (0, 0), 2.2)
    tex_fg = 170.0 + 24.0 * cv2.GaussianBlur(rng_wg.standard_normal((h, w)).astype(np.float32), (0, 0), 2.0)
    bar_mask = (yy >= 140) & (yy < 580) & (
        ((xx >= 360) & (xx < 400))
        | ((xx >= 412) & (xx < 452))  # 12 px gap at x in [400, 412)
        | ((xx >= 468) & (xx < 508))  # 16 px gap at x in [452, 468)
        | ((xx >= 528) & (xx < 568))  # 20 px gap at x in [508, 528)
    )
    gap_mask = (yy >= 160) & (yy < 560) & (
        ((xx >= 400) & (xx < 412))
        | ((xx >= 452) & (xx < 468))
        | ((xx >= 508) & (xx < 528))
    )
    gt_bars = np.where(bar_mask, 28.0, 20.0).astype(np.float32)
    l_b, r_b, non_occ_bars = warp_pair(np.where(bar_mask, tex_fg, tex_bg), gt_bars, 950)
    yuyv_b = np.empty((h, 2560, 2), dtype=np.uint8)
    yuyv_b[:, :1280, 0] = l_b
    yuyv_b[:, 1280:, 0] = r_b
    yuyv_b[:, :, 1] = 128
    yp_b = work / "narrow_gaps.yuyv"
    yp_b.write_bytes(yuyv_b.tobytes())
    pref_b = work / "narrow_gaps_out"
    if not Path(str(pref_b) + ".completed_disparity.f32").exists():
        subprocess.run(
            [
                str(REFINE_BENCH),
                "--coarse-manifest", str(COARSE_4X_MANIFEST),
                "--target-manifest", str(DEFAULT_MANIFEST),
                "--synthetic-yuyv", str(yp_b),
                "--variant", "raw_borrow",
                "--frames", "1", "--warmup", "0",
                "--coarse-disparities", "32",
                "--coarse-schedule", "layered-per-path",
                "--coarse-right-reference-mode", "independent-sgm",
                "--descriptor-mode", "ortho-hybrid",
                "--fast-mode", "1",
                "--selective-repair", "1",
                "--planar-recovery", "1",
                "--planar-min-anchors", "10",
                "--planar-coarse-affine", "1",
                "--planar-ray-crossing-guard", "1",
                "--planar-coarse-lr-guard", "1",
                "--radius", "2", "--subpixel", "1",
                "--dump-prefix", str(pref_b),
                "--output", str(pref_b) + ".json",
            ],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    d_b_comp = np.fromfile(str(pref_b) + ".completed_disparity.f32", dtype="<f4").reshape(-1, h, w)[0]
    p_b_comp = np.fromfile(str(pref_b) + ".completed_provenance.u32", dtype="<u4").reshape(-1, h, w)[0]
    gap_inf = gap_mask & (p_b_comp == int(Provenance.PLANAR_INFERRED))
    gap_bridged_false = gap_inf & (np.abs(d_b_comp - 28.0) <= 4.0)
    gap_non_occ = gap_mask & non_occ_bars
    gap_bridged_false_non_occ = gap_bridged_false & non_occ_bars

    return {
        "slanted_tabletop_woodgrain_low_texture": {
            "mode_1_measured_stereo_only": _gt_eval(d_wg_meas, v_wg_meas, gt_wg, non_occ_wg),
            "mode_2_spatial_planar_only": _gt_eval(d_wg_sp, v_wg_sp, gt_wg, non_occ_wg),
            "mode_2_inferred_prov6_only": _gt_eval(
                d_wg_sp, (v_wg_sp & (~v_wg_meas)), gt_wg, non_occ_wg
            ),
            "mode_3_temporal_accumulation_4frame": _gt_eval(
                d_wg_temp, v_wg_temp & non_occ_wg, gt_wg, non_occ_wg
            ),
            "mode_4_combined_temporal_plus_planar": _gt_eval(d_wg_comb, v_wg_comb, gt_wg, non_occ_wg),
        },
        "narrow_gap_bars_counterexample": {
            "gap_widths_px": [12, 16, 20],
            "total_gap_pixels": int(gap_mask.sum()),
            "non_occluded_gap_pixels": int(gap_non_occ.sum()),
            "inferred_planar_pixels_inside_gaps": int(gap_inf.sum()),
            "false_foreground_bridged_pixels_in_gaps": int(gap_bridged_false.sum()),
            "false_foreground_bridged_pixels_in_non_occluded_gaps": int(gap_bridged_false_non_occ.sum()),
            "false_foreground_bridged_pct_of_non_occluded_gaps": round(
                100.0 * int(gap_bridged_false_non_occ.sum()) / max(int(gap_non_occ.sum()), 1), 2
            ),
            "non_occluded_inferred_cohort": _gt_eval(
                d_b_comp, p_b_comp == int(Provenance.PLANAR_INFERRED), gt_bars, non_occ_bars
            ),
        },
    }


def run_v94_qualification_suite(work_dir: Path, desk_env_dir: Path) -> dict[str, Any]:
    """Execute the complete StereoForge v9.4 Dense Low-Texture Surface Recovery qualification suite."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    env = gpu_env()
    h, w = 720, 1280

    zed_intr = {
        "fx": 527.7903039292192,
        "fy": 527.7903039292192,
        "cx": 648.5904159545898,
        "cy": 352.4145050048828,
        "baseline_m": 0.119444,
        "cx_offset": 0.0,
    }

    # 1. Multi-frame 4-mode surface recovery on 8-frame ZED 2 desk scene (desk_fixture_env)
    desk_rois = _desk_rois(h, w)
    desk_multiframe = run_multiframe_4mode_surface_recovery(
        desk_env_dir / "target",
        desk_env_dir / "coarse/manifest.json",
        work,
        "zed2_desk_8frame",
        desk_rois,
        zed_intr,
        num_frames=8,
        fast_mode="0",
    )
    last_arrs = desk_multiframe.pop("_last_arrays")

    # Extended 192-disparity ZNCC reference for near-field y >= 463 diagnostic
    zncc_d_192, zncc_v_192 = build_independent_zncc_epipolar_reference(
        last_arrs["luma_l"].astype(np.uint32),
        last_arrs["luma_r"].astype(np.uint32),
        max_disp=192,
    )

    # 2. 11-Stage Unrecovered-Pixel Taxonomy & Single-Guard Relaxation Ceiling Matrix
    taxonomy = compute_unrecovered_pixel_taxonomy_and_relaxation_matrix(
        last_arrs["d_meas"],
        last_arrs["v_meas"],
        last_arrs["d_spatial"],
        last_arrs["prov_spatial"],
        last_arrs["pred"],
        last_arrs["pred_v"],
        last_arrs["luma_l"],
        last_arrs["luma_r"],
        last_arrs["zncc_d_128"],
        last_arrs["zncc_v_128"],
        zncc_d_192,
        zncc_v_192,
        zed_intr,
    )

    # 3. Cross-Sensor Active-IR Intel RealSense D435i Reference Plane Comparison
    rs_json_path = Path(
        "/home/bjoern/.gemini/antigravity-cli/brain/6a5df588-91c3-4b8d-8e19-d968e1de4f3f/scratch/calibration_target_comparison.json"
    )
    realsense_cross_sensor_ref: dict[str, Any] = {}
    if rs_json_path.exists():
        rs_doc = json.loads(rs_json_path.read_text())
        rs_tag = rs_doc.get("target_1_upright_apriltag_plane", {}).get("realsense_d435i", {})
        rs_char_in = rs_doc.get("target_2_flat_charuco_board_in_range_band", {}).get("realsense_d435i", {})
        rs_char_full = rs_doc.get("target_3_flat_charuco_board_full_polygon", {}).get("realsense_d435i", {})
        tag_v94 = desk_multiframe["independent_zncc_reference_planes"]["upright_apriltag_target"]["modes"][
            "mode_4_combined_temporal_plus_planar"
        ]["completed_total"]
        char_v94 = desk_multiframe["independent_zncc_reference_planes"]["charuco_board_inrange"]["modes"][
            "mode_4_combined_temporal_plus_planar"
        ]["completed_total"]
        desk_v94 = desk_multiframe["independent_zncc_reference_planes"]["horizontal_desk_plane_clean"]["modes"][
            "mode_4_combined_temporal_plus_planar"
        ]["completed_total"]
        realsense_cross_sensor_ref = {
            "reference_sensor": "Intel RealSense D435i Active-IR Structured Light (848x480)",
            "realsense_apriltag_ransac_plane_rmse_mm": rs_tag.get("precision_3d_plane_fit", {}).get("ransac_inlier_rmse_mm"),
            "realsense_apriltag_all_points_rmse_mm": rs_tag.get("precision_3d_plane_fit", {}).get("all_points_svd_rmse_mm"),
            "realsense_apriltag_coverage_pct": rs_tag.get("coverage", {}).get("valid_coverage_mean_pct"),
            "realsense_charuco_inrange_ransac_plane_rmse_mm": rs_char_in.get("precision_3d_plane_fit", {}).get("ransac_inlier_rmse_mm"),
            "realsense_charuco_inrange_all_points_rmse_mm": rs_char_in.get("precision_3d_plane_fit", {}).get("all_points_svd_rmse_mm"),
            "realsense_charuco_inrange_coverage_pct": rs_char_in.get("coverage", {}).get("valid_coverage_mean_pct"),
            "realsense_charuco_full_ransac_plane_rmse_mm": rs_char_full.get("precision_3d_plane_fit", {}).get("ransac_inlier_rmse_mm"),
            "realsense_charuco_full_all_points_rmse_mm": rs_char_full.get("precision_3d_plane_fit", {}).get("all_points_svd_rmse_mm"),
            "realsense_charuco_full_coverage_pct": rs_char_full.get("coverage", {}).get("valid_coverage_mean_pct"),
            "stereoforge_v94_apriltag_orth_plane_rmse_mm": tag_v94.get("orth_plane_residual_rmse_mm"),
            "stereoforge_v94_charuco_orth_plane_rmse_mm": char_v94.get("orth_plane_residual_rmse_mm"),
            "stereoforge_v94_desk_orth_plane_rmse_mm": desk_v94.get("orth_plane_residual_rmse_mm"),
            "stereoforge_v94_apriltag_coverage_pct": tag_v94.get("roi_coverage_pct"),
            "stereoforge_v94_charuco_coverage_pct": char_v94.get("roi_coverage_pct"),
            "stereoforge_v94_desk_coverage_pct": desk_v94.get("roi_coverage_pct"),
        }

    # 4. Canonical 5-Scene 720p Synthetic Ground-Truth Suite + 2 Targeted Stress Tests
    gt_5scene = evaluate_known_ground_truth_suite_720p(work)
    stress_suite = evaluate_synthetic_woodgrain_and_counterexample_suite(work)

    # 5. 5-Trial Interleaved 240 FPS GPU Performance Qualification (wait for low GPU utilization + hysteresis)
    def _wait_gpu_idle_hysteresis(max_busy_pct: int = 12, required_streak: int = 4, timeout_s: float = 45.0) -> None:
        busy_paths = sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
        if not busy_paths:
            return
        t0 = time.monotonic()
        streak = 0
        while time.monotonic() - t0 < timeout_s:
            vals = []
            for bp in busy_paths:
                try:
                    vals.append(int(bp.read_text().strip()))
                except Exception:
                    pass
            cur = max(vals) if vals else 0
            if cur <= max_busy_pct:
                streak += 1
                if streak >= required_streak:
                    return
            else:
                streak = 0
            time.sleep(0.25)

    _wait_gpu_idle_hysteresis()
    num_trials = 5
    trial_docs: dict[str, list[dict[str, Any]]] = {
        "dual4x_selective_repair_only": [],
        "dual4x_selective_repair_plus_planar_v94": [],
    }
    for trial in range(num_trials):
        _wait_gpu_idle_hysteresis()
        for label, p_flag in (
            ("dual4x_selective_repair_only", "0"),
            ("dual4x_selective_repair_plus_planar_v94", "1"),
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
                    "--planar-recovery", p_flag,
                    "--planar-min-anchors", "10",
                    "--planar-coarse-affine", "1",
                    "--planar-ray-crossing-guard", "1",
                    "--planar-coarse-lr-guard", "1",
                    "--radius", "2", "--subpixel", "1",
                    "--coarse-schedule", "shared-census-sequential",
                    "--coarse-right-reference-mode", "left-aggregated-selection",
                    "--frames", "120", "--warmup", "20",
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
        docs_sorted = sorted(docs, key=lambda d: float(d["gpu_combined_queue_span_ms"]["p95"]))
        med = docs_sorted[len(docs_sorted) // 2]
        p50s = [round(float(d["gpu_combined_queue_span_ms"]["p50"]), 4) for d in docs]
        p95s = [round(float(d["gpu_combined_queue_span_ms"]["p95"]), 4) for d in docs]
        p99s = [round(float(d["gpu_combined_queue_span_ms"]["p99"]), 4) for d in docs]
        gpu_perf[label] = {
            "interleaved_trials_count": num_trials,
            "selection_policy": "median_by_gpu_combined_p95_across_5_interleaved_trials",
            "gpu_combined_queue_span_ms": med["gpu_combined_queue_span_ms"],
            "gpu_stage_sum_ms": med["gpu_stage_sum_ms"],
            "host_wall_ms": med["host_wall_ms"],
            "gpu_planar_recovery_ms": med.get("gpu_planar_recovery_ms"),
            "gpu_stage_timings_ms": med["gpu_stage_timings_ms"],
            "interleaved_trial_distributions": {
                "gpu_combined_p50_ms": p50s,
                "gpu_combined_p95_ms": p95s,
                "gpu_combined_p99_ms": p99s,
                "gpu_combined_p95_mean_ms": round(float(np.mean(p95s)), 4),
                "gpu_combined_p95_std_ms": round(float(np.std(p95s)), 4),
                "all_trials_meet_240fps_gpu_p95": all(x <= 4.167 for x in p95s),
            },
            "measured_valid_pixels": med.get("measured_valid_pixels"),
            "completed_valid_pixels": med.get("completed_valid_pixels"),
            "inferred_planar_pixels": med.get("inferred_planar_pixels"),
            "meets_240fps_4_167ms_gpu_p50": float(med["gpu_combined_queue_span_ms"]["p50"]) <= 4.167,
            "meets_240fps_4_167ms_gpu_p95": float(med["gpu_combined_queue_span_ms"]["p95"]) <= 4.167,
            "meets_240fps_4_167ms_gpu_p99": float(med["gpu_combined_queue_span_ms"]["p99"]) <= 4.167,
            "validation_errors_after_teardown": med["validation_errors_after_teardown"],
        }

    r_only = gpu_perf["dual4x_selective_repair_only"]
    r_plan = gpu_perf["dual4x_selective_repair_plus_planar_v94"]
    m_no_plan = float(r_only["gpu_stage_timings_ms"]["refinement_metric_ms"])
    m_with_plan = float(r_plan["gpu_stage_timings_ms"]["refinement_metric_ms"])
    p_kern = float(r_plan["gpu_stage_timings_ms"]["refinement_planar_recovery_ms"])
    r_plan["planar_feature_overhead_accounting_ms"] = {
        "planar_kernel_ms": round(p_kern, 4),
        "metric_stage_without_planar_ms": round(m_no_plan, 4),
        "metric_stage_with_planar_ms": round(m_with_plan, 4),
        "metric_completed_buffer_init_delta_ms": round(m_with_plan - m_no_plan, 4),
        "total_planar_stage_overhead_ms": round(p_kern + (m_with_plan - m_no_plan), 4),
    }

    return {
        "schema": "stereoforge.v9_4_dense_surface_qualification.v1",
        "provenance": current_provenance_manifest(),
        "provenance_enum_contract": {
            "INVALID": int(Provenance.INVALID),
            "STEREO": int(Provenance.STEREO),
            "REPROJECTED": int(Provenance.REPROJECTED),
            "FUSED": int(Provenance.FUSED),
            "NEURAL": int(Provenance.NEURAL),
            "PLANAR_INFERRED": int(Provenance.PLANAR_INFERRED),
            "TEMPORAL_PLANAR_INFERRED": int(Provenance.TEMPORAL_PLANAR_INFERRED),
            "distinct_provenance_codes": len(
                {
                    int(Provenance.INVALID),
                    int(Provenance.STEREO),
                    int(Provenance.REPROJECTED),
                    int(Provenance.FUSED),
                    int(Provenance.PLANAR_INFERRED),
                    int(Provenance.TEMPORAL_PLANAR_INFERRED),
                }
            )
            == 6,
        },
        "unrecovered_pixel_taxonomy_and_relaxation_matrix": taxonomy,
        "multiframe_4mode_zed2_desk_evaluation": desk_multiframe,
        "cross_sensor_realsense_d435i_comparison": realsense_cross_sensor_ref,
        "canonical_5scene_720p_gt_suite": gt_5scene["aggregate_across_5_scenes"],
        "canonical_5scene_720p_per_scene_inferred_cohorts": {
            sname: sdata["newly_admitted_repair_pixels_vs_gt"]
            for sname, sdata in gt_5scene["per_scene"].items()
        },
        "synthetic_woodgrain_and_counterexample_stress_suite": stress_suite,
        "gpu_240fps_qualification": gpu_perf,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    scratch_desk = Path(
        "/home/bjoern/.gemini/antigravity-cli/brain/6a5df588-91c3-4b8d-8e19-d968e1de4f3f/scratch/desk_fixture_env"
    )
    out_dir = root / "benchmarks/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = run_v94_qualification_suite(Path("/tmp/sf_v94_qual"), scratch_desk)
    out_file = out_dir / "v9_4_dense_surface_qualification.json"
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_file}")
