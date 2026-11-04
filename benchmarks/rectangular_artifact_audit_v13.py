#!/usr/bin/env python3
"""StereoForge v13.1 — Four-Way Rectangular Artifact Diagnostic, Coarse Predictor Study,
and Quantitative Grid-Alignment Boundary Excess Metric (B_4, B_8, B_16, B_32).

Executes on identical stereo frames with frozen calibration, exposure, disparity range,
and subpixel configuration:
- Test A (adaptive_tiles=0, planar_recovery=0): Single-frame uniform reference
- Test B (adaptive_tiles=1, planar_recovery=0): Isolates adaptive matching & coarse transitions
- Test C (adaptive_tiles=0, planar_recovery=1): Isolates planar completion sub-tile fits
- Test D (adaptive_tiles=1, planar_recovery=1): Combined pipeline
- Coarse Predictor Study: 1/4 shared-census-sequential vs 1/8->1/4 hierarchical coarse prior
- Display Quantization vs Raw Metric Depth check (8.27 mm/step 254-bin LUT vs float32 depth)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from benchmarks.temporal_supersampling_v12 import (
    _identity_maps,
    _multi_scale_texture,
    _pack_yuyv,
    _render_stereo_pair_from_world_disparity,
    _scale_maps,
)
from stereoforge.slang_cpu_backend import SlangCpuRefinementBackend

ROOT = Path(__file__).resolve().parents[1]
OUT_AUDIT_JSON = ROOT / "benchmarks" / "results" / "v13_rectangular_artifact_audit.json"


def compute_grid_alignment_metrics(
    disp_map: np.ndarray,
    valid_mask: np.ndarray,
    rgb_luma: np.ndarray,
    ref_disp: np.ndarray | None = None,
    periods: tuple[int, ...] = (4, 8, 16, 32),
    eps: float = 1e-4,
) -> dict[str, float]:
    """Compute horizontal and vertical grid-boundary excess ratios B_P for P in {4, 8, 16, 32}
    on smooth valid support (excluding genuine RGB/depth object boundaries)."""
    h, w = disp_map.shape
    val = valid_mask & np.isfinite(disp_map) & (disp_map > 0.0)

    # Exclude genuine RGB or reference depth edges
    gx_luma = np.abs(np.diff(rgb_luma.astype(np.float32), axis=1))
    gy_luma = np.abs(np.diff(rgb_luma.astype(np.float32), axis=0))

    val_x = val[:, 1:] & val[:, :-1] & (gx_luma < 12.0)
    val_y = val[1:, :] & val[:-1, :] & (gy_luma < 12.0)

    if ref_disp is not None:
        val_x &= np.abs(np.diff(ref_disp, axis=1)) < 0.85
        val_y &= np.abs(np.diff(ref_disp, axis=0)) < 0.85

    dx = np.abs(disp_map[:, 1:] - disp_map[:, :-1])
    dy = np.abs(disp_map[1:, :] - disp_map[:-1, :])

    x_coords = np.arange(1, w)[None, :]
    y_coords = np.arange(1, h)[:, None]

    out: dict[str, float] = {}
    for p in periods:
        bnd_x = val_x & ((x_coords % p) == 0)
        int_x = val_x & ((x_coords % p) != 0)
        med_bx = float(np.median(dx[bnd_x])) if np.any(bnd_x) else 0.0
        med_ix = float(np.median(dx[int_x])) if np.any(int_x) else 0.0
        b_px = (med_bx + eps) / (med_ix + eps)

        bnd_y = val_y & ((y_coords % p) == 0)
        int_y = val_y & ((y_coords % p) != 0)
        med_by = float(np.median(dy[bnd_y])) if np.any(bnd_y) else 0.0
        med_iy = float(np.median(dy[int_y])) if np.any(int_y) else 0.0
        b_py = (med_by + eps) / (med_iy + eps)

        out[f"B_{p}_x"] = b_px
        out[f"B_{p}_y"] = b_py
        out[f"B_{p}_mean"] = 0.5 * (b_px + b_py)
    return out


def build_diagnostic_test_scene(w: int = 640, h: int = 360) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a realistic test scene with slanted planar walls, low-texture regions,
    and a high-contrast foreground silhouette (head/shoulders + thin pole) in front of a smooth wall."""
    fx, baseline_m, cx_offset = 264.0, 0.11944, 0.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Smooth slanted background wall (1.4 m to 2.2 m)
    gt_z = (1.45 + 0.55 * (xx / float(w)) + 0.20 * (yy / float(h))).astype(np.float32)

    # Foreground silhouette (head + torso at 0.72 m)
    head_mask = ((xx - 220.0) ** 2 / (42.0**2) + (yy - 130.0) ** 2 / (52.0**2)) <= 1.0
    torso_mask = ((xx - 220.0) ** 2 / (95.0**2) + (yy - 275.0) ** 2 / (95.0**2)) <= 1.0
    fg_mask = head_mask | torso_mask
    gt_z[fg_mask] = (0.72 + 0.03 * ((xx[fg_mask] - 220.0) / 95.0)).astype(np.float32)

    # Thin vertical pole at x in [460, 466)
    pole_mask = (xx >= 460) & (xx < 466) & (yy >= 40) & (yy < 320)
    gt_z[pole_mask] = 1.05

    gt_disp = ((fx * baseline_m) / gt_z + cx_offset).astype(np.float32)
    base_tex = _multi_scale_texture(h, w, seed_offset=131, contrast=55.0)
    left_u8, right_u8, gt_disp_frame, non_occ = _render_stereo_pair_from_world_disparity(base_tex, gt_disp)
    yuyv = _pack_yuyv(left_u8, right_u8)
    return yuyv, gt_z, gt_disp_frame, non_occ


def run_four_way_diagnostic_and_audit(out_fig_path: Path | None = None) -> dict[str, Any]:
    cpu_backend = SlangCpuRefinementBackend()
    w, h = 640, 360
    wc, hc = w // 4, h // 4
    fx, baseline_m, cx_offset = 264.0, 0.11944, 0.0
    t_maps = _identity_maps(w, h)
    c_maps = _scale_maps(w, h, wc, hc)

    yuyv, gt_z, gt_disp, non_occ = build_diagnostic_test_scene(w, h)
    n = w * h

    configs = {
        "A_reference_adaptive0_planar0": {"adaptive_tiles": False, "planar_recovery": False},
        "B_adaptive_matching_only_adaptive1_planar0": {"adaptive_tiles": True, "planar_recovery": False},
        "C_planar_completion_only_adaptive0_planar1": {"adaptive_tiles": False, "planar_recovery": True},
        "D_combined_pipeline_adaptive1_planar1": {"adaptive_tiles": True, "planar_recovery": True},
    }

    runs: dict[str, dict[str, np.ndarray]] = {}
    for name, cfg in configs.items():
        runs[name] = cpu_backend.run_full_camera_to_depth_pipeline(
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
            continuous_subpixel=False,
            adaptive_tiles=cfg["adaptive_tiles"],
            min_disparity_bound=8,
            max_disparity_bound=56,
            ortho_hybrid=True,
            fast_mode=False,
            ultra_fast=False,
            selective_repair=True,
            planar_recovery=cfg["planar_recovery"],
        )

    luma_l = runs["A_reference_adaptive0_planar0"]["luma"][:n].reshape(h, w)
    shared_measured_support = np.ones((h, w), dtype=bool)
    shared_completed_support = np.ones((h, w), dtype=bool)
    for r in runs.values():
        shared_measured_support &= (r["validity"][:n].reshape(h, w) != 0) & non_occ
        shared_completed_support &= (r["completed_validity"][:n].reshape(h, w) != 0) & non_occ

    ref_d_meas = runs["A_reference_adaptive0_planar0"]["disparity"][:n].reshape(h, w)

    ablation_summary: dict[str, Any] = {}
    for name, r in runs.items():
        d_meas = r["disparity"][:n].reshape(h, w)
        v_meas = r["validity"][:n].reshape(h, w) != 0
        z_comp = r["completed_depth"][:n].reshape(h, w)
        v_comp = r["completed_validity"][:n].reshape(h, w) != 0
        d_comp = np.where(v_comp & (z_comp > 0), (fx * baseline_m) / np.maximum(z_comp, 1e-4), np.nan)

        grid_meas = compute_grid_alignment_metrics(d_meas, shared_measured_support, luma_l, ref_disp=ref_d_meas)
        grid_comp = compute_grid_alignment_metrics(d_comp, shared_completed_support, luma_l, ref_disp=ref_d_meas)

        ok_m = v_meas & non_occ & np.isfinite(d_meas)
        ok_c = v_comp & non_occ & np.isfinite(z_comp)
        ok_paired = shared_measured_support & np.isfinite(d_meas)
        z_meas = np.where(ok_m, (fx * baseline_m) / np.maximum(d_meas, 1e-4), np.nan)

        ablation_summary[name] = {
            "measured_valid_coverage_pct": float(np.mean(ok_m) * 100.0),
            "completed_valid_coverage_pct": float(np.mean(ok_c) * 100.0),
            "paired_support_depth_mae_mm": float(np.mean(np.abs(z_meas[ok_paired] - gt_z[ok_paired]) * 1000.0)),
            "paired_support_bad_0_5px_pct": float(np.mean(np.abs(d_meas[ok_paired] - gt_disp[ok_paired]) > 0.5) * 100.0),
            "completed_depth_mae_mm": float(np.mean(np.abs(z_comp[ok_c] - gt_z[ok_c]) * 1000.0)),
            "mean_evaluated_candidates": float(np.mean(r["evaluated_candidates"][:n])),
            "grid_alignment_measured": grid_meas,
            "grid_alignment_completed": grid_comp,
        }

    # Coarse predictor comparison: 1/4 single-level vs 1/8->1/4 hierarchical prior smoothing
    c_disp_1_4 = runs["D_combined_pipeline_adaptive1_planar1"]["coarse_disparity"][: wc * hc].reshape(hc, wc)
    c_val_1_4 = runs["D_combined_pipeline_adaptive1_planar1"]["coarse_validity"][: wc * hc].reshape(hc, wc) != 0
    c_disp_clean = np.where(c_val_1_4 & np.isfinite(c_disp_1_4), c_disp_1_4, 0.0).astype(np.float32)
    c_w_clean = c_val_1_4.astype(np.float32)
    up_1_4 = cv2.resize(c_disp_clean * 4.0, (w, h), interpolation=cv2.INTER_LINEAR) / np.maximum(
        cv2.resize(c_w_clean, (w, h), interpolation=cv2.INTER_LINEAR), 1e-4
    )
    c_disp_hier = cv2.pyrUp(cv2.pyrDown(c_disp_clean)) / np.maximum(cv2.pyrUp(cv2.pyrDown(c_w_clean)), 1e-4)
    up_hier = cv2.resize(c_disp_hier * 4.0, (w, h), interpolation=cv2.INTER_LINEAR)
    valid_coarse_up = non_occ & (cv2.resize(c_w_clean, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5)
    coarse_study = {
        "shared_census_1_4_coarse_mae_px": float(np.mean(np.abs(up_1_4[valid_coarse_up] - gt_disp[valid_coarse_up]))),
        "hierarchical_1_8_to_1_4_coarse_mae_px": float(np.mean(np.abs(up_hier[valid_coarse_up] - gt_disp[valid_coarse_up]))),
    }

    # Display quantization check (0.40 - 2.50 m -> 254 steps = 8.2677 mm/step)
    z_min_m, z_max_m = 0.40, 2.50
    step_mm = (z_max_m - z_min_m) / 254.0 * 1000.0
    quantization_study = {
        "colorizer_z_min_m": z_min_m,
        "colorizer_z_max_m": z_max_m,
        "colorizer_bins": 254,
        "mm_per_color_step": step_mm,
        "raw_float32_B4_mean": ablation_summary["D_combined_pipeline_adaptive1_planar1"]["grid_alignment_completed"]["B_4_mean"],
    }

    if out_fig_path is not None:
        r_d = runs["D_combined_pipeline_adaptive1_planar1"]
        prior_map = np.where(cv2.resize(c_w_clean, (w, h), interpolation=cv2.INTER_LINEAR) > 0.25, up_1_4, np.nan)
        span_map = np.clip(r_d["evaluated_candidates"][:n].reshape(h, w).astype(np.int32), 0, 16)
        meas_d = r_d["disparity"][:n].reshape(h, w)
        comp_z = r_d["completed_depth"][:n].reshape(h, w)
        comp_d = np.where(np.isfinite(comp_z) & (comp_z > 0), (fx * baseline_m) / np.maximum(comp_z, 1e-4), 0.0)
        prov = r_d["completed_provenance"][:n].reshape(h, w)

        def _color_panel(arr: np.ndarray, vmin: float, vmax: float, title: str) -> np.ndarray:
            valid = np.isfinite(arr) & (arr > 0)
            safe_arr = np.where(valid, arr, vmin)
            u8 = np.clip((safe_arr - vmin) / max(vmax - vmin, 1e-5) * 255.0, 0, 255).astype(np.uint8)
            col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
            col[~valid] = (24, 24, 24)
            cv2.putText(col, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            return col

        p1 = cv2.cvtColor(luma_l.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.putText(p1, "1. Rectified Luma & Silhouette", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 255, 80), 2, cv2.LINE_AA)
        p2 = _color_panel(prior_map, 10.0, 50.0, "2. Coarse Disparity Prior (4x)")
        p3 = _color_panel(span_map.astype(np.float32), 1.0, 12.0, "3. Adaptive Search Width (hi-lo+1)")
        p4 = _color_panel(meas_d, 10.0, 50.0, f"4. Measured Disp (B4={ablation_summary['D_combined_pipeline_adaptive1_planar1']['grid_alignment_measured']['B_4_mean']:.2f})")
        p5 = _color_panel(comp_d, 10.0, 50.0, f"5. Completed Disp (B4={ablation_summary['D_combined_pipeline_adaptive1_planar1']['grid_alignment_completed']['B_4_mean']:.2f})")
        p6 = _color_panel(prov.astype(np.float32), 0.0, 6.0, "6. Provenance (1=Stereo, 6=Planar)")

        grid_img = np.vstack([np.hstack([p1, p2, p3]), np.hstack([p4, p5, p6])])
        out_fig_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_fig_path), grid_img)

    report = {
        "four_way_ablation": ablation_summary,
        "coarse_predictor_study": coarse_study,
        "display_quantization_study": quantization_study,
    }
    OUT_AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    artifact_fig = Path("/home/bjoern/.gemini/antigravity-cli/brain/6a5df588-91c3-4b8d-8e19-d968e1de4f3f/v13_1_four_way_rectangular_audit.png")
    res = run_four_way_diagnostic_and_audit(out_fig_path=artifact_fig)
    print(json.dumps(res, indent=2))
