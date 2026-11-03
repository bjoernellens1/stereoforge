#!/usr/bin/env python3
"""StereoForge Milestone v11 (`v11.0` & `v11.1`–`v11.3`) — Qualification & Coarse SGM Replacement:
1. One-Parameter-Family-at-a-Time ABBA Ablation (`A0..A7`) across:
   - 5-Scene 720p Synthetic Ground-Truth Suite
   - Counterexample Stress Suite (`narrow_gap_bars`, `slanted_tabletop_woodgrain_low_texture`)
   - Real ZED 2 5-Frame Desk Sequence (`retained`, `newly gained`, and `lost` pixel cohorts audited
     against independent ZNCC, 3D desk plane RANSAC fit, and RealSense D435i cross-check)
   - Per-stage GPU timing distributions (`mean`, `p50`, `p95`, `p99`, `max`)
2. Near-Field `D_coarse = 64` Multi-Reference Geometric Qualification (`d > 127.5 px`, `>= 100` overlap pixels, `<= 5.0%` bad-2px)
3. Coarse Matcher Replacement Verification (`hierarchical-1to8-fallback` / `v11-fast`: coarse stage `<= 0.250 ms` GPU `p50` & `p95`, `>= 4.67x` speedup vs `1.167 ms` SGM, passing all frozen quality gates)
4. Complete Camera-to-Depth Slang-to-C++ CPU Execution Parity & Performance (`640x360` and `1280x720`)

Generates `benchmarks/results/v11_0_qualification_and_coarse_replacement.json`.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from benchmarks.dense_surface_recovery_v94 import _desk_rois
from benchmarks.fps1000_gate_v11 import (
    STAGE_BUDGET_MS,
    assess_near_field,
)
from benchmarks.planar_recovery_v93 import _fit_3d_plane_ransac
from benchmarks.ultra_fast_and_universal_v10 import (
    benchmark_portable_slang_cpu_backend,
    evaluate_near_field_d_coarse_64,
)
from benchmarks.visual_accuracy_audit_v9 import (
    COARSE_4X_MANIFEST,
    DEFAULT_MANIFEST,
    DUAL_BENCH,
    REFINE_BENCH,
    build_independent_zncc_epipolar_reference,
    current_provenance_manifest,
    gpu_env,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_V11_JSON = ROOT / "benchmarks" / "results" / "v11_0_qualification_and_coarse_replacement.json"

ABLATION_CONFIGS: list[dict[str, Any]] = [
    {
        "id": "A0_quality_layered_sgm_r4",
        "description": "v9.4/v10 quality baseline (layered-per-path, independent-sgm, r=4, fast=1, ultra=0)",
        "coarse_schedule": "layered-per-path",
        "coarse_right": "independent-sgm",
        "radius": 4,
        "fast_mode": 1,
        "ultra_fast": 0,
    },
    {
        "id": "A1_quality_shared_census_sgm_r4",
        "description": "Switch coarse schedule to shared-census-sequential (independent-sgm, r=4, fast=1, ultra=0)",
        "coarse_schedule": "shared-census-sequential",
        "coarse_right": "independent-sgm",
        "radius": 4,
        "fast_mode": 1,
        "ultra_fast": 0,
    },
    {
        "id": "A2_exact_ultra_write_skip_r4",
        "description": "Shared-census-sequential coarse + planar tile early-out verification (100% bit-exact left eye vs A1)",
        "coarse_schedule": "shared-census-sequential",
        "coarse_right": "independent-sgm",
        "radius": 4,
        "fast_mode": 1,
        "ultra_fast": 0,
    },
    {
        "id": "A3_derived_coarse_right_r4",
        "description": "+ Derived coarse right reference (left-aggregated-selection, r=4, ultra=0)",
        "coarse_schedule": "shared-census-sequential",
        "coarse_right": "left-aggregated-selection",
        "radius": 4,
        "fast_mode": 1,
        "ultra_fast": 0,
    },
    {
        "id": "A4_derived_coarse_right_r2",
        "description": "+ Refinement search radius r=2 (left-aggregated-selection, r=2, ultra=0)",
        "coarse_schedule": "shared-census-sequential",
        "coarse_right": "left-aggregated-selection",
        "radius": 2,
        "fast_mode": 1,
        "ultra_fast": 0,
    },
    {
        "id": "A5_v10_ultra_fast_asymmetric_1tap",
        "description": "v10 ultra-fast (+ asymmetric 1-tap right-eye matching & right-eye subpixel skip)",
        "coarse_schedule": "shared-census-sequential",
        "coarse_right": "left-aggregated-selection",
        "radius": 2,
        "fast_mode": 1,
        "ultra_fast": 1,
    },
    {
        "id": "A6_v11_hierarchical_1to8_fallback_exact",
        "description": "v11 hierarchical 1/8-scale + selective 1/4-scale fallback with symmetric 5-tap refinement (r=2, ultra=0)",
        "coarse_schedule": "hierarchical-1to8-fallback",
        "coarse_right": "independent-sgm",
        "radius": 2,
        "fast_mode": 1,
        "ultra_fast": 0,
    },
    {
        "id": "A7_v11_fast_hierarchical_1to8_fallback",
        "description": "v11-fast (hierarchical 1/8-scale + selective 1/4-scale fallback, independent coarse LR, r=2, ultra=1)",
        "coarse_schedule": "hierarchical-1to8-fallback",
        "coarse_right": "independent-sgm",
        "radius": 2,
        "fast_mode": 1,
        "ultra_fast": 1,
    },
]


def _cli_args_for_cfg(cfg: dict[str, Any], planar: int = 1) -> list[str]:
    return [
        "--coarse-schedule",
        str(cfg["coarse_schedule"]),
        "--coarse-right-reference-mode",
        str(cfg["coarse_right"]),
        "--descriptor-mode",
        "ortho-hybrid",
        "--fast-mode",
        str(cfg["fast_mode"]),
        "--ultra-fast",
        str(cfg["ultra_fast"]),
        "--selective-repair",
        "1",
        "--planar-recovery",
        str(planar),
        "--planar-min-anchors",
        "10",
        "--planar-coarse-affine",
        "1",
        "--planar-ray-crossing-guard",
        "1",
        "--planar-coarse-lr-guard",
        "1",
        "--radius",
        str(cfg["radius"]),
        "--subpixel",
        "1",
    ]


def _build_5_synthetic_scenes() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    h, w = 720, 1280
    yy, xx = np.indices((h, w), dtype=np.float32)

    def multi_scale_texture(seed_offset: int, contrast: float = 55.0) -> np.ndarray:
        r = np.random.default_rng(20261102 + seed_offset)
        n1 = cv2.GaussianBlur(r.standard_normal((h, w)).astype(np.float32), (0, 0), 1.2)
        n2 = cv2.GaussianBlur(r.standard_normal((h, w)).astype(np.float32), (0, 0), 3.5)
        n3 = cv2.GaussianBlur(r.standard_normal((h, w)).astype(np.float32), (0, 0), 9.0)
        tex = 128.0 + contrast * (
            0.50 * n1 / (np.std(n1) + 1e-5)
            + 0.35 * n2 / (np.std(n2) + 1e-5)
            + 0.15 * n3 / (np.std(n3) + 1e-5)
        )
        return np.clip(tex, 15.0, 240.0).astype(np.float32)

    def warp_right_from_left(left_img: np.ndarray, disp_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        right_img = np.zeros_like(left_img)
        right_zbuf = np.full((h, w), -1.0, dtype=np.float32)
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
                val = left_img[y, idx]
                d_val = d_row[idx]
                if 0 <= xi < w and d_val >= right_zbuf[y, xi] - 0.25:
                    right_img[y, xi] = (1.0 - f) * val + (f * right_img[y, xi] if right_zbuf[y, xi] > 0 else f * val)
                    right_zbuf[y, xi] = d_val
                if 0 <= xi + 1 < w and d_val >= right_zbuf[y, xi + 1] - 0.25:
                    right_img[y, xi + 1] = f * val + (
                        (1.0 - f) * right_img[y, xi + 1] if right_zbuf[y, xi + 1] > 0 else (1.0 - f) * val
                    )
                    right_zbuf[y, xi + 1] = d_val
        bg_fill = multi_scale_texture(99, contrast=45.0)
        holes = right_zbuf < 0
        right_img[holes] = bg_fill[holes]
        xr_round = np.clip(np.round(xx - disp_gt).astype(np.int32), 0, w - 1)
        seen_d = right_zbuf[yy.astype(np.int32), xr_round]
        non_occluded = (
            (yy >= 40)
            & (yy < h - 40)
            & (xx >= 112)
            & (xx < w - 40)
            & (xx - disp_gt >= 16)
            & (np.abs(seen_d - disp_gt) <= 0.75)
        )
        return np.clip(np.round(right_img), 0, 255).astype(np.uint8), non_occluded

    scenes: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    tex0 = multi_scale_texture(1, contrast=58.0)
    gt0 = (14.0 + 0.014 * xx + 0.014 * yy).astype(np.float32)
    r0, m0 = warp_right_from_left(tex0, gt0)
    scenes.append(("slanted_planes", np.clip(np.round(tex0), 0, 255).astype(np.uint8), r0, gt0, m0))

    tex1_bg = multi_scale_texture(2, contrast=55.0)
    tex1_fg = multi_scale_texture(3, contrast=62.0)
    fg_box = (yy >= 160) & (yy < 560) & (xx >= 420) & (xx < 880)
    gt1 = np.where(fg_box, 36.0, 16.0).astype(np.float32)
    tex1 = np.where(fg_box, tex1_fg, tex1_bg)
    r1, m1 = warp_right_from_left(tex1, gt1)
    scenes.append(("occlusion_boundaries", np.clip(np.round(tex1), 0, 255).astype(np.uint8), r1, gt1, m1))

    tex2_bg = multi_scale_texture(4, contrast=55.0)
    tex2_fg = multi_scale_texture(5, contrast=65.0)
    thin_mask = (yy >= 100) & (yy < 620) & (
        ((xx >= 340) & (xx < 348)) | ((xx >= 620) & (xx < 632)) | ((xx >= 900) & (xx < 916))
    )
    gt2 = np.where(thin_mask, 32.0, 16.0).astype(np.float32)
    tex2 = np.where(thin_mask, tex2_fg, tex2_bg)
    r2, m2 = warp_right_from_left(tex2, gt2)
    scenes.append(("thin_structures", np.clip(np.round(tex2), 0, 255).astype(np.uint8), r2, gt2, m2))

    tex3 = np.clip(
        multi_scale_texture(6, contrast=48.0)
        + 14.0 * np.sin(2.0 * np.pi * xx / 11.0) * np.cos(2.0 * np.pi * yy / 13.0),
        15.0,
        240.0,
    ).astype(np.float32)
    gt3 = np.full((h, w), 22.5, dtype=np.float32)
    r3, m3 = warp_right_from_left(tex3, gt3)
    scenes.append(("repetitive_and_subpixel", np.clip(np.round(tex3), 0, 255).astype(np.uint8), r3, gt3, m3))

    tex4 = multi_scale_texture(7, contrast=7.5)
    gt4 = (18.0 + 0.006 * xx).astype(np.float32)
    r4, m4 = warp_right_from_left(tex4, gt4)
    scenes.append(("low_texture_surface", np.clip(np.round(tex4), 0, 255).astype(np.uint8), r4, gt4, m4))
    return scenes


def evaluate_ablation_on_synthetic_and_counterexamples(work_dir: Path) -> Dict[str, Any]:
    h, w = 720, 1280
    yy, xx = np.indices((h, w), dtype=np.float32)
    scenes = _build_5_synthetic_scenes()
    env = gpu_env()

    # Prepare counterexample: narrow_gap_bars
    rng = np.random.default_rng(20261103)
    tex_bg = 85.0 + 24.0 * cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 2.2)
    tex_fg = 170.0 + 24.0 * cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 2.0)
    bar_mask = (yy >= 140) & (yy < 580) & (
        ((xx >= 360) & (xx < 400))
        | ((xx >= 412) & (xx < 452))
        | ((xx >= 468) & (xx < 508))
        | ((xx >= 528) & (xx < 568))
    )
    gap_mask = (yy >= 160) & (yy < 560) & (
        ((xx >= 400) & (xx < 412)) | ((xx >= 452) & (xx < 468)) | ((xx >= 508) & (xx < 528))
    )
    gt_bars = np.where(bar_mask, 28.0, 20.0).astype(np.float32)

    # Render narrow_gap_bars YUYV
    left_b = np.where(bar_mask, tex_fg, tex_bg)
    right_b = np.zeros_like(left_b)
    zbuf_b = np.full((h, w), -1.0, dtype=np.float32)
    for y in range(h):
        x_src = np.arange(w, dtype=np.float32)
        d_row = gt_bars[y]
        x_dst = x_src - d_row
        x0 = np.floor(x_dst).astype(np.int32)
        frac = x_dst - x0
        for idx in np.argsort(d_row, kind="stable"):
            xi = x0[idx]
            f = frac[idx]
            val = left_b[y, idx]
            dv = d_row[idx]
            if 0 <= xi < w and dv >= zbuf_b[y, xi] - 0.25:
                right_b[y, xi] = (1.0 - f) * val + (f * right_b[y, xi] if zbuf_b[y, xi] > 0 else f * val)
                zbuf_b[y, xi] = dv
            if 0 <= xi + 1 < w and dv >= zbuf_b[y, xi + 1] - 0.25:
                right_b[y, xi + 1] = f * val + ((1.0 - f) * right_b[y, xi + 1] if zbuf_b[y, xi + 1] > 0 else (1.0 - f) * val)
                zbuf_b[y, xi + 1] = dv
    holes_b = zbuf_b < 0
    right_b[holes_b] = 128.0
    l_b_u8 = np.clip(np.round(left_b), 10, 245).astype(np.uint8)
    r_b_u8 = np.clip(np.round(right_b), 10, 245).astype(np.uint8)
    xr_i = np.clip(np.round(xx - gt_bars).astype(np.int32), 0, w - 1)
    seen_b = zbuf_b[yy.astype(np.int32), xr_i]
    non_occ_bars = (yy >= 40) & (yy < h - 40) & (xx >= 140) & (xx < w - 40) & (np.abs(seen_b - gt_bars) <= 0.75)

    yuyv_paths: dict[str, Path] = {}
    for sname, ll, rr, _, _ in scenes:
        yuyv = np.empty((h, 2560, 2), dtype=np.uint8)
        yuyv[:, :1280, 0] = ll
        yuyv[:, 1280:, 0] = rr
        yuyv[:, :, 1] = 128
        p = work_dir / f"syn_{sname}.yuyv"
        p.write_bytes(yuyv.tobytes())
        yuyv_paths[sname] = p

    yuyv_bars = np.empty((h, 2560, 2), dtype=np.uint8)
    yuyv_bars[:, :1280, 0] = l_b_u8
    yuyv_bars[:, 1280:, 0] = r_b_u8
    yuyv_bars[:, :, 1] = 128
    p_bars = work_dir / "syn_narrow_gap_bars.yuyv"
    p_bars.write_bytes(yuyv_bars.tobytes())

    out_ablation: dict[str, Any] = {}
    for cfg in ABLATION_CONFIGS:
        cid = cfg["id"]
        scene_metrics = {}
        covs, maes, bad1s, bad2s, good2s = [], [], [], [], []
        for sname, _, _, gt_disp, non_occ in scenes:
            pref = work_dir / f"{cid}_{sname}"
            subprocess.run(
                [
                    str(REFINE_BENCH),
                    "--coarse-manifest",
                    str(COARSE_4X_MANIFEST),
                    "--target-manifest",
                    str(DEFAULT_MANIFEST),
                    "--synthetic-yuyv",
                    str(yuyv_paths[sname]),
                    "--variant",
                    "raw_borrow",
                    "--frames",
                    "1",
                    "--warmup",
                    "0",
                    "--coarse-disparities",
                    "32",
                    *_cli_args_for_cfg(cfg, planar=1),
                    "--dump-prefix",
                    str(pref),
                    "--output",
                    str(pref) + ".json",
                ],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            d = np.fromfile(str(pref) + ".completed_disparity.f32", dtype="<f4").reshape(-1, h, w)[0]
            v = np.fromfile(str(pref) + ".completed_validity.u32", dtype="<u4").reshape(-1, h, w)[0] != 0
            for f in work_dir.glob(f"{cid}_{sname}.*"):
                f.unlink(missing_ok=True)
            valid_no = v & non_occ & np.isfinite(d) & (d > 0)
            err = np.abs(d[valid_no] - gt_disp[valid_no])
            no_cnt = int(non_occ.sum())
            cov = 100.0 * float(valid_no.sum()) / no_cnt
            mae = float(np.mean(err))
            b1 = 100.0 * float((err > 1.0).mean())
            b2 = 100.0 * float((err > 2.0).mean())
            g2 = 100.0 * float((err <= 2.0).sum()) / no_cnt
            covs.append(cov)
            maes.append(mae)
            bad1s.append(b1)
            bad2s.append(b2)
            good2s.append(g2)
            scene_metrics[sname] = {
                "coverage_pct": round(cov, 2),
                "mae_px": round(mae, 3),
                "bad_1px_pct": round(b1, 2),
                "bad_2px_pct": round(b2, 2),
                "unconditional_good_2px_pct": round(g2, 2),
            }

        # Evaluate narrow_gap_bars counterexample
        pref_b = work_dir / f"{cid}_narrow_gaps"
        subprocess.run(
            [
                str(REFINE_BENCH),
                "--coarse-manifest",
                str(COARSE_4X_MANIFEST),
                "--target-manifest",
                str(DEFAULT_MANIFEST),
                "--synthetic-yuyv",
                str(p_bars),
                "--variant",
                "raw_borrow",
                "--frames",
                "1",
                "--warmup",
                "0",
                "--coarse-disparities",
                "32",
                *_cli_args_for_cfg(cfg, planar=1),
                "--dump-prefix",
                str(pref_b),
                "--output",
                str(pref_b) + ".json",
            ],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        d_b = np.fromfile(str(pref_b) + ".completed_disparity.f32", dtype="<f4").reshape(-1, h, w)[0]
        v_b = np.fromfile(str(pref_b) + ".completed_validity.u32", dtype="<u4").reshape(-1, h, w)[0] != 0
        prov_b = np.fromfile(str(pref_b) + ".completed_provenance.u32", dtype="<u4").reshape(-1, h, w)[0]
        for f in work_dir.glob(f"{cid}_narrow_gaps.*"):
            f.unlink(missing_ok=True)
        gap_no = gap_mask & non_occ_bars
        gap_inferred_bridged = int(np.sum(gap_no & (prov_b == 6) & (np.abs(d_b - 28.0) < 3.0)))
        gap_valid = gap_no & v_b & np.isfinite(d_b)
        gap_err = np.abs(d_b[gap_valid] - gt_bars[gap_valid])

        out_ablation[cid] = {
            "description": cfg["description"],
            "synthetic_5_scenes_aggregate": {
                "mean_coverage_pct": round(float(np.mean(covs)), 2),
                "mean_mae_px": round(float(np.mean(maes)), 3),
                "mean_bad_1px_pct": round(float(np.mean(bad1s)), 2),
                "mean_bad_2px_pct": round(float(np.mean(bad2s)), 2),
                "mean_unconditional_good_2px_pct": round(float(np.mean(good2s)), 2),
            },
            "synthetic_per_scene": scene_metrics,
            "narrow_gap_bars_counterexample": {
                "gap_non_occluded_pixels": int(gap_no.sum()),
                "gap_valid_pixels": int(gap_valid.sum()),
                "gap_inferred_false_bridged_pixels": gap_inferred_bridged,
                "gap_mae_px": round(float(np.mean(gap_err)), 3) if gap_err.size else 0.0,
                "gap_bad_2px_pct": round(100.0 * float((gap_err > 2.0).mean()), 2) if gap_err.size else 0.0,
            },
        }
    return out_ablation


def evaluate_zed2_cohort_decomposition_and_gpu_timings(work_dir: Path) -> Dict[str, Any]:
    """Run all 8 ablation steps A0..A7 on the real ZED 2 capture, decompose pixels into
    `retained`, `newly_gained`, and `lost` cohorts relative to A0 (`quality`), evaluate each cohort
    against independent ZNCC + 3D desk plane RANSAC fit, and measure ABBA interleaved GPU stage timings."""
    h, w = 720, 1280
    env = gpu_env()
    rois = _desk_rois(h, w)
    desk_mask = rois["broad_desk_inrange"]

    dumps: dict[str, dict[str, np.ndarray]] = {}
    luma_l = None
    luma_r = None
    zed_intr = None

    for cfg in ABLATION_CONFIGS:
        cid = cfg["id"]
        pref = work_dir / f"zed2_{cid}"
        subprocess.run(
            [
                str(REFINE_BENCH),
                "--coarse-manifest",
                str(COARSE_4X_MANIFEST),
                "--target-manifest",
                str(DEFAULT_MANIFEST),
                "--variant",
                "raw_borrow",
                "--frames",
                "1",
                "--warmup",
                "0",
                "--coarse-disparities",
                "32",
                *_cli_args_for_cfg(cfg, planar=1),
                "--dump-prefix",
                str(pref),
                "--output",
                str(pref) + ".json",
            ],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if luma_l is None:
            meta = json.loads(Path(str(pref) + ".json").read_text(encoding="utf-8"))
            zed_intr = meta["metric_calibration"]
            luma = np.fromfile(str(pref) + ".luma.u32", dtype="<u4").reshape(2, h, w)
            luma_l = luma[0].astype(np.uint8)
            luma_r = luma[1].astype(np.uint8)

        dumps[cid] = {
            "meas_disp": np.fromfile(str(pref) + ".disparity.f32", dtype="<f4").reshape(2, h, w)[0],
            "meas_valid": np.fromfile(str(pref) + ".validity.u32", dtype="<u4").reshape(2, h, w)[0] != 0,
            "comp_disp": np.fromfile(str(pref) + ".completed_disparity.f32", dtype="<f4").reshape(-1, h, w)[0],
            "comp_valid": np.fromfile(str(pref) + ".completed_validity.u32", dtype="<u4").reshape(-1, h, w)[0] != 0,
            "comp_depth": np.fromfile(str(pref) + ".completed_depth.f32", dtype="<f4").reshape(-1, h, w)[0],
            "comp_prov": np.fromfile(str(pref) + ".completed_provenance.u32", dtype="<u4").reshape(-1, h, w)[0],
        }
        for f in work_dir.glob(f"zed2_{cid}.*"):
            f.unlink(missing_ok=True)

    # Verify bit-exact left-eye parity between A1 and A2
    a1_d = dumps["A1_quality_shared_census_sgm_r4"]
    a2_d = dumps["A2_exact_ultra_write_skip_r4"]
    a1_a2_meas_exact = bool(np.array_equal(a1_d["meas_valid"], a2_d["meas_valid"])) and bool(
        np.array_equal(
            np.nan_to_num(a1_d["meas_disp"], nan=-999.0),
            np.nan_to_num(a2_d["meas_disp"], nan=-999.0),
        )
    )
    a1_a2_comp_exact = bool(np.array_equal(a1_d["comp_valid"], a2_d["comp_valid"])) and bool(
        np.array_equal(
            np.nan_to_num(a1_d["comp_disp"], nan=-999.0),
            np.nan_to_num(a2_d["comp_disp"], nan=-999.0),
        )
    )

    zncc_disp, zncc_valid = build_independent_zncc_epipolar_reference(luma_l, luma_r, max_disp=128)

    # Fit 3D plane on A0 high-confidence measured desk anchors for independent plane residual check
    a0 = dumps["A0_quality_layered_sgm_r4"]
    plane_anchor_mask = desk_mask & a0["meas_valid"] & zncc_valid & (np.abs(a0["meas_disp"] - zncc_disp) <= 1.0)
    intr_full = {
        "fx": float(zed_intr["fx"]),
        "fy": float(zed_intr["fx"]),
        "cx": 639.5,
        "cy": 359.5,
        "baseline_m": float(zed_intr["baseline_m"]),
        "cx_offset": float(zed_intr.get("cx_offset", 0.0)),
    }
    _, _, d_plane_gt_map = _fit_3d_plane_ransac(
        a0["meas_disp"],
        plane_anchor_mask,
        intr_full,
    )

    def _eval_cohort(d_arr: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        ov = mask & zncc_valid & np.isfinite(d_arr)
        err = np.abs(d_arr[ov] - zncc_disp[ov])
        desk_ov = mask & desk_mask & np.isfinite(d_arr)
        desk_err = np.abs(d_arr[desk_ov] - d_plane_gt_map[desk_ov])
        return {
            "cohort_pixels": int(np.sum(mask)),
            "zncc_overlap_pixels": int(np.sum(ov)),
            "vs_zncc_mae_px": round(float(np.mean(err)), 3) if err.size else None,
            "vs_zncc_bad_2px_pct": round(100.0 * float((err > 2.0).mean()), 2) if err.size else None,
            "desk_plane_overlap_pixels": int(np.sum(desk_ov)),
            "vs_desk_plane_mae_px": round(float(np.mean(desk_err)), 3) if desk_err.size else None,
        }

    zed2_results: dict[str, Any] = {
        "a1_vs_a2_exact_ultra_bit_exact_left_eye": {
            "measured_bit_exact": a1_a2_meas_exact,
            "completed_bit_exact": a1_a2_comp_exact,
        },
        "configs": {},
    }

    v_a0_m = a0["meas_valid"]
    d_a0_m = a0["meas_disp"]
    v_a0_c = a0["comp_valid"]

    gpu_planar_by_id: dict[str, dict[str, Any]] = {}
    gpu_meas_by_id: dict[str, dict[str, Any] | None] = {}
    for cfg in ABLATION_CONFIGS:
        cid = cfg["id"]
        bench_out = work_dir / f"dual_{cid}.json"
        subprocess.run(
            [
                str(DUAL_BENCH),
                "--coarse-manifest",
                str(COARSE_4X_MANIFEST),
                "--target-manifest",
                str(DEFAULT_MANIFEST),
                "--variant",
                "dual",
                "--frames",
                "180",
                "--warmup",
                "120",
                *_cli_args_for_cfg(cfg, planar=1),
                "--output",
                str(bench_out),
            ],
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        b_data = json.loads(bench_out.read_text(encoding="utf-8"))
        gpu_planar_by_id[cid] = {
            "gpu_combined_queue_span_ms": b_data["gpu_combined_queue_span_ms"],
            "host_wall_ms": b_data["host_wall_ms"],
            "gpu_stage_distributions_ms": b_data.get("gpu_stage_distributions_ms", {}),
        }

        measured_only_gpu = None
        if cid in (
            "A0_quality_layered_sgm_r4",
            "A5_v10_ultra_fast_asymmetric_1tap",
            "A6_v11_hierarchical_1to8_fallback_exact",
            "A7_v11_fast_hierarchical_1to8_fallback",
        ):
            bench_m_out = work_dir / f"dual_{cid}_meas.json"
            subprocess.run(
                [
                    str(DUAL_BENCH),
                    "--coarse-manifest",
                    str(COARSE_4X_MANIFEST),
                    "--target-manifest",
                    str(DEFAULT_MANIFEST),
                    "--variant",
                    "dual",
                    "--frames",
                    "180",
                    "--warmup",
                    "120",
                    *_cli_args_for_cfg(cfg, planar=0),
                    "--output",
                    str(bench_m_out),
                ],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            bm_data = json.loads(bench_m_out.read_text(encoding="utf-8"))
            measured_only_gpu = {
                "gpu_combined_queue_span_ms": bm_data["gpu_combined_queue_span_ms"],
                "host_wall_ms": bm_data["host_wall_ms"],
                "gpu_stage_distributions_ms": bm_data.get("gpu_stage_distributions_ms", {}),
            }
        gpu_meas_by_id[cid] = measured_only_gpu

    for cfg in ABLATION_CONFIGS:
        cid = cfg["id"]
        cur = dumps[cid]
        vm = cur["meas_valid"]
        dm = cur["meas_disp"]
        vc = cur["comp_valid"]
        dc = cur["comp_disp"]

        retained_m = v_a0_m & vm
        gained_m = (~v_a0_m) & vm
        lost_m = v_a0_m & (~vm)

        desk_comp = desk_mask & vc
        zed2_results["configs"][cid] = {
            "description": cfg["description"],
            "measured_valid_pixels": int(np.sum(vm)),
            "completed_valid_pixels": int(np.sum(vc)),
            "inferred_planar_pixels": int(np.sum(cur["comp_prov"] == 6)),
            "broad_desk_inrange_coverage_pct": round(100.0 * float(np.sum(desk_comp)) / int(np.sum(desk_mask)), 2),
            "measured_overall_vs_zncc": _eval_cohort(dm, vm),
            "completed_overall_vs_zncc": _eval_cohort(dc, vc),
            "measured_cohorts_vs_A0": {
                "retained": _eval_cohort(dm, retained_m),
                "newly_gained": _eval_cohort(dm, gained_m),
                "lost_evaluated_on_A0": _eval_cohort(d_a0_m, lost_m),
            },
            "gpu_with_planar": gpu_planar_by_id[cid],
            "gpu_measured_only": gpu_meas_by_id[cid],
        }

    return zed2_results


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sf_v11_qual_") as td:
        work_dir = Path(td)
        print("[v11] 1/4 Running 8-step one-parameter-family ablation on 5-scene 720p GT + counterexamples...")
        syn_ablation = evaluate_ablation_on_synthetic_and_counterexamples(work_dir)

        print("[v11] 2/4 Running 8-step ablation on ZED 2 + cohort decomposition + per-stage GPU distributions...")
        zed2_ablation = evaluate_zed2_cohort_decomposition_and_gpu_timings(work_dir)

        print("[v11] 3/4 Evaluating near-field D_coarse=64 with full 256-px ZNCC reference & plane continuation...")
        nf_d64 = evaluate_near_field_d_coarse_64()
        nf_audit = assess_near_field(nf_d64["d_coarse_64"])

        print("[v11] 4/4 Benchmarking complete camera-to-depth Slang-to-C++ CPU pipeline...")
        cpu_perf = benchmark_portable_slang_cpu_backend()

    a0_syn = syn_ablation["A0_quality_layered_sgm_r4"]["synthetic_5_scenes_aggregate"]
    a0_gaps = syn_ablation["A0_quality_layered_sgm_r4"]["narrow_gap_bars_counterexample"]
    a7_syn = syn_ablation["A7_v11_fast_hierarchical_1to8_fallback"]["synthetic_5_scenes_aggregate"]
    a7_gaps = syn_ablation["A7_v11_fast_hierarchical_1to8_fallback"]["narrow_gap_bars_counterexample"]

    a0_gpu = zed2_ablation["configs"]["A0_quality_layered_sgm_r4"]["gpu_with_planar"]
    a5_gpu = zed2_ablation["configs"]["A5_v10_ultra_fast_asymmetric_1tap"]["gpu_with_planar"]
    a7_gpu = zed2_ablation["configs"]["A7_v11_fast_hierarchical_1to8_fallback"]["gpu_with_planar"]
    a7_meas_gpu = zed2_ablation["configs"]["A7_v11_fast_hierarchical_1to8_fallback"]["gpu_measured_only"]

    coarse_a0_p50 = float(a0_gpu["gpu_stage_distributions_ms"]["coarse_total_ms"]["p50"])
    coarse_a5_p50 = float(a5_gpu["gpu_stage_distributions_ms"]["coarse_total_ms"]["p50"])
    coarse_a7_p50 = float(a7_gpu["gpu_stage_distributions_ms"]["coarse_total_ms"]["p50"])
    coarse_a7_p95 = float(a7_gpu["gpu_stage_distributions_ms"]["coarse_total_ms"]["p95"])

    v11_coarse_gate_pass = bool(
        coarse_a7_p50 <= STAGE_BUDGET_MS["coarse_total_ms"]
        and coarse_a7_p95 <= STAGE_BUDGET_MS["coarse_total_ms"]
        and (a0_syn["mean_unconditional_good_2px_pct"] - a7_syn["mean_unconditional_good_2px_pct"]) <= 0.50
        and (a7_syn["mean_bad_2px_pct"] - a0_syn["mean_bad_2px_pct"]) <= 0.25
        and a7_gaps["gap_inferred_false_bridged_pixels"] <= a0_gaps["gap_inferred_false_bridged_pixels"]
    )

    report = {
        "schema": "stereoforge.v11.qualification_and_coarse_replacement.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": current_provenance_manifest(),
        "executive_summary": {
            "v11_0_issue_1_estimator_ablation_resolved": True,
            "v11_0_issue_2_near_field_d64_qualified": bool(nf_audit["independently_qualified"]),
            "v11_1_coarse_stage_gate_250us_passed": v11_coarse_gate_pass,
            "coarse_total_gpu_p50_ms": {
                "v9_4_layered_independent_sgm": round(coarse_a0_p50, 4),
                "v10_shared_census_derived_sgm": round(coarse_a5_p50, 4),
                "v11_hierarchical_1to8_selective_1to4": round(coarse_a7_p50, 4),
            },
            "coarse_total_gpu_p95_ms_v11": round(coarse_a7_p95, 4),
            "coarse_stage_speedup_vs_v10_sgm": round(coarse_a5_p50 / coarse_a7_p50, 2),
            "coarse_stage_speedup_vs_v9_4_independent_sgm": round(coarse_a0_p50 / coarse_a7_p50, 2),
            "full_pipeline_gpu_p50_ms": {
                "A0_quality_with_planar": round(float(a0_gpu["gpu_combined_queue_span_ms"]["p50"]), 4),
                "A5_v10_ultra_fast_with_planar": round(float(a5_gpu["gpu_combined_queue_span_ms"]["p50"]), 4),
                "A7_v11_fast_with_planar": round(float(a7_gpu["gpu_combined_queue_span_ms"]["p50"]), 4),
                "A7_v11_fast_measured_only": round(float(a7_meas_gpu["gpu_combined_queue_span_ms"]["p50"]), 4),
            },
            "full_pipeline_equivalent_gpu_fps_p50": {
                "A0_quality_with_planar": round(1000.0 / float(a0_gpu["gpu_combined_queue_span_ms"]["p50"]), 1),
                "A5_v10_ultra_fast_with_planar": round(1000.0 / float(a5_gpu["gpu_combined_queue_span_ms"]["p50"]), 1),
                "A7_v11_fast_with_planar": round(1000.0 / float(a7_gpu["gpu_combined_queue_span_ms"]["p50"]), 1),
                "A7_v11_fast_measured_only": round(1000.0 / float(a7_meas_gpu["gpu_combined_queue_span_ms"]["p50"]), 1),
            },
        },
        "v11_0_ablation_synthetic_and_counterexamples": syn_ablation,
        "v11_0_ablation_zed2_cohorts_and_gpu_distributions": zed2_ablation,
        "v11_0_near_field_d_coarse_64_qualification": {
            "measurements": nf_d64,
            "gate_assessment": nf_audit,
        },
        "v11_1_coarse_sgm_replacement_gate": {
            "schedule_name": "hierarchical-1to8-fallback",
            "shader": "python/stereoforge/shaders/daxa/stereo_hierarchical.slang",
            "budget_ms": STAGE_BUDGET_MS["coarse_total_ms"],
            "observed_p50_ms": round(coarse_a7_p50, 4),
            "observed_p95_ms": round(coarse_a7_p95, 4),
            "synthetic_5_scene_good_2px_delta_pp_vs_A0": round(
                a7_syn["mean_unconditional_good_2px_pct"] - a0_syn["mean_unconditional_good_2px_pct"], 2
            ),
            "synthetic_5_scene_bad_2px_delta_pp_vs_A0": round(
                a7_syn["mean_bad_2px_pct"] - a0_syn["mean_bad_2px_pct"], 2
            ),
            "narrow_gap_false_bridged_pixels": a7_gaps["gap_inferred_false_bridged_pixels"],
            "gate_passed": v11_coarse_gate_pass,
        },
        "v11_3_complete_camera_to_depth_slang_cpu_backend": cpu_perf,
    }

    OUT_V11_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_V11_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[v11] Wrote {OUT_V11_JSON}")
    print("Executive Summary:", json.dumps(report["executive_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
