#!/usr/bin/env python3
"""Milestone v9 (Steps 8 & 9): Fixed-Region Visual & Quantitative Accuracy Audit
and Budgeted Selective 4x Refinement Evaluation on Identical Spatial Support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from benchmarks.lr_confidence_audit_v6 import read_fixture_luma
from benchmarks.matching_cost_inspector_v8 import geometric_qualification_metrics
from benchmarks.stereo_reference_crosscheck_v5 import cv_sgbm, lr_pair, valid_map

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json"


def build_fixed_spatial_regions(
    luma_l: np.ndarray,
    consensus_ref_valid: np.ndarray,
    consensus_ref_disp: np.ndarray,
) -> dict[str, Any]:
    """Construct deterministic, estimator-independent spatial regions on the 1280x720 left image."""
    h, w = luma_l.shape
    luma_f = luma_l.astype(np.float32)

    gx = cv2.Sobel(luma_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)

    interior = np.zeros((h, w), dtype=bool)
    interior[48 : h - 48, 128 : w - 48] = True

    # 1. Fixed boundary bands: +/- 2 px around strong Sobel edges in interior
    edge_thresh = float(np.percentile(grad_mag[interior], 85.0))
    strong_edges = (grad_mag >= edge_thresh) & interior
    kernel = np.ones((5, 5), dtype=np.uint8)
    boundary_mask = (cv2.dilate(strong_edges.astype(np.uint8), kernel) > 0) & interior

    # Tile-level classification on a fixed 32x32 grid inside [64..672, 160..1216]
    tile = 32
    planar_tiles: list[tuple[int, int, int, int]] = []
    low_texture_mask = np.zeros((h, w), dtype=bool)
    repetitive_mask = np.zeros((h, w), dtype=bool)
    planar_candidates: list[tuple[float, int, int]] = []

    gxx = np.abs(cv2.Laplacian(luma_f, cv2.CV_32F, ksize=3))

    for y0 in range(64, h - 64 - tile + 1, tile):
        for x0 in range(160, w - 64 - tile + 1, tile):
            patch_luma = luma_f[y0 : y0 + tile, x0 : x0 + tile]
            patch_grad = grad_mag[y0 : y0 + tile, x0 : x0 + tile]
            patch_lap = gxx[y0 : y0 + tile, x0 : x0 + tile]
            std_l = float(np.std(patch_luma))
            p90_g = float(np.percentile(patch_grad, 90.0))
            mean_g = float(np.mean(patch_grad)) + 1e-3
            osc_ratio = float(np.mean(patch_lap)) / mean_g

            # Low-texture tile: smooth intensity variation without strong edges
            if 2.0 <= std_l <= 9.5 and p90_g < edge_thresh * 0.85:
                low_texture_mask[y0 : y0 + tile, x0 : x0 + tile] = True

            # Repetitive / high-frequency texture tile: high Laplacian-to-gradient ratio & moderate std
            if std_l >= 8.0 and osc_ratio >= 0.72:
                repetitive_mask[y0 : y0 + tile, x0 : x0 + tile] = True

            # Candidate for fixed planar tile: smooth gradient + high consensus reference support + planar consensus
            ref_v = consensus_ref_valid[y0 : y0 + tile, x0 : x0 + tile]
            if float(ref_v.mean()) >= 0.25 and p90_g < edge_thresh * 1.35:
                ref_d = consensus_ref_disp[y0 : y0 + tile, x0 : x0 + tile]
                yy, xx = np.nonzero(ref_v)
                vals = ref_d[yy, xx].astype(np.float64)
                A = np.column_stack([xx.astype(np.float64), yy.astype(np.float64), np.ones_like(xx, dtype=np.float64)])
                coef, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
                rmse = float(np.sqrt(np.mean((A @ coef - vals) ** 2)))
                if rmse <= 1.35:
                    planar_candidates.append((rmse, y0, x0))

    planar_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    for _, y0, x0 in planar_candidates[:20]:
        planar_tiles.append((int(y0), int(x0), tile, tile))

    planar_mask = np.zeros((h, w), dtype=bool)
    for y0, x0, th, tw in planar_tiles:
        planar_mask[y0 : y0 + th, x0 : x0 + tw] = True

    return {
        "interior_mask": interior,
        "planar_mask": planar_mask,
        "planar_tiles": planar_tiles,
        "boundary_mask": boundary_mask,
        "low_texture_mask": low_texture_mask,
        "repetitive_mask": repetitive_mask,
    }


def evaluate_fixed_planar_tiles(
    disp: np.ndarray,
    valid: np.ndarray,
    planar_tiles: list[tuple[int, int, int, int]],
    ref_disp: np.ndarray,
    ref_valid: np.ndarray,
) -> dict[str, Any]:
    """Evaluate planar SVD fit RMSE and reference error on the EXACT SAME fixed planar tiles."""
    fit_rmses: list[float] = []
    ref_rmses: list[float] = []
    coverages: list[float] = []
    for y0, x0, th, tw in planar_tiles:
        v = valid[y0 : y0 + th, x0 : x0 + tw] & np.isfinite(disp[y0 : y0 + th, x0 : x0 + tw])
        cov = float(v.mean())
        coverages.append(cov)
        if int(v.sum()) >= 32:
            d_patch = disp[y0 : y0 + th, x0 : x0 + tw]
            yy, xx = np.nonzero(v)
            vals = d_patch[yy, xx].astype(np.float64)
            A = np.column_stack([xx.astype(np.float64), yy.astype(np.float64), np.ones_like(xx, dtype=np.float64)])
            coef, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
            fit_rmses.append(float(np.sqrt(np.mean((A @ coef - vals) ** 2))))
        rv = v & ref_valid[y0 : y0 + th, x0 : x0 + tw]
        if int(rv.sum()) >= 16:
            diff = disp[y0 : y0 + th, x0 : x0 + tw][rv] - ref_disp[y0 : y0 + th, x0 : x0 + tw][rv]
            ref_rmses.append(float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))))

    return {
        "fixed_tiles_total": len(planar_tiles),
        "fixed_tiles_with_min_32px_support": len(fit_rmses),
        "mean_tile_coverage_pct": round(100.0 * float(np.mean(coverages)), 2) if coverages else 0.0,
        "median_planar_svd_fit_rmse_px": round(float(np.median(fit_rmses)), 3) if fit_rmses else None,
        "mean_planar_svd_fit_rmse_px": round(float(np.mean(fit_rmses)), 3) if fit_rmses else None,
        "median_vs_consensus_ref_rmse_px": round(float(np.median(ref_rmses)), 3) if ref_rmses else None,
    }


def evaluate_region_accuracy(
    disp: np.ndarray,
    valid: np.ndarray,
    region_mask: np.ndarray,
    ref_disp: np.ndarray,
    ref_valid: np.ndarray,
    fxb: float,
    cx_offset: float,
) -> dict[str, Any]:
    """Compute coverage, disparity accuracy, depth accuracy, bad-match rates, and completeness within a fixed spatial mask."""
    r_pixels = int(region_mask.sum())
    if r_pixels == 0:
        return {"region_pixels": 0}
    cand_in_r = valid & region_mask & np.isfinite(disp) & (disp > 0)
    ref_in_r = ref_valid & region_mask & np.isfinite(ref_disp) & (ref_disp > 0)
    joint = cand_in_r & ref_in_r

    cov_pct = round(100.0 * float(cand_in_r.sum()) / r_pixels, 2)
    ref_count = int(ref_in_r.sum())
    joint_count = int(joint.sum())
    if joint_count == 0:
        return {
            "region_pixels": r_pixels,
            "coverage_pct": cov_pct,
            "joint_ref_pixels": 0,
        }

    d_cand = disp[joint].astype(np.float64)
    d_ref = ref_disp[joint].astype(np.float64)
    abs_err = np.abs(d_cand - d_ref)

    z_cand_mm = 1000.0 * fxb / np.maximum(d_cand - cx_offset, 0.5)
    z_ref_mm = 1000.0 * fxb / np.maximum(d_ref - cx_offset, 0.5)
    z_err_mm = np.abs(z_cand_mm - z_ref_mm)

    return {
        "region_pixels": r_pixels,
        "coverage_pct": cov_pct,
        "joint_ref_pixels": joint_count,
        "disparity_mae_px": round(float(np.mean(abs_err)), 3),
        "disparity_rmse_px": round(float(np.sqrt(np.mean(abs_err ** 2))), 3),
        "depth_mae_mm": round(float(np.mean(z_err_mm)), 1),
        "depth_rmse_mm": round(float(np.sqrt(np.mean(z_err_mm ** 2))), 1),
        "bad_1px_pct_of_valid": round(100.0 * float((abs_err > 1.0).mean()), 2),
        "bad_2px_pct_of_valid": round(100.0 * float((abs_err > 2.0).mean()), 2),
        "bad_4px_pct_of_valid": round(100.0 * float((abs_err > 4.0).mean()), 2),
        "completeness_good_1px_of_ref_pct": round(100.0 * float((abs_err <= 1.0).sum()) / max(ref_count, 1), 2),
        "completeness_good_2px_of_ref_pct": round(100.0 * float((abs_err <= 2.0).sum()) / max(ref_count, 1), 2),
    }


def evaluate_pipeline_on_identical_support(
    name: str,
    disp: np.ndarray,
    valid: np.ndarray,
    conf: np.ndarray,
    luma_l: np.ndarray,
    regions: dict[str, Any],
    ref_full_ind_disp: np.ndarray,
    ref_full_ind_val: np.ndarray,
    ref_ocv_disp: np.ndarray,
    ref_ocv_val: np.ndarray,
    ref_consensus_disp: np.ndarray,
    ref_consensus_val: np.ndarray,
    fxb: float,
    cx_offset: float,
    fps_p50: float | None = None,
    host_wall_p50_ms: float | None = None,
    host_wall_p95_ms: float | None = None,
) -> dict[str, Any]:
    geom = geometric_qualification_metrics(disp, valid, luma_l)
    fixed_planar = evaluate_fixed_planar_tiles(
        disp, valid, regions["planar_tiles"], ref_consensus_disp, ref_consensus_val
    )
    region_metrics = {}
    for rname, rmask in (
        ("all_interior", regions["interior_mask"]),
        ("fixed_planar_20tiles", regions["planar_mask"]),
        ("fixed_boundary_bands", regions["boundary_mask"]),
        ("fixed_low_texture_tiles", regions["low_texture_mask"]),
        ("fixed_repetitive_texture_tiles", regions["repetitive_mask"]),
    ):
        region_metrics[rname] = evaluate_region_accuracy(
            disp, valid, rmask, ref_consensus_disp, ref_consensus_val, fxb, cx_offset
        )

    vs_full_ind = evaluate_region_accuracy(
        disp, valid, regions["interior_mask"], ref_full_ind_disp, ref_full_ind_val, fxb, cx_offset
    )
    vs_opencv = evaluate_region_accuracy(
        disp, valid, regions["interior_mask"], ref_ocv_disp, ref_ocv_val, fxb, cx_offset
    )

    return {
        "pipeline": name,
        "fps_p50": fps_p50,
        "host_wall_p50_ms": host_wall_p50_ms,
        "host_wall_p95_ms": host_wall_p95_ms,
        "overall_lr_valid_pct": round(100.0 * float(valid.mean()), 2),
        "coherent_coverage_pct": round(100.0 * float(geom["coherent_surface_fraction_all_pixels"]), 2),
        "speckle_survival_pct": round(100.0 * float(geom["speckle_survival_rate_of_lr_accepted"]), 2),
        "legacy_variable_patch_planar_rmse_px": round(float(geom["planar_disparity_rmse_median_px"]), 3),
        "legacy_variable_patches_count": int(geom["planar_patches_evaluated"]),
        "fixed_planar_20tiles": fixed_planar,
        "regions_vs_consensus_ref": region_metrics,
        "interior_vs_full_res_independent": vs_full_ind,
        "interior_vs_opencv_sgbm": vs_opencv,
    }


def render_diagnostic_panels(
    luma_l: np.ndarray,
    regions: dict[str, Any],
    pipelines: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, float | None]],
    ref_consensus_disp: np.ndarray,
    ref_consensus_val: np.ndarray,
    fxb: float,
    cx_offset: float,
    out_panels_path: Path,
    out_regions_path: Path,
) -> None:
    """Render synchronized multi-panel visual audit PNGs."""
    h, w = luma_l.shape
    gray_u8 = np.clip(luma_l, 0, 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    # 1. Render Fixed Spatial Support Overlay
    overlay = base_bgr.copy()
    overlay[regions["low_texture_mask"]] = (
        0.55 * overlay[regions["low_texture_mask"]] + 0.45 * np.array([255, 180, 0])
    ).astype(np.uint8)
    overlay[regions["repetitive_mask"]] = (
        0.55 * overlay[regions["repetitive_mask"]] + 0.45 * np.array([220, 0, 220])
    ).astype(np.uint8)
    overlay[regions["boundary_mask"]] = (
        0.45 * overlay[regions["boundary_mask"]] + 0.55 * np.array([0, 215, 255])
    ).astype(np.uint8)
    for y0, x0, th, tw in regions["planar_tiles"]:
        cv2.rectangle(overlay, (x0, y0), (x0 + tw - 1, y0 + th - 1), (0, 255, 0), 2)

    cv2.putText(
        overlay,
        "Fixed Spatial Support Regions: Green=20 Planar Tiles | Yellow=Sobel Boundaries | Cyan=Low-Texture | Magenta=Repetitive",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    out_regions_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_regions_path), overlay)

    # 2. Render Synchronized Multi-Pipeline Audit Grid (downscaled 2x per panel: 640x360)
    pw, ph = 480, 270
    rows_img = []
    for title, disp, val, conf, fps in pipelines:
        valid_mask = val & np.isfinite(disp) & (disp > 0)
        # Col 1: Disparity (0..96 px -> turbo)
        d_norm = np.clip(np.where(valid_mask, disp, 0.0) / 96.0 * 255.0, 0, 255).astype(np.uint8)
        d_col = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
        d_col[~valid_mask] = (18, 18, 24)

        # Col 2: Depth (0.5m..6.0m -> viridis)
        z_m = np.where(valid_mask, fxb / np.maximum(disp - cx_offset, 0.5), 0.0)
        z_norm = np.clip((6.0 - np.clip(z_m, 0.5, 6.0)) / 5.5 * 255.0, 0, 255).astype(np.uint8)
        z_col = cv2.applyColorMap(z_norm, cv2.COLORMAP_VIRIDIS)
        z_col[~valid_mask] = (18, 18, 24)

        # Col 3: Disparity Error vs Consensus Reference (0..4 px)
        joint = valid_mask & ref_consensus_val
        err = np.where(joint, np.abs(disp - ref_consensus_disp), 0.0)
        err_col = np.full((h, w, 3), (22, 22, 28), dtype=np.uint8)
        err_col[valid_mask & ~ref_consensus_val] = (90, 90, 90)
        err_col[joint & (err <= 1.0)] = (40, 210, 60)      # <= 1 px: green
        err_col[joint & (err > 1.0) & (err <= 2.0)] = (0, 210, 240)  # 1..2 px: yellow
        err_col[joint & (err > 2.0) & (err <= 4.0)] = (0, 128, 255)  # 2..4 px: orange
        err_col[joint & (err > 4.0)] = (30, 30, 245)      # > 4 px: red

        # Col 4: Confidence (0..1 -> plasma)
        c_norm = np.clip(np.where(valid_mask, conf, 0.0) * 255.0, 0, 255).astype(np.uint8)
        c_col = cv2.applyColorMap(c_norm, cv2.COLORMAP_PLASMA)
        c_col[~valid_mask] = (18, 18, 24)

        c1 = cv2.resize(d_col, (pw, ph), interpolation=cv2.INTER_NEAREST)
        c2 = cv2.resize(z_col, (pw, ph), interpolation=cv2.INTER_NEAREST)
        c3 = cv2.resize(err_col, (pw, ph), interpolation=cv2.INTER_NEAREST)
        c4 = cv2.resize(c_col, (pw, ph), interpolation=cv2.INTER_NEAREST)

        cov = 100.0 * float(valid_mask.mean())
        bad1 = 100.0 * float((err[joint] > 1.0).mean()) if int(joint.sum()) > 0 else 0.0
        fps_str = f"{fps:.1f} FPS" if fps else "CPU ref"
        cv2.putText(c1, f"{title} ({fps_str})", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(c1, f"Disparity | Cov: {cov:.1f}%", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 255, 220), 1, cv2.LINE_AA)
        cv2.putText(c2, "Metric Depth (0.5m - 6.0m)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(c3, f"|d - d_ref| (Bad>1px: {bad1:.1f}%)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(c4, "Confidence Margin", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        rows_img.append(np.hstack([c1, c2, c3, c4]))

    grid = np.vstack(rows_img)
    cv2.imwrite(str(out_panels_path), grid)


def load_depth_planes(prefix: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    report = json.loads(Path(str(prefix) + ".json").read_text())
    h, w = int(report["height"]), int(report["width"])
    disp = np.fromfile(str(prefix) + ".lr_disparity.f32", dtype="<f4").reshape(2, h, w)[0]
    val = np.fromfile(str(prefix) + ".lr_validity.u32", dtype="<u4").reshape(2, h, w)[0] != 0
    conf = np.fromfile(str(prefix) + ".lr_confidence.f32", dtype="<f4").reshape(2, h, w)[0]
    return disp, val, conf, report


def load_refinement_planes(prefix: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    report = json.loads(Path(str(prefix) + ".json").read_text())
    h, w = int(report["height"]), int(report["width"])
    disp = np.fromfile(str(prefix) + ".disparity.f32", dtype="<f4").reshape(2, h, w)[0]
    val = np.fromfile(str(prefix) + ".validity.u32", dtype="<u4").reshape(2, h, w)[0] != 0
    conf = np.fromfile(str(prefix) + ".confidence.f32", dtype="<f4").reshape(2, h, w)[0]
    return disp, val, conf, report


def main() -> None:
    luma_l, luma_r = read_fixture_luma(DEFAULT_MANIFEST.parent)
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    fx = float(manifest.get("fx", 527.790283203125))
    baseline_m = float(manifest.get("baseline_m", 0.11944399774074554))
    cx_offset = float(manifest.get("principal_point_offset_px", 0.0))
    fxb = fx * baseline_m

    step8_dir = ROOT / "artifacts/depth-video/v9-milestone/step8-dumps"
    step9_dir = ROOT / "artifacts/depth-video/v9-milestone/step9-dumps"
    v8_step6_json = json.loads(
        (ROOT / "docs/evidence/v8-review-milestone/step6-multiresolution-refinement-240fps-v8.json").read_text()
    )["runs"]
    v8_step5_json = json.loads(
        (ROOT / "docs/evidence/v8-review-milestone/step5-native-ortho-hybrid-depth-benchmark.json").read_text()
    )["runs"]

    # Load Full-Res Independent & OpenCV SGBM to build Consensus Reference
    d_full_ind, v_full_ind, c_full_ind, _ = load_depth_planes(step8_dir / "depth_ortho-hybrid_layered")
    dl_ocv, dr_ocv, _ = cv_sgbm(luma_l, luma_r, disparities=128, block=5, uniqueness=10)
    lv_ocv = valid_map(dl_ocv, disparities=128)
    rv_ocv = valid_map(dr_ocv, disparities=128)
    lr_ok_ocv, _, _ = lr_pair(dl_ocv, dr_ocv, lv_ocv, rv_ocv, tol=1.0)
    d_ocv = np.where(lr_ok_ocv, dl_ocv, np.nan).astype(np.float32)
    v_ocv = lr_ok_ocv.astype(bool)
    c_ocv = np.where(v_ocv, 0.75, 0.0).astype(np.float32)

    # Consensus Reference: pixels where BOTH Independent Full-Res Ortho-Hybrid SGM and OpenCV SGBM
    # pass independent LR check AND agree within <= 1.0 px
    v_consensus = v_full_ind & v_ocv & (np.abs(d_full_ind - d_ocv) <= 1.0) & (d_full_ind > 0)
    d_consensus = np.where(v_consensus, 0.5 * (d_full_ind + d_ocv), np.nan).astype(np.float32)

    regions = build_fixed_spatial_regions(luma_l, v_consensus, d_consensus)

    d_census, v_census, c_census, _ = load_depth_planes(step8_dir / "depth_census_sequential")
    d_full_der, v_full_der, c_full_der, _ = load_depth_planes(step8_dir / "depth_ortho-hybrid_derived")
    d_ref2x_der, v_ref2x_der, c_ref2x_der, _ = load_refinement_planes(step8_dir / "ref2x_ortho-hybrid_derived_fast1")
    d_ref2x_rep, v_ref2x_rep, c_ref2x_rep, r_ref2x_rep = load_refinement_planes(
        step8_dir / "ref2x_ortho-hybrid_derived_fast1_repair1"
    )
    d_ref4x_ind, v_ref4x_ind, c_ref4x_ind, r_ref4x_ind = load_refinement_planes(
        step9_dir / "ref4x_ortho-hybrid_layered_fast1"
    )
    d_ref4x_ind_rep, v_ref4x_ind_rep, c_ref4x_ind_rep, r_ref4x_ind_rep = load_refinement_planes(
        step9_dir / "ref4x_ortho-hybrid_layered_fast1_repair1"
    )
    d_ref4x_der, v_ref4x_der, c_ref4x_der, r_ref4x_der = load_refinement_planes(
        step9_dir / "ref4x_ortho-hybrid_derived_fast1"
    )
    d_ref4x_der_rep, v_ref4x_der_rep, c_ref4x_der_rep, r_ref4x_der_rep = load_refinement_planes(
        step9_dir / "ref4x_ortho-hybrid_derived_fast1_repair1"
    )

    dual_ind_rep = json.loads((step9_dir / "dual4x_ortho-hybrid_layered_fast1_repair1.json").read_text())
    dual_der_rep = json.loads((step9_dir / "dual4x_ortho-hybrid_derived_fast1_repair1.json").read_text())

    def hw_stats(entry: dict[str, Any]) -> tuple[float, float, float]:
        hw = entry.get("host_wall_ms") or entry.get("timings_ms", {}).get("shared_dual_refinement", {}).get("host_wall_ms", {})
        p50 = float(hw["p50"])
        p95 = float(hw["p95"])
        return round(1000.0 / p50, 1), round(p50, 3), round(p95, 3)

    candidates = [
        ("Legacy Census 24b Full-Res", d_census, v_census, c_census, *hw_stats(v8_step5_json["depth_census_sequential"])),
        ("Full-Res Ortho-Hybrid (Independent)", d_full_ind, v_full_ind, c_full_ind, *hw_stats(v8_step5_json["depth_ortho-hybrid_layered"])),
        ("Full-Res Ortho-Hybrid (Derived Right)", d_full_der, v_full_der, c_full_der, *hw_stats(v8_step5_json["depth_ortho-hybrid_derived"])),
        ("2x Refinement (Derived Right)", d_ref2x_der, v_ref2x_der, c_ref2x_der, *hw_stats(v8_step6_json["dual2x_ortho-hybrid_derived_fast1"])),
        ("2x Selective Repair (Derived Right)", d_ref2x_rep, v_ref2x_rep, c_ref2x_rep, *hw_stats(r_ref2x_rep)),
        ("4x Refinement (Independent)", d_ref4x_ind, v_ref4x_ind, c_ref4x_ind, *hw_stats(v8_step6_json["dual4x_ortho-hybrid_layered_fast1"])),
        ("4x Selective Repair (Independent)", d_ref4x_ind_rep, v_ref4x_ind_rep, c_ref4x_ind_rep, *hw_stats(dual_ind_rep)),
        ("4x Refinement (Derived Right)", d_ref4x_der, v_ref4x_der, c_ref4x_der, *hw_stats(v8_step6_json["dual4x_ortho-hybrid_derived_fast1"])),
        ("4x Selective Repair (Derived Right)", d_ref4x_der_rep, v_ref4x_der_rep, c_ref4x_der_rep, *hw_stats(dual_der_rep)),
        ("OpenCV StereoSGBM (Exact Luma Ref)", d_ocv, v_ocv, c_ocv, None, None, None),
    ]

    evaluated: dict[str, Any] = {}
    panel_list = []
    for label, d_arr, v_arr, c_arr, fps_val, p50_ms, p95_ms in candidates:
        evaluated[label] = evaluate_pipeline_on_identical_support(
            label,
            d_arr,
            v_arr,
            c_arr,
            luma_l,
            regions,
            d_full_ind,
            v_full_ind,
            d_ocv,
            v_ocv,
            d_consensus,
            v_consensus,
            fxb,
            cx_offset,
            fps_p50=fps_val,
            host_wall_p50_ms=p50_ms,
            host_wall_p95_ms=p95_ms,
        )
        panel_list.append((label, d_arr, v_arr, c_arr, fps_val))

    # Compute Step 9 Selective Repair efficiency summary (coherent coverage recovered per ms)
    ind_base_p50 = evaluated["4x Refinement (Independent)"]["host_wall_p50_ms"]
    ind_rep_p50 = evaluated["4x Selective Repair (Independent)"]["host_wall_p50_ms"]
    ind_delta_ms = max(ind_rep_p50 - ind_base_p50, 0.001)
    ind_delta_coh = (
        evaluated["4x Selective Repair (Independent)"]["coherent_coverage_pct"]
        - evaluated["4x Refinement (Independent)"]["coherent_coverage_pct"]
    )
    ind_delta_lr = (
        evaluated["4x Selective Repair (Independent)"]["overall_lr_valid_pct"]
        - evaluated["4x Refinement (Independent)"]["overall_lr_valid_pct"]
    )

    der_base_p50 = evaluated["4x Refinement (Derived Right)"]["host_wall_p50_ms"]
    der_rep_p50 = evaluated["4x Selective Repair (Derived Right)"]["host_wall_p50_ms"]
    der_delta_ms = max(der_rep_p50 - der_base_p50, 0.001)
    der_delta_coh = (
        evaluated["4x Selective Repair (Derived Right)"]["coherent_coverage_pct"]
        - evaluated["4x Refinement (Derived Right)"]["coherent_coverage_pct"]
    )
    der_delta_lr = (
        evaluated["4x Selective Repair (Derived Right)"]["overall_lr_valid_pct"]
        - evaluated["4x Refinement (Derived Right)"]["overall_lr_valid_pct"]
    )

    out_dir = ROOT / "docs/evidence/v9-live-and-audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_png = out_dir / "v9_visual_accuracy_audit_panels.png"
    regions_png = out_dir / "v9_fixed_regions_overlay.png"
    render_diagnostic_panels(
        luma_l, regions, panel_list, d_consensus, v_consensus, fxb, cx_offset, panels_png, regions_png
    )

    summary_doc = {
        "schema": "stereoforge.visual_accuracy_and_selective_repair_v9.v1",
        "consensus_reference_definition": "Pixels where both Full-Res Independent Ortho-Hybrid SGM and OpenCV StereoSGBM pass LR consistency and agree within <= 1.0 px",
        "consensus_reference_coverage_pct": round(100.0 * float(v_consensus.mean()), 2),
        "fixed_regions_summary": {
            "interior_pixels": int(regions["interior_mask"].sum()),
            "fixed_planar_20tiles_pixels": int(regions["planar_mask"].sum()),
            "fixed_planar_20tiles_coords_yxwh": regions["planar_tiles"],
            "fixed_boundary_bands_pixels": int(regions["boundary_mask"].sum()),
            "fixed_low_texture_tiles_pixels": int(regions["low_texture_mask"].sum()),
            "fixed_repetitive_texture_tiles_pixels": int(regions["repetitive_mask"].sum()),
        },
        "selective_repair_efficiency_v9": {
            "independent_4x": {
                "baseline_host_wall_p50_ms": ind_base_p50,
                "selective_repair_host_wall_p50_ms": ind_rep_p50,
                "selective_repair_host_wall_p95_ms": evaluated["4x Selective Repair (Independent)"]["host_wall_p95_ms"],
                "delta_gpu_ms": round(ind_delta_ms, 3),
                "delta_lr_valid_pct": round(ind_delta_lr, 2),
                "delta_coherent_coverage_pct": round(ind_delta_coh, 2),
                "coherent_coverage_pct_per_ms": round(ind_delta_coh / ind_delta_ms, 2),
                "all_30_planes_exact_between_variants": bool(dual_ind_rep.get("all_30_planes_exact_between_variants")),
                "within_240fps_4_167ms_budget_p50_and_p95": ind_rep_p50 <= 4.167
                and evaluated["4x Selective Repair (Independent)"]["host_wall_p95_ms"] <= 4.167,
            },
            "derived_4x": {
                "baseline_host_wall_p50_ms": der_base_p50,
                "selective_repair_host_wall_p50_ms": der_rep_p50,
                "selective_repair_host_wall_p95_ms": evaluated["4x Selective Repair (Derived Right)"]["host_wall_p95_ms"],
                "delta_gpu_ms": round(der_delta_ms, 3),
                "delta_lr_valid_pct": round(der_delta_lr, 2),
                "delta_coherent_coverage_pct": round(der_delta_coh, 2),
                "coherent_coverage_pct_per_ms": round(der_delta_coh / der_delta_ms, 2),
                "all_30_planes_exact_between_variants": bool(dual_der_rep.get("all_30_planes_exact_between_variants")),
                "within_240fps_4_167ms_budget_p50_and_p95": der_rep_p50 <= 4.167
                and evaluated["4x Selective Repair (Derived Right)"]["host_wall_p95_ms"] <= 4.167,
            },
        },
        "pipelines": evaluated,
        "visual_artifacts": {
            "panels_png": str(panels_png.relative_to(ROOT)),
            "regions_overlay_png": str(regions_png.relative_to(ROOT)),
        },
    }

    out_json = out_dir / "step8-9-visual-accuracy-and-selective-repair-v9.json"
    out_json.write_text(json.dumps(summary_doc, indent=2) + "\n")
    print(f"Wrote {out_json}, {panels_png}, and {regions_png}")


if __name__ == "__main__":
    main()
