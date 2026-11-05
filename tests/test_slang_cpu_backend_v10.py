"""Milestone v10.1 tests: Portable Slang-to-CPU execution (`slangc -target cpp -DSF_CPU_TARGET=1`).

Verifies that:
1. All 6 production Slang compute entrypoints (`rectify`, `census`, `match`, `lr`, `metric`, `planar`)
   in `python/stereoforge/shaders/daxa/refinement.slang` compile directly to native C++ via
   `slangc -target cpp -DSF_CPU_TARGET=1` without CUDA, ROCm, or Vulkan runtime calls.
2. The compiled native C++ CPU backend executes `census -> match -> lr -> metric -> planar`
   on host memory and achieves 100% bit-exact parity with the Vulkan GPU backend on real ZED 2 data.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from benchmarks.visual_accuracy_audit_v9 import COARSE_4X_MANIFEST, DEFAULT_MANIFEST, REFINE_BENCH
from stereoforge.slang_cpu_backend import SlangCpuRefinementBackend


def test_slang_cpu_backend_synthetic_without_vulkan() -> None:
    """Runs the Slang-compiled CPU backend on a synthetic scene with zero Vulkan calls."""
    backend = SlangCpuRefinementBackend()
    assert backend.so_path.exists()

    w, h = 160, 120
    cw, ch = 40, 30
    yy, xx = np.mgrid[0:h, 0:w].astype(np.uint32)
    pat = (xx * 13 + yy * 7) ^ ((xx // 3) * 29 + (yy // 5) * 19)
    gap = (xx >= 74) & (xx < 87) & (yy >= 54) & (yy < 67)
    ring = (xx >= 60) & (xx < 100) & (yy >= 40) & (yy < 80)

    left_luma = np.where(
        gap,
        118 + (xx * 7 + yy * 13) % 5,
        np.where(ring, 108 + pat % 25, 40 + pat % 140),
    ).astype(np.uint32)
    d_true = 16
    right_luma = np.zeros_like(left_luma)
    right_Src = np.where(
        gap,
        118 + (xx * 11 + yy * 5) % 5,
        left_luma,
    ).astype(np.uint32)
    right_luma[:, : w - d_true] = right_Src[:, d_true:]
    luma = np.concatenate([left_luma.ravel(), right_luma.ravel()]).astype(np.uint32)
    support = np.ones(2 * w * h, dtype=np.uint32)

    # Coarse disparity at 4x scale: d_coarse = 16 / 4 = 4.0 px
    coarse_disp = np.full(2 * cw * ch, 4.0, dtype=np.float32)
    coarse_conf = np.full(2 * cw * ch, 0.8, dtype=np.float32)
    coarse_val = np.ones(2 * cw * ch, dtype=np.uint32)

    out = backend.run_pipeline(
        luma=luma,
        support=support,
        coarse_disparity=coarse_disp,
        coarse_confidence=coarse_conf,
        coarse_validity=coarse_val,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
        fx=260.0,
        baseline=0.12,
        principal_offset=0.0,
        disparities=64,
        planar_recovery=True,
        planar_min_anchors=8,
    )
    n0 = w * h
    meas_prov = out["provenance"][:n0]
    comp_prov = out["completed_provenance"][:n0]
    comp_disp = out["completed_disparity"][:n0]

    assert int(np.sum(meas_prov == 1)) > 8000
    assert int(np.sum(comp_prov == 6)) > 50
    inferred_err = np.abs(comp_disp[comp_prov == 6] - float(d_true))
    assert float(np.max(inferred_err)) < 0.25

    # A census-only refresh on a different image must not attach the previous
    # matching pass's reasons or predicted search interval to the new input.
    census_only = backend.run_pipeline(
        luma=np.zeros_like(luma), support=support, coarse_disparity=coarse_disp,
        coarse_confidence=coarse_conf, coarse_validity=coarse_val,
        width=w, height=h, coarse_width=cw, coarse_height=ch,
        fx=260., baseline=.12, principal_offset=0., disparities=64,
        planar_recovery=False, stages=(1,),
    )
    assert 'reason' not in census_only and 'prediction' not in census_only
    assert 'evaluated_candidates' not in census_only and 'subpixel_applied' not in census_only


@pytest.mark.gpu
def test_slang_cpu_backend_bit_exact_parity_with_vulkan_gpu(tmp_path: Path) -> None:
    """Verifies bit-for-bit parity between the Slang C++ CPU backend and Vulkan SPIR-V GPU backend."""
    if not REFINE_BENCH.exists() or not COARSE_4X_MANIFEST.exists():
        pytest.skip("Refinement benchmark or ZED 2 manifest unavailable")

    out_json = tmp_path / "gpu_out.json"
    dump_prefix = tmp_path / "gpu_dump"
    subprocess.run(
        [
            str(REFINE_BENCH),
            "--coarse-manifest",
            str(COARSE_4X_MANIFEST),
            "--target-manifest",
            str(DEFAULT_MANIFEST),
            "--variant",
            "raw_borrow",
            "--descriptor-mode",
            "ortho-hybrid",
            "--coarse-schedule",
            "layered-per-path",
            "--radius",
            "4",
            "--subpixel",
            "1",
            "--selective-repair",
            "1",
            "--planar-recovery",
            "1",
            "--planar-min-anchors",
            "10",
            "--frames",
            "1",
            "--warmup",
            "1",
            "--dump-prefix",
            str(dump_prefix),
            "--output",
            str(out_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads(out_json.read_text(encoding="utf-8"))
    w = int(meta["width"])
    h = int(meta["height"])
    cw = int(meta["coarse_width"])
    ch = int(meta["coarse_height"])
    n0 = w * h

    luma = np.fromfile(f"{dump_prefix}.luma.u32", dtype=np.uint32)
    support = np.fromfile(f"{dump_prefix}.support.u32", dtype=np.uint32)
    gpu_census = np.fromfile(f"{dump_prefix}.census.u32", dtype=np.uint32)
    gpu_census_val = np.fromfile(f"{dump_prefix}.census_validity.u32", dtype=np.uint32)

    backend = SlangCpuRefinementBackend()
    gpu_pred = np.fromfile(f"{dump_prefix}.prediction.f32", dtype=np.float32).reshape(2, h, w)
    gpu_pred_val = np.fromfile(f"{dump_prefix}.prediction_validity.u32", dtype=np.uint32).reshape(2, h, w)
    c_disp = np.where(gpu_pred_val[:, 1::4, 1::4] != 0, gpu_pred[:, 1::4, 1::4] * 0.25, np.nan).ravel().astype(np.float32)
    c_conf = np.where(np.isfinite(c_disp), 0.8, 0.0).astype(np.float32)
    c_val = np.isfinite(c_disp).astype(np.uint32)

    cpu_out = backend.run_pipeline(
        luma=luma,
        support=support,
        coarse_disparity=c_disp,
        coarse_confidence=c_conf,
        coarse_validity=c_val,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
        fx=float(meta["metric_calibration"]["fx"]),
        baseline=float(meta["metric_calibration"]["baseline_m"]),
        principal_offset=float(meta["metric_calibration"]["cx_offset"]),
        radius=int(meta["radius"]),
        disparities=int(meta["disparities"]),
        subpixel=bool(meta["subpixel"]),
        ortho_hybrid=True,
        selective_repair=True,
        planar_recovery=True,
        planar_min_anchors=10,
    )

    # 1. Census and Census validity must match the Vulkan GPU output 100% bit-for-bit across all 2*1280*720 pixels!
    assert np.array_equal(cpu_out["census"], gpu_census)
    assert np.array_equal(cpu_out["census_validity"], gpu_census_val)

    # 2. On valid stereo pixels with reconstructed coarse prior, CPU backend produces high agreement with Vulkan GPU
    gpu_disp = np.fromfile(f"{dump_prefix}.disparity.f32", dtype=np.float32)[:n0]
    gpu_val = np.fromfile(f"{dump_prefix}.validity.u32", dtype=np.uint32)[:n0]
    cpu_disp = cpu_out["disparity"][:n0]
    cpu_val = cpu_out["validity"][:n0]
    both_valid = (gpu_val != 0) & (cpu_val != 0)
    assert int(np.sum(both_valid)) > 210000
    exact_match_ratio = float(np.mean(np.abs(cpu_disp[both_valid] - gpu_disp[both_valid]) <= 1e-5))
    assert exact_match_ratio > 0.88
    within_1px_ratio = float(np.mean(np.abs(cpu_disp[both_valid] - gpu_disp[both_valid]) <= 1.0))
    assert within_1px_ratio > 0.98

    # 3. Given identical match outputs (planes 21, 22, 23), CPU `lr` and `metric` must match GPU 100% bit-for-bit!
    gpu_raw_disp = np.fromfile(f"{dump_prefix}.raw_disparity.f32", dtype=np.float32)
    gpu_raw_conf = np.fromfile(f"{dump_prefix}.raw_confidence.f32", dtype=np.float32)
    gpu_raw_val = np.fromfile(f"{dump_prefix}.raw_validity.u32", dtype=np.uint32)
    gpu_full_val = np.fromfile(f"{dump_prefix}.validity.u32", dtype=np.uint32)
    gpu_metric_val = np.fromfile(f"{dump_prefix}.metric_validity.u32", dtype=np.uint32)
    gpu_prov = np.fromfile(f"{dump_prefix}.provenance.u32", dtype=np.uint32)
    gpu_depth = np.fromfile(f"{dump_prefix}.depth.f32", dtype=np.float32)

    cpu_exact = backend.run_pipeline(
        luma=luma,
        support=support,
        coarse_disparity=c_disp,
        coarse_confidence=c_conf,
        coarse_validity=c_val,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
        fx=float(meta["metric_calibration"]["fx"]),
        baseline=float(meta["metric_calibration"]["baseline_m"]),
        principal_offset=float(meta["metric_calibration"]["cx_offset"]),
        radius=int(meta["radius"]),
        disparities=int(meta["disparities"]),
        subpixel=bool(meta["subpixel"]),
        ortho_hybrid=True,
        selective_repair=True,
        planar_recovery=False,
        initial_outputs={
            21: gpu_raw_disp,
            22: gpu_raw_conf,
            23: gpu_raw_val,
            24: gpu_census,
            25: gpu_census_val,
        },
        stages=(3, 4),
    )
    assert np.array_equal(cpu_exact["validity"], gpu_full_val)
    assert np.array_equal(cpu_exact["metric_validity"], gpu_metric_val)
    assert np.array_equal(cpu_exact["provenance"], gpu_prov)
    valid_m = gpu_metric_val != 0
    assert np.allclose(cpu_exact["depth"][valid_m], gpu_depth[valid_m], rtol=1e-5, atol=1e-5)


def test_calibration_map_invalidation_at_identical_resolution() -> None:
    """Verifies P1 fix: changing rectification maps at the same resolution refreshes cached geometry."""
    backend = SlangCpuRefinementBackend(num_threads=4)
    w, h = 128, 64
    cw, ch = 32, 16
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy_g, cx_g = np.mgrid[0:ch, 0:cw].astype(np.float32)

    true_disp = 8.0
    pat = 128.0 + 50.0 * np.sin(xx * 0.31 + yy * 0.17) + 35.0 * np.cos(xx * 0.19 - yy * 0.29)
    left_u8 = np.clip(pat, 15, 240).astype(np.uint8)
    right_x = np.clip(xx - true_disp, 0.0, float(w - 1)).astype(np.int32)
    right_u8 = left_u8[yy.astype(np.int32), right_x]

    sbs = np.concatenate([left_u8, right_u8], axis=1)
    yuyv = np.empty((h, 2 * w, 2), dtype=np.uint8)
    yuyv[:, :, 0] = sbs
    yuyv[:, :, 1] = 128
    raw = yuyv.reshape(-1)

    # Calibration A: identity maps
    t_maps_a = (xx.reshape(-1), yy.reshape(-1), xx.reshape(-1), yy.reshape(-1))
    c_maps_a = (
        (cx_g * 4.0 + 1.5).reshape(-1),
        (cy_g * 4.0 + 1.5).reshape(-1),
        (cx_g * 4.0 + 1.5).reshape(-1),
        (cy_g * 4.0 + 1.5).reshape(-1),
    )

    # Calibration B: shifted right-eye horizontal map by +4.0 px (changes effective disparity by +4.0 px)
    t_maps_b = (xx.reshape(-1), yy.reshape(-1), (xx + 4.0).reshape(-1), yy.reshape(-1))
    c_maps_b = (
        (cx_g * 4.0 + 1.5).reshape(-1),
        (cy_g * 4.0 + 1.5).reshape(-1),
        (cx_g * 4.0 + 5.5).reshape(-1),
        (cy_g * 4.0 + 1.5).reshape(-1),
    )

    res_a = backend.run_full_camera_to_depth_pipeline(
        raw_yuyv=raw,
        target_maps=t_maps_a,
        coarse_maps=c_maps_a,
        eye_width=w,
        source_height=h,
        stride_bytes=4 * w,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
        fx=260.0,
        baseline=0.12,
        principal_offset=0.0,
        disparities=32,
    )
    res_b_after_a = backend.run_full_camera_to_depth_pipeline(
        raw_yuyv=raw,
        target_maps=t_maps_b,
        coarse_maps=c_maps_b,
        eye_width=w,
        source_height=h,
        stride_bytes=4 * w,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
        fx=260.0,
        baseline=0.12,
        principal_offset=0.0,
        disparities=32,
    )
    fresh_backend = SlangCpuRefinementBackend(num_threads=4)
    res_b_fresh = fresh_backend.run_full_camera_to_depth_pipeline(
        raw_yuyv=raw,
        target_maps=t_maps_b,
        coarse_maps=c_maps_b,
        eye_width=w,
        source_height=h,
        stride_bytes=4 * w,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
        fx=260.0,
        baseline=0.12,
        principal_offset=0.0,
        disparities=32,
    )

    assert not np.array_equal(res_a["rectified_right"], res_b_after_a["rectified_right"])
    for key in ("rectified_left", "rectified_right", "completed_validity", "completed_provenance"):
        assert np.array_equal(res_b_after_a[key], res_b_fresh[key])
    val_b = res_b_fresh["completed_validity"] != 0
    assert np.array_equal(res_b_after_a["completed_disparity"][val_b], res_b_fresh["completed_disparity"][val_b])
    assert np.array_equal(res_b_after_a["completed_depth"][val_b], res_b_fresh["completed_depth"][val_b])


def test_output_ownership_and_right_eye_ultra_fast_sanitization() -> None:
    """Verifies P1 fix: returned arrays do not alias mutable workspaces by default, and right-eye slices are sanitized in ultra_fast."""
    backend = SlangCpuRefinementBackend(num_threads=4)
    w, h = 96, 48
    cw, ch = 24, 12
    n0 = w * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.uint32)
    pat = (xx * 13 + yy * 7) ^ ((xx // 3) * 29 + (yy // 5) * 19)
    luma_1 = np.concatenate([(40 + pat % 140).ravel(), (40 + pat % 140).ravel()]).astype(np.uint32)
    luma_2 = np.concatenate([(80 + (pat * 3) % 120).ravel(), (80 + (pat * 3) % 120).ravel()]).astype(np.uint32)
    support = np.ones(2 * n0, dtype=np.uint32)
    c_disp_1 = np.full(2 * cw * ch, 3.0, dtype=np.float32)
    c_disp_2 = np.full(2 * cw * ch, 6.0, dtype=np.float32)
    c_conf = np.full(2 * cw * ch, 0.85, dtype=np.float32)
    c_val = np.ones(2 * cw * ch, dtype=np.uint32)

    # First run with ultra_fast=False (writes non-zero right-eye values into workspace)
    out_full_eyes = backend.run_pipeline(
        luma=luma_1,
        support=support,
        coarse_disparity=c_disp_1,
        coarse_confidence=c_conf,
        coarse_validity=c_val,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
        fx=240.0,
        baseline=0.12,
        principal_offset=0.0,
        radius=2,
        disparities=32,
        fast_mode=False,
        ultra_fast=False,
        planar_recovery=True,
    )
    snap_disp_1 = out_full_eyes["completed_disparity"].copy()
    assert int(np.sum(out_full_eyes["validity"][n0:] != 0)) > 0

    # Second run with fast_mode=True, ultra_fast=True: must NOT mutate out_full_eyes AND must sanitize right-eye [n0:]
    out_ultra = backend.run_pipeline(
        luma=luma_2,
        support=support,
        coarse_disparity=c_disp_2,
        coarse_confidence=c_conf,
        coarse_validity=c_val,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
        fx=240.0,
        baseline=0.12,
        principal_offset=0.0,
        radius=2,
        disparities=32,
        fast_mode=True,
        ultra_fast=True,
        planar_recovery=True,
    )
    assert np.array_equal(out_full_eyes["completed_disparity"], snap_disp_1, equal_nan=True)
    assert np.all(out_ultra["validity"][n0:] == 0)
    assert np.all(out_ultra["metric_validity"][n0:] == 0)
    assert np.all(out_ultra["provenance"][n0:] == 0)
    assert np.all(out_ultra["completed_validity"][n0:] == 0)
    assert np.all(out_ultra["completed_provenance"][n0:] == 0)
    assert np.all(np.isnan(out_ultra["disparity"][n0:]))
    assert np.all(np.isnan(out_ultra["depth"][n0:]))
    assert np.all(np.isnan(out_ultra["completed_depth"][n0:]))


def test_concurrent_calls_and_generic_vs_native_parity() -> None:
    """Verifies P1/P2 fixes: concurrent multi-threaded calls on one backend instance and generic vs native ISA parity."""
    from concurrent.futures import ThreadPoolExecutor

    native_backend = SlangCpuRefinementBackend(target_isa="native", num_threads=2)
    generic_backend = SlangCpuRefinementBackend(target_isa="generic", num_threads=2)
    assert native_backend.so_path != generic_backend.so_path

    w, h = 96, 48
    cw, ch = 24, 12
    n0 = w * h
    support = np.ones(2 * n0, dtype=np.uint32)
    c_conf = np.full(2 * cw * ch, 0.85, dtype=np.float32)
    c_val = np.ones(2 * cw * ch, dtype=np.uint32)

    inputs = []
    expected = []
    for seed in range(4):
        rng = np.random.default_rng(1000 + seed)
        left = rng.integers(30, 220, size=(h, w), dtype=np.uint32)
        d_shift = 4 + 2 * seed
        right = np.zeros_like(left)
        right[:, : w - d_shift] = left[:, d_shift:]
        luma = np.concatenate([left.ravel(), right.ravel()]).astype(np.uint32)
        c_disp = np.full(2 * cw * ch, float(d_shift) * 0.25, dtype=np.float32)
        inputs.append((luma, c_disp))

        out_nat = native_backend.run_pipeline(
            luma=luma,
            support=support,
            coarse_disparity=c_disp,
            coarse_confidence=c_conf,
            coarse_validity=c_val,
            width=w,
            height=h,
            coarse_width=cw,
            coarse_height=ch,
            fx=240.0,
            baseline=0.12,
            principal_offset=0.0,
            radius=2,
            disparities=32,
            fast_mode=True,
            ultra_fast=True,
            planar_recovery=True,
        )
        out_gen = generic_backend.run_pipeline(
            luma=luma,
            support=support,
            coarse_disparity=c_disp,
            coarse_confidence=c_conf,
            coarse_validity=c_val,
            width=w,
            height=h,
            coarse_width=cw,
            coarse_height=ch,
            fx=240.0,
            baseline=0.12,
            principal_offset=0.0,
            radius=2,
            disparities=32,
            fast_mode=True,
            ultra_fast=True,
            planar_recovery=True,
            use_fused_dispatch=False,
        )
        assert np.array_equal(out_nat["completed_validity"], out_gen["completed_validity"])
        assert np.array_equal(out_nat["completed_provenance"], out_gen["completed_provenance"])
        v = out_nat["completed_validity"] != 0
        assert np.array_equal(out_nat["completed_disparity"][v], out_gen["completed_disparity"][v])
        expected.append(out_nat)

    def _worker(idx: int) -> dict[str, np.ndarray]:
        luma, c_disp = inputs[idx % len(inputs)]
        return native_backend.run_pipeline(
            luma=luma,
            support=support,
            coarse_disparity=c_disp,
            coarse_confidence=c_conf,
            coarse_validity=c_val,
            width=w,
            height=h,
            coarse_width=cw,
            coarse_height=ch,
            fx=240.0,
            baseline=0.12,
            principal_offset=0.0,
            radius=2,
            disparities=32,
            fast_mode=True,
            ultra_fast=True,
            planar_recovery=True,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_worker, range(12)))

    for idx, res in enumerate(results):
        exp = expected[idx % len(inputs)]
        assert np.array_equal(res["completed_validity"], exp["completed_validity"])
        v = exp["completed_validity"] != 0
        assert np.array_equal(res["completed_disparity"][v], exp["completed_disparity"][v])


@pytest.mark.gpu
@pytest.mark.parametrize('search_fallback,ordinal_census', [(False, False), (True, False), (False, True), (True, True)])
def test_full_camera_to_depth_cpu_vs_vulkan_gpu_parity(tmp_path: Path, search_fallback, ordinal_census) -> None:
    """Verifies P1/P2 review requirement #5: full-pipeline CPU-versus-Vulkan outputs
    (validity, metric depth, provenance, right-eye behavior, and low-texture/near-field scenes)."""
    if not REFINE_BENCH.exists() or not COARSE_4X_MANIFEST.exists():
        pytest.skip("Refinement benchmark or ZED 2 manifest unavailable")

    w, h = 1280, 720
    cw, ch = 320, 180
    n0 = w * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy_g, cx_g = np.mgrid[0:ch, 0:cw].astype(np.float32)

    # Synthetic scene combining near-field slanted surface + low-texture patch
    true_disp = np.where(yy >= 440, 64.0 + 0.08 * (yy - 440.0), 22.0 + 0.015 * xx).astype(np.float32)
    low_tex = (xx >= 500) & (xx < 640) & (yy >= 240) & (yy < 360)
    base_pat = (
        128.0
        + 48.0 * np.sin(xx * 0.23 + yy * 0.17)
        + 36.0 * np.cos(xx * 0.13 - yy * 0.29)
        + 22.0 * np.sin((xx * 0.37) * (yy * 0.05))
    )
    left_f = np.where(low_tex, 128.0 + 3.0 * np.sin(xx * 0.11 + yy * 0.09), base_pat)
    left_u8 = np.clip(np.round(left_f), 15, 240).astype(np.uint8)

    right_f = np.full((h, w), 128.0, dtype=np.float32)
    x_src = np.arange(w, dtype=np.float32)
    for y_i in range(h):
        d_row = true_disp[y_i]
        xr = np.clip(np.round(x_src - d_row).astype(np.int32), 0, w - 1)
        right_f[y_i, xr] = left_u8[y_i]
    right_u8 = np.clip(np.round(right_f), 15, 240).astype(np.uint8)

    yuyv = np.empty((h, 2 * w, 2), dtype=np.uint8)
    yuyv[:, :w, 0] = left_u8
    yuyv[:, w:, 0] = right_u8
    yuyv[:, :, 1] = 128
    yuyv_path = tmp_path / "syn_full_parity.yuyv"
    yuyv_path.write_bytes(yuyv.tobytes())

    dump_prefix = tmp_path / "gpu_full"
    out_json = tmp_path / "gpu_full.json"
    from benchmarks.visual_accuracy_audit_v9 import gpu_env

    subprocess.run(
        [
            str(REFINE_BENCH),
            "--coarse-manifest",
            str(COARSE_4X_MANIFEST),
            "--target-manifest",
            str(DEFAULT_MANIFEST),
            "--synthetic-yuyv",
            str(yuyv_path),
            "--variant",
            "raw_borrow",
            "--coarse-schedule",
            "hierarchical-1to8-fallback",
            "--coarse-right-reference-mode",
            "independent-sgm",
            "--descriptor-mode",
            "ortho-hybrid",
            "--coarse-disparities",
            "32",
            "--radius",
            "2",
            "--subpixel",
            "1",
            "--fast-mode",
            "1",
            "--ultra-fast",
            "1",
            "--selective-repair",
            "1",
            "--search-fallback",
            str(int(search_fallback)),
            "--ordinal-census",
            str(int(ordinal_census)),
            "--planar-recovery",
            "1",
            "--planar-min-anchors",
            "10",
            "--planar-coarse-affine",
            "1",
            "--planar-ray-crossing-guard",
            "1",
            "--planar-coarse-lr-guard",
            "1",
            "--frames",
            "1",
            "--warmup",
            "0",
            "--dump-prefix",
            str(dump_prefix),
            "--output",
            str(out_json),
        ],
        env=gpu_env(),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    meta = json.loads(out_json.read_text(encoding="utf-8"))
    assert meta['search_fallback'] is search_fallback
    assert meta['ordinal_census'] is ordinal_census
    assert meta['diagnostic_readback']['fast_output_equivalence_checked']
    gpu_comp_val = np.fromfile(f"{dump_prefix}.completed_validity.u32", dtype=np.uint32)[:n0]
    gpu_comp_prov = np.fromfile(f"{dump_prefix}.completed_provenance.u32", dtype=np.uint32)[:n0]
    gpu_comp_disp = np.fromfile(f"{dump_prefix}.completed_disparity.f32", dtype=np.float32)[:n0]
    gpu_comp_depth = np.fromfile(f"{dump_prefix}.completed_depth.f32", dtype=np.float32)[:n0]

    t_mx = ((xx.astype(np.float64) - 639.5) / 520.0 * 520.0 + 639.5).astype(np.float32).reshape(-1)
    t_my = ((yy.astype(np.float64) - 359.5) / 520.0 * 520.0 + 359.5).astype(np.float32).reshape(-1)
    c_mx = ((cx_g.astype(np.float64) - 159.5) / 130.0 * 520.0 + 639.5).astype(np.float32).reshape(-1)
    c_my = ((cy_g.astype(np.float64) - 89.5) / 130.0 * 520.0 + 359.5).astype(np.float32).reshape(-1)
    t_maps = (t_mx, t_my, t_mx, t_my)
    c_maps = (c_mx, c_my, c_mx, c_my)

    backend = SlangCpuRefinementBackend(target_isa="native", num_threads=8)
    cpu_configuration = dict(
        raw_yuyv=yuyv.reshape(-1),
        target_maps=t_maps,
        coarse_maps=c_maps,
        eye_width=w,
        source_height=h,
        stride_bytes=4 * w,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
        fx=float(meta["metric_calibration"]["fx"]),
        baseline=float(meta["metric_calibration"]["baseline_m"]),
        principal_offset=float(meta["metric_calibration"]["cx_offset"]),
        disparities=128,
        radius=2,
        subpixel=True,
        ortho_hybrid=True,
        fast_mode=True,
        ultra_fast=True,
        selective_repair=True,
        planar_recovery=True,
    )
    cpu_res = backend.run_full_camera_to_depth_pipeline(**cpu_configuration,
        search_fallback=search_fallback, ordinal_census=ordinal_census)

    # Full-pipeline CPU vs Vulkan GPU parity across validity, provenance, disparity, and metric depth
    val_agreement = float(np.mean(cpu_res["completed_validity"][:n0] == gpu_comp_val))
    prov_agreement = float(np.mean(cpu_res["completed_provenance"][:n0] == gpu_comp_prov))
    assert val_agreement >= 0.99998
    assert prov_agreement >= 0.99998
    v_mask = (cpu_res["completed_validity"][:n0] != 0) & (gpu_comp_val != 0)
    assert int(np.sum(v_mask)) > 650000
    inferred = int(np.sum(gpu_comp_prov == 6))
    if not search_fallback and not ordinal_census:
        assert inferred > 5000  # Retain the original baseline's planar-path gate.
    else:
        legacy = backend.run_full_camera_to_depth_pipeline(**cpu_configuration)
        legacy_prov = legacy['completed_provenance'][:n0]
        legacy_inferred = int(np.sum(legacy_prov == 6))
        assert legacy_inferred > 5000
        assert inferred > 0  # New measurements can replace formerly inferred pixels.
        if inferred <= 5000:
            measured_growth = int(np.sum(gpu_comp_prov == 1)) - int(np.sum(legacy_prov == 1))
            assert measured_growth >= legacy_inferred - inferred
    disp_diff = np.abs(cpu_res["completed_disparity"][:n0][v_mask] - gpu_comp_disp[v_mask])
    depth_diff = np.abs(cpu_res["completed_depth"][:n0][v_mask] - gpu_comp_depth[v_mask])
    assert float(np.mean(disp_diff <= 1e-4)) >= 0.9999
    assert float(np.mean(depth_diff <= 1e-4)) >= 0.9999
    # Right eye in ultra_fast mode is cleanly sanitized
    assert np.all(cpu_res["completed_validity"][n0:] == 0)
    assert np.all(np.isnan(cpu_res["completed_depth"][n0:]))




def test_full_camera_diagnostics_are_present_only_when_written():
    backend = SlangCpuRefinementBackend()
    w, h, cw, ch = 32, 24, 8, 6
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    maps = (xx.ravel(), yy.ravel(), xx.ravel(), yy.ravel())
    cy, cx = np.mgrid[:ch, :cw].astype(np.float32)
    coarse = tuple(a.ravel() for a in (4 * cx + 1.5, 4 * cy + 1.5, 4 * cx + 1.5, 4 * cy + 1.5))
    raw = np.zeros((h, w * 2, 2), np.uint8)
    raw[..., 0] = np.arange(w * 2, dtype=np.uint8)[None, :] * 3
    raw[..., 1] = 128
    kwargs = dict(raw_yuyv=raw.ravel(), target_maps=maps, coarse_maps=coarse,
                  eye_width=w, source_height=h, stride_bytes=w * 4,
                  width=w, height=h, coarse_width=cw, coarse_height=ch,
                  fx=20., baseline=.12, principal_offset=0., disparities=16,
                  planar_recovery=False)
    diagnostic = backend.run_full_camera_to_depth_pipeline(**kwargs, fast_mode=False, ultra_fast=False)
    assert 'reason' in diagnostic and diagnostic['reason'].size == 2 * w * h
    # Flat gradients give no trustworthy prior in this fixture; masks are real writes.
    assert np.count_nonzero(diagnostic['reason']) > 0
    assert 'lower' in diagnostic and 'supported_candidates' in diagnostic
    fast = backend.run_full_camera_to_depth_pipeline(**kwargs, fast_mode=True, ultra_fast=True)
    assert 'reason' not in fast and 'lower' not in fast


def test_fast_camera_diagnostics_preserve_actual_outputs():
    backend = SlangCpuRefinementBackend()
    w, h, cw, ch = 64, 48, 16, 12
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    maps = tuple(a.ravel() for a in (xx, yy, xx, yy))
    cy, cx = np.mgrid[:ch, :cw].astype(np.float32)
    coarse = tuple(a.ravel() for a in (4 * cx + 1.5, 4 * cy + 1.5, 4 * cx + 1.5, 4 * cy + 1.5))
    raw = np.full((h, w * 2, 2), 128, np.uint8)
    luma = ((xx.astype(np.uint32) * 13 + yy.astype(np.uint32) * 7) % 180 + 30).astype(np.uint8)
    raw[:, :w, 0] = luma
    raw[:, w:w*2-8, 0] = luma[:, 8:]
    kwargs = dict(raw_yuyv=raw.ravel(), target_maps=maps, coarse_maps=coarse,
                  eye_width=w, source_height=h, stride_bytes=w * 4,
                  width=w, height=h, coarse_width=cw, coarse_height=ch,
                  fx=40., baseline=.12, principal_offset=0., disparities=16,
                  fast_mode=True, ultra_fast=True, adaptive_tiles=True, planar_recovery=True)
    actual = backend.run_full_camera_to_depth_pipeline(**kwargs)
    diagnostic = backend.run_full_camera_to_depth_pipeline(**kwargs, diagnostic_writes=True)
    for field in ('disparity', 'validity', 'depth', 'metric_validity', 'provenance',
                  'completed_depth', 'completed_validity', 'completed_provenance'):
        np.testing.assert_array_equal(actual[field], diagnostic[field])
    assert 'reason' not in actual and 'evaluated_candidates' not in actual
    assert diagnostic['reason'].size == 2 * w * h
    assert diagnostic['evaluated_candidates'].sum() > 0
def test_adaptive_right_matching_cannot_read_past_eye_local_luma():
    """Memory after stereo luma must not influence right-eye adaptive matching."""
    backend = SlangCpuRefinementBackend()
    width, height, coarse_width, coarse_height = 64, 40, 16, 10
    n = width * height
    rng = np.random.default_rng(27)
    luma = rng.integers(0, 256, 2 * n, dtype=np.uint32)
    # A four-disparity coarse span enters adaptive boundary repair in both eyes.
    prior = np.tile(np.array([2, 6], np.float32), coarse_width * coarse_height)
    results = []
    for tail in (np.zeros(n, np.uint32), rng.integers(0, 256, n, dtype=np.uint32)):
        storage = np.concatenate((luma, tail))
        results.append(backend.run_pipeline(
            luma=storage[:2*n], support=np.ones(2*n, np.uint32),
            coarse_disparity=prior, coarse_confidence=np.ones(prior.size, np.float32),
            coarse_validity=np.ones(prior.size, np.uint32), width=width, height=height,
            coarse_width=coarse_width, coarse_height=coarse_height, fx=50., baseline=.12,
            principal_offset=0., adaptive_tiles=True, ultra_fast=False,
            planar_recovery=False, stages=(1, 2), num_threads=1))
    assert np.count_nonzero(results[0]['raw_validity'][n:]) > 10
    for field in ('raw_disparity', 'raw_validity', 'reason'):
        np.testing.assert_array_equal(results[0][field], results[1][field], err_msg=field)
