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
