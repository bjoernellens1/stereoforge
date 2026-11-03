"""Pure-CPU safeguards for 1,000-FPS qualification evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.fps1000_gate_v11 import (
    MIN_INDEPENDENT_NEAR_FIELD_OVERLAP,
    STAGE_BUDGET_MS,
    assess_near_field,
    assess_profile,
    assess_v10,
)

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "benchmarks/results/v10_ultra_fast_and_universal_qualification.json"
V11 = ROOT / "benchmarks/results/v11_0_qualification_and_coarse_replacement.json"


def _profile() -> dict:
    return {
        "gpu_combined_p50_ms": 2.5544,
        "gpu_combined_p95_ms": 2.5708,
        "host_wall_p95_ms": 2.8239,
        "measured_valid_pixels": 294942,
        "completed_valid_pixels": 294942,
        "frames": 180,
        "warmup": 20,
        "stage_timings_ms": {name: budget * 3 for name, budget in STAGE_BUDGET_MS.items()},
    }


def test_timing_is_not_sustained_throughput() -> None:
    result = assess_profile(_profile())
    assert result["timing_verdict"] == "FAIL"
    assert result["fps_from_gpu_p50_only"] == 391.5
    assert result["measured_coverage_pct"] == 32.0
    assert result["stage_snapshot_vs_allocations"]["coarse_total_ms"]["needed_speedup"] == 3.0
    assert "repeated" in result["input_limitation"]


def test_hypothetical_under_one_ms_is_still_unqualified() -> None:
    profile = _profile()
    profile.update(gpu_combined_p50_ms=0.7, gpu_combined_p95_ms=0.9, host_wall_p95_ms=0.95)
    assert assess_profile(profile)["timing_verdict"] == "UNQUALIFIED"


def test_nearfield_count_does_not_qualify_accuracy() -> None:
    obj = {
        "near_field_completed_d_gt_127_5_pixels": 25064,
        "vs_zncc_d_gt_127_5_overlap_pixels": 1,
        "vs_zncc_d_gt_127_5_bad_2px_pct": 100,
    }
    result = assess_near_field(obj)
    assert result["reported_high_disparity_completed_pixels"] > 20000
    assert result["verdict"] == "UNQUALIFIED"
    assert result["independently_qualified"] is False
    obj["vs_zncc_d_gt_127_5_overlap_pixels"] = MIN_INDEPENDENT_NEAR_FIELD_OVERLAP
    obj["vs_zncc_d_gt_127_5_bad_2px_pct"] = 0.5
    assert assess_near_field(obj)["verdict"] == "PASS_PRELIMINARY"


def test_invalid_timing_rejected() -> None:
    profile = _profile()
    profile["gpu_combined_p50_ms"] = 0.0
    with pytest.raises(ValueError):
        assess_profile(profile)


@pytest.mark.skipif(not V10.exists(), reason="v10 report not present in checkout")
def test_real_v10_is_no_go_and_reports_nearfield_and_competition_status() -> None:
    report = assess_v10(json.loads(V10.read_text(encoding="utf-8")))
    assert report["global_verdict"] == "NO_GO"
    assert report["near_field_d64"]["verdict"] == "PASS_PRELIMINARY"
    assert report["near_field_d64"]["independent_reference_overlap_pixels"] >= MIN_INDEPENDENT_NEAR_FIELD_OVERLAP
    assert report["near_field_d64"]["independent_bad_2px_pct_on_overlap"] <= 5.0
    assert report["competing_workload_provenance"]["field_present"] is True
    assert "not a matched algorithmic A/B" in report["speedup_comparison_caveat"]
    for prof_key in ("v11_fast_measured_only", "v11_fast_with_planar"):
        assert prof_key in report["v10_profiles"]
        coarse_obs = report["v10_profiles"][prof_key]["stage_snapshot_vs_allocations"]["coarse_total_ms"]["observed_ms"]
        assert coarse_obs <= STAGE_BUDGET_MS["coarse_total_ms"]


@pytest.mark.skipif(not V11.exists(), reason="v11 report not present in checkout")
def test_v11_qualification_and_coarse_replacement_contracts() -> None:
    data = json.loads(V11.read_text(encoding="utf-8"))
    exec_sum = data["executive_summary"]
    assert exec_sum["v11_0_issue_1_estimator_ablation_resolved"] is True
    assert exec_sum["v11_0_issue_2_near_field_d64_qualified"] is True
    assert exec_sum["v11_1_coarse_stage_gate_250us_passed"] is True

    zed2 = data["v11_0_ablation_zed2_cohorts_and_gpu_distributions"]
    assert zed2["a1_vs_a2_exact_ultra_bit_exact_left_eye"]["measured_bit_exact"] is True
    assert zed2["a1_vs_a2_exact_ultra_bit_exact_left_eye"]["completed_bit_exact"] is True

    gate = data["v11_1_coarse_sgm_replacement_gate"]
    assert gate["gate_passed"] is True
    assert gate["observed_p50_ms"] <= STAGE_BUDGET_MS["coarse_total_ms"]
    assert gate["observed_p95_ms"] <= STAGE_BUDGET_MS["coarse_total_ms"]
    assert gate["synthetic_5_scene_good_2px_delta_pp_vs_A0"] >= -0.50
    assert gate["synthetic_5_scene_bad_2px_delta_pp_vs_A0"] <= 0.25

    cpu_sec = data["v11_3_complete_camera_to_depth_slang_cpu_backend"]
    assert cpu_sec["samples_per_configuration"] >= 100
    assert cpu_sec["interleaved_blocks"] >= 3
    cpu_640 = cpu_sec["resolutions"]["640x360_mobile"]
    assert cpu_640["controlled_same_config_bit_exact_parity"] is True
    assert cpu_640["full_camera_to_depth_cpu_p50_ms"] <= 10.0
    assert cpu_640["full_camera_to_depth_valid_completed_pixels"] > 180000
    assert "thread_scaling_640x360" in cpu_sec
    assert "git_head_commit_timestamp" in data["provenance"]


def test_slang_cpu_full_camera_to_depth_pipeline_end_to_end() -> None:
    import numpy as np
    from stereoforge.slang_cpu_backend import SlangCpuRefinementBackend

    backend = SlangCpuRefinementBackend()
    w, h = 160, 96
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    true_disp = (8.0 + 0.04 * yy).astype(np.float32)
    pat = (
        128.0
        + 45.0 * np.sin(xx * 0.27 + yy * 0.13)
        + 35.0 * np.cos(xx * 0.19 - yy * 0.23)
        + 20.0 * np.sin((xx * 0.41) * (yy * 0.07))
    )
    left_u8 = np.clip(pat, 15, 240).astype(np.uint8)
    right_x = np.clip(xx + true_disp, 0.0, float(w - 1))
    x0 = np.floor(right_x).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    a = right_x - x0
    y_idx = yy.astype(np.int32)
    right_u8 = np.clip((1.0 - a) * left_u8[y_idx, x0] + a * left_u8[y_idx, x1], 15, 240).astype(np.uint8)

    sbs = np.concatenate([left_u8, right_u8], axis=1)
    yuyv = np.empty((h, 2 * w, 2), dtype=np.uint8)
    yuyv[:, :, 0] = sbs
    yuyv[:, :, 1] = 128
    cw, ch = w // 4, h // 4
    cy_g, cx_g = np.mgrid[0:ch, 0:cw].astype(np.float32)
    t_maps = (xx.reshape(-1), yy.reshape(-1), xx.reshape(-1), yy.reshape(-1))
    c_maps = (
        (cx_g * 4.0 + 1.5).reshape(-1),
        (cy_g * 4.0 + 1.5).reshape(-1),
        (cx_g * 4.0 + 1.5).reshape(-1),
        (cy_g * 4.0 + 1.5).reshape(-1),
    )

    res = backend.run_full_camera_to_depth_pipeline(
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
        fx=260.0,
        baseline=0.12,
        principal_offset=0.0,
        disparities=32,
        radius=2,
        ultra_fast=True,
        planar_recovery=True,
    )
    comp_val = res["completed_validity"][: w * h].reshape(h, w)
    comp_disp = res["completed_disparity"][: w * h].reshape(h, w)
    valid = (comp_val != 0) & (xx >= 24) & (xx < w - 16) & (yy >= 16) & (yy < h - 16)
    assert int(np.sum(valid)) > 5000
    mae = float(np.mean(np.abs(comp_disp[valid] - true_disp[valid])))
    assert mae < 0.25
