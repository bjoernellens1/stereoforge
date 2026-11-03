"""Tests for StereoForge Milestone v12 (`P0`–`P4`) Temporal Supersampling & Multi-Frame Enhancement:
- `P0` single-frame continuous 5-point half-pixel zero-mean 3x3 photometric subpixel refinement (`continuous_subpixel=True`),
- `P1` full-resolution `v11` (`DaxaRefinement::Frame`) integration with pose-warped `DaxaTemporal`,
- `P2` calibrated inverse-depth uncertainty fusion, foreground depth-ordering disocclusion guard, and 50% `left_eye_only` memory reduction,
- `P3` pose-warped temporal stereo cost refinement (`d_pred = fx * B / z_warp + cx_offset` Census re-search + L-R consistency check) with strict provenance honesty (`Provenance::Stereo = 1` with `age_ns = 0` only when current-frame Census + L-R check succeeds; historical fills strictly remain `Provenance::Reprojected = 2` or `7` with `age_ns > 0`),
- `P4` 2x joint-luma edge-preserving inverse-depth super-resolution and qualification artifact (`benchmarks/results/v12_temporal_supersampling_qualification.json`).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from benchmarks.temporal_supersampling_v12 import (
    OUT_V12_JSON,
    PROVENANCE_REPROJECTED,
    PROVENANCE_STEREO,
    run_daxa_temporal_step,
    upsample_depth_2x_joint_luma,
)
from benchmarks.visual_accuracy_audit_v9 import gpu_env

ROOT = Path(__file__).resolve().parents[1]


def test_p0_continuous_subpixel_improves_subpixel_disparity_accuracy() -> None:
    """P0: Continuous 5-point half-pixel zero-mean photometric subpixel refinement reduces
    disparity MAE and bad-0.5px error on half-pixel repetitive/subpixel surfaces."""
    assert OUT_V12_JSON.exists(), "Run benchmarks/temporal_supersampling_v12.py first"
    data = json.loads(OUT_V12_JSON.read_text(encoding="utf-8"))
    suite_a = data["suite_a_synthetic_known_pose_translation"]
    for scene_name in ("repetitive_and_subpixel", "slanted_planes"):
        v11 = suite_a[scene_name]["v11_single_frame"]
        p0 = suite_a[scene_name]["v11_p0_continuous_subpixel"]
        assert p0["disparity_mae_px"] < v11["disparity_mae_px"]
        assert p0["depth_mae_mm"] < v11["depth_mae_mm"]
        assert p0["bad_0_5px_pct"] <= v11["bad_0_5px_pct"]


def test_p1_p2_p3_multi_frame_translation_and_superresolution_gains() -> None:
    """P1–P4: Pose-warped inverse-depth fusion (`P1+P2`) and temporal stereo cost refinement (`P3+P4`)
    further reduce depth MAE and improve valid coverage over single-frame `v11` and unwarped RealSense EMA."""
    data = json.loads(OUT_V12_JSON.read_text(encoding="utf-8"))
    suite_a = data["suite_a_synthetic_known_pose_translation"]
    for scene_name in ("repetitive_and_subpixel", "slanted_planes"):
        v11 = suite_a[scene_name]["v11_single_frame"]
        ema = suite_a[scene_name]["realsense_ema_temporal"]
        p12 = suite_a[scene_name]["daxa_temporal_p1_p2"]
        p34 = suite_a[scene_name]["temporal_cost_refinement_p3_p4"]
        assert p12["depth_mae_mm"] < v11["depth_mae_mm"]
        assert p34["depth_mae_mm"] <= p12["depth_mae_mm"]
        assert p34["valid_coverage_pct"] >= v11["valid_coverage_pct"]
        assert p34["superres_2x_depth_mae_mm"] < v11["superres_2x_depth_mae_mm"]
        # Pose-warped temporal fusion beats unwarped EMA under camera translation
        assert p34["depth_mae_mm"] < ema["depth_mae_mm"]


def test_p2_disocclusion_guard_suppresses_moving_object_ghosting() -> None:
    """P2: Moving-object disocclusion guard suppresses false-obstacle ghosting in newly uncovered
    background wakes compared to unwarped EMA and unguarded temporal persistence."""
    data = json.loads(OUT_V12_JSON.read_text(encoding="utf-8"))
    suite_b = data["suite_b_moving_object_disocclusion"]
    mean_ghost = suite_b["mean_disocclusion_ghosting_false_obstacle_pct"]
    assert mean_ghost["daxa_temporal_p1_p2"] < mean_ghost["realsense_ema_temporal"]
    assert mean_ghost["daxa_temporal_p1_p2"] < mean_ghost["daxa_temporal_no_disocclusion_guard"]
    assert mean_ghost["temporal_cost_refinement_p3_p4"] <= mean_ghost["daxa_temporal_p1_p2"] + 0.5
    assert suite_b["provenance_honesty_violations"] == 0


def test_p3_provenance_honesty_and_temporal_stereo_promotion() -> None:
    """P3: Verify strict provenance separation:
    - Missing pixels confirmed by current-frame Census + L-R check around `d_pred` are promoted to
      `Provenance::Stereo = 1` with `age_ns = 0` and `source_id = current_frame_id`.
    - Missing pixels filled solely from warped history (when current-frame stereo fails) MUST retain
      `Provenance::Reprojected = 2` (or `7` for planar) with `age_ns > 0` and historical `source_id`."""
    h, w = 16, 32
    n = h * w
    fx, fy, cx, cy = 100.0, 100.0, 15.5, 7.5
    baseline_m = 0.10
    cx_offset = 0.0
    # True disparity = 5.0 px -> z = 2.0 m
    z_true = 2.0

    # Frame 1: valid everywhere at z = 2.0 m
    f1 = {
        "completed_depth": np.full(2 * n, z_true, dtype=np.float32),
        "completed_confidence": np.full(2 * n, 0.85, dtype=np.float32),
        "completed_validity": np.ones(2 * n, dtype=np.uint32),
        "completed_provenance": np.full(2 * n, PROVENANCE_STEREO, dtype=np.uint32),
        "census": np.full(2 * n, 0xAAAAAAAA, dtype=np.uint32),
        "census_validity": np.ones(2 * n, dtype=np.uint32),
        "raw_disparity": np.full(2 * n, 5.0, dtype=np.float32),
        "raw_validity": np.ones(2 * n, dtype=np.uint32),
        "luma": np.full(2 * n, 128, dtype=np.uint32),
        "support": np.ones(2 * n, dtype=np.uint32),
    }
    st1 = run_daxa_temporal_step(
        f1,
        None,
        width=w,
        height=h,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        baseline_m=baseline_m,
        cx_offset=cx_offset,
        current_from_previous=np.eye(4, dtype=np.float32),
        frame_id=10,
        inverse_depth_fusion=True,
        disocclusion_guard=True,
        temporal_stereo_research=True,
    )

    # Frame 2:
    # - Pixel (8, 16) is missing in `completed_validity`, but has matching current-frame Census & right-eye L-R at d=5 -> should promote to PROVENANCE_STEREO (frame_id=11, age_ns=0)
    # - Pixel (8, 20) is missing in `completed_validity` AND has invalid current-frame Census -> must remain PROVENANCE_REPROJECTED (frame_id=10, age_ns > 0)
    f2 = {k: v.copy() for k, v in f1.items()}
    idx_promote = 8 * w + 16
    idx_reproj = 8 * w + 20
    f2["completed_validity"][idx_promote] = 0
    f2["completed_depth"][idx_promote] = np.nan
    f2["completed_provenance"][idx_promote] = 0

    f2["completed_validity"][idx_reproj] = 0
    f2["completed_depth"][idx_reproj] = np.nan
    f2["completed_provenance"][idx_reproj] = 0
    f2["census_validity"][idx_reproj] = 0

    # Set distinct Census descriptors along row 8 so (8, 16) uniquely matches (8, 11) at d=5 with 0 Hamming cost
    for x in range(w):
        f2["census"][8 * w + x] = np.uint32((x * 2654435761) & 0xFFFFFFFF)
        f2["census"][n + 8 * w + x] = np.uint32(((x + 5) * 2654435761) & 0xFFFFFFFF)
    f2["census"][8 * w + 16] = np.uint32(0xCAFEBABE)
    f2["census"][n + 8 * w + 11] = np.uint32(0xCAFEBABE)

    st2 = run_daxa_temporal_step(
        f2,
        st1,
        width=w,
        height=h,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        baseline_m=baseline_m,
        cx_offset=cx_offset,
        current_from_previous=np.eye(4, dtype=np.float32),
        frame_id=11,
        inverse_depth_fusion=True,
        disocclusion_guard=True,
        temporal_stereo_research=True,
    )

    # Confirmed match at (8, 16) is promoted to fresh current-frame Stereo
    assert st2["validity"][8, 16] == 1
    assert st2["provenance"][8, 16] == PROVENANCE_STEREO
    assert st2["source_ids"][8, 16] == 11
    assert st2["age_ns"][8, 16] == 0
    assert st2["promoted_stereo_pixels"][8, 16]

    # Historical-only fill at (8, 20) strictly retains Reprojected provenance, source_id=10, and age_ns > 0
    assert st2["validity"][8, 20] == 1
    assert st2["provenance"][8, 20] == PROVENANCE_REPROJECTED
    assert st2["source_ids"][8, 20] == 10
    assert st2["age_ns"][8, 20] > 0
    assert not st2["promoted_stereo_pixels"][8, 20]


def test_p4_joint_luma_2x_superresolution_preserves_depth_edges() -> None:
    """P4: 2x joint-luma super-resolution preserves sharp step edges without flying-pixel bridging."""
    h1, w1 = 16, 32
    depth_1x = np.where(np.arange(w1)[None, :] < 16, 1.5, 3.5).repeat(h1, axis=0).astype(np.float32)
    valid_1x = np.ones((h1, w1), dtype=np.uint32)
    luma_2x = np.where(np.arange(w1 * 2)[None, :] < 32, 200, 50).repeat(h1 * 2, axis=0).astype(np.uint8)

    depth_2x, valid_2x = upsample_depth_2x_joint_luma(depth_1x, valid_1x, luma_2x)
    assert depth_2x.shape == (h1 * 2, w1 * 2)
    assert np.all(valid_2x == 1)
    # No flying-pixel intermediate depths (e.g. 2.0..3.0 m) across the step discontinuity
    bridged = (depth_2x > 1.65) & (depth_2x < 3.35)
    assert int(np.sum(bridged)) == 0


def test_zed2_sequence_and_memory_footprint_qualification() -> None:
    """Suite C: Verify ZED 2 5-frame temporal qualification and 50% GPU memory footprint reduction."""
    data = json.loads(OUT_V12_JSON.read_text(encoding="utf-8"))
    suite_c = data["suite_c_zed2_real_sequence_and_gpu_audit"]
    tiers = suite_c["zed2_5frame_tier_comparison"]
    v11 = tiers["v11_single_frame"]
    p0 = tiers["v11_p0_continuous_subpixel"]
    p12 = tiers["daxa_temporal_p1_p2"]
    p34 = tiers["temporal_cost_refinement_p3_p4"]

    assert p0["independent_zncc_mae_px"] <= v11["independent_zncc_mae_px"]
    assert p12["total_valid_coverage_pct"] >= p0["total_valid_coverage_pct"]
    assert p34["fresh_stereo_coverage_pct"] >= p12["fresh_stereo_coverage_pct"]
    assert p34["temporal_depth_jitter_median_std_mm"] <= p0["temporal_depth_jitter_median_std_mm"]

    mem = suite_c["temporal_gpu_memory_optimization_1280x720"]
    assert mem["reduction_pct"] == pytest.approx(50.0, abs=0.1)
    assert mem["left_eye_only_mb"] < 170.0


@pytest.mark.gpu
def test_gpu_daxa_temporal_and_refinement_v12_contracts() -> None:
    """GPU: Verify native C++ `daxa_temporal_contracts --gpu` (covering `DaxaRefinement::Frame` + P1/P2/P3 + `left_eye_only`)."""
    bin_path = ROOT / "build" / "ros-daxa-native" / "daxa_temporal_contracts"
    res = subprocess.run([str(bin_path), "--gpu"], env=gpu_env(), capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
