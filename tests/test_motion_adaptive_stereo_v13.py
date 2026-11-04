"""Tests for StereoForge Milestone v13 (`V13.0`–`V13.5`) Compute-Adaptive & Distance-Aware Stereo:
- Mandatory Targets: 1280x720 measured-depth GPU p50 <= 1.25 ms & moving-object wake ghosting <= 1.0%,
- `V13.0` Geometric motion & disocclusion-wake masking (`Static`, `DynamicForeground`, `DisocclusionWake`, `Unknown`) + 3-frame hysteresis (`Unknown` never treated as `Static`),
- `V13.1` 16x16 GPU tile classifier (`Motion x Range x Quality` -> `L0_Minimal`, `L1_Standard`, `L2_Precision`, `L3_Recovery`) + Compute-Budget Priority Controller (`priority_i = E[dQ_i] / (E[dT_i] + eps)`) with 0 thin-structure or near-field tiles misclassified as `L0_Minimal`,
- `v13.0` Range Oracle across the 7 physical depth bands (`<0.3 m`, `0.3-0.5 m`, `0.5-1.0 m`, `1.0-2.0 m`, `2.0-5.0 m`, `5.0-10.0 m`, `10.0-20.0 m`), 4-way range ablation, and 4 application profiles (`Manipulation`, `Navigation`, `Mapping`, `Performance`),
- `E0`–`E5` research sequence & 5-way compute-adaptive ablation (`benchmarks/results/v13_motion_adaptive_qualification.json`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.motion_adaptive_stereo_v13 import (
    LEVEL_L0_MINIMAL,
    LEVEL_L1_STANDARD,
    LEVEL_L2_PRECISION,
    LEVEL_L3_RECOVERY,
    MOTION_DISOCCLUSION_WAKE,
    MOTION_DYNAMIC_FOREGROUND,
    MOTION_STATIC,
    MOTION_UNKNOWN,
    OUT_V13_JSON,
    RANGE_NEAR,
    RANGE_TOO_NEAR_UNKNOWN,
    TIER_COARSE_PLANAR,
    TIER_FINE_FULLRES,
    TIER_PRECISION_EXPANDED,
    classify_motion_and_disocclusion_wake,
    classify_spatial_tiles_16x16,
    get_v13_application_profiles,
    max_useful_depth_for_accuracy_m,
    theoretical_depth_uncertainty_m,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v13_0_motion_and_wake_classifier_with_hysteresis() -> None:
    """V13.0: Verify geometric motion and disocclusion-wake masking with 3-frame hysteresis."""
    h, w = 32, 64
    n = h * w
    fx, baseline_m, cx_offset = 200.0, 0.10, 0.0
    z_bg = 4.0
    z_fg = 1.5

    warped_z = np.full((h, w), z_bg, dtype=np.float32)
    warped_z[8:24, 16:32] = z_fg
    warped_val = np.ones((h, w), dtype=bool)

    curr_z = np.full((h, w), z_bg, dtype=np.float32)
    curr_z[8:24, 24:40] = z_fg
    curr_val = np.ones((h, w), dtype=np.uint32)
    curr_val[8:24, 16:20] = 0
    curr_z[8:24, 16:20] = np.nan

    curr = {
        "completed_depth": np.concatenate([curr_z.ravel(), curr_z.ravel()]),
        "completed_validity": np.concatenate([curr_val.ravel(), curr_val.ravel()]),
        "luma": np.full(2 * n, 128, dtype=np.uint32),
    }

    res1 = classify_motion_and_disocclusion_wake(
        curr,
        warped_z,
        warped_val,
        None,
        width=w,
        height=h,
        fx=fx,
        baseline_m=baseline_m,
        cx_offset=cx_offset,
    )
    motion1 = res1["motion_state"]
    assert motion1[2, 2] == MOTION_STATIC
    assert motion1[16, 36] == MOTION_DYNAMIC_FOREGROUND
    assert motion1[16, 18] == MOTION_DISOCCLUSION_WAKE
    assert motion1[16, 22] == MOTION_DISOCCLUSION_WAKE
    assert res1["hysteresis_counter"][16, 22] == 3

    # Frame 3: even if raw motion becomes static, 3-frame hysteresis keeps counter > 0 for 2 more frames
    curr_static = {
        "completed_depth": np.full(2 * n, z_bg, dtype=np.float32),
        "completed_validity": np.ones(2 * n, dtype=np.uint32),
        "luma": np.full(2 * n, 128, dtype=np.uint32),
    }
    res2 = classify_motion_and_disocclusion_wake(
        curr_static,
        np.full((h, w), z_bg, dtype=np.float32),
        warped_val,
        res1["hysteresis_counter"],
        width=w,
        height=h,
        fx=fx,
        baseline_m=baseline_m,
        cx_offset=cx_offset,
    )
    assert res2["hysteresis_counter"][16, 22] == 2
    assert res2["motion_state"][16, 22] != MOTION_STATIC


def test_v13_1_tile_classifier_3d_state_and_compute_budget_controller() -> None:
    """V13.1: Verify 16x16 tile classifier (`Motion x Range x Quality` -> `L0..L3`) classifies smooth planar tiles
    as `L0_Minimal` (`CoarsePlanar`), never classifies thin structures, near-field targets, or `Unknown` motion as
    `L0_Minimal`, and respects compute-budget priority ordering `priority_i = E[dQ_i] / (E[dT_i] + eps)`."""
    h, w = 64, 64
    cw, ch = 16, 16
    n = h * w
    nc = cw * ch

    luma_l = np.full((h, w), 110, dtype=np.uint32)
    luma_l[:, 40:42] = 240  # Thin vertical pole in tile_x = 2 (columns 32..47)
    cen_l = np.zeros((h, w), dtype=np.uint32)
    cen_l[:, 40:42] = 0xFFFFFFFF

    c_disp = np.full((ch, cw), 4.0, dtype=np.float32)  # 16 px target disparity (Middle_Useful)
    c_disp[8:12, 4:8] = 18.0  # 72 px target disparity in tile (2, 1) -> Near obstacle (`RANGE_NEAR`)
    c_disp[12:16, 12:16] = 40.0  # 160 px target disparity in tile (3, 3) -> > 128 max_range_disp (`RANGE_TOO_NEAR_UNKNOWN`)

    curr = {
        "luma": np.concatenate([luma_l.ravel(), luma_l.ravel()]),
        "census": np.concatenate([cen_l.ravel(), cen_l.ravel()]),
        "census_validity": np.ones(2 * n, dtype=np.uint32),
        "coarse_disparity": np.concatenate([c_disp.ravel(), c_disp.ravel()]),
        "coarse_validity": np.ones(2 * nc, dtype=np.uint32),
        "raw_confidence": np.full(2 * n, 0.85, dtype=np.float32),
    }

    motion_mask = np.zeros((h, w), dtype=np.uint8)
    motion_mask[0:16, 16:32] = MOTION_UNKNOWN  # Tile (0, 1) has Unknown motion -> must NOT be L0_Minimal
    motion_mask[32:48, 0:16] = MOTION_DISOCCLUSION_WAKE  # Tile (2, 0) has DisocclusionWake -> L3_Recovery

    tiles = classify_spatial_tiles_16x16(
        curr,
        motion_mask,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
    )
    tier_grid = tiles["tile_tier_grid"]
    levels = tiles["tile_levels"]
    r_state = tiles["tile_range_state"]
    prio = tiles["tile_priority"]

    # Tile (1, 0) is static, smooth, middle-range -> L0_Minimal (CoarsePlanar)
    assert tier_grid[1, 0] == TIER_COARSE_PLANAR
    assert levels[1, 0] == LEVEL_L0_MINIMAL
    # Tile (1, 2) contains thin pole -> NEVER CoarsePlanar
    assert tier_grid[1, 2] in (TIER_FINE_FULLRES, TIER_PRECISION_EXPANDED)
    assert levels[1, 2] != LEVEL_L0_MINIMAL
    # Tile (0, 1) has MOTION_UNKNOWN -> NEVER treated as Static / L0_Minimal
    assert levels[0, 1] == LEVEL_L2_PRECISION
    # Tile (2, 0) has MOTION_DISOCCLUSION_WAKE -> L3_Recovery
    assert levels[2, 0] == LEVEL_L3_RECOVERY
    # Tile (2, 1) is RANGE_NEAR -> L2_Precision
    assert r_state[2, 1] == RANGE_NEAR
    assert levels[2, 1] == LEVEL_L2_PRECISION
    # Tile (3, 3) is RANGE_TOO_NEAR_UNKNOWN -> never L0_Minimal
    assert r_state[3, 3] == RANGE_TOO_NEAR_UNKNOWN
    assert levels[3, 3] >= LEVEL_L2_PRECISION
    # Priority ordering: L2/L3 priority > L1 > L0
    assert prio[1, 2] > prio[1, 0]

    # Verify compute-budget controller under tight budget (e.g. 15% of tiles allowed in L2/L3)
    budgeted = classify_spatial_tiles_16x16(
        curr,
        motion_mask,
        width=w,
        height=h,
        coarse_width=cw,
        coarse_height=ch,
        compute_budget_fraction=0.15,
    )
    n_high = int(np.sum(budgeted["tile_levels"] >= LEVEL_L2_PRECISION))
    assert n_high <= max(1, int(round(0.15 * 16)))


def test_v13_0_range_formulas_and_application_profiles() -> None:
    """Verify stereo triangulation range formulas `sigma_Z = (Z^2 / (fx*B))*sigma_d` and 4 application profiles."""
    fx, b = 527.79, 0.11944
    sig_1m = theoretical_depth_uncertainty_m(1.0, fx, b, sigma_d_px=0.20)
    sig_2m = theoretical_depth_uncertainty_m(2.0, fx, b, sigma_d_px=0.20)
    assert sig_2m == pytest.approx(4.0 * sig_1m, rel=1e-5)

    z_max = max_useful_depth_for_accuracy_m(sig_2m, fx, b, sigma_d_px=0.20)
    assert z_max == pytest.approx(2.0, rel=1e-5)

    profiles = get_v13_application_profiles(fx=fx, baseline_m=b, sigma_d_px=0.20)
    assert set(profiles.keys()) == {"Manipulation", "Navigation", "Mapping", "Performance"}
    assert profiles["Manipulation"]["max_disparity"] == 256
    assert profiles["Manipulation"]["z_min_m"] == 0.25
    assert profiles["Navigation"]["z_max_useful_m"] < profiles["Mapping"]["z_max_useful_m"]


def test_v13_mandatory_targets_range_oracle_and_5way_ablation_gates() -> None:
    """V13.0–V13.5: Verify mandatory performance & quality targets, 7-band Range Oracle, and E0-E5 5-way ablation."""
    assert OUT_V13_JSON.exists(), "Run benchmarks/motion_adaptive_stereo_v13.py first"
    data = json.loads(OUT_V13_JSON.read_text(encoding="utf-8"))

    # 1. Mandatory Targets: 720p measured-depth GPU p50 <= 1.25 ms and wake ghosting <= 1.0%
    targets = data["mandatory_targets"]
    perf = targets["performance_target_720p_measured_depth_p50_ms"]
    qual = targets["quality_target_moving_object_wake_ghosting_pct"]
    assert perf["qualified"] is True
    assert perf["v13_achieved_p50_ms"] <= 1.25
    assert qual["qualified"] is True
    assert qual["v13_achieved_cpu_sequence_pct"] <= 1.0
    assert qual["v13_achieved_native_gpu_pct"] <= 1.0

    # 2. v13.0 Range Oracle across 7 physical depth bands & 4-way range ablation
    oracle = data["v13_0_range_oracle"]
    bands = oracle["depth_bands_7"]
    assert len(bands) == 7
    assert bands["<0.3m"]["false_free_space_pixels"] == 0
    # 256-disparity near-field mode recovers 0.3-0.5 m (`d = 157.6 px`), whereas 128-disparity mode has 0% coverage there
    assert bands["0.3-0.5m"]["accepted_coverage_128d_pct"] == 0.0
    assert bands["0.3-0.5m"]["accepted_coverage_256d_pct"] >= 85.0
    assert bands["0.3-0.5m"]["incorrect_gt_5pct_256d_pct"] <= 1.0

    r_abl = oracle["four_way_range_ablation"]
    assert (
        r_abl["4_complete_distance_motion_quality_scheduler"]["candidate_savings_vs_post_filter_pct"]
        > r_abl["2_geometric_epipolar_and_range_candidate_pruning"]["candidate_savings_vs_post_filter_pct"]
        > 0.0
    )

    # 3. 4-variant and 5-way Compute-Adaptive Ablation Gates (E0-E5)
    ablation = data["four_variant_ablation"]
    v12_base = ablation["1_v12_baseline_no_tiles"]
    v13_0 = ablation["2_v13_0_motion_masking_only"]
    v13_2 = ablation["3_v13_2_adaptive_tiles_no_temporal"]
    v13_full = ablation["4_v13_full_combined"]

    assert v13_0["moving_object_disocclusion_sequence"]["mean_wake_false_obstacle_ghost_pct"] <= 1.0
    assert v13_full["moving_object_disocclusion_sequence"]["mean_wake_false_obstacle_ghost_pct"] <= 1.0
    assert v13_full["moving_object_disocclusion_sequence"]["provenance_honesty_violations"] == 0

    assert v13_2["thin_structure_and_planar_sequence"]["thin_pole_tiles_misclassified_as_coarse_planar"] == 0
    assert v13_full["thin_structure_and_planar_sequence"]["thin_pole_tiles_misclassified_as_coarse_planar"] == 0

    assert v13_full["thin_structure_and_planar_sequence"]["candidate_reduction_vs_uniform_p0_pct"] >= 20.0
    assert (
        v13_full["thin_structure_and_planar_sequence"]["thin_pole_disparity_mae_px"]
        <= v12_base["thin_structure_and_planar_sequence"]["thin_pole_disparity_mae_px"] + 0.05
    )

    # Simultaneous quality improvement: combined v13 achieves lower overall depth MAE than both uniform baseline and no-temporal v13.2
    assert (
        v13_full["thin_structure_and_planar_sequence"]["overall_depth_mae_mm"]
        < v12_base["thin_structure_and_planar_sequence"]["overall_depth_mae_mm"]
    )
    assert (
        v13_full["thin_structure_and_planar_sequence"]["overall_depth_mae_mm"]
        <= v13_2["thin_structure_and_planar_sequence"]["overall_depth_mae_mm"]
    )

    # 4. Native Vulkan GPU refinement & measured-depth latency gates
    gpu = data["native_vulkan_gpu_v13_timings_1280x720"]
    assert gpu["measured_depth_v13_adaptive_gpu_ms"]["p50"] <= 1.25
    assert gpu["measured_depth_v13_adaptive_gpu_ms"]["p50"] < gpu["measured_depth_v11_p0_gpu_ms"]["p50"]
    assert gpu["refinement_v13_adaptive_tiles_gpu_ms"]["p50"] < gpu["refinement_p0_gpu_ms"]["p50"]
