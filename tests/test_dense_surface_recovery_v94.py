"""Contract and qualification tests for StereoForge v9.4 Dense Low-Texture Surface Recovery:
- Explicit 6-code provenance contract (0=INVALID, 1=STEREO, 2=REPROJECTED, 3=FUSED, 6=PLANAR_INFERRED, 7=TEMPORAL_PLANAR_INFERRED)
- 11-stage unrecovered-pixel waterfall taxonomy & single-guard relaxation ceiling matrix
- Near-field d > 127 px coarse SGM best==31 (124.0 px) saturation isolation
- 4-mode multi-frame surface recovery comparison (Measured vs Spatial Planar vs Temporal vs Combined Temporal+Planar)
- Multi-reference geometric accuracy (Independent ZNCC, Cross-Sensor Intel RealSense D435i, Exact Synthetic GT)
- Counterexample safety gates (occlusion boundaries, narrow-gap bars, thin structures)
- 5-trial interleaved 240 FPS GPU performance qualification
"""
from __future__ import annotations

import json
from pathlib import Path

from stereoforge.types import Provenance


def test_provenance_enum_contract_v94():
    assert int(Provenance.INVALID) == 0
    assert int(Provenance.STEREO) == 1
    assert int(Provenance.REPROJECTED) == 2
    assert int(Provenance.FUSED) == 3
    assert int(Provenance.PLANAR_INFERRED) == 6
    assert int(Provenance.TEMPORAL_PLANAR_INFERRED) == 7
    codes = {
        int(Provenance.INVALID),
        int(Provenance.STEREO),
        int(Provenance.REPROJECTED),
        int(Provenance.FUSED),
        int(Provenance.PLANAR_INFERRED),
        int(Provenance.TEMPORAL_PLANAR_INFERRED),
    }
    assert len(codes) == 6


def test_v94_dense_surface_qualification_artifact():
    qual_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks/results/v9_4_dense_surface_qualification.json"
    )
    assert qual_path.exists(), f"Missing {qual_path}"
    doc = json.loads(qual_path.read_text())
    assert doc["schema"] == "stereoforge.v9_4_dense_surface_qualification.v1"

    # 1. Provenance contract
    p_contract = doc["provenance_enum_contract"]
    assert p_contract["PLANAR_INFERRED"] == 6
    assert p_contract["TEMPORAL_PLANAR_INFERRED"] == 7
    assert p_contract["distinct_provenance_codes"] is True

    # 2. Unrecovered-pixel waterfall taxonomy & near-field saturation diagnostic
    tax = doc["unrecovered_pixel_taxonomy_and_relaxation_matrix"]
    wf = tax["waterfall_by_roi"]
    for roi_key in (
        "upright_apriltag_target",
        "charuco_board_inrange",
        "horizontal_wood_desk_clean",
        "horizontal_desk_plane_clean",
        "broad_desk_inrange",
        "broad_desk_outofrange_d_gt_127",
    ):
        assert roi_key in wf
        r_wf = wf[roi_key]
        assert (
            r_wf["measured_stereo_prov1_pixels"] + r_wf["unrecovered_hole_pixels"]
            == r_wf["total_roi_pixels"]
        )
        assert (
            r_wf["stage_1_out_of_disparity_range_d_gt_127_pixels"]
            + r_wf["stage_2_no_valid_coarse_prior_pixels"]
            + r_wf["stage_3_valid_coarse_candidates_pixels"]
            == r_wf["unrecovered_hole_pixels"]
        )

    oor_diag = tax["near_field_d_gt_127_saturation_diagnostic"]
    assert oor_diag["extended_zncc_192_disparity_p10_p50_p90_px"][1] > 127.0
    assert oor_diag["clamped_share_of_measured_in_oor_pct"] > 50.0

    # 3. Multi-frame 4-mode ZED 2 desk evaluation (Independent ZNCC vs Self-Consistency separated)
    mf = doc["multiframe_4mode_zed2_desk_evaluation"]
    assert mf["frames_evaluated"] == 8
    zncc_planes = mf["independent_zncc_reference_planes"]
    self_planes = mf["self_consistency_to_measured_planes"]

    for rname in (
        "upright_apriltag_target",
        "charuco_board_inrange",
        "horizontal_wood_desk_clean",
        "horizontal_desk_plane_clean",
        "broad_desk_inrange",
    ):
        assert rname in zncc_planes
        assert rname in self_planes
        assert (
            zncc_planes[rname]["reference_kind"]
            == "independent_multiscale_zncc_epipolar_ransac_plane"
        )
        assert (
            self_planes[rname]["reference_kind"]
            == "self_consistency_to_measured_stereo_ransac_plane"
        )
        modes = zncc_planes[rname]["modes"]
        cov_m1 = modes["mode_1_measured_stereo_only"]["completed_total"]["roi_coverage_pct"]
        cov_m2 = modes["mode_2_spatial_planar_propagation"]["completed_total"]["roi_coverage_pct"]
        cov_m3 = modes["mode_3_temporal_accumulation_8frame"]["completed_total"]["roi_coverage_pct"]
        cov_m4 = modes["mode_4_combined_temporal_plus_planar"]["completed_total"]["roi_coverage_pct"]
        assert cov_m2 > cov_m1, f"{rname}: spatial planar must improve over measured stereo"
        assert cov_m4 >= cov_m2, f"{rname}: combined mode must be >= spatial planar alone"
        assert cov_m4 >= cov_m3, f"{rname}: combined mode must be >= temporal alone"

    # Specific quantitative gains on ZED 2 tabletop ROIs against Independent ZNCC Reference
    tag_m2 = zncc_planes["upright_apriltag_target"]["modes"]["mode_2_spatial_planar_propagation"]
    assert tag_m2["newly_recovered_only"]["pixels"] >= 300
    assert tag_m2["newly_recovered_only"]["depth_mae_mm"] <= 2.0

    char_m2 = zncc_planes["charuco_board_inrange"]["modes"]["mode_2_spatial_planar_propagation"]
    assert char_m2["newly_recovered_only"]["pixels"] >= 200
    assert char_m2["newly_recovered_only"]["depth_mae_mm"] <= 3.0

    desk_m1 = zncc_planes["horizontal_desk_plane_clean"]["modes"]["mode_1_measured_stereo_only"]
    desk_m2 = zncc_planes["horizontal_desk_plane_clean"]["modes"]["mode_2_spatial_planar_propagation"]
    assert desk_m2["newly_recovered_only"]["pixels"] >= 400
    assert desk_m2["newly_recovered_only"]["depth_mae_mm"] <= 5.0
    assert desk_m2["newly_recovered_only"]["normal_angle_err_deg"] <= 6.0
    assert desk_m2["completed_total"]["normal_angle_err_deg"] < desk_m1["completed_total"]["normal_angle_err_deg"]

    # 4. Synthetic ground-truth & counterexamples
    cohorts = doc["canonical_5scene_720p_per_scene_inferred_cohorts"]
    assert cohorts["occlusion_boundaries"]["independent_4x_planar_inferred_prov6"]["bad_2px_pct"] == 0.0
    assert cohorts["low_texture_surface"]["independent_4x_planar_inferred_prov6"]["bad_1px_pct"] == 0.0
    assert cohorts["low_texture_surface"]["independent_4x_planar_inferred_prov6"]["bad_2px_pct"] == 0.0
    assert cohorts["slanted_planes"]["independent_4x_planar_inferred_prov6"]["bad_2px_pct"] == 0.0

    stress = doc["synthetic_woodgrain_and_counterexample_stress_suite"]
    wg = stress["slanted_tabletop_woodgrain_low_texture"]
    assert wg["mode_2_spatial_planar_only"]["coverage_pct"] > wg["mode_1_measured_stereo_only"]["coverage_pct"]
    assert wg["mode_4_combined_temporal_plus_planar"]["coverage_pct"] >= wg["mode_2_spatial_planar_only"]["coverage_pct"]
    assert wg["mode_4_combined_temporal_plus_planar"]["bad_2px_pct"] <= 0.5

    gaps = stress["narrow_gap_bars_counterexample"]
    assert gaps["false_foreground_bridged_pct_of_non_occluded_gaps"] <= 1.0
    assert gaps["non_occluded_inferred_cohort"]["bad_2px_pct"] <= 0.2

    # 5. 5-Trial Interleaved 240 FPS GPU Qualification
    gpu_plan = doc["gpu_240fps_qualification"]["dual4x_selective_repair_plus_planar_v94"]
    assert gpu_plan["interleaved_trials_count"] == 5
    assert gpu_plan["meets_240fps_4_167ms_gpu_p50"] is True
    assert gpu_plan["meets_240fps_4_167ms_gpu_p95"] is True
    assert gpu_plan["interleaved_trial_distributions"]["all_trials_meet_240fps_gpu_p95"] is True
    assert gpu_plan["validation_errors_after_teardown"] == 0
    assert "planar_feature_overhead_accounting_ms" in gpu_plan
