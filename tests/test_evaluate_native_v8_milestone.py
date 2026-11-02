"""Unit tests for the Step 5 and Step 6 native GPU benchmark evaluation script."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_step5_native_ortho_hybrid_depth_benchmark_artifact() -> None:
    path = ROOT / "docs/evidence/v8-review-milestone/step5-native-ortho-hybrid-depth-benchmark.json"
    assert path.is_file()
    doc = json.loads(path.read_text())
    assert doc["schema"] == "stereoforge.native_ortho_hybrid_depth_v8.v1"
    runs = doc["runs"]
    assert "depth_census_sequential" in runs
    assert "depth_ortho-hybrid_sequential" in runs
    assert "depth_ortho-hybrid_derived" in runs

    c_seq = runs["depth_census_sequential"]
    o_seq = runs["depth_ortho-hybrid_sequential"]
    o_der = runs["depth_ortho-hybrid_derived"]

    for r in (c_seq, o_seq, o_der):
        assert r["reference_checks"]["lr_mask_metadata_reference_checked"] is True
        assert r["reference_checks"]["validation_errors_after_teardown"] == 0
        assert r["reference_checks"]["validation_warnings_after_teardown"] == 0

    assert o_seq["reference_checks"]["both_stereo_integer_reference_checked"] is True
    assert o_der["reference_checks"]["left_stereo_integer_reference_checked"] is True
    assert o_der["reference_checks"]["derived_right_mask_reference_checked"] is True

    assert o_seq["lr_valid_left_pct"] > c_seq["lr_valid_left_pct"] + 7.0
    assert o_seq["lr_agreement_among_both_selected_pct"] > c_seq["lr_agreement_among_both_selected_pct"] + 10.0
    assert o_seq["geometry"]["planar_disparity_rmse_median_px"] < c_seq["geometry"]["planar_disparity_rmse_median_px"] * 0.3
    assert o_seq["geometry"]["speckle_survival_rate_of_lr_accepted"] > c_seq["geometry"]["speckle_survival_rate_of_lr_accepted"] + 0.14


def test_step6_multiresolution_refinement_240fps_artifact() -> None:
    path = ROOT / "docs/evidence/v8-review-milestone/step6-multiresolution-refinement-240fps-v8.json"
    assert path.is_file()
    doc = json.loads(path.read_text())
    assert doc["schema"] == "stereoforge.multiresolution_refinement_v8.v1"
    runs = doc["runs"]

    # Check 2x and 4x ortho-hybrid refinement vs census
    for key in (
        "ref2x_ortho-hybrid_layered_fast1",
        "ref2x_ortho-hybrid_derived_fast1",
        "ref4x_ortho-hybrid_layered_fast1",
        "ref4x_ortho-hybrid_derived_fast1",
        "dual2x_ortho-hybrid_layered_fast1",
        "dual2x_ortho-hybrid_derived_fast1",
        "dual4x_ortho-hybrid_layered_fast1",
        "dual4x_ortho-hybrid_derived_fast1",
    ):
        assert key in runs
        assert runs[key]["validation_errors_after_teardown"] == 0
        assert runs[key]["validation_warnings_after_teardown"] == 0

    for dual_key in (
        "dual2x_ortho-hybrid_layered_fast1",
        "dual2x_ortho-hybrid_derived_fast1",
        "dual4x_ortho-hybrid_layered_fast1",
        "dual4x_ortho-hybrid_derived_fast1",
    ):
        assert runs[dual_key]["all_30_planes_exact_between_variants"] is True

    # 240 FPS budget: <= 4.167 ms p50 host wall on 4x coarse->720p refinement
    assert runs["dual4x_ortho-hybrid_derived_fast1"]["host_wall_ms"]["p50"] <= 4.167
    assert runs["dual4x_ortho-hybrid_layered_fast1"]["host_wall_ms"]["p50"] <= 4.167
    assert runs["ref4x_ortho-hybrid_derived_fast1"]["lr_valid_left_pct"] >= 24.0
    assert runs["ref4x_ortho-hybrid_derived_fast1"]["geometry"]["planar_disparity_rmse_median_px"] <= 1.5

    # Temporal stability improvement
    temp = doc["temporal_5frame_capture_stability"]
    assert temp["ortho_hybrid_u32"]["mean_confirmed_current_pct"] > temp["census_24bit"]["mean_confirmed_current_pct"] + 6.0
    assert temp["ortho_hybrid_u32"]["mean_confirmed_plus_100ms_hold_pct"] > temp["census_24bit"]["mean_confirmed_plus_100ms_hold_pct"] + 7.0
    assert temp["ortho_hybrid_u32"]["mean_frame_to_frame_raw_dropout_pct"] < temp["census_24bit"]["mean_frame_to_frame_raw_dropout_pct"]
