"""Deterministic CPU contract tests for provenance-tagged planar recovery (Provenance.PLANAR_INFERRED = 6)
and D-Y / D-UV / D-YUV / D-UV-Shuffled causal tie-break ablation."""
import cv2
import numpy as np

from benchmarks.planar_recovery_v93 import (
    PlanarRecoveryConfig,
    evaluate_color_tiebreak_causal_ablation,
    recover_planar_reference,
)
from stereoforge.types import Provenance


def _scene(disparity: float = 24.0):
    h, w = 96, 160
    rng = np.random.default_rng(42)
    raw = rng.normal(128, 35, (h, w)).astype(np.float32)
    left = np.clip(cv2.GaussianBlur(raw, (3, 3), 0.7), 0, 255)
    right = np.empty_like(left)
    di = int(disparity)
    right[:, :-di] = left[:, di:]
    right[:, -di:] = 128
    valid = np.ones((h, w), dtype=bool)
    valid[38:54, 70:86] = False
    valid[:6] = False
    valid[-6:] = False
    measured = np.where(valid, disparity, np.nan).astype(np.float32)
    coarse = np.full((h // 4, w // 4), disparity / 4, np.float32)
    coarse_valid = np.ones_like(coarse, dtype=bool)
    return measured, valid, coarse, coarse.copy(), coarse_valid, coarse_valid.copy(), left, right


def test_flat_desk_fills_with_distributed_measured_anchors_and_provenance_6():
    args = _scene()
    result = recover_planar_reference(*args, config=PlanarRecoveryConfig(max_luma_gradient_dn=200))
    assert result.diagnostics["inferred_pixels"] > 100, result.diagnostics
    # Measured output must never be modified or contain PLANAR_INFERRED
    assert np.all(result.measured_provenance[args[1]] == int(Provenance.STEREO))
    assert np.all(result.measured_provenance[~args[1]] == int(Provenance.INVALID))
    assert np.array_equal(result.measured_disparity[args[1]], args[0][args[1]])
    assert np.all(np.isnan(result.measured_disparity[~args[1]]))
    # Completed output carries Provenance.PLANAR_INFERRED = 6 (never Provenance.REPROJECTED = 2)
    assert int(Provenance.PLANAR_INFERRED) == 6
    assert int(Provenance.REPROJECTED) == 2
    assert np.all(result.completed_provenance[args[1]] == int(Provenance.STEREO))
    inferred = result.completed_provenance == int(Provenance.PLANAR_INFERRED)
    assert int(inferred.sum()) == result.diagnostics["inferred_pixels"]
    assert np.all(result.completed_provenance != int(Provenance.REPROJECTED))
    assert np.allclose(result.completed_disparity[inferred], 24.0, atol=0.1)
    assert np.all(np.isnan(result.completed_disparity[result.completed_provenance == 0]))


def test_no_fill_without_anchors_or_outside_disparity_range():
    args = list(_scene())
    args[1][:] = False
    args[0][:] = np.nan
    assert recover_planar_reference(*args).diagnostics["inferred_pixels"] == 0
    args = list(_scene())
    args[2][:] = 45
    args[3][:] = 45
    assert recover_planar_reference(*args).diagnostics["inferred_pixels"] == 0


def test_coarse_stereo_lr_disagreement_rejected():
    args = list(_scene())
    args[3][:] = 10
    assert recover_planar_reference(*args).diagnostics["inferred_pixels"] == 0


def test_mismatched_coarse_geometry_rejected():
    args = list(_scene())
    args[3] = args[3][:, :-1]
    try:
        recover_planar_reference(*args)
    except ValueError:
        return
    raise AssertionError("mismatched coarse right geometry must be rejected")


def test_slanted_plane_preserves_affine_disparity():
    h, w = 96, 160
    y, x = np.indices((h, w), dtype=np.float32)
    gt = (22 + 0.025 * x + 0.040 * y).astype(np.float32)
    valid = np.ones((h, w), bool)
    valid[38:53, 72:87] = False
    measured = np.where(valid, gt, np.nan)
    cy, cx = np.indices((h // 4, w // 4), dtype=np.float32)
    coarse_d = (22 + 0.025 * (4 * cx + 1.5) + 0.04 * (4 * cy + 1.5)) / 4
    gray = np.full((h, w), 128, np.float32)
    result = recover_planar_reference(
        measured,
        valid,
        coarse_d,
        coarse_d.copy(),
        np.ones_like(coarse_d, bool),
        np.ones_like(coarse_d, bool),
        gray,
        gray,
        config=PlanarRecoveryConfig(max_luma_gradient_dn=200),
    )
    inferred = result.completed_provenance == int(Provenance.PLANAR_INFERRED)
    assert inferred.sum() > 50, result.diagnostics
    assert np.max(np.abs(result.completed_disparity[inferred] - gt[inferred])) < 0.25


def test_step_discontinuity_does_not_bridge_depth_edge():
    args = list(_scene())
    args[2][:, 20:] = 10.0
    args[3][:, 20:] = 10.0
    args[0][:, 80:] = np.where(args[1][:, 80:], 40.0, np.nan)
    result = recover_planar_reference(
        *args, config=PlanarRecoveryConfig(max_luma_gradient_dn=200)
    )
    assert np.sum(result.completed_provenance[:, 76:84] == int(Provenance.PLANAR_INFERRED)) == 0
    assert np.sum(result.completed_provenance[:, 68:75] == int(Provenance.PLANAR_INFERRED)) > 0


def test_color_tiebreak_causal_ablation_separates_luma_from_chroma():
    h, w = 64, 96
    rng = np.random.default_rng(7)
    rgb_l = rng.integers(80, 180, size=(h, w, 3), dtype=np.uint8)
    rgb_r = np.roll(rgb_l, -12, axis=1)
    base_d = np.full((h, w), 12.0, dtype=np.float32)
    base_v = np.ones((h, w), dtype=bool)
    base_v[16:48, 24:72] = False
    reason = np.where(~base_v, np.uint32(32), np.uint32(0))
    coarse_p = np.full((h, w), 12.0, dtype=np.float32)
    coarse_v = np.ones((h, w), dtype=bool)
    res = evaluate_color_tiebreak_causal_ablation(
        rgb_l, rgb_r, base_d, base_v, reason, coarse_p, coarse_v
    )
    assert set(res["variants"].keys()) == {
        "D_Y_luma_only",
        "D_UV_chroma_only",
        "D_YUV_combined",
        "D_UV_shuffled_control",
    }
    assert res["variants"]["D_UV_chroma_only"]["admitted_pixels"] > res["variants"]["D_UV_shuffled_control"]["admitted_pixels"]


def test_v93_qualification_artifact_contract():
    import json
    from pathlib import Path

    qual_path = Path(__file__).resolve().parents[1] / "benchmarks/results/v9_3_planar_qualification.json"
    assert qual_path.exists()
    doc = json.loads(qual_path.read_text())
    assert doc["schema"] == "stereoforge.v9_3_planar_qualification.v1"
    assert doc["provenance_enum_contract"]["PLANAR_INFERRED"] == 6
    assert doc["provenance_enum_contract"]["REPROJECTED"] == 2
    assert doc["provenance_enum_contract"]["planar_never_collides_with_reprojected"] is True
    gpu_plan = doc["gpu_240fps_qualification"]["dual4x_selective_repair_plus_planar"]
    assert gpu_plan["meets_240fps_4_167ms_gpu_p50"] is True
    assert gpu_plan["meets_240fps_4_167ms_gpu_p95"] is True
    assert gpu_plan["meets_240fps_4_167ms_gpu_p99"] is True
    assert gpu_plan["interleaved_trials_count"] == 5
    assert gpu_plan["interleaved_trial_distributions"]["all_trials_meet_240fps_gpu_p95"] is True
    assert gpu_plan["gpu_planar_recovery_ms"]["p50"] <= 0.45
    assert gpu_plan["validation_errors_after_teardown"] == 0
    assert doc["canonical_5scene_720p_per_scene_inferred_cohorts"]["occlusion_boundaries"]["independent_4x_planar_inferred_prov6"]["inferred_pixels"] == 0
    assert doc["canonical_5scene_720p_per_scene_inferred_cohorts"]["low_texture_surface"]["independent_4x_planar_inferred_prov6"]["bad_1px_pct"] == 0.0
    desk_def = doc["desk_scene_gpu_ablation"]["v93_planar_a10_affine3x3_default"]
    assert desk_def["measured_plane_bit_exact_with_nonplanar"] is True
    assert "independent_zncc_reference_planes" in desk_def
    assert "consistency_to_measured_planes" in desk_def
    for pkey in ("upright_apriltag_target", "charuco_board_inrange", "horizontal_desk_plane_clean"):
        assert desk_def["independent_zncc_reference_planes"][pkey]["reference_kind"] == "independent_multiscale_zncc_epipolar_ransac_plane"
        assert desk_def["consistency_to_measured_planes"][pkey]["reference_kind"] == "self_consistency_to_measured_stereo_ransac_plane"
        assert desk_def["independent_zncc_reference_planes"][pkey]["inferred_prov6"]["pixels"] > 0
        assert desk_def["consistency_to_measured_planes"][pkey]["inferred_prov6"]["pixels"] > 0

