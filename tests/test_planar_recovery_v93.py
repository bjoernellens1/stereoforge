"""Deterministic CPU contract tests for opt-in planar-reference experiment."""
import numpy as np
import cv2

from benchmarks.planar_recovery_v93 import recover_planar_reference, PlanarRecoveryConfig


def _scene(disparity=24.0):
    h, w = 96, 160
    rng = np.random.default_rng(42)
    raw = rng.normal(128, 35, (h, w)).astype(np.float32)
    left = np.clip(cv2.GaussianBlur(raw, (3, 3), 0.7), 0, 255)
    right = np.empty_like(left)
    di = int(disparity)
    right[:, :-di] = left[:, di:]
    right[:, -di:] = 128
    valid = np.ones((h, w), dtype=bool)
    valid[35:63, 60:95] = False
    valid[:6] = False
    valid[-6:] = False
    measured = np.where(valid, disparity, np.nan).astype(np.float32)
    coarse = np.full((h // 4, w // 4), disparity / 4, np.float32)
    coarse_valid = np.ones_like(coarse, dtype=bool)
    return measured, valid, coarse, coarse.copy(), coarse_valid, coarse_valid.copy(), left, right


def test_flat_desk_fills_with_distributed_measured_anchors():
    args = _scene()
    result = recover_planar_reference(*args, config=PlanarRecoveryConfig(max_luma_gradient_dn=200))
    assert result.diagnostics["inferred_pixels"] > 100, result.diagnostics
    assert np.all(result.provenance[args[1]] == 1)
    assert np.array_equal(result.depth_disparity[args[1]], args[0][args[1]])
    assert np.allclose(result.depth_disparity[result.provenance == 2], 24, atol=0.1)
    assert np.all(np.isnan(result.depth_disparity[result.provenance == 0]))


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
    valid[35:62, 68:96] = False
    measured = np.where(valid, gt, np.nan)
    cy, cx = np.indices((h // 4, w // 4), dtype=np.float32)
    coarse_d = (22 + 0.025 * (4 * cx + 1.5) + 0.04 * (4 * cy + 1.5)) / 4
    gray = np.full((h, w), 128, np.float32)
    result = recover_planar_reference(
        measured, valid, coarse_d, coarse_d.copy(),
        np.ones_like(coarse_d, bool), np.ones_like(coarse_d, bool),
        gray, gray, config=PlanarRecoveryConfig(max_luma_gradient_dn=200),
    )
    inferred = result.provenance == 2
    assert inferred.sum() > 50, result.diagnostics
    assert np.max(np.abs(result.depth_disparity[inferred] - gt[inferred])) < 0.25


def test_step_discontinuity_does_not_bridge_depth_edge():
    args = list(_scene())
    args[2][:, 20:] = 10.0
    args[3][:, 20:] = 10.0
    args[0][:, 80:] = np.where(args[1][:, 80:], 40.0, np.nan)
    result = recover_planar_reference(
        *args, config=PlanarRecoveryConfig(max_luma_gradient_dn=200)
    )
    assert np.sum(result.provenance[:, 76:84] == 2) == 0
    assert np.sum(result.provenance[:, 55:75] == 2) > 0
