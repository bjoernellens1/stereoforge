"""Offline synthetic geometry evidence; these tests do not qualify SDK support."""
import math

import cv2
import numpy as np
import pytest

from benchmarks.zed_comparison.sdk_geometry import fit_ray_rotation, compare_rig_hypotheses


def rotation(v):
    return cv2.Rodrigues(np.asarray(v, dtype=np.float64))[0]


def brown5(point, K, d):
    # Independent scalar forward model, not OpenCV projection/undistortion.
    x, y = float(point[0] / point[2]), float(point[1] / point[2])
    r2 = x*x + y*y
    k1, k2, p1, p2, k3 = d
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    return [K[0, 0]*xd + K[0, 2], K[1, 1]*yd + K[1, 2]]


@pytest.fixture
def scene():
    raw_K = np.array([[740., 0, 640], [0, 725, 360], [0, 0, 1]])
    rect_K = np.array([[710., 0, 632], [0, 710, 358], [0, 0, 1]])
    d = np.array([-.12, .025, .0011, -.0008, -.003])
    Q = rotation([.035, -.052, .021])
    rays = np.array([[x, y, 1.] for y in [-.33, -.08, .21, .37]
                     for x in [-.45, -.17, .13, .42]])
    raw = np.array([brown5(p, raw_K, d) for p in rays])
    rect = np.array([brown5(Q @ p, rect_K, np.zeros(5)) for p in rays])
    train = np.arange(len(rays)) % 2 == 0
    return raw, rect, raw_K, d, rect_K, train, Q


def test_training_and_independent_holdout_recover_rotation(scene):
    args, Q = scene[:6], scene[6]
    saved = [a.copy() for a in args]
    result = fit_ray_rotation(*args)
    np.testing.assert_allclose(result['Q'], Q, atol=2e-11)
    assert result['status'] == 'diagnostic_unqualified'
    assert np.linalg.det(result['Q']) == pytest.approx(1.)
    for subset in ('train', 'validation'):
        assert result[subset]['count'] == 8
        assert result[subset]['max'] < 1e-8
        assert result[subset]['median'] <= result[subset]['p95'] <= result[subset]['max']
    assert result['residuals_px'].shape == (16,)
    for a, b in zip(args, saved):
        np.testing.assert_array_equal(a, b)
    wrong = np.array([brown5(Q.T @ np.array([(p[0]-632)/710, (p[1]-358)/710, 1]),
                             args[4], np.zeros(5)) for p in args[1]])
    assert np.median(np.linalg.norm(wrong - args[1], axis=1)) > 20


def test_validation_corruption_never_changes_training_fit(scene):
    raw, rect, K, d, Kr, mask, _ = scene
    baseline = fit_ray_rotation(raw, rect, K, d, Kr, mask)
    changed = rect.copy()
    changed[~mask] += [14., -9.]
    result = fit_ray_rotation(raw, changed, K, d, Kr, mask)
    np.testing.assert_array_equal(result['Q'], baseline['Q'])
    assert result['validation']['median'] == pytest.approx(math.hypot(14, 9), abs=1e-8)


@pytest.mark.parametrize('field,value', [
    (0, np.zeros((5, 2))), (0, np.full((16, 2), np.nan)),
    (0, np.full((16, 2), 1e308)), (1, np.zeros((16, 3))),
    (2, np.eye(3)), (2, np.diag([-1., 2., 1.])),
    (2, [[2, .1, 0], [0, 2, 0], [0, 0, 1]]),
    (2, [[2, 0, 0], [.1, 2, 0], [0, 0, 1]]),
    (2, [[2, 0, 0], [0, 2, 0], [0, .1, 1]]),
    (3, [0]*6), (3, [np.inf]*5), (3, [1e308]*5),
    (5, np.ones(16, dtype=int)), (5, np.ones(15, dtype=bool)),
    (5, np.ones(16, dtype=bool)), (5, np.arange(16) < 2),
])
def test_bad_fit_inputs_rejected(scene, field, value):
    args = list(scene[:6])
    args[field] = value
    # Identity K is a valid camera; this case instead creates a non-pinhole last row.
    if field == 2 and np.array_equal(value, np.eye(3)):
        args[field] = np.diag([1., 1., 2.])
    with pytest.raises(ValueError):
        fit_ray_rotation(*args)


@pytest.mark.parametrize('which', [0, 1])
@pytest.mark.parametrize('shape', ['duplicate', 'collinear'])
def test_degenerate_bearings_rejected(scene, which, shape):
    args = list(scene[:6])
    args[which] = np.tile([640., 360.], (16, 1))
    if shape == 'collinear':
        args[which][:, 0] += np.arange(16)
    # Use zero distortion so a pixel line is a bearing plane.
    args[3] = np.zeros(5)
    with pytest.raises(ValueError):
        fit_ray_rotation(*args)


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def test_rig_direction_candidates_and_inversion():
    Ql = rotation([.04, -.025, .018])
    R = rotation([-.012, .022, .031])
    t = np.array([.12, -.004, .007])
    Qr = Ql @ R
    sdk = rigid(R, t)
    rect = rigid(np.eye(3), Ql @ t)
    saved = [x.copy() for x in (Ql, Qr, sdk, rect)]
    result = compare_rig_hypotheses(Ql, Qr, sdk, rect)
    assert result['status'] == 'diagnostic_unqualified'
    direct, inverse = result['direct_right_pose'], result['inverse_mapping']
    assert direct['orientation_error_rad'] < 3e-8
    assert direct['translation_error'] < 1e-14
    assert direct['translation_alignment'] == pytest.approx(1.)
    assert inverse['orientation_error_rad'] > .05
    assert inverse['translation_error'] > .2
    flipped = compare_rig_hypotheses(Ql, Qr, np.linalg.inv(sdk), rect)
    assert flipped['inverse_mapping']['translation_error'] < 1e-14
    assert flipped['inverse_mapping']['orientation_error_rad'] < 3e-8
    for a, b in zip((Ql, Qr, sdk, rect), saved):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('field,value', [
    (0, np.diag([1., 1., -1.])), (1, np.ones((3, 3))),
    (2, np.eye(4)), (2, np.full((4, 4), 1e308)),
    (2, rigid(np.diag([1., 1., -1.]), [.12, 0, 0])),
    (3, rigid(np.eye(3), [0, 0, 0])), (3, np.full((4, 4), np.nan)),
])
def test_invalid_rig_rejected(field, value):
    args = [np.eye(3), np.eye(3), rigid(np.eye(3), [.12, 0, 0]),
            rigid(np.eye(3), [.12, 0, 0])]
    args[field] = value
    with pytest.raises(ValueError):
        compare_rig_hypotheses(*args)


def test_full_rectified_rotation_is_used():
    Ql, Qr = rotation([.12, -.04, .025]), rotation([-.07, .03, -.02])
    R, t = rotation([.02, .04, -.01]), np.array([.12, .002, -.003])
    rect_R = Ql @ R @ Qr.T
    good = compare_rig_hypotheses(Ql, Qr, rigid(R, t), rigid(rect_R, Ql @ t))
    wrong = compare_rig_hypotheses(Ql, Qr, rigid(R, t), rigid(np.eye(3), Ql @ t))
    assert good['direct_right_pose']['orientation_error_rad'] < 3e-8
    assert wrong['direct_right_pose']['orientation_error_rad'] > .1


@pytest.mark.parametrize('length', [0, 4, 8, 12])
def test_supported_distortion_lengths(scene, length):
    raw, rect, K, d, Kr, mask, Q = scene
    coefficients = np.zeros(length)
    coefficients[:min(length, 5)] = d[:min(length, 5)]
    scalar_d = np.zeros(5)
    scalar_d[:min(length, 5)] = coefficients[:min(length, 5)]
    rays = np.array([[x, y, 1.] for y in [-.33, -.08, .21, .37]
                     for x in [-.45, -.17, .13, .42]])
    raw = np.array([brown5(p, K, scalar_d) for p in rays])
    result = fit_ray_rotation(raw, rect, K, coefficients, Kr, mask)
    np.testing.assert_allclose(result['Q'], Q, atol=2e-11)


def test_negative_heldout_depth_is_rejected():
    Q = rotation([0, 1.3, 0])
    rays = np.array([[-.8, -.2, 1.], [-.6, .3, 1.], [-.4, -.1, 1.],
                     [1., -.3, 1.], [.9, .1, 1.], [1.1, .2, 1.]])
    rect = np.array([brown5(Q @ p, np.eye(3), np.zeros(5)) for p in rays])
    with pytest.raises(ValueError, match='depth'):
        fit_ray_rotation(rays[:, :2], rect, np.eye(3), [], np.eye(3), np.arange(6) < 3)


def test_float32_rig_rotations_project_explicitly_without_mutating_metadata():
    R, rect_R = rotation([.17, -.11, .08]), rotation([-.13, .07, -.03])
    sdk = rigid(R.astype(np.float32), [.12, -.004, .007])
    rect = rigid(rect_R.astype(np.float32), [.12, 0, 0])
    saved = [sdk.copy(), rect.copy()]
    result = compare_rig_hypotheses(np.eye(3), np.eye(3), sdk, rect)
    assert result['status'] == 'diagnostic_unqualified'
    for name, value in (('sdk', sdk), ('rectified', rect)):
        stats = result[name + '_rotation_normalization']
        original = value[:3, :3]
        assert stats['representation_budget'] == 1e-6
        assert stats['orthogonality_max'] == pytest.approx(
            np.max(np.abs(original.T @ original - np.eye(3))))
        assert 1e-10 < stats['orthogonality_max'] < 1e-6
        assert stats['input_determinant'] == pytest.approx(np.linalg.det(original))
        assert 0 < stats['projection_delta_max'] < 1e-6
    expected = math.acos(np.clip((np.trace(rect_R.T @ R) - 1) / 2, -1, 1))
    assert result['direct_right_pose']['orientation_error_rad'] == pytest.approx(expected, abs=1e-7)
    for a, b in zip((sdk, rect), saved):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('field', [2, 3])
@pytest.mark.parametrize('bad_R', [np.diag([1.000001, 1., 1.]),
                                  np.diag([1., 1., -1.])])
def test_rig_representation_budget_rejects_nonrigid_and_reflections(field, bad_R):
    args = [np.eye(3), np.eye(3), rigid(np.eye(3), [.12, 0, 0]),
            rigid(np.eye(3), [.12, 0, 0])]
    args[field] = rigid(bad_R, [.12, 0, 0])
    with pytest.raises(ValueError):
        compare_rig_hypotheses(*args)


@pytest.mark.parametrize('field', [0, 1])
def test_q_rotation_remains_strict(field):
    args = [np.eye(3), np.eye(3), rigid(np.eye(3), [.12, 0, 0]),
            rigid(np.eye(3), [.12, 0, 0])]
    args[field] = rotation([.17, -.11, .08]).astype(np.float32)
    with pytest.raises(ValueError):
        compare_rig_hypotheses(*args)


def test_sdk_550_first_probe_rotation_is_admitted_as_unqualified():
    # Pinned metadata from the first recorded SDK 5.5.0 calibration probe.
    sdk = np.array([
        [.99999886751174927, .00057511741761118174, -.0014009603764861822, .11974000185728073],
        [-.0005788117996416986, .9999963641166687, -.0026380761992186308, -9.4671697297599167e-05],
        [.0013994380133226514, .0026388838887214661, .99999552965164185, 5.514690656127641e-06],
        [0., 0., 0., 1.],
    ])
    rect = rigid(np.eye(3), [.11974003911018372, 0., 0.])
    saved = sdk.copy()
    result = compare_rig_hypotheses(np.eye(3), np.eye(3), sdk, rect)
    assert result['status'] == 'diagnostic_unqualified'
    stats = result['sdk_rotation_normalization']
    assert stats['orthogonality_max'] == pytest.approx(2.8474633539943284e-8)
    assert stats['input_determinant'] == pytest.approx(1.0000000163243448)
    assert stats['projection_delta_max'] < 2e-8
    for name in ('direct_right_pose', 'inverse_mapping'):
        assert np.isfinite(list(result[name].values())).all()
    np.testing.assert_array_equal(sdk, saved)
