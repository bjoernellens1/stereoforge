"""Independent analytic world truth; no matching implementation is consulted."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import refinement_quality_fixtures as truth
import compare_daxa_preprocess as native
from stereoforge.types import Calibration, CameraModel


def plane(z=3., **extra):
    return dict(id=1, slope_x=0., slope_y=0., intercept_m=z,
                bounds_xy_m=[-100., 100., -100., 100.], texture='textured', region='far', **extra)


def test_plane_and_independent_cross_eye_geometry():
    cal = truth.synthetic_calibration(64, 36)
    scene = dict(schema=truth.SCENE_SCHEMA, seed=42, surfaces=[plane()])
    model = truth.rectified_models(cal)
    left = truth.render_grid_truth(cal, model, scene, 'left', disparities=128)
    right = truth.render_grid_truth(cal, model, scene, 'right', disparities=128)
    assert np.all(left['axial_z_m'] == 3.)
    assert np.all(right['axial_z_m'] == 3.)
    expected = model['P1'][0][0] * .12 / 3.
    np.testing.assert_allclose(left['disparity'], expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(right['disparity'], expected, rtol=0, atol=1e-13)
    assert left['off_image'][:, 0].all() and right['off_image'][:, -1].all()
    assert left['visible'][:, 2:-2].all() and not left['occluded'].any()


def test_principal_point_offset_and_domain_are_explicit():
    cal = truth.synthetic_calibration(64, 36)
    scene = dict(schema=truth.SCENE_SCHEMA, seed=42, surfaces=[plane()])
    model = truth.rectified_models(cal)
    model['P2'][0][2] += .75
    expected = model['P1'][0][0] * .12 / 3. - .75
    for eye in ('left', 'right'):
        result = truth.render_grid_truth(cal, model, scene, eye, disparities=1)
        np.testing.assert_allclose(result['disparity'], expected, rtol=0, atol=1e-13)
        assert result['out_of_range'].all()


def test_nearest_intersection_occlusion_and_opposite_eyes():
    cal = truth.synthetic_calibration(128, 72)
    near = dict(id=2, slope_x=0., slope_y=0., intercept_m=.7,
                bounds_xy_m=[-.1, .1, -.4, .4], texture='textured', region='thin')
    scene = dict(schema=truth.SCENE_SCHEMA, seed=42, surfaces=[plane(), near])
    model = truth.rectified_models(cal)
    a = truth.render_grid_truth(cal, model, scene, 'left')
    b = truth.render_grid_truth(cal, model, scene, 'right')
    assert a['occluded'].any() and b['occluded'].any()
    assert np.all(a['surface_id'][a['occluded']] == 1)
    assert np.all(b['surface_id'][b['occluded']] == 1)
    assert np.all(a['axial_z_m'][a['surface_id'] == 2] == .7)
    assert not np.array_equal(a['surface_id'], b['surface_id'])
    assert not (a['visible'] & a['occluded']).any()


def test_nonidentity_world_pose_and_distorted_original_rays():
    cal = truth.synthetic_calibration(64, 36)
    c = cal.left
    distorted = CameraModel(c.width, c.height, c.fx, c.fy, c.cx, c.cy, (.03, -.002, .001, -.001, 0.))
    cal = Calibration(distorted, distorted, cal.right_to_left)
    scene = dict(schema=truth.SCENE_SCHEMA, seed=7, surfaces=[plane()])
    pose = np.eye(4); pose[:3, 3] = [.2, -.1, .5]
    rendered = truth.render_original(cal, scene, 'left', world_from_left=pose)
    assert np.all(rendered['axial_z_m'] == 2.5)
    points = rendered['points_world']
    np.testing.assert_allclose(points[..., 2], 3., atol=1e-14)
    # Nonidentity orientation is exercised with a known centre ray.
    angle = .1
    pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    origins = np.array([[.2, -.1, .5]])
    directions = np.array([[np.sin(angle), 0., np.cos(angle)]])
    hit = truth.render_rays(scene, origins, directions)
    np.testing.assert_allclose(hit['points_world'][0], origins[0] + directions[0] * (2.5 / np.cos(angle)))
    rotated = truth.render_original(cal, scene, 'right', world_from_left=pose)
    np.testing.assert_allclose(rotated['points_world'][..., 2], 3., rtol=0, atol=1e-14)
    right_pose = pose @ cal.right_to_left
    camera_points = (rotated['points_world'] - right_pose[:3, 3]) @ right_pose[:3, :3]
    np.testing.assert_allclose(camera_points[..., 2], rotated['axial_z_m'], rtol=0, atol=1e-14)
    import cv2
    projected = cv2.projectPoints(camera_points.reshape(-1, 3), np.zeros(3), np.zeros(3), distorted.K,
                                 np.asarray(distorted.distortion))[0].reshape(36, 64, 2)
    y, x = np.indices((36, 64))
    np.testing.assert_allclose(projected, np.stack((x, y), axis=-1), rtol=0, atol=1e-7)


def test_ties_missing_surfaces_and_bad_rays():
    scene = dict(schema=truth.SCENE_SCHEMA, seed=42, surfaces=[dict(plane(), id=7), plane()])
    hit = truth.render_rays(scene, np.array([0., 0., 0.]), np.array([[0., 0., 1.], [0., 0., -1.], [1., 0., 0.]]))
    assert hit['surface_id'].tolist() == [1, 0, 0]
    assert hit['valid'].tolist() == [True, False, False]
    assert np.isnan(hit['points_world'][1:]).all()
    for ray in ([[0., 0., 0.]], [[np.nan, 0., 1.]], [[1., 2.]]):
        with pytest.raises(ValueError):
            truth.render_rays(scene, np.zeros(3), ray)


def test_mixed_footprint_is_separate_from_centre_geometry():
    ids = np.array([[1, 2], [1, 2]], np.int32)
    mx = np.array([[.5, 0., -.25]], np.float32)
    my = np.array([[.5, 0., 0.]], np.float32)
    centre = np.array([[1, 1, 1]], np.int32)
    mixed = truth._mixed_footprints(mx, my, ids, centre)
    assert mixed.tolist() == [[True, False, False]]
    # Unsupported sampling and geometric mixing are deliberately independent.
    assert native.support_mask(mx, my, 2, 2).tolist() == [[1, 1, 0]]
    assert centre.tolist() == [[1, 1, 1]]


@pytest.mark.parametrize('rotated', [False, True])
def test_thin_patches_have_requested_left_phase_and_centreline_width(rotated):
    # Project world points independently of the fixture construction. In
    # particular, adding phase*z/fx to an arbitrary world x does NOT fix phase.
    cal = truth.synthetic_calibration(1280, 720)
    if rotated:
        a = .027
        pose = np.array(cal.right_to_left, copy=True)
        pose[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        cal = Calibration(cal.left, cal.right, pose)
    model = truth.rectified_models(cal)
    scene = truth.default_scene(cal)
    def project(point, eye):
        raw = np.asarray(point, float)
        if eye == 'right':
            raw = np.linalg.solve(cal.right_to_left, np.r_[raw, 1.])[:3]
        q = np.asarray(model['R1' if eye == 'left' else 'R2']) @ raw
        p = np.asarray(model['P1' if eye == 'left' else 'P2'])
        return np.array([p[0, 0]*q[0]/q[2]+p[0, 2], p[1, 1]*q[1]/q[2]+p[1, 2]])
    thin = [p for p in scene['surfaces'] if p['region'] == 'thin']
    for p, (width, phase) in zip(thin, ((1, 0.), (2, .25), (4, .5))):
        centre = p['full_grid_design']['world_centre_m']
        actual = project(centre, 'left')
        phase_error = ((actual[0] - phase + .5) % 1.) - .5
        assert abs(phase_error) < 1e-12
        lo, hi, _, _ = p['bounds_xy_m']
        left_boundary = project([lo, centre[1], .7], 'left')
        right_boundary = project([hi, centre[1], .7], 'left')
        assert abs(right_boundary[0] - left_boundary[0] - width) < 1e-12
        assert p['full_grid_design']['requested_left_phase_pixels'] == phase
        assert p['full_grid_design']['requested_left_centreline_width_pixels'] == width
        for eye in ('left', 'right'):
            projected = project(centre, eye)
            recorded = p['full_grid_design']['actual_projection'][eye]
            np.testing.assert_allclose([recorded['centre_u'], recorded['centre_v']], projected, rtol=0, atol=1e-12)
            assert abs(recorded['phase_pixels'] - projected[0] % 1.) < 1e-12
        assert 'target_phase_pixels' not in p


def test_half_pixel_truth_is_recomputed_not_resized():
    cal = truth.synthetic_calibration(64, 36)
    scene = dict(schema=truth.SCENE_SCHEMA, seed=1, surfaces=[dict(plane(), slope_x=.3)])
    full = truth.rectified_models(cal)
    half = truth.scaled_models(full, 32, 18)
    high = truth.render_grid_truth(cal, full, scene, 'left')
    low = truth.render_grid_truth(cal, half, scene, 'left', disparities=64)
    assert not np.array_equal(low['axial_z_m'], high['axial_z_m'][::2, ::2])
    fx = half['P1'][0][0]; cx = half['P1'][0][2]
    expected = 3. / (1 - .3 * (np.arange(32) - cx) / fx)
    np.testing.assert_allclose(low['axial_z_m'], np.broadcast_to(expected, (18, 32)), rtol=0, atol=1e-14)


def test_unequal_distorted_original_models_actual_rotations_and_cross_epipolar_geometry():
    left = CameraModel(64, 36, 28., 27., 30.7, 17.2, (.01, -.002, .001, -.001, 0.))
    right = CameraModel(64, 36, 29., 28., 31.3, 17.7, (-.01, .001, -.001, .002, 0.))
    angle = .03
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    pose[:3, 3] = [.12, .002, .001]
    cal = Calibration(left, right, pose)
    model = truth.rectified_models(cal)
    assert not np.array_equal(model['R1'], model['R2'])
    scene = dict(schema=truth.SCENE_SCHEMA, seed=3, surfaces=[dict(plane(), slope_x=.1)])
    for eye in ('left', 'right'):
        result = truth.render_grid_truth(cal, model, scene, eye)
        y = np.indices((36, 64))[0]
        np.testing.assert_allclose(result['cross_v'], y, rtol=0, atol=1e-10)
        expected = model['P1'][0][0] * cal.baseline_m / result['axial_z_m'] + model['P1'][0][2] - model['P2'][0][2]
        np.testing.assert_allclose(result['disparity'], expected, rtol=0, atol=1e-10)
        assert result['visible'].any() and not result['occluded'].any()


def test_export_family_hashes_original_bytes_and_no_truth_resizing(tmp_path, monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, 'resize', lambda *a, **k: pytest.fail('truth must be ray rendered'))
    cal = truth.synthetic_calibration(64, 36)
    scene = truth.default_scene(cal)
    before_scene = native.canonical(scene)
    before_calibration = native.canonical(cal.to_dict())
    first = truth.export_scene_family(tmp_path / 'a', width=64, height=36, calibration=cal, scene=scene)
    second = truth.export_scene_family(tmp_path / 'b', width=64, height=36)
    assert native.canonical(scene) == before_scene
    assert native.canonical(cal.to_dict()) == before_calibration
    def files(root):
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert files(first.parent) == files(second.parent)
    family = native.strict_json(first.read_text())
    profiles = family['profiles']
    assert set(profiles) == {'full', 'half'}
    captures = []
    for name, p in profiles.items():
        manifest_path = first.parent / p['native_manifest']['file']
        manifest, _ = native.validate_fixture(manifest_path)
        captures.append(native.fixture_file(manifest_path.parent, manifest['capture']).read_bytes())
        gt = native.strict_json((first.parent / p['truth_manifest']['file']).read_text())
        for eye in ('left', 'right'):
            for entry in gt['eyes'][eye].values():
                path = native.fixture_file((first.parent / p['truth_manifest']['file']).parent, entry)
                assert path.stat().st_size == np.prod(entry['shape']) * np.dtype(entry['dtype']).itemsize
    assert captures[0] == captures[1]
    with pytest.raises(ValueError, match='fresh'):
        truth.export_scene_family(first.parent, width=64, height=36)


@pytest.mark.parametrize('change', [dict(intercept_m=0), dict(slope_x=float('nan')), dict(id=0), dict(bounds_xy_m=[1, -1, 0, 1]), dict(texture='unknown')])
def test_bad_geometry_fails(change):
    scene = dict(schema=truth.SCENE_SCHEMA, seed=1, surfaces=[dict(plane(), **change)])
    with pytest.raises(ValueError):
        truth.render_rays(scene, np.zeros((1, 3)), np.array([[0., 0., 1.]]))


def test_invalid_export_and_unsafe_destination(tmp_path):
    for dims in ((63, 36), (64, 35), (0, 0)):
        with pytest.raises(ValueError):
            truth.export_scene_family(tmp_path / 'bad', width=dims[0], height=dims[1])
    target = tmp_path / 'real'; target.mkdir()
    link = tmp_path / 'link'; link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        truth.export_scene_family(link / 'child', width=64, height=36)


def test_photometric_is_bounded_documented_and_does_not_change_geometry(tmp_path):
    a = truth.export_scene_family(tmp_path / 'normal', width=64, height=36)
    b = truth.export_scene_family(tmp_path / 'exposure', width=64, height=36,
                                 photometric={'right': dict(gain=2., offset=30.)})
    original, exposure = [native.strict_json(p.read_text()) for p in (a, b)]
    assert original['original_capture']['sha256'] != exposure['original_capture']['sha256']
    assert exposure['photometric']['clipped_pixels']['right'] > 0
    for profile in ('full', 'half'):
        ma = native.strict_json((a.parent / original['profiles'][profile]['truth_manifest']['file']).read_text())
        mb = native.strict_json((b.parent / exposure['profiles'][profile]['truth_manifest']['file']).read_text())
        assert ma['eyes'] == mb['eyes']
    for setting in ({'right': {'gain': 0., 'offset': 0.}}, {'other': {}}, {'left': {'gain': 1.}}):
        with pytest.raises(ValueError):
            truth.export_scene_family(tmp_path / 'bad-exposure', width=64, height=36, photometric=setting)
