import copy

import numpy as np
import pytest

from benchmarks.physical_depth_v14 import native_reference as N


def factory():
    transform = np.eye(4)
    transform[:3, 3] = [.02, -.01, .1]
    return dict(schema='stereoforge.native_realsense_factory.v1', depth_scale_m=.001,
                depth=dict(width=3, height=2, fx=2., fy=2., ppx=1., ppy=0.,
                           model='distortion.none', coeffs=[0.] * 5),
                T_color_from_depth=transform.tolist())


def test_native_points_keep_axial_z_then_transform_to_color():
    rs = pytest.importorskip('pyrealsense2')
    depth = np.full((2, 3), 1000, np.uint16)
    points = N.native_points(depth, factory(), np.array([[1, 0], [2, 1]]), rs)
    np.testing.assert_allclose(points, [[0, 0, 1], [.5, .5, 1]], atol=1e-7)
    color = N.color_points(points, factory())
    np.testing.assert_allclose(color, [[.02, -.01, 1.1], [.52, .49, 1.1]], atol=1e-7)
    assert np.linalg.norm(points[1]) > points[1, 2]  # ray range differs from axial depth


def test_native_reference_does_not_clip_borders_or_fill_zero_depth():
    rs = pytest.importorskip('pyrealsense2')
    depth = np.full((2, 3), 1000, np.uint16)
    depth[0, 0] = 0
    pixels = np.array([[0, 0], [-1, 0], [3, 0], [1, 2], [1, 0]])
    points = N.native_points(depth, factory(), pixels, rs)
    assert np.isnan(points[:4]).all()
    np.testing.assert_allclose(points[4], [0, 0, 1])


@pytest.mark.parametrize('model', ['none', 'brown_conrady', 'inverse_brown_conrady'])
def test_recorded_distortion_matches_installed_sdk(model):
    rs = pytest.importorskip('pyrealsense2')
    f = factory()
    f['depth'].update(model='distortion.' + model, coeffs=[.05, -.01, .002, -.001, .001])
    intr = rs.intrinsics()
    for key in ('width', 'height', 'fx', 'fy', 'ppx', 'ppy', 'coeffs'):
        setattr(intr, key, f['depth'][key])
    intr.model = getattr(rs.distortion, model)
    expected = rs.rs2_deproject_pixel_to_point(intr, [2., 1.], 2.)
    actual = N.native_points(np.full((2, 3), 2000, np.uint16), f, np.array([[2, 1]]), rs)
    np.testing.assert_array_equal(actual[0], expected)


@pytest.mark.parametrize('change', ['shape', 'dtype', 'scale', 'fx', 'coeffs', 'fractional', 'model', 'schema'])
def test_invalid_geometry_is_rejected_before_sdk(change):
    f, depth, pixels = factory(), np.ones((2, 3), np.uint16), np.array([[1, 0]])
    if change == 'shape':
        depth = depth[:, :2]
    elif change == 'dtype':
        depth = depth.astype(np.float32)  # aligned metric preview must never enter here
    elif change == 'scale':
        f['depth_scale_m'] = float('nan')
    elif change == 'fx':
        f['depth']['fx'] = 0
    elif change == 'coeffs':
        f['depth']['coeffs'][0] = float('inf')
    elif change == 'fractional':
        pixels = np.array([[1.5, 0]])
    elif change == 'model':
        f['depth']['model'] = 'distortion.modified_brown_conrady'
    else:
        f['schema'] = 'legacy-color-aligned'
    with pytest.raises(ValueError):
        N.native_points(depth, f, pixels, None)


def test_color_mapping_rejects_nonrigid_transform_and_keeps_invalid_points():
    f = factory()
    points = np.array([[0., 0., 1.], [np.nan] * 3])
    assert np.isnan(N.color_points(points, f)[1]).all()
    for row, col, value in [(0, 0, 2), (3, 0, 1), (0, 3, np.nan)]:
        bad = copy.deepcopy(f)
        bad['T_color_from_depth'][row][col] = value
        with pytest.raises(ValueError, match='rigid'):
            N.color_points(points, bad)
