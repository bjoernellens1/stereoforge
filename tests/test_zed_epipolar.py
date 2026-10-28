import copy
import hashlib

import numpy as np
import pytest

from benchmarks.zed_comparison.epipolar import evaluate_epipolar, compare_epipolar_profiles


def envelope():
    return dict(schema='stereoforge.zed_comparison.epipolar.v1', profile='analytic', space='rectified',
                geometry=dict(width=10, height=10, fx=10, fy=10, baseline_m=.1, cx_left=5,
                              cx_right=5, projection='perspective', depth_convention='axial_z',
                              units='m', coordinate_frame='left', pixel_centres='integer_centres',
                              reference_camera='left', coordinate_axes='x_right_y_down_z_forward'),
                calibration_sha256='a'*64, original_pixel_identity='b'*64,
                correspondence_provenance=dict(method='manual', selection='all',
                    population_identity=dict(status='available', value='c'*64)))


def test_identity_and_signed_offset():
    result = evaluate_epipolar(np.array([[2, 3, 1, 1], [3, 1, 1, 3]]), envelope())
    assert result['vertical']['signed_median_px'] == 0
    assert result['vertical']['absolute_median_px'] == 2
    assert result['horizontal']['signed_median_px'] == 1.5
    assert result['metric_truth'] is result['quality_qualified'] is False
    assert result['matches_sha256'] == hashlib.sha256(np.array([[2,3,1,1],[3,1,1,3]], dtype='<f8').tobytes()).hexdigest()


def test_nearest_rank_and_empty():
    points = np.zeros((20, 4)); points[:, 1] = np.arange(20)/3
    assert evaluate_epipolar(points, envelope())['vertical']['absolute_p95_px'] == 6
    assert evaluate_epipolar(np.empty((0, 4)), envelope())['status'] == 'unavailable'


@pytest.mark.parametrize('points', [np.zeros((1,3)), np.array([[True]*4]), np.array([[0,0,np.nan,0]]), np.array([[-.5001,0,0,0]]), np.array([[9.5,0,0,0]])])
def test_invalid_matches(points):
    with pytest.raises(ValueError): evaluate_epipolar(points, envelope())


def test_bounds_and_strict_geometry():
    evaluate_epipolar(np.full((1,4), -.5), envelope())
    for mutation in [lambda e: e.update(space='raw'), lambda e: e['geometry'].update(projection='fisheye'), lambda e: e.update(extra=1)]:
        e = envelope(); mutation(e)
        with pytest.raises(ValueError): evaluate_epipolar(np.zeros((1,4)), e)


def test_comparison_requires_matched_population():
    e = envelope(); a = evaluate_epipolar(np.array([[1,2,1,1]]), e)
    e['profile'] = 'effective'; b = evaluate_epipolar(np.array([[1,1,1,1]]), e)
    assert compare_epipolar_profiles([a,b])['deltas']['status'] == 'available'
    e['correspondence_provenance']['selection'] = 'other'
    b = evaluate_epipolar(np.array([[1,1,1,1]]), e)
    assert compare_epipolar_profiles([a,b])['deltas']['status'] == 'unavailable'


def test_hash_normalizes_endian_integer_and_signed_zero():
    a = evaluate_epipolar(np.array([[0, 1, 2, 3]], dtype='>f8'), envelope())
    b = evaluate_epipolar(np.array([[-0., 1, 2, 3]], dtype='<f8'), envelope())
    c = evaluate_epipolar(np.array([[0, 1, 2, 3]], dtype=np.int32), envelope())
    assert a['matches_sha256'] == b['matches_sha256'] == c['matches_sha256']


@pytest.mark.parametrize('change', [
    lambda e: e.update(calibration_sha256='A'*64),
    lambda e: e['geometry'].update(width=True),
    lambda e: e['geometry'].update(fx=float('inf')),
    lambda e: e['correspondence_provenance'].update(method=''),
    lambda e: e['correspondence_provenance'].update(extra='bad'),
])
def test_invalid_envelope(change):
    e = envelope(); change(e)
    with pytest.raises(ValueError): evaluate_epipolar(np.zeros((1,4)), e)


def test_unknown_population_and_different_count_no_delta():
    e = envelope()
    e['correspondence_provenance']['population_identity'] = dict(status='unavailable', reason='unknown')
    a = evaluate_epipolar(np.zeros((1,4)), e)
    assert compare_epipolar_profiles([a,a])['deltas']['status'] == 'unavailable'
    e = envelope()
    a = evaluate_epipolar(np.zeros((1,4)), e)
    b = evaluate_epipolar(np.zeros((2,4)), e)
    assert compare_epipolar_profiles([a,b])['deltas']['status'] == 'unavailable'


@pytest.mark.parametrize('mutate', [
    lambda r: r['vertical'].update(absolute_p95_px=float('nan')),
    lambda r: r['horizontal'].update(signed_median_px=True),
    lambda r: r.update(count=True),
    lambda r: r.update(count=0),
    lambda r: r.update(status='invented'),
    lambda r: r.pop('geometry'),
    lambda r: r.update(extra=1),
    lambda r: r.update(diagnostic=False),
    lambda r: r.update(matches_sha256='bad'),
    lambda r: r['vertical'].update(count=2),
    lambda r: r['vertical'].update(zero_count=2),
    lambda r: r['vertical'].update(absolute_median_px=-1),
    lambda r: r['vertical'].update(absolute_p95_px=2),
    lambda r: r['vertical'].update(signed_median_px=2),
    lambda r: r['vertical'].pop('absolute_max_px'),
])
def test_comparison_rejects_malformed_result(mutate):
    result = evaluate_epipolar(np.array([[1., 2., 1., 1.]]), envelope())
    mutate(result)
    with pytest.raises(ValueError): compare_epipolar_profiles([result])


def test_comparison_valid_empty_and_invalid_unavailable():
    result = evaluate_epipolar(np.empty((0,4)), envelope())
    assert compare_epipolar_profiles([result, result])['deltas']['status'] == 'unavailable'
    for mutation in ({'count': 1}, {'reason': ''}, {'vertical': {}}, {'reason': None}):
        bad = copy.deepcopy(result); bad.update(mutation)
        with pytest.raises(ValueError): compare_epipolar_profiles([bad])
