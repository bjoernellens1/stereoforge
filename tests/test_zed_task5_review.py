"""Independent Task 5 functional regressions, written without production edits."""
from copy import deepcopy
import numpy as np
import pytest

from tests.test_zed_comparison_schema import record, available
from tests.test_zed_evaluate_temporal import sequence, run
from tests.test_zed_epipolar import envelope
from benchmarks.zed_comparison.epipolar import evaluate_epipolar, compare_epipolar_profiles
from benchmarks.zed_comparison.evaluate_temporal import _Moments


def frozen_lane(record, root, **kwargs):
    data = sequence(record, root, **kwargs)
    p = data[2]
    p['frozen'] = True
    p['freeze_parameters'].update(
        region_definitions={r: dict(description='declared region') for r in p['required_regions']},
        zero_reference_error=dict(method='absolute_margin', absolute_mae_margin_m=.01),
        uncertainty=dict(method='analytic bounds', confidence_level=.95, registration_max_m=.01, timestamp_max_ns=10),
        statistics=dict(method='controlled', unit='scene_sequence', min_scenes=1, min_sequences=1),
        partition_digest=data[3]['partition_digest'])
    return data


def test_review_measured_temporal_requires_physical_reference(record, tmp_path):
    data = frozen_lane(record, tmp_path)
    data[1]['provenance']['kind'] = 'measured'
    result = run(data, tmp_path)
    assert not result['quality_qualified'], result
    assert 'missing_physical_reference_metadata' in result['reasons']


def test_review_stationary_grid_rejects_changing_supplied_pose(record, tmp_path):
    data = frozen_lane(record, tmp_path, explicit=True)
    data[3]['sequences'][0]['scene_motion'] = 'dynamic'
    for i, (r, f) in enumerate(zip(data[0], data[3]['sequences'][0]['frames'])):
        pose = np.eye(4)
        pose[0, 3] = i
        f['world_from_left'] = available(dict(matrix=pose.tolist(), capture=r['timestamps']['capture']))
    with pytest.raises(ValueError, match='stationary'):
        run(data, tmp_path)


def test_review_missing_method_captures_cannot_qualify(record, tmp_path):
    data = frozen_lane(record, tmp_path)
    candidates = deepcopy(data[0][:2])
    for candidate in candidates:
        candidate['implementation']['settings']['variant'] = 'only-easy-first-two-frames'
    data[0].extend(candidates)
    result = run(data, tmp_path)
    assert not result['quality_qualified'], result
    assert 'method_input_set_mismatch' in result['reasons']


def test_review_malformed_epipolar_summary_rejected():
    a = evaluate_epipolar(np.array([[1., 2., 1., 1.]]), envelope())
    b = deepcopy(a)
    b['profile'] = 'effective'
    b['vertical']['absolute_p95_px'] = float('nan')
    with pytest.raises(ValueError):
        compare_epipolar_profiles([a, b])


def test_review_finite_epipolar_inputs_cannot_overflow_delta():
    a = evaluate_epipolar(np.array([[1., 2., 1., 1.]]), envelope())
    b = deepcopy(a)
    for result, signed in ((a, 1e308), (b, -1e308)):
        result['vertical'].update(signed_median_px=signed, absolute_median_px=1e308,
            absolute_p95_px=1e308, absolute_max_px=1e308,
            positive_count=int(signed > 0), negative_count=int(signed < 0), zero_count=0)
    with pytest.raises(ValueError, match='profile delta'):
        compare_epipolar_profiles([a, b])


def test_review_moments_fastpath_after_reset_and_reordered_updates():
    state = _Moments(2)
    ids = np.array([3, 7], dtype=np.int64)
    state.update(ids, np.array([999., 999.]))
    state.reset()
    state.update(ids, np.array([1., 10.]))
    state.update(ids[::-1], np.array([12., 3.]))
    state.update(ids, np.array([5., 14.]))
    out = state.report()
    assert out['track_count'] == 2 and out['observation_count'] == 6
    assert out['mean_track_bias_m'] == pytest.approx(7.5)
    assert out['mean_track_error_std_m'] == pytest.approx(2.)
