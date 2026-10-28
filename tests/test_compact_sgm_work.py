import numpy as np
import pytest
from benchmarks.compact_sgm_work import estimate_eye, rounded_lanes


def test_zero_and_wave_rounding():
    np.testing.assert_array_equal(rounded_lanes(np.array([0, 1, 32, 33, 64, 65]), 32), [0, 32, 32, 64, 64, 96])
    np.testing.assert_array_equal(rounded_lanes(np.array([0, 1, 64, 65]), 64), [0, 64, 64, 128])


def test_inclusive_bounds_missing_and_unsupported_fall_back():
    valid = np.array([[1, 1, 0, 1]], dtype=bool)
    support = np.array([[1, 1, 1, 0]], dtype=bool)
    lower = np.array([[0, 7, 0, 1]])
    upper = np.array([[0, 9, 0, 1]])
    out = estimate_eye(valid, support, lower, upper, 10)
    assert out['active_candidates'] == 24
    assert out['dense_candidates'] == 40
    assert out['fallback_pixels'] == 2
    assert out['unsupported_pixels'] == 1
    assert out['wave32_lanes'] == 128
    assert out['wave64_lanes'] == 256


def test_full_and_invalid_intervals():
    x = np.ones((1, 1), bool)
    assert estimate_eye(x, x, np.array([[0]]), np.array([[127]]), 128)['active_candidates'] == 128
    for lo, hi in [(-1, 0), (1, 0), (0, 128)]:
        with pytest.raises(ValueError):
            estimate_eye(x, x, np.array([[lo]]), np.array([[hi]]), 128)


def test_truth_candidate_inclusive_tolerance_and_exclusive_upper():
    from benchmarks.compact_sgm_work import truth_coverage
    lo = np.array([[3, 3, 3, 3, 0, 0]])
    hi = np.array([[5, 5, 5, 5, 128, 128]])
    truth = np.array([[2., 5., 5.0001, 1.9999, 127.9, 50.]])
    support = np.array([[1, 1, 1, 1, 1, 0]], bool)
    out = truth_coverage(lo, hi, truth, support, {'all': np.ones_like(support), 'thin': np.array([[1, 1, 1, 0, 0, 1]])})
    assert out['all']['eligible'] == 5
    assert out['all']['excluded'] == 1
    assert out['all']['unrepresentable'] == 2
    assert out['thin']['eligible'] == 3
    assert out['thin']['unrepresentable'] == 1
    assert out['thin']['excluded'] == 1
