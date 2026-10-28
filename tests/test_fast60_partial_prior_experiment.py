import numpy as np
import pytest
from benchmarks.fast60_partial_prior_experiment import partial_prior, merge_matches, validate_controls, validate_profile


def test_partial_prior_requires_two_taps_and_half_weight():
    d = np.full((2, 2), 10, np.float32)
    c = np.ones((2, 2), np.float32)
    valid = np.array([[1, 1], [0, 0]], np.uint32)
    p = partial_prior(d, c, valid)
    assert p.valid[1, 1]
    assert p.disparity[1, 1] == 20
    assert (p.lower[1, 1], p.upper[1, 1]) == (18, 22)
    assert not p.valid[2, 1]  # weights for the two valid taps total only 1/4
    valid[0, 1] = 0
    assert not partial_prior(d, c, valid).valid[1, 1]


def test_partial_prior_rejects_unreliable_taps_and_discontinuity():
    d = np.array([[10, 13], [10, 13]], np.float32)
    c = np.ones((2, 2), np.float32)
    v = np.ones((2, 2), np.uint32)
    assert not partial_prior(d, c, v).valid[1, 1]
    d[:] = 10
    c[0, 0] = .01
    c[0, 1] = .01
    assert not partial_prior(d, c, v).valid[1, 1]
    assert not partial_prior(d, np.ones_like(c), v).valid[0, 0]  # no border clamping


def test_merge_preserves_existing_prior_even_if_its_match_was_invalid():
    base_d = np.array([1, np.nan, np.nan], np.float32)
    base_v = np.array([1, 0, 0], bool)
    out_d, out_v = merge_matches(base_d, base_v, np.array([1, 1, 0], bool),
                                  np.array([5, 5, 5], np.float32), np.ones(3, bool))
    np.testing.assert_array_equal(out_v, [1, 0, 1])
    assert out_d[0] == 1 and np.isnan(out_d[1]) and out_d[2] == 5


def test_partial_prior_excludes_nonfinite_and_out_of_domain():
    d = np.full((2, 2), 10, np.float32)
    c = np.ones_like(d)
    v = np.ones((2, 2), np.uint32)
    d[0, 0], d[1, 0] = np.nan, 64
    assert not partial_prior(d, c, v).valid[1, 1]
    d[:] = 10
    c[0, 0], c[1, 0] = np.nan, 1.1
    assert not partial_prior(d, c, v).valid[1, 1]


def test_control_rejects_nonbinary_and_mismatched_census_support():
    actual = {k: np.zeros((2, 3, 4), np.uint32) for k in ('census_validity.u32', 'raw_validity.u32', 'validity.u32')}
    support = np.zeros((2, 3, 4), bool)
    validate_controls(actual, support)
    actual['raw_validity.u32'][0, 0, 0] = 2
    with pytest.raises(ValueError, match='nonbinary'):
        validate_controls(actual, support)
    actual['raw_validity.u32'][0, 0, 0] = 0
    support[0, 0, 0] = True
    with pytest.raises(ValueError, match='Census support'):
        validate_controls(actual, support)


def test_profile_rejects_tampered_setting_and_device():
    expected = {'schema': 'expected', 'radius': 2}
    frozen = {'report_identity': {'shader': 'abc'}, 'device_identity': {'device': 'gpu'}}
    report = {'schema': 'expected', 'radius': 2, 'shader': 'abc', 'device': {'device': 'gpu'}}
    validate_profile(report, expected, frozen)
    report['radius'] = 4
    with pytest.raises(ValueError, match='radius'):
        validate_profile(report, expected, frozen)
    report['radius'] = 2
    report['device']['device'] = 'other'
    with pytest.raises(ValueError, match='device'):
        validate_profile(report, expected, frozen)


def test_profile_rejects_type_forgery():
    expected = {'radius': 2, 'subpixel': False}
    frozen = {'report_identity': {'shader_version': 2}, 'device_identity': {'id': 1}}
    report = {'radius': 2, 'subpixel': False, 'shader_version': 2, 'device': {'id': 1}}
    for key, forged in [('radius', 2.0), ('subpixel', 0), ('shader_version', True)]:
        broken = {**report, key: forged}
        with pytest.raises(ValueError):
            validate_profile(broken, expected, frozen)
    with pytest.raises(ValueError):
        validate_profile({**report, 'device': {'id': True}}, expected, frozen)
