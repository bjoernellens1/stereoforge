
import cv2
import numpy as np
import pytest

from benchmarks.physical_depth_v14 import reference_sgm as R
from benchmarks.physical_depth_v14.calibrate_sigma import sigma_d


def _pair(shift: float):
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((180, 320)) * 255).astype(np.float32), (0, 0), 1.5)
    right = cv2.warpAffine(img, np.float32([[1, 0, -shift], [0, 1, 0]]), (320, 180), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(img, 0, 255).astype(np.uint8), np.clip(right, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("shift", [12.0, 12.3, 12.5])
def test_recovers_known_subpixel_shift_without_pixel_locking(shift):
    out = R.estimate(*_pair(shift), R.Params(disparities=32))
    sel = out["valid"].copy()
    sel[:, :40] = False
    assert sel.mean() > 0.7
    err = out["disparity"][sel] - shift
    assert abs(np.median(err)) < 0.15
    assert np.sqrt(np.mean(err ** 2)) < 0.25


def test_flat_region_gets_low_confidence_and_textured_region_high():
    left, right = _pair(10.0)
    left[:, 160:] = 128
    right[:, 160:] = 128
    out = R.estimate(left, right, R.Params(disparities=32))
    assert out["uniqueness"][:, 60:140].mean() > 4 * max(out["uniqueness"][:, 190:300].mean(), 1e-3)


def test_sigma_model_is_monotone_in_uniqueness():
    coef = np.array([-1.9, -0.78, -0.01])
    u = np.array([0.05, 0.2, 1.0, 5.0])
    s = sigma_d(u, np.ones_like(u), coef)
    assert np.all(np.diff(s) < 0) and s[-1] >= 0.03


def test_right_reference_is_independently_aggregated(monkeypatch):
    calls = []
    aggregate = R.aggregate

    def record(volume, gray, params):
        calls.append(gray.copy())
        return aggregate(volume, gray, params)

    monkeypatch.setattr(R, "aggregate", record)
    left, right = _pair(8)
    out = R.estimate(left[:12, :48], right[:12, :48], R.Params(disparities=16))
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1], right[:12, :48][:, ::-1])
    assert out["right_disparity"].shape == (12, 48)


def test_lr_rejects_out_of_image_instead_of_clipping(monkeypatch):
    shape = (3, 8)

    def winner(agg, params):
        return (np.full(shape, 2., np.float32), np.ones(shape),
                np.ones(shape), np.zeros(shape))

    monkeypatch.setattr(R, "subpixel_and_sigma", winner)
    out = R.estimate(np.zeros(shape, np.uint8), np.zeros(shape, np.uint8), R.Params(disparities=4))
    assert not out["valid"][:, :2].any()
    assert np.isinf(out["lr_diff"][:, :2]).all()
    assert out["valid"][:, 2:].all()


def test_reference_rejects_invalid_input_and_handles_disparities_wider_than_image():
    with pytest.raises(ValueError):
        R.estimate(np.zeros((4, 8), np.uint8), np.zeros((4, 7), np.uint8))
    out = R.estimate(np.zeros((4, 8), np.uint8), np.zeros((4, 8), np.uint8), R.Params(disparities=16))
    assert not out["valid"].any()


def test_lr_cannot_validate_against_ambiguous_right_evidence(monkeypatch):
    shape = (3, 8)
    calls = 0

    def winner(agg, params):
        nonlocal calls
        calls += 1
        return (np.full(shape, 2., np.float32), np.ones(shape),
                np.full(shape, 1. if calls == 1 else 0.), np.zeros(shape))

    monkeypatch.setattr(R, 'subpixel_and_sigma', winner)
    out = R.estimate(np.zeros(shape, np.uint8), np.zeros(shape, np.uint8), R.Params(disparities=4))
    assert not out['valid'].any()


@pytest.mark.parametrize('paths', [4, 8])
def test_census_reference_recovers_integer_shift_with_independent_eyes(paths):
    left, right = _pair(8)
    out = R.estimate(left[:40, :100], right[:40, :100],
                     R.Params(disparities=16, matching_cost='census', paths=paths, subpixel='none'))
    valid = out['valid'][4:-4, 24:-8]
    assert valid.mean() > .8
    assert np.max(np.abs(out['disparity'][4:-4, 24:-8][valid] - 8)) <= 1
    assert not out['valid'][:2].any()
    assert not out['valid'][:, -2:].any()


def test_census_cost_uses_hamming_and_rejects_missing_window_support():
    rng = np.random.default_rng(9)
    left = rng.integers(0, 128, (9, 12), dtype=np.uint8)
    right = left + np.uint8(50)  # monotonic radiometric change preserves Census
    params = R.Params(disparities=4, matching_cost='census')
    cost = R.cost_volume(left, right, params)
    assert np.all(cost[2:-2, 2:-2, 0] == 0)
    assert np.all(cost[:2] == R.BIG)
    assert np.all(cost[:, :2] == R.BIG)
    assert np.all(cost[:, :5, 3] == R.BIG)
    assert np.any(cost[2:-2, 5:-2, 3] > 0)


@pytest.mark.parametrize('paths', [4, 8])
def test_sgm_aggregates_exact_requested_number_of_paths(paths):
    volume = np.full((5, 7, 4), 2., np.float32)
    total = R.aggregate(volume, np.zeros((5, 7), np.uint8), R.Params(paths=paths))
    np.testing.assert_array_equal(total, paths * volume)


@pytest.mark.parametrize('params', [dict(paths=6), dict(matching_cost='unknown'), dict(subpixel='unknown')])
def test_invalid_reference_ablation_settings_fail_explicitly(params):
    with pytest.raises(ValueError):
        R.estimate(np.ones((5, 8), np.uint8), np.ones((5, 8), np.uint8), R.Params(**params))


@pytest.mark.parametrize('fit,curve,expected', [
    ('parabolic', [5.0625, 1.5625, .0625, .5625, 3.0625], 2.25),
    ('equiangular', [2.25, 1.25, .25, .75, 1.75], 2.25),
    ('none', [2.25, 1.25, .25, .75, 1.75], 2.),
])
def test_subpixel_fit_matches_its_declared_cost_shape(fit, curve, expected):
    disparity, *_ = R.subpixel_and_sigma(np.array([[curve]], np.float32), R.Params(subpixel=fit))
    np.testing.assert_allclose(disparity, expected)


def test_linear_lr_samples_subpixel_disparity_without_crossing_a_depth_edge():
    left = np.full((1, 6), 1.5, np.float32)
    right = np.arange(6, dtype=np.float32)[None]
    valid = np.ones_like(right, dtype=bool)
    nearest, *_ = R.sample_right_disparity(left, right, valid, mode='nearest')
    linear, supported, inside, edge = R.sample_right_disparity(left, right, valid, mode='linear')
    assert nearest[0, 3] == 2 and linear[0, 3] == 1.5
    assert supported[0, 3] and inside[0, 3] and not edge[0, 3]
    right[0, 1:3] = [.5, 2.5]  # naive interpolation would falsely match1.5
    linear, supported, _, edge = R.sample_right_disparity(left, right, valid, mode='linear')
    assert not supported[0, 3] and edge[0, 3] and np.isnan(linear[0, 3])


def test_linear_lr_requires_only_contributing_reliable_taps():
    left = np.full((1, 6), 2., np.float32)
    right = np.full_like(left, np.nan)
    right[0, 2] = 2.
    valid = np.zeros_like(left, dtype=bool)
    valid[0, 2] = True
    sampled, supported, _, _ = R.sample_right_disparity(left, right, valid, mode='linear')
    assert supported[0, 4] and sampled[0, 4] == 2  # exact coordinate needs no neighbor
    left[0, 4] = 2.25
    sampled, supported, _, _ = R.sample_right_disparity(left, right, valid, mode='linear')
    assert not supported[0, 4] and np.isnan(sampled[0, 4])


@pytest.mark.parametrize('mode', ['nearest', 'linear'])
def test_lr_sampler_rejects_outside_nonfinite_and_unreliable_coordinates(mode):
    left = np.array([[1, np.nan, -4, 0, np.inf]], np.float32)
    right = np.zeros_like(left)
    sampled, supported, inside, _ = R.sample_right_disparity(left, right, np.ones_like(left, bool), mode=mode)
    np.testing.assert_array_equal(inside, [[False, False, False, True, False]])
    assert supported[0, 3] and sampled[0, 3] == 0
    assert np.isnan(sampled[~inside]).all()


@pytest.mark.parametrize('params', [dict(lr_sampling='unknown'), dict(lr_max_jump=-1), dict(lr_max_jump=np.nan), dict(lr_tol=np.inf)])
def test_reference_rejects_invalid_lr_configuration(params):
    with pytest.raises(ValueError):
        R.estimate(np.zeros((4, 8), np.uint8), np.zeros((4, 8), np.uint8), R.Params(**params))


@pytest.mark.parametrize('sampling,expected', [('nearest', False), ('linear', True)])
def test_estimate_uses_declared_lr_sampler(monkeypatch, sampling, expected):
    calls = 0
    shape = (2, 8)
    def winner(agg, params):
        nonlocal calls
        calls += 1
        disparity = np.full(shape, 1.5, np.float32) if calls == 1 else np.tile(np.arange(8, dtype=np.float32)[::-1], (2, 1))
        return disparity, np.ones(shape), np.ones(shape), np.zeros(shape)
    monkeypatch.setattr(R, 'subpixel_and_sigma', winner)
    result = R.estimate(np.zeros(shape, np.uint8), np.zeros(shape, np.uint8),
                        R.Params(disparities=16, lr_sampling=sampling, lr_tol=.1))
    assert bool(result['valid'][0, 3]) is expected
    assert result['right_validity'][0, 1:3].all()


def test_right_reference_must_have_in_image_correspondence_even_with_wide_lr_tolerance(monkeypatch):
    calls = 0
    shape = (2, 8)
    def winner(agg, params):
        nonlocal calls
        calls += 1
        return np.full(shape, 2. if calls == 1 else 10., np.float32), np.ones(shape), np.ones(shape), np.zeros(shape)
    monkeypatch.setattr(R, 'subpixel_and_sigma', winner)
    result = R.estimate(np.zeros(shape, np.uint8), np.zeros(shape, np.uint8), R.Params(disparities=16, lr_tol=16))
    assert not result['right_validity'].any()
    assert not result['valid'].any()
