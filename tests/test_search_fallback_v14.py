"""Known stereo controls for bounded, opt-in missing/contradictory-prior search."""
import numpy as np
import pytest

from stereoforge.slang_cpu_backend import SlangCpuRefinementBackend


def inputs(*, shift=8, prior=2., supported=True, corrupt=False):
    rng = np.random.default_rng(45)
    w, h, cw, ch = 96, 40, 24, 10
    left = rng.integers(0, 256, (h, w), dtype=np.uint32)
    right = np.roll(left, -shift, axis=1) if not corrupt else np.full_like(left, 127)
    return dict(luma=np.r_[left.ravel(), right.ravel()],
                support=np.full(2*w*h, supported, np.uint32),
                coarse_disparity=np.full(2*cw*ch, prior or 0, np.float32),
                coarse_confidence=np.ones(2*cw*ch, np.float32),
                coarse_validity=np.full(2*cw*ch, prior is not None, np.uint32),
                width=w, height=h, coarse_width=cw, coarse_height=ch,
                fx=50., baseline=.12, principal_offset=0., disparities=32,
                radius=2, ortho_hybrid=True, adaptive_tiles=True, subpixel=False,
                planar_recovery=False, num_threads=1)


def interior(array):
    return array[:96*40].reshape(40, 96)[5:-5, 36:-8]


@pytest.mark.parametrize('prior', [None, 4.])
def test_selective_search_recovers_missing_or_contradicted_priors(prior):
    backend = SlangCpuRefinementBackend()
    config = inputs(prior=prior)
    result = backend.run_pipeline(**config, search_fallback=True)
    valid = interior(result['metric_validity']) != 0
    assert valid.mean() > .9
    np.testing.assert_allclose(interior(result['disparity'])[valid], 8.)
    assert np.all(interior(result['method']) == 2)
    assert np.all(interior(result['reason']) & 4096)
    assert np.max(result['evaluated_candidates']) <= 64  # at most2D per pixel
    if prior is None:
        assert not result['prediction_validity'].any()  # no invented coarse evidence


def test_selective_search_leaves_reliable_priors_local():
    backend = SlangCpuRefinementBackend()
    config = inputs()
    baseline = backend.run_pipeline(**config)
    result = backend.run_pipeline(**config, search_fallback=True)
    for name in ('raw_disparity', 'raw_validity', 'disparity', 'metric_validity'):
        np.testing.assert_array_equal(interior(result[name]), interior(baseline[name]))
    assert np.all(interior(result['method']) == 1)
    assert np.max(interior(result['evaluated_candidates'])) < 32


def test_full_search_photometric_confirmation_accepts_gain_and_offset():
    config = inputs(prior=None)
    w, h = config['width'], config['height']
    luma = config['luma'].reshape(2, h, w).copy()
    # Apply the same compression first, then an independent right-eye exposure.
    luma = 30 + luma // 2
    luma[1] = (1.2*luma[1] + 7).astype(np.uint32)
    result = SlangCpuRefinementBackend().run_pipeline(**(config | dict(luma=luma.ravel())),
                                                    search_fallback=True, ordinal_census=True)
    valid = interior(result['metric_validity']) != 0
    assert valid.mean() > .9
    np.testing.assert_allclose(interior(result['disparity'])[valid], 8.)


def test_ordinal_census_is_exposure_invariant_with_hybrid_aggregation():
    config = inputs()
    neutral = 30 + config['luma'] // 2
    exposed = neutral.reshape(2, 40, 96).copy()
    exposed[1] = (1.2*exposed[1]+7).astype(np.uint32)
    backend = SlangCpuRefinementBackend()
    before = backend.run_pipeline(**(config | dict(luma=neutral)), ordinal_census=True)
    after = backend.run_pipeline(**(config | dict(luma=exposed.ravel())), ordinal_census=True)
    np.testing.assert_array_equal(before['census'], after['census'])
    np.testing.assert_array_equal(before['raw_validity'], after['raw_validity'])
    np.testing.assert_array_equal(before['disparity'], after['disparity'])
    assert (interior(after['metric_validity']) != 0).mean() > .9


def test_unconfirmed_global_retry_preserves_previous_local_proposals():
    import cv2
    config = inputs()
    left = config['luma'][:96*40].reshape(40, 96).astype(np.float32)
    right = cv2.warpAffine(left, np.float32([[1, 0, -8.35], [0, 1, 0]]), (96, 40),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.uint32)
    config['luma'] = np.r_[left.astype(np.uint32).ravel(), right.ravel()]
    backend = SlangCpuRefinementBackend()
    before = backend.run_pipeline(**config)
    after = backend.run_pipeline(**config, search_fallback=True)
    previous = interior(before['raw_validity']) != 0
    assert previous.mean() > .5
    assert interior(after['raw_validity'])[previous].all()
    assert np.count_nonzero(interior(after['method']) == 3) > 100
    retained = interior(after['method']) == 3
    assert np.all(interior(after['reason'])[retained] & 8192)


@pytest.mark.parametrize('change', [dict(supported=False), dict(corrupt=True), dict(shift=0), dict(shift=31)])
def test_full_search_rejects_unsupported_ambiguous_and_global_boundaries(change):
    result = SlangCpuRefinementBackend().run_pipeline(**inputs(prior=None, **change), search_fallback=True)
    assert not interior(result['metric_validity']).any()
    assert np.max(result['evaluated_candidates']) <= 64
    if not change.get('supported', True):
        assert not result['evaluated_candidates'].any()


def test_full_search_respects_declared_disparity_bounds():
    result = SlangCpuRefinementBackend().run_pipeline(**inputs(prior=None), search_fallback=True,
                                                    min_disparity_bound=10, max_disparity_bound=20)
    assert np.all(interior(result['lower']) >= 10)
    assert np.all(interior(result['upper']) <= 20)
    assert not interior(result['metric_validity']).any()  # true disparity8 excluded


@pytest.mark.parametrize('change', [dict(ortho_hybrid=False), dict(disparities=512)])
def test_full_search_rejects_incompatible_cost_scale_or_unbounded_work(change):
    config = inputs() | change
    with pytest.raises(ValueError, match='search fallback'):
        SlangCpuRefinementBackend().run_pipeline(**config, search_fallback=True)
