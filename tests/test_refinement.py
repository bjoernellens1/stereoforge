"""Independent hand-cost and known-shift contracts for the R0 CPU oracle."""
import importlib.util

import numpy as np
import pytest


def oracle():
    assert importlib.util.find_spec('stereoforge.refinement'), 'R0 CPU reference is missing'
    from stereoforge import refinement
    return refinement


def curve(cost, supported=None, lo=0, hi=None, **settings):
    r = oracle()
    c = np.asarray(cost, np.uint32)[:, None, None]
    ok = np.ones(c.shape, bool) if supported is None else np.asarray(supported, bool)[:, None, None]
    return r.select_costs(c, ok, np.array([[lo]], np.int32),
                          np.array([[len(cost)-1 if hi is None else hi]], np.int32),
                          r.Settings(disparities=len(cost), **settings))


def test_equal_cost_second_best_is_other_tie_and_invalid():
    r = oracle(); m = curve([8, 1, 1, 9])
    assert m.integer_winner.item() == 1
    assert m.best.item() == m.second.item() == 1
    assert not m.valid.item()
    assert m.reasons.item() & r.Reason.AMBIGUOUS
    assert m.disparity.view(np.uint32).item() == 0x7fc00000
    assert m.confidence.item() == 0


def test_two_hypotheses_required_even_with_perfect_single_cost():
    r = oracle(); m = curve([7, 0, 8], [False, True, False])
    assert m.candidates.item() == 1 and not m.valid.item()
    assert m.reasons.item() & r.Reason.TOO_FEW


def test_restricted_endpoint_invalid_but_global_endpoint_unique_valid():
    r = oracle(); m = curve([0, 3, 7, 9], lo=1)
    assert not m.valid.item() and m.reasons.item() & r.Reason.RESTRICTED_ENDPOINT
    m = curve([0, 3, 7, 9])
    assert m.valid.item() and m.disparity.item() == 0
    assert m.reasons.item() & r.Reason.GLOBAL_ENDPOINT
    assert not m.subpixel_applied.item()
    assert m.reasons.item() & r.Reason.MISSING_SUBPIXEL_NEIGHBOUR


def test_unique_subpixel_uses_only_evaluated_neighbours_and_exact_integer_margin():
    m = curve([12, 6, 2, 4, 10])
    assert m.valid.item() and m.integer_winner.item() == 2
    assert m.second.item() == 4 and m.confidence.item() == np.float32(.5)
    assert m.disparity.item() == np.float32(2 + 1/6)
    assert m.subpixel_applied.item() and m.candidates.item() == 5
    m = curve([12, 6, 2, 4, 10], [True, False, True, True, True])
    assert m.valid.item() and m.disparity.item() == 2
    assert not m.subpixel_applied.item() and m.candidates.item() == 4
    assert not curve([12, 6, 2, 4, 10], min_margin=3).valid.item()


def test_prediction_half_pixel_phase_and_discontinuity_gate():
    r = oracle(); d = np.array([[2, 3], [3, 4]], np.float32)
    p = r.predict_prior(d, np.ones_like(d), np.ones_like(d, bool), (4, 4), r.Settings())
    assert p.disparity[1, 1] == 5  # coarse(.25,.25)=2.5; target disparity doubles.
    assert not p.valid[0, 0] and p.valid[1:3, 1:3].all()
    assert (p.lower[1, 1], p.upper[1, 1]) == (3, 7)
    d[1, 1] = np.float32(4.00001)
    p = r.predict_prior(d, np.ones_like(d), np.ones_like(d, bool), (4, 4), r.Settings())
    assert not p.valid[1, 1] and p.reasons[1, 1] & r.Reason.PRIOR_DISCONTINUITY


@pytest.mark.parametrize('field,value,reason', [('confidence', .049, 'PRIOR_LOW_CONFIDENCE'), ('valid', False, 'PRIOR_MISSING'), ('disparity', np.nan, 'PRIOR_NONFINITE')])
def test_prediction_requires_every_nonzero_weight_neighbour(field, value, reason):
    r = oracle(); a = dict(disparity=np.full((3, 3), 2, np.float32), confidence=np.ones((3, 3), np.float32), valid=np.ones((3, 3), bool))
    a[field][1, 1] = value
    p = r.predict_prior(**a, target_shape=(6, 6), settings=r.Settings())
    assert not p.valid[2, 2] and p.reasons[2, 2] & getattr(r.Reason, reason)
    assert p.disparity.view(np.uint32)[2, 2] == 0x7fc00000


def test_census_support_is_whole_footprint_and_order_is_known():
    r = oracle(); image = np.arange(81, dtype=np.uint8).reshape(9, 9)
    support = np.ones_like(image, bool)
    c = r.census(image, support)
    assert c.descriptor[4, 4] == (1 << 12)-1
    assert c.support.sum() == 25 and not c.support[:2].any()
    support[2, 2] = False
    c = r.census(image, support)
    assert not c.support[4, 4] and c.support[5, 5]


def shifted():
    # Independently specified raw scene: right(x)=left(x+4), disparity exactly4.
    left = np.random.default_rng(17).integers(0, 256, (18, 64), dtype=np.uint8)
    right = np.zeros_like(left); right[:, :-4] = left[:, 4:]
    return np.stack((left, right))


@pytest.mark.parametrize('radius', [2, 4, 8])
def test_known_shift_independent_eyes_metric_nonzero_cx_and_work(radius):
    r = oracle(); image = shifted(); support = np.ones_like(image, bool)
    settings = r.Settings(disparities=16, radius=radius, subpixel=False)
    prior = np.full((2, 9, 32), 2, np.float32)
    out = r.refine_pair(image, support, prior, np.ones_like(prior), np.ones_like(prior, bool),
                        r.Geometry(fx=100, baseline_m=.1, cx_offset=1), 57, settings)
    area = np.s_[:, 4:14, 20:44]
    assert out.metric_validity[area].mean() > .95  # Genuine equal Census ties remain holes.
    valid = out.metric_validity[area]
    assert np.all(out.disparity[area][valid] == 4)
    assert np.all(out.depth[area][valid] == np.float32(np.float32(100)*np.float32(.1)/np.float32(3)))
    assert np.all(out.source_ids[area][valid] == 57) and not out.age_ns.any()
    assert np.all(out.provenance[area][valid] == 1) and np.all(out.contributor_ids[area][..., 0][valid] == 57)
    assert np.all(out.contributor_ids[area][..., 1:] == -1)
    expected = min(15, 4+radius)-max(0, 4-radius)+1
    assert np.all(out.matches[0].candidates[4:14, 20:44] == expected)
    assert np.all(out.matches[1].candidates[4:14, 20:44] == expected)


def test_missing_prior_never_becomes_fresh_and_bounded_recovery_can_find_shift():
    r = oracle(); image = shifted(); prior = np.full((2, 9, 32), 2, np.float32)
    valid = np.zeros_like(prior, bool); support = np.ones_like(image, bool)
    args = (image, support, prior, np.ones_like(prior), valid, r.Geometry(100, .1), 5)
    guided = r.refine_pair(*args, r.Settings(disparities=16, subpixel=False))
    assert not guided.metric_validity.any() and not guided.matches[0].candidates.any()
    recovered = r.refine_pair(*args, r.Settings(disparities=16, subpixel=False, recovery=True))
    area = np.s_[:, 4:14, 20:40]
    assert np.count_nonzero(recovered.metric_validity[area]) == 374  # Exact frozen tuple-Hamming truth, including ties.
    assert np.all(recovered.disparity[area][recovered.metric_validity[area]] == 4)
    for m in recovered.matches:
        assert np.all(m.method[4:14, 20:40] == r.Method.RECOVERED)
        assert np.all(m.recovery_candidates[4:14, 20:40] == 16)
        assert not m.guided_candidates.any()
    assert not guided.provenance.any() and np.all(guided.source_ids == -1)


def test_lr_half_ties_symmetric_and_failure_is_not_fresh():
    r = oracle(); d = np.zeros((2, 1, 8), np.float32); v = np.ones_like(d, bool)
    d[0, 0, 4] = 1.5; d[1, 0, 3] = 1.5
    lr = r.lr_consistency(d, v, 0)
    assert lr[0, 0, 4] and not lr[1, 0, 3]  # L4->R3, R3->L5.
    d[1, 0, 3] = 3
    assert not r.lr_consistency(d, v, 1)[0, 0, 4]


def test_excluded_regions_stay_invalid_even_with_recovery():
    r = oracle(); image = shifted(); prior = np.full((2, 9, 32), 2, np.float32)
    regions = np.ones_like(image, bool); regions[:, :, 20:30] = False
    out = r.refine_pair(image, np.ones_like(image, bool), prior, np.ones_like(prior),
                        np.ones_like(prior, bool), r.Geometry(100, .1), 7,
                        r.Settings(disparities=16, recovery=True), regions=regions)
    assert not out.metric_validity[:, :, 20:30].any()
    assert not out.matches[0].candidates[:, 20:30].any()


@pytest.mark.parametrize('kwargs', [{'radius': 3}, {'disparities': 1}, {'min_margin': 0}, {'prior_confidence_floor': np.nan}, {'prior_span': -1}, {'lr_tolerance': -1}])
def test_settings_reject_invalid_parameters(kwargs):
    with pytest.raises(ValueError): oracle().Settings(**kwargs)


def test_shape_source_and_geometry_validation():
    r = oracle(); image = shifted(); prior = np.full((2, 9, 32), 2, np.float32)
    args = [image, np.ones_like(image, bool), prior, np.ones_like(prior), np.ones_like(prior, bool), r.Geometry(100, .1), 0]
    with pytest.raises(ValueError): r.predict_prior(prior[0], prior[0], prior[0] > 0, (17, 64), r.Settings())
    for bad in [-1, 1 << 63, True]:
        args[-1] = bad
        with pytest.raises(ValueError): r.refine_pair(*args)
    with pytest.raises(ValueError): r.Geometry(1e-50, .1)


def test_arithmetic_bounds_are_operation_specific_and_fail_closed_outside_domain():
    r = oracle(); c = r.confidence_bounds(np.array([2, 0]), np.array([7, 3]))
    exact = np.array([5/7, 1.])
    assert np.all(c.lower <= exact) and np.all(c.upper >= exact)
    assert np.all(c.upper-c.lower < 1e-6)
    s = r.subpixel_bounds(np.array([6]), np.array([2]), np.array([4]), np.array([2]))
    assert s.lower[0] <= 2+1/6 <= s.upper[0]
    z = r.metric_bounds(s, r.Geometry(100, .1, 1))
    exact_z = float(np.float32(100))*float(np.float32(.1))/(2+1/6-1)
    assert z.lower[0] <= exact_z <= z.upper[0]
    with pytest.raises(ValueError): r.metric_bounds(r.FloatBounds(np.array([1.]), np.array([1.])), r.Geometry(100, .1, 1))
    assert 'unqualified' in r.ARITHMETIC_QUALIFICATION.lower()


def scalar_expected_full_search(images, disparities):
    """Test-only tuple descriptors and Python integer Hamming, not oracle calls."""
    _, h, w = images.shape
    descriptors = []
    for image in images:
        rows = {}
        for y in range(2, h-2):
            for x in range(2, w-2):
                rows[y, x] = tuple(int(image[y+dy, x+dx]) < int(image[y, x])
                                   for dy in range(-2, 3) for dx in range(-2, 3)
                                   if (dy, dx) != (0, 0))
        descriptors.append(rows)
    winners = np.zeros(images.shape, np.int32)
    valid = np.zeros(images.shape, bool)
    for eye in (0, 1):
        for (y, x), own in descriptors[eye].items():
            hypotheses = []
            for d in range(disparities):
                target = x+d if eye else x-d
                other = descriptors[1-eye].get((y, target))
                if other is not None:
                    hypotheses.append((sum(a != b for a, b in zip(own, other)), d))
            hypotheses.sort()
            if len(hypotheses) >= 2 and hypotheses[1][0] > hypotheses[0][0]:
                valid[eye, y, x] = True
                winners[eye, y, x] = hypotheses[0][1]
    metric = np.zeros_like(valid)
    for eye, y, x in np.ndindex(images.shape):
        d = int(winners[eye, y, x]); target = x+d if eye else x-d
        metric[eye, y, x] = (valid[eye, y, x] and d > 0 and 0 <= target < w
                             and valid[1-eye, y, target]
                             and abs(d-int(winners[1-eye, y, target])) <= 1)
    return winners, valid, metric


def test_recovery_exact_masks_against_independent_tuple_hamming():
    r = oracle(); image = shifted(); prior = np.full((2, 9, 32), 2, np.float32)
    out = r.refine_pair(image, np.ones_like(image, bool), prior, np.ones_like(prior),
                        np.zeros_like(prior, bool), r.Geometry(100, .1), 17,
                        r.Settings(disparities=16, subpixel=False, recovery=True))
    winners, valid, metric = scalar_expected_full_search(image, 16)
    np.testing.assert_array_equal(out.validity, valid)
    np.testing.assert_array_equal(out.metric_validity, metric)
    np.testing.assert_array_equal(out.disparity[valid], winners[valid])
    assert np.all(out.depth.view(np.uint32)[~metric] == 0x7fc00000)
    assert not out.metric_confidence[~metric].any()
    assert not out.provenance[~metric].any() and np.all(out.source_ids[~metric] == -1)


def test_restricted_boundary_recovery_and_explicit_selection_count_once():
    r = oracle(); image = shifted(); prior = np.full((2, 9, 32), 3, np.float32)
    args = (image, np.ones_like(image, bool), prior, np.ones_like(prior), np.ones_like(prior, bool), r.Geometry(100, .1), 8)
    guided = r.refine_pair(*args, r.Settings(disparities=16, subpixel=False))
    assert not guided.validity[:, 6, 30].any()
    assert all(m.reasons[6, 30] & r.Reason.RESTRICTED_ENDPOINT for m in guided.matches)
    out = r.refine_pair(*args, r.Settings(disparities=16, subpixel=False, recovery=True))
    assert out.metric_validity[:, 6, 30].all() and np.all(out.disparity[:, 6, 30] == 4)
    for m in out.matches:
        assert m.candidates[6, 30] == 21
        assert m.guided_candidates[6, 30] == 5 and m.recovery_candidates[6, 30] == 16
        assert m.method[6, 30] == r.Method.RECOVERED
    prior[:] = 2
    explicit = np.zeros_like(image, bool); explicit[:, 6, 30] = True
    out = r.refine_pair(*args, r.Settings(disparities=16, subpixel=False, recovery=True), recovery_regions=explicit)
    for m in out.matches:
        assert m.reasons[6, 30] & r.Reason.EXPLICIT_RECOVERY
        assert m.candidates[6, 30] == 21


def test_uninformative_target_remains_invalid_after_one_recovery_round():
    r = oracle(); image = np.ones((2, 18, 64), np.uint8); prior = np.full((2, 9, 32), 2, np.float32)
    out = r.refine_pair(image, np.ones_like(image, bool), prior, np.ones_like(prior), np.ones_like(prior, bool),
                        r.Geometry(100, .1), 19, r.Settings(disparities=16, recovery=True))
    assert not out.validity.any() and not out.metric_validity.any()
    for m in out.matches:
        assert m.candidates[6, 30] == 21 and m.method[6, 30] == r.Method.RECOVERED
        assert m.best[6, 30] == m.second[6, 30] == 0


def test_actual_work_excludes_out_of_image_and_unsupported_hypotheses():
    r = oracle(); image = shifted(); descriptors = [r.census(i, np.ones_like(i, bool)) for i in image]
    p = r.predict_prior(np.full((9, 32), 2, np.float32), np.ones((9, 32), np.float32), np.ones((9, 32), bool), (18, 64))
    left = r.match_eye(descriptors[0], descriptors[1], p, 0, r.Settings(disparities=16))
    assert left.requested[6, 4] == 5 and left.candidates[6, 4] == 1  # Only d2 leaves R Census inbounds.
    assert not left.valid[6, 4] and left.reasons[6, 4] & r.Reason.TOO_FEW
    assert left.requested[0, 4] == left.candidates[0, 4] == 0


def test_nonzero_prior_confidence_threshold_is_inclusive_and_no_input_mutation():
    r = oracle(); d = np.full((3, 3), 2, np.float32); c = np.full_like(d, np.float32(.05)); valid = np.ones_like(d, bool)
    snapshots = [a.copy() for a in (d, c, valid)]
    p = r.predict_prior(d, c, valid, (6, 6))
    assert p.valid[1:5, 1:5].all()
    for value, original in zip((d, c, valid), snapshots): np.testing.assert_array_equal(value, original)
    with pytest.raises(ValueError): r.predict_prior(d.astype(float), c, valid, (6, 6))
    with pytest.raises(ValueError): r.predict_prior(d, c*100, valid, (6, 6))


def test_public_cost_and_census_input_validation():
    r = oracle()
    with pytest.raises(ValueError): r.select_costs(np.array([-1, 1, 2], np.int32)[:, None, None], np.ones((3, 1, 1), bool), np.zeros((1, 1), np.int32), np.full((1, 1), 2, np.int32), r.Settings(disparities=3))
    with pytest.raises(ValueError): r.census(np.full((5, 5), 256, np.uint32), np.ones((5, 5), bool))
    with pytest.raises(ValueError): r.census(np.zeros((5, 5), np.uint8), np.full((5, 5), 2, np.uint32))
    c = r.census(np.zeros((3, 7), np.uint8), np.ones((3, 7), bool))
    assert not c.support.any() and not c.descriptor.any()


def test_metric_bounds_reject_underflow_overflow_and_unstable_predicate():
    r = oracle()
    with pytest.raises(ValueError): r.confidence_bounds(np.array([3]), np.array([3]))
    with pytest.raises(ValueError): r.subpixel_bounds(np.array([25]), np.array([2]), np.array([3]), np.array([2]))
    with pytest.raises(ValueError): r.FloatBounds(np.array([2.]), np.array([1.]))
    with pytest.raises(ValueError): r.metric_bounds(r.FloatBounds(np.array([.9]), np.array([1.1])), r.Geometry(100, .1, 1))
    with pytest.raises(ValueError): r.metric_bounds(r.FloatBounds(np.array([1.]), np.array([2.])), r.Geometry(1e-30, 1e-30))
    with pytest.raises(ValueError): r.metric_bounds(r.FloatBounds(np.array([1.]), np.array([2.])), r.Geometry(1e30, 1e30))


def test_all_integer_confidence_ratios_and_subpixel_curves_have_derived_bounds():
    r = oracle()
    from fractions import Fraction
    for best in range(24):
        second = np.arange(best+1, 25, dtype=np.int32)
        b = r.confidence_bounds(np.full_like(second, best), second)
        exact = np.array([float(Fraction(int(s)-best, int(s))) for s in second])
        nominal = (second.astype(np.float32)-np.float32(best))/second.astype(np.float32)
        assert np.all(b.lower <= exact) and np.all(exact <= b.upper)
        assert np.all(b.lower <= nominal) and np.all(nominal <= b.upper)
    # Every pair of supported neighbour costs above a unique integer minimum.
    best, minus, plus = np.indices((25, 25, 25))
    keep = (minus > best) & (plus > best)
    best, minus, plus = best[keep], minus[keep], plus[keep]
    winner = np.full_like(best, 63)
    b = r.subpixel_bounds(minus, best, plus, winner)
    denominator = ((minus-2*best)+plus).astype(np.float32)
    offset = np.float32(.5)*(minus-plus).astype(np.float32)/denominator
    nominal = np.float32(63)+np.clip(offset, -1, 1)
    assert np.all(b.lower <= nominal) and np.all(nominal <= b.upper)
    assert np.all(b.upper-b.lower <= 2*np.spacing(np.float32(63)))


def test_region_uint32_wire_mask_and_minimum_margin_not_probability_gate():
    r = oracle(); image = shifted(); prior = np.full((2, 9, 32), 2, np.float32)
    region = np.ones_like(image, np.uint32); region[:, :, 20:30] = 0
    out = r.refine_pair(image, np.ones_like(image, np.uint32), prior, np.ones_like(prior),
                        np.ones_like(prior, np.uint32), r.Geometry(100, .1), (1 << 63)-1,
                        r.Settings(disparities=16, recovery=True), regions=region)
    assert not out.metric_validity[:, :, 20:30].any()
    assert np.any(out.source_ids == (1 << 63)-1)
    m = curve([20, 22, 24], subpixel=False)
    assert m.valid.item() and m.confidence.item() == np.float32(2/22)


def test_provisional_lr_recovery_runs_after_both_initial_eyes_and_final_lr_runs_again():
    r = oracle(); image = shifted(); prior = np.full((2, 9, 32), 2, np.float32); prior[1] = 4
    args = (image, np.ones_like(image, bool), prior, np.ones_like(prior), np.ones_like(prior, bool), r.Geometry(100, .1), 31)
    guided = r.refine_pair(*args, r.Settings(disparities=16, subpixel=False))
    out = r.refine_pair(*args, r.Settings(disparities=16, subpixel=False, recovery=True))
    assert guided.validity[0, 6, 30] and not guided.lr_validity[0, 6, 30]
    assert out.metric_validity[:, 6, 30].all()
    for m in out.matches:
        assert m.method[6, 30] == r.Method.RECOVERED
        assert m.recovery_candidates[6, 30] == 16
    assert out.matches[0].reasons[6, 30] & r.Reason.LR_FAILED


def test_out_of_domain_prior_and_luma_fail_without_overflow():
    r = oracle(); d = np.full((3, 3), np.finfo(np.float32).max, np.float32)
    p = r.predict_prior(d, np.ones_like(d), np.ones_like(d, bool), (6, 6))
    assert not p.valid.any() and np.all(p.disparity.view(np.uint32) == 0x7fc00000)
    with pytest.raises(ValueError): r.census(np.full((5, 5), -1, np.int32), np.ones((5, 5), bool))
    with pytest.raises(ValueError): r.subpixel_bounds(np.array([2]), np.array([0]), np.array([4]), np.array([-1]))


def test_metric_bounds_preserve_tiny_nonzero_offset_before_round_enclosure():
    r = oracle()
    # Exact 1-cx is below1 although binary64 subtraction rounds it to1.
    # RTZ can return the lower binary32 neighbour as the denominator. The
    # following allowed division is2.4999999702 ULP above its exact quotient.
    bounds = r.metric_bounds(r.FloatBounds(np.array([1.]), np.array([1.])), r.Geometry(1, 1, 1e-30))
    allowed = float(np.float32(1.0000003576278687))
    assert bounds.upper[0] >= allowed
    bounds = r.metric_bounds(r.FloatBounds(np.array([1.]), np.array([1.])), r.Geometry(1, 1, -1e-30))
    denominator = np.nextafter(np.float32(1), np.float32(np.inf))
    assert bounds.lower[0] <= float(np.float32(1)/denominator)


def test_lr_uses_separately_rounded_binary32_coordinate_operations():
    r = oracle()
    d = np.zeros((2, 1, 8), np.float32); v = np.zeros_like(d, bool)
    above = np.nextafter(np.float32(1.5), np.float32(np.inf))
    d[0, 0, 4] = above; d[1, 0, 3] = above
    v[0, 0, 4] = v[1, 0, 3] = True
    # Hand bit proof:4-above=2.5-2^-23, halfway between two binary32
    # numbers; RNE rounds to2.5, then+.5=3. Double floor gives2.
    assert np.float32(np.float32(4)-above).view(np.uint32) == np.float32(2.5).view(np.uint32)
    assert r.lr_consistency(d, v, 0)[0, 0, 4]
    below = np.nextafter(np.float32(1.5), np.float32(-np.inf))
    d[:] = 0; v[:] = False
    d[1, 0, 2] = below; d[0, 0, 4] = below
    v[1, 0, 2] = v[0, 0, 4] = True
    #2+below rounds to3.5, then+.5=4. Double floor gives3.
    assert np.float32(np.float32(2)+below).view(np.uint32) == np.float32(3.5).view(np.uint32)
    assert r.lr_consistency(d, v, 0)[1, 0, 2]


def test_lr_threshold_uses_binary32_subtraction_without_tolerance_epsilon():
    r = oracle(); d = np.ones((2, 1, 8), np.float32); v = np.ones_like(d, bool)
    d[0, 0, 4] = 2
    assert r.lr_consistency(d, v, 1)[0, 0, 4]
    d[0, 0, 4] = np.nextafter(np.float32(2), np.float32(np.inf))
    assert not r.lr_consistency(d, v, 1)[0, 0, 4]


def test_exact_round_enclosures_at_zero_and_both_sides_of_power_two():
    r = oracle()
    from fractions import Fraction
    one = Fraction(1); tiny = Fraction(1, 10**40)
    b = r._round_enclosure(np.array([one-tiny, one, one+tiny, Fraction(0)], object),
                            np.array([one-tiny, one, one+tiny, Fraction(0)], object))
    np.testing.assert_array_equal(b.lower, [float(np.nextafter(np.float32(1), np.float32(0))), 1, 1, 0])
    np.testing.assert_array_equal(b.upper, [1, 1, float(np.nextafter(np.float32(1), np.float32(2))), 0])
    assert r._ulp(Fraction(1)) == Fraction(1, 2**24)
    assert r._ulp(Fraction(1)+tiny) == Fraction(1, 2**23)
    assert r._ulp(Fraction(1)-tiny) == Fraction(1, 2**24)


def test_metric_bound_contains_allowed_directed_intermediate_and_division_results():
    r = oracle()
    from fractions import Fraction

    def bracket(real):
        guess = np.float32(float(real)); actual = Fraction.from_float(float(guess))
        lo = np.nextafter(guess, np.float32(-np.inf)) if actual > real else guess
        hi = np.nextafter(guess, np.float32(np.inf)) if actual < real else guess
        return (lo, hi)

    for offset in [1e-30, -1e-30, 0, .25, -.25]:
        geometry = r.Geometry(1, 1, offset)
        bounds = r.metric_bounds(r.FloatBounds(np.array([1.]), np.array([1.])), geometry)
        exact_den = Fraction(1)-Fraction.from_float(float(geometry.cx_offset))
        for denominator in bracket(exact_den):
            quotient = Fraction(1)/Fraction.from_float(float(denominator))
            # Near1, enumerate nearby representable answers and retain only
            # those satisfying the actual2.5ULP contract in exact rational math.
            answer = np.float32(float(quotient))
            candidates = [answer]
            lo = hi = answer
            for _ in range(4):
                lo = np.nextafter(lo, np.float32(-np.inf)); hi = np.nextafter(hi, np.float32(np.inf))
                candidates.extend([lo, hi])
            q0, q1 = bracket(quotient)
            if q0 == q1:
                gap = min(Fraction.from_float(float(q0))-Fraction.from_float(float(np.nextafter(q0, np.float32(-np.inf)))),
                          Fraction.from_float(float(np.nextafter(q0, np.float32(np.inf))))-Fraction.from_float(float(q0)))
            else:
                gap = Fraction.from_float(float(q1))-Fraction.from_float(float(q0))
            for candidate in candidates:
                if abs(Fraction.from_float(float(candidate))-quotient) <= Fraction(5, 2)*gap:
                    assert bounds.lower[0] <= candidate <= bounds.upper[0]
