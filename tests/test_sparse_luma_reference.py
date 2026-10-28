"""Contracts for the exploratory, CPU-only luma tie-break matcher."""
import numpy as np
import pytest
from benchmarks import sparse_luma_reference as luma
from benchmarks import sparse_stereo_reference as base


def test_known_shift_both_eyes_and_same_points_as_baseline():
    left = np.random.default_rng(123).integers(0, 256, (16, 256), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-137] = left[:, 137:]
    support = np.ones_like(left, bool)
    support[5:8, 80:84] = False
    kwargs = dict(source_id='frame-12', cell_size=8, disparities=224, selection='combined')
    result = luma.sparse_stereo(left, right, support, support, **kwargs)
    reference = base.sparse_stereo(left, right, support, support, **kwargs)
    assert [(m.eye, m.y, m.x) for m in result.matches] == [(m.eye, m.y, m.x) for m in reference.matches]
    assert result.source_id == 'frame-12'
    for eye in ('left', 'right'):
        matched = [m for m in result.matches if m.eye == eye and m.valid]
        assert matched and all(m.disparity == 137 for m in matched)
        assert all(m.other_x == m.x + (-137 if eye == 'left' else 137) for m in matched)
    assert result.forward_hamming_upper_bound == result.forward_luma_upper_bound == result.forward_candidates
    assert result.reverse_hamming_upper_bound == result.reverse_luma_upper_bound == result.reverse_candidates
    assert not hasattr(result.matches[0], 'depth')


def test_equal_lexicographic_scores_rejected_and_support_guard():
    d = np.zeros((1, 8), np.uint32)
    i = np.zeros((1, 8), np.uint8)
    s = np.ones((1, 8), bool)
    out = luma.match_descriptors(d, d, i, i, s, s, [(0, 4)], 'left', disparities=4)
    assert out.matches[0].reason == 'ambiguous'
    s[0, 4] = False
    assert luma.match_descriptors(d, d, i, i, s, s, [(0, 4)], 'left', disparities=4).matches[0].reason == 'unsupported'
    with pytest.raises(ValueError):
        luma.match_descriptors(d, d, i, i, s, s, [(0, 8)], 'left', disparities=4)


def test_reverse_ambiguity_and_mismatch():
    left = np.full((1, 8), 0xffffff, np.uint32)
    right = left.copy()
    left[0, 4:6] = 0
    right[0, 3] = 0
    i = np.zeros((1, 8), np.uint8)
    s = np.ones((1, 8), bool)
    result = luma.match_descriptors(left, right, i, i, s, s, [(0, 5)], 'left', disparities=4)
    assert result.matches[0].reason == 'reverse_ambiguous'
    left[0, 4] = right[0, 3] = 1
    result = luma.match_descriptors(left, right, i, i, s, s, [(0, 5)], 'left', disparities=4)
    assert result.matches[0].reason == 'reverse_mismatch'
    assert not result.matches[0].valid


def test_luma_tiebreak_and_exposure_change_affects_winner():
    d = np.zeros((1, 8), np.uint32)
    left = np.full((1, 8), 100, np.uint8)
    right = np.full((1, 8), 200, np.uint8)
    right[0, 3], right[0, 2] = 100, 130
    s = np.ones((1, 8), bool)
    # Other left pixels have distinct intensities so reverse search is unique.
    left[0, 2:] = [160, 140, 120, 100, 180, 190]
    a = luma.match_descriptors(d, d, left, right, s, s, [(0, 5)], 'left', disparities=4).matches[0]
    assert a.valid and a.disparity == 2 and a.best_cost == 0
    right[0, 3], right[0, 2] = 150, 105
    b = luma.match_descriptors(d, d, left, right, s, s, [(0, 5)], 'left', disparities=4).matches[0]
    assert b.disparity == 3
    assert b.disparity != a.disparity


def test_census_is_primary_key_and_zero_disparity_not_metric():
    left = np.zeros((1, 5), np.uint32)
    right = np.full((1, 5), 0xffffff, np.uint32)
    right[0, 3], right[0, 2] = 0, 1
    li = np.full((1, 5), 255, np.uint8)
    ri = np.zeros((1, 5), np.uint8)
    ri[0, 2] = 255
    s = np.ones((1, 5), bool)
    m = luma.match_descriptors(left, right, li, ri, s, s, [(0, 3)], 'left', disparities=3).matches[0]
    assert m.disparity == 0 and m.best_cost == 255 and m.second_cost == 256
    assert not hasattr(m, 'depth')
