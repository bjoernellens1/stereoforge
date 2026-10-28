"""Contract tests for the offline vertical Census experiment."""
import numpy as np
import pytest

from benchmarks.sparse_vertical_reference import match_descriptors, sparse_stereo
from benchmarks.sparse_stereo_reference import sparse_stereo as baseline


def test_vertical_evidence_resolves_center_tie_and_counts_work():
    left = np.full((5, 8), 0xffffff, np.uint32)
    right = np.full_like(left, 0xffffff)
    left[1:4, 5] = [1, 0, 2]
    right[1:4, 4] = [1, 0, 2]
    right[1:4, 3] = [0, 0, 0]
    support = np.ones_like(left, bool)
    r = match_descriptors(left, right, support, support, [(2, 5)], 'left', disparities=3)
    assert r.matches[0].valid and r.matches[0].disparity == 1
    assert r.forward_candidates == r.reverse_candidates == 3
    assert r.forward_hamming_ops == r.reverse_hamming_ops == 9


def test_ties_unsupported_and_border_fail_closed():
    d = np.zeros((5, 8), np.uint32)
    good = np.ones_like(d, bool)
    assert match_descriptors(d, d, good, good, [(2, 5)], 'left', 3).matches[0].reason == 'ambiguous'
    assert match_descriptors(d, d, good, good, [(0, 5)], 'left', 3).matches[0].reason == 'unsupported'
    good[1, 5] = False
    r = match_descriptors(d, d, good, np.ones_like(good), [(2, 5)], 'left', 3)
    assert r.matches[0].reason == 'unsupported' and r.forward_candidates == 0
    with pytest.raises(ValueError):
        match_descriptors(d, d, good, good, [(2, 8)], 'left')


def test_candidate_support_requires_all_three_rows():
    d = np.zeros((5, 8), np.uint32)
    ref = np.ones_like(d, bool)
    other = ref.copy()
    other[1, 4] = False
    r = match_descriptors(d, d, ref, other, [(2, 5)], 'left', disparities=3)
    assert r.forward_candidates == 2 and r.forward_hamming_ops == 6
    assert r.matches[0].reason == 'ambiguous'


def test_reverse_search_rejects_distinct_winner():
    left = np.full((5, 8), 0xffffff, np.uint32)
    right = np.full_like(left, 0xffffff)
    left[1:4, 5] = 1
    left[1:4, 6] = 0
    right[1:4, 4] = 0
    support = np.ones_like(left, bool)
    r = match_descriptors(left, right, support, support, [(2, 5)], 'left', disparities=4)
    assert r.matches[0].reason == 'reverse_mismatch'


def test_long_shift_both_eyes_and_zero_disparity():
    rng = np.random.default_rng(4)
    left = rng.integers(0, 1 << 24, (5, 256), dtype=np.uint32)
    right = np.zeros_like(left)
    right[:, :-137] = left[:, 137:]
    support = np.ones_like(left, bool)
    for eye, x in [('left', 200), ('right', 63)]:
        m = match_descriptors(left, right, support, support, [(2, x)], eye, disparities=224).matches[0]
        assert m.valid and m.disparity == 137
        assert m.other_x == x + (-137 if eye == 'left' else 137)
    m = match_descriptors(left, left, support, support, [(2, 150)], 'left', disparities=224).matches[0]
    assert m.valid and m.disparity == 0


@pytest.mark.parametrize('selector', ['combined', 'combined_edge'])
def test_selector_points_identical_to_baseline(selector):
    image = np.random.default_rng(19).integers(0, 256, (20, 40), dtype=np.uint8)
    support = np.ones_like(image, bool)
    a = baseline(image, image, support, support, source_id='a', cell_size=8, disparities=8, selection=selector)
    b = sparse_stereo(image, image, support, support, source_id='a', cell_size=8, disparities=8, selection=selector)
    assert [(m.eye, m.y, m.x) for m in a.matches] == [(m.eye, m.y, m.x) for m in b.matches]
    assert b == sparse_stereo(image, image, support, support, source_id='a', cell_size=8, disparities=8, selection=selector)
