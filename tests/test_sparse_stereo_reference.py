"""Behavioral tests for the CPU-only sparse stereo research oracle."""
import numpy as np
import pytest

from benchmarks.sparse_stereo_reference import match_descriptors, select_grid, sparse_stereo


def test_grid_selection_is_image_only_deterministic_and_skips_unsupported():
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    support = np.ones_like(image, bool)
    first = select_grid(image, support, cell_size=4)
    assert first == select_grid(image.copy(), support.copy(), cell_size=4)
    assert len(first) == 4
    for y, x in first:
        support[y, x] = False
    assert all(support[y, x] for y, x in select_grid(image, support, cell_size=4))
    assert select_grid(image, np.zeros_like(support), cell_size=4) == []


def test_sparse_pair_known_shift_and_both_eyes():
    left = np.random.default_rng(123).integers(0, 256, (16, 256), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-137] = left[:, 137:]
    support = np.ones_like(left, bool)
    result = sparse_stereo(left, right, support, support, source_id='frame-12',
                           cell_size=8, disparities=224)
    assert result.source_id == 'frame-12'
    for eye in ('left', 'right'):
        matches = [m for m in result.matches if m.eye == eye and m.valid]
        assert matches and all(m.disparity == 137 for m in matches)
        assert all(m.other_x == m.x + (-137 if eye == 'left' else 137) for m in matches)
    assert result.forward_candidates > 0 and result.reverse_candidates > 0


def test_ties_and_single_candidate_are_rejected():
    desc = np.zeros((1, 8), np.uint32)
    valid = np.ones_like(desc, bool)
    out = match_descriptors(desc, desc, valid, valid, [(0, 4)], 'left', disparities=4)
    assert out.matches[0].reason == 'ambiguous' and not out.matches[0].valid
    valid[:] = False
    valid[0, 4] = True
    out = match_descriptors(desc, desc, valid, valid, [(0, 4)], 'left', disparities=4)
    assert out.matches[0].reason == 'too_few_candidates'


def test_reverse_is_independent_and_rejects_reverse_ambiguity():
    left = np.full((1, 8), 0xffffff, np.uint32)
    right = np.full((1, 8), 0xffffff, np.uint32)
    left[0, 4:6] = 0
    right[0, 3] = 0
    valid = np.ones_like(left, bool)
    out = match_descriptors(left, right, valid, valid, [(0, 5)], 'left', disparities=4)
    assert out.matches[0].reason == 'reverse_ambiguous'
    assert out.forward_candidates == 4 and out.reverse_candidates == 4


def test_invalid_support_and_coordinate_bounds_fail_closed():
    desc = np.zeros((1, 8), np.uint32)
    valid = np.ones_like(desc, bool)
    valid[0, 4] = False
    out = match_descriptors(desc, desc, valid, np.ones_like(valid), [(0, 4)], 'left', disparities=4)
    assert out.matches[0].reason == 'unsupported'
    with pytest.raises(ValueError):
        match_descriptors(desc, desc, valid, valid, [(0, 8)], 'left', disparities=4)


def test_reverse_search_rejects_a_different_unique_winner():
    left = np.full((1, 8), 0xffffff, np.uint32)
    right = np.full((1, 8), 0xffffff, np.uint32)
    left[0, 5] = 0
    left[0, 4] = 1
    right[0, 3] = 1
    support = np.ones_like(left, bool)
    result = match_descriptors(left, right, support, support, [(0, 5)], 'left', disparities=4)
    assert result.matches[0].disparity == 2
    assert result.matches[0].reason == 'reverse_mismatch'
    assert not result.matches[0].valid


def test_descriptor_grid_prefers_balanced_bits_then_gradient_and_row_major():
    image = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
    support = np.ones_like(image, bool)
    descriptor = np.zeros_like(image, np.uint32)
    descriptor[2, 2] = (1 << 12) - 1
    descriptor[3, 3] = (1 << 12) - 1
    descriptor[5, 5] = (1 << 11) - 1
    # Equal descriptor balance and gradient use row-major order.
    assert select_grid(image, support, cell_size=8, descriptor=descriptor) == [(2, 2)]
    # Gradient breaks equal-balance ties, even for a later pixel.
    image[3, 4] = 100
    assert select_grid(image, support, cell_size=8, descriptor=descriptor) == [(3, 3)]
    # A stronger response with less balanced bits is still ranked lower.
    image[5, 6] = 255
    assert select_grid(image, support, cell_size=8, descriptor=descriptor) == [(3, 3)]
    support[2, 2] = False
    assert select_grid(image, support, cell_size=8, descriptor=descriptor) == [(3, 3)]


def test_descriptor_grid_ignores_flat_and_unsupported_footprints():
    image = np.full((8, 8), 50, np.uint8)
    support = np.ones_like(image, bool)
    descriptor = np.full_like(image, (1 << 12) - 1, dtype=np.uint32)
    assert select_grid(image, support, cell_size=8, descriptor=descriptor) == []
    image[3, 4] = 100
    support[:] = False
    support[3, 3] = True
    assert select_grid(image, support, cell_size=8, descriptor=descriptor) == []


def test_sparse_selection_default_is_unchanged_and_ablation_is_explicit():
    left = np.random.default_rng(27).integers(0, 256, (16, 32), dtype=np.uint8)
    right = np.roll(left, -3, axis=1)
    support = np.ones_like(left, bool)
    default = sparse_stereo(left, right, support, support, source_id='f', disparities=8)
    assert default == sparse_stereo(left, right, support, support, source_id='f', disparities=8,
                                    selection='gradient')
    selected = sparse_stereo(left, right, support, support, source_id='f', disparities=8,
                             selection='descriptor')
    assert selected == sparse_stereo(left, right, support, support, source_id='f', disparities=8,
                                     selection='descriptor')
    with pytest.raises(ValueError):
        sparse_stereo(left, right, support, support, source_id='f', selection='unknown')


def test_combined_selection_is_exact_sorted_union_and_deduplicated():
    from benchmarks.sparse_stereo_reference import select_points
    image = np.random.default_rng(28).integers(0, 256, (23, 37), dtype=np.uint8)
    support = np.ones_like(image, bool)
    descriptor = np.random.default_rng(29).integers(0, 1 << 24, image.shape, dtype=np.uint32)
    gradient = select_grid(image, support, 16)
    balanced = select_grid(image, support, 16, descriptor=descriptor)
    combined = select_points(image, support, descriptor, 16, selection='combined')
    assert combined == sorted(set(gradient) | set(balanced))
    assert len(combined) < len(gradient) + len(balanced) or set(gradient).isdisjoint(balanced)
    assert combined == select_points(image.copy(), support.copy(), descriptor.copy(), 16,
                                     selection='combined')


def test_combined_edge_bounds_support_and_matching_work():
    from benchmarks.sparse_stereo_reference import select_points
    image = np.random.default_rng(30).integers(0, 256, (33, 35), dtype=np.uint8)
    support = np.ones_like(image, bool)
    support[10:14, 10:14] = False
    descriptor = np.random.default_rng(31).integers(0, 1 << 24, image.shape, dtype=np.uint32)
    combined = select_points(image, support, descriptor, 16, selection='combined')
    edge = select_points(image, support, descriptor, 16, selection='combined_edge')
    assert edge == sorted(set(edge))
    assert set(combined) <= set(edge)
    assert len(edge) <= 18 * 3 * 3  # at most two base and 16 extra per cell
    for y, x in edge:
        assert support[y, x] and support[y - 1, x] and support[y + 1, x]
        assert support[y, x - 1] and support[y, x + 1]
    right = np.roll(image, -3, axis=1)
    result = sparse_stereo(image, right, support, support, source_id='f',
                           cell_size=16, disparities=8, selection='combined_edge')
    assert len(result.matches) <= 2 * 18 * 3 * 3
    assert result.forward_candidates <= len(result.matches) * 8
    assert result.reverse_candidates <= len(result.matches) * 8
    for eye in ('left', 'right'):
        points = [(m.y, m.x) for m in result.matches if m.eye == eye]
        assert points == sorted(set(points))


def test_subcell_selection_preserves_cross_cell_gradient_footprints():
    image = np.zeros((8, 8), np.uint8)
    image[4, 4] = 255
    support = np.ones_like(image, bool)
    points = select_grid(image, support, cell_size=8, subdivisions=4)
    assert (3, 4) in points and (4, 3) in points
    with pytest.raises(ValueError):
        select_grid(image, support, subdivisions=5)
