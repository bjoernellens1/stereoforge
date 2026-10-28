import numpy as np
import pytest

from benchmarks.compact_sgm_reference import compact_sgm, dense_masked_sgm, DIRECTIONS


def compare(cost, lo, hi, p1=2, p2=7):
    dense = dense_masked_sgm(cost, lo, hi, p1, p2)
    compact = compact_sgm(cost, lo, hi, p1, p2)
    np.testing.assert_array_equal(compact.paths, dense.paths)
    np.testing.assert_array_equal(compact.sums, dense.sums)
    indices = np.arange(cost.shape[2])
    active = np.isfinite(cost) & (indices >= lo[..., None]) & (indices < hi[..., None])
    assert compact.candidate_updates == len(DIRECTIONS) * int(active.sum())
    assert compact.interval_slots == len(DIRECTIONS) * int(np.sum(hi - lo))
    assert compact.dense_slots == len(DIRECTIONS) * cost.size
    return compact


def test_random_intervals_invalid_candidates_and_directions():
    rng = np.random.default_rng(731)
    for h, w, d in [(1, 13, 7), (13, 1, 7), (5, 7, 9)]:
        for _ in range(12):
            cost = rng.integers(0, 8, (h, w, d)).astype(float)
            cost[rng.random(cost.shape) < .22] = np.inf
            lo = rng.integers(0, d + 1, (h, w))
            hi = np.array([[rng.integers(x, d + 1) for x in row] for row in lo])
            compare(cost, lo, hi)


def test_disjoint_expansion_empty_and_reset():
    cost = np.zeros((1, 6, 6))
    lo = np.array([[0, 4, 1, 3, 0, 1]])
    hi = np.array([[2, 6, 5, 3, 6, 2]])
    result = compare(cost, lo, hi, 2, 5)
    # Left-to-right: disjoint predecessor contributes P2 even with no overlap.
    np.testing.assert_array_equal(result.paths[0, 0, 1, 4:6], [5, 5])
    # Empty predecessor resets, so expanded candidates recover raw costs.
    np.testing.assert_array_equal(result.paths[0, 0, 4], np.zeros(6))
    assert np.isinf(result.sums[0, 3]).all()


def test_full_range_matches_existing_scalar_for_finite_costs():
    from stereoforge.reference import aggregate_sgm
    rng = np.random.default_rng(7)
    cost = rng.integers(0, 25, (4, 6, 8)).astype(float)
    lo = np.zeros(cost.shape[:2], dtype=int)
    hi = np.full(cost.shape[:2], cost.shape[2], dtype=int)
    result = compare(cost, lo, hi, 8, 32)
    np.testing.assert_array_equal(result.sums, aggregate_sgm(cost.astype(np.uint32), paths=4, p1=8, p2=32))


@pytest.mark.parametrize('lo,hi,p1,p2', [(-1, 1, 1, 2), (0, 4, 1, 2), (2, 1, 1, 2), (0, 2, 3, 2)])
def test_invalid_configuration(lo, hi, p1, p2):
    with pytest.raises(ValueError):
        compact_sgm(np.zeros((1, 1, 3)), np.array([[lo]]), np.array([[hi]]), p1, p2)


def test_hand_calculated_neighbors_and_invalid_holes():
    cost = np.array([[[0., 9., 4., np.inf], [3., 0., 2., 1.]]])
    lo = np.zeros((1, 2), int)
    hi = np.full((1, 2), 4, int)
    result = compare(cost, lo, hi, 2, 7)
    np.testing.assert_array_equal(result.paths[0, 0, 1], [3, 2, 6, 7])


@pytest.mark.parametrize('bad', [np.nan, -np.inf, -1., 1.5, 1e308, 10001.])
def test_cost_domain_rejected(bad):
    with pytest.raises(ValueError):
        compact_sgm(np.array([[[bad]]]), np.array([[0]]), np.array([[1]]))


@pytest.mark.parametrize('p1,p2', [(1.5, 3), (1, 1e308), (1, 10001), (-1, 2)])
def test_penalty_domain_rejected(p1, p2):
    with pytest.raises(ValueError):
        compact_sgm(np.zeros((1, 1, 1)), np.array([[0]]), np.array([[1]]), p1, p2)
