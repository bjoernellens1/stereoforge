import numpy as np
import pytest

from benchmarks.compact_sgm_reference import compact_sgm
from benchmarks.compact_sgm_witness import trace_witness


def test_scanline_witness_matches_full_scalar_paths_with_unsupported_and_interval_gaps():
    rng = np.random.default_rng(82)
    costs = rng.integers(0, 24, (5, 7, 8)).astype(float)
    costs[2, 1, :] = np.inf
    costs[3, 3, 4] = np.inf
    lo = rng.integers(0, 4, (5, 7))
    hi = rng.integers(4, 9, (5, 7))
    reference = compact_sgm(costs, lo, hi)
    for y, x in [(0, 0), (2, 4), (4, 6), (2, 1)]:
        result = trace_witness(costs.shape, lo, hi, y, x, lambda yy, xx, d: costs[yy, xx, d])
        np.testing.assert_array_equal(result['paths'], reference.paths[:, y, x])
        np.testing.assert_array_equal(result['sums'], reference.sums[y, x])
        assert result['winner'] == (int(np.argmin(reference.sums[y, x])) if np.isfinite(reference.sums[y, x]).any() else -1)


def test_rejects_bad_coordinate():
    with pytest.raises(ValueError):
        trace_witness((2, 2, 2), np.zeros((2, 2)), np.full((2, 2), 2), -1, 0, lambda *_: 0)


@pytest.mark.parametrize('penalties', [(1.5, 32), (8, 32.5), (8, float('inf')), (8, 10001)])
def test_rejects_invalid_penalties(penalties):
    bounds = np.zeros((1, 1), dtype=int)
    with pytest.raises(ValueError):
        trace_witness((1, 1, 2), bounds, bounds + 2, 0, 0, lambda *_: 0, *penalties)


@pytest.mark.parametrize('bad_cost', [float('nan'), -float('inf'), -1, 0.5, 10001])
def test_rejects_invalid_costs(bad_cost):
    bounds = np.zeros((1, 1), dtype=int)
    with pytest.raises(ValueError):
        trace_witness((1, 1, 2), bounds, bounds + 2, 0, 0, lambda *_: bad_cost)


def test_compiled_runner_parity_for_all_pixels(tmp_path):
    import subprocess
    from pathlib import Path
    from benchmarks.compact_sgm_quality import run_costs

    root = Path(__file__).resolve().parents[1]
    runner = tmp_path / 'runner'
    subprocess.run(['c++', '-std=c++20', '-O2', str(root / 'benchmarks/compact_sgm_runner.cpp'), '-o', str(runner)], check=True)
    # Ties, a completely unsupported predecessor and disjoint intervals, with
    # another partial support hole. Cost-input runner is eye-independent.
    costs = np.zeros((3, 5, 5), dtype=float)
    costs[1, 1, :] = np.inf
    costs[0, 2, 2] = np.inf
    costs[2, 3, :] = [2, 3, 0, 4, 1]
    lo = np.array([[0, 0, 2, 0, 0], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0]])
    hi = np.array([[5, 2, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]])
    paths, sums, winners = run_costs(runner, tmp_path / 'data', costs, lo, hi, 8, 32, 'left')
    for y in range(3):
        for x in range(5):
            actual = trace_witness(costs.shape, lo, hi, y, x, lambda yy, xx, d: costs[yy, xx, d])
            np.testing.assert_array_equal(actual['paths'], paths[:, y, x])
            np.testing.assert_array_equal(actual['sums'], sums[y, x])
            assert actual['winner'] == winners[y, x]


def test_provenance_includes_direction_source():
    from benchmarks.compact_sgm_witness import provenance_dependencies
    dependencies = provenance_dependencies()
    assert any(path.name == 'compact_sgm_reference.py' for path in dependencies)
    assert all(path.is_file() for path in dependencies)


def test_compiled_image_runner_both_eye_mappings(tmp_path):
    import subprocess
    from pathlib import Path
    from benchmarks.compact_sgm_quality import run_eye
    from stereoforge.refinement import Census

    root = Path(__file__).resolve().parents[1]
    runner = tmp_path / 'runner'
    subprocess.run(['c++', '-std=c++20', '-O2', str(root / 'benchmarks/compact_sgm_runner.cpp'), '-o', str(runner)], check=True)
    h, w, d = 3, 6, 4
    l = np.array([[0, 1, 3, 7, 15, 31], [7, 3, 1, 0, 2, 6], [5, 5, 2, 2, 1, 1]], dtype=np.uint32)
    r = np.roll(l, -1, axis=1)
    supports = [np.ones((h, w), bool), np.ones((h, w), bool)]
    supports[0][1, 1] = False
    supports[1][1, 4] = False
    descriptors = [Census(l, supports[0]), Census(r, supports[1])]
    lo = np.zeros((h, w), dtype=np.uint32)
    hi = np.full((h, w), d, dtype=np.uint32)
    lo[1, 2], hi[1, 2] = 2, 4
    for eye in (0, 1):
        actual = run_eye(runner, tmp_path / str(eye), descriptors[eye], descriptors[1-eye], lo, hi, eye, d, dump_scores=True)
        def cost(y, x, disparity):
            ox = x + disparity if eye else x - disparity
            if not supports[eye][y, x] or not 0 <= ox < w or not supports[1-eye][y, ox]:
                return np.inf
            return (int(descriptors[eye].descriptor[y, x]) ^ int(descriptors[1-eye].descriptor[y, ox])).bit_count()
        for y in range(h):
            for x in range(w):
                witness = trace_witness((h, w, d), lo, hi, y, x, cost)
                np.testing.assert_array_equal(witness['sums'], actual['scores'][y, x])
                assert witness['winner'] == actual['winners'][y, x]
