import subprocess
from pathlib import Path
import numpy as np
import pytest

from benchmarks.compact_sgm_gradient_ablation import adaptive_paths, run_compiled


@pytest.fixture(scope='module')
def runner(tmp_path_factory):
    output = tmp_path_factory.mktemp('grad-runner') / 'runner'
    subprocess.run(['c++', '-std=c++20', '-O2', 'benchmarks/compact_sgm_gradient_runner.cpp', '-o', str(output)], check=True)
    return output


@pytest.mark.parametrize('eye', [0, 1])
def test_compiled_paths_match_scalar_with_edges_invalids_and_disjoint_intervals(runner, tmp_path, eye):
    rng = np.random.default_rng(117)
    shape = (5, 7, 8)
    cost = rng.integers(0, 24, shape).astype(float)
    cost[rng.random(shape) < .18] = np.inf
    image = rng.choice([0, 12, 255], shape[:2]).astype(np.uint8)
    lo = rng.integers(0, 4, shape[:2]); hi = rng.integers(4, 9, shape[:2])
    lo[2,2] = hi[2,2] = 2
    expected = adaptive_paths(cost, lo, hi, image)
    paths, sums, winners = run_compiled(runner, tmp_path, cost, lo, hi, image, eye)
    np.testing.assert_array_equal(paths, expected)
    np.testing.assert_array_equal(sums, expected.sum(axis=0))
    expected_sum = expected.sum(axis=0)
    expected_winner = np.where(np.isfinite(expected_sum).any(axis=2), np.argmin(expected_sum, axis=2), -1)
    np.testing.assert_array_equal(winners, expected_winner)


def test_flat_image_matches_constant_sgm():
    from benchmarks.compact_sgm_reference import compact_sgm
    rng = np.random.default_rng(12)
    cost = rng.integers(0, 24, (3, 6, 5)).astype(float)
    lo = np.zeros((3,6), dtype=int); hi = np.full((3,6), 5)
    actual = adaptive_paths(cost, lo, hi, np.zeros((3,6), np.uint8))
    np.testing.assert_array_equal(actual, compact_sgm(cost, lo, hi).paths)


def test_large_gradient_reduces_directional_penalty_to_p1():
    cost = np.array([[[0,100,100], [100,100,0]]], dtype=float)
    lo = np.zeros((1,2), dtype=int); hi = np.full((1,2), 3)
    flat = adaptive_paths(cost, lo, hi, np.zeros((1,2), np.uint8))
    edge = adaptive_paths(cost, lo, hi, np.array([[0,255]], dtype=np.uint8))
    assert flat[0,0,1,2] == 32
    assert edge[0,0,1,2] == 8


@pytest.mark.parametrize('eye', [0, 1])
def test_compiled_image_mode_independently_matches_reference_geometry(runner, tmp_path, eye):
    from stereoforge.refinement import census
    from benchmarks.compact_sgm_gradient_ablation import run_eye
    rng = np.random.default_rng(417)
    images = rng.integers(0, 256, (2,8,12), dtype=np.uint8)
    support = np.ones_like(images, dtype=bool); support[1,3,4] = False
    a, b = (census(images[i], support[i]) for i in (eye,1-eye))
    h,w,d=8,12,6
    lo=rng.integers(0,d+1,(h,w)); hi=np.array([[rng.integers(v,d+1) for v in row] for row in lo])
    cost=np.full((h,w,d),np.inf)
    for y,x,q in np.ndindex(h,w,d):
        ox=x+q if eye else x-q
        if a.support[y,x] and 0 <= ox < w and b.support[y,ox]:
            cost[y,x,q]=(int(a.descriptor[y,x]) ^ int(b.descriptor[y,ox])).bit_count()
    scores=adaptive_paths(cost,lo,hi,images[eye]).sum(axis=0)
    expected=np.where(np.isfinite(scores).any(axis=2), np.argmin(scores,axis=2), -1)
    actual=run_eye(runner,tmp_path,a,b,lo,hi,images[eye].astype(np.uint32),eye,d)
    np.testing.assert_array_equal(actual['winners'], expected)
    for y,x in np.ndindex(h,w):
        ordered=np.sort(scores[y,x][np.isfinite(scores[y,x])])
        inf=np.iinfo(np.uint32).max
        assert actual['count'][y,x] == len(ordered)
        assert actual['best'][y,x] == (int(ordered[0]) if len(ordered) else inf)
        assert actual['second'][y,x] == (int(ordered[1]) if len(ordered)>1 else inf)


def test_frozen_reference_dependency_mutation_is_rejected(tmp_path):
    import hashlib
    from benchmarks.compact_sgm_gradient_experiment import verify_dependencies
    dependency = tmp_path / 'compact_sgm_reference.py'
    dependency.write_text('original')
    frozen = {dependency.name: hashlib.sha256(dependency.read_bytes()).hexdigest()}
    verify_dependencies(tmp_path, frozen)
    dependency.write_text('changed')
    with pytest.raises(ValueError, match='dependency changed'):
        verify_dependencies(tmp_path, frozen)
