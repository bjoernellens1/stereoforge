import subprocess
from pathlib import Path

import numpy as np
import pytest

from benchmarks.compact_sgm_reference import compact_sgm
from benchmarks.compact_sgm_quality import run_costs


@pytest.fixture(scope='module')
def runner(tmp_path_factory):
    out = tmp_path_factory.mktemp('compact-runner') / 'runner'
    source = Path(__file__).resolve().parents[1] / 'benchmarks/compact_sgm_runner.cpp'
    subprocess.run(['c++', '-std=c++20', '-O2', str(source), '-o', str(out)], check=True)
    return out


@pytest.mark.parametrize('shape', [(1, 6, 6), (7, 1, 9), (4, 7, 8)])
@pytest.mark.parametrize('eye', ['left', 'right'])
def test_compiled_matches_reference_paths_sums_and_winners(runner, tmp_path, shape, eye):
    rng = np.random.default_rng(741)
    cost = rng.integers(0, 25, shape).astype(float)
    cost[rng.random(shape) < .25] = np.inf
    lo = rng.integers(0, shape[2] + 1, shape[:2])
    hi = np.array([[rng.integers(v, shape[2] + 1) for v in row] for row in lo])
    # Disjoint, invalid and empty predecessor states explicitly occur.
    lo.flat[0], hi.flat[0] = 0, 0
    result = compact_sgm(cost, lo, hi, 2, 7)
    paths, sums, winners = run_costs(runner, tmp_path, cost, lo, hi, 2, 7, eye)
    np.testing.assert_array_equal(paths, result.paths)
    np.testing.assert_array_equal(sums, result.sums)
    expected = np.argmin(result.sums, axis=2).astype(np.int32)
    expected[~np.isfinite(result.sums).any(axis=2)] = -1
    np.testing.assert_array_equal(winners, expected)


def test_disjoint_reset_and_tie_selection(runner, tmp_path):
    cost = np.zeros((1, 6, 6))
    lo = np.array([[0, 4, 1, 3, 0, 1]])
    hi = np.array([[2, 6, 5, 3, 6, 2]])
    expected = compact_sgm(cost, lo, hi, 2, 5)
    paths, sums, winners = run_costs(runner, tmp_path, cost, lo, hi, 2, 5, 'left')
    np.testing.assert_array_equal(paths, expected.paths)
    np.testing.assert_array_equal(sums, expected.sums)
    assert winners[0, 3] == -1
    _, _, tie = run_costs(runner, tmp_path, np.zeros((1, 1, 6)), np.zeros((1, 1), dtype=int), np.full((1, 1), 6), 2, 5, 'right')
    assert tie[0, 0] == 0


@pytest.mark.parametrize('eye', [0, 1])
def test_real_image_eye_geometry_and_scores(runner, tmp_path, eye):
    from stereoforge.refinement import census
    from benchmarks.compact_sgm_quality import run_eye
    rng = np.random.default_rng(273)
    images = rng.integers(0, 256, (2, 8, 12), dtype=np.uint8)
    support = np.ones_like(images, dtype=bool)
    support[1, 3, 4] = False
    a, b = (census(images[i], support[i]) for i in (eye, 1-eye))
    h, w = images.shape[1:]
    d = 6
    lo = rng.integers(0, d + 1, (h, w))
    hi = np.array([[rng.integers(v, d + 1) for v in row] for row in lo])
    costs = np.full((h, w, d), np.inf)
    for y in range(h):
        for x in range(w):
            for q in range(d):
                xx = x + q if eye else x - q
                if a.support[y, x] and 0 <= xx < w and b.support[y, xx]:
                    costs[y, x, q] = int(int(a.descriptor[y, x] ^ b.descriptor[y, xx]).bit_count())
    expected = compact_sgm(costs, lo, hi, 8, 32)
    actual = run_eye(runner, tmp_path, a, b, lo, hi, eye, d, dump_scores=True)
    np.testing.assert_array_equal(actual['scores'], expected.sums)
    assert actual['candidate_updates'] == expected.candidate_updates
    assert actual['interval_slots'] == expected.interval_slots
    win = np.argmin(expected.sums, axis=2).astype(np.int32)
    win[~np.isfinite(expected.sums).any(axis=2)] = -1
    np.testing.assert_array_equal(actual['winners'], win)
    finite = np.isfinite(expected.sums)
    np.testing.assert_array_equal(actual['count'], finite.sum(axis=2))
    for y, x in np.ndindex(h, w):
        ordered = np.sort(expected.sums[y, x][finite[y, x]])
        assert actual['best'][y, x] == (int(ordered[0]) if len(ordered) else np.iinfo(np.uint32).max)
        assert actual['second'][y, x] == (int(ordered[1]) if len(ordered) > 1 else np.iinfo(np.uint32).max)


def test_selection_rejects_ties_restricted_endpoints_and_missing_candidates():
    from benchmarks.compact_sgm_quality import select_eye, INF
    r = {'winners': np.array([[0, 2, 2, 3, -1]]),
         'best': np.array([[1, 1, 1, 1, INF]], np.uint32),
         'second': np.array([[2, 1, 2, 2, INF]], np.uint32),
         'count': np.array([[2, 2, 2, 2, 0]], np.uint32)}
    lo = np.array([[0, 0, 2, 0, 0]])
    hi = np.array([[5, 5, 5, 5, 5]])
    d, c, v = select_eye(r, lo, hi, 5)
    np.testing.assert_array_equal(v, [[False, False, False, True, False]])
    assert np.isnan(d[0, 0]) and d[0, 3] == 3
    assert np.isnan(d[0, 1]) and c[0, 1] == 0


def test_main_small_fixture_report_integration(runner, tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    import benchmarks.compact_sgm_quality as experiment
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    import compare_daxa_refinement as cmp
    import refinement_quality_fixtures as fixture
    from stereoforge import refinement
    family = fixture.export_scene_family(tmp_path / 'family', width=64, height=36)
    context = cmp.load_context(family.parent / 'half/manifest.json', family.parent / 'full/manifest.json')
    settings = refinement.Settings(radius=2, subpixel=False)
    prior = np.full((2, 18, 32), 2, np.float32)
    result = refinement.refine_pair(context['images'], context['support'], prior, np.ones_like(prior),
                                    np.ones_like(prior, bool), context['geometry'], context['source_id'], settings)
    dump = tmp_path / 'dump'; dump.mkdir()
    planes = cmp.oracle_planes(result, context)
    files = {}
    for name, plane in planes.items():
        dtype = '<u4' if name.endswith('.u32') else '<i8' if name.endswith('.i64') else '<f4'
        p = dump / name
        np.asarray(plane, dtype=dtype).tofile(p)
        files[name] = {'path': str(p), 'sha256': cmp.sha256(p), 'elements': plane.size}
    report = cmp.expected_report(context, settings, False)
    report['files'] = files
    report_path = tmp_path / 'report.json'; report_path.write_text(json.dumps(report))
    source = Path(__file__).resolve().parents[1] / 'benchmarks/compact_sgm_runner.cpp'
    build = {'command': ['c++', '-std=c++20', '-O2', str(source), '-o', str(runner)],
             'compiler_version': 'test compiler', 'source_sha256': cmp.sha256(source), 'binary_sha256': cmp.sha256(runner)}
    build_path = tmp_path / 'build.json'; build_path.write_text(json.dumps(build))
    output = tmp_path / 'output.json'
    args = ['quality', '--runner', str(runner), '--build-manifest', str(build_path),
            '--coarse-manifest', str(family.parent / 'half/manifest.json'),
            '--target-manifest', str(family.parent / 'full/manifest.json'), '--family', str(family),
            '--report', str(report_path), '--dump-root', str(dump), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', args)
    experiment.main()
    saved = json.loads(output.read_text())
    assert saved['build']['binary_sha256'] == cmp.sha256(runner)
    assert saved['quality']['left']['all']['eligible'] > 0
    assert len(saved['output_plane_hashes']['depth.f32']) == 64
    assert 'outside clean truth support' in saved['quality_caveat']
