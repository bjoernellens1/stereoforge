import numpy as np
import pytest


def test_edge_neighborhood_widens_guided_intervals_and_missing_prior_is_full():
    from benchmarks.compact_sgm_edge_experiment import intervals
    image = np.zeros((7, 9), np.uint8)
    image[3, 4] = 255
    pred = np.ones_like(image, bool)
    pred[0, 0] = False
    support = np.ones_like(pred)
    lo = np.full(image.shape, 20)
    hi = np.full(image.shape, 24)
    low, high, edge = intervals(image, pred, support, lo, hi, 128)
    assert (low[0, 0], high[0, 0]) == (0, 128)
    assert (low[0, 8], high[0, 8]) == (20, 25)
    assert edge[3, 4] and (low[3, 4], high[3, 4]) == (12, 33)


def test_both_eye_selection_and_surface_false_accept_counts():
    from benchmarks.compact_sgm_edge_experiment import surface_counts
    truth = {'left': {'surface_id': np.array([[1, 1, 2, 2]]), 'valid': np.ones((1,4)),
                      'visible': np.array([[1, 1, 1, 0]]), 'disparity': np.array([[3., 3., 4., 4.]])},
             'right': {'surface_id': np.array([[1, 2, 2, 0]]), 'valid': np.ones((1,4)),
                       'visible': np.ones((1,4)), 'disparity': np.array([[3., 4., 4., 4.]])}}
    actual = {'validity.u32': np.array([[[1, 1, 1, 1]], [[1, 1, 0, 1]]]),
              'raw_disparity.f32': np.array([[[3., 5., 4., 4.]], [[3., 8., 4., 4.]]])}
    rows = surface_counts(actual, truth)
    assert rows['left']['1'] == {'pixels': 2, 'accepted': 2, 'false_accepted': 1}
    assert rows['left']['2']['false_accepted'] == 1
    assert rows['right']['2']['false_accepted'] == 1
    assert rows['right']['0']['accepted'] == 1


def test_rejects_bad_images():
    from benchmarks.compact_sgm_edge_experiment import intervals
    a = np.zeros((3,3), np.uint8)
    with pytest.raises(ValueError):
        intervals(a.astype(float), a, a, a, a, 16)


def test_experiment_freezes_before_truth_and_reports_both_eyes(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys
    from pathlib import Path
    from benchmarks import compact_sgm_edge_experiment as experiment
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
    files = {}
    for name, plane in cmp.oracle_planes(result, context).items():
        dtype = '<u4' if name.endswith('.u32') else '<i8' if name.endswith('.i64') else '<f4'
        path = dump / name
        np.asarray(plane, dtype=dtype).tofile(path)
        files[name] = {'path': str(path), 'sha256': cmp.sha256(path), 'elements': plane.size}
    report = cmp.expected_report(context, settings, False); report['files'] = files
    report_path = tmp_path / 'report.json'; report_path.write_text(json.dumps(report))
    source = Path(__file__).resolve().parents[1] / 'benchmarks/compact_sgm_runner.cpp'
    runner = tmp_path / 'runner'
    command = ['c++', '-std=c++20', '-O2', str(source), '-o', str(runner)]
    subprocess.run(command, check=True)
    build = {'command': command, 'compiler_version': 'test compiler', 'source_sha256': cmp.sha256(source), 'binary_sha256': cmp.sha256(runner)}
    build_path = tmp_path / 'build.json'; build_path.write_text(json.dumps(build))
    frozen, output = tmp_path / 'frozen.npz', tmp_path / 'output.json'
    original = cmp.load_truth
    def verify_frozen(*args):
        assert frozen.exists() and Path(str(frozen) + '.selection.npz').exists()
        return original(*args)
    monkeypatch.setattr(cmp, 'load_truth', verify_frozen)
    monkeypatch.setattr(sys, 'argv', ['experiment', '--runner', str(runner), '--build-manifest', str(build_path),
        '--coarse-manifest', str(family.parent / 'half/manifest.json'), '--target-manifest', str(family.parent / 'full/manifest.json'),
        '--family', str(family), '--report', str(report_path), '--dump-root', str(dump), '--frozen', str(frozen), '--output', str(output)])
    experiment.main()
    saved = json.loads(output.read_text())
    with np.load(frozen, allow_pickle=False) as frozen_data:
        assert json.loads(str(frozen_data['config'])) == experiment.CONFIG
        assert json.loads(str(frozen_data['gates'])) == experiment.GATES
        frozen_hashes = json.loads(str(frozen_data['input_hashes']))
        assert frozen_hashes['family'] == cmp.sha256(family)
        assert frozen_hashes['runner'] == cmp.sha256(runner)
        assert frozen_hashes['report'] == cmp.sha256(report_path)
    assert saved['surface_counts_domain'] != saved['surface_quality_domain']
    assert saved['hashes']['frozen_intervals'] == cmp.sha256(frozen)
    assert saved['hashes']['frozen_selection'] == cmp.sha256(Path(str(frozen) + '.selection.npz'))
    assert set(saved['surface_counts']) == {'left', 'right'}
    assert 0 < saved['work']['interval_fraction'] <= 1
    with pytest.raises(FileExistsError):
        experiment.main()
    frozen.unlink(); Path(str(frozen) + '.selection.npz').unlink(); output.unlink()
    def tamper_selection(*args):
        truth = original(*args)
        with Path(str(frozen) + '.selection.npz').open('ab') as f:
            f.write(b'tampered')
        return truth
    monkeypatch.setattr(cmp, 'load_truth', tamper_selection)
    with pytest.raises(ValueError, match='selection changed'):
        experiment.main()
    assert not output.exists()


def test_thin_representability_separates_missing_invalid_and_wrong():
    from benchmarks.compact_sgm_edge_experiment import thin_representability
    truth = {'valid': np.ones((1, 4)), 'visible': np.ones((1, 4)), 'out_of_range': np.zeros((1, 4)),
             'off_image': np.zeros((1, 4)), 'remap_support': np.ones((1, 4)), 'mixed_bilinear': np.zeros((1, 4)),
             'region_thin': np.ones((1, 4)), 'disparity': np.array([[3., 3., 3., 3.]])}
    actual = {'metric_validity.u32': np.array([[1, 0, 1, 0]]), 'raw_disparity.f32': np.array([[3., np.nan, 6., np.nan]])}
    low = np.array([[2, 2, 2, 8]])
    high = np.array([[5, 5, 5, 9]])
    assert thin_representability(truth, actual, np.ones((1, 4), bool), low, high) == {
        'eligible': 4, 'representable': 3, 'not_representable': 1,
        'representable_metric_invalid': 1, 'representable_metric_incorrect': 1,
        'representable_metric_correct': 1}


def test_surface_clean_domain_counts():
    from benchmarks.compact_sgm_edge_experiment import surface_quality
    t = {'surface_id': np.array([[1, 1, 1, 2]]), 'valid': np.ones((1, 4)), 'visible': np.ones((1, 4)),
         'out_of_range': np.zeros((1, 4)), 'off_image': np.zeros((1, 4)), 'remap_support': np.ones((1, 4)),
         'mixed_bilinear': np.array([[0, 0, 1, 0]]), 'disparity': np.array([[3., 3., 3., 3.]])}
    a = {'validity.u32': np.array([[1, 1, 1, 1]]), 'metric_validity.u32': np.array([[1, 1, 1, 0]]),
         'raw_disparity.f32': np.array([[3., 5., 3., 3.]])}
    rows = surface_quality(t, a, np.ones((1, 4), bool))
    assert rows['1'] == {'eligible': 2, 'metric_accepted': 2, 'correct_metric': 1, 'lr_false_accepted': 1}
    assert rows['2']['eligible'] == 1 and rows['2']['metric_accepted'] == 0
