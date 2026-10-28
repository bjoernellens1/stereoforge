import numpy as np
import pytest

from benchmarks.fast60_prior_diagnostic import classify, region_counts


def test_partition_and_order_with_invalid_prediction_windows_and_nan_raw():
    truth = np.full((1, 9), 5.2)
    prediction = np.array([[np.nan, 5, 5, 5, 5, 5, 5, 5, 5.]])
    prediction_valid = np.ones((1, 9), bool)
    lower = np.array([[0, 0, 9, 4, 4, 4, 4, 4, 0]])
    upper = np.array([[10, 2, 3, 6, 6, 6, 6, 6, 10]])
    raw = np.array([[5, 5, 5, np.nan, np.nan, 8, 5, 5, 5.]])
    raw_valid = np.array([[1, 1, 1, 0, 1, 1, 1, 1, 1]], bool)
    lr = np.array([[1, 1, 1, 0, 1, 1, 0, 1, 1]], bool)
    support = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0]], bool)
    masks = classify(truth, support, prediction, prediction_valid, lower, upper, raw, raw_valid, lr)
    assert [int(m.sum()) for m in masks.values()] == [1, 2, 1, 2, 1, 1]
    assert np.array_equal(sum(m.astype(int) for m in masks.values()), support.astype(int))
    rows = region_counts(masks, {'all': np.ones_like(support), 'thin': np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0]], bool)})
    assert rows['all']['denominator'] == 8
    assert rows['thin']['denominator'] == 4
    assert sum(rows['thin']['counts'].values()) == 4


def test_inclusive_integer_window_and_prediction_validity():
    # truth=4.2 has qualifying integer disparities 4 and 5; 6 is too far.
    t = np.array([[4.2, 4.2, 4.2, 4.2]])
    masks = classify(t, np.ones_like(t, bool), np.ones_like(t), np.array([[0, 1, 1, 1]], bool),
                     np.array([[4, 4, 5, 6]]), np.array([[4, 4, 5, 6]]),
                     t, np.ones_like(t, bool), np.ones_like(t, bool))
    assert masks['no_prediction'][0, 0]
    assert masks['window_misses_truth'][0, 3]
    assert masks['correct_lr_valid'][0, 1:3].all()


def test_nonfinite_truth_in_support_rejected():
    a = np.array([[np.nan]])
    with pytest.raises(ValueError, match='truth'):
        classify(a, np.ones_like(a, bool), a, np.zeros_like(a, bool), a, a, a, np.zeros_like(a, bool), np.zeros_like(a, bool))


def test_nonbinary_wire_masks_rejected():
    from benchmarks.fast60_prior_diagnostic import validate_wire_masks
    with pytest.raises(ValueError, match='prediction_validity'):
        validate_wire_masks({'prediction_validity.u32': np.array([0, 2], dtype=np.uint32)})


def test_provenance_validation_rejects_changed_metadata(monkeypatch):
    from benchmarks.fast60_prior_diagnostic import validate_report
    class Comparator:
        @staticmethod
        def expected_report(context, settings, validation):
            return {'schema': 'expected', 'actual_coarse_pipeline': True,
                    'diagnostic_injection': False, 'radius': 2}
        @staticmethod
        def _same(a, b, label):
            if type(a) is not type(b) or a != b:
                raise ValueError('changed ' + label)
    report = {'schema': 'expected', 'actual_coarse_pipeline': True,
              'diagnostic_injection': False, 'radius': 2, 'validation_requested': True}
    validate_report(Comparator, report, None, None)
    for key, bad in [('schema', 'wrong'), ('actual_coarse_pipeline', 1),
                     ('diagnostic_injection', True), ('radius', 4)]:
        changed = dict(report, **{key: bad})
        with pytest.raises(ValueError, match=key):
            validate_report(Comparator, changed, None, None)


def test_cpu_census_support_rejects_native_tampering():
    from benchmarks.fast60_prior_diagnostic import verified_census_support
    from stereoforge import refinement
    images = np.zeros((2, 7, 7), np.uint32)
    support = np.ones_like(images, np.uint32)
    expected = np.stack([refinement.census(images[i], support[i]).support for i in range(2)])
    assert np.array_equal(verified_census_support(images, support, expected, refinement), expected)
    expected[0, 3, 3] = False
    with pytest.raises(ValueError, match='Census support'):
        verified_census_support(images, support, expected, refinement)


def test_valid_wire_values_checked():
    from benchmarks.fast60_prior_diagnostic import validate_wire_values
    a = {'prediction.f32': np.array([5., np.nan]), 'prediction_validity.u32': np.array([1, 0]),
         'lower.u32': np.array([3, 0xffffffff]), 'upper.u32': np.array([7, 0xffffffff]),
         'raw_disparity.f32': np.array([5., np.nan]), 'raw_validity.u32': np.array([1, 0]),
         'disparity.f32': np.array([5., np.nan]), 'validity.u32': np.array([1, 0])}
    validate_wire_values(a, 128)
    for name, value in [('prediction.f32', np.nan), ('lower.u32', 8),
                        ('upper.u32', 128), ('raw_disparity.f32', np.nan), ('disparity.f32', 4.)]:
        broken = {k: v.copy() for k, v in a.items()}
        broken[name][0] = value
        with pytest.raises(ValueError):
            validate_wire_values(broken, 128)


def test_cli_rejects_tampered_provenance_and_rehashed_census(tmp_path, monkeypatch):
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from benchmarks import fast60_prior_diagnostic as diagnostic
    path = Path(__file__).with_name('test_compare_daxa_refinement.py')
    spec = importlib.util.spec_from_file_location('refinement_wire_fixture', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    h, report, _, _, dump, _, family = module.wire.__wrapped__(tmp_path)
    report_path = tmp_path / 'native.json'
    def save(): report_path.write_text(json.dumps(report))
    args = ['diagnostic', '--coarse-manifest', str(family.parent/'half/manifest.json'),
            '--target-manifest', str(family.parent/'full/manifest.json'), '--family', str(family),
            '--report', str(report_path), '--dump-root', str(dump), '--output', str(tmp_path/'output.json'),
            '--radius', '4']
    monkeypatch.setattr(sys, 'argv', args)
    save(); diagnostic.main()
    report['diagnostic_injection'] = True; save()
    with pytest.raises(ValueError, match='diagnostic_injection'): diagnostic.main()
    report['diagnostic_injection'] = False
    plane = report['files']['census_validity.u32']; p = Path(plane['path'])
    a = np.fromfile(p, '<u4'); a[0] ^= np.uint32(1); p.write_bytes(a.tobytes())
    plane['sha256'] = h.sha256(p); save()
    with pytest.raises(ValueError, match='Census support'): diagnostic.main()


def test_guided_window_must_match_rounded_prediction_and_raw_requires_prior():
    from benchmarks.fast60_prior_diagnostic import validate_wire_values
    a = {'prediction.f32': np.array([5.49, np.nan], dtype=np.float32),
         'prediction_validity.u32': np.array([1, 0]), 'lower.u32': np.array([3, 0]),
         'upper.u32': np.array([7, 0xffffffff]), 'raw_disparity.f32': np.array([5., np.nan]),
         'raw_validity.u32': np.array([1, 0]), 'disparity.f32': np.array([5., np.nan]),
         'validity.u32': np.array([1, 0])}
    validate_wire_values(a, 128, 2)
    for name, pos, val in [('lower.u32', 0, 4), ('upper.u32', 0, 6), ('raw_validity.u32', 1, 1)]:
        broken = {k: v.copy() for k, v in a.items()}; broken[name][pos] = val
        if name == 'raw_validity.u32': broken['raw_disparity.f32'][pos] = 5.
        with pytest.raises(ValueError): validate_wire_values(broken, 128, 2)
