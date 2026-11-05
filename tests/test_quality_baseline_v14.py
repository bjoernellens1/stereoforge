"""Quality scores must punish holes and keep inferred support separate."""
import numpy as np
import pytest

from benchmarks.physical_depth_v14 import quality_contract as Q


def test_finite_pixels_are_not_correct_completeness():
    reference = np.ones((2, 3))
    support = np.ones((2, 3), bool)
    depth = np.array([[1., 1.005, 4.], [np.nan, 1., 1.]])
    provenance = np.array([[1, 1, 1], [0, 2, 6]], np.uint32)
    r = Q.score(depth, reference, support, provenance, reference_uncertainty_m=.001)
    assert r['support_pixels'] == 6
    assert r['measured_pixels'] == 3
    assert r['correct_measured_pixels'] == 2
    assert r['correct_completeness_pct'] == pytest.approx(100 * 2 / 6)
    assert r['temporal_pixels'] == r['spatial_pixels'] == 1
    assert r['bad_20mm_pct_of_measured'] == pytest.approx(100 / 3)
    assert r['numerical_accuracy_gate_passed'] is False  # p95 rejects the wrong point


def test_reference_uncertainty_and_empty_support_fail_closed():
    z = np.ones((2, 3))
    r = Q.score(z, z, np.ones_like(z, bool), np.ones_like(z), reference_uncertainty_m=.004)
    assert r['reference_qualified_pixels'] == 0
    assert not r['numerical_accuracy_gate_passed']
    assert r['correct_completeness_pct'] == 100  # diagnostic only
    empty = Q.score(z, z, np.zeros_like(z, bool), np.ones_like(z), reference_uncertainty_m=0)
    assert empty['correct_completeness_pct'] is None
    assert not empty['numerical_accuracy_gate_passed']
    assert empty['p95_absolute_error_mm'] is None


def test_bad_reference_on_fixed_support_is_an_error():
    z = np.ones((2, 3))
    z[0, 0] = np.nan
    with pytest.raises(ValueError, match='reference'):
        Q.score(z, z, np.ones_like(z, bool), np.ones_like(z), reference_uncertainty_m=.001)


def test_trace_counts_overlaps_and_partitions_first_failures():
    masks = {'support': np.array([[1, 0, 1]], bool),
             'lr': np.array([[1, 1, 0]], bool)}
    result, bits, first = Q.failure_accounting(masks)
    assert result['all_reasons'] == {'support': 2, 'lr': 2}
    assert result['first_failure'] == {'support': 2, 'lr': 1, 'unattributed': 0}
    np.testing.assert_array_equal(bits, [[3, 2, 1]])
    np.testing.assert_array_equal(first, [[1, 2, 1]])


def test_trace_cannot_claim_missing_diagnostic_equivalence():
    with pytest.raises(ValueError, match='equivalence'):
        Q.require_equivalent({'depth': np.array([1.])}, {'depth': np.array([2.])})
    Q.require_equivalent({'depth': np.array([np.nan, 1.])}, {'depth': np.array([np.nan, 1.])})


def test_dynamic_reference_requires_motion_times_timing_budget():
    assert Q.dynamic_reference_available(1., .001, .002)
    assert not Q.dynamic_reference_available(1., .003, .002)
    assert not Q.dynamic_reference_available(1., None, .002)


def test_corpus_audit_does_not_infer_mounting_session(tmp_path):
    from benchmarks.physical_depth_v14.quality_baseline import audit_scene
    (tmp_path / 'scene.json').write_text('{"scene":"S1", "tape_m":1.0}')
    np.save(tmp_path / 'rs_depth.npy', np.ones((1, 2, 3), np.float32))
    audit = audit_scene(tmp_path)
    assert audit['rig_session'] is None
    assert audit['absolute_reference_qualified'] is False
    assert audit['dynamic_reference_available'] is False
    assert 'native_depth' in audit['missing_reference_evidence']
    assert audit['split'] == 'development'


def test_all_native_provenance_classes_stay_separate():
    z = np.ones(8)
    r = Q.score(z, z, np.ones(8, bool), np.arange(8), reference_uncertainty_m=.001)
    assert r['measured_pixels'] == 2  # stereo and stereo-left-aggregated
    assert r['temporal_pixels'] == 3  # reprojected, fused, temporal-planar
    assert r['spatial_pixels'] == 2
    assert r['neural_pixels'] == 1


def test_nonpositive_metric_denominator_is_invalid():
    from benchmarks.physical_depth_v14.quality_baseline import metric_depth
    z, valid = metric_depth(np.array([1., 2., 3., np.nan]), dict(fx=10., baseline=.1, principal_offset=2.))
    np.testing.assert_array_equal(valid, [False, False, True, False])
    assert np.isnan(z[:2]).all() and np.isnan(z[3]) and z[2] == 1.


def test_coarse_geometry_requires_same_calibration_and_half_pixel_scaling():
    import copy
    from benchmarks.physical_depth_v14.quality_baseline import validate_coarse_geometry
    target = dict(source=dict(eye_width=32, height=24, stride_bytes=128), rectified=dict(width=32, height=24),
                  calibration=dict(original_identity='a', R1=np.eye(3).tolist(), R2=np.eye(3).tolist(),
                                   P1=[[20,0,16,0],[0,20,12,0],[0,0,1,0]], P2=[[20,0,16,-2],[0,20,12,0],[0,0,1,0]]))
    coarse = copy.deepcopy(target)
    coarse['rectified'] = dict(width=8, height=6)
    a = np.array([[.25,0,-.375],[0,.25,-.375],[0,0,1.]])
    for key in ('P1', 'P2'):
        coarse['calibration'][key] = (a @ np.asarray(target['calibration'][key])).tolist()
    validate_coarse_geometry(target, coarse)
    coarse['calibration']['original_identity'] = 'other'
    with pytest.raises(ValueError, match='calibration'):
        validate_coarse_geometry(target, coarse)
    coarse['calibration']['original_identity'] = 'a'
    coarse['calibration']['P1'][0][2] += .375
    with pytest.raises(ValueError, match='half-pixel'):
        validate_coarse_geometry(target, coarse)


@pytest.mark.parametrize('ablation', [False, True])
def test_baseline_retains_bound_artifacts_and_completion_state(tmp_path, ablation):
    import argparse
    import hashlib
    import json
    from benchmarks.physical_depth_v14 import quality_baseline as B
    fixture = tmp_path / 'fixture'
    fixture.mkdir()
    h, w = 12, 32
    yy, xx = np.mgrid[:h, :w].astype('<f4')
    maps = {}
    for key, array in zip(('left_x', 'left_y', 'right_x', 'right_y'), (xx, yy, xx, yy)):
        path = fixture / (key + '.f32')
        path.write_bytes(array.tobytes())
        maps[key] = dict(file=path.name, sha256=B.sha256(path))
    original = fixture / 'original.json'
    original.write_text('{}')
    rectified = fixture / 'rectified.json'
    rectified.write_text('{"synthetic": true}')
    cal = dict(P1=[[20,0,16,0],[0,20,6,0],[0,0,1,0]], P2=[[20,0,16,-2],[0,20,6,0],[0,0,1,0]])
    for key, path in [('original', original), ('rectified', rectified)]:
        cal[key + '_json'] = dict(file=path.name, sha256=B.sha256(path))
        cal[key + '_identity'] = B.sha256(path)
    manifest = fixture / 'manifest.json'
    manifest.write_text(json.dumps(dict(source=dict(eye_width=w, height=h), rectified=dict(width=w, height=h), calibration=cal, maps=maps)))
    scene = tmp_path / 'corpus' / 'synthetic'
    scene.mkdir(parents=True)
    (scene / 'scene.json').write_text('{"scene":"synthetic"}')
    raw = np.zeros((1, h, w * 2, 2), np.uint8)
    raw[..., 1] = 128
    np.save(scene / 'zed_raw.npy', raw)
    args = argparse.Namespace(out=tmp_path / 'result', manifest=manifest, cpu=False,
                              corpus=scene.parent, scenes=['synthetic'], indices=[0], disparities=[128])
    if ablation:
        args.reference_costs = ['normalized', 'census']
        args.reference_paths = [4, 8]
        args.reference_subpixel = ['none']
        args.reference_lr_sampling = 'linear'
        args.reference_lr_tolerance = .25
        args.reference_lr_max_jump = .5
    result = B.run(args)
    assert result['status'] == 'complete_development_diagnosis'
    assert result['requested_scenes'] == ['synthetic'] and result['requested_indices'] == [0]
    modes = result['scenes']['synthetic']['frames'][0]['modes']
    if ablation:
        assert len(modes) == 4
        assert {m['params']['matching_cost'] for m in modes.values()} == {'normalized', 'census'}
        assert {m['params']['paths'] for m in modes.values()} == {4, 8}
        mode = modes['independent_reference_d128_census_4path_none_lrlinear_t0.25_j0.5']
        assert mode['params']['lr_sampling'] == 'linear'
        assert mode['params']['lr_tol'] == .25 and mode['params']['lr_max_jump'] == .5
        assert mode['all_reasons']['support_failure'] == w * h - (w - 4) * (h - 4)
    else:
        mode = modes['independent_reference_d128']
    artifact = args.out / mode['artifact']['file']
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == mode['artifact']['sha256']
    assert mode['correct_completeness'] is None
    assert sum(mode['first_failure'].values()) == w * h
    with np.load(artifact) as planes:
        assert planes['reason_mask'].shape == (h, w)


def test_equivalence_rejects_signed_zero_or_storage_type_changes():
    with pytest.raises(ValueError, match='equivalence'):
        Q.require_equivalent({'depth': np.array([-0.], np.float32)}, {'depth': np.array([0.], np.float32)})
    with pytest.raises(ValueError, match='equivalence'):
        Q.require_equivalent({'depth': np.array([1.], np.float32)}, {'depth': np.array([1.], np.float64)})


def test_observed_cpu_rejects_borrowed_workspace_aliases():
    from benchmarks.physical_depth_v14.quality_baseline import run_observed_cpu

    class ReusedWorkspace:
        value = np.array([1.], np.float32)

        def run_full_camera_to_depth_pipeline(self, **kwargs):
            self.value += 1
            return {'depth': self.value}

    with pytest.raises(ValueError, match='owned'):
        run_observed_cpu(ReusedWorkspace(), fast_mode=True, borrow_workspace=True)
