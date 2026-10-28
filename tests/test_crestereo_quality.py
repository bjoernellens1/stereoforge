import numpy as np
import pytest
from benchmarks.crestereo_quality import depth_from_disparity, score, shifted_pair, restore_disparity


def test_shifted_pair_sign_and_no_wrap():
    image = np.arange(12, dtype=np.uint8).reshape(2, 6)
    for shift in (-2, 0, 2):
        left, right, support = shifted_pair(image, shift)
        ys, xs = np.where(support)
        assert np.array_equal(left[ys, xs], right[ys, xs-shift])
        assert support.sum() == 2 * (6 - abs(shift))


def test_resize_horizontal_units_and_principal_offset():
    raw = np.full((240, 320), 5., np.float32)
    d = restore_disparity(raw, 1280, 720)
    assert d.shape == (720, 1280)
    assert np.all(d == 20)
    depth, valid = depth_from_disparity(d, 500., .12, 2.)
    assert valid.all()
    assert np.allclose(depth, 60/18)
    _, valid = depth_from_disparity(np.array([[np.nan, 2., -1., 3.]]), 500., .12, 2.)
    assert valid.tolist() == [[False, False, False, True]]


def test_fixed_support_counts_missing_and_wrong():
    truth = np.full((2, 3), 10.)
    pred = np.array([[10., 9., 20.], [np.nan, 10., 10.]])
    support = np.array([[1,1,1],[1,0,0]], bool)
    q = score(pred, truth, support, threshold=1.)
    assert q['eligible'] == 4 and q['numerical_valid'] == 3
    assert q['correct'] == 2 and q['false_accepted'] == 1
    assert q['correct_completeness'] == .5
    assert q['valid_coverage'] == .75


def test_reject_bad_calibration():
    with pytest.raises(ValueError):
        depth_from_disparity(np.ones((1, 1)), -1, .12, 0)


def test_prefilter_is_explicit_and_reduces_alias():
    from benchmarks.crestereo_quality import prefilter
    image = np.zeros((32, 32, 3), np.uint8)
    image[:, ::2] = 255
    assert np.array_equal(prefilter(image, 0), image)
    smooth = prefilter(image, 2)
    assert smooth.dtype == np.uint8 and smooth.shape == image.shape
    assert np.std(smooth.astype(float)) < np.std(image.astype(float))
    with pytest.raises(ValueError):
        prefilter(image, 1)


def test_evaluator_rejects_changed_frozen_prediction(tmp_path):
    import hashlib, json
    from benchmarks.crestereo_quality import evaluate
    fixture = tmp_path / 'fixture'; fixture.mkdir()
    frozen = tmp_path / 'frozen'; frozen.mkdir()
    manifest = fixture / 'manifest.json'; manifest.write_text('{}')
    (frozen / 'prediction.f32').write_bytes(b'changed')
    (frozen / 'freeze.json').write_text(json.dumps({
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'prediction_sha256': '0' * 64,
    }))
    with pytest.raises(ValueError, match='frozen binding'):
        evaluate(fixture, frozen, tmp_path / 'out.json')
    assert not (tmp_path / 'out.json').exists()


def test_scoring_uses_effective_disparity_with_both_offset_signs():
    pred = np.array([[1., -1., 3., np.nan]])
    truth = np.zeros((1, 4)); domain = np.ones((1, 4), bool)
    plus = score(pred, truth, domain, principal_offset=2.)
    minus = score(pred, truth, domain, principal_offset=-2.)
    assert plus['numerical_positive'] == minus['numerical_positive'] == 2
    assert plus['numerical_valid'] == 1 and minus['numerical_valid'] == 3


def test_evaluate_small_fixture_truth_masks_and_hashes(tmp_path):
    import hashlib, json
    from benchmarks.crestereo_quality import evaluate
    fixture = tmp_path / 'fixture'; fixture.mkdir()
    frozen = tmp_path / 'frozen'; frozen.mkdir()
    def digest(x): return hashlib.sha256(x).hexdigest()
    p1 = [[100.,0.,4.,0.],[0.,100.,4.,0.],[0.,0.,1.,0.]]
    p2 = [[100.,0.,3.99,-12.],[0.,100.,4.,0.],[0.,0.,1.,0.]]
    manifest = json.dumps({'calibration':{'P1':p1,'P2':p2}}).encode()
    (fixture/'manifest.json').write_bytes(manifest)
    shape=(8,8)
    planes={}
    for name in ('disparity','visible','out_of_range','mixed_bilinear','remap_support','occluded','region_thin','region_edges','region_near','region_far'):
        if name == 'disparity': a=np.full(shape, .025, dtype='<f8')
        else: a=np.zeros(shape, dtype='u1')
        if name in ('visible','remap_support','region_far'): a[:]=1
        if name == 'mixed_bilinear': a[3,3]=1
        if name == 'occluded': a[0,0]=1
        if name == 'visible': a[0,0]=0
        raw=a.tobytes(); (fixture/name).write_bytes(raw)
        planes[name]=dict(file=name, bytes=len(raw),sha256=digest(raw),dtype=a.dtype.str,shape=list(shape))
    truth={'native_manifest_sha256':digest(manifest),'eyes':{'left':planes}}
    (fixture/'truth.json').write_text(json.dumps(truth))
    prediction=np.ones((240,320),dtype='<f4').tobytes()
    (frozen/'prediction.f32').write_bytes(prediction)
    (frozen/'freeze.json').write_text(json.dumps({'manifest_sha256':digest(manifest), 'prediction_sha256':digest(prediction), 'width':8,'height':8}))
    out=tmp_path/'out.json'; evaluate(fixture,frozen,out)
    report=json.loads(out.read_text())
    assert report['metrics']['native_clean']['eligible'] == 15
    assert report['metrics']['native_clean']['correct'] == 15
    assert report['occluded_positive_metric_denominator'] == 1
    (fixture/'visible').write_bytes(b'bad')
    with pytest.raises(ValueError, match='fixture hash mismatch'):
        evaluate(fixture,frozen,tmp_path/'tampered.json')
