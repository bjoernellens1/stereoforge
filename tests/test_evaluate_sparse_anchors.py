from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import evaluate_sparse_anchors as evaluator
from sparse_stereo_reference import Match, Result


def truth():
    shape = (1, 6)
    t = {k: np.zeros(shape, bool) for k in ('occluded', 'off_image', 'mixed_bilinear', 'region_thin', 'region_edges', 'region_near', 'region_far')}
    t.update(valid=np.ones(shape, bool), visible=np.ones(shape, bool), remap_support=np.ones(shape, bool),
             disparity=np.array([[3., 4., 130., 2., 2., 2.]]), surface_id=np.array([[1, 1, 2, 3, 3, 3]]),
             cross_u=np.arange(6, dtype=float).reshape(shape), cross_v=np.zeros(shape))
    t['region_thin'][0, 2] = True
    t['region_near'][0, 2] = True
    t['region_edges'][0, 1] = True
    t['mixed_bilinear'][0, 3] = True
    t['occluded'][0, 4] = True
    t['off_image'][0, 5] = True
    return {'left': t, 'right': {k: v.copy() for k, v in t.items()}}


def matches():
    return [asdict(Match('left', 0, x, x, disp, 0, 2, True, 'ok')) for x, disp in enumerate((3, 7, 130, 2, 2, 2))]


def test_fixed_denominators_range_exclusions_and_surfaces():
    t = truth(); support = np.ones((2, 1, 6), bool)
    q128 = evaluator.evaluate(matches(), t, support, 128)['left']
    q224 = evaluator.evaluate(matches(), t, support, 224)['left']
    assert q128['all']['eligible_pixels'] == 2
    assert q128['all']['correct'] == 1
    assert q128['excluded']['out_of_range'] == 1
    assert q224['all']['eligible_pixels'] == 3
    assert q224['all']['correct'] == 2
    assert q224['all']['false_accepted_eligible'] == 1
    assert q224['all']['accepted_total'] == 6
    assert q224['all']['correct_over_all_accepted'] == 2 / 6
    assert q224['surfaces']['2']['correct'] == 1
    assert q224['excluded']['mixed'] == q224['excluded']['occluded'] == q224['excluded']['off_image'] == 1
    assert q224['thin']['correct_over_eligible_pixels'] == 1
    assert evaluator.evaluate([], t, support, 224)['right']['all']['correct_over_all_accepted'] is None
    support[0, 0, 0] = False
    assert evaluator.evaluate(matches(), t, support, 224)['left']['excluded']['unsupported'] == 1


def test_policy_and_decisions_written_before_truth(tmp_path, monkeypatch):
    for n in ('coarse', 'target', 'family'):
        (tmp_path/n).write_text(n)
    out = tmp_path/'output'
    images = np.arange(12, dtype=np.uint32).reshape(2, 1, 6)
    context = {'images': images, 'support': np.ones_like(images), 'source_id': 42}
    monkeypatch.setattr(evaluator.fixture, 'load_context', lambda *args: context)
    calls = []
    def fake_match(*args, **kwargs):
        assert (out/'policy.json').is_file()
        calls.append((kwargs['selection'], kwargs['disparities']))
        d = 3 if kwargs['selection'] == 'gradient' else 4
        return Result('42', (Match('left', 0, 0, 0, d, 0, 2, True, 'ok'),), 7, 5)
    monkeypatch.setattr(evaluator.matcher, 'sparse_stereo', fake_match)
    def load_truth(*args):
        assert (out/'decisions.json').is_file()
        return truth()
    monkeypatch.setattr(evaluator.fixture, 'load_truth', load_truth)
    class C:
        support = np.ones((1, 6), bool)
    monkeypatch.setattr(evaluator, 'census', lambda *args: C())
    report = evaluator.run(tmp_path/'coarse', tmp_path/'target', tmp_path/'family', out)
    assert calls == [('gradient', 128), ('gradient', 224), ('descriptor', 128), ('descriptor', 224)]
    assert report['profiles']['gradient']['224']['forward_candidates'] == 7
    assert report['profiles']['gradient']['128']['eyes']['left']['all']['correct'] == 1
    assert report['profiles']['descriptor']['128']['eyes']['left']['all']['correct'] == 1
    assert report['policy']['selectors'] == ['gradient', 'descriptor']
    import json
    decisions = json.loads((out/'decisions.json').read_text())
    assert decisions['runs']['gradient']['128']['matches'][0]['disparity'] == 3
    assert decisions['runs']['descriptor']['128']['matches'][0]['disparity'] == 4
    with pytest.raises(FileExistsError):
        evaluator.run(tmp_path/'coarse', tmp_path/'target', tmp_path/'family', out)


def test_thin_surface_gate_requires_correct_anchor_inside_thin_region():
    t = truth()
    t['left']['surface_id'][0, 0] = 2
    # Surface 2 has an eligible thin pixel at x=2, but only x=0 was matched.
    match = asdict(Match('left', 0, 0, 0, 3, 0, 2, True, 'ok'))
    q = evaluator.evaluate([match], t, np.ones((2, 1, 6), bool), 224)['left']
    assert q['surfaces']['2']['correct'] == 1
    assert q['surfaces']['2']['thin_correct'] == 0
    assert not q['provisional_gates']['thin_each_eligible_surface']


def test_accepted_exclusions_report_each_overlapping_reason():
    t = truth()
    t['left']['mixed_bilinear'][0, 4] = True  # Both mixed and occluded.
    support = np.ones((2, 1, 6), bool)
    support[0, 0, 0] = False
    q = evaluator.evaluate(matches(), t, support, 128)['left']
    ex = q['accepted_excluded']
    assert ex['outside_eligible'] == 5
    assert ex['unsupported'] == 1
    assert ex['mixed'] == 2
    assert ex['occluded'] == 1
    assert ex['off_image'] == 1
    assert ex['out_of_range'] == 1
    assert ex['supported_but_outside_eligible'] == 4


def test_counterpart_pair_support_nearest_pixel_and_boundaries():
    t = truth()
    left = t['left']
    left['mixed_bilinear'][:] = False
    left['occluded'][:] = False
    left['off_image'][:] = False
    left['cross_u'][0] = [0.49, 0.5, 1.49, 1.5, -0.51, 5.5]
    support = np.ones((2, 1, 6), bool)
    support[1, 0, 1] = False
    decisions = [asdict(Match('left', 0, x, x, int(d), 0, 2, True, 'ok'))
                 for x, d in enumerate(left['disparity'][0])]
    q = evaluator.evaluate(decisions, t, support, 224)['left']
    assert q['all']['geometric_eligible_pixels'] == 6
    assert q['all']['eligible_pixels'] == 2  # x=0 maps to 0, x=3 maps to 2
    assert q['accepted_excluded']['opposite_unsupported'] == 2
    assert q['accepted_excluded']['counterpart_outside_integer_grid'] == 2
    assert q['all']['correct'] == 2


def test_counterpart_requires_rectified_same_integer_row():
    t = truth()
    t['left']['cross_v'][0, 0] = 0.5
    q = evaluator.evaluate(matches(), t, np.ones((2, 1, 6), bool), 224)['left']
    assert q['accepted_excluded']['counterpart_outside_integer_grid'] == 1


def test_right_eye_uses_left_eye_counterpart_support():
    t = truth()
    support = np.ones((2, 1, 6), bool)
    # The right reference is supported, but its nearest left counterpart is not.
    support[0, 0, 0] = False
    decision = asdict(Match('right', 0, 0, 0, 3, 0, 2, True, 'ok'))
    q = evaluator.evaluate([decision], t, support, 224)['right']
    assert q['all']['geometric_eligible_pixels'] == 3
    assert q['all']['eligible_pixels'] == 2
    assert q['all']['accepted_total'] == 1
    assert q['accepted_excluded']['opposite_unsupported'] == 1
    assert q['all']['correct'] == 0
